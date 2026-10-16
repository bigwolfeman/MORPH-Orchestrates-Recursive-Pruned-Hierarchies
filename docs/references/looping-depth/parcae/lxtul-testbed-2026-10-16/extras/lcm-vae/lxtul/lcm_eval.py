"""Score a finished stage-2 LCM run (or a PURE span-NCM run): token CE of the scored spans, per backbone depth.

  python -m lxtul.lcm_eval --out DIR [--rows-from RUN] [--compare RUN]... [--rows 480] [--depths 1 2 3 4 6 8] RUN [RUN ...]

`--compare` is repeated, one run per flag (a variadic `--compare` swallowed the positional run list and the chain died
with "the following arguments are required: runs"). RUN is a stage-2 `lcm` run or a PURE `ncm` run
(`ncm.coda_span_local`, it scores this same set); a plan-arm `ncm` run scores every token label of the row and is
paired with `lxtul.paired_eval` instead.

Scored set (exactly): the target tokens of the complete spans of each row with span index >= 1. A complete
span is one closed by a boundary inside the row; span 0 (no earlier concept) and the open tail (not a span)
are dropped, and their token counts are written to `summary.json["scope"]`. `--rows-from RUN` takes the
EVAL ROWS of another run (its morph config, its cell count): with a minimal-TUL run these are the 1088-position
rows of `minimal-5k-s2`, so the LCM and that run score the same text. `--compare RUN` also scores that run on the
same target tokens and writes the paired bootstrap CI of (run under test - compare run) per depth; the token counts of
the two sides must agree row by row. A TUL (`lxtul`) run is scored on the positions whose label is such a token
(`lcm.tul_target_mask`). A `gpt`-kind run (gpt8, gpt8_pointer) is scored on its per-token NLL at the same targets,
after checking row by row that its label ids are those target tokens (`gpt_sums_on_set`).

Per run it writes `DIR/<run>/sweep.json` (per-row sums per depth: the existing sweep format plus `oracle`,
`null`, `scope` for an LCM run; the ELBO keys of `ncm.NCMModel.depth_sweep` for a pure NCM run) and `summary.json`
(CE per depth, K1-K6, oracle/null CE for an LCM run, the compare block).
  ce_pred   mse: D on the predicted concept. diffusion: -log mean_K p(span | z_k) over K sampled concepts
            (per span, then summed), plus `ce_pred_sample_mean` = mean over the K samples of the per-sample CE.
            pure NCM: the bound (code NLL + token NLL given the true codes) at m = gen_levels.
  lcm_vae   ce = the negative ELBO (one posterior sample + exact KL), ce_iwae = the K-sample importance-weighted bound,
            ce_prior_sample = -log mean_K p(span | c), c ~ prior (generation mode), ce_recon / ce_kl = the parts of ce.
  ce_oracle D on the true concept (the autoencoder ceiling).   ce_null   D on a zero concept (no context).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from lxtul.data import layout_for_cells, morph_cfg, tul_runtime
from lxtul.evaluate import _stats, eval_rows, plain_view, summarize
from lxtul.lcm import build_lcm, compact, rows_to, tul_target_mask
from lxtul.ncm import build_ncm


def run_config(run: Path):
    ck = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    return ck, OmegaConf.create(ck["config"])


def rows_runtime(cfg):
    """(mcfg, rt, eval_cells) of the rows a run was evaluated on (`train.py` / `build.py` rules)."""
    mcfg = morph_cfg(str(cfg.morph.config))
    rt = tul_runtime(mcfg)
    kind, eval_cells = str(cfg.arm.kind), 0
    if kind == "lxtul":
        ov = OmegaConf.to_container(cfg.arm.lxtul, resolve=True) if "lxtul" in cfg.arm else {}
        base = OmegaConf.to_container(cfg.lxtul, resolve=True)
        cells = int({**base, **ov}.get("cells", mcfg.tul.fan_k))
        rt = layout_for_cells(rt, cells)
        if {**base, **ov}.get("cell_count", "fixed") != "fixed":
            eval_cells = int({**base, **ov}.get("cell_count_eval", 0))
    elif kind in ("lcm", "lcm_ae", "lcm_vae"):
        rt = layout_for_cells(rt, int(cfg.arm.lcm.cells))
    elif kind in ("ncm", "ncm_tok"):
        rt = layout_for_cells(rt, int(cfg.arm.ncm.cells))
    return mcfg, rt, eval_cells


@torch.no_grad()
def tul_sums_on_set(model, rows, depths, S: int, device="cuda"):
    """Per-row CE sums and counts of a TUL (lxtul-kind) model over `tul_target_mask`, per depth."""
    model.eval()
    sums, counts = {k: [] for k in depths}, []
    for batch in rows:
        x, y, lay = rows_to(batch, device)
        sel = tul_target_mask(lay, S)
        counts.append(sel.sum(1).double().cpu())
        for k in depths:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model(x, None, lay, depth=k)["hidden"]
                nll = model.token_nll(h, x, y, lay.slot_mask, lay)
            sums[k].append((nll.float() * sel).sum(1).double().cpu())
    return {str(k): torch.cat(v).numpy() for k, v in sums.items()}, torch.cat(counts).numpy()


@torch.no_grad()
def gpt_sums_on_set(model, rows, depths, S: int, slot_id: int, device="cuda"):
    """Per-row CE sums and counts of a `gpt`-kind model (gpt8, gpt8_pointer) on the LCM scored set: the target tokens
    of complete spans with index >= 1. The gpt reads the token positions of the row (`plain_view`); its NLL at compact
    position t scores the label x_c[t + 1], so the set is the positions t whose label sits in a complete span with
    index >= 1 (the row's last token has no label in the row). The gpt is flat: one score serves every depth.
    Raises unless the gpt view IS the compact row and its label ids ARE the target tokens, row by row."""
    model.eval()
    sums, counts = [], []
    for batch in rows:
        x, y, lay = rows_to(batch, device)
        c = compact(x, lay, S)
        xp, yp = plain_view(x, y, lay, slot_id)
        if not torch.equal(xp, c.x):
            raise RuntimeError("the gpt view of the row is not the compact row")
        T = x.shape[1]
        n_all = (~lay.slot_mask).sum(1)
        lbag = c.bag[:, 1:]
        sel = (lbag >= 1) & (lbag < S) & (torch.arange(T - 1, device=x.device)[None] + 1 < n_all[:, None])
        if not torch.equal(yp[:, : T - 1][sel], c.x[:, 1:][sel]):
            raise RuntimeError("gpt label ids differ from the target tokens of the scored set")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if hasattr(model, "pointer"):
                nll = model.token_nll(xp, yp)
            else:
                lg = model(xp, return_logits=True)["logits"].float()
                lg[..., slot_id] = float("-inf")
                nll = F.cross_entropy(lg.flatten(0, 1), yp.clamp_min(0).flatten(), reduction="none").view(yp.shape)
        counts.append(sel.sum(1).double().cpu())
        sums.append((nll.float()[:, : T - 1] * sel).sum(1).double().cpu())
    s = torch.cat(sums).numpy()
    return {str(k): s for k in depths}, torch.cat(counts).numpy()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows-from", type=Path, default=None)
    ap.add_argument("--compare", type=Path, action="append", default=[],
                    help="a run to score on the same tokens; repeat the flag, one run each")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("runs", type=Path, nargs="+")
    return ap


def main() -> None:
    a = build_parser().parse_args()
    if 6 not in a.depths:
        raise ValueError("--depths must include 6 (the reference depth of K1-K6)")

    src_cfg = run_config(a.rows_from)[1] if a.rows_from else None
    cmp_runs = []
    for r in a.compare:
        from lxtul.gap_probe import load_run
        m, kind, cfg, rrt = load_run(r)
        if kind not in ("lxtul", "gpt"):
            raise ValueError(f"--compare {r.name}: kind {kind!r}; only lxtul-kind and gpt-kind runs can be scored on "
                             "this set")
        cmp_runs.append((r, m.cuda(), kind, int(rrt.data_cfg.prefix_k)))

    for run in a.runs:
        ck, cfg = run_config(run)
        kind = str(cfg.arm.kind)
        if kind == "ncm" and not bool(cfg.arm.ncm.coda_span_local):
            raise ValueError(f"{run}: a plan-arm ncm run scores every token label; pair it with lxtul.paired_eval")
        if kind not in ("lcm", "lcm_vae", "ncm"):
            raise ValueError(f"{run}: kind {kind!r}, expected a stage-2 'lcm' run, an 'lcm_vae' run or a pure 'ncm' run")
        mcfg, rt, eval_cells = rows_runtime(src_cfg if src_cfg is not None else cfg)
        if kind in ("lcm", "lcm_vae"):
            model, _ = build_lcm(cfg, mcfg, kind, compile=False, load=False)        # the ae weights come from final.pt
        else:
            model, _ = build_ncm(cfg, mcfg, "ncm", compile=False, load=False)       # the tokenizer comes from final.pt
        model.load_state_dict({k.replace("._orig_mod", ""): v for k, v in ck["model"].items()}, strict=True)
        model = model.cuda().eval()
        rows = eval_rows(mcfg, rt, a.rows, 6, eval_cells=eval_cells)
        sweep = model.depth_sweep(rows, a.depths)
        summ = summarize(sweep, ref=6)
        cnt = np.asarray(sweep["count"])
        n = float(cnt.sum())
        if kind == "lcm_vae":              # ce = the negative ELBO (an upper bound on CE); the rest are readings of it
            summ["ce_oracle"] = float(np.sum(sweep["oracle"]) / n)
            for key in ("iwae", "prior_sample", "recon", "kl"):
                summ[f"ce_{key}"] = {k: float(np.sum(v) / n) for k, v in sweep[key].items()}
            summ["loss"] = "elbo"
        elif kind == "lcm":
            summ["ce_oracle"] = float(np.sum(sweep["oracle"]) / n)
            summ["ce_null"] = float(np.sum(sweep["null"]) / n)
            summ["loss"] = str(cfg.arm.lcm.loss)
        else:
            summ["scalars"] = sweep["scalars"]
        if "sums_sample_mean" in sweep:
            summ["ce_pred_sample_mean"] = {k: float(np.sum(v) / n) for k, v in sweep["sums_sample_mean"].items()}
        summ["K1_minus_K6"] = summ["ce"]["1"] - summ["ce"]["6"]
        summ["scope"] = sweep["scope"]
        summ["scored_set"] = ("target tokens of complete spans with index >= 1; rows of "
                              f"{a.rows_from.name if a.rows_from else run.name} ({rt.data_cfg.prefix_k} cells per slot)")
        if cmp_runs:
            summ["compare"] = {}
            ci = _stats().paired_bootstrap_ci
            diff_key = "ncm_minus_run" if kind == "ncm" else "lcm_minus_run"
            for r, m, rkind, width in cmp_runs:
                if rkind == "lxtul":
                    if width != int(rt.data_cfg.prefix_k):
                        raise ValueError(f"--compare {r.name} has {width} cells per slot, the rows have {rt.data_cfg.prefix_k}")
                    t_sums, t_cnt = tul_sums_on_set(m, rows, a.depths, int(mcfg.tul.max_slots))
                else:
                    t_sums, t_cnt = gpt_sums_on_set(m, rows, a.depths, int(mcfg.tul.max_slots), int(rt.data_cfg.slot_id))
                if not np.array_equal(t_cnt, cnt):
                    raise RuntimeError(f"{r.name}: scored token counts differ from {run.name}'s on the same rows")
                summ["compare"][r.name] = {
                    "ce": {k: float(v.sum() / n) for k, v in t_sums.items()},
                    diff_key: {k: ci(np.asarray(sweep["sums"][k]), v, cnt) for k, v in t_sums.items()}}
        out = a.out / run.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.json").write_text(json.dumps(sweep))
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        keys = ("ce", "K1_minus_K6") + {"lcm": ("ce_oracle", "ce_null"), "lcm_vae": ("ce_oracle", "ce_iwae", "ce_prior_sample")}.get(kind, ())
        print(run.name, json.dumps({k: summ[k] for k in keys}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
