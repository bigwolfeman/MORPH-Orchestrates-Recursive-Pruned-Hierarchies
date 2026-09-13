"""Why the trainer's `val/slot_eff_rank` does not reproduce on the checkpoint.

`slot-spandec-strict` logs `val/slot_eff_rank` 5.7598 / `val/slot_pairwise_cos` 0.7104 on
the last `[VAL]` line of its run, and the same probe on `step_5000.pt` reads 13.85 / 0.52
(`lab/experiments/results/2026-09-13-rank-anatomy/`). This instrument scores the SAME
checkpoint on the SAME rows under the candidate causes, one variant per column, so the
gap is attributed to a mechanism instead of argued.

THE VARIANTS (all on `model.eval()`, the trainer's mode inside `evaluate`)

  shipped   `tul_slot_state_probe` as it is at HEAD: the front runs under the model's OWN
            TG relation (`_tul_tg_kwargs`).
  bare      the probe as it was BEFORE commit 7a24adf (2026-09-13 08:01): a bare
            `_tul_front(input_ids, layout)`. On a `tg_geometry="strict"` (or
            `tg_restrict_scope="all"`) model that runs the PRELUDE UNRESTRICTED — the
            model was trained same-span-only and every state downstream is
            off-distribution. This is the code the 2026-09-12 runs logged from.
  bf16      inside `torch.autocast("cuda", bfloat16)` — the rank-anatomy instrument's
            context.
  fp32      outside it — `morph/training/train.py::evaluate` closes the autocast block
            after `tul_forward_with_plan_nats` and calls the probe in plain fp32, so
            `bare_fp32` is the trainer's exact context.

THE ROWS. The trainer's own val loader (`create_dataloader(..., seq_len, batch_size,
split="validation", skip_samples=50_000, tul=val_data_cfg)`), not the probe family's
stream. With no curriculum, `_curr_val_batches` is None and the loader is NOT rebuilt
before each eval, so the periodic evals walk the stream: at `eval_every=250`,
`n_eval_batches=20` over 5000 steps, 20 evals consume 400 batches and the FINAL eval —
the one whose numbers reach the `Final val_loss=` line — reads batches 400..419.
`--stream-offsets 0,400` scores both, so a stream-position effect cannot hide inside a
code effect.

PART 2 — where the prefix write loses the rank. `W_prefix` is `[K, d, d]` on a non-Linear
module, so no QAT scope touches it (`morph/model/ternary_qat.py` quantises `nn.Linear` /
`nn.Embedding` / CMS only); this prints its spectrum and checks the value set. The write
stage of the rank anatomy is `h_s W_k` for every slot and every k, so with the exit's
stream-mean states in hand the whole stage is reproducible offline: per-row centered
participation ratio of the exit, of each `W_k`'s image, and of the two concatenated (the
instrument's `sW_write`), in the raw and the readout view, plus the gain `W_k` puts on the
row's MEAN direction against the gain it puts on the centered residual — which is what
decides whether the drop is the matrix's own low rank or a re-inflated shared offset.

Usage:
  python lab/divergence/slot_rank_probe_variants.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --batches 20 --batch 6 --stream-offsets 0,400 --out results/variants.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg                                   # noqa: E402
from slot_rank_anatomy import rank_stats, readout_view, quantiles     # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")


# ── the probe, both versions ─────────────────────────────────────────────────

@torch.no_grad()
def probe(model, inp, layout, bare: bool) -> dict:
    """`tul_slot_state_probe`'s body, with the front's TG relation as the one variable.

    Mirrors HEAD's body for the ruler family only: a `cond_layers`, `vq_codes`,
    `center_exit` or `slot_cells > 1` model takes extra stages inside the shipped probe
    and this copy would silently skip them, so it RAISES instead.
    """
    from morph.model.fm_planner import effective_rank, mean_pairwise_cos

    for attr in ("tul_cond", "tul_vq", "tul_center"):
        if getattr(model, attr, None) is not None:
            raise SystemExit(f"this variant probe does not mirror {attr}; use the shipped one")
    if int(model.cfg.tul.slot_cells) != 1:
        raise SystemExit("this variant probe does not mirror the register (slot_cells > 1)")

    if bare:
        fkw = freset = None
    else:
        fkw, freset, _c, _cr = model._tul_tg_kwargs(layout)
    x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    _xn, h_slots, _d, _g, *_ = model._tul_core(x, x0, bigram, layout, input_ids=inp)
    z = model._readout(h_slots).float()
    valid = layout.slot_valid
    rows = z[valid]
    z_cpu, valid_cpu = z.cpu(), valid.cpu()
    return {
        "slot_eff_rank": effective_rank(z_cpu, valid_cpu),
        "slot_pairwise_cos": mean_pairwise_cos(z_cpu, valid_cpu),
        "slot_norm_mean": float(rows.norm(dim=-1).mean()),
        "slot_component_std": float(rows.std()),
        "slot_component_mean": float(rows.mean()),
    }


@torch.no_grad()
def exit_states(model, inp, layout) -> torch.Tensor:
    """The loop exit `[B, S, C]` (stream mean), at the model's own relation."""
    fkw, freset, _c, _cr = model._tul_tg_kwargs(layout)
    x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    _xn, h_slots, *_ = model._tul_core(x, x0, bigram, layout, input_ids=inp)
    return h_slots


# ── part 2 ───────────────────────────────────────────────────────────────────

def matrix_spectrum(w: torch.Tensor) -> dict:
    """Singular spectrum of one `[d, d]` matrix, through the symmetric eigenproblem."""
    w = w.double()
    lam = torch.linalg.eigvalsh(w.T @ w).clamp_min(0.0)          # = sigma^2
    s = lam.sqrt()
    s1, s2 = float(lam.sum()), float((lam * lam).sum())
    sv = s.sort(descending=True).values
    return {
        "eff_rank_sigma2": (s1 * s1 / s2) if s2 > 0 else 0.0,
        "sigma_max": float(sv[0]), "sigma_min": float(sv[-1]),
        "sigma_mean": float(sv.mean()),
        "top1_share_sigma2": float(lam.max()) / s1 if s1 > 0 else 0.0,
        "top8_share_sigma2": float(sv[:8].pow(2).sum()) / s1 if s1 > 0 else 0.0,
        "fro": float(w.norm()),
        "n_distinct_values": int(torch.unique(w).numel()),
        "frac_zero": float((w == 0).double().mean()),
        "offdiag_over_diag": float(
            (w - torch.diag(torch.diagonal(w))).norm() / w.diagonal().norm().clamp_min(1e-12)),
    }


def write_stage(model, h_exit_raw: torch.Tensor, valid: torch.Tensor, wp: torch.Tensor,
                device: str) -> dict:
    """Per-row rank of the exit, of each `W_k` image, and of the K images concatenated.

    `h_exit_raw` is `[B, S, C]` (the anatomy's raw view: the carrier's stream mean), so
    `mean_streams(h W_k) == (mean_streams h) W_k` and the write stage is exact here.
    """
    K = wp.shape[0]
    series: dict[str, dict[str, list[float]]] = {}

    @torch.no_grad()
    def ro(m: torch.Tensor) -> torch.Tensor:
        """The readout view, on the model's device, back on the CPU."""
        return readout_view(model, m.to(device)).float().cpu()

    def add(name: str, view: str, m: torch.Tensor):
        if m.shape[0] < 2:
            return
        for k, v in rank_stats(m).items():
            series.setdefault(f"{name}/{view}", {}).setdefault(k, []).append(float(v))

    gains: dict[str, list[float]] = {}
    for b in range(h_exit_raw.shape[0]):
        sel = valid[b]
        h = h_exit_raw[b][sel]                                   # [n, C]
        if h.shape[0] < 2:
            continue
        add("exit", "raw", h)
        add("exit", "readout", ro(h))
        imgs = []
        mu = h.double().mean(0)
        resid = h.double() - mu
        for k in range(K):
            p = h @ wp[k].to(h.dtype)
            imgs.append(p)
            add(f"W{k}", "raw", p)
            add(f"W{k}", "readout", ro(p))
            wk = wp[k].double()
            gains.setdefault(f"W{k}/gain_mean_dir", []).append(
                float((mu @ wk).norm() / mu.norm().clamp_min(1e-12)))
            gains.setdefault(f"W{k}/gain_resid", []).append(
                float((resid @ wk).norm() / resid.norm().clamp_min(1e-12)))
        cat = torch.cat(imgs, dim=0)
        add("write_cat", "raw", cat)
        add("write_cat", "readout", ro(cat))
        # The concatenated stage minus each CELL's own mean. `write_cat` is centered by
        # the mean of all K*n vectors, so a per-cell offset (`mu W_0` != `mu W_1`) enters
        # its spectrum as one big direction shared by every vector of a cell, with the
        # sign flipping between cells. Removing each cell's own mean deletes exactly that
        # term and nothing else, so the difference between these two rows IS the
        # inter-cell offset's cost in participation ratio.
        cc = torch.cat([p - p.mean(0, keepdim=True) for p in imgs], dim=0)
        add("write_cat_cellcentered", "raw", cc)
        add("write_cat_cellcentered", "readout", ro(cc))
        d01 = (mu @ wp[0].double()) - (mu @ wp[1].double())
        gains.setdefault("cell_offset_norm", []).append(float(d01.norm()))
        gains.setdefault("cell_offset_over_resid_rms", []).append(
            float(d01.norm() / (resid.norm() / (resid.shape[0] ** 0.5)).clamp_min(1e-12)))
    out = {k: {kk: quantiles(vv) for kk, vv in v.items()} for k, v in series.items()}
    out["gains"] = {k: quantiles(v) for k, v in gains.items()}
    return out


# ── the run ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="label=config=path")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=0, help="0 = the config's training.batch_size")
    ap.add_argument("--stream-offsets", default="0",
                    help="comma-separated batches to DROP before scoring")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-write-stage", action="store_true")
    ap.add_argument("--skip-variants", action="store_true",
                    help="skip the four probe readings (for a CE-only control)")
    ap.add_argument("--ce", action="store_true",
                    help="also score the TRAINER's val CE on these rows — the control "
                         "for 'the checkpoint holds different weights than the run's "
                         "live ones': if the CE reproduces and the rank does not, the "
                         "weights are not the cause")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    label, config, path = a.ckpt.split("=", 2)
    t0 = time.time()
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit(f"{label}: needs a SLOT-LOOP model (tul.tokens_through_core false)")
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    assert not model.training
    batch = a.batch or int(cfg.training.batch_size)
    offsets = [int(v) for v in a.stream_offsets.split(",")]
    tc = model.cfg.tul
    print(f"{label}: step {step}  batch {batch}  seq {cfg.data.seq_len}  "
          f"tg_geometry={tc.tg_geometry} tg_restrict={tc.tg_restrict} "
          f"scope={tc.tg_restrict_scope} prefix_k={tc.prefix_k} "
          f"slot_cells={tc.slot_cells}", flush=True)

    out: dict = {"label": label, "config": config, "ckpt": path, "step": step,
                 "batch": batch, "n_batches": a.batches,
                 "seq_len": int(cfg.data.seq_len),
                 "tg_geometry": str(tc.tg_geometry), "prefix_k": int(tc.prefix_k),
                 "variants": {}, "wall_s": None}

    for off in offsets:
        loader = iter(create_dataloader(
            cfg.data.tokenizer, cfg.data.dataset, cfg.data.seq_len, batch,
            split="validation", skip_samples=50_000, tul=tul_rt.val_data_cfg))
        for i in range(off):
            next(loader)
            if i % 100 == 0:
                print(f"  [offset {off}] dropped {i} batches "
                      f"({time.time() - t0:.0f}s)", flush=True)
        acc: dict[str, list[float]] = {}
        h_rows: list[torch.Tensor] = []
        v_rows: list[torch.Tensor] = []
        for i in range(a.batches):
            x, _y, layout = next(loader)
            x, layout = x.to(a.device), layout.to(a.device)
            if a.ce:
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=a.device == "cuda"):
                    o = model.tul_forward_with_plan_nats(x, _y.to(a.device), layout)
                # evaluate()'s val loss: the model CE, every auxiliary term removed.
                _l = float(o["loss"])
                for _k in ("mux_weighted", "sigreg_weighted", "gain_reg_weighted",
                           "spandec_weighted", "egrad_weighted", "pass_res_weighted",
                           "oracle_z_weighted", "spandec_pass_weighted",
                           "coda_span_weighted", "core_token_aux_weighted",
                           "critic_weighted", "vq_weighted", "row_contrast_weighted",
                           "fm_weighted", "fm_sigreg_weighted"):
                    if o.get(_k) is not None:
                        _l -= float(o[_k])
                acc.setdefault("ce/val_loss", []).append(_l)
                acc.setdefault("ce/ce_tokens", []).append(float(o["ce_tokens"]))
                acc.setdefault("ce/spandec_ce", []).append(float(o.get("spandec_ce", 0.0)))
            for bare in (() if a.skip_variants else (False, True)):
                for dt in ("bf16", "fp32"):
                    name = f"{'bare' if bare else 'shipped'}_{dt}"
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(dt == "bf16" and a.device == "cuda")):
                        r = probe(model, x, layout, bare=bare)
                    for k, v in r.items():
                        acc.setdefault(f"{name}/{k}", []).append(float(v))
            if off == offsets[0] and not a.skip_write_stage:
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=a.device == "cuda"):
                    h = exit_states(model, x, layout)
                raw = (h.mean(dim=2) if h.dim() == 4 else h).float()
                h_rows.append(raw.cpu())
                v_rows.append(layout.slot_valid.cpu())
            if i % 5 == 0:
                live = "  ".join(f"{k.split('/')[0]} {v[-1]:.4f}" for k, v in acc.items()
                                 if k.endswith("slot_eff_rank") or k == "ce/ce_tokens")
                print(f"  [offset {off}] batch {i}/{a.batches} {live} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        out["variants"][f"offset{off}"] = {k: sum(v) / len(v) for k, v in acc.items()}
        print(f"[offset {off}] " + "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(out["variants"][f"offset{off}"].items())
            if k.endswith(("slot_eff_rank", "slot_pairwise_cos"))
            or k.startswith("ce/")), flush=True)

        if h_rows and not a.skip_write_stage:
            wp = model.tul.W_prefix.detach()
            out["W_prefix"] = {
                "shape": list(wp.shape), "dtype": str(wp.dtype),
                "requires_grad": bool(model.tul.W_prefix.requires_grad),
                "is_parametrized": bool(getattr(model.tul, "parametrizations", None)
                                        and "W_prefix" in model.tul.parametrizations),
                "per_k": [matrix_spectrum(wp[k].float().cpu()) for k in range(wp.shape[0])],
            }
            model_cpu_wp = wp.float().cpu()
            out["write_stage"] = write_stage(model, torch.cat(h_rows, 0),
                                             torch.cat(v_rows, 0), model_cpu_wp,
                                             a.device)
            print("  W_prefix:", json.dumps(out["W_prefix"]["per_k"], indent=1), flush=True)
            for k, v in out["write_stage"].items():
                if k == "gains":
                    continue
                print(f"  {k}: eff_rank_raw {v['eff_rank_raw']['median']:.3f} "
                      f"eff_rank_centered {v['eff_rank_centered']['median']:.3f} "
                      f"cos_raw {v['cos_raw']['median']:.3f} "
                      f"norm_mean {v['norm_mean']['median']:.3f}", flush=True)
            for k, v in out["write_stage"]["gains"].items():
                print(f"  {k}: {v['median']:.4f}", flush=True)

    out["wall_s"] = time.time() - t0
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out, f"({out['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
