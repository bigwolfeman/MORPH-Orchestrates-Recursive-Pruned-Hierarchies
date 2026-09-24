"""LXTUL-E Stage 0: the width budget B0 of the frozen ruler's exit cell.

Spec: ``.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md``,
"The cheapest decisive test", acceptance clause P-S0. The two Stage 0 arms
(``morph/configs/tul_slot_spandec_strict_e0k{1,4}.yaml``) fit a parallel span head
(``morph/model/tul_spandec_parallel.py``) on the FROZEN ``slot-spandec-strict`` step-5000
model, one with no code (K = 1) and one with four enumerated codes under the exact mixture
likelihood (K = 4). This file scores both heads on the SAME validation spans::

    B0 = CE_par(K = 1) - CE_par(K = 4 mixture)        nats per span token

TERMS (one meaning each):

  z             the slot's exit state as the head reads it, ``model._readout(h_slots)``,
                captured at the model's own seam (``_tul_spandec_par_loss``) inside the
                ordinary EVAL forward (eval depth, no dropout). Computed ONCE per batch with
                the K = 4 model and handed to BOTH heads, so the pair differs in the head
                alone. The frozen tensors of the two checkpoints are checked byte-identical
                (and, with ``--ruler``, identical to the ruler's) before anything is scored.
  span token    a valid target of ``ParallelSpanHead.targets``: token j of span s+1 for a
                supervised slot s (the ruler decoder's own target, ``span_slots``).
  CE_par(K=1)   ``-log p(t_j | z, j)`` per span token, the K = 1 head.
  CE_par(mix)   the K = 4 mixture's NLL split over a span's tokens by the chain rule
                (``mixture_token_nll``): token j pays ``-log sum_k w_k(<j) p_k(t_j)`` with
                ``w`` the posterior over codes after the span's first j tokens. A slot's
                tokens sum to exactly its mixture term, so the overall mean IS the
                mixture CE and the per-offset mean is the sequential Bayes read.
  CE_tf         the frozen ruler decoder's TEACHER-FORCED CE on the same span tokens
                (``SpanDecoder.decode``: z plus the true prefix). A reference line: the
                hedging reader the Lean note says width cannot help.
  offset        j, the target token's index inside its span; binned by ``_earning.BINS``.
  block         the stream index of a SLOT's first target token // 1,024
                (``sweep_score.BLOCK``). A slot's tokens never split across two units.

STATISTICS. Every CE mean and B0 are token-weighted; the 95 % CI is the paired block
bootstrap of ``_stats.paired_bootstrap_ci`` (2,000 resamples, seed 0), the same resamples
for K = 1 and the mixture. Code usage on the held-out spans: the mean entropy of the
per-span responsibilities over codes, the fraction of spans each code wins, each code's
CE read alone, the best single code, and the mixture's gain over it.

SELF-CHECK. The model's own eval forward also computes the K = 4 term (``out["par_ce"]``)
and the ruler decoder's term (``out["spandec_ce"]``) on the same batch. The offline
recomputation must match both per batch (reported as ``max_abs_dev``; the run RAISES above
``--tol``), so the numbers here are the numbers the trainer optimised.

DECISION (the note's, reported, not acted on): ``B0 < 0.005`` stops the design; ``B0 >=
0.02`` is P-S0's prediction; in between, Stage 1 runs with its width priors halved.

Usage (GPU):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/lxtul_e_stage0_score.py \\
      --k1 checkpoints/morph/lxtul-e0k1/step_2000.pt \\
      --k4 checkpoints/morph/lxtul-e0k4/step_2000.pt --rows 480 --out .../stage0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _earning import BINS  # noqa: E402  (ONE home for the offset bins)
from _stats import paired_bootstrap_ci  # noqa: E402  (ONE home for the paired bootstrap)
from sweep_score import BLOCK  # noqa: E402  (1,024 stream tokens per bootstrap unit)

RULER = "/home/wolfe/morph-to/checkpoints/morph/slot-spandec-strict/step_5000.pt"
HEAD = "tul_spandec_par."
N_BOOT = 2000


def _canon(k: str) -> str:
    return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")


# ── the frozen model is the same in both arms ───────────────────────────────────────

def check_frozen_identical(m1, m4, ruler_path: str | None) -> dict:
    """Byte-compare every non-head tensor of the two loaded models, and of the K = 4 model
    against the ruler checkpoint. RAISES on any difference: a Stage 0 arm whose frozen
    model moved is not measuring the ruler's cell."""
    s1 = {_canon(k): v for k, v in m1.state_dict().items()}
    s4 = {_canon(k): v for k, v in m4.state_dict().items()}
    base = sorted(k for k in s4 if not k.startswith(HEAD))
    if sorted(k for k in s1 if not k.startswith(HEAD)) != base:
        raise RuntimeError("the two Stage 0 models do not hold the same frozen tensor set")
    diff = [k for k in base if not torch.equal(s1[k], s4[k])]
    if diff:
        raise RuntimeError(f"{len(diff)} frozen tensor(s) differ between the K = 1 and K = 4 "
                           f"checkpoints: {diff[:6]}")
    out = {"n_frozen_tensors": len(base), "k1_vs_k4": "identical"}
    if ruler_path:
        ck = torch.load(ruler_path, map_location="cpu", weights_only=False, mmap=True)
        rs = {_canon(k): v for k, v in ck["model"].items()}
        miss = [k for k in base if k not in rs]
        if miss:
            raise RuntimeError(f"{len(miss)} frozen tensor(s) are absent from the ruler "
                               f"checkpoint: {miss[:6]}")
        bad = [k for k in base if not torch.equal(s4[k].cpu(), rs[k].to(s4[k].dtype))]
        if bad:
            raise RuntimeError(f"{len(bad)} frozen tensor(s) differ from the ruler "
                               f"{ruler_path}: {bad[:6]}")
        out["vs_ruler"] = "identical"
        out["ruler"] = ruler_path
    return out


# ── one batch ───────────────────────────────────────────────────────────────────────

class _Seam:
    """Capture ``(h_slots, input_ids, layout)`` at ``_tul_spandec_par_loss``, the model's own
    seam, for the duration of the block. Counts its calls; the caller raises on zero."""

    def __init__(self, model):
        self.model, self.got = model, []

    def __enter__(self):
        real = self.model._tul_spandec_par_loss

        def wrap(h_slots, input_ids, layout, stats=None):
            self.got.append((h_slots, input_ids, layout))
            return real(h_slots, input_ids, layout, stats=stats)
        self.model._tul_spandec_par_loss = wrap
        return self

    def __exit__(self, *exc):
        del self.model._tul_spandec_par_loss
        return False


@torch.no_grad()
def score_batch(m4, head1, inp, labels, layout, idx, device: str) -> dict:
    """Per-token arrays for one packed batch, plus the self-check deviations."""
    from morph.model.fused_ce import fused_linear_label_logprob
    from morph.model.tul_spandec import span_slots
    from morph.model.tul_spandec_parallel import mixture_token_nll
    head4 = m4.tul_spandec_par
    dec = m4.tul_spandec
    tc = m4.cfg.tul
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    with ac, _Seam(m4) as seam:
        out = m4(inp.to(device), labels=labels.to(device), slot_layout=layout)
    if len(seam.got) != 1:
        raise RuntimeError(f"the parallel-head seam was reached {len(seam.got)} times")
    h_slots, input_ids, lay = seam.got[0]
    with ac:
        z = m4._readout(h_slots)                                      # [B, S, C]
        B, S, C = z.shape
        ids, valid = head4.targets(input_ids, lay)
        w = m4.embed.lm_weight().detach()
        K = head4.n_codes
        lp4, sup, val_s = head4.token_logp(
            head4.with_codes(z.reshape(B * S, C)).view(K, B, S, C), ids, valid, w,
            chunk_size=m4.cfg.ce_chunk_size, mask_token_id=tc.slot_id)
        lp1, sup1, val1 = head1.token_logp(z.unsqueeze(0), ids, valid, w,
                                           chunk_size=m4.cfg.ce_chunk_size,
                                           mask_token_id=tc.slot_id)
        st = dec.decode(z, ids, valid, w)                              # [B, S, J, C]
        lp_tf = fused_linear_label_logprob(st[valid], w, ids[valid],
                                           chunk_size=m4.cfg.ce_chunk_size,
                                           mask_token_id=tc.slot_id)
    if not (torch.equal(sup, sup1) and torch.equal(val_s, val1)):
        raise RuntimeError("the two heads selected different span tokens")
    mix_tok = mixture_token_nll(lp4.float(), val_s)                    # [Nv]
    n = int(val_s.sum())
    dev_par = abs(float(mix_tok.double().sum() / n) - float(out["par_ce"]))
    dev_tf = abs(float(-lp_tf.double().sum() / n) - float(out["spandec_ce"]))
    # stream index of each target token: the same scatter the target takes, applied to the
    # row's stream map (a slot position holds -1 there and is never a target)
    sidx, _v = span_slots(idx.to(ids.device), lay, int(ids.shape[-1]),
                          shift=head4.target_offset)
    J = int(ids.shape[-1])
    off = torch.arange(J, device=ids.device).view(1, 1, J).expand_as(ids)
    # per-slot bookkeeping: slot number in batch order, and its first token's stream index
    slot_of = torch.arange(int(sup.sum()), device=ids.device).view(-1, 1).expand_as(val_s)
    first = sidx[sup][:, 0]
    tok_slot = slot_of[val_s].cpu().numpy()
    lp4_np = lp4.double().cpu().numpy()
    M = int(sup.sum())
    # per-slot span log-likelihood per code, summed on the host in fp64 (no atomics)
    per_slot_lp4 = np.stack([np.bincount(tok_slot, weights=lp4_np[k], minlength=M)
                             for k in range(K)])
    return {
        "tok_index": sidx[valid].cpu().numpy(),
        "offset": off[valid].cpu().numpy(),
        "slot": tok_slot,
        "slot_first": first.cpu().numpy(),
        "ce_k1": (-lp1[0]).float().cpu().numpy(),
        "ce_mix": mix_tok.float().cpu().numpy(),
        "ce_code": (-lp4).float().cpu().numpy(),                        # [K, Nv]
        "ce_tf": (-lp_tf).float().cpu().numpy(),
        "slot_logp": per_slot_lp4,                                      # [K, M]
        "dev_par": dev_par, "dev_tf": dev_tf,
    }


# ── statistics ──────────────────────────────────────────────────────────────────────

def _block_sums(vals: np.ndarray, blocks: np.ndarray, sel: np.ndarray, nb: int
                ) -> tuple[np.ndarray, np.ndarray]:
    s = np.bincount(blocks[sel], weights=vals[sel], minlength=nb)
    c = np.bincount(blocks[sel], minlength=nb).astype(np.float64)
    return s, c


def _mean_ci(a: np.ndarray, b: np.ndarray | None, blocks: np.ndarray, sel: np.ndarray,
             n_boot: int, seed: int) -> dict:
    """Token-weighted ``mean(a) - mean(b)`` (``b = None``: ``mean(a)``) over ``sel`` with the
    paired block-bootstrap CI. Blocks that hold no selected token are dropped as units."""
    nb = int(blocks.max()) + 1
    sa, c = _block_sums(a, blocks, sel, nb)
    sb = (np.zeros_like(sa) if b is None else _block_sums(b, blocks, sel, nb)[0])
    keep = c > 0
    if not keep.any():
        return {"point": None, "lo": None, "hi": None, "n_units": 0, "n_tokens": 0}
    r = paired_bootstrap_ci(sa[keep], sb[keep], c[keep], n_boot=n_boot, seed=seed)
    r["n_tokens"] = int(sel.sum())
    return r


def stage0_stats(arr: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """B0 overall and per offset bin, the three CE lines, and the code usage."""
    blocks_raw = arr["slot_first"][arr["slot"]] // BLOCK
    _u, blocks = np.unique(blocks_raw, return_inverse=True)
    allsel = np.ones(arr["ce_k1"].shape[0], dtype=bool)
    res: dict = {"n_tokens": int(allsel.sum()), "n_spans": int(arr["slot_logp"].shape[1]),
                 "n_blocks": int(blocks.max()) + 1}
    res["B0"] = _mean_ci(arr["ce_k1"], arr["ce_mix"], blocks, allsel, n_boot, seed)
    res["ce_par_k1"] = _mean_ci(arr["ce_k1"], None, blocks, allsel, n_boot, seed)
    res["ce_par_mix"] = _mean_ci(arr["ce_mix"], None, blocks, allsel, n_boot, seed)
    res["ce_tf_ruler"] = _mean_ci(arr["ce_tf"], None, blocks, allsel, n_boot, seed)
    res["by_offset_bin"] = []
    for lo, hi in BINS:
        sel = (arr["offset"] >= lo) & (arr["offset"] <= hi)
        res["by_offset_bin"].append({
            "bin": [lo, hi], "n_tokens": int(sel.sum()),
            "B0": _mean_ci(arr["ce_k1"], arr["ce_mix"], blocks, sel, n_boot, seed),
            "ce_par_k1": _mean_ci(arr["ce_k1"], None, blocks, sel, n_boot, seed),
            "ce_par_mix": _mean_ci(arr["ce_mix"], None, blocks, sel, n_boot, seed),
            "ce_tf_ruler": _mean_ci(arr["ce_tf"], None, blocks, sel, n_boot, seed)})
    # code usage on the held-out spans
    lp = arr["slot_logp"]                                              # [K, M]
    K = lp.shape[0]
    mx = lp.max(0, keepdims=True)
    resp = np.exp(lp - mx)
    resp /= resp.sum(0, keepdims=True)
    ent = -(resp * np.log(np.clip(resp, 1e-300, None))).sum(0)
    wins = np.bincount(lp.argmax(0), minlength=K) / lp.shape[1]
    ce_code = [float(arr["ce_code"][k].mean()) for k in range(K)]
    best = int(np.argmin(ce_code))
    res["codes"] = {
        "K": K,
        "resp_entropy_mean": float(ent.mean()), "resp_entropy_max": math.log(K),
        "win_frac": [float(x) for x in wins],
        "ce_code": ce_code, "best_code": best,
        "mix_gain_over_best_code": _mean_ci(arr["ce_code"][best], arr["ce_mix"], blocks,
                                            allsel, n_boot, seed),
    }
    b0 = res["B0"]
    if b0["point"] < 0.005:
        verdict = "stop: B0 < 0.005 (the note stops the design)"
    elif b0["point"] >= 0.02:
        verdict = "P-S0 holds: B0 >= 0.02"
    else:
        verdict = "between: Stage 1 runs with its width priors halved"
    res["decision"] = {"verdict": verdict,
                       "ci_clear_of_0.005": bool(b0["lo"] >= 0.005 or b0["hi"] < 0.005),
                       "ci_clear_of_0.02": bool(b0["lo"] >= 0.02 or b0["hi"] < 0.02)}
    return res


def _fmt(d: dict) -> str:
    if d.get("point") is None:
        return "n/a"
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['n_tokens']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k1", required=True, help="the K = 1 arm's checkpoint")
    ap.add_argument("--k4", required=True, help="the K = 4 arm's checkpoint")
    ap.add_argument("--config-k1", default="tul_slot_spandec_strict_e0k1")
    ap.add_argument("--config-k4", default="tul_slot_spandec_strict_e0k4")
    ap.add_argument("--ruler", default=RULER,
                    help="the ruler checkpoint the frozen tensors must equal; 'none' skips")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max |offline - forward| per batch for par_ce and spandec_ce")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ovr", action="append", default=[],
                    help="extra Hydra override for BOTH arms (repeatable). A CPU run needs "
                         "model.tg_scoped_kernels=false and model.hc_use_kernel=false (the "
                         "Triton kernels have no CPU driver); neither changes a tensor.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    from _build import ROOT, build_cfg
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    def _abs(p: str) -> str:
        return p if p.startswith("/") else os.path.join(ROOT, p)

    models, steps = {}, {}
    for key, config, path in (("k1", a.config_k1, _abs(a.k1)), ("k4", a.config_k4, _abs(a.k4))):
        cfg = build_cfg(config, ["model.use_kernels=false", *a.ovr])
        rt = build_tul_runtime(cfg)
        if rt is None or not rt.model_cfg.spandec_parallel:
            raise SystemExit(f"{config} is not a parallel-head arm")
        m, step = load_ckpt(cfg, path, a.device, rt.model_cfg)
        models[key], steps[key] = m.eval(), step
        if key == "k4":
            cfg4, rt4 = cfg, rt
    if models["k1"].tul_spandec_par.n_codes != 1 or models["k4"].tul_spandec_par.n_codes < 2:
        raise SystemExit("--k1 must be a K = 1 head and --k4 a K > 1 head")
    frozen = check_frozen_identical(models["k1"], models["k4"],
                                    None if a.ruler == "none" else _abs(a.ruler))
    head1 = models["k1"].tul_spandec_par
    m4 = models["k4"]
    loader = create_dataloader(cfg4.data.tokenizer, cfg4.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = rt4.data_cfg.spec_for(cfg4.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, rt4, cfg4, a.batch, False)[:n_batches]
    rows = sum(b[0].shape[0] for b in batches)
    if rows < a.rows:
        raise SystemExit(f"packed {rows} rows, asked for {a.rows}")
    t1 = time.time()
    parts: dict[str, list] = {}
    devs = {"par": 0.0, "tf": 0.0}
    slot_base = 0
    for inp, labels, layout, idx in batches:
        r = score_batch(m4, head1, inp, labels, layout.to(a.device), idx, a.device)
        devs["par"] = max(devs["par"], r.pop("dev_par"))
        devs["tf"] = max(devs["tf"], r.pop("dev_tf"))
        r["slot"] = r["slot"] + slot_base
        slot_base += r["slot_logp"].shape[1]
        for k, v in r.items():
            parts.setdefault(k, []).append(v)
    if devs["par"] > a.tol or devs["tf"] > a.tol:
        raise RuntimeError(f"offline recomputation disagrees with the model's own forward: "
                           f"par_ce {devs['par']:.2e}, spandec_ce {devs['tf']:.2e} (tol "
                           f"{a.tol})")
    arr = {k: np.concatenate(v, axis=-1) for k, v in parts.items()}
    res = stage0_stats(arr, n_boot=a.n_boot, seed=a.seed)
    res.update({"k1": _abs(a.k1), "k4": _abs(a.k4), "step_k1": steps["k1"],
                "step_k4": steps["k4"], "configs": [a.config_k1, a.config_k4],
                "rows": rows, "batch": a.batch, "ovr": a.ovr, "frozen": frozen,
                "self_check_max_abs_dev": devs, "block_tokens": BLOCK,
                "wall_s": {"load": round(t1 - t0, 1), "score": round(time.time() - t1, 1)}})
    npz = a.out.rsplit(".", 1)[0] + ".tokens.npz"
    np.savez_compressed(npz, tok_index=arr["tok_index"].astype(np.int64),
                        offset=arr["offset"].astype(np.int16),
                        slot=arr["slot"].astype(np.int32),
                        slot_first=arr["slot_first"].astype(np.int64),
                        ce_k1=arr["ce_k1"], ce_mix=arr["ce_mix"], ce_tf=arr["ce_tf"],
                        ce_code=arr["ce_code"], slot_logp=arr["slot_logp"])
    res["tokens_npz"] = npz
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"Stage 0: {res['n_spans']} spans, {res['n_tokens']} span tokens, "
          f"{res['n_blocks']} blocks, rows {rows}; frozen {frozen}")
    print(f"  CE_par K=1   {_fmt(res['ce_par_k1'])}")
    print(f"  CE_par mix   {_fmt(res['ce_par_mix'])}")
    print(f"  CE_tf ruler  {_fmt(res['ce_tf_ruler'])}")
    print(f"  B0           {_fmt(res['B0'])}")
    for b in res["by_offset_bin"]:
        print(f"    offset {b['bin'][0]:>2}-{min(b['bin'][1], 99):<2} B0 {_fmt(b['B0'])}")
    c = res["codes"]
    print(f"  codes: resp entropy {c['resp_entropy_mean']:.4f} of {c['resp_entropy_max']:.4f}, "
          f"wins {[round(x, 3) for x in c['win_frac']]}, ce_code "
          f"{[round(x, 4) for x in c['ce_code']]}, mix gain over best "
          f"{_fmt(c['mix_gain_over_best_code'])}")
    print(f"  self-check max |dev|: {devs}")
    print(f"  DECISION: {res['decision']}")
    print(f"wrote {a.out} and {npz}")


if __name__ == "__main__":
    main()
