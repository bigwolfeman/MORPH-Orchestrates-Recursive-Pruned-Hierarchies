"""LXTUL-G instruments: the exposure gap, the Bayesian per-token read over N prior samples.

Prereg: lab/experiments/failures/2026-09-23-lxtul-g-panel.md (Instruments). Design:
/home/wolfe/morph-scratch/arc/notes/2026-09-23-lxtul-g-paper-synthesis.md, Decisions 6-7.
Model: morph/model/tul_gram.py (`tul.gram`).

On the first ``--rows`` rows of the validation stream (the SAME stream, packer and row cut
``core_depth_sweep.py`` reads, so the per-token arrays pair with a sweep's by stream index),
it reports as JSON:

    ce_post          the coda on ONE posterior sample (gram_mode="post"; reads the answer —
                     never quote it alone)
    ce_prior@1       the coda on ONE prior sample (seed ``--seed``)
    ce_prior_mean    the coda on the prior MEAN (gram_mode="mean", no noise)
    ce_iw@N          the exact per-token Bayesian read over N prior samples (seeds
                     seed .. seed+N-1). Per sample n the coda is decoded ONCE; at token j of
                     a span w_n(j) = softmax_n( sum_{i<j, same span} log p_n(tok_i) ) and
                     p(tok_j) = sum_n w_n(j) p_n(tok_j). Causal (earlier tokens only); summed
                     over a span it equals log (1/N) sum_n prod_j p_n(tok_j), the
                     multi-sample bound. A "span" is a run of token positions with one
                     `bag_id`, so the weights reset at every slot; tokens before the first
                     slot read no cell and come out sample-independent.
    ce_iw_identity@4 the identity null: 4 COPIES of sample 0 through the same read. Must
                     equal ce_prior@1 exactly, and the script ASSERTS it (an aggregator that
                     reads a gain on copies is broken; Synthesis Controls A3).
    ce_zero          the written cells zeroed (plan_mode="zero"), the worth floor
    ce_elbo          ce_post + (sum over slots and passes of KL) / n_tokens
    kl               per-pass KL mean / p50 / p90 over the slots that reach the pass, the
                     per-slot sum's mean / p50 / p90, from the post-mode forward
    sigma_ratio      sigma / r of the prior (every mode) and of the posterior (post mode)

and saves per-token arrays (``ce_<key>`` keyed by the stream index ``tok_index``, the
``core_depth_sweep.py`` convention) to an npz beside the JSON; the JSON carries
``tokens_npz`` so ``sweep_score.tokens`` / ``sweep_score.paired`` read it unchanged
(``paired(a, "iw4", b, "6")`` pairs this read against a sweep's depth-6 column).

``--depths 1,6`` also reads ce_prior@1 and ce_iw@4 at each forced depth (the depth x
width corner, Synthesis B3), forcing depth the way ``core_depth_sweep.py`` does
(``slot_mean_depth`` / ``slot_max_depth`` / ``slot_depth_fixed``). The eval noise at pass t
is a function of (seed, t) only, so the depths are paired per pass.

Usage:
  python lab/divergence/lxtul_g_probe.py --ckpt checkpoints/morph/lxtul-g/step_5000.pt \\
      --config tul_slot_spandec_strict_gram --rows 192 --out .../lxtul-g_5000.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

N_LIST = (1, 4, 16)


# ── the Bayesian per-token read ─────────────────────────────────────────────────────

def bayes_read(lp: np.ndarray, bag: np.ndarray) -> np.ndarray:
    """``lp`` ``[N, n_tok]`` float64 log p of each scored token under sample n, in row
    order; ``bag`` ``[n_tok]`` its span id. Returns ``[n_tok]`` log p of the read.

    Written so that N identical rows give back the row EXACTLY (the identity null): the
    weights are ``exp(A - max A) / sum``, which is exactly ``1/N`` on ties, and the mixture
    is taken as ``M + log(sum_n w_n exp(lp_n - M))`` with ``M = max_n lp_n``, which is
    ``M + log(1)`` = ``M`` on ties. ``N = 1`` gives ``lp[0]`` exactly for the same reason.
    """
    N, n = lp.shape
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    # contiguous runs of one span id
    cut = np.flatnonzero(np.diff(bag) != 0) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [n]])
    for a, b in zip(starts, ends):
        seg = lp[:, a:b]                                               # [N, m]
        A = np.cumsum(seg, axis=1) - seg                               # exclusive prefix
        logw = A - A.max(axis=0, keepdims=True)
        w = np.exp(logw)
        w = w / w.sum(axis=0, keepdims=True)
        M = seg.max(axis=0, keepdims=True)
        out[a:b] = M[0] + np.log((w * np.exp(seg - M)).sum(axis=0))
    return out


# ── one forward -> per-position log p of the label ───────────────────────────────────

@torch.no_grad()
def label_logp(model, inp, layout, labels, device, *, gram_mode=None, seed=None,
               plan_mode="normal"):
    """``(logp [B, L] float32 of the label token, out)``; 0 where the label is < 0."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode=plan_mode,
                                        gram_mode=gram_mode, gram_sample_seed=seed)
    logits = out["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    return (-ce).cpu(), out


@contextlib.contextmanager
def forced_depth(model, d: int | None):
    """The ``core_depth_sweep.py`` depth lever, restored on exit. ``None`` = unchanged."""
    if d is None:
        yield
        return
    tc = model.cfg.tul
    orig = (int(tc.slot_mean_depth), int(tc.slot_max_depth), int(tc.slot_depth_fixed))
    try:
        tc.slot_mean_depth = int(d)
        tc.slot_max_depth = max(int(d), orig[1] or int(model.cfg.max_depth))
        if orig[2] > 0:
            tc.slot_depth_fixed = int(d)
        yield
    finally:
        tc.slot_mean_depth, tc.slot_max_depth, tc.slot_depth_fixed = orig


def _quant(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else float("nan")


def gram_probe(model, batches, device: str, *, n_list=N_LIST, seed: int = 0,
               depths: list[int] | None = None) -> tuple[dict, dict]:
    """The core of the probe, on already-packed batches ``[(inp, labels, layout, idx)]``
    (``idx`` the stream index per position, ``-1`` off the stream; ``None`` is accepted and
    gives ``tok_index = -1``). Returns ``(result JSON dict, per-token arrays)``.

    Asserts the identity null (``ce_iw_identity@4 == ce_prior@1``, exactly) and that
    ``ce_iw@1 == ce_prior@1`` exactly (the same sample through the read)."""
    if getattr(model, "tul_gram", None) is None:
        raise ValueError("lxtul_g_probe needs a model built with tul.gram=true")
    model.eval()
    n_max = max(n_list)
    tok: dict[str, list[np.ndarray]] = {}
    tok_index: list[np.ndarray] = []
    kl_sum = 0.0
    kl_pass: dict[int, list[np.ndarray]] = {}
    kl_slot: list[np.ndarray] = []
    sig_prior: list[float] = []
    sig_post: list[float] = []

    def _put(key: str, v: np.ndarray) -> None:
        tok.setdefault(key, []).append(v.astype(np.float64))

    def _iw(lps: list[np.ndarray], bags: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Per-row Bayesian read; ``lps`` N arrays ``[n_tok]`` in (row, position) order."""
        stack = np.stack(lps, 0)
        out = np.empty(stack.shape[1], dtype=np.float64)
        for r in np.unique(rows):
            sel = rows == r
            out[sel] = bayes_read(stack[:, sel], bags[sel])
        return out

    for inp, labels, layout, idx in batches:
        layout = layout.to(device)
        tp = ((~layout.slot_mask.cpu()) & (labels >= 0))                 # [B, L]
        rows = torch.arange(inp.shape[0]).view(-1, 1).expand_as(tp)[tp].numpy()
        bags = layout.bag_id.cpu()[tp].numpy()
        tok_index.append((idx[tp].numpy() if idx is not None
                          else np.full(int(tp.sum()), -1)).astype(np.int64))
        # prior samples, seeds seed .. seed + n_max - 1 (sample 0 is ce_prior@1)
        lps = []
        for n in range(n_max):
            lp, out = label_logp(model, inp, layout, labels, device,
                                 gram_mode="prior", seed=seed + n)
            lps.append(lp[tp].numpy().astype(np.float64))
            if n == 0:
                sig_prior.append(float(out["gram_sigma_ratio_prior"]))
        _put("prior1", lps[0])
        for N in n_list:
            _put(f"iw{N}", _iw(lps[:N], bags, rows))
        _put("iw_identity4", _iw([lps[0]] * 4, bags, rows))
        # posterior sample (the instrument that reads the answer) and its KL
        lp, out = label_logp(model, inp, layout, labels, device, gram_mode="post", seed=seed)
        _put("post", lp[tp].numpy())
        kp = out["gram_kl_pass"].float().cpu().numpy()                   # [B, S, T]
        km = out["gram_kl_mask"].cpu().numpy()
        kl_sum += float((kp * km).sum())
        for t in range(kp.shape[-1]):
            kl_pass.setdefault(t + 1, []).append(kp[..., t][km[..., t]])
        has = km.any(-1)
        kl_slot.append((kp * km).sum(-1)[has])
        sig_post.append(float(out["gram_sigma_ratio_post"]))
        # prior mean, cells zeroed
        lp, _ = label_logp(model, inp, layout, labels, device, gram_mode="mean")
        _put("prior_mean", lp[tp].numpy())
        lp, _ = label_logp(model, inp, layout, labels, device, gram_mode="prior", seed=seed,
                           plan_mode="zero")
        _put("zero", lp[tp].numpy())
        # the depth x width corner
        for d in depths or []:
            with forced_depth(model, d):
                lps_d = []
                for n in range(4):
                    lp, _ = label_logp(model, inp, layout, labels, device,
                                       gram_mode="prior", seed=seed + n)
                    lps_d.append(lp[tp].numpy().astype(np.float64))
            _put(f"prior1_d{d}", lps_d[0])
            _put(f"iw4_d{d}", _iw(lps_d, bags, rows))

    arr = {k: -np.concatenate(v) for k, v in tok.items()}              # CE = -log p
    n_tok = int(arr["prior1"].size)
    ce = {k: float(v.mean()) for k, v in arr.items()}
    # THE IDENTITY NULL, and the N = 1 read of sample 0: both EXACT by construction.
    if not np.array_equal(arr["iw_identity4"], arr["prior1"]):
        raise AssertionError(
            f"identity null failed: ce_iw_identity@4 {ce['iw_identity4']!r} != ce_prior@1 "
            f"{ce['prior1']!r} (max |diff| "
            f"{np.abs(arr['iw_identity4'] - arr['prior1']).max()})")
    if 1 in n_list and not np.array_equal(arr["iw1"], arr["prior1"]):
        raise AssertionError("ce_iw@1 != ce_prior@1: the read changed a single sample")
    res = {
        "n_tokens": n_tok, "seed": seed, "n_list": list(n_list),
        "ce_post": ce["post"], "ce_prior@1": ce["prior1"], "ce_prior_mean": ce["prior_mean"],
        **{f"ce_iw@{N}": ce[f"iw{N}"] for N in n_list},
        "ce_iw_identity@4": ce["iw_identity4"], "ce_zero": ce["zero"],
        "ce_elbo": ce["post"] + kl_sum / max(n_tok, 1),
        "kl_per_token": kl_sum / max(n_tok, 1),
        "exposure_gap": ce["prior1"] - ce["post"],
        "width_gain@4": ce["prior1"] - ce.get("iw4", float("nan")),
        "worth_zero": ce["zero"] - ce["prior1"],
        "sigma_ratio_prior": float(np.mean(sig_prior)) if sig_prior else float("nan"),
        "sigma_ratio_post": float(np.mean(sig_post)) if sig_post else float("nan"),
    }
    ks = np.concatenate(kl_slot) if kl_slot else np.zeros(0)
    res["kl_slot"] = {"mean": float(ks.mean()) if ks.size else float("nan"),
                      "p50": _quant(ks, 0.5), "p90": _quant(ks, 0.9), "n_slots": int(ks.size)}
    res["kl_pass"] = {}
    for t, parts in sorted(kl_pass.items()):
        v = np.concatenate(parts) if parts else np.zeros(0)
        res["kl_pass"][str(t)] = {"mean": float(v.mean()) if v.size else float("nan"),
                                  "p50": _quant(v, 0.5), "p90": _quant(v, 0.9),
                                  "n_slots": int(v.size)}
    if depths:
        res["depths"] = {str(d): {"ce_prior@1": ce[f"prior1_d{d}"], "ce_iw@4": ce[f"iw4_d{d}"]}
                         for d in depths}
    arr["_tok_index"] = np.concatenate(tok_index)
    return res, arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint path (relative = repo root)")
    ap.add_argument("--config", required=True, help="Hydra config name the arm trained with")
    ap.add_argument("--ovr", action="append", default=[],
                    help="Hydra override the arm trained with (repeatable)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--rows", type=int, default=192)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="first prior-sample seed")
    ap.add_argument("--depths", default="", help="e.g. 1,6: the depth x width corner")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from _build import ROOT, build_cfg
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    path = a.ckpt if a.ckpt.startswith("/") else os.path.join(ROOT, a.ckpt)
    label = a.label or os.path.basename(os.path.dirname(path))
    cfg = build_cfg(a.config, ["model.use_kernels=false", *a.ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or not tul_rt.model_cfg.gram:
        raise SystemExit(f"{a.config} is not a tul.gram arm")
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]
    depths = [int(x) for x in a.depths.split(",") if x.strip()]
    res, arr = gram_probe(model, batches, a.device, seed=a.seed, depths=depths or None)
    res.update({"label": label, "config": a.config, "ckpt": path, "step": step,
                "rows": sum(b[0].shape[0] for b in batches), "batch": a.batch})
    npz = a.out.rsplit(".", 1)[0] + f".{label}.tokens.npz"
    tix = arr.pop("_tok_index").astype(np.int32)
    np.savez_compressed(npz, tok_index=tix,
                        **{f"ce_{k}": v.astype(np.float32) for k, v in arr.items()})
    res["tokens_npz"] = npz
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("kl_pass",)}, indent=1))
    print(f"wrote {a.out} and {npz}")


if __name__ == "__main__":
    main()
