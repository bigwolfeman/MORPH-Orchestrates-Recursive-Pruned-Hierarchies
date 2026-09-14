"""LoopMTP's own instruments (arXiv 2608.03624, Fig 2 bottom / Fig 3 / Fig 4 right).

The forced-depth K-curve (`core_depth_sweep.py`) asks what the iterations are WORTH in
nats. It cannot say whether the iterations are DOING DIFFERENT THINGS, which is the claim
LoopMTP makes: each iteration is trained against a different horizon, so iteration `t`
should be the one that knows about the token `t` steps ahead, and consecutive iterations
should stop being near-copies of one another. That is what this reads.

Three numbers per checkpoint, all at eval, all on the same packed rows:

* **Fig 3 — horizon rank.** For every iteration `t` and every offset `k`, the MEDIAN rank
  of the true token `u_{i+k}` when iteration `t`'s state is read out. Two read-outs, both
  reported, because MORPH is not the paper's architecture:
    - `rank_head`: the model's OWN LM head applied to the iterate, skipping the coda
      (`embed.attend(_readout(x))`). This is the paper's "read the LM head off each
      iteration's raw hidden state".
    - `rank_cos`: the rank under the cosine score the alignment loss actually optimises
      (`cos(proj(x), E_v)` over the whole vocabulary, `proj` = the arm's shared
      `loopmtp_proj` where it has one). On an arm with `loopmtp_weight > 0` this is the
      objective's own read-out; on a control it is the same formula with no projection.
  The paper's headline is the DIAGONAL (`t == k`) against iteration 1's row: "iteration
  t's rank of u_{i+t} beats iteration 1's rank of u_{i+t}". The full matrix is kept so a
  reader can see whether the horizons separate or every iteration just predicts u_{i+1}.
* **Fig 2 bottom — consecutive cosine.** `cos(x^(t), x^(t+1))` and `cos(x^(1), x^(t))`,
  per position, meaned. The ruler's band is the number a control arm reads; the claim is
  that horizon supervision pushes later iterations into different subspaces.
* **Fig 4 right — gate mass.** Mean and standard deviation of the normalised aggregator
  gate each iteration received, per iteration. Only on a `core_readout: gated` arm.

Runs on a control (`core_readout: last`, `loopmtp_weight: 0`) too: the iterates are
collected by the same `_loopmtp_capture` hook, which only needs `_loopmtp_states`, so a
control must be built with `model.loopmtp_weight=1e-12` (or `core_readout=gated`) to be
readable — and that changes the model. The honest route for a control is `--collect-only`,
which forces the collection on WITHOUT touching the read-out: it flips
`model._loopmtp_states` after construction, which cannot change any weight or any forward
op except that the per-iteration carriers are kept. See `--collect-only`'s help.

Usage:
  python lab/divergence/loopmtp_iteration_probe.py \
    --ckpt d6mtp=notul_norm_match_20k_d6_loopmtp=checkpoints/morph/norm-match-20k-d6-loopmtp/step_20000.pt \
    --ckpt d6=notul_norm_match_20k_d6fixed=checkpoints/morph/norm-match-20k-d6fixed/step_20000.pt \
    --rows 48 --batch 3 --out .../loopmtp_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def _iterates(model, inp, device: str):
    """The per-iteration carriers ``x^(1) .. x^(T)``, in batch order, plus the gate mass."""
    model._loopmtp_capture = True
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            model(inp.to(device), labels=None)
        if not getattr(model, "_loopmtp_iterates", None):
            raise RuntimeError(
                "no per-iteration carriers were collected. The model's `_loopmtp_states` "
                "is off (core_readout 'last' AND loopmtp_weight 0) — run with "
                "--collect-only to turn the collection on without changing the read-out.")
        return list(model._loopmtp_iterates), model._loopmtp_gate_mass
    finally:
        model._loopmtp_capture = False
        model._loopmtp_iterates = None
        model._loopmtp_gate_mass = None


@torch.no_grad()
def _ranks(scores_fn, targets: torch.Tensor, valid: torch.Tensor, n_pos: int,
           chunk: int = 256) -> np.ndarray:
    """Median 1-based rank of `targets` under `scores_fn(lo, hi)` -> [n, V].

    Chunked over positions: the full ``[B*S, V]`` score matrix is 0.6 GB in fp32 at
    seq 1024 / batch 3 / vocab 49152, and there is one per (iteration, offset) pair.
    """
    out: list[torch.Tensor] = []
    for lo in range(0, n_pos, chunk):
        hi = min(lo + chunk, n_pos)
        v = valid[lo:hi]
        if not bool(v.any()):
            continue
        sc = scores_fn(lo, hi).float()                       # [n, V]
        tg = targets[lo:hi].clamp(min=0).unsqueeze(1)        # [n, 1]
        own = sc.gather(1, tg)                               # [n, 1]
        rank = (sc > own).sum(dim=1) + 1                     # 1-based
        out.append(rank[v].cpu())
    if not out:
        return np.array([])
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=256,
                    help="positions per vocabulary-score chunk (memory knob)")
    ap.add_argument("--collect-only", action="store_true",
                    help="turn the per-iteration collection on for an arm that does not "
                         "need it (a control). It sets `model._loopmtp_states = True` "
                         "AFTER construction: no parameter is added, no read-out changes, "
                         "and `_core_region` only keeps the carriers it already computed. "
                         "It cannot be used to score a Poisson-depth model — the "
                         "collection asserts a full active set at every iteration, which "
                         "is exactly what a Poisson draw breaks.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        parts = triple.split("=", 3)
        label, config, path = parts[0], parts[1], parts[2]
        ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is not None:
            raise SystemExit(
                f"{label}: LoopMTP is a PLAIN-model arm; this probe has no defined "
                f"read-out on a TUL layout (the slot positions carry no u_{{i+t}} label).")
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, None)
        model.eval()
        if a.collect_only:
            model._loopmtp_states = True
        T = int(model.cfg.mean_depth)
        if int(model.cfg.mean_depth) != int(model.cfg.max_depth):
            raise SystemExit(
                f"{label}: a Poisson-depth model (mean {model.cfg.mean_depth}, max "
                f"{model.cfg.max_depth}) has no per-iteration read-out — different rows "
                f"run different numbers of iterations. Score a fixed-depth arm.")

        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = int(cfg.data.seq_len) + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        n_batches = -(-a.rows // a.batch)
        batches = pack_rows(stream, None, cfg, a.batch, True)[:n_batches]
        rows_done = sum(inp.shape[0] for inp, _, _, _ in batches)

        w = model.embed.lm_weight().detach().float()                 # [V, d] — sg[E]
        w_n = F.normalize(w, dim=-1)
        proj = getattr(model, "loopmtp_proj", None)

        # rank[t][k] -> list of per-position ranks; cos_next[t], cos_first[t]
        rank_head = {t: {k: [] for k in range(1, T + 1)} for t in range(1, T + 1)}
        rank_cos = {t: {k: [] for k in range(1, T + 1)} for t in range(1, T + 1)}
        cos_next = {t: [] for t in range(1, T)}
        cos_first = {t: [] for t in range(2, T + 1)}
        gate_mass: list[np.ndarray] = []

        for inp, labels, _layout, _idx in batches:
            states, mass = _iterates(model, inp, device)
            if len(states) != T:
                raise RuntimeError(f"{label}: collected {len(states)} iterates, expected {T}")
            if mass is not None:
                gate_mass.append(mass.float().cpu().numpy())         # [T, B*S]
            lab = labels.to(device)
            B, S = lab.shape
            flat = [s.mean(dim=2) if s.dim() == 4 else s for s in states]   # [B, S, d]

            # Fig 2 bottom — the pass geometry, before anything is projected.
            for t in range(1, T):
                c = F.cosine_similarity(flat[t - 1].float(), flat[t].float(), dim=-1)
                cos_next[t].append(float(c.mean()))
            for t in range(2, T + 1):
                c = F.cosine_similarity(flat[0].float(), flat[t - 1].float(), dim=-1)
                cos_first[t].append(float(c.mean()))

            # Fig 3 — the horizon ranks.
            for t in range(1, T + 1):
                h = flat[t - 1]
                head = model.embed.attend(model._readout(states[t - 1])).float()
                head = head.reshape(B * S, -1)
                hp = proj(h) if proj is not None else h
                hp_n = F.normalize(hp.float().reshape(B * S, -1), dim=-1)
                for k in range(1, T + 1):
                    # u_{i+k} == labels[i + k - 1]; the last k-1 positions have no target.
                    tk = (lab if k == 1
                          else F.pad(lab[:, k - 1:], (0, k - 1), value=-100))
                    tk = tk.reshape(B * S)
                    v = (tk >= 0)
                    rank_head[t][k].append(
                        _ranks(lambda lo, hi: head[lo:hi], tk, v, B * S, a.chunk))
                    rank_cos[t][k].append(
                        _ranks(lambda lo, hi: hp_n[lo:hi] @ w_n.t(), tk, v, B * S, a.chunk))

        def _med(d):
            return {str(t): {str(k): (float(np.median(np.concatenate(v))) if v and
                                      len(np.concatenate(v)) else None)
                             for k, v in row.items()} for t, row in d.items()}

        arm = {
            "step": step, "rows": rows_done, "batch": a.batch, "T": T,
            "config": config, "collect_only": bool(a.collect_only),
            "core_readout": str(model.cfg.core_readout),
            "loopmtp_weight": float(model.cfg.loopmtp_weight),
            "loopmtp_free_first": bool(model.cfg.loopmtp_free_first),
            "loopmtp_proj": str(model.cfg.loopmtp_proj),
            "loopmtp_ponder_weight": float(model.cfg.loopmtp_ponder_weight),
            # Fig 3
            "median_rank_head": _med(rank_head),
            "median_rank_cos": _med(rank_cos),
            # Fig 2 bottom
            "cos_consecutive": {str(t): float(np.mean(v)) for t, v in cos_next.items() if v},
            "cos_to_first": {str(t): float(np.mean(v)) for t, v in cos_first.items() if v},
        }
        if gate_mass:
            g = np.concatenate(gate_mass, axis=1)                    # [T, N]
            arm["gate_mean"] = [float(x) for x in g.mean(axis=1)]
            arm["gate_std"] = [float(x) for x in g.std(axis=1)]
        results[label] = arm

        print(f"\n=== {label} @ step {step}  (T={T}, readout={arm['core_readout']}, "
              f"lambda_align={arm['loopmtp_weight']})")
        print("  Fig 3 — median rank of u_{i+k} read from iteration t (head read-out)")
        print("    t \\ k " + " ".join(f"{k:>9d}" for k in range(1, T + 1)))
        for t in range(1, T + 1):
            row = arm["median_rank_head"][str(t)]
            print(f"    {t:>5d} " + " ".join(
                f"{row[str(k)]:>9.1f}" if row[str(k)] is not None else "        -"
                for k in range(1, T + 1)))
        print("  Fig 3 — same, under the alignment loss's own cosine score")
        for t in range(1, T + 1):
            row = arm["median_rank_cos"][str(t)]
            print(f"    {t:>5d} " + " ".join(
                f"{row[str(k)]:>9.1f}" if row[str(k)] is not None else "        -"
                for k in range(1, T + 1)))
        diag = [arm["median_rank_head"][str(t)][str(t)] for t in range(1, T + 1)]
        first = [arm["median_rank_head"]["1"][str(t)] for t in range(1, T + 1)]
        print(f"  diagonal (iteration t at its own horizon): {diag}")
        print(f"  iteration 1 at the same horizons:          {first}")
        print(f"  cos(x^t, x^t+1): {arm['cos_consecutive']}")
        print(f"  cos(x^1, x^t):   {arm['cos_to_first']}")
        if "gate_mean" in arm:
            print(f"  gate mean/std per iteration: "
                  f"{[f'{m:.3f}+-{s:.3f}' for m, s in zip(arm['gate_mean'], arm['gate_std'])]}")

    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
