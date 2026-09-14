"""The horizon-arm per-pass instrument (arm `slot-spandec-strict-horizon`).

Loads a checkpoint of a `tul.pass_readout in ("last", "gated")` slot-loop model built
with `tul.slot_depth_fixed = T`, packs rows from the validation stream (the
`core_depth_sweep.py` cut: the trainer's packer, one paired batch set) and reports, per
valid slot:

  1. THE T x T MATRIX -- does pass t's state predict horizon h's span better than pass
     1's does? `M[t, h] = mean_slot cos( proj(readout(db_traj[t])), target_h )`, where
     `target_h` is the SAME detached, mean-pooled tied-embedding target
     `_tul_horizon_loss` grades against (span slot+h), and `proj` is the model's own
     `tul_horizon_proj` when the checkpoint was trained with `horizon_weight > 0`, or
     the identity otherwise (`--no-proj` forces this, e.g. to probe a `-fixed6` control
     that never built the projection). Diagonal `M[t, t]` is what the shipped loss (at
     `t >= 2`) actually trains; the off-diagonal answers the question this arm exists to
     ask -- reading UP a row (fixed h, varying t) says whether later passes carry more
     of horizon h than pass 1 does.
  2. CONSECUTIVE-PASS COSINE OF THE UPDATES: `cos(h_t - h_{t-1}, h_{t+1} - h_t)`, mean
     over valid slots, one number per consecutive pair -- the "do the passes cancel"
     reading (`web-text-does-loop-target-is-the-lever` / the per-pass gradient probe
     lineage: -0.2 to -0.6 on every arm run so far).
  3. THE GATE'S MEAN WEIGHT PER PASS (only when the checkpoint has `tul_pass_gate`,
     i.e. `pass_readout="gated"`): `mean_slot g~_t` for t=1..T, the SAME normalised
     weight `TULPassGate.forward` computes, so this instrument reads exactly what the
     coda's write actually used, not an offline re-derivation of it.

Usage:
  python lab/divergence/horizon_pass_probe.py \
    --ckpt horizon=tul_slot_spandec_strict_horizon=checkpoints/morph/.../step_5000.pt \
    --rows 48 --out .../horizon_pass_probe.json

NOT RUN AGAINST A REAL CHECKPOINT as part of writing this file -- no GPU, no checkpoint,
in the CPU-only worktree this was authored in (see the prereg's "Not verified before
launch"). The T x T / consecutive-cosine / gate-weight MATH is exercised directly, on
the tiny CPU test fixture, by `tests/test_tul_horizon.py`
(`test_the_gate_normalises_to_one`, `test_pass_t_reads_span_i_plus_t_via_span_slots_shift_t`)
-- this script's own arithmetic mirrors those functions, it does not reimplement them.
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
def _horizon_matrix(model, inp, layout, input_ids, T: int, J: int, use_proj: bool
                    ) -> tuple[np.ndarray, np.ndarray, list]:
    """One batch's contribution: ``(cos_sum [T, T], cos_n [T, T], db_traj)``.

    ``cos_sum[t-1, h-1] / cos_n[t-1, h-1]`` is the batch's mean cosine of pass t's
    (projected) state against horizon h's target, over the slots for which span
    (slot + h) exists. ``db_traj`` is returned so the caller can also read the
    consecutive-pass cosine and the gate weights off the SAME forward -- one `_tul_core`
    call per batch, not three.
    """
    from morph.model.tul_spandec import span_slots

    real = model._tul_core
    captured: dict = {}

    def spy(*a, **kw):
        out = real(*a, **kw)
        captured["db_traj"] = out[4]
        return out

    model._tul_core = spy
    try:
        model(inp, labels=None, slot_layout=layout)
    finally:
        model._tul_core = real
    db_traj = captured["db_traj"]
    if db_traj is None:
        raise RuntimeError(
            "no db_traj returned -- the checkpoint's tul.slot_depth_fixed is probably 0 "
            "(Poisson depth), which this instrument is not defined against (the same "
            "rule TULConfig.__post_init__ enforces for tul.horizon_weight/pass_readout).")
    if len(db_traj) - 1 != T:
        raise RuntimeError(f"expected T={T} passes, db_traj has {len(db_traj) - 1}")

    w_tied = model.embed.lm_weight().detach()
    proj = model.tul_horizon_proj if (use_proj and model.tul_horizon_proj is not None) \
        else (lambda x: x)

    cos_sum = np.zeros((T, T), dtype=np.float64)
    cos_n = np.zeros((T, T), dtype=np.float64)
    S = layout.slot_index.shape[1]
    for h in range(1, T + 1):
        if h >= S:
            continue
        ids, valid = span_slots(input_ids, layout, J, shift=h)
        keep = valid.any(dim=-1) & layout.slot_valid
        if not bool(keep.any()):
            continue
        e = F.embedding(ids, w_tied)
        m = valid.unsqueeze(-1).to(e.dtype)
        tgt = (e * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        for t in range(1, T + 1):
            z = proj(model._readout(db_traj[t]).float())
            cos = F.cosine_similarity(z, tgt.float(), dim=-1)
            cos_sum[t - 1, h - 1] += float(cos[keep].sum())
            cos_n[t - 1, h - 1] += float(keep.sum())
    return cos_sum, cos_n, db_traj


@torch.no_grad()
def _consecutive_cosine(db_traj: list, slot_valid: torch.Tensor) -> list[float]:
    """``cos(h_t - h_{t-1}, h_{t+1} - h_t)`` per consecutive pair, mean over valid slots."""
    deltas = [(db_traj[t] - db_traj[t - 1]) for t in range(1, len(db_traj))]
    out = []
    for i in range(len(deltas) - 1):
        # HC stream mean (dim 2), NOT a flatten-and-mean over stream*channel -- that
        # would collapse the channel axis too and leave nothing for cosine to compare.
        a = deltas[i].mean(dim=2) if deltas[i].dim() == 4 else deltas[i]
        b = deltas[i + 1].mean(dim=2) if deltas[i + 1].dim() == 4 else deltas[i + 1]
        cos = F.cosine_similarity(a.float(), b.float(), dim=-1)
        out.append(float(cos[slot_valid].mean()))
    return out


@torch.no_grad()
def _gate_weights(model, db_traj: list, slot_valid: torch.Tensor) -> list[float] | None:
    """``mean_slot g~_t`` for t=1..T, the SAME computation `TULPassGate.forward` makes."""
    if model.tul_pass_gate is None:
        return None
    g = model.tul_pass_gate
    states = db_traj[1:]
    gates = [F.softplus(g.Wg(x) + g.beta[t]) for t, x in enumerate(states)]
    denom = gates[0]
    for gt in gates[1:]:
        denom = denom + gt
    denom = denom + g.eps
    out = []
    for gt in gates:
        frac = (gt / denom)
        frac = frac.mean(dim=2) if frac.dim() == 4 else frac   # HC stream mean, not channel
        out.append(float(frac[slot_valid].mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--tokens-per-span", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-proj", action="store_true",
                     help="ignore tul_horizon_proj even if the checkpoint has one "
                          "(read the RAW readout instead) -- for a fixed6 control that "
                          "was never trained with the projection.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        cfg = build_cfg(config, ["model.use_kernels=false"])
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is None:
            raise ValueError(f"{label}: not a TUL config, this instrument needs a slot loop")
        T = int(tul_rt.model_cfg.slot_depth_fixed)
        if T <= 0:
            raise ValueError(
                f"{label}: tul.slot_depth_fixed={T} (Poisson depth); this instrument "
                f"needs a fixed-depth model, the same rule the model's own knobs enforce.")
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, tul_rt.model_cfg)
        model.eval()

        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        n_batches = -(-a.rows // a.batch)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, plain=False)[:n_batches]

        cos_sum = np.zeros((T, T), dtype=np.float64)
        cos_n = np.zeros((T, T), dtype=np.float64)
        consec: list[list[float]] = []
        gate_w: list[list[float]] = []
        n_rows_done = 0
        for inp, labels, layout, idx in batches:
            inp = inp.to(device)
            layout = layout.to(device)   # the packer returns CPU tensors (2026-09-14 Spark run)
            cs, cn, db_traj = _horizon_matrix(model, inp, layout, inp, T,
                                              a.tokens_per_span, use_proj=not a.no_proj)
            cos_sum += cs
            cos_n += cn
            consec.append(_consecutive_cosine(db_traj, layout.slot_valid))
            gw = _gate_weights(model, db_traj, layout.slot_valid)
            if gw is not None:
                gate_w.append(gw)
            n_rows_done += inp.shape[0]

        matrix = np.divide(cos_sum, cos_n, out=np.full_like(cos_sum, np.nan), where=cos_n > 0)
        results[label] = {
            "step": step, "T": T, "rows": n_rows_done,
            "cosine_matrix_pass_by_horizon": matrix.tolist(),
            "cosine_matrix_n": cos_n.tolist(),
            "consecutive_pass_cosine_mean": np.mean(consec, axis=0).tolist() if consec else [],
            "gate_mean_weight_per_pass": (np.mean(gate_w, axis=0).tolist() if gate_w else None),
        }
        print(f"{label}: T={T} rows={n_rows_done} diag={np.diag(matrix).round(4).tolist()}")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
