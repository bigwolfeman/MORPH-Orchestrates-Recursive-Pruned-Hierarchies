"""The amplitude-PIN K-curve of an LXTUL-E arm: coda K1-K6 and K3-K6 with every slot's
non-injected channels held at their entry RMS after every loop pass, against the unpinned
curve on the same rows.

Sibling of ``lx_amp_matched.py`` (which gives a shallow exit a LOUDER code). This probe
removes the other half of the confound from below: outside ``DiagonalInjection``'s ctx
slice ``[start, end)`` (512:832 on fp01, read from the built model, never hard-coded) the
carry is the identity, so the LX code ``r * rms(f(h)) * u_k`` re-added at every pass piles
up there roughly linearly with depth. The PIN rescales those channels back to the RMS they
entered the loop with, after every pass, so every depth hands the coda a carrier whose
non-injected part has the SAME size. What is left of K1-K6 under the pin is the passes'
work on direction, not on amplitude.

TERMS (one meaning each; the Stage 1 terms carry over, ``lxtul_e_stage1_score.py``):

  depth d        forced slot-loop depth of every valid slot (``slot_depths`` table).
  outer channels the channels outside ``model.injection.start:end``.
  entry RMS      per (rollout row, slot, Hyper-Connection stream): the RMS over the outer
                 channels of the carrier the loop STARTS from (``h`` at pass 0, after
                 ``core_init``), recorded at pass 0 of every ``_tul_core`` call.
  pin            after pass t's output is complete (the core step, then the LX code term),
                 every VALID slot's outer channels are multiplied by
                 ``entry RMS / (their RMS now + 1e-6)``, per stream; the ctx slice and pad
                 slots are untouched. Identical at every pass. The same rule at every depth.
  coda CE        per-token coda CE under the per-span Bayes read over the K rollouts, on the
                 480 validation rows, token-paired by stream index.
  K{a}-K{b}      CE(a) - CE(b); ``pinned`` and ``free`` (no pin) are separate curves on the
                 SAME tokens; ``pin effect @d`` = CE_pinned(d) - CE_free(d).

The pin is installed through ``MORPHTransformer._slot_pass_hook`` (None by default; an eval
seam in ``_tul_core`` right after the code term) and REMOVED on exit, also on error. With
``--pin off`` nothing is installed. Eval only: the hook RAISES under ``model.training``.

STATISTICS. Token-weighted means; paired block bootstrap over 1,024-token stream blocks,
2,000 resamples, seed 0 (``lxtul_e_stage1_score._ci``). Self-check: the Stage 1 scorer's
per-batch check against the forward's own ``ce_tokens``.

Usage (GPU):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/lx_amp_pin.py \\
      --arm fp01_5k=tul_slot_spandec_strict_e4probe_fp01=/home/wolfe/morph-to/checkpoints/morph/lxtul-e4probe-fp01/step_5000.pt \\
      --rows 480 --out .../pin_fp01_5k.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _build import parse_ckpt_spec  # noqa: E402
from lxtul_e_stage1_score import (  # noqa: E402  (ONE copy of the Stage 1 instrument)
    N_BOOT, _blocks, _ci, _fmt, abs_path, load_arm, score_arm, val_batches)


def outer_mask(m) -> torch.Tensor:
    """``[C]`` bool: True on the channels outside the DiagonalInjection ctx slice."""
    inj = m.injection
    if type(inj).__name__ != "DiagonalInjection":
        raise RuntimeError(f"model.injection is {type(inj).__name__}, not DiagonalInjection")
    C = int(m.cfg.d_model)
    s, e = int(inj.start), int(inj.end)
    if not 0 <= s < e <= C or e - s >= C:
        raise RuntimeError(f"injection slice {s}:{e} leaves no outer channel (C = {C})")
    mask = torch.ones(C, dtype=torch.bool)
    mask[s:e] = False
    return mask


class AmpPin:
    """The pin as a ``_slot_pass_hook``: ``(t, h, h_new, layout) -> h_new``.

    At ``t == 0`` it records the entry RMS of ``h`` (the loop's entry carrier). Every call
    returns ``h_new`` with each valid slot's outer channels rescaled to that entry RMS, per
    stream. ``stats`` accumulates, per pass t, the mean over valid (row, slot, stream) of the
    outer RMS BEFORE the pin divided by the entry RMS (the growth the pin removes)."""

    def __init__(self, model, eps: float = 1e-6):
        self.model = model
        self.mask = outer_mask(model)
        self.eps = eps
        self.entry: torch.Tensor | None = None
        self.stats: dict[int, list[float]] = {}

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask.to(x.device)
        return x.float()[..., m].pow(2).mean(-1).sqrt()               # [B, S, (n)]

    def __call__(self, t: int, h: torch.Tensor, h_new: torch.Tensor, layout) -> torch.Tensor:
        if self.model.training:
            raise RuntimeError("the amplitude pin is an EVAL instrument; model is training")
        if t == 0:
            self.entry = self._rms(h).detach()
        if self.entry is None or self.entry.shape != self._rms(h_new).shape:
            raise RuntimeError("pin called before pass 0 of this _tul_core call")
        now = self._rms(h_new)
        valid = layout.slot_valid
        vm = valid.view(*valid.shape, *([1] * (now.dim() - 2))).expand_as(now)
        with torch.no_grad():
            ratio = (now / (self.entry + 1e-12))[vm]
            self.stats.setdefault(int(t), []).append(float(ratio.mean()) if ratio.numel() else float("nan"))
        scale = torch.where(vm, self.entry / (now + self.eps), torch.ones_like(now))
        m = self.mask.to(h_new.device)
        full = torch.where(m, scale.unsqueeze(-1), torch.ones_like(scale).unsqueeze(-1))
        return (h_new.float() * full).to(h_new.dtype)


@contextlib.contextmanager
def amp_pin(m, on: bool = True):
    """Install :class:`AmpPin` on ``m`` for the block (``on=False``: install nothing) and
    remove it after, also when the block raises. Yields the pin (or None)."""
    if getattr(m, "_slot_pass_hook", None) is not None:
        raise RuntimeError("a slot-pass hook is already installed")
    if not on:
        yield None
        return
    pin = AmpPin(m)
    m._slot_pass_hook = pin
    try:
        yield pin
    finally:
        m._slot_pass_hook = None


def k_pairs(ce: dict[int, np.ndarray], blk: np.ndarray, nb: int, sd: int) -> dict:
    """K1-K6 and K3-K6 (and K1-Kmax) of one curve, when the depths exist."""
    out = {}
    top = max(ce)
    for a, b in ((1, 6), (3, 6), (1, top)):
        if a in ce and b in ce and a != b and f"K{a}-K{b}" not in out:
            out[f"K{a}-K{b}"] = _ci(ce[a], ce[b], blk, None, nb, sd)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, help="LABEL=CONFIG=PATH[=ovr,...]")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depths", default="1,2,3,4,5,6")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ovr", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    depths = [int(x) for x in a.depths.split(",")]
    label, config, path, ovr = parse_ckpt_spec(a.arm)
    m, step, cfg, rt = load_arm(config, path, a.device, [*a.ovr, *ovr])
    if not getattr(m, "_code_enum_k", 0):
        raise SystemExit(f"{label}: not an LXTUL-E arm (tul.code_enum_k > 1)")
    mask = outer_mask(m)
    batches, rows = val_batches(cfg, rt, a.rows, a.batch)
    res: dict = {"label": label, "config": config, "path": abs_path(path), "step": step,
                 "rows": rows, "depths": depths,
                 "ctx_slice": [int(m.injection.start), int(m.injection.end)],
                 "n_outer_channels": int(mask.sum())}
    print(f"[{label}] step {step}: ctx slice {res['ctx_slice']}, {res['n_outer_channels']} "
          f"outer channels pinned", flush=True)
    ce: dict[str, dict[int, np.ndarray]] = {"free": {}, "pinned": {}}
    idx0 = None
    growth: dict[int, dict] = {}
    for mode in ("free", "pinned"):
        for d in depths:
            with amp_pin(m, on=mode == "pinned") as pin:
                p = score_arm(m, batches, [d], a.device, a.tol)["per"][d]
            if idx0 is None:
                idx0 = p["coda_idx"]
            if not np.array_equal(p["coda_idx"], idx0):
                raise RuntimeError(f"{mode} d{d}: the coda token set moved")
            ce[mode][d] = p["coda_mix"].astype(np.float64)
            if pin is not None:
                growth[d] = {int(t): float(np.nanmean(v)) for t, v in sorted(pin.stats.items())}
            print(f"  {mode:6s} d{d}: coda {ce[mode][d].mean():.4f}", flush=True)
    blk = _blocks(idx0)
    nb, sd = a.n_boot, a.seed
    res["coda_ce"] = {mode: {d: float(v.mean()) for d, v in c.items()} for mode, c in ce.items()}
    res["free"] = k_pairs(ce["free"], blk, nb, sd)
    res["pinned"] = k_pairs(ce["pinned"], blk, nb, sd)
    res["pin_effect"] = {d: _ci(ce["pinned"][d], ce["free"][d], blk, None, nb, sd)
                         for d in depths}
    res["outer_rms_over_entry_before_pin"] = growth
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    npz = a.out.rsplit(".", 1)[0] + ".tokens.npz"
    np.savez_compressed(npz, coda_idx=idx0.astype(np.int64),
                        **{f"{mode}_d{d}": v.astype(np.float32)
                           for mode, c in ce.items() for d, v in c.items()})
    res["tokens_npz"] = npz
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"[{label}] amplitude-pin K-curve, {rows} rows, {idx0.size} coda tokens")
    for mode in ("free", "pinned"):
        for k, v in res[mode].items():
            print(f"  {mode:6s} {k:7s} {_fmt(v)}")
    for d in depths:
        print(f"  pin effect @{d}  {_fmt(res['pin_effect'][d])}")
    print(f"  outer RMS / entry before the pin, per pass, at the deepest depth: "
          f"{growth.get(max(depths))}")
    print(f"wrote {a.out} and {npz}")


if __name__ == "__main__":
    main()
