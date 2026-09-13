"""Does the critic's direction actually improve the state? (Arm `slot-spandec-strict-critic`.)

The question
------------
``tul.grad_pass_energy: critic`` hands every pass the gradient of a learned scorer
``c_phi(z, ctx)`` that was fitted to whether one pass lowered the REAL coda's next-span CE
(``morph/model/tul_egrad.py::CriticEnergy``). Two things can go wrong and they look
identical on a K-curve:

* the critic learned nothing and its gradient is noise — the arm is then its ruler with an
  extra injection channel, which is what ``gp_rel_t0`` near 0 meant for
  ``slot-mnext-gradpass``;
* the critic learned something real and the passes cannot USE a direction — which is a
  statement about ``W_g`` and the pass transition, not about the objective.

This probe separates them. It moves the loop's EXIT state a small step along the critic's
direction, replays the real coda, and asks whether the next span's CE fell by more than it
falls along an rms-matched RANDOM direction. It is the decisive reading of prediction P-9
in ``lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md``.

What it measures, per valid slot
--------------------------------
``delta = CE_next_span(z + eps * rms(z) * d) - CE_next_span(z)``, for

* ``d = -grad_z E / rms(grad_z E)``, ``E = -c_phi(z, ctx)`` — the critic's own direction,
  i.e. the way it says the state gets better;
* ``d = n / rms(n)``, ``n ~ N(0, I)``, averaged over ``--n_random`` draws — the control.

Negative is an improvement. Reported: both means, the paired mean gap
``delta_random - delta_critic``, the fraction of scored slots where the critic's direction
is better AT ALL, and the fraction where it is better by more than ``--margin`` (P-9's
0.01). A critic whose direction is noise reads a gap of 0 and a win rate of 0.5 by
construction, so the null is exactly the coin flip and there is nothing to interpret
generously.

Three properties this leans on, none of them new here
-----------------------------------------------------
* **The split point.** ``ZSplit`` (``slot_z_optimize.py``) records the trained forward once
  and then re-runs ONLY what reads ``z``, asserting on every replay that the tensor
  reaching ``TULSlots.prefix_project`` IS the substituted one. So "the coda saw the moved
  state" is a checked claim.
* **The per-slot CE map.** ``slot_outcome_labels`` (``morph/model/tul_egrad.py``) is the
  SAME function the critic is trained against and the same one
  ``slot_state_linear_probe.py`` fits, so the probe cannot drift away from the arm.
* **The context.** ``ctx`` is ``model._tul_egrad_ctx``, the slot's prelude-entry state,
  stashed by ``_tul_core`` on the recording forward — the exact tensor the critic was
  trained with, not a re-derivation.

Eval only, dropout off, ``slot_gain_lambda`` forced to 0: an instrument on a fixed
function.

Usage
  python lab/divergence/critic_direction_probe.py \
      --ckpt LABEL=CONFIG=PATH --rows 96 --batch 3 --depth 6 \
      --out .../critic_direction_LABEL.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from slot_z_optimize import ZSplit, guard_split_point  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

from morph.model.tul_egrad import CriticEnergy, slot_outcome_labels  # noqa: E402

__all__ = ["critic_direction", "random_direction", "step_along", "slot_ce",
           "direction_deltas"]


def _rms(t: torch.Tensor) -> torch.Tensor:
    """Per-slot RMS over everything past the slot axis -> ``[B, S]``."""
    return t.float().flatten(2).pow(2).mean(-1).sqrt()


def critic_direction(model, z: torch.Tensor, ctx: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """The unit-RMS direction the critic says the state improves along, ``z``-shaped.

    ``E = -c_phi(readout(z), ctx)`` is the energy the passes descend, so the improving
    direction is ``-dE/dz``. Taken at a DETACHED leaf with ``create_graph=False``, exactly
    as ``_egrad_feature`` takes it inside the loop, so this is the same vector the arm
    injects — not a re-derivation of it.
    """
    eg = model.tul_egrad
    if not isinstance(eg, CriticEnergy):
        raise SystemExit(
            "critic_direction_probe needs a model built with "
            "tul.grad_pass_energy='critic'; this one has "
            f"{type(eg).__name__ if eg is not None else 'no energy module'}.")
    with torch.enable_grad():
        leaf = z.detach().requires_grad_(True)
        e = eg.energy(model._readout(leaf), ctx, mask)
        g, = torch.autograd.grad(e, leaf, create_graph=False, allow_unused=True)
    if g is None:
        return torch.zeros_like(z)
    d = -g.detach().float()
    return (d / _rms(d).clamp(min=1e-12).view(*d.shape[:2],
                                              *([1] * (d.dim() - 2)))).to(z.dtype)


def random_direction(z: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """An rms-matched Gaussian direction, ``z``-shaped. The control."""
    n = torch.randn(z.shape, generator=generator, device=z.device, dtype=torch.float32)
    return (n / _rms(n).clamp(min=1e-12).view(*n.shape[:2],
                                              *([1] * (n.dim() - 2)))).to(z.dtype)


def step_along(z: torch.Tensor, d: torch.Tensor, eps: float) -> torch.Tensor:
    """``z + eps * rms(z) * d``. ``d`` is unit-RMS, so the step is ``eps`` of the state."""
    s = (eps * _rms(z)).view(*z.shape[:2], *([1] * (z.dim() - 2))).to(z.dtype)
    return z + s * d


@torch.no_grad()
def slot_ce(model, split, inp, labels, layout, z) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mean_ce [B, S], scored [B, S])`` for a substituted ``z``, through the REAL coda."""
    split.replay(inp, labels, layout, z, want_groups=False)
    _y, scored, ce = slot_outcome_labels(
        split.xh, labels, layout, model.embed.lm_weight().detach(),
        model.cfg.ce_chunk_size)
    return ce, scored


def direction_deltas(model, split, inp, labels, layout, eps: float, n_random: int,
                     seed: int) -> dict:
    """One batch -> per-slot arrays. The whole measurement, so a test can call it.

    Returns numpy arrays over the SCORED slots (a slot whose next span holds at least one
    real label under EVERY replay): ``d_critic``, ``d_random`` (the mean over the draws)
    and ``d_random_all`` ``[n_random, n_slots]``.
    """
    _out, z, _h0 = split.record(inp, labels, layout, want_groups=False)
    ctx = getattr(model, "_tul_egrad_ctx", None)
    if ctx is None:
        raise RuntimeError(
            "_tul_core did not stash the critic's context — the recording forward did not "
            "reach the slot loop, so this model is not the one the arm trains.")
    base_ce, base_ok = slot_ce(model, split, inp, labels, layout, z)

    d_c = critic_direction(model, z, ctx, layout.slot_valid)
    ce_c, ok_c = slot_ce(model, split, inp, labels, layout, step_along(z, d_c, eps))

    gen = torch.Generator(device=z.device).manual_seed(seed)
    rs, oks = [], [base_ok, ok_c]
    for _ in range(n_random):
        d_r = random_direction(z, gen)
        ce_r, ok_r = slot_ce(model, split, inp, labels, layout, step_along(z, d_r, eps))
        rs.append(ce_r)
        oks.append(ok_r)
    ok = oks[0]
    for o in oks[1:]:
        ok = ok & o
    sel = ok & layout.slot_valid
    b = base_ce[sel].float().cpu().numpy()
    return {
        "d_critic": ce_c[sel].float().cpu().numpy() - b,
        "d_random_all": np.stack([r[sel].float().cpu().numpy() - b for r in rs]),
        "d_random": np.stack([r[sel].float().cpu().numpy() for r in rs]).mean(0) - b,
        "base_ce": b,
        # The direction is only meaningful if it is not the zero vector. A critic whose
        # gradient vanishes reads a perfect tie against random and would otherwise look
        # like a clean null rather than a broken probe.
        "critic_dir_rms": float(_rms(d_c)[sel].mean()),
    }


def _summary(d_c: np.ndarray, d_r: np.ndarray, margin: float) -> dict:
    gap = d_r - d_c                      # > 0 means the critic's direction is better
    n = int(d_c.size)
    se = float(gap.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n_slots": n,
        "delta_critic_mean": float(d_c.mean()) if n else float("nan"),
        "delta_random_mean": float(d_r.mean()) if n else float("nan"),
        "gap_mean": float(gap.mean()) if n else float("nan"),
        "gap_ci95": [float(gap.mean() - 1.96 * se), float(gap.mean() + 1.96 * se)],
        "win_rate": float((gap > 0).mean()) if n else float("nan"),
        f"win_rate_margin_{margin}": float((gap > margin).mean()) if n else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.1,
                    help="step size as a fraction of rms(z); the arm's tul.critic_eps")
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.01, help="P-9's margin, in nats")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    label, config, path = a.ckpt.split("=", 2)

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("critic_direction_probe needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false)")
    if str(tul_rt.model_cfg.grad_pass_energy) != "critic":
        raise SystemExit("critic_direction_probe needs tul.grad_pass_energy='critic'; "
                         f"this config has {tul_rt.model_cfg.grad_pass_energy!r}")
    model, step = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)
    guard_split_point(model)

    tc = model.cfg.tul
    tc.slot_depth_fixed = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth or model.cfg.max_depth))
    model.cfg.slot_gain_lambda = 0.0
    model.eval()
    model.requires_grad_(False)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    parts: list[dict] = []
    with ZSplit(model) as split:
        for bi, (inp, labels, layout, _) in enumerate(batches):
            inp, labels = inp.cuda(), labels.cuda()
            layout = layout.to("cuda")
            parts.append(direction_deltas(model, split, inp, labels, layout,
                                          a.eps, a.n_random, a.seed + bi))
    d_c = np.concatenate([p["d_critic"] for p in parts])
    d_r = np.concatenate([p["d_random"] for p in parts])
    res = _summary(d_c, d_r, a.margin)
    res.update({
        "label": label, "config": config, "step": int(step), "depth": a.depth,
        "eps": a.eps, "n_random": a.n_random, "margin": a.margin, "rows": a.rows,
        "critic_dir_rms": float(np.mean([p["critic_dir_rms"] for p in parts])),
        "note": ("delta = CE_next_span(z + eps*rms(z)*d) - CE_next_span(z), per valid "
                 "slot, through the REAL coda. Negative is an improvement. gap = "
                 "delta_random - delta_critic, so POSITIVE means the critic's direction "
                 "is the better one. A critic whose gradient is noise reads gap 0 and "
                 "win_rate 0.5 by construction."),
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
