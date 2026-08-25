"""H24 screen — revive the CORE's dead HCA compressed branch at fixed weights.

Pre-registration: docs/experiments/planned/2026-08-25-h24-hca-branch-screen.md
(Method Amendment 1, 2026-08-25, is what this file implements.)

Phase 0 of H18 measured that on the TUL slot path (S = 64) the three HCA core blocks have
`n_blocks = S // hca_compress_ratio = 64 // 256 = 0`, so their compressed branch output is
identically 0.0000 while the gate still spends ~0.50 of its mixture on it. The token path
at the same weights has `n_blocks = 4` and a live branch. A1 (slots) diverges, A0 (tokens)
does not.

WHY THIS IS NOT A PLAIN CONFIG OVERRIDE. The pre-registration claimed `m` is in no weight
shape. THAT WAS WRONG, and the run that proved it is in the record: `GatedPoolCompressor`
carries `B_a` of shape `[m, c]`, a learned gate bias per WITHIN-BLOCK position, so
`model.hca_compress_ratio=16` fails to load the checkpoint with seven size mismatches.

Two consequences, both handled here rather than hidden:

  1. The screen needs an APPROXIMATION. `B_a` is sliced to its first `m_new` rows. Those
     rows are trained — every HCA block in prelude and coda uses all 256 of them at
     S = 1152 — but the slice repurposes a 256-wide positional gate as a 16-wide one. The
     screen is therefore a forward-map probe of "what if this branch contributed", not a
     faithful model of the trained alternative.
  2. The change is confined to the CORE. Rewriting prelude and coda too would move the
     carrier that ENTERS the loop, and the pre-registered validity gate V2 requires
     iteration 0 to be identical between the arms. Scoping it to the core is also the
     counterfactual the hypothesis is about.

Usage:
    PYTHONPATH=$PWD python lab/divergence/h24_screen.py \\
        --ckpt-dir checkpoints/morph/onset-capture --m 16 --out h24_revive.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import morph.model.attention as _attn                            # noqa: E402
from lab.divergence.jac_ladder import build, state_geometry      # noqa: E402
from morph.training.train import load_checkpoint                 # noqa: E402

__all__ = ["revive_core_hca"]


def revive_core_hca(model, m_new: int):
    """Set the CORE's HCA compressed ratio to `m_new`, slicing `B_a`. In place.

    Returns `(touched, restore)`. `restore()` puts the original Parameters and ratios back,
    so the next `load_checkpoint` sees the shapes it expects. A caller that touches no
    block gets a raise, not a silent no-op.
    """
    root = getattr(model, "_orig_mod", model)
    touched, saved = [], []
    for li, blk in enumerate(root.core):
        impl = blk.attention._impl
        if not isinstance(impl, _attn._CCAHCAAttention):
            continue
        comp = impl.compressor
        if comp.m <= m_new:
            raise RuntimeError(f"core.{li} already has m={comp.m} <= {m_new}; the screen "
                               "would be widening, not reviving")
        saved.append((impl, comp, comp.B_a, comp.m))
        with torch.no_grad():
            b = comp.B_a.data[:m_new].clone()
        comp.B_a = torch.nn.Parameter(b)
        comp.m = m_new
        impl.compress_ratio = m_new
        touched.append(f"core.{li}")
    if not touched:
        raise RuntimeError("no HCA core blocks found — the layer alternation changed")

    def restore():
        for impl, comp, b, m in saved:
            comp.B_a = b
            comp.m = m
            impl.compress_ratio = m
    return touched, restore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a1")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="run WITHOUT the surgery, through this same file, so the two arms "
                         "differ by the surgery alone and not by which script ran them")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ov = [o for o in a.overrides.split(",") if o.strip()]
    _cfg, model, x, y, layout = build(a.config_name, ov)
    model.train()                       # Poisson depths, as in training

    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    out = []
    for path in ckpts:
        step = int(torch.load(path, map_location="cpu",
                              weights_only=False).get("step", -1))
        # Re-load BEFORE the surgery every rung: the surgery replaces a Parameter, so a
        # later load_checkpoint would hit the same size mismatch that produced this file.
        load_checkpoint(path, model, scaler, torch.device("cuda"))
        touched, restore = ([], lambda: None) if a.control else revive_core_hca(model, a.m)
        st = state_geometry(model, x, y, layout, a.seed)
        restore()
        out.append({"ckpt": os.path.basename(path), "step": step,
                    "m": None if a.control else a.m, "touched": touched, "state": st})
        pi = st["per_iter"]
        eu = " ".join(f"{r['eff_rank_unit']:.2f}" for r in pi)
        ratio = pi[-1]["eff_rank_unit"] / max(pi[0]["eff_rank_unit"], 1e-30)
        print(f"{os.path.basename(path):<22} unit_rank/iter={eu}  ratio={ratio:.3f}",
              flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
