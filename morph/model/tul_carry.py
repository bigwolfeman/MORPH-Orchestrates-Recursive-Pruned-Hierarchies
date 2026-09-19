"""``tul.loop_carry`` — the slot loop KEEPS what it read from its neighbours.

THE MEASURED PROBLEM (`lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md`).
On `slot-spandec-strict-prev-reach1` at 5000 steps, content ``h`` spans back arrives at
pass ``h - 1`` and then DECAYS under the cell's own later passes: with the reach cut so
that nothing arrives after pass 0, a planted copy two spans back falls from 0.148 to
0.035 nats of benefit between depth 1 and depth 6, and the tokens that depend on that
span lose 0.083 nats. The cell's OWN span does the opposite under the same cut (0.181 ->
0.291), because own-span content is RE-SUPPLIED every pass by the per-layer x0 / bigram
injection while neighbour content enters once, through the layer-0 reach attention, and
then lives only in the recurrent carrier the core re-processes.

WHAT THIS MODULE IS. A per-cell state ``c_k`` that accumulates the cross-cell read and is
re-injected at the entry of every later pass, so neighbour content stands on the footing
own-span content already has.

    c_k(0) = 0
    sum :   c_k(t) = c_k(t-1) + r_k(t)
    gate:   c_k(t) = c_k(t-1) + sigmoid(W_g [r_k(t) ; c_k(t-1)]) * r_k(t)

``r_k(t)`` is the layer-0 window-branch read — see
``MORPHTransformer._tul_core`` for where it is captured and why that tensor IS the
cross-cell content. ``W_g`` is ``d x 2d``, ZERO-init (the gate opens at exactly 1/2), a
plain ``nn.Parameter`` that is never ternarised (``_ternary_exclude``, the
``TULReread`` / ``TULCodeProj`` rule) and drawn from no RNG at all, so building the module
leaves every other weight of the model byte-identical.

THE NORMALISATION, stated once and tested (`tests/test_tul_loop_carry.py`). Before the
add, the carry is scaled so that its per-cell RMS EQUALS the per-cell RMS of the carrier
it is added to:

    rms_h[b,k] = sqrt(mean over (streams, channels) of h[b,k]^2)
    rms_c[b,k] = sqrt(mean over channels of c[b,k]^2)
    term       = c * rms_h / rms_c        (and EXACTLY 0 where rms_c <= eps)

Both in fp32, cast back to the carrier's dtype at the end. The scale is the reason an
unbounded sum cannot grow the carrier's norm: whatever ``c`` accumulates, what is added is
always one carrier-RMS worth of the carry's DIRECTION. A cell whose carry is exactly zero
adds exactly zero and takes no gradient through the match -- see `carry_rms_match` for
the two guards that make that true and not a NaN. Such cells are common, not exotic: a
pad slot for the whole forward, and cell 0 of every row forever (its window row under a
reach budget is empty, so its read is identically 0).

NAMED, not hidden: this arm is therefore NOT a no-op at initialisation, unlike every
zero-init mechanism in this tree. Normalising to the carrier's RMS cancels any constant
scale in front of ``c``, so a zero-init gate cannot make step 0 identical to the ruler —
``gate`` at ``W_g = 0`` injects exactly what ``sum`` injects at pass 1 (half of the same
vector, renormalised to the same RMS). The two modes differ only from pass 2 on, where
``gate`` can weight the new read against what it already holds. The bit-identity claim of
this key is ``loop_carry: "none"``, where nothing is built and nothing runs.

Record: .agents/notes/proposed/architecture/2026-09-19-loop-carry-reinjection.md
Prereg: lab/experiments/planned/2026-09-19-loop-carry-prev-reach1.md
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

LOOP_CARRY_MODES: tuple[str, ...] = ("none", "sum", "gate")


def carry_rms_match(c: Tensor, h: Tensor, eps: float = 1e-6) -> Tensor:
    """``[B, S, C]`` carry scaled to the per-cell RMS of the carrier ``h``.

    ``h`` is ``[B, S, n, C]`` (the Hyper-Connection carrier) or ``[B, S, C]``; the target
    RMS pools every axis after the cell axis, which is the RMS the broadcast term will sit
    at in each of the ``n`` streams. Returns the carrier's dtype.
    """
    cf = c.float()
    rms_h = h.detach().float().flatten(2).pow(2).mean(dim=-1).sqrt()          # [B, S]
    ms_c = cf.pow(2).mean(dim=-1)                                             # [B, S]
    # TWO guards, and both are load-bearing, because cells with an EXACTLY zero carry are
    # not an edge case here: every cell at pass 0, every pad slot for the whole forward,
    # and cell 0 of every row forever (under `loop_reach w` its window row is cells
    # -w..-1, which is empty, so its read is exactly 0).
    #   * `+ eps*eps` INSIDE the sqrt: `sqrt(0)` has an infinite derivative, and an inf
    #     multiplied by the zero cotangent a masked branch hands back is NaN, not 0. This
    #     is the difference between a finite gradient and a NaN loss (measured 2026-09-19
    #     on the tiny fixture: without it the total gradient is nan).
    #   * the `where`: a zero carry is scaled by EXACTLY zero, so it adds nothing and
    #     receives no gradient through the match at all, rather than riding a 1/eps scale.
    rms_c = (ms_c + eps * eps).sqrt()                                         # [B, S]
    scale = torch.where(ms_c > eps * eps, rms_h / rms_c, torch.zeros_like(rms_c))
    return (cf * scale.unsqueeze(-1)).to(h.dtype)


class TULLoopCarry(nn.Module):
    """The per-cell carry state (``tul.loop_carry``). See the module docstring."""

    def __init__(self, d_model: int, mode: str):
        super().__init__()
        if mode not in ("sum", "gate"):
            raise ValueError(
                f"TULLoopCarry(mode={mode!r}): 'none' builds no module at all; the only "
                f"modes that do are 'sum' and 'gate'.")
        self.mode = mode
        self.d_model = d_model
        self.eps = 1e-6
        if mode == "gate":
            # d x 2d, ZERO: sigmoid(0) = 1/2 at every channel of every cell. No RNG is
            # drawn — a zeros_ is deterministic — so a carry model shares every other
            # weight with its same-seed ruler.
            self.W_g: nn.Parameter | None = nn.Parameter(torch.zeros(d_model, 2 * d_model))
        else:
            self.W_g = None
        # The ternary QAT quantises nn.Linear leaves; this module holds a bare Parameter
        # and is excluded explicitly, like TULReread / TULCodeProj.
        self._ternary_exclude = True

    def update(self, r: Tensor, c_prev: Tensor) -> tuple[Tensor, Tensor | None]:
        """``(c_k(t), gate)`` from the read ``r`` ``[B, S, C]`` and ``c_k(t-1)``.

        ``gate`` is the elementwise sigmoid under ``mode='gate'`` and ``None`` under
        ``'sum'`` — the caller reports its mean as ``carry/gate_mean_t{t}``.
        """
        if self.mode == "sum":
            return c_prev + r, None
        g = torch.sigmoid(F.linear(torch.cat([r, c_prev], dim=-1),
                                   self.W_g.to(r.dtype)))
        return c_prev + g * r, g

    def inject_term(self, c: Tensor, h: Tensor) -> Tensor:
        """The ``[B, S, C]`` term added to the carrier ``h`` at the entry of the next pass."""
        return carry_rms_match(c, h, self.eps)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, d_model={self.d_model}"
