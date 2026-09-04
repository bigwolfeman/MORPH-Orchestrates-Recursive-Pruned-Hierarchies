"""Attention-lift instrument (MUX arXiv 2607.18264 §8.3), eval-only.

THE QUESTION. Under ``tg_restrict`` the slot is the only route from an earlier span to
a later token — but "is the only route" is an architectural fact, not a behavioural one.
A token can be forced to route through slots and still put almost no attention mass on
them. MUX §8.3 measures exactly this and calls it the reasoning attention lift:

    lift = (attention mass on the latent tokens) / (their proportional share)

with share ``K / N_pre``. A lift of 1.0 means attention is spread uniformly over what is
visible; above 1.0 means the model prefers the latents to their share. Their reported
numbers are 0.633 (answer bridge) and 0.544 (final token) — i.e. BELOW 1.0 even for the
method that wins, which is the calibration to read ours against.

WHICH ATTENTION IS COUNTED, precisely. A MORPH attention layer has two branches whose
outputs a learned sigmoid gate blends (``_CCABase._gate_combine_up``):

* the WINDOW branch (``_window_fallback``) — tokens and slots compete inside ONE softmax.
  This is the branch where "lift" has the paper's meaning, and it is the only branch this
  instrument counts.
* the COMPRESSED branch — under ``tg_restrict`` this is ``_tg_slot_attention``, which
  attends slot keys and nothing else, so its slot mass is 1.0 BY CONSTRUCTION. Folding it
  in would produce a number driven by the gate, not by any attention preference, and the
  gate is a different mechanism measured a different way. Unrestricted, the same branch is
  a pooled-block compressor whose keys are not positions at all, so there is no slot mass
  to speak of. Counted in neither case; say so when quoting the number.

So: **window-branch lift**, restricted and unrestricted alike, which is what makes the two
arms comparable on this metric.

HOW IT HOOKS. Eager only, and by swapping ``attention._window_fallback`` for a wrapper —
the same eval-time monkey-patch ``tests/test_tg_restrict.py::_severed_forward`` already
uses, so the hot path carries no instrument code at all. The wrapper recomputes the
attention weights and returns ``weights @ v`` itself, and
``tests/test_tul_gl1b.py::test_the_lift_wrapper_reproduces_the_shipped_attention``
pins that output to the shipped function's. If the shipped mask ever changes, that test
fails rather than the instrument silently measuring a different mask.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

import morph.model.attention as _attn
from morph.model.tul_layout import SlotLayout

__all__ = ["AttnLiftStats", "capture_attn_lift", "first_token_of_span_mask"]


def first_token_of_span_mask(layout: SlotLayout) -> Tensor:
    """``[B, L]`` True at the FIRST token of each span.

    These are the positions where cross-span information matters most: the token has no
    same-span context yet, so anything it knows about earlier spans arrived through a
    slot. MUX's analogue is the answer-bridge / final-prediction token.
    """
    sm = layout.slot_mask
    prev_is_slot = torch.zeros_like(sm)
    prev_is_slot[:, 1:] = sm[:, :-1]
    at_start = torch.zeros_like(sm)
    at_start[:, 0] = True
    return (~sm) & (prev_is_slot | at_start)


@dataclass
class AttnLiftStats:
    """Accumulated per-call window-branch statistics."""

    calls: list[dict] = field(default_factory=list)

    def _agg(self, keys: tuple[str, ...], last_only: bool) -> dict:
        rows = self.calls[-1:] if last_only else self.calls
        rows = [c for c in rows if c["n_q"] > 0]
        if not rows:
            return {k: float("nan") for k in keys}
        n = sum(c["n_q"] for c in rows)
        return {k: sum(c[k] * c["n_q"] for c in rows) / n for k in keys}

    def summary(self) -> dict:
        """The reported metrics. ``attn_lift`` is the LAST layer, per MUX §8.3's
        'last-layer attention'; ``*_alllayers`` averages every window call in the
        forward, which is the broader read and is reported beside it rather than
        instead of it."""
        keys = ("slot_mass", "slot_share", "lift")
        keys_ft = ("slot_mass_ft", "slot_share_ft", "lift_ft")
        last = self._agg(keys, True)
        alll = self._agg(keys, False)
        last_ft = self._agg(keys_ft, True)
        all_ft = self._agg(keys_ft, False)
        return {
            "attn_lift": last["lift"],
            "attn_slot_mass": last["slot_mass"],
            "attn_slot_share": last["slot_share"],
            "attn_lift_first_tok": last_ft["lift_ft"],
            "attn_slot_mass_first_tok": last_ft["slot_mass_ft"],
            "attn_lift_alllayers": alll["lift"],
            "attn_lift_first_tok_alllayers": all_ft["lift_ft"],
            "attn_lift_n_calls": float(len(self.calls)),
        }


def _window_mask(S: int, window_size: int, n_skip_rope: int, device,
                 extra_mask: Tensor | None) -> Tensor:
    """The SHIPPED window/XSA/skip-rope mask. Kept byte-identical to
    ``attention._window_fallback``'s, and pinned to it by a test."""
    row = torch.arange(S, device=device).unsqueeze(1)
    col = torch.arange(S, device=device).unsqueeze(0)
    dist = row - col
    mask = (dist >= 0) & (dist < window_size) & (dist != 0)
    if n_skip_rope > 0:
        mask = mask | (col >= S - n_skip_rope) | (row >= S - n_skip_rope)
    mask = mask.unsqueeze(0).unsqueeze(0)
    if extra_mask is not None:
        mask = mask & extra_mask
    return mask


@contextlib.contextmanager
def capture_attn_lift(layout: SlotLayout, stats: AttnLiftStats):
    """Swap the window branch for a measuring wrapper for the duration of one forward."""
    orig = _attn._window_fallback
    slot_mask = layout.slot_mask                       # [B, L]
    ft_mask = first_token_of_span_mask(layout)         # [B, L]

    def measured(q, k, v, window_size, device, scale, n_skip_rope=0, extra_mask=None):
        S = q.shape[2]
        if S != slot_mask.shape[1]:
            # The core's compact slot-gathered sequence (every position IS a slot) and
            # any other non-row-length call: not a token-vs-slot competition at all.
            return orig(q, k, v, window_size, device, scale, n_skip_rope, extra_mask)
        # THE MODEL'S OUTPUT COMES FROM THE SHIPPED FUNCTION, ALWAYS. An earlier
        # revision returned `weights @ v` computed here; a fully-masked query row (row 0
        # under XSA has no visible key at all) softmaxes to NaN, that NaN entered the
        # residual stream, and every later layer then measured NaN attention which
        # `nan_to_num` silently turned into "zero slot mass". The instrument was
        # reporting a number it had itself created. Measurement never touches the graph
        # now: the weights below are recomputed purely to be counted.
        out = orig(q, k, v, window_size, device, scale, n_skip_rope, extra_mask)

        with torch.no_grad():
            mask = _window_mask(S, window_size, n_skip_rope, device, extra_mask)
            bias = torch.where(mask, 0.0, float("-inf"))
            scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale + bias
            w = torch.softmax(scores, dim=-1)
            allow = mask.expand(q.shape[0], 1, S, S)[:, 0]                # [B, S, S]
            n_keys = allow.sum(-1).float()                                # [B, S]
            n_slot_keys = (allow & slot_mask[:, None, :]).sum(-1).float()
            # A fully-masked row softmaxes to NaN. Excluded, not zero-filled: counting it
            # as "no slot mass" is exactly the error described above.
            live = (n_keys > 0) & torch.isfinite(w).all(-1).all(1)
            share = torch.where(live, n_slot_keys / n_keys.clamp(min=1.0),
                                torch.zeros_like(n_keys))
            mass = (w.mean(1) * slot_mask[:, None, :].float()).sum(-1)     # [B, S]
            mass = torch.where(live, mass, torch.zeros_like(mass))
            tok_q = (~slot_mask) & live & (share > 0)
            ft_q = tok_q & ft_mask
            rec = {"n_q": int(tok_q.sum())}
            for tag, sel in (("", tok_q), ("_ft", ft_q)):
                n = int(sel.sum())
                if n == 0:
                    rec[f"slot_mass{tag}"] = float("nan")
                    rec[f"slot_share{tag}"] = float("nan")
                    rec[f"lift{tag}"] = float("nan")
                    continue
                mm, ss = mass[sel], share[sel]
                rec[f"slot_mass{tag}"] = float(mm.mean())
                rec[f"slot_share{tag}"] = float(ss.mean())
                rec[f"lift{tag}"] = float((mm / ss.clamp(min=1e-9)).mean())
            rec["n_q_ft"] = int(ft_q.sum())
            stats.calls.append(rec)
        return out

    _attn._window_fallback = measured
    try:
        yield stats
    finally:
        _attn._window_fallback = orig
