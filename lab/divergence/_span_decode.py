"""The ONE home of the strict-geometry span decode the three LCM instruments share.

`code_roundtrip_probe.py`, `code_context_mi_probe.py` and `fan_stream_decode_probe.py`
all need the same three things on a packed validation row:

1. PIN the cells the coda reads, so a decode that rewrites the row's tokens cannot move
   them (:class:`PinnedCells`). The shipped forward recomputes the cells from the prelude
   every call; span ``s+1``'s tokens feed slot ``s+1``'s seed, so without the pin the
   cells of every later slot drift while the decode runs and the probe would score a
   rolled generation instead of the row's own cells.
2. DECODE every eligible span at once (:func:`greedy_decode`). Affordable only because
   the strict geometry (`tul.tg_geometry: strict`) makes a row's spans mutually invisible
   at the TOKEN level — a coda token reads its own span's tokens plus the earlier slots'
   PREFIX CELLS, and the cells are pinned — so one forward advances every span by one
   token. `--check-parity` falsifies that instead of assuming it: the same decode with
   only the even-indexed slots live must reproduce the even slots' tokens exactly.
3. SCORE a span teacher-forced (:func:`span_nll`), which is ONE forward and J gathers.

The positions are the ones ``MORPHTransformer._tul_code_grade`` uses and the layout check
below is that method's, copied deliberately: token ``j`` of span ``s+1`` is read at the
position that PREDICTS it — the boundary token of span ``s`` (``slot_index - 1``) for
``j = 0``, the span's own token ``j-1`` (``slot_index + prefix_k + j - 1``) after that.
The logits come from the SHIPPED forward's ``out["logits"]``, indexed at those positions,
with the structural ``slot_id`` masked to −inf exactly as the CE helpers and
``_code_grade_logits`` mask it. The forward is called whole rather than through
``_coda_state_only`` because that entry point does not exist at every commit these probes
must load a checkpoint from (`1eefe85`, the LCTUL 20k arm's commit, predates it), and one
read path that works on both is worth the transient ``[B, L, V]`` tensor (340 MB in bf16
at B 3, L 1152, V 49169).

THE DECODE IS LENGTH-MATCHED, not free-running: span ``s+1`` is decoded to exactly the
true span's token count and written back over those positions, so the row's length, its
slot positions and its ``bag_id`` map are untouched and a re-encode of the decoded row is
positionally identical to the encode of the true one. A free-running decode would move
every boundary and the comparison would confound content with geometry.
"""
from __future__ import annotations

import torch
from torch import Tensor

from morph.model.tul_code import code_target_valid
from morph.model.tul_layout import SlotLayout

__all__ = ["PinnedCells", "span_slots", "check_span_layout", "row_logits",
           "position_logits", "greedy_decode", "span_nll"]


class PinnedCells:
    """Hold the cells the coda reads FIXED across forwards, and capture the live ones.

    ``kind="code"`` patches ``MORPHTransformer._tul_code_core`` (LCTUL: the cells are
    E's code or the thinker's sample); ``kind="loop"`` patches ``_tul_core`` (a slot-loop
    arm: the cells are the loop's exit states, ``[B, S*M, C]`` on the compact cell axis
    the register/fan reshape to ``[B, S, M, C]``).

    While ``pin`` is None the patch is a pure capture and the forward is the shipped one;
    the identity check in each probe asserts that by comparing a pinned forward at the
    captured value against the unpinned forward, elementwise.
    """

    def __init__(self, model, kind: str):
        if kind not in ("code", "loop"):
            raise ValueError(f"kind must be 'code' or 'loop', got {kind!r}")
        self.model, self.kind = model, kind
        self.pin: Tensor | None = None
        self.last: Tensor | None = None
        self.last_xn: Tensor | None = None
        self._restore = None

    def __enter__(self):
        from morph.model.transformer import MORPHTransformer
        box = self
        if self.kind == "code":
            real = MORPHTransformer._tul_code_core

            def patched(self, x, x0, bigram_emb, layout, code_mode, code_steps, plan_mode,
                        *a, **kw):
                xn, cells, fm, stats, h_slots, depths = real(
                    self, x, x0, bigram_emb, layout, code_mode, code_steps, plan_mode,
                    *a, **kw)
                box.last = cells.detach()
                box.last_xn = xn.detach()
                if box.pin is not None:
                    cells = box.pin.to(cells.dtype)
                    h_slots = cells.mean(dim=2)
                return xn, cells, fm, stats, h_slots, depths

            MORPHTransformer._tul_code_core = patched
            self._restore = lambda: setattr(MORPHTransformer, "_tul_code_core", real)
        else:
            real = MORPHTransformer._tul_core

            def patched(self, x, x0, bigram_emb, layout, *a, **kw):
                xn, h_slots, depths, g_traj, db_traj, gain_reg, mep_keep = real(
                    self, x, x0, bigram_emb, layout, *a, **kw)
                box.last = h_slots.detach()
                box.last_xn = xn.detach()
                if box.pin is not None:
                    h_slots = box.pin.to(h_slots.dtype)
                return xn, h_slots, depths, g_traj, db_traj, gain_reg, mep_keep

            MORPHTransformer._tul_core = patched
            self._restore = lambda: setattr(MORPHTransformer, "_tul_core", real)
        return self

    def __exit__(self, *exc):
        self._restore()
        return False


def span_slots(layout: SlotLayout, max_tokens: int):
    """``(elig, lens, first, src0)`` — the slots whose NEXT span this probe can score.

    ``elig`` is ``code_target_valid`` (slot ``s`` and slot ``s+1`` both valid, i.e. span
    ``s+1`` is a complete span with a slot of its own) narrowed to next spans of 2 to
    ``max_tokens`` tokens — the grader's own eligibility rule. A one-token span carries no
    within-span prefix and a span over the cap cannot be decoded in ``max_tokens`` steps.
    """
    dev = layout.slot_index.device
    B, S = layout.slot_valid.shape
    s_ar = torch.arange(S, device=dev)
    n_tok = ((layout.bag_id.unsqueeze(1) == s_ar.view(1, S, 1))
             & (~layout.slot_mask).unsqueeze(1)).sum(-1)                      # [B, S]
    nxt = torch.zeros_like(n_tok)
    nxt[:, :S - 1] = n_tok[:, 1:]
    elig = code_target_valid(layout) & (nxt >= 2) & (nxt <= int(max_tokens))
    first = layout.slot_index + int(layout.prefix_k)
    src0 = layout.slot_index - 1
    return elig, nxt, first, src0


def check_span_layout(layout: SlotLayout, elig: Tensor, first: Tensor, src0: Tensor) -> None:
    """``_tul_code_grade``'s packer check: a real gate, not a decorative one.

    If the packer ever stops putting a slot's cells immediately after its span, every
    position here is off by ``prefix_k`` and the decode would write over the wrong tokens.
    """
    r, s = elig.nonzero(as_tuple=True)
    p = first[r, s]
    if bool((layout.bag_id[r, p] != (s + 1)).any()) or bool(layout.slot_mask[r, p].any()) \
            or bool(layout.slot_mask[r, src0[r, s]].any()):
        raise RuntimeError(
            "slot_index + prefix_k is not the first TOKEN of the next span (or "
            "slot_index - 1 is not the boundary token): the packer's layout and this "
            "probe disagree and nothing below would score the right positions.")


@torch.no_grad()
def row_logits(model, ids: Tensor, layout: SlotLayout, code_mode: str | None = None) -> Tensor:
    """``[B, L, V]`` the SHIPPED forward's logits on this row. One read path, every commit."""
    kw = {} if code_mode is None else {"code_mode": code_mode}
    return model(ids, slot_layout=layout, **kw)["logits"]


def position_logits(model, logits: Tensor, src: Tensor, live: Tensor) -> Tensor:
    """``[n_live, V]`` fp32 at the LIVE source positions, ``slot_id`` at −inf.

    ``src`` / ``live`` are ``[B, S]``; the rows come back in ``live.nonzero()`` order, the
    order every caller gathers its targets in. The ``slot_id`` mask is the one the CE
    helpers apply (``fused_ce``'s ``mask_token_id``) — a structural position id is not a
    token the model may emit, and the generator refuses one.
    """
    r, c = live.nonzero(as_tuple=True)
    lg = logits[r, src[r, c]].float()
    return lg.index_fill(-1, torch.tensor([model.cfg.tul.slot_id], device=lg.device),
                         float("-inf"))


@torch.no_grad()
def greedy_decode(model, ids: Tensor, layout: SlotLayout, elig: Tensor, lens: Tensor,
                  first: Tensor, src0: Tensor, max_tokens: int,
                  code_mode: str | None = None) -> tuple[Tensor, Tensor]:
    """Greedy, length-matched decode of every eligible next span at once.

    Returns ``(ids_out, cand [B, S, J])``: ``ids_out`` is ``ids`` with each eligible span
    replaced by its decode (same positions, same count), ``cand`` the decoded tokens with
    zeros past a span's own length.
    """
    ids = ids.clone()
    B, S = elig.shape
    J = int(max_tokens)
    cand = torch.zeros(B, S, J, dtype=torch.long, device=ids.device)
    for j in range(J):
        live = elig & (lens > j)
        if not bool(live.any()):
            break
        lg_all = row_logits(model, ids, layout, code_mode)
        src = src0 if j == 0 else first + (j - 1)
        draw = position_logits(model, lg_all, src, live).argmax(dim=-1)
        del lg_all
        r, s = live.nonzero(as_tuple=True)
        ids[r, first[r, s] + j] = draw
        cand[r, s, j] = draw
    return ids, cand


@torch.no_grad()
def span_nll(model, ids: Tensor, layout: SlotLayout, live: Tensor, lens: Tensor,
             first: Tensor, src0: Tensor, max_tokens: int,
             code_mode: str | None = None) -> Tensor:
    """``[B, S]`` SUMMED token NLL of the span already written into ``ids``.

    Teacher forced, so ONE forward and ``max_tokens`` gathers. The sum (not the mean) is
    returned because every table in these probes is token weighted.
    """
    lg_all = row_logits(model, ids, layout, code_mode)
    total = torch.zeros(live.shape, dtype=torch.float32, device=ids.device)
    for j in range(int(max_tokens)):
        step = live & (lens > j)
        if not bool(step.any()):
            break
        src = src0 if j == 0 else first + (j - 1)
        lp = torch.log_softmax(position_logits(model, lg_all, src, step), dim=-1)
        r, s = step.nonzero(as_tuple=True)
        tgt = ids[r, first[r, s] + j]
        total[r, s] = total[r, s] - lp.gather(1, tgt.view(-1, 1)).squeeze(1)
    return total
