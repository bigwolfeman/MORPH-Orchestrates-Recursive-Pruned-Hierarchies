"""Residual-stream building blocks for the MORPH transformer block.

Contents:
  ChannelInject — additive injection of a signal (x0 skip, value embeds, diagonal
                  injection) into a fixed channel slice of the residual stream.
  MORPHBlock    — pre-norm attention + MLP block whose residual is a
                  HyperConnectionResidual (Cayley/JPmHC n-stream mixer; see
                  hyper_connections.py), with an optional parallel GLA retention branch.

The residual stream is conceptually split into channels (DEFAULT_CHANNEL_DIMS) so each
injected signal type has a dedicated slice; ChannelInject targets those slices.

Design notes
------------
- bf16 compatible: dtype casts handled at injection boundaries.
- torch.compile friendly: no Python control flow on tensor values, no in-place ops on
  views, no dynamic shapes.
- MORPHBlock takes pre-built attention and MLP modules, keeping this file decoupled from
  the model internals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Channel layout ────────────────────────────────────────────────────────────

# Channel layout of the residual stream (must sum to the configured d_model). ChannelInject targets
# these slices so each injected signal type has a dedicated region:
#   Ch0 — Compute (384): attention + MLP primary output.
#   Ch1 — Context (256): x0 skip, value embeds, loop injection.
#   Ch2 — Slow    (128): low-rate slice.
DEFAULT_CHANNEL_DIMS: tuple[int, ...] = (384, 256, 128)


# ── ChannelInject ─────────────────────────────────────────────────────────────

class ChannelInject(nn.Module):
    """Inject a signal into a specific channel slice of the residual stream.

    Useful for targeted injection of:
      - x0 skip connection → Context channel (e.g. dims 384:640)
      - value embeddings   → Context channel
      - diagonal injection → Context channel

    Injection: h[..., start:end] += scale · project(signal)

    A learned raw scalar `log_scale` modulates magnitude (not sigmoid-gated —
    allows negative scales, simpler gradient flow early in training).
    An optional Linear projection handles d_signal ≠ channel_width.

    All tensors constructed without in-place ops for torch.compile safety.

    Args:
        channel_start: start index into d_model.
        channel_end:   end index into d_model.
        d_signal:      dimension of the injected signal.
        init_scale:    initial value of the raw scalar gate. Use 0.0 to
                       start with zero injection (safe for all signal types).
    """

    def __init__(
        self,
        channel_start: int,
        channel_end: int,
        d_signal: int,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.start = channel_start
        self.end   = channel_end
        channel_width = channel_end - channel_start

        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)))

        if d_signal != channel_width:
            self.proj: nn.Module = nn.Linear(d_signal, channel_width, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)      # type: ignore[union-attr]
        else:
            self.proj = nn.Identity()

    def precompute(self, signal: Tensor) -> Tensor:
        """Project + scale a signal into the channel-width additive term.

        Returns ``scale · project(signal)`` of shape ``[..., channel_width]``.

        This is the loop-invariant part of :meth:`forward` for a signal that
        does not change across iterations (e.g. the cloned ``x0`` skip).
        Compute it once outside the loop, then feed each iteration through
        :meth:`apply_precomputed` to avoid recomputing the projection.
        """
        if isinstance(self.proj, nn.Linear):
            s = F.linear(signal, self.proj.weight.to(signal.dtype))
        else:
            s = signal
        scale = self.log_scale.to(s.dtype)
        return scale * s

    def apply_precomputed(self, h: Tensor, term: Tensor) -> Tensor:
        """Add a pre-projected additive ``term`` to the channel slice of h.

        ``term`` must be the output of :meth:`precompute` (shape
        ``[..., channel_width]``). Equivalent to :meth:`forward` but skips the
        projection + scale, which the caller has already done once.

        Stream-adaptive: when the carrier ``h`` is an ``[B, S, n, C]`` Hyper-Connection
        n-stream tensor but ``term`` is a single-stream ``[B, S, W]`` signal (x0 / value
        embeds, which live outside the streams), broadcast the signal into every stream by
        inserting the stream axis. A native n-stream signal (term.ndim == h.ndim) is added
        as-is. Single-stream carriers (h.ndim == 3) are unaffected.
        """
        if term.ndim == h.ndim - 1:
            term = term.unsqueeze(-2)            # [B,S,W] → [B,S,1,W] broadcasts over n streams
        prefix = h[..., :self.start]
        target = h[..., self.start:self.end] + term.to(h.dtype)
        suffix = h[..., self.end:]
        return torch.cat([prefix, target, suffix], dim=-1)

    def forward(self, h: Tensor, signal: Tensor) -> Tensor:
        """Inject signal into channel slice of h.

        Args:
            h:      [B, S, D] full residual stream.
            signal: [B, S, d_signal] signal to inject.

        Returns:
            [B, S, D] h with channel slice updated. No in-place ops.
        """
        # forward == precompute (projection) then apply. Kept as one path for
        # the prelude/coda (called once each, no loop-invariance to exploit).
        return self.apply_precomputed(h, self.precompute(signal))


# ── PassLoRA ──────────────────────────────────────────────────────────────────

class PassLoRA(nn.Module):
    """Per-pass low-rank deltas on ONE shared core block (``tul.pass_lora_rank``).

    Bae et al. 2024, "Relaxed Recursive Transformers: Effective Parameter Sharing with
    Layer-wise LoRA". A looped core applies the SAME weights at every pass, so every pass
    computes the identical map; their relaxation keeps the shared weights and gives each
    position in the recursion its own small low-rank delta. Here the recursion index is the
    slot loop's pass ``t``, and the delta is an additive rank-``r`` branch on a sublayer:

        y_t = sublayer(x) + B_t (A_t x),   A_t [r, C], B_t [C, r], B zero-init

    so pass 0 of a fresh model is bit-identical to the block without the module.

    GRANULARITY, stated plainly because it differs from the paper. Bae puts a LoRA on each
    linear of the shared layer (q, k, v, o, up, gate, down). This puts ONE delta on each
    targeted SUBLAYER — the attention sublayer (input ``norm_attn(x)``, output the
    attention branch's ``[B, S, C]``) and the MLP sublayer (input ``norm_mlp(x)``, output
    the SwiGLU's ``[B, S, C]``). Both are C -> C, so a delta is a genuine rank-r
    perturbation of the map each sublayer computes, and the attention delta lands exactly
    where the output projection's does. What this granularity CANNOT express is a delta
    that acts INSIDE the SwiGLU nonlinearity (a separate one on ``gate_up``) or on the
    attention's q/k/v before the score. It was chosen for three reasons: one mechanism at
    one code site instead of two; the attention projections are not reachable as modules
    (several read ``.weight`` and matmul it themselves, and the block's attention takes no
    iteration argument); and, decisively, plain ``nn.Parameter`` tensors on a non-Linear
    module are invisible to BOTH schedules that rewrite this tree's weights — ternary QAT
    walks ``nn.Linear`` / ``nn.Embedding`` / CMS modules, and prune/carve/the deploy packer
    walk ``MortarLinear`` / ``CMSBlockLinear``. The deltas are therefore never ternarised,
    never pruned, never carved and never packed, by construction rather than by an
    exclusion list someone has to maintain.

    The pass index is an INDEX INTO A STACKED PARAMETER (``self.A_attn[t]``), never a
    Python branch, so ``torch.compile`` sees no data-dependent control flow — the same
    shape the ReMoE router's ``iter_embed`` already uses.

    Init draws from a PRIVATE generator, so attaching this module leaves the model's RNG
    stream and every other weight byte-identical to a build without it.
    """

    LEGAL_TARGETS = ("attn", "mlp")

    def __init__(self, d_model: int, n_passes: int, rank: int,
                 targets: tuple[str, ...], seed: int = 0x10A):
        super().__init__()
        if n_passes < 1:
            raise ValueError(f"PassLoRA needs n_passes >= 1, got {n_passes}")
        if rank < 1:
            raise ValueError(f"PassLoRA needs rank >= 1, got {rank}")
        bad = [t for t in targets if t not in self.LEGAL_TARGETS]
        if bad or not targets:
            raise ValueError(
                f"PassLoRA targets must be a non-empty subset of {self.LEGAL_TARGETS}, "
                f"got {tuple(targets)}")
        self.n_passes, self.rank = int(n_passes), int(rank)
        self.targets = tuple(targets)
        self.has_attn = "attn" in self.targets
        self.has_mlp = "mlp" in self.targets
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        std = d_model ** -0.5
        for name in self.targets:
            # A: the standard LoRA down-projection, drawn small. B: EXACTLY zero, so the
            # whole module is a no-op at step 0 and dL/dA is zero there too (the standard
            # LoRA cold start — B moves first, then A follows).
            self.register_parameter(f"A_{name}", nn.Parameter(
                torch.empty(self.n_passes, self.rank, d_model).normal_(0.0, std, generator=g)))
            self.register_parameter(f"B_{name}", nn.Parameter(
                torch.zeros(self.n_passes, d_model, self.rank)))

    def delta(self, which: str, x: Tensor, t: int) -> Tensor:
        """``B_t (A_t x)`` for pass ``t``, in ``x``'s dtype (bf16 under autocast)."""
        a = getattr(self, f"A_{which}")[t].to(x.dtype)
        b = getattr(self, f"B_{which}")[t].to(x.dtype)
        return F.linear(F.linear(x, a), b)

    def extra_repr(self) -> str:
        return (f"n_passes={self.n_passes}, rank={self.rank}, "
                f"targets={self.targets}")


# ── MORPHBlock ────────────────────────────────────────────────────────────────

class MORPHBlock(nn.Module):
    """Pre-norm attention + MLP transformer block with a HyperConnection residual.

    Each sublayer (attention, MLP) is wrapped in a HyperConnectionResidual (Cayley/JPmHC
    n-stream mixer; see hyper_connections.py) — the carrier is [B, S, n, C]. The attention
    and MLP modules are unchanged (single [B,S,C] in/out); only the residual connection
    mixes streams. An optional GLA retention branch can be attached in parallel to the
    attention sublayer (attach_retention), gated off at init.

    Accepts pre-built attention and MLP modules so this file stays decoupled from the model
    internals. RMSNorm operates on the full stream-averaged input, not per-channel.

    Args:
        norm_attn: normalization module for the attention sublayer.
        attn:      attention module. forward(x) → [B, T, D].
        norm_mlp:  normalization module for the MLP sublayer.
        mlp:       MLP module. forward(x) → [B, T, D].
        dropout:   dropout rate applied after each sublayer output.
        d_model:   per-stream feature width C (required — HyperConnectionResidual needs it).
        hc_kwargs: kwargs forwarded to HyperConnectionResidual (n_streams, tau, cayley_*, …).

    Note: the residual attributes are named ``mrr_attn`` / ``mrr_mlp`` for checkpoint
    compatibility with earlier runs; they hold HyperConnectionResidual modules.
    """

    def __init__(
        self,
        norm_attn: nn.Module,
        attn: nn.Module,
        norm_mlp: nn.Module,
        mlp: nn.Module,
        dropout: float = 0.0,
        d_model: int | None = None,
        hc_kwargs: dict | None = None,
    ):
        super().__init__()
        self.norm_attn = norm_attn
        self.attention = attn
        self.norm_mlp  = norm_mlp
        self.mlp       = mlp
        self.drop      = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Residual = HyperConnectionResidual (Cayley/JPmHC), the sole supported residual:
        # an n-stream [B,S,n,C] carrier with an orthogonal stream mixer (exact dynamical
        # isometry for the weight-tied loop). Built once here (branch-free hot path). The
        # attribute names mrr_attn / mrr_mlp are kept for checkpoint compatibility.
        from .hyper_connections import HyperConnectionResidual
        assert d_model is not None, "HyperConnectionResidual needs d_model"
        hk = dict(hc_kwargs or {})
        self.mrr_attn: nn.Module = HyperConnectionResidual(d_model, **hk)
        self.mrr_mlp:  nn.Module = HyperConnectionResidual(d_model, **hk)

        # Retention branch (#230) — attached post-construction so it does NOT perturb
        # the base init RNG (keeps the rest of the model byte-identical to the baseline, so the
        # ablation isolates the retention branch and nothing else). None unless attach_retention.
        self.retention: nn.Module | None = None
        self.norm_ret: nn.Module | None = None
        self.ret_gate: nn.Parameter | None = None

        # Per-pass low-rank deltas (tul.pass_lora_rank). Attached post-construction for
        # the SAME reason as retention: PassLoRA draws from a private generator, and
        # attaching it last leaves every other weight of the model byte-identical.
        # None — the default — makes both call sites below Python-level no-ops.
        self.pass_lora: PassLoRA | None = None

    def attach_retention(self, gla: nn.Module, norm: nn.Module, gate_init: float) -> None:
        """Add a gated GLA branch in PARALLEL to the attention sublayer.

        sublayer output becomes  attn(x) + sigmoid(ret_gate) · gla(norm_ret(x), state).
        gate_init very negative → sigmoid ≈ 0 → branch ≈ off at init (identity to baseline);
        the gate is learnable so the model can open it if retention helps (the key diagnostic).
        """
        self.retention = gla
        self.norm_ret = norm
        self.ret_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def attach_pass_lora(self, lora: "PassLoRA") -> None:
        """Give this block its per-pass low-rank deltas (Bae et al. 2024).

        Adds NEW parameters, so the optimizer must be built after this. Only the CORE
        blocks get one: prelude and coda run once and have no pass index.
        """
        self.pass_lora = lora

    def forward(
        self,
        h: Tensor,
        attn_kwargs: dict | None = None,
        mlp_kwargs: dict | None = None,
        next_inject_term: Tensor | None = None,
        ret_state: Tensor | None = None,
        ret_capture: dict | None = None,
        ret_reset_mask: Tensor | None = None,
        pass_idx: int = 0,
    ) -> Tensor:
        """Forward pass: attention sublayer then MLP sublayer with HC residuals.

        Args:
            h:           [B, T, D] residual stream.
            attn_kwargs: optional keyword arguments forwarded to attention.
            mlp_kwargs:  optional keyword arguments forwarded to mlp.
            next_inject_term: [B, S, C] | None — carrier-engine (HC only): the NEXT layer's
                         injection term, folded into THIS block's MLP-residual POST write
                         (the block's last carrier write), so the next layer skips a separate
                         _apply_injection. Only set in HC carrier-engine mode.
            ret_state:   [B, H, dk, dv] | None — retention (GLA) initial state for this block's
                         branch (the cross-iteration carry in the core loop). None → zero state.
            ret_capture: dict | None — if given and this block has retention, the new GLA state
                         is written to ret_capture["state"]. The CALLER must RETURN that state
                         from any checkpointed region (side-channel capture is not checkpoint-safe
                         on its own); _core_step does exactly that.
            ret_reset_mask: [B, T] bool | None — GLA segment-reset mask (docs/tul-tg-spec.md
                         §4), forwarded to the retention branch untouched. Ignored when this
                         block has no retention branch.
            pass_idx:    which loop pass is applying this shared block. Read ONLY by
                         `pass_lora` (tul.pass_lora_rank), which indexes its stacked
                         parameters with it. Ignored — and the default 0 is never even
                         looked at — on a block without one, which is every block of every
                         model that does not set the knob.

        Returns:
            [B, T, D] updated residual stream.
        """
        attn_kwargs = attn_kwargs or {}
        mlp_kwargs  = mlp_kwargs  or {}
        # Python-level constant per module instance (it is set once at construction and
        # never rebound), so both branches below trace out on a model without the knob.
        lora = self.pass_lora

        # ── tul.loop_carry="persist" on a Thought Register — the READER FIX ────────
        # (2026-09-22 correction; `morph/model/transformer.py::_tul_core`, "THE READER
        # FIX", is the one place that builds this payload — ONLY for
        # `tul.loop_carry="persist"` with `tul.slot_cells > 1`.) `"tg_persist_capture"`
        # is a private transport, not a `MORPHAttention` kwarg (it would TypeError if
        # forwarded), so it is popped HERE, once, into its own local. The `dict(...)`
        # copy fires ONLY when the key is present: `_core_akw[0]` is the SAME dict
        # object shared by every pass of the loop, so popping it IN PLACE would strip
        # the key after pass 1 and silently stop capturing from pass 2 on — every other
        # call site (every arm without this key) keeps the zero-copy path it always
        # had. `None` is what makes the `if` inside `_attn_fn` below a Python-level
        # no-op on every block of every model without a register+persist arm.
        _persist_capture = None
        if "tg_persist_capture" in attn_kwargs:
            attn_kwargs = dict(attn_kwargs)
            _persist_capture = attn_kwargs.pop("tg_persist_capture")

        def _attn_fn(x: Tensor) -> Tensor:
            xa = self.norm_attn(x)
            if _persist_capture is not None:
                # ONE extra core-layer-0 attention application, on the EXACT SAME `xa`
                # the real call two lines down is about to consume — `x` here is
                # already `x_bar`, the Hyper-Connection residual's OWN mixed
                # single-stream view (`HyperConnectionResidual.forward`'s pre-map),
                # which is why this sits INSIDE `_attn_fn` and not one level up in
                # `_apply_core_step`: nothing outside this closure has access to
                # `x_bar` without re-deriving the Cayley/softmax mapping by hand.
                # `cross` excludes a cell's OWN slot entirely (not merely self, unlike
                # the real call's `tg_relation` below), which is the whole fix — see
                # the transformer.py comment for why. WITH GRAD, no `torch.no_grad()`:
                # the persist term must carry gradient back to the PREVIOUS slot's
                # cells this call's K/V read. Its own output tensor is otherwise
                # unused; only `capture_dict["win"]` (`_CCABase._gate_combine_up`) is
                # read, by `_core_step`'s `want_carry` branch, exactly where the
                # ordinary `tg_win_capture` path (below, on every OTHER arm) writes it.
                _p_relation, _p_seg, _p_cap = _persist_capture
                self.attention(xa, tg_relation=_p_relation, tg_seg=_p_seg,
                               tg_win_capture=_p_cap)
            a = self.attention(xa, **attn_kwargs)
            if lora is not None and lora.has_attn:
                # The delta rides the attention branch's own input and output, i.e. it
                # lands where the output projection's contribution does. Cast to `a`'s
                # dtype first: RMSNorm returns fp32 even under autocast, and computing the
                # delta in fp32 would both cost more and promote the whole sublayer.
                a = a + lora.delta("attn", xa.to(a.dtype), pass_idx)
            if self.retention is not None:
                g_out, s_out = self.retention(self.norm_ret(x), initial_state=ret_state,
                                              reset_mask=ret_reset_mask)
                if ret_capture is not None:
                    ret_capture["state"] = s_out
                a = a + torch.sigmoid(self.ret_gate).to(a.dtype) * g_out
            return self.drop(a)

        def _mlp_fn(x: Tensor) -> Tensor:
            xm = self.norm_mlp(x)
            y = self.mlp(xm, **mlp_kwargs)
            if lora is not None and lora.has_mlp:
                y = y + lora.delta("mlp", xm.to(y.dtype), pass_idx)
            return self.drop(y)

        h = self.mrr_attn(h, _attn_fn)
        if next_inject_term is not None:
            # HC carrier-engine: fold the next layer's inject into the MLP POST write.
            h = self.mrr_mlp(h, _mlp_fn, post_inject=next_inject_term)
        else:
            h = self.mrr_mlp(h, _mlp_fn)
        return h
