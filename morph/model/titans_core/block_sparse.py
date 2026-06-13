"""Block-sparse linear layer with CMS tile scoring and MORTAR carving.

CMSBlockLinear is a drop-in replacement for nn.Linear with two phases:

Pre-carve (dense mode): weights stored as a standard [out, in] matrix
(F.linear → cuBLAS). Gradient-based tile saliency (block_score_ema) drives
structured masked-dense pruning (prune_step_blocks zeroes 128×128 blocks in
the dense matrix — no sparse overhead, no optimizer rebuild).

Post-carve (MORTAR mode): carve() packs the masked-dense weight into
128×128 BCSR blocks executed by the vendored stk Triton backend
(morph.sparse.stk) — measured 3.09× faster than dense at 0.25 density.

MORTAR is the ONLY sparse backend. The legacy 16×16 Block-ELL format
(compact()/values/col_indices + its Triton kernels) was removed 2026-06-11;
its kernel measured slower than dense (below MMA granularity).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .topology_scorer import TopologyScorer, compute_gradient_frobenius_norms


# ============================================================================
# FP8 dense-mode matmul (sm_120 native torch._scaled_mm)
# ============================================================================
# E4M3 has a max representable magnitude of 448.0; E5M2 max is 57344.0.
# Dynamic per-tensor scaling: fp8 = hp * (FP8_MAX / amax). torch._scaled_mm
# computes (a_fp8 * scale_a) @ (b_fp8 * scale_b).T in fp32 accum, so the
# scale we feed back is the *dequant* factor = amax / FP8_MAX = 1 / cast_scale.
_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
# Global counter so the verify script can assert how many times the weight
# was actually cast to FP8 (proves the per-step cache eliminates loop recasts).
_FP8_WEIGHT_CAST_COUNT = 0


def _amax_to_dequant_scale(amax: Tensor, fp8_max: float) -> Tensor:
    """Return the per-tensor dequant scale (amax / fp8_max) as fp32 [1,1].

    This is what torch._scaled_mm multiplies the fp8 operand by to recover
    high-precision magnitude. The cast (quantize) factor is its reciprocal.
    """
    amax = amax.to(torch.float32).clamp(min=1e-12)
    return (amax / fp8_max).reshape(1, 1)


def _hp_to_fp8(t: Tensor, fp8_dtype: torch.dtype, fp8_max: float) -> Tuple[Tensor, Tensor]:
    """Dynamic per-tensor cast of a high-precision tensor to fp8.

    Returns (t_fp8, dequant_scale[1,1] fp32). Matches torchao E4M3 dynamic
    semantics: scale by FP8_MAX/amax then cast (saturating).

    Perf note: the amax reduction goes to fp32 (scalar), but the
    element-wise scale stays in the input dtype (bf16 under autocast) to avoid
    a full fp32 materialization of the activation — that upcast was measured at
    2-3x the cost of the whole cast on a 5090, and it dominated the FP8 path.
    """
    # fp32 amax scalar (cheap reduction) for an accurate, overflow-safe scale.
    amax = t.abs().amax().to(torch.float32)
    dequant = _amax_to_dequant_scale(amax, fp8_max)  # amax / fp8_max [1,1] fp32
    # cast_scale = fp8_max / amax, applied in the tensor's own dtype.
    cast_scale = (fp8_max / amax.clamp(min=1e-12)).to(t.dtype)
    t_fp8 = (t * cast_scale).to(fp8_dtype)
    return t_fp8, dequant


class _FP8DenseMatmul(torch.autograd.Function):
    """Autograd-correct FP8 dense matmul: y = x @ w.T + bias.

    Forward inputs (E4M3 for both x and w), fp32 accumulation, bf16 out.
    Backward uses E5M2 for grad_output (more dynamic range, standard FP8
    training recipe) and E4M3 for the re-cast operands:
        grad_input  = scaled_mm(grad_out_e5m2, w_e4m3)        -> [.., in]
        grad_weight = scaled_mm(grad_out_e5m2.T, x_e4m3)      -> [out, in]

    The forward weight is supplied PRE-CAST + detached (from the per-step
    cache), so no autograd flows through the cast. The full-precision weight
    `w_hp` is saved ONLY to re-cast it in backward for grad_input — grad_weight
    is produced by the scaled_mm above and routed back to the leaf via the
    pass-through of `w_hp` (its .grad slot), giving a real gradient to
    self.weight.
    """

    @staticmethod
    def forward(ctx, x, w_hp, w_fp8, w_dequant, bias):
        # x: [..., in] (bf16 under autocast). Flatten to 2D for scaled_mm.
        orig_shape = x.shape
        in_features = orig_shape[-1]
        x2d = x.reshape(-1, in_features)

        x_fp8, x_dequant = _hp_to_fp8(x2d, torch.float8_e4m3fn, _E4M3_MAX)

        # y = (x_fp8 * x_dequant) @ (w_fp8 * w_dequant).T
        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=x_dequant,
            scale_b=w_dequant,
            bias=None,
            out_dtype=torch.bfloat16,
        )
        if bias is not None:
            out = out + bias.to(out.dtype)

        ctx.save_for_backward(x2d, w_hp, bias)
        ctx.orig_shape = orig_shape
        ctx.out_features = w_hp.shape[0]
        return out.reshape(*orig_shape[:-1], ctx.out_features)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, w_hp, bias = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        out_features = ctx.out_features

        grad_out2d = grad_out.reshape(-1, out_features).contiguous()

        grad_input = grad_weight = grad_bias = None

        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        need_b = bias is not None and ctx.needs_input_grad[4]

        if need_x or need_w:
            # grad_output cast as E5M2 (more range for gradient magnitudes).
            go_fp8, go_dequant = _hp_to_fp8(grad_out2d, torch.float8_e5m2, _E5M2_MAX)

        # torch._scaled_mm contract: mat1 row-major, mat2 COL-major. Eager is
        # lenient but torch.compile's fake-tensor path enforces it strictly
        # ("mat2 must be col_major"). A col-major [K,N] operand is a contiguous
        # [N,K] tensor viewed via .t(), so we cast the TRANSPOSED tensor
        # contiguous, then .t() it back for the scaled_mm call.
        if need_x:
            # grad_input = grad_out[M,out] @ w[out,in] -> [M, in].
            # mat2 = w as col-major [out,in] == contiguous [in,out] .t().
            # Re-cast the full-precision weight to E4M3 (consistent with fwd).
            w_t_fp8, w_dequant_e4m3 = _hp_to_fp8(
                w_hp.t().contiguous(), torch.float8_e4m3fn, _E4M3_MAX
            )  # [in, out] contiguous
            grad_input2d = torch._scaled_mm(
                go_fp8,
                w_t_fp8.t(),  # col-major [out, in], contract over out
                scale_a=go_dequant,
                scale_b=w_dequant_e4m3,
                bias=None,
                out_dtype=torch.bfloat16,
            )
            grad_input = grad_input2d.reshape(orig_shape)

        if need_w:
            # grad_weight = grad_out.T[out,M] @ x[M,in] -> [out, in].
            # mat1 = go.t() row-major [out,M] == contiguous(go.t()).
            # mat2 = x as col-major [M,in] == contiguous [in,M] .t().
            x_t_fp8, x_dequant = _hp_to_fp8(
                x2d.t().contiguous(), torch.float8_e4m3fn, _E4M3_MAX
            )  # [in, M] contiguous
            grad_weight = torch._scaled_mm(
                go_fp8.t().contiguous(),  # row-major [out, M]
                x_t_fp8.t(),  # col-major [M, in], contract over M
                scale_a=go_dequant,
                scale_b=x_dequant,
                bias=None,
                out_dtype=torch.float32,
            )

        if need_b:
            grad_bias = grad_out2d.sum(dim=0).to(bias.dtype)

        # Order matches forward args: (x, w_hp, w_fp8, w_dequant, bias)
        # grad_weight is routed through w_hp (the leaf), w_fp8/w_dequant get None
        # (they are detached cache tensors, not differentiable inputs).
        return grad_input, grad_weight, None, None, grad_bias


class CMSBlockLinear(nn.Module):
    """Dual-mode linear layer: dense pre-carve, MORTAR 128×128 BCSR post-carve.

    Drop-in replacement for nn.Linear with:
    - Dense [out, in] weight storage pre-carve (cuBLAS GEMM)
    - Gradient-based tile saliency (block_score_ema) for structured pruning
    - prune_step_blocks: masked-dense, block-aligned density reduction
    - carve(): pack the masked weight into MORTAR BCSR storage (stk backend)

    Args:
        in_features: Input dimension (must be divisible by tile_size)
        out_features: Output dimension (must be divisible by tile_size)
        tile_size: Saliency tile size (default 16; execution blocks are 128×128)
        density: Fraction of active tiles per row (0.1 to 1.0)
        bias: Include bias term (default True)
        score_ema_alpha: EMA momentum for gradient scores (default 0.95)
        device: Target device
        dtype: Parameter dtype

    Raises:
        ValueError: If dimensions not divisible by tile_size
        ValueError: If density not in [0.1, 1.0]

    Example:
        >>> layer = CMSBlockLinear(640, 2560, tile_size=16, density=1.0)
        >>> x = torch.randn(32, 640)
        >>> y = layer(x)  # [32, 2560]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_size: int = 16,
        density: float = 0.5,
        bias: bool = True,
        score_ema_alpha: float = 0.95,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the dual-mode (dense → MORTAR) linear layer."""
        super().__init__()

        # Validate inputs
        if in_features % tile_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tile_size ({tile_size})"
            )
        if out_features % tile_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tile_size ({tile_size})"
            )
        if not (0.1 <= density <= 1.0):
            raise ValueError(f"density ({density}) must be in [0.1, 1.0]")

        # Core dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.density = density
        self.score_ema_alpha = score_ema_alpha
        # Importance scoring mode for accumulate_scores / prune_step_blocks.
        #   "grad"      → ‖∇W_block‖_F   (default — bit-identical to the original behaviour)
        #   "taylor"    → ‖W_block ⊙ ∇W_block‖_F   (Molchanov first-order saliency, est. Δloss)
        #   "magnitude" → ‖W_block‖_F   (lottery-ticket criterion)
        self.score_mode: str = "grad"
        # Structured masked-dense pruning state (pre-carve). prune_mask is [R, C] bool
        # (True = alive); None until the first prune_step_blocks. _prune_elem_mask is the
        # cached [out, in] elementwise mask used to re-zero dead tiles in the live weight
        # every step.
        self._prune_mask: Optional[Tensor] = None
        self._prune_elem_mask: Optional[Tensor] = None

        # Derived dimensions
        self.R = out_features // tile_size  # output block-rows
        self.C = in_features // tile_size  # input block-columns
        self.K = max(1, int(self.C * density))  # active blocks per row

        # Dense mode: store weights as standard [out, in] matrix for cuBLAS speed.
        # After carve(), transitions to MORTAR BCSR storage.
        self._dense_mode = True
        self._ternary_mode = False
        # Post-carve ternary QAT on mortar_data (Phase C: continue ternary pretraining
        # after carving). Internal STE flag; mortar_data stays the smooth shadow and the
        # carved forward applies the STE.
        self._values_ternary_mode = False
        self._values_threshold = 0.5
        # MORTAR (Macro-Orchestrated Routing and Tile-Aligned Recompaction): post-carve
        # BCSR block-sparse mode executing 128×128 blocks via the vendored stk Triton
        # backend (morph.sparse.stk). A layer is dense OR MORTAR-carved.
        # Set by carve(); storage = mortar_data [nnz, blk, blk] + 6 BCSR index buffers.
        self._mortar = False
        self._mortar_blocking = 0
        # FP8 dense-mode matmul (sm_120 torch._scaled_mm). Off by default →
        # bit-identical to the existing F.linear path. Enable via enable_fp8().
        self._fp8_mode = False
        # Per-optimizer-step FP8 weight cache (keyed on self.weight._version):
        self._fp8_weight_cache = None     # (w_fp8 e4m3, dequant_scale fp32 [1,1])
        self._fp8_weight_version = -1     # last weight._version we cast for

        # Dense weight parameter — same as nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # Gradient importance EMA [R, K] — the prune saliency accumulator.
        self.register_buffer(
            "block_score_ema",
            torch.zeros(self.R, self.K, device=device, dtype=dtype or torch.float32),
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weight parameters matching nn.Linear's kaiming_uniform(a=sqrt(5))."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def enable_ternary(self) -> None:
        """Transition from dense → dense+ternary (Phase 1 → Phase 2).
        Shadow weight IS self.weight — zero extra memory. STE passes gradients through."""
        assert self._dense_mode, "Ternary only supported in dense mode"
        self._ternary_mode = True
        B = self.tile_size
        w_tiles = self.weight.data.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
        tile_means = w_tiles.abs().mean(dim=(2, 3))  # [R, C]
        self.register_buffer("ternary_scale", tile_means.clamp(min=1e-6))

    def enable_fp8(self) -> None:
        """Turn on the FP8 dense-mode matmul path (sm_120 torch._scaled_mm).

        E4M3 inputs+weight, fp32 accumulation, bf16 out, dynamic per-tensor
        scaling. Mutually exclusive with ternary (a weight can't be both).
        Only meaningful in dense mode; the sparse/post-compact path is untouched.
        """
        assert not self._ternary_mode, "FP8 and ternary are mutually exclusive"
        assert self._dense_mode, "FP8 dense path only applies in dense mode"
        self._fp8_mode = True
        # Invalidate cache so the first forward recomputes against the current
        # weight version.
        self._fp8_weight_cache = None
        self._fp8_weight_version = -1

    def _get_cached_fp8_weight(self) -> Tuple[Tensor, Tensor]:
        """Return (w_fp8 E4M3, dequant_scale) for the current weight, recasting
        ONLY when self.weight._version has changed (i.e. once per optimizer
        step, after the in-place update). Reused across the ~6 loop iterations.

        The cached tensors are DETACHED (no autograd through the cache) — the
        weight gradient is produced by the backward scaled_mm in
        _FP8DenseMatmul, not by differentiating this cast. This is what makes
        it checkpoint-safe: on the recomputed forward inside
        torch.utils.checkpoint, weight._version is unchanged → cache hit →
        identical FP8 weight.
        """
        version = self.weight._version
        cache = self._fp8_weight_cache
        if cache is not None and self._fp8_weight_version == version:
            return cache

        global _FP8_WEIGHT_CAST_COUNT
        _FP8_WEIGHT_CAST_COUNT += 1
        with torch.no_grad():
            w_fp8, w_dequant = _hp_to_fp8(
                self.weight.detach(), torch.float8_e4m3fn, _E4M3_MAX
            )
        self._fp8_weight_cache = (w_fp8, w_dequant)
        self._fp8_weight_version = version
        return self._fp8_weight_cache

    def _ternary_ste(self, w: Tensor) -> Tensor:
        """Quantize to {-1, 0, +1} × per-tile scale with straight-through estimator."""
        B = self.tile_size
        w_tiles = w.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)  # [R, C, B, B]
        scale = self.ternary_scale.unsqueeze(-1).unsqueeze(-1)  # [R, C, 1, 1]
        w_scaled = w_tiles / scale.clamp(min=1e-6)
        w_ternary = torch.sign(w_scaled) * (w_scaled.abs() > 0.5).float()
        result = scale * (w_scaled + (w_ternary - w_scaled).detach())
        return result.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def enable_values_ternary(self, threshold: float = 0.5) -> None:
        """Continue ternary QAT on the post-carve ``mortar_data`` (per-tensor symmetric).

        ``mortar_data`` stays the smooth trainable shadow; the carved forward applies the
        STE so the effective sparse weight is ternary. Mirrors the dense ``_ternary_mode``
        path. Per-tensor symmetric (the deploy ``scale_group='tensor'`` config).
        """
        assert not self._dense_mode, "enable_values_ternary requires a carved (sparse) layer"
        self._values_ternary_mode = True
        self._values_threshold = float(threshold)
        # MORTAR persists these flags in a buffer so they survive save/load.
        mt = getattr(self, "mortar_ternary", None)
        if mt is not None:
            mt[0] = 1.0
            mt[1] = float(threshold)

    def update_ternary_scales(self) -> None:
        """Recompute per-tile scales from current shadow weights."""
        if not self._ternary_mode:
            return
        B = self.tile_size
        w_tiles = self.weight.data.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
        self.ternary_scale = w_tiles.abs().mean(dim=(2, 3)).clamp(min=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: dense (cuBLAS) pre-carve, MORTAR BCSR post-carve.

        Pre-carve (dense mode): F.linear(x, weight, bias) → cuBLAS GEMM.
        Post-carve (MORTAR mode): stk dds Triton kernel over 128×128 blocks.

        Args:
            x: Input tensor [batch, in_features] or [batch, seq, in_features]

        Returns:
            Output tensor with same batch/seq dims, out_features last
        """
        if self._mortar:
            return self._forward_mortar(x)

        if self._fp8_mode:
            # FP8 dense matmul. Cached fp8 weight (per opt-step), fresh fp8
            # activation cast every call. Grad flows to self.weight via the
            # backward scaled_mm, and to x via grad_input.
            w_fp8, w_dequant = self._get_cached_fp8_weight()
            return _FP8DenseMatmul.apply(
                x, self.weight, w_fp8, w_dequant, self.bias
            )
        w = self._ternary_ste(self.weight) if self._ternary_mode else self.weight
        return F.linear(x, w, self.bias)

    def accumulate_scores(self) -> None:
        """Accumulate gradient statistics for importance scoring (pre-carve only).

        Call after backward() each step. Updates block_score_ema (the EMA of
        per-tile gradient Frobenius norms) that prune_step_blocks pools into
        block saliency. Post-carve scoring is over — this is a no-op.

        Contract:
            - Safe to call even if grad is None (no-op)
            - Accumulates into existing EMA (doesn't reset)
        """
        import torch.nn.utils.parametrize as parametrize

        if not self._dense_mode:
            return
        B = self.tile_size
        # Under ternary QAT the weight is a parametrization (STE) → self.weight is a
        # non-leaf computed tensor with .grad = None; the real gradient lives on the
        # smooth shadow leaf. Fall back to it so scoring is NOT a silent no-op.
        w_grad = self.weight.grad
        if w_grad is None and parametrize.is_parametrized(self, "weight"):
            w_grad = self.parametrizations.weight.original.grad
        if w_grad is None:
            return
        grad_tiles = w_grad.reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
        if self.score_mode == "grad":
            grad_norms = compute_gradient_frobenius_norms(grad_tiles)
        elif self.score_mode == "taylor":
            # self.weight = effective (ternary) weight; pairs with dL/d(effective) → |W·∇W|.
            w_tiles = self.weight.detach().reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
            grad_norms = compute_gradient_frobenius_norms(w_tiles * grad_tiles)
        elif self.score_mode == "magnitude":
            w_tiles = self.weight.detach().reshape(self.R, B, self.C, B).permute(0, 2, 1, 3)
            grad_norms = compute_gradient_frobenius_norms(w_tiles)
        else:
            raise ValueError(f"unknown score_mode {self.score_mode!r}")

        scorer = TopologyScorer(self.R, self.C, self.K, ema_alpha=self.score_ema_alpha)

        with torch.no_grad():
            self.block_score_ema = scorer.update_gradient_ema(grad_norms, self.block_score_ema)

    # ── Structured masked-dense pruning (pre-carve) ──────────────────────────
    # Drives real density reduction by ZEROING the lowest-saliency blocks in the
    # dense weight. Kept masked (not carved) so there is no parameter swap /
    # optimizer rebuild during the prune window: the network function at 25%
    # density is identical to its carved twin.

    def _ensure_prune_mask(self) -> None:
        if self._prune_mask is None:
            self._prune_mask = torch.ones(
                self.R, self.C, dtype=torch.bool, device=self.block_score_ema.device
            )

    def prune_density(self) -> float:
        """Fraction of core tiles still alive (1.0 if never pruned)."""
        if self._prune_mask is None:
            return 1.0
        return float(self._prune_mask.float().mean().item())

    def _rebuild_prune_elem_mask(self) -> None:
        """Expand the [R, C] tile mask to a cached [out, in] elementwise mask."""
        B = self.tile_size
        m = self._prune_mask.to(torch.float32)  # [R, C]
        elem = (
            m.view(self.R, 1, self.C, 1)
            .expand(self.R, B, self.C, B)
            .reshape(self.R * B, self.C * B)
        )
        self._prune_elem_mask = elem.contiguous()

    def _prune_target_weight(self) -> Tensor:
        """The LEAF holding the live dense weight — the ternary shadow when parametrized."""
        import torch.nn.utils.parametrize as parametrize

        if parametrize.is_parametrized(self, "weight"):
            return self.parametrizations.weight.original
        return self.weight

    def apply_prune_mask(self) -> None:
        """Re-zero dead tiles (and their grads) in the live weight. Call EVERY step so
        neither the optimizer momentum nor the ternary STE can revive a pruned tile."""
        if not self._dense_mode:
            return
        if self._prune_elem_mask is None:
            # Lazily rebuild from the tile mask (e.g. after a resume) if anything is dead.
            if self._prune_mask is None or bool(self._prune_mask.all()):
                return
            self._rebuild_prune_elem_mask()
        tgt = self._prune_target_weight()
        with torch.no_grad():
            tgt.data.mul_(self._prune_elem_mask.to(tgt.dtype))
            if tgt.grad is not None:
                tgt.grad.mul_(self._prune_elem_mask.to(tgt.grad.dtype))

    def prune_step_blocks(
        self, prune_rate: float, target_density: float, blocking: int = 128
    ) -> dict:
        """MORTAR block-aligned pruning: zero the lowest-saliency 128×128 BLOCKS.

        Score at tile, execute at block: tile saliency (block_score_ema, whatever
        score_mode produced it) is pooled over each (blocking/tile_size)² tile group
        into a block score; the lowest-score alive blocks are dropped GLOBALLY across
        the layer (not per-row) subject to a ≥1-block-per-row floor. Global selection
        is what makes 0.25 density reachable on gate_up's 6 input block-columns
        (row-uniform K would force 1/6 or 2/6) — BCSR tolerates the resulting ragged
        rows, verified in Gate G1.

        The resulting tile mask is exactly block-aligned, so the later carve() is
        LOSSLESS: every weight the mask keeps lands in a kept block.

        Returns {"pruned": n_blocks_dropped, "density": new_block_density}.
        """
        if not self._dense_mode:
            return {"pruned": 0, "density": self.prune_density()}
        B = self.tile_size
        if blocking % B != 0:
            raise ValueError(f"blocking ({blocking}) must be divisible by tile_size ({B})")
        if self.out_features % blocking or self.in_features % blocking:
            raise ValueError(
                f"prune_step_blocks: [{self.out_features},{self.in_features}] not divisible "
                f"by blocking {blocking}"
            )
        tpb = blocking // B                       # tiles per block edge
        Rb = self.out_features // blocking        # block-rows
        Cb = self.in_features // blocking         # block-cols

        self._ensure_prune_mask()
        with torch.no_grad():
            m4 = self._prune_mask.view(Rb, tpb, Cb, tpb)
            block_alive = m4.any(dim=3).any(dim=1)            # [Rb, Cb]
            total = Rb * Cb
            cur_alive = int(block_alive.sum().item())
            target_keep = max(Rb, int(round(target_density * total)))
            if cur_alive <= target_keep:
                return {"pruned": 0, "density": cur_alive / total}
            new_alive = int(math.floor(cur_alive * (1.0 - prune_rate)))
            new_alive = max(target_keep, min(new_alive, cur_alive - 1))  # guarantee progress
            n_drop = cur_alive - new_alive

            # Pool tile saliency → block scores (dead tiles contribute 0, not stale EMA).
            sal_tiles = self.block_score_ema.detach().float() * self._prune_mask.float()
            sal = sal_tiles.view(Rb, tpb, Cb, tpb).sum(dim=(1, 3))      # [Rb, Cb]

            # ≥1-per-row floor: protect each row's best alive block from dropping.
            sal_best = sal.masked_fill(~block_alive, float("-inf"))
            row_best = sal_best.argmax(dim=1)                            # [Rb]
            droppable = sal.masked_fill(~block_alive, float("inf"))      # dead → never re-dropped
            droppable[torch.arange(Rb, device=sal.device), row_best] = float("inf")

            _, drop_idx = droppable.flatten().topk(n_drop, largest=False)
            new_block_alive = block_alive.clone()
            new_block_alive.view(-1)[drop_idx] = False

            # Expand block mask → tile mask (exactly block-aligned by construction).
            self._prune_mask = (
                new_block_alive.view(Rb, 1, Cb, 1)
                .expand(Rb, tpb, Cb, tpb)
                .reshape(self.R, self.C)
                .contiguous()
            )
        self._rebuild_prune_elem_mask()
        self.apply_prune_mask()
        return {"pruned": n_drop, "density": new_alive / total}

    def get_density(self) -> float:
        """Get actual current density.

        Returns:
            K / C (should match configured density)
        """
        return self.K / self.C

    # ─────────────────────────────────────────────────────────────────────
    # MORTAR: carve (dense masked → BCSR 128×128) + carved forward
    # ─────────────────────────────────────────────────────────────────────

    def _init_mortar_storage(
        self,
        data: Tensor,
        row_indices: Tensor,
        column_indices: Tensor,
        offsets: Tensor,
        column_indices_t: Tensor,
        offsets_t: Tensor,
        block_offsets_t: Tensor,
        blocking: int,
    ) -> None:
        """Install MORTAR BCSR storage on this layer (shared by carve() and ckpt load).

        Deletes the dense `weight` leaf (removing any ternary parametrization first,
        keeping the smooth shadow semantics — the discrete ternary is reapplied at
        forward via the values-STE, never baked into storage), registers mortar_data
        as the trainable parameter and the 6 BCSR index tensors as buffers (→ they
        ride through state_dict automatically), and flips the layer into mortar mode.
        """
        import torch.nn.utils.parametrize as _parametrize

        if _parametrize.is_parametrized(self, "weight"):
            _parametrize.remove_parametrizations(self, "weight", leave_parametrized=False)
        if "weight" in self._parameters:
            del self._parameters["weight"]

        self.mortar_data = nn.Parameter(data)
        self.register_buffer("mortar_row_indices", row_indices)
        self.register_buffer("mortar_column_indices", column_indices)
        self.register_buffer("mortar_offsets", offsets)
        self.register_buffer("mortar_column_indices_t", column_indices_t)
        self.register_buffer("mortar_offsets_t", offsets_t)
        self.register_buffer("mortar_block_offsets_t", block_offsets_t)
        # Persisted ternary-QAT state: [flag, threshold]. A plain attr would be lost
        # across save/load; a buffer rides state_dict. Only exists post-carve, so no
        # back-compat impact on pre-MORTAR checkpoints.
        self.register_buffer(
            "mortar_ternary",
            torch.tensor(
                [1.0 if self._values_ternary_mode else 0.0, self._values_threshold],
                dtype=torch.float32, device=data.device,
            ),
        )

        self._mortar = True
        self._mortar_blocking = int(blocking)
        self._dense_mode = False
        self._ternary_mode = False
        self._fp8_mode = False
        # Keep the [R, C] tile _prune_mask (tiny) for exact density logging;
        # drop the [out, in] elementwise mask — pruning is over, it's pure VRAM.
        self._prune_elem_mask = None

    def carve(self, blocking: int = 128) -> int:
        """MORTAR: carve the masked-dense weight into BCSR 128×128 block-sparse storage.

        Packs the alive blocks into hardware-grain 128×128 BCSR storage executed
        by the vendored stk Triton backend (measured 3.09× FASTER than dense at
        0.25 density, Gate G1).

        Lossless when pruning was block-aligned (prune_step_blocks): the kept-block
        union exactly covers the alive tiles. With a legacy tile-level mask, any
        partially-alive block is kept whole (dead tiles ride along as zeros) —
        still numerically exact, just less compression; a warning reports the delta.

        Ternary deploy stack: callers restore the smooth shadow
        (remove_parametrizations leave_parametrized=False) before calling, and
        re-enable per-tensor STE on the carved data after via
        enable_values_ternary() — QAT continues on mortar_data uninterrupted.

        Returns nnz (number of kept blocks).
        """
        assert self._dense_mode, "carve() requires dense mode (pre-carve)"
        assert not self._mortar, "carve() already applied"
        if self.out_features % blocking or self.in_features % blocking:
            raise ValueError(
                f"carve: [{self.out_features},{self.in_features}] not divisible by "
                f"blocking {blocking}"
            )
        B = self.tile_size
        if blocking % B != 0:
            raise ValueError(f"carve: blocking ({blocking}) not divisible by tile_size ({B})")

        from morph.sparse.stk.matrix import _transpose as _stk_transpose

        with torch.no_grad():
            tpb = blocking // B
            Rb = self.out_features // blocking
            Cb = self.in_features // blocking
            w = self._prune_target_weight().data  # smooth shadow when ternary-parametrized
            device, dtype = w.device, w.dtype

            # Tile alive mask → block mask. Never-pruned layer carves fully dense.
            if self._prune_mask is not None:
                tile_mask = self._prune_mask.to(device)
                # Enforce the mask (defense in depth — apply_prune_mask should have).
                if self._prune_elem_mask is None:
                    self._rebuild_prune_elem_mask()
                w = w * self._prune_elem_mask.to(dtype=dtype, device=device)
            else:
                tile_mask = torch.ones(self.R, self.C, dtype=torch.bool, device=device)

            m4 = tile_mask.view(Rb, tpb, Cb, tpb)
            block_mask = m4.any(dim=3).any(dim=1)            # [Rb, Cb]
            full_blocks = m4.all(dim=3).all(dim=1)
            if not bool((full_blocks == block_mask).all()):
                n_partial = int((block_mask & ~full_blocks).sum().item())
                print(
                    f"[carve] WARNING: tile mask not {blocking}-block-aligned — "
                    f"{n_partial}/{int(block_mask.sum())} kept blocks are partial "
                    f"(lossless, but block density {block_mask.float().mean():.3f} > "
                    f"tile density {tile_mask.float().mean():.3f}). "
                    f"Use prune_step_blocks for aligned pruning.", flush=True,
                )
            if not bool(block_mask.any(dim=1).all()):
                raise RuntimeError("carve: a block-row has zero kept blocks")

            # BCSR metadata. nonzero() returns row-major order = sorted by row, the
            # layout stk's kernels assume.
            idx = block_mask.nonzero()                        # [nnz, 2]
            row_indices = idx[:, 0].to(torch.int16).contiguous()
            column_indices = idx[:, 1].to(torch.int16).contiguous()
            row_nnz = block_mask.sum(dim=1).to(torch.int32)
            offsets = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device),
                 row_nnz.cumsum(0, dtype=torch.int32)]
            ).contiguous()

            blocks = (
                w.reshape(Rb, blocking, Cb, blocking).permute(0, 2, 1, 3)
            )                                                  # [Rb, Cb, blk, blk]
            data = blocks[idx[:, 0], idx[:, 1]].contiguous()   # [nnz, blk, blk]

            column_indices_t, offsets_t, block_offsets_t = _stk_transpose(
                (self.out_features, self.in_features),
                data, row_indices, column_indices, offsets,
            )

            self._init_mortar_storage(
                data.clone(),
                row_indices, column_indices, offsets,
                column_indices_t.contiguous(), offsets_t.contiguous(),
                block_offsets_t.contiguous(),
                blocking,
            )

        nnz = int(self.mortar_data.shape[0])
        # Density bookkeeping for log_stats / extra_repr (block density == tile
        # density when the mask was block-aligned).
        self.density = nnz / (Rb * Cb)
        return nnz

    def _mortar_effective_data(self) -> Tensor:
        """Carved weight actually used in forward: per-tensor symmetric ternary STE on
        mortar_data when values-ternary QAT is on (forward = scale·q, gradient =
        identity into the smooth mortar_data shadow)."""
        d = self.mortar_data
        if not getattr(self, "_values_ternary_mode", False):
            return d
        scale = d.detach().abs().mean().clamp(min=1e-8)
        d_norm = d / scale
        q = torch.sign(d_norm) * (d_norm.abs() > self._values_threshold).to(d.dtype)
        return d + (scale * q - d).detach()

    def _forward_mortar(self, x: Tensor) -> Tensor:
        """Carved forward: y = x @ W_sparse^T via the morph_stk::dds custom op.

        torch.compile-friendly: no Matrix wrapper object, no bare
        torch.autograd.Function — the custom op (with a register_fake meta) traces
        opaquely through Dynamo/AOTAutograd, so the surrounding elementwise
        (ternary STE, SwiGLU gate, bias) fuses around the sparse GEMM instead of
        graph-breaking. Numerics are byte-identical to the legacy stk
        Matrix+dds path: same backend kernels, same tensors/strides, and the
        explicit autocast cast below mirrors stk's custom_fwd (cast lhs+data to
        the autocast dtype, run the kernel autocast-off). Gradient checkpointing
        and the ternary STE stay correct for the same reason as before: the
        recomputed forward re-reads the CURRENT autograd-tracked data.

        M (= flattened batch·seq) is padded to a multiple of blocking when needed
        (decode shapes); training deploy shape 4·4096 = 16384 needs no pad.
        """
        from morph.sparse.stk.backend.custom_ops import stk_dds

        d = self._mortar_effective_data()
        lead_shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        M = x2.shape[0]
        pad = (-M) % self._mortar_blocking
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        if torch.is_autocast_enabled():
            # Mirror stk backend/autocast.py custom_fwd byte-for-byte: under
            # autocast, the legacy DDS cast lhs+data to the autocast dtype inside
            # the Function. Here the casts are explicit graph nodes (the .to()
            # backward re-casts grads to fp32 exactly like autograd's
            # validate_outputs did for the Function's bf16 grads).
            _dt = torch.get_autocast_gpu_dtype()
            x2 = x2.to(_dt)
            d = d.to(_dt)
        y = stk_dds(
            x2, d,
            self.mortar_offsets,
            self.mortar_row_indices,
            self.mortar_column_indices,
            self.mortar_offsets_t,
            self.mortar_column_indices_t,
            self.mortar_block_offsets_t,
            self.in_features, self.out_features, True,
        )
        if pad:
            y = y[:M]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead_shape, self.out_features)

    # State-dict keys saved by older revisions of this class (Block-ELL topology +
    # CMS topology-evolution bookkeeping, all removed 2026-06-11). Popped on load so
    # legacy checkpoints — including the validated MORTAR winner ckpts, which carry
    # these buffers — still load strict. New checkpoints simply do not contain them.
    _LEGACY_STATE_KEYS = (
        "col_indices", "activation_norm_acc", "error_norm_acc", "block_age",
        "_score_snapshot", "col_usage_count", "score_history", "crystallized_mask",
        "block_score_historical_ema", "last_swap_step", "swap_count",
        "gradient_coherence_ema", "prev_grad_direction",
        "_acc_steps", "_score_history_idx", "_swap_rate_history", "_topology_step_count",
    )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Transparently restructure dense→mortar when loading a carved checkpoint.

        nn.Module.load_state_dict recurses through THIS hook, so a freshly-built
        dense model loads a post-carve checkpoint without external surgery: detect
        mortar keys, install empty storage with the checkpoint's shapes, let
        super() fill it. Also drops removed-legacy keys (see _LEGACY_STATE_KEYS)
        and refuses 16×16 Block-ELL `values` checkpoints outright — that format
        was removed; only dense and MORTAR-carved checkpoints are loadable.
        """
        if prefix + "values" in state_dict:
            raise RuntimeError(
                f"{prefix}values: this checkpoint uses the removed 16×16 Block-ELL "
                f"compact() format. Block-ELL was removed — re-train or re-carve "
                f"with MORTAR (carve())."
            )
        for k in self._LEGACY_STATE_KEYS:
            state_dict.pop(prefix + k, None)

        data_key = prefix + "mortar_data"
        if data_key in state_dict and not self._mortar:
            ckpt = state_dict[data_key]
            blocking = int(ckpt.shape[-1])
            dev = self.block_score_ema.device

            def _like(key: str) -> Tensor:
                t = state_dict[prefix + key]
                return torch.empty_like(t, device=dev)

            # Restore persisted ternary-QAT flags BEFORE storage init so the buffer
            # is created with the right values even if super()'s copy is elided.
            mt = state_dict.get(prefix + "mortar_ternary")
            if mt is not None:
                self._values_ternary_mode = bool(float(mt[0]) > 0)
                self._values_threshold = float(mt[1])

            self._init_mortar_storage(
                torch.empty_like(ckpt, device=dev),
                _like("mortar_row_indices"),
                _like("mortar_column_indices"),
                _like("mortar_offsets"),
                _like("mortar_column_indices_t"),
                _like("mortar_offsets_t"),
                _like("mortar_block_offsets_t"),
                blocking,
            )
            Rb = self.out_features // blocking
            Cb = self.in_features // blocking
            self.density = int(ckpt.shape[0]) / (Rb * Cb)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for print(layer)."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tile_size={self.tile_size}, density={self.density:.2f}, "
            f"K={self.K}, bias={self.bias is not None}"
        )
