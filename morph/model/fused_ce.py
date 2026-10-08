"""Fused (chunked) linear cross-entropy for the weight-tied LM head.

The eager path ``F.cross_entropy(x @ W.T, labels)`` materialises a full
``[N, V]`` logits tensor (N = B·S tokens, V = vocab). At scale that tensor is
the dominant activation-memory cost — e.g. B8·S4096·V49152 fp32 ≈ 6.4 GiB, plus
an equal-size softmax buffer in the autograd graph for the backward. That is the
single biggest thing capping batch size / sequence length on a fixed VRAM budget.

This module computes the loss **and its gradients** in row-chunks, never holding
more than one ``[chunk, V]`` logits tile at a time. Peak extra memory becomes
``grad_x [N, d] + grad_w [V, d] + one [chunk, V] tile`` instead of ``[N, V]`` ×2.

Numerics
--------
Chunking is over the **row (token)** dimension. Each output logit
``logit[n, v] = Σ_d x[n, d]·W[v, d]`` is computed identically regardless of how
rows are grouped — there is *no* cross-chunk accumulation for any single logit.
The only reductions that change accumulation order are the per-token loss sum and
``grad_w = Σ_n gradlogit[n]ᵀ x[n]`` (both summed over tokens, fp32). So the result
matches ``F.cross_entropy`` to fp32 reduction error (~1e-6), verified in __main__.

Reduction is ``mean`` over non-ignored tokens, matching ``F.cross_entropy``
defaults with ``ignore_index``.

Contract
--------
    loss = fused_linear_cross_entropy(x, w, labels, ignore_index=-100,
                                      chunk_size=1024)
    x:      [N, d]   hidden states (any float dtype; matmul done in x.dtype,
                     softmax/loss reduction in fp32 for stability).
    w:      [V, d]   weight-tied LM-head matrix (built outside, e.g. cat of the
                     euclidean + lorentz-tangent embedding weights). Gradient
                     flows back to w → autograd handles cat/log-map to params.
    labels: [N]      int64 targets; ``ignore_index`` rows contribute 0.
    Returns: scalar mean loss. grad flows to BOTH x and w.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _pad_vocab(w_cast: Tensor) -> tuple[Tensor, int, int]:
    """GEMM-align the vocab dim. An odd V (e.g. StarCoder2 49152 + 17 olympiad
    specials = 49169) knocks every head matmul off the tensor-core fast path —
    measured 94.7 vs 202.9 TFLOPS on a 5090 for [16384,768]@[768,V]. Zero-pad
    the weight rows to the next multiple of 128 and run all three GEMMs at V′;
    pad logits are masked to -inf before lse/softmax so their probability is
    EXACTLY 0 → loss and both grads are bit-equivalent to the unpadded math
    (pad grad_w rows stay 0 and are sliced off). Costs one [V′-V, d] zero-fill
    + a [c, V′-V] mask per chunk — noise next to the 2× GEMM win."""
    V = w_cast.shape[0]
    if V % 64 == 0:
        return w_cast, V, V
    V_pad = ((V + 127) // 128) * 128
    return F.pad(w_cast, (0, 0, 0, V_pad - V)), V, V_pad


def _scale_inv(t: Tensor, inv: Tensor) -> None:
    """``t`` in place times the 0-dim fp32 DEVICE scalar ``inv``, bit-identical to the
    old ``t.div_(n)`` by a Python number. A Python ``inv`` (the CPU path) is the old
    divisor ``n`` itself and divides as before.

    ATen divides a CUDA tensor by a CPU scalar as ``a * (1/b)``, the reciprocal taken in
    fp32 (``div_true_kernel_cuda``'s CPU-scalar branch), and the product in fp32 before
    rounding to ``t``'s dtype. ``inv = torch.reciprocal(n)`` is the same IEEE fp32
    reciprocal on the device. An fp32 ``t`` then multiplies in fp32 directly; a bf16
    ``t`` must NOT multiply by a 0-dim fp32 tensor directly (the operand would be cast
    to bf16 first), so it goes through fp32 and rounds once on the copy back, which is
    the old kernel's rounding. Pinned by ``tests/test_fused_ce_sync_free.py``."""
    if not torch.is_tensor(inv):
        t.div_(inv)                 # CPU: the old host-scalar division, unchanged
    elif t.dtype == torch.float32:
        t.mul_(inv)
    else:
        t.copy_(t.float().mul_(inv))


def _acc_grad_w_(grad_w: Tensor, tile: Tensor, x_c: Tensor) -> None:
    """``grad_w += tile.t() @ x_c`` with the GEMM accumulating straight into the fp32
    ``grad_w`` (cuBLAS beta = 1, fp32 output), the Triton-kernel paths' accumulate. The eager
    paths round each chunk's product to bf16 and then add it in a separate fp32 pass over the
    [V', d] accumulator; this skips that pass (about 0.26 ms of a 0.81 ms chunk on a 5090 at
    [49280, 1024] x [1024, 1024]) and the bf16 rounding of the product (max error vs fp64
    0.50 -> 2.8e-4 on that chunk). Not bit-identical to the eager accumulate."""
    if tile.dtype in (torch.bfloat16, torch.float16) and grad_w.dtype == torch.float32:
        torch.addmm(grad_w, tile.t(), x_c, out_dtype=torch.float32, out=grad_w)
    else:
        grad_w.add_(tile.t() @ x_c)


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, labels: Tensor,
                ignore_index: int, chunk_size: int, mask_token_id: int = -1,
                weights: Tensor | None = None, softmax_kernel: bool = False,
                compact_rows: bool = False, row_cap: int = 0) -> Tensor:
        # x: [N, d], w: [V, d], labels: [N]
        N, d = x.shape
        x_full = x
        # A DETACHED head (the span decoder's `tul.mux_detach_head` read) never receives its
        # gradient, so its [V, d] accumulator and the chunk's third GEMM are not computed.
        # Bit-identical for the loss and grad_x: grad_w feeds neither.
        need_w = bool(ctx.needs_input_grad[1])
        compute_dtype = x.dtype  # match eager autocast matmul precision

        valid = labels != ignore_index
        if weights is None:
            n_valid_t = valid.sum().clamp_min(1).to(torch.float32)
            row_w = None
        else:
            # Per-row loss weights (TUL spec §5: the first token of a span is predicted
            # twice — from the span's last token and from the slot — and each term is
            # weighted 0.5 so it is not counted twice). The reduction becomes the
            # WEIGHTED mean Σ wᵢ·CEᵢ / Σ wᵢ, which is the plain mean when every weight
            # is 1. Folding it into this kernel keeps the loss to ONE vocab GEMM and one
            # [V, d] grad accumulator instead of one per label group.
            row_w = weights.to(torch.float32) * valid.to(torch.float32)
            n_valid_t = row_w.sum().clamp_min(1e-6)
        # The normaliser stays ON THE DEVICE on CUDA (2026-10-04): the old
        # `int(valid.sum().item())` was a host sync at the head of every CE call, which
        # drained the queue in the middle of the forward. `_scale_inv` reproduces the old
        # CUDA division by a Python number bit for bit (see its docstring). A CPU tensor
        # keeps the old host scalar: there is no queue to drain, and the CPU kernel divides
        # where the CUDA kernel multiplies by the reciprocal.
        inv_n = torch.reciprocal(n_valid_t) if x.is_cuda else float(n_valid_t.item())

        # `compact_rows` (model.ce_compact_rows): drop every row that carries no loss (an
        # ignored label, or a zero row weight) BEFORE the vocab GEMMs. Their loss and their
        # gradient are exactly zero on the full path too, so only the summation order of
        # the loss and of grad_w changes (the chunks hold different rows). The row count is
        # needed on the host to size the loop: ONE sync per call, placed in front of the
        # GEMM-bound chunk loop where the GPU has the most queued work to drain.
        keep_idx = None
        if compact_rows:
            keep = valid if row_w is None else row_w != 0
            keep_idx = keep.nonzero().squeeze(1)
            x = x.index_select(0, keep_idx)
            labels = labels.index_select(0, keep_idx)
            valid = valid.index_select(0, keep_idx)
            if row_w is not None:
                row_w = row_w.index_select(0, keep_idx)
            N = x.shape[0]

        # `row_cap` (model.spandec_ce_row_cap): the same row drop at a FIXED row count, for a
        # CUDA-graph-captured step (no host sync, one shape for every batch). The caller
        # passes a cap it has PROVEN no batch exceeds (`MORPHTransformer._spandec_row_cap`).
        # The rows that carry loss are packed in order to the front of a [row_cap] buffer by
        # a prefix sum; the buffer's tail reads row 0 under a zero weight (zero loss, zero
        # gradient). A batch over the cap would lose rows silently, so it makes the loss NaN
        # on the device instead (`over_cap` below): loud, and still no sync.
        cap_back = over_cap = None
        if 0 < row_cap < N and not compact_rows:
            keep = valid if row_w is None else row_w != 0
            pos = keep.cumsum(0) - 1
            land = keep & (pos < row_cap)
            over_cap = keep.sum() > row_cap
            ar = torch.arange(N, device=x.device)
            # every index unique (a row that does not land goes to its own slot past the
            # buffer), so the scatter is deterministic
            src = torch.full((row_cap + N,), N, device=x.device, dtype=torch.long)
            src.scatter_(0, torch.where(land, pos, ar + row_cap), ar)
            src = src[:row_cap]
            used = src < N
            src = torch.where(used, src, torch.zeros_like(src))
            x = x.index_select(0, src)
            labels = torch.where(used, labels.index_select(0, src),
                                 torch.full_like(src, ignore_index))
            valid = labels != ignore_index
            if row_w is not None:
                row_w = torch.where(used, row_w.index_select(0, src), torch.zeros_like(src,
                                    dtype=row_w.dtype))
            # grad_x back to the full rows by a GATHER (deterministic; a dropped row reads
            # the appended zero row)
            cap_back = torch.where(land, pos, torch.full_like(pos, row_cap))
            N = row_cap

        # Cast the weight to compute dtype ONCE, in its natural [V, d] layout, and reuse
        # it for both matmuls. The logits matmul wants [d, V]; instead of materialising a
        # separate transposed-contiguous copy (a full [d, V] = [768, 49152] byte-move every
        # forward — costly for our hybrid weight-tied head), pass w_cast.t() and let cuBLAS
        # transpose via strides for free. Bit-identical: cast-then-transpose == transpose-
        # then-cast (cast is per-element, transpose only reindexes).
        w_cast, V, V_pad = _pad_vocab(w.to(compute_dtype))  # [V′, d]
        neg_inf = torch.finfo(torch.float32).min
        # The GEMM operands, cast ONCE (2026-10-04). Under CUDA autocast every chunk's three
        # matmuls cast their fp32 operands to the autocast dtype again: the [V′, d] weight
        # twice per chunk, 20 chunks a step on the slot-loop arm. Casting here is the same
        # round-to-nearest cast, so the GEMMs see the same inputs and the result is
        # bit-identical; without autocast both are `compute_dtype` and nothing changes.
        gemm_dtype = (torch.get_autocast_dtype("cuda")
                      if x.is_cuda and torch.is_autocast_enabled("cuda") else compute_dtype)
        w_g = w_cast.to(gemm_dtype)
        x_g = x.to(gemm_dtype)
        # `softmax_kernel` (model.ce_softmax_kernel): the chunk's elementwise body runs as
        # ONE Triton kernel over the low-precision logits tile (morph/kernels/triton/
        # ce_softmax_grad.py). Not bit-identical to the eager body (a few fp32 ulps before
        # the gradient's bf16 rounding); only when the GEMMs run in bf16/fp16 on CUDA.
        use_kernel = bool(softmax_kernel) and x.is_cuda and gemm_dtype in (
            torch.bfloat16, torch.float16)
        if use_kernel:
            from morph.kernels.triton.ce_softmax_grad import ce_softmax_grad_

        # Accumulators sized by the inputs, NOT by [N, V].
        grad_x = torch.empty_like(x)
        grad_w = (torch.zeros((V_pad, d), device=w.device, dtype=torch.float32)
                  if need_w else None)
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_c = x_g[start:end]                     # [c, d] in the GEMM dtype
            lab_c = labels[start:end]                # [c]
            valid_c = valid[start:end].float().unsqueeze(-1)  # [c, 1]

            if use_kernel:
                w_c = valid_c if row_w is None else row_w[start:end].unsqueeze(-1)
                probs_c = x_c @ w_g.t()                  # [c, V′] bf16 logits
                loss_c = ce_softmax_grad_(probs_c, lab_c.clamp(min=0), w_c.squeeze(-1),
                                          V, mask_token_id)   # tile := weighted grad
                loss_sum = loss_sum + loss_c.sum()
                grad_x[start:end] = probs_c @ w_g
                if need_w:
                    _acc_grad_w_(grad_w, probs_c, x_c)
                del probs_c
                continue

            logits_c = (x_c @ w_g.t()).float()       # [c, V′] fp32 (freed each iter)
            if V_pad != V:
                logits_c[:, V:] = neg_inf            # pad cols → prob exactly 0
            if mask_token_id >= 0:
                # TUL (.agents/specs/tul-spec.md §3.1, invariant 4): the slot id is a structural
                # token the rule inserts — the LM head must never predict it. Same
                # mechanism as the pad columns above: -inf ⇒ softmax probability EXACTLY
                # 0 ⇒ that vocab row's grad_w stays 0 and it is excluded from the
                # partition function. The id is never a LABEL (asserted at data prep),
                # so the target gather below is unaffected.
                logits_c[:, mask_token_id] = neg_inf

            # log-softmax cross-entropy, fp32 (numerically load-bearing).
            lse = torch.logsumexp(logits_c, dim=-1)  # [c]
            lab_safe = lab_c.clamp(min=0)            # avoid gather OOB on ignore rows
            tgt = logits_c.gather(-1, lab_safe.unsqueeze(-1)).squeeze(-1)  # [c]
            w_c = valid_c if row_w is None else row_w[start:end].unsqueeze(-1)
            loss_c = (lse - tgt) * w_c.squeeze(-1)
            loss_sum = loss_sum + loss_c.sum()

            # grad wrt logits (unnormalised by n_valid; scaled once at the end):
            #   softmax - onehot, zeroed on ignored rows. softmax in fp32.
            probs = torch.softmax(logits_c, dim=-1)  # [c, V′] fp32; pad cols exactly 0
            probs.scatter_add_(
                -1, lab_safe.unsqueeze(-1),
                -torch.ones_like(lab_safe, dtype=probs.dtype).unsqueeze(-1),
            )
            probs.mul_(w_c)                          # zero ignored rows; scale by wᵢ
            del logits_c

            # Grad matmuls in compute_dtype (bf16 → tensor cores; matches the
            # eager autograd backward of the bf16 x@wᵀ). grad_w accumulates in
            # fp32 across chunks to avoid cancellation. Pad probs are 0 → pad
            # grad_w rows stay 0, grad_x unaffected (0 · w_pad_row = 0).
            probs_c = probs.to(compute_dtype)        # [c, V′]
            grad_x[start:end] = probs_c @ w_g        # [c, d]
            # `add_` of the bf16 product promotes it to fp32 inside the add (exact), the
            # same value as `+= (...).float()` without the [V′, d] fp32 temporary.
            if need_w:
                grad_w.add_(probs_c.t() @ x_c)       # [V′, d] fp32 accumulate
            del probs, probs_c

        loss = loss_sum * inv_n if torch.is_tensor(inv_n) else loss_sum / inv_n
        _scale_inv(grad_x, inv_n)
        if keep_idx is not None:
            grad_x = torch.zeros_like(x_full).index_copy_(0, keep_idx, grad_x)
        if cap_back is not None:
            grad_x = torch.cat([grad_x, grad_x.new_zeros(1, d)]).index_select(0, cap_back)
            loss = torch.where(over_cap, torch.full_like(loss, float("nan")), loss)
        if need_w:
            grad_w = grad_w[:V] if V_pad != V else grad_w
            _scale_inv(grad_w, inv_n)

        ctx.save_for_backward(grad_x, grad_w)
        ctx.x_dtype = x.dtype
        ctx.w_dtype = w.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_x, grad_w = ctx.saved_tensors
        go = grad_output  # scalar
        gx = (grad_x * go).to(ctx.x_dtype)
        gw = None if grad_w is None else (grad_w * go).to(ctx.w_dtype)
        return gx, gw, None, None, None, None, None, None, None, None


def fused_linear_cross_entropy(
    x: Tensor,
    w: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    chunk_size: int = 1024,
    mask_token_id: int = -1,
    weights: Tensor | None = None,
    softmax_kernel: bool = False,
    compact_rows: bool = False,
    row_cap: int = 0,
) -> Tensor:
    """Memory-efficient ``mean`` cross-entropy of a weight-tied linear head.

    See module docstring. ``x`` is ``[N, d]``, ``w`` is ``[V, d]``, ``labels``
    is ``[N]``. Returns a scalar; gradient flows to both ``x`` and ``w``.

    ``mask_token_id`` (default −1 = off, bit-identical to before): force that vocab
    row's logit to −inf in every chunk, so it has probability exactly 0 and receives
    exactly zero gradient. Used for TUL's structural ``slot_id`` (spec §3.1).

    ``weights`` (default None = off, bit-identical to before): ``[N]`` per-row loss
    weights; the reduction becomes ``Σ wᵢ·CEᵢ / Σ wᵢ``. Used for TUL's half-weight
    double label (spec §5).

    ``softmax_kernel`` (default False = the eager chunk body, unchanged): the per-chunk
    softmax gradient in one Triton pass over the bf16 logits tile (CUDA bf16/fp16 only;
    a few fp32 ulps from the eager body, see ``morph/kernels/triton/ce_softmax_grad.py``).

    ``compact_rows`` (default False = every row): the vocab GEMMs run on the rows that
    carry loss only; one host sync per call; not bit-identical (summation order).

    ``row_cap`` (default 0 = off): the same row drop with no host sync, into a FIXED
    ``[row_cap]`` buffer (CUDA-graph safe). The caller must prove no batch carries more than
    ``row_cap`` rows with loss; a batch that does returns a NaN loss. Not bit-identical.

    A ``w`` that does not require grad gets no ``grad_w`` (the third GEMM is skipped).
    """
    return _FusedLinearCE.apply(x, w, labels, ignore_index, chunk_size, mask_token_id,
                                weights, softmax_kernel, compact_rows, int(row_cap))


# ── Spectral decoupling: an L2 penalty on the coda's TOKEN logits ───────────
# (Pezeshki et al., "Gradient Starvation: A Learning Proclivity in Neural Networks",
# arXiv 2011.09468. `tul.coda_logit_l2`, docs/tul-... none — see morph/model/tul.py.)
#
# A SEPARATE autograd.Function/entry point from `_FusedLinearCE` above, not a branch
# inside it: `fused_linear_cross_entropy` is called from a dozen sites across
# `transformer.py`, several of them hot; this keeps every one of them, and the default
# (`tul.coda_logit_l2 == 0.0`) path of `_tul_group_losses`, byte-for-byte untouched —
# bit-identical by construction, not by a runtime `if` inside the shared kernel.
#
# WHAT IT ADDS: `lambda/2 * mean_i row_w_i * ||z_i||^2 / n_valid_f`, where `z_i` is the
# position's RAW (pre-softmax, pre-mask) logit vector over the real vocab — i.e. the
# same weighted-mean convention `_FusedLinearCE` already uses for the CE term (so
# "labelled token positions" means exactly the positions the CE reduction already
# counts, TUL's half-weight double label included). `z_i` is read straight out of the
# chunk tile BEFORE the pad-column and `mask_token_id` overrides — the pad columns are
# exactly 0 there (the padded weight ROWS are zero, so the matmul already gives 0, no
# masking needed for them to contribute 0 to the sum), and the `mask_token_id` COLUMN
# (TUL's structural slot id) is included: it is a real learned output of the tied head
# and "the coda's token logits" is the full vocab vector, not the vocab the CE actually
# scores. That is a deliberate reading, not an oversight — the reference in
# tests/test_coda_logit_l2.py encodes the same choice, so the exactness check pins it.
#
# Exact gradient (no [N, V] materialised beyond the one-chunk tile already needed for
# the CE): with S = lambda / n_valid_f,
#   dLoss_l2/dz_i = S * row_w_i * z_i            (V-dim; 0 at every pad column already)
#   dLoss_l2/dh_i = (dLoss_l2/dz_i) @ W           (same contraction the CE grad uses)
#   dLoss_l2/dW   = sum_i row_w_i * z_i ⊗ h_i * S (same accumulation pattern as grad_w)
# so it is folded into the SAME per-chunk `grad_x` / `grad_w` accumulation as the CE
# term, at the cost of one extra `[chunk, V']` elementwise buffer per chunk.

class _FusedLinearCEWithLogitL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, labels: Tensor, ignore_index: int,
                chunk_size: int, mask_token_id: int, weights: Tensor | None,
                logit_l2_lambda: float) -> tuple[Tensor, Tensor]:
        N, d = x.shape
        compute_dtype = x.dtype

        valid = labels != ignore_index
        if weights is None:
            n_valid = int(valid.sum().item())
            n_valid_f = float(max(n_valid, 1))
            row_w = None
        else:
            row_w = weights.to(torch.float32) * valid.to(torch.float32)
            n_valid_f = float(max(float(row_w.sum().item()), 1e-6))

        w_cast, V, V_pad = _pad_vocab(w.to(compute_dtype))
        neg_inf = torch.finfo(torch.float32).min

        grad_x = torch.empty_like(x)
        grad_w = torch.zeros((V_pad, d), device=w.device, dtype=torch.float32)
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        logit_sq_sum = torch.zeros((), device=x.device, dtype=torch.float32)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_c = x[start:end]
            lab_c = labels[start:end]
            valid_c = valid[start:end].float().unsqueeze(-1)
            w_c = valid_c if row_w is None else row_w[start:end].unsqueeze(-1)

            # RAW logits — kept apart from the masked copy used for the softmax/CE so
            # the penalty reads the true (unmasked) coda output. Pad columns are 0 by
            # construction (zero-padded weight rows); the mask_token_id column is real.
            z_c = (x_c @ w_cast.t()).float()             # [c, V'] fp32
            logits_c = z_c.clone()
            if V_pad != V:
                logits_c[:, V:] = neg_inf
            if mask_token_id >= 0:
                logits_c[:, mask_token_id] = neg_inf

            lse = torch.logsumexp(logits_c, dim=-1)
            lab_safe = lab_c.clamp(min=0)
            tgt = logits_c.gather(-1, lab_safe.unsqueeze(-1)).squeeze(-1)
            loss_c = (lse - tgt) * w_c.squeeze(-1)
            loss_sum = loss_sum + loss_c.sum()

            sq_c = z_c.pow(2).sum(dim=-1)                 # [c], real V only (pad = 0)
            logit_sq_sum = logit_sq_sum + (sq_c * w_c.squeeze(-1)).sum()

            probs = torch.softmax(logits_c, dim=-1)
            probs.scatter_add_(
                -1, lab_safe.unsqueeze(-1),
                -torch.ones_like(lab_safe, dtype=probs.dtype).unsqueeze(-1),
            )
            probs = probs * w_c
            del logits_c

            # raw (pre n_valid_f-division) dLoss/dz: CE term + penalty term, summed
            # BEFORE the shared matmul into grad_x / grad_w (same accumulation the CE
            # alone uses, extended by one elementwise term).
            grad_z = probs + logit_l2_lambda * w_c * z_c  # [c, V']
            del probs, z_c

            grad_z_c = grad_z.to(compute_dtype)
            grad_x[start:end] = grad_z_c @ w_cast
            grad_w += (grad_z_c.t() @ x_c).float()
            del grad_z, grad_z_c

        loss_sum = loss_sum + (logit_l2_lambda / 2.0) * logit_sq_sum
        loss = loss_sum / n_valid_f
        grad_x.div_(n_valid_f)
        grad_w = grad_w[:V] if V_pad != V else grad_w
        grad_w.div_(n_valid_f)
        logit_sq_mean = (logit_sq_sum / n_valid_f).detach()

        ctx.save_for_backward(grad_x, grad_w)
        ctx.x_dtype = x.dtype
        ctx.w_dtype = w.dtype
        ctx.mark_non_differentiable(logit_sq_mean)
        return loss, logit_sq_mean

    @staticmethod
    def backward(ctx, grad_output: Tensor, _grad_logit_sq_mean=None):
        grad_x, grad_w = ctx.saved_tensors
        go = grad_output
        gx = (grad_x * go).to(ctx.x_dtype)
        gw = (grad_w * go).to(ctx.w_dtype)
        return gx, gw, None, None, None, None, None, None


def fused_linear_cross_entropy_logit_l2(
    x: Tensor,
    w: Tensor,
    labels: Tensor,
    logit_l2_lambda: float,
    ignore_index: int = -100,
    chunk_size: int = 1024,
    mask_token_id: int = -1,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """CE + the spectral-decoupling logit penalty (``tul.coda_logit_l2``), fused.

    Returns ``(loss, logit_sq_mean)``: ``loss`` is the single differentiable scalar
    ``CE + lambda/2 * mean(row_w * ||z||^2) / n_valid_f`` (gradient exact, see module
    comment above); ``logit_sq_mean`` is a DETACHED fp32 scalar, the same weighted mean
    of ``||z||^2`` WITHOUT the ``lambda`` scale — the raw ``tul/coda_logit_sq`` stat.

    Never materialises ``[N, V]``: same chunked-row contract as
    :func:`fused_linear_cross_entropy`, one extra ``[chunk, V']`` fp32 tile held per
    chunk iteration (freed before the next).
    """
    return _FusedLinearCEWithLogitL2.apply(x, w, labels, ignore_index, chunk_size,
                                           mask_token_id, weights, logit_l2_lambda)


# ── Per-row log-probability of the label (tul.gram_objective="iw", 2026-09-23) ──
#
# The multi-sample bound (morph/model/tul_gram.py::iw_span_bound) needs log p(label) at
# EVERY position of EVERY rollout, as a tensor with a gradient, because the bound is a
# log-mean-exp over rollouts of per-span SUMS of these values: its gradient into row i of
# rollout k is a per-row weight (the rollout's posterior credit for that span) that is
# only known after every row has been scored. `_FusedLinearCE` cannot serve it: it folds
# the backward into its forward (it precomputes grad_x / grad_w for a SCALAR loss whose
# per-row weights are known up front) and returns one number.
#
# So this is the other contract: return ``[N]`` fp32 ``log softmax(x W^T)[label]`` (0 on
# ignored rows) and save only the per-row log-partition ``lse`` ``[N]`` fp32; the backward
# recomputes each ``[chunk, V']`` logits tile, forms ``g_i * (onehot - softmax)`` from the
# saved ``lse`` and runs the two grad GEMMs. Peak extra memory is the same as the CE
# kernel's: ``grad_x [N, d] + grad_w [V', d] fp32 + one [chunk, V'] tile``. Cost: one
# logits GEMM in the forward and three in the backward (recompute + grad_x + grad_w),
# against three in `_FusedLinearCE`'s forward.
#
# Precision. The forward runs whatever matmul precision the caller's autocast gives it
# (the `_FusedLinearCE` rule: "match eager autocast matmul precision"), and the backward
# RE-ENTERS the same autocast state before it recomputes the tile, so the recomputed
# logits are the forward's logits and ``exp(logit - lse)`` is the forward's softmax. The
# autograd engine does not carry the autocast state into a custom Function's backward on
# its own (torch.amp.custom_bwd exists for that, but it is bound to one device type).
#
# `softmax_kernel` (model.ce_softmax_kernel, 2026-10-08): both elementwise bodies as one Triton
# pass each over the bf16 tile (`morph/kernels/triton/ce_softmax_grad.py::ce_row_lse` and
# `ce_logprob_grad_`) instead of ~6 eager passes over an fp32 copy. The GEMM operands are cast
# ONCE to the GEMM dtype (the autocast dtype, as `_FusedLinearCE` does), so the GEMMs see the
# same operands as the eager path and the tile is the same bf16 tile; only the fp32 arithmetic
# on it differs (online log-sum-exp, exp2), a few fp32 ulps before the gradient's bf16 rounding.
# This is the pointer / DITTO token loss of `_tul_group_losses`: there it is the whole LM head.
def _gemm_operands(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor, int, int]:
    """``(x_g, w_g [V', d], V, V')`` cast once to the dtype the eager GEMM runs in."""
    gemm_dtype = (torch.get_autocast_dtype("cuda")
                  if x.is_cuda and torch.is_autocast_enabled("cuda") else x.dtype)
    w_cast, V, V_pad = _pad_vocab(w.to(x.dtype))
    return x.to(gemm_dtype), w_cast.to(gemm_dtype), V, V_pad


def _kernel_ok(x: Tensor, softmax_kernel: bool) -> bool:
    gemm_dtype = (torch.get_autocast_dtype("cuda")
                  if x.is_cuda and torch.is_autocast_enabled("cuda") else x.dtype)
    return bool(softmax_kernel) and x.is_cuda and gemm_dtype in (torch.bfloat16, torch.float16)


class _FusedLinearLabelLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, labels: Tensor, ignore_index: int,
                chunk_size: int, mask_token_id: int, softmax_kernel: bool = False) -> Tensor:
        N, _d = x.shape
        dev = x.device.type
        ctx.ac_enabled = torch.is_autocast_enabled(dev)
        ctx.ac_dtype = torch.get_autocast_dtype(dev) if ctx.ac_enabled else None
        ctx.use_kernel = _kernel_ok(x, softmax_kernel)
        ctx.ignore_index, ctx.chunk_size = ignore_index, chunk_size
        ctx.mask_token_id = mask_token_id
        if ctx.use_kernel:
            from morph.kernels.triton.ce_softmax_grad import ce_row_lse
            x_g, w_g, V, _V_pad = _gemm_operands(x, w)
            valid = labels != ignore_index
            lab_safe = labels.clamp(min=0)
            out = torch.empty(N, device=x.device, dtype=torch.float32)
            lse = torch.empty(N, device=x.device, dtype=torch.float32)
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                out[s:e], lse[s:e] = ce_row_lse(x_g[s:e] @ w_g.t(), lab_safe[s:e],
                                                valid[s:e], V, mask_token_id)
            ctx.save_for_backward(x, w, labels, lse)
            return out
        w_cast, V, V_pad = _pad_vocab(w.to(x.dtype))
        neg_inf = torch.finfo(torch.float32).min
        valid = labels != ignore_index
        lab_safe = labels.clamp(min=0)
        out = torch.empty(N, device=x.device, dtype=torch.float32)
        lse = torch.empty(N, device=x.device, dtype=torch.float32)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            logits = (x[s:e] @ w_cast.t()).float()
            if V_pad != V:
                logits[:, V:] = neg_inf
            if mask_token_id >= 0:
                logits[:, mask_token_id] = neg_inf
            lse_c = torch.logsumexp(logits, dim=-1)
            tgt = logits.gather(-1, lab_safe[s:e].unsqueeze(-1)).squeeze(-1)
            out[s:e] = torch.where(valid[s:e], tgt - lse_c, torch.zeros_like(lse_c))
            lse[s:e] = lse_c
            del logits
        ctx.save_for_backward(x, w, labels, lse)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, w, labels, lse = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_x or need_w):
            return None, None, None, None, None, None, None
        dev = x.device.type
        chunk = ctx.chunk_size
        N, d = x.shape
        if ctx.use_kernel:
            from morph.kernels.triton.ce_softmax_grad import ce_logprob_grad_
            with torch.autocast(device_type=dev, dtype=ctx.ac_dtype or torch.bfloat16,
                                enabled=ctx.ac_enabled):
                x_g, w_g, V, V_pad = _gemm_operands(x, w)
            g = torch.where(labels != ctx.ignore_index, grad_out.float(),
                            torch.zeros_like(lse))
            lab_safe = labels.clamp(min=0)
            grad_x = torch.empty_like(x) if need_x else None
            grad_w = (torch.zeros((V_pad, d), device=w.device, dtype=torch.float32)
                      if need_w else None)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                x_c = x_g[s:e]
                dl = ce_logprob_grad_(x_c @ w_g.t(), lab_safe[s:e], g[s:e], lse[s:e], V,
                                      ctx.mask_token_id)          # tile := g (onehot - p)
                if need_x:
                    grad_x[s:e] = dl @ w_g
                if need_w:
                    _acc_grad_w_(grad_w, dl, x_c)
                del dl
            if need_w:
                grad_w = (grad_w[:V] if V_pad != V else grad_w).to(w.dtype)
            return grad_x, grad_w, None, None, None, None, None
        with torch.autocast(device_type=dev, dtype=ctx.ac_dtype or torch.bfloat16,
                            enabled=ctx.ac_enabled):
            w_cast, V, V_pad = _pad_vocab(w.to(x.dtype))
            neg_inf = torch.finfo(torch.float32).min
            valid = labels != ctx.ignore_index
            lab_safe = labels.clamp(min=0)
            g = torch.where(valid, grad_out.float(), torch.zeros_like(lse))     # [N]
            grad_x = torch.empty_like(x) if need_x else None
            grad_w = (torch.zeros((V_pad, d), device=w.device, dtype=torch.float32)
                      if need_w else None)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                x_c = x[s:e]
                logits = (x_c @ w_cast.t()).float()
                if V_pad != V:
                    logits[:, V:] = neg_inf
                if ctx.mask_token_id >= 0:
                    logits[:, ctx.mask_token_id] = neg_inf
                # d log p(y) / d logit = onehot(y) - softmax; times the row's upstream g.
                probs = torch.exp(logits - lse[s:e].unsqueeze(-1))
                del logits
                g_c = g[s:e].unsqueeze(-1)
                dl = probs.mul_(-g_c)
                dl.scatter_add_(-1, lab_safe[s:e].unsqueeze(-1), g_c)
                dl_c = dl.to(x.dtype)
                del probs, dl
                if need_x:
                    grad_x[s:e] = dl_c @ w_cast
                if need_w:
                    grad_w += (dl_c.t() @ x_c).float()
                del dl_c
        if need_w:
            grad_w = (grad_w[:V] if V_pad != V else grad_w).to(w.dtype)
        return grad_x, grad_w, None, None, None, None, None


def fused_linear_label_logprob(
    x: Tensor, w: Tensor, labels: Tensor, ignore_index: int = -100,
    chunk_size: int = 1024, mask_token_id: int = -1, softmax_kernel: bool = False,
) -> Tensor:
    """``[N]`` fp32 ``log softmax(x @ w.T)[label]`` per row, 0 on ``ignore_index`` rows, with
    an exact gradient to ``x`` and ``w`` for ANY upstream ``[N]`` gradient. Never
    materialises ``[N, V]`` (see the block comment above). ``mask_token_id`` removes that
    vocab row from the partition function, exactly as :func:`fused_linear_cross_entropy`
    does, so ``-out[i]`` equals that kernel's per-row CE.

    ``softmax_kernel`` (default False = the eager bodies, unchanged): the forward's
    log-sum-exp and the backward's softmax gradient as one Triton pass each over the bf16
    tile (CUDA bf16/fp16 GEMMs only; see the block comment above)."""
    return _FusedLinearLabelLogProb.apply(x, w, labels, ignore_index, chunk_size,
                                          mask_token_id, bool(softmax_kernel))


# ── Multi-hot cross-entropy (MCE) for Token-Superposition Training ──────────
class _FusedLinearMCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, labels: Tensor,
                ignore_index: int, chunk_size: int, mask_token_id: int = -1) -> Tensor:
        N, d = x.shape
        compute_dtype = x.dtype
        valid = labels != ignore_index
        k_per_row = valid.sum(dim=1)
        row_valid = k_per_row > 0
        n_valid_f = float(max(int(row_valid.sum().item()), 1))
        inv_k = torch.zeros(N, device=x.device, dtype=torch.float32)
        inv_k[row_valid] = 1.0 / k_per_row[row_valid].float()
        grad_x = torch.empty_like(x)
        w_cast, V, V_pad = _pad_vocab(w.to(compute_dtype))   # GEMM-aligned head (see _pad_vocab)
        neg_inf = torch.finfo(torch.float32).min
        grad_w = torch.zeros((V_pad, x.shape[1]), device=w.device, dtype=torch.float32)
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_c = x[start:end]
            lab_c = labels[start:end]
            valid_c = valid[start:end].to(torch.float32)
            invk_c = inv_k[start:end]
            rowv_c = row_valid[start:end].to(torch.float32)
            lab_safe = lab_c.clamp(min=0)
            logits_c = (x_c @ w_cast.t()).float()
            if V_pad != V:
                logits_c[:, V:] = neg_inf                    # pad cols → prob exactly 0
            if mask_token_id >= 0:
                # Same contract as the single-hot path above: the TUL slot id is never a
                # prediction. It matters here because a caller may mix the two kernels to
                # express one soft target (morph/model/tul_egrad.py) and the two halves
                # must share a partition function.
                logits_c[:, mask_token_id] = neg_inf
            lse = torch.logsumexp(logits_c, dim=-1)
            tgt = logits_c.gather(-1, lab_safe) * valid_c
            sum_tgt = tgt.sum(dim=1)
            loss_c = (lse - invk_c * sum_tgt) * rowv_c
            loss_sum = loss_sum + loss_c.sum()
            probs = torch.softmax(logits_c, dim=-1)
            sub = (invk_c.unsqueeze(1) * valid_c).to(probs.dtype)
            probs.scatter_add_(-1, lab_safe, -sub)
            probs = probs * rowv_c.unsqueeze(-1)
            del logits_c
            probs_c = probs.to(compute_dtype)
            grad_x[start:end] = probs_c @ w_cast
            grad_w += (probs_c.t() @ x_c).float()
            del probs, probs_c
        loss = loss_sum / n_valid_f
        grad_x.div_(n_valid_f)
        grad_w = grad_w[:V] if V_pad != V else grad_w
        grad_w.div_(n_valid_f)
        ctx.save_for_backward(grad_x, grad_w)
        ctx.x_dtype = x.dtype
        ctx.w_dtype = w.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_x, grad_w = ctx.saved_tensors
        go = grad_output
        return ((grad_x * go).to(ctx.x_dtype), (grad_w * go).to(ctx.w_dtype),
                None, None, None, None)


def fused_linear_cross_entropy_mce(
    x: Tensor, w: Tensor, labels: Tensor, ignore_index: int = -100, chunk_size: int = 1024,
    mask_token_id: int = -1,
) -> Tensor:
    """Memory-efficient multi-hot cross-entropy. labels: [N, K]. Reduces to single-hot when K=1.

    ``mask_token_id`` (default −1 = off, bit-identical to before) forces that vocab row's
    logit to −inf, exactly as :func:`fused_linear_cross_entropy` does."""
    return _FusedLinearMCE.apply(x, w, labels, ignore_index, chunk_size, mask_token_id)


def multi_hot_cross_entropy_reference(
    logits: Tensor, labels: Tensor, ignore_index: int = -100,
) -> Tensor:
    """Eager full-logits MCE reference. logits: [N, V], labels: [N, K]."""
    f = logits.float()
    valid = (labels != ignore_index)
    k = valid.sum(dim=1).clamp(min=1).float()
    row_valid = (valid.sum(dim=1) > 0).float()
    lse = torch.logsumexp(f, dim=-1)
    tgt = f.gather(-1, labels.clamp(min=0)) * valid.float()
    per_row = (lse - tgt.sum(dim=1) / k) * row_valid
    return per_row.sum() / row_valid.sum().clamp(min=1.0)


# ── Pure-reference + self-test ──────────────────────────────────────────────

def _reference(x: Tensor, w: Tensor, labels: Tensor, ignore_index: int = -100):
    """Eager F.cross_entropy over the full materialised logits."""
    logits = (x @ w.t()).float()
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def _run_case(dtype, device, N=4096, d=768, V=49152, ignore_frac=0.1, chunk=1024):
    torch.manual_seed(0)
    x = torch.randn(N, d, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(V, d, device=device, dtype=dtype, requires_grad=True) * (d ** -0.5)
    w = w.detach().requires_grad_(True)
    labels = torch.randint(0, V, (N,), device=device)
    # inject some ignore_index
    mask = torch.rand(N, device=device) < ignore_frac
    labels = labels.masked_fill(mask, -100)

    # ── reference ──
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    loss_ref = _reference(xr, wr, labels)
    loss_ref.backward()

    # ── fused ──
    xf = x.detach().clone().requires_grad_(True)
    wf = w.detach().clone().requires_grad_(True)
    loss_f = fused_linear_cross_entropy(xf, wf, labels, chunk_size=chunk)
    loss_f.backward()

    def rel(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)

    loss_rel = abs(loss_f.item() - loss_ref.item()) / (abs(loss_ref.item()) + 1e-12)
    gx_rel = rel(xf.grad, xr.grad)
    gw_rel = rel(wf.grad, wr.grad)
    gx_cos = F.cosine_similarity(xf.grad.flatten().float(),
                                 xr.grad.flatten().float(), dim=0).item()
    gw_cos = F.cosine_similarity(wf.grad.flatten().float(),
                                 wr.grad.flatten().float(), dim=0).item()
    print(f"  [{str(dtype):>14}] loss_ref={loss_ref.item():.6f} fused={loss_f.item():.6f}  "
          f"loss_rel={loss_rel:.2e}  gx_rel={gx_rel:.2e} (cos {gx_cos:.6f})  "
          f"gw_rel={gw_rel:.2e} (cos {gw_cos:.6f})")
    return loss_rel, gx_cos, gw_cos


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"fused_linear_cross_entropy self-test on {dev}")

    print("\nfp32 (exactness gate — chunking must be near-bit-exact):")
    lr32, gxc32, gwc32 = _run_case(torch.float32, dev)
    assert lr32 < 1e-5, f"fp32 loss mismatch {lr32}"
    assert gxc32 > 0.99999 and gwc32 > 0.99999, f"fp32 grad cos {gxc32},{gwc32}"

    print("\nbf16 (autocast-precision gate):")
    lrb, gxcb, gwcb = _run_case(torch.bfloat16, dev)
    assert lrb < 1e-2, f"bf16 loss mismatch {lrb}"
    assert gxcb > 0.99 and gwcb > 0.99, f"bf16 grad cos {gxcb},{gwcb}"

    print("\nODD vocab V=49169 (exercises the _pad_vocab GEMM-align path):")
    lr32o, gxc32o, gwc32o = _run_case(torch.float32, dev, V=49169)
    assert lr32o < 1e-5, f"fp32 odd-V loss mismatch {lr32o}"
    assert gxc32o > 0.99999 and gwc32o > 0.99999, f"fp32 odd-V grad cos {gxc32o},{gwc32o}"
    lrbo, gxcbo, gwcbo = _run_case(torch.bfloat16, dev, V=49169)
    assert lrbo < 1e-2, f"bf16 odd-V loss mismatch {lrbo}"
    assert gxcbo > 0.99 and gwcbo > 0.99, f"bf16 odd-V grad cos {gxcbo},{gwcbo}"

    print("\nMCE odd-V parity (fused padded vs full-logits reference, K=6 bags):")
    torch.manual_seed(1)
    Nm, dm, Vm, K = 4096, 768, 49169, 6
    for dt in (torch.float32, torch.bfloat16):
        xm = torch.randn(Nm, dm, device=dev, dtype=dt, requires_grad=True)
        wm = (torch.randn(Vm, dm, device=dev, dtype=dt) * dm ** -0.5).requires_grad_(True)
        labm = torch.randint(0, Vm, (Nm, K), device=dev)
        labm[torch.rand(Nm, K, device=dev) < 0.05] = -100
        xr = xm.detach().clone().requires_grad_(True)
        wr = wm.detach().clone().requires_grad_(True)
        ref = multi_hot_cross_entropy_reference((xr @ wr.t()).float(), labm)
        ref.backward()
        lf = fused_linear_cross_entropy_mce(xm, wm, labm, chunk_size=1024)
        lf.backward()
        lrel = abs(lf.item() - ref.item()) / (abs(ref.item()) + 1e-12)
        gxc = F.cosine_similarity(xm.grad.flatten().float(), xr.grad.flatten().float(), dim=0).item()
        gwc = F.cosine_similarity(wm.grad.flatten().float(), wr.grad.flatten().float(), dim=0).item()
        print(f"  [{str(dt):>14}] ref={ref.item():.6f} fused={lf.item():.6f} rel={lrel:.2e} "
              f"gx_cos={gxc:.6f} gw_cos={gwc:.6f}")
        tol = 1e-5 if dt == torch.float32 else 1e-2
        assert lrel < tol and gxc > 0.99 and gwc > 0.99, "MCE odd-V parity FAILED"

    print("\n── peak memory: fused vs eager (bf16, N=B8·S4096) ──")
    if dev.type == "cuda":
        N, d, V = 8 * 4096, 768, 49152
        labels = torch.randint(0, V, (N,), device=dev)
        for tag, fn in [("eager", "eager"), ("fused", "fused")]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            x = torch.randn(N, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
            w = (torch.randn(V, d, device=dev, dtype=torch.bfloat16) * d ** -0.5).requires_grad_(True)
            if fn == "eager":
                loss = _reference(x, w, labels)
            else:
                loss = fused_linear_cross_entropy(x, w, labels, chunk_size=1024)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(dev) / 2**20
            print(f"  {tag:6s} peak={peak:.0f} MiB  loss={loss.item():.4f}")
            del x, w, loss

    print("\nALL FUSED-CE CHECKS PASSED")
