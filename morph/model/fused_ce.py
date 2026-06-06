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


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, labels: Tensor,
                ignore_index: int, chunk_size: int) -> Tensor:
        # x: [N, d], w: [V, d], labels: [N]
        N, d = x.shape
        V = w.shape[0]
        compute_dtype = x.dtype  # match eager autocast matmul precision

        valid = labels != ignore_index
        n_valid = int(valid.sum().item())
        n_valid_f = float(max(n_valid, 1))

        # Accumulators sized by the inputs, NOT by [N, V].
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(w, dtype=torch.float32)
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)

        # Cast the weight to compute dtype ONCE, in its natural [V, d] layout, and reuse
        # it for both matmuls. The logits matmul wants [d, V]; instead of materialising a
        # separate transposed-contiguous copy (a full [d, V] = [768, 49152] byte-move every
        # forward — costly for our hybrid weight-tied head), pass w_cast.t() and let cuBLAS
        # transpose via strides for free. Bit-identical: cast-then-transpose == transpose-
        # then-cast (cast is per-element, transpose only reindexes).
        w_cast = w.to(compute_dtype)               # [V, d] — used for logits (.t()) AND grad_x

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_c = x[start:end]                       # [c, d]
            lab_c = labels[start:end]                # [c]
            valid_c = valid[start:end].float().unsqueeze(-1)  # [c, 1]

            logits_c = (x_c @ w_cast.t()).float()    # [c, V] fp32 (freed each iter); cuBLAS transposes w

            # log-softmax cross-entropy, fp32 (numerically load-bearing).
            lse = torch.logsumexp(logits_c, dim=-1)  # [c]
            lab_safe = lab_c.clamp(min=0)            # avoid gather OOB on ignore rows
            tgt = logits_c.gather(-1, lab_safe.unsqueeze(-1)).squeeze(-1)  # [c]
            loss_c = (lse - tgt) * valid_c.squeeze(-1)
            loss_sum = loss_sum + loss_c.sum()

            # grad wrt logits (unnormalised by n_valid; scaled once at the end):
            #   softmax - onehot, zeroed on ignored rows. softmax in fp32.
            probs = torch.softmax(logits_c, dim=-1)  # [c, V] fp32
            probs.scatter_add_(
                -1, lab_safe.unsqueeze(-1),
                -torch.ones_like(lab_safe, dtype=probs.dtype).unsqueeze(-1),
            )
            probs = probs * valid_c                  # zero ignored rows
            del logits_c

            # Grad matmuls in compute_dtype (bf16 → tensor cores; matches the
            # eager autograd backward of the bf16 x@wᵀ). grad_w accumulates in
            # fp32 across chunks to avoid cancellation.
            probs_c = probs.to(compute_dtype)        # [c, V]
            grad_x[start:end] = probs_c @ w_cast     # [c, d]
            grad_w += (probs_c.t() @ x_c).float()    # [V, d] fp32 accumulate
            del probs, probs_c

        loss = loss_sum / n_valid_f
        grad_x.div_(n_valid_f)
        grad_w.div_(n_valid_f)

        ctx.save_for_backward(grad_x, grad_w)
        ctx.x_dtype = x.dtype
        ctx.w_dtype = w.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_x, grad_w = ctx.saved_tensors
        go = grad_output  # scalar
        gx = (grad_x * go).to(ctx.x_dtype)
        gw = (grad_w * go).to(ctx.w_dtype)
        return gx, gw, None, None, None


def fused_linear_cross_entropy(
    x: Tensor,
    w: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> Tensor:
    """Memory-efficient ``mean`` cross-entropy of a weight-tied linear head.

    See module docstring. ``x`` is ``[N, d]``, ``w`` is ``[V, d]``, ``labels``
    is ``[N]``. Returns a scalar; gradient flows to both ``x`` and ``w``.
    """
    return _FusedLinearCE.apply(x, w, labels, ignore_index, chunk_size)


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
