"""Guards for the Triton-kernel Dynamo fences + the window force_eager fix (2026-07-02).

Two related enablement changes let a MORPH model compile with kernels ON:

1. force_eager consistency — fused_window_attention now honours the process-global
   force_eager() switch (every other fused entry point already did). Before, the
   window path gated ONLY on the DISABLE_FUSED_KERNELS import-time env, so
   use_kernels=False silently kept sliding-window attention on the Triton kernel.

2. Dynamo fences — the fused Triton autograd Functions are opaque to Dynamo
   (tracing into their Triton IR mis-launches the kernel / feeds fp64 to tl.dot).
   Each kernel's dispatch is wrapped in @torch.compiler.disable so torch.compile
   over a whole block/model graph-breaks AT the kernel and compiles everything
   around it. The pure-PyTorch reference branches stay OUTSIDE the fence so
   inductor still fuses them (the kernels-OFF compile path the d=256 seed uses).

The CUDA tests build the smallest kernels-ON config that avoids the GLA kernel's
SM120 shared-memory OOM at tiny dims (retention off) — an orthogonal kernel-tuning
limit, not a fence property.
"""
import pytest
import torch

from morph.model.transformer import MORPHTransformer, MORPHConfig
from morph.kernels.triton._eager_flag import set_force_eager


def _compilable_cfg(**kw):
    """Small kernels-ON, n_core=0 config (straight-line graph, no GLA OOM)."""
    base = dict(
        d_model=256, n_heads=2, n_kv_heads=2, d_ff=688, vocab_size=512,
        max_seq_len=512, n_prelude=2, n_core=0, n_coda=2,
        channel_dims=(128, 80, 48), compression=2,
        csa_compress_ratio=8, hca_compress_ratio=256, top_k=256, window_size=256,
        retention=False, use_kernels=True, hc_use_kernel=True, dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


# ── force_eager consistency (CPU-safe: reference paths only) ────────────────

def test_window_attention_honours_force_eager():
    """fused_window_attention must route to its reference when force_eager()."""
    from morph.kernels.triton import fused_window_attention as fwa

    q = torch.randn(1, 2, 16, 64)
    k = torch.randn(1, 2, 16, 64)
    v = torch.randn(1, 2, 16, 64)

    set_force_eager(True)
    try:
        out = fwa.fused_window_attention(q, k, v, window_size=8, scale=0.125)
    finally:
        set_force_eager(False)
    ref = fwa.fused_window_attention_reference(q, k, v, 8, 0, False, 0.125)
    # CPU + force_eager → both are the reference: exact match.
    assert torch.equal(out, ref)


# ── Dynamo fences (CUDA) ────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_full_model_compiles_with_kernels_on():
    """torch.compile over the whole model with kernels ON must fwd+bwd cleanly.

    Regression guard: removing any kernel's @torch.compiler.disable fence makes
    Dynamo trace into the Triton IR and this raises (fp64 tl.dot / bad launch).
    """
    torch.manual_seed(0)
    set_force_eager(False)
    model = MORPHTransformer(_compilable_cfg()).to("cuda").train()
    x = torch.randint(0, 512, (4, 256), device="cuda")

    compiled = torch.compile(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = compiled(x, labels=None, bag_size=0)
        loss = torch.nn.functional.cross_entropy(
            out["logits"][:, :-1].reshape(-1, 512).float(),
            x[:, 1:].reshape(-1),
        )
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_and_reference_paths_agree():
    """Kernels-ON (fenced dispatchers) ≈ reference (force_eager); a scrambled
    dispatch arg-list would blow this up or crash. Tolerance is bf16/tf32
    kernel-vs-reference, not exactness."""
    torch.manual_seed(0)
    set_force_eager(False)
    model = MORPHTransformer(_compilable_cfg()).to("cuda").eval()
    x = torch.randint(0, 512, (4, 256), device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lk = model(x, labels=None, bag_size=0)["logits"].float()
    set_force_eager(True)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lr = model(x, labels=None, bag_size=0)["logits"].float()
    finally:
        set_force_eager(False)

    rel = (lk - lr).abs().max() / lr.abs().max()
    assert rel < 5e-2, f"kernel vs reference rel {rel:.2e} too large — dispatch arg bug?"
