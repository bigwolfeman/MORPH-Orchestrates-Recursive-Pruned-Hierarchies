"""Cross-layer L2 residency: mark the HC carrier's address range PERSISTING so it survives
the sublayer GEMMs' streaming traffic between HC ops (carrier reads hit L2, not HBM).

Mechanism proven on sm_120 (ignore/l2_residency.py: -19.6% on the isolated reread pattern).
cc8.0+ (all deploy archs). Numerically a NO-OP (caching hint only) — never changes results.

Lazy, cached, graceful: the CUDA extension compiles on first use; any failure (no nvcc, wrong
arch, non-cuda) degrades to no-ops so training is never blocked. Per-call host cost ~µs.

CAVEAT (measured): MORPH's carrier is functional (new alloc per HC op → hopping addresses),
so a window on one tensor only persists THAT address. Full benefit needs a ping-pong carrier
(≤2 fixed buffers). This module is the mechanism + the cheap per-iteration hook for measurement.
"""
from __future__ import annotations

import os
import functools

_CUDA = "/opt/cuda/targets/x86_64-linux"

_CPP = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstring>

static size_t g_max_persist = 0;

void set_window(uintptr_t ptr, int64_t nbytes, double hitRatio) {
    int dev; cudaGetDevice(&dev);
    if (g_max_persist == 0) {
        int mp = 0; cudaDeviceGetAttribute(&mp, cudaDevAttrMaxPersistingL2CacheSize, dev);
        g_max_persist = (size_t)mp;
    }
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, g_max_persist);
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    cudaStreamAttrValue av; memset(&av, 0, sizeof(av));
    av.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
    av.accessPolicyWindow.num_bytes = (size_t)nbytes;
    av.accessPolicyWindow.hitRatio  = (float)hitRatio;
    av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &av);
}
void reset_window() {
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    cudaStreamAttrValue av; memset(&av, 0, sizeof(av));
    av.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &av);
    cudaCtxResetPersistingL2Cache();
}
int64_t max_persist_bytes() {
    int dev; cudaGetDevice(&dev);
    int mp = 0; cudaDeviceGetAttribute(&mp, cudaDevAttrMaxPersistingL2CacheSize, dev);
    return (int64_t)mp;
}
"""


@functools.lru_cache(maxsize=1)
def _ext():
    """Compile (cached) + return the extension, or None on any failure (→ no-op)."""
    try:
        import torch  # noqa
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        os.environ.setdefault("CXX", "g++-13")
        os.environ.setdefault("CC", "gcc-13")
        return load_inline(
            name="morph_l2_persist",
            cpp_sources=[_CPP],
            functions=["set_window", "reset_window", "max_persist_bytes"],
            extra_cflags=[f"-I{_CUDA}/include"],
            extra_ldflags=[f"-L{_CUDA}/lib", "-lcudart"],
            extra_cuda_cflags=["-arch=sm_120", "-O3", "-ccbin", "g++-13"],
            verbose=False,
        )
    except Exception as e:  # pragma: no cover
        print(f"[l2_persist] extension unavailable ({type(e).__name__}: {e}) — L2 hints are no-ops")
        return None


@functools.lru_cache(maxsize=1)
def _max_persist() -> int:
    e = _ext()
    return int(e.max_persist_bytes()) if e is not None else 0


def set_carrier(tensor) -> None:
    """Mark up to max-persisting bytes of `tensor`'s storage as L2-persisting. No-op on failure."""
    e = _ext()
    if e is None or not tensor.is_cuda:
        return
    nbytes = min(tensor.numel() * tensor.element_size(), _max_persist())
    if nbytes > 0:
        e.set_window(tensor.data_ptr(), nbytes, 1.0)


def reset() -> None:
    """Clear the persisting window + reset persisting L2. No-op on failure."""
    e = _ext()
    if e is not None:
        e.reset_window()


def available() -> bool:
    return _ext() is not None
