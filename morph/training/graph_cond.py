"""A two-way branch inside a replayed step: a CUDA IF/ELSE conditional node whose two bodies
are recorded torch CUDA graphs, chosen ON THE DEVICE by a bool tensor (2026-10-10, phase-2
tracker C4; used by morph/training/graph_step.py).

Why. The gain hinge's backward (`model.slot_gain_lambda`, two extra core applications at the
drawn pass) is about 11 ms of the lxtul_pointer_fast step, and on most steps its gradient is
exactly zero: no row's gain is over the target, so the hinge `relu(gain - target)` is 0. Which
case a step is in is known only on the device, after the forward. A host read would put a
sync in the middle of every step; a conditional node reads the flag where it lives.

How. PyTorch 2.11 has no conditional nodes, so the graph is assembled with the CUDA driver
API (cuda.bindings, NVIDIA's Python bindings):

    [set kernel: handle <- *flag] -> IF handle: [child graph: if_true] ELSE: [child graph: if_false]

The two bodies are ordinary torch recordings (``torch.cuda.CUDAGraph(keep_graph=True)``, so
torch keeps the raw graph and never instantiates it); the conditional node clones each one
into its body as a child-graph node. The set kernel is four lines of CUDA C compiled once per
device architecture with NVRTC (``cudaGraphSetConditional`` is a device-runtime builtin; no
device runtime library is linked).

What the bodies may hold. A conditional body takes kernel, memset, memcpy, child-graph and
conditional nodes; a recording with an event or host node fails at instantiation (loud).
A body must draw NO random numbers: a torch graph's RNG offsets are set by its own
``replay()`` (the generator's replay prologue), which this launch does not run, so a draw in
a body would reuse the offsets of the graph replayed before it. The caller checks this
(graph_step.py: the generator offset across the recording).
"""
from __future__ import annotations

import ctypes
from typing import Optional

import torch

_SRC = rb"""
typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(
    cudaGraphConditionalHandle handle, unsigned int value);
extern "C" __global__ void morph_set_cond(cudaGraphConditionalHandle h, const bool* flag) {
    cudaGraphSetConditional(h, *flag ? 1u : 0u);
}
"""

_KERNELS: dict = {}           # device index -> (module, function); compiled once per process


def _ck(res, what: str):
    """Unwrap a cuda.bindings call: raise on any error code, return the remaining values."""
    err, *rest = res if isinstance(res, tuple) else (res,)
    if int(err) != 0:
        raise RuntimeError(f"graph_cond: {what} failed: {err!r}")
    return rest[0] if len(rest) == 1 else (tuple(rest) if rest else None)


def _set_kernel(device: int):
    if device in _KERNELS:
        return _KERNELS[device][1]
    from cuda.bindings import driver as drv
    from cuda.bindings import nvrtc
    major, minor = torch.cuda.get_device_capability(device)
    prog = _ck(nvrtc.nvrtcCreateProgram(_SRC, b"morph_set_cond.cu", 0, [], []),
               "nvrtcCreateProgram")
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-default-device"]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(res[0]) != 0:
        n = _ck(nvrtc.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"graph_cond: NVRTC failed: {log.decode(errors='replace')}")
    n = _ck(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")
    cubin = b" " * n
    _ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
    nvrtc.nvrtcDestroyProgram(prog)
    with torch.cuda.device(device):
        torch.cuda.current_stream().synchronize()            # the primary context is current
        mod = _ck(drv.cuModuleLoadData(cubin), "cuModuleLoadData")
        fn = _ck(drv.cuModuleGetFunction(mod, b"morph_set_cond"), "cuModuleGetFunction")
    _KERNELS[device] = (mod, fn)
    return fn


def graph_node_count(g: torch.cuda.CUDAGraph) -> int:
    """The node count of a recording kept with ``keep_graph=True`` (the tests read it: the
    hinge branch's B body must be a smaller graph than A)."""
    from cuda.bindings import driver as drv
    _nodes, n = _ck(drv.cuGraphGetNodes(drv.CUgraph(g.raw_cuda_graph()), 0),
                    "cuGraphGetNodes")
    return int(n)


class CondBranch:
    """``if flag: if_true else: if_false``, as one instantiated CUDA graph.

    ``flag``: a CUDA bool scalar at a fixed address (a recorded graph's output); read on the
    device at every launch. ``if_true`` / ``if_false``: finished recordings made with
    ``torch.cuda.CUDAGraph(keep_graph=True)``; they are cloned into the branch, but the
    caller keeps them (and their memory pool) alive as long as the branch is launched,
    because the clones read and write the pool's memory."""

    def __init__(self, flag: torch.Tensor, if_true: torch.cuda.CUDAGraph,
                 if_false: torch.cuda.CUDAGraph):
        from cuda.bindings import driver as drv
        if not (flag.is_cuda and flag.dtype == torch.bool and flag.numel() == 1):
            raise ValueError(f"graph_cond: the flag must be one CUDA bool, got "
                             f"{flag.dtype} {tuple(flag.shape)} on {flag.device}")
        self.flag = flag
        self.device = flag.device.index
        fn = _set_kernel(self.device)
        with torch.cuda.device(self.device):
            ctx = _ck(drv.cuCtxGetCurrent(), "cuCtxGetCurrent")
            if int(ctx) == 0:
                raise RuntimeError("graph_cond: no current CUDA context")
            self._graph = _ck(drv.cuGraphCreate(0), "cuGraphCreate")
            self._exec = None
            try:
                handle = _ck(drv.cuGraphConditionalHandleCreate(
                    self._graph, ctx, 0, drv.CU_GRAPH_COND_ASSIGN_DEFAULT),
                    "cuGraphConditionalHandleCreate")
                # the set kernel: one thread, (handle, flag pointer)
                self._args = (ctypes.c_ulonglong(int(handle)), ctypes.c_void_p(flag.data_ptr()))
                self._argv = (ctypes.c_void_p * 2)(*[ctypes.addressof(a) for a in self._args])
                kp = drv.CUDA_KERNEL_NODE_PARAMS()
                kp.func = fn
                kp.gridDimX = kp.gridDimY = kp.gridDimZ = 1
                kp.blockDimX = kp.blockDimY = kp.blockDimZ = 1
                kp.sharedMemBytes = 0
                kp.kernelParams = ctypes.addressof(self._argv)
                n_set = _ck(drv.cuGraphAddKernelNode(self._graph, [], 0, kp),
                            "cuGraphAddKernelNode")
                cp = drv.CUgraphNodeParams()
                cp.type = drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
                cp.conditional.handle = handle
                cp.conditional.type = drv.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
                cp.conditional.size = 2                    # body 0: value != 0; body 1: else
                cp.conditional.ctx = ctx
                _ck(drv.cuGraphAddNode(self._graph, [n_set], 1, cp), "cuGraphAddNode(cond)")
                bodies = cp.conditional.phGraph_out
                for body, g in ((bodies[0], if_true), (bodies[1], if_false)):
                    _ck(drv.cuGraphAddChildGraphNode(body, [], 0,
                                                     drv.CUgraph(g.raw_cuda_graph())),
                        "cuGraphAddChildGraphNode")
                fl = drv.CUgraphInstantiate_flags
                self._exec = _ck(drv.cuGraphInstantiate(
                    self._graph, int(fl.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                    | int(fl.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY)),
                    "cuGraphInstantiate")
            except Exception:
                self.close()
                raise

    def launch(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Run the branch on ``stream`` (default: the current stream), after the work queued
        there before it, like a torch graph replay."""
        from cuda.bindings import driver as drv
        if self._exec is None:
            raise RuntimeError("graph_cond: the branch was closed")
        s = stream if stream is not None else torch.cuda.current_stream(self.device)
        _ck(drv.cuGraphLaunch(self._exec, drv.CUstream(s.cuda_stream)), "cuGraphLaunch")

    def close(self) -> None:
        """Free the instantiated branch (the caller first makes sure no launch is running)."""
        from cuda.bindings import driver as drv
        if getattr(self, "_exec", None) is not None:
            _ck(drv.cuGraphExecDestroy(self._exec), "cuGraphExecDestroy")
            self._exec = None
        if getattr(self, "_graph", None) is not None:
            _ck(drv.cuGraphDestroy(self._graph), "cuGraphDestroy")
            self._graph = None
