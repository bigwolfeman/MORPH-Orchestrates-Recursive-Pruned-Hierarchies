"""Side CUDA streams for work that does not sit on the critical path (``model.slot_gain_stream``,
``model.split_wgrad``).

Rules every user of a side stream in this tree follows (graph_step records the step, so a
stream mistake is a race on EVERY replay, not a one-off):

  * A stream is created once, in an eager step, never inside a capture
    (:func:`side_stream` raises if asked to).
  * A fork is ``side.wait_stream(current)`` and a join is ``current.wait_event(event)``;
    inside a capture these two calls are what pull the side stream into, and back out of,
    the recording. A side stream the capture never joins makes the capture fail loudly.
  * A tensor allocated on one stream and read on another is kept alive for the reader:
    :func:`record_on` (``Tensor.record_stream``) in eager mode; the same call inside a
    capture defers the free to the end of the capture.
"""
from __future__ import annotations

import torch

__all__ = ["side_stream", "record_on"]


_STREAMS: dict[tuple[str, int], torch.cuda.Stream] = {}


def side_stream(name: str, device: torch.device | int | None = None) -> torch.cuda.Stream:
    """The process-wide side stream called ``name`` on ``device``, made on first use. Refuses
    to make one inside a capture (a stream the recording creates is absent from the replay's
    accounting) and on a machine without CUDA (no overlap exists there to ask for). Kept in
    a registry, not on a module, so a deepcopy of a model never meets a stream."""
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    key = (name, idx)
    st = _STREAMS.get(key)
    if st is None:
        if not torch.cuda.is_available():
            raise RuntimeError(f"side stream {name!r} needs CUDA; the key that asked for it "
                               f"cannot run on this machine")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"side stream {name!r} must be created in an eager step, before "
                               f"the CUDA graph is recorded (the warm-up steps run first)")
        st = _STREAMS[key] = torch.cuda.Stream(device=idx)
    return st


def record_on(stream: torch.cuda.Stream, *objs) -> None:
    """``record_stream(stream)`` on every tensor in ``objs`` (tensors, and any nesting of
    tuples, lists and dicts of them; other objects are skipped). Call it for each tensor a
    side stream READS that another stream allocated, so the allocator keeps the block until
    the side stream's work on it is done."""
    for o in objs:
        if torch.is_tensor(o):
            if o.is_cuda:
                o.record_stream(stream)
        elif isinstance(o, (tuple, list)):
            record_on(stream, *o)
        elif isinstance(o, dict):
            record_on(stream, *o.values())
