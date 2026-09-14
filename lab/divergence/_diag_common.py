"""Shared loader/plumbing for the four loop-diagnostic scripts (2026-09-14).

One home for: building a model from ``LABEL=CONFIG=PATH``, packing validation rows with
``_rows.py``'s stream-index-preserving cut, forcing the loop's depth (slot-loop OR plain
model) for a paired K1-K6 read, and Spearman's rho with no scipy dependency (the Spark
venv is not guaranteed to carry it).

Every instrument that rebuilds the front (``_tul_front``) MUST call
``model._tul_tg_kwargs(layout)`` first and pass its front kwargs through — a bare
``_tul_front(input_ids, layout)`` on a ``tg_geometry=strict`` model (the strict ruler,
``slot-spandec-strict``) runs the prelude UNRESTRICTED and every state downstream of it is
off-distribution (2026-09-13, `slot_rank_anatomy.py`'s docstring, `_tul_tg_kwargs`'s own
docstring). On the plain model (``tul is None``) there is no front to restrict; callers
branch on ``layout is None``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")


class Arm:
    """One loaded checkpoint: the model, whether it is a slot-loop TUL model, and the
    packed evaluation batches (each row kept whole — batch size is fixed at 1 per row so
    every per-row instrument gets an isolated mask)."""

    def __init__(self, label: str, config: str, path: str, device: str,
                 rows: int, row_batch: int = 1, skip_samples: int = 0):
        from morph.training.data import create_dataloader
        from morph.training.tul_setup import build_tul_runtime
        from tul_samples import load_ckpt

        self.label = label
        self.config = config
        self.device = device
        # tg_scoped_kernels=false as well: these instruments monkey-patch and re-invoke
        # eager functions (_window_fallback, core_init), which the scoped Triton kernel
        # path does not know about (docs/tul-tg-spec.md; lab/divergence/
        # slot_state_linear_probe.py's own --override note). Correctness over speed here
        # — every run in this file is eval-only.
        # ++ (not +) because a plain (no-tul) config's model.* struct never declares this
        # key at all — a bare + would refuse to override it where it DOES already exist
        # (the TG arms), and a bare override= would refuse to ADD it where it doesn't.
        cfg = build_cfg(config, ["model.use_kernels=false", "++model.tg_scoped_kernels=false"])
        self.cfg = cfg
        tul_rt = build_tul_runtime(cfg)
        self.tul_rt = tul_rt
        self.is_slot_loop = tul_rt is not None and not bool(tul_rt.model_cfg.tokens_through_core)
        self.plain = tul_rt is None
        if tul_rt is not None and tul_rt.model_cfg.tokens_through_core:
            raise SystemExit(
                f"{label}: this instrument needs a SLOT-LOOP model (tokens_through_core "
                "false) or a PLAIN model (no tul block); the paid loop is neither.")
        model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg if tul_rt else None)
        model.eval()
        self.model = model
        self.step = step

        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=skip_samples,
                                   bag_size=0, tul=None)
        row_tokens = (int(cfg.data.seq_len) + 1 if self.plain
                     else tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1)
        stream = stream_from_loader(loader, rows * row_tokens)
        batches = pack_rows(stream, tul_rt, cfg, row_batch, self.plain)
        n_needed = -(-rows // row_batch)
        self.batches = batches[:n_needed]
        self.n_rows = sum(inp.shape[0] for inp, _, _, _ in self.batches)

    def rows(self):
        """Yield ``(inp[1,L], labels[1,L], layout_or_None)`` one row at a time, on device."""
        for inp, labels, layout, _idx in self.batches:
            for b in range(inp.shape[0]):
                lay_b = None
                if layout is not None:
                    from morph.model.tul_layout import SlotLayout
                    lay_b = SlotLayout(
                        slot_mask=layout.slot_mask[b:b + 1],
                        bag_id=layout.bag_id[b:b + 1],
                        slot_index=layout.slot_index[b:b + 1],
                        slot_valid=layout.slot_valid[b:b + 1],
                        prefix_k=layout.prefix_k,
                    ).to(self.device)
                yield inp[b:b + 1].to(self.device), labels[b:b + 1].to(self.device), lay_b

    # ── forced-depth per-row token CE, paired across a depth ladder ──────────────
    # ``core_depth_sweep.py``'s rule: a k-fixed arm (``tul.slot_depth_fixed > 0``)
    # ignores ``slot_mean_depth`` at eval, so forcing depth must go through whichever
    # knob the checkpoint actually reads. Neither target checkpoint here sets
    # ``slot_depth_fixed`` (verified: 0 on both), but this stays correct for others.
    def set_depth(self, depth: int) -> None:
        if self.plain:
            self.model.cfg.mean_depth = int(depth)
            return
        tc = self.model.cfg.tul
        if int(tc.slot_depth_fixed) > 0:
            tc.slot_depth_fixed = int(depth)
        else:
            tc.slot_mean_depth = int(depth)
        tc.slot_max_depth = max(int(depth), int(tc.slot_max_depth) or int(self.cfg.model.max_depth))

    def restore_depth(self, orig_mean: int, orig_max: int, orig_fixed: int = 0) -> None:
        if self.plain:
            self.model.cfg.mean_depth = orig_mean
            return
        tc = self.model.cfg.tul
        tc.slot_mean_depth = orig_mean
        tc.slot_max_depth = orig_max
        tc.slot_depth_fixed = orig_fixed

    def orig_depth(self) -> tuple[int, int, int]:
        """``(mean, max, fixed)``; ``fixed`` is always 0 for a plain model."""
        if self.plain:
            return int(self.model.cfg.mean_depth), 0, 0
        tc = self.model.cfg.tul
        return int(tc.slot_mean_depth), int(tc.slot_max_depth), int(tc.slot_depth_fixed)

    def own_eval_depth(self) -> int:
        """The depth the checkpoint ACTUALLY runs at eval today (no override)."""
        if self.plain:
            return int(self.model.cfg.mean_depth)
        tc = self.model.cfg.tul
        return int(tc.slot_depth_fixed) or int(tc.slot_mean_depth) or int(self.model.cfg.mean_depth)


@torch.no_grad()
def row_token_ce(model, inp: torch.Tensor, labels: torch.Tensor, layout, device: str) -> tuple[float, int]:
    """(sum of per-token CE, count) over the row's real TOKEN positions (label >= 0,
    never a slot position). One row (B=1) at a time."""
    import torch.nn.functional as F
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        if layout is None:
            res = model(inp, labels=None)
        else:
            res = model.tul_forward_ablated(inp, None, layout, plan_mode="normal")
    logits = res["logits"].float()
    B, L, V = logits.shape
    tokpos = (labels >= 0) if layout is None else ((~layout.slot_mask) & (labels >= 0))
    lab = labels.clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L), reduction="none").reshape(B, L)
    ce = ce[tokpos]
    return float(ce.sum().item()), int(tokpos.sum().item())


def jacobian_top2(fn, h0: torch.Tensor, mask: torch.Tensor, n_iter: int = 20,
                  seed: int = 0) -> tuple[float, float]:
    """Top-two singular values of ``d fn / d h`` at ``h0``, restricted to ``mask``.

    Subspace (orthogonal) iteration on ``A = J^T J`` with a 2-column block: apply ``A``
    to both columns, re-orthonormalise the pair with a real QR every iteration (a
    hand-rolled single-vector Gram-Schmidt deflation converges the SECOND column onto
    the FIRST eigenvalue whenever the top estimate has not yet fully converged — measured
    directly, ``tests/test_loop_diagnostics.py::test_jacobian_top2_recovers_known_singular_values``
    reads sigma2 == sigma1 on a synthetic 5/3 spectrum with that scheme), then read the
    eigenvalues off the small 2x2 Rayleigh quotient ``Q^T A Q`` rather than off
    ``||A q_i||`` alone, which stays accurate even a few iterations before ``Q`` fully
    converges. ``J v`` and ``J^T w`` are the matrix-free double-backward identity of
    ``morph/training/core_jacobian.py::_jacobian_stats``. ``fn`` must map a tensor shaped
    like ``h0`` to a tensor shaped like ``h0``.
    """
    h = h0.detach().float().requires_grad_(True)
    y = fn(h)
    if y.shape != h.shape:
        raise RuntimeError(f"jacobian_top2: fn is not an endomorphism: in {tuple(h.shape)} "
                           f"out {tuple(y.shape)}")
    y = y.float()
    u = torch.zeros_like(y, requires_grad=True)
    (g,) = torch.autograd.grad(y, h, grad_outputs=u, create_graph=True)
    shape = h0.shape
    n_free = h0.numel()

    def jv(v):
        (out,) = torch.autograd.grad(g, u, grad_outputs=v.view(shape), retain_graph=True)
        return (out * mask).reshape(-1)

    def jtw(w):
        (out,) = torch.autograd.grad(y, h, grad_outputs=w.view(shape), retain_graph=True)
        return (out * mask).reshape(-1)

    def av(v):
        return jtw(jv(v))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    Q0 = torch.randn(n_free, 2, generator=gen).to(device=h0.device, dtype=torch.float32)
    mask_flat = mask.expand(shape).reshape(-1)
    Q0 = Q0 * mask_flat.unsqueeze(1)
    Q, _ = torch.linalg.qr(Q0)

    for _ in range(max(1, n_iter)):
        AQ = torch.stack([av(Q[:, 0]), av(Q[:, 1])], dim=1)
        Q, _ = torch.linalg.qr(AQ)

    AQ = torch.stack([av(Q[:, 0]), av(Q[:, 1])], dim=1)
    M = Q.t() @ AQ
    M = 0.5 * (M + M.t())                                       # symmetrise (A is PSD)
    evals = torch.linalg.eigvalsh(M).clamp_min(0.0)              # ascending
    sigma1 = float(evals[-1].sqrt())
    sigma2 = float(evals[0].sqrt())
    return sigma1, sigma2


def spearman_rho(x: list[float], y: list[float]) -> tuple[float, int]:
    """Spearman's rank correlation, no scipy. Average ranks on ties. Returns (rho, n)."""
    n = len(x)
    if n < 2:
        return float("nan"), n

    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(n, dtype=np.float64)
        vs = np.asarray(v)[order]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[j + 1] == vs[i]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(np.asarray(x, dtype=np.float64)), ranks(np.asarray(y, dtype=np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0.0:
        return float("nan"), n
    return float((rx * ry).sum() / denom), n
