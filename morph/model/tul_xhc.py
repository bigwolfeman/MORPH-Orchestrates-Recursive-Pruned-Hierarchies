"""Expanded Hyper-Connections (xHC, Zhang et al., arXiv 2607.14530) for the SLOT LOOP's core.

Plan C (`.agents/notes/proposed/architecture/2026-09-26-plan-c-xhc-slot-loop.md`). On a model
built with `tul.xhc_streams = N > 0` the core blocks' two residuals (`mrr_attn`, `mrr_mlp`,
the legacy names of `HyperConnectionResidual`) are REPLACED by :class:`XHCResidual`. The core
blocks run only inside `_tul_core` (the slot loop), so the prelude and the coda keep their
n = 4 Cayley carrier and the token path never sees the 16 streams.

One sublayer's update, per slot position, on the carrier ``X`` ``[N, C]``:

    rms       = RMS(vec X)                                   fp32, full state
    H_pre     = softmax(W_pre vec X / rms + b_pre)           [N]   DENSE read, one row
    x_bar     = sum_j H_pre[j] X[j]                          the sublayer's one input
    y         = F(x_bar)                                     attention or MLP, run once
    s         = sigmoid(W_r vec X / rms + b_r)               [N - m] router scores
    A         = {0 .. m-1} + the (k - m) best of s           stable sort, ties -> lower index
    g         = [1] * m + s[routed]                          write gates of the k active streams
    X_A       = X[A]                                         [k, C], gathered in that order
    rms_A     = RMS(vec X_A)
    H_res     = Cayley(W_res vec X_A / rms_A + b_res)        [k, k] ORTHOGONAL (k = 4: closed form)
    H_post    = rowsum(softmax_col(W_post vec X_A / rms_A + b_post))    [k], MORPH's HC form
    X'[A]     = H_res X_A + g * H_post * y   (+ g * H_aug @ aug(y) under temporal augmentation)
    X'[i]     = X[i]                        for every idle stream i (exactly unchanged)

The router's gradient. TopK is not differentiable in its index. The selected streams' WRITE
(``H_post y`` and the augmented components) is multiplied by the router score ``s_i``, the
MoE gate rule; the m fixed streams have gate 1. So the router learns through the size of
what a routed stream receives, and the orthogonal mixer ``H_res`` is never scaled (a
convex blend of I and an orthogonal matrix is not orthogonal, and the Cayley isometry is
the reason MORPH uses Cayley in the weight-shared loop). No straight-through estimator, no
noise, no load-balancing loss (the paper has none either). The indices are a pure function
of the state, so the router is deterministic and the checkpoint recompute selects the same
streams.

Temporal augmentation (``temporal_kernels``, MLP sublayer only, as in the paper): the
sublayer output ``y`` ``[B, S, C]`` over the SLOT axis S, pads zeroed by ``valid``, gives
``R`` extra components ``DWConv_kappa(y)`` (causal depthwise, kernel ``kappa`` slots; slot s
reads slots s-kappa+1 .. s). They are made orthogonal to ``y`` and to each other by modified
Gram-Schmidt per position (projection removal only, no renormalisation, so a component
keeps the size of its novel part), and written into the k active streams through
``H_aug = tanh(W_aug vec X_A / rms_A + b_aug)`` ``[k, R]``, gated by ``g``. ``b_aug`` starts at
atanh(0.01), so the extra writes start at 1 % (the paper's alpha = 0.01 init rule) and every
conv weight has a gradient from the first step. The conv weights start as the causal moving
average (``1 / kappa``), a deterministic init.

Never ternarised, never pruned: every weight is a plain ``nn.Parameter`` on a module that is
not ``nn.Linear`` / ``nn.Embedding`` / CMS (the `PassLoRA` precedent), and the module carries
``_ternary_exclude`` as well. Built from a PRIVATE generator, so attaching it moves no
global RNG state. Biases are named ``*_bias`` so the optimizer's ``bias`` keyword puts them
in the no-decay group, as `HyperConnectionResidual.proj.bias` is.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hyper_connections import cayley_orthogonal


def xhc_select(score: Tensor, n_fixed: int, n_routed: int) -> tuple[Tensor, Tensor]:
    """The active stream set and its write gates from the router scores.

    ``score`` ``[..., N - n_fixed]`` (sigmoid outputs for the candidate streams
    ``n_fixed .. N-1``). Returns ``(idx, gate)``, both ``[..., n_fixed + n_routed]``: the fixed
    streams ``0 .. n_fixed-1`` first (gate 1), then the ``n_routed`` best candidates in
    descending score order (gate = their score). A STABLE descending sort, so equal scores
    keep the lower stream index first on every device (``torch.topk`` makes no tie
    promise); an all-equal row, e.g. a pad slot whose state is 0, therefore always picks
    ``n_fixed .. n_fixed + n_routed - 1``.
    """
    lead = score.shape[:-1]
    order = torch.sort(score, dim=-1, descending=True, stable=True).indices[..., :n_routed]
    fixed = torch.arange(n_fixed, device=score.device).expand(*lead, n_fixed)
    idx = torch.cat([fixed, order + n_fixed], dim=-1)
    gate = torch.cat([score.new_ones(*lead, n_fixed), score.gather(-1, order)], dim=-1)
    return idx, gate


def causal_slot_conv(y: Tensor, weight: Tensor) -> Tensor:
    """Causal depthwise convolution over the slot axis.

    ``y`` ``[B, S, C]``, ``weight`` ``[C, 1, kappa]``. Output slot ``s`` is
    ``sum_j weight[:, 0, j] * y[s - (kappa - 1 - j)]``: tap ``kappa - 1`` is the slot itself
    and tap 0 the slot ``kappa - 1`` back; slots before 0 are zero (left padding only).
    """
    kappa = weight.shape[-1]
    x = F.pad(y.transpose(1, 2), (kappa - 1, 0))                   # [B, C, S + kappa - 1]
    return F.conv1d(x, weight.to(x.dtype), groups=y.shape[-1]).transpose(1, 2)


def gram_schmidt_components(base: Tensor, comps: Sequence[Tensor], eps: float = 1e-12) -> Tensor:
    """Modified Gram-Schmidt, per position over the last axis, fp32.

    ``base`` ``[..., C]`` is kept as it is (it is the sublayer output the main write already
    carries); each of ``comps`` has the part along ``base`` and along every EARLIER
    orthogonalised component removed, one projection at a time (the modified form). No
    renormalisation. Returns the orthogonalised ``comps`` stacked on a new axis ``-2``:
    ``[..., R, C]``. A zero basis vector (a pad slot) removes nothing and divides nothing.
    """
    basis = [base.float()]
    out = []
    for c in comps:
        u = c.float()
        for q in basis:
            u = u - ((u * q).sum(-1, keepdim=True) / ((q * q).sum(-1, keepdim=True) + eps)) * q
        basis.append(u)
        out.append(u)
    return torch.stack(out, dim=-2)


class XHCResidual(nn.Module):
    """Sparse-write, dense-read N-stream residual for one core sublayer (see module doc).

    Args:
        d_model:          per-stream width C.
        n_streams:        N, the carrier's stream count (16 on plan C).
        n_active:         k, streams written per sublayer (4). The k x k mixer is the Cayley
                          map, which has the closed 4 x 4 form at k = 4.
        n_fixed:          m, streams always written (2); ``n_fixed < n_active``.
        temporal_kernels: slot-axis kernel sizes of the temporal augmentation; ``()`` = off
                          (the attention sublayer always passes ``()``).
        tau, cayley_alpha, init_gain: as `HyperConnectionResidual` (the model's hc_* keys).
        generator:        the PRIVATE generator every random init draws from.
    """

    def __init__(self, d_model: int, n_streams: int, n_active: int, n_fixed: int,
                 temporal_kernels: Sequence[int] = (), tau: float = 1.0,
                 cayley_alpha: float = 0.1, init_gain: float = 0.1,
                 generator: torch.Generator | None = None):
        super().__init__()
        if not (0 <= n_fixed < n_active <= n_streams):
            raise ValueError(
                f"XHCResidual needs 0 <= n_fixed < n_active <= n_streams, got "
                f"n_fixed={n_fixed}, n_active={n_active}, n_streams={n_streams}")
        ks = tuple(int(k) for k in temporal_kernels)
        if any(k < 2 for k in ks) or len(set(ks)) != len(ks):
            raise ValueError(f"temporal_kernels must be distinct ints >= 2, got {ks}")
        self.d_model, self.n, self.k, self.m = int(d_model), int(n_streams), int(n_active), \
            int(n_fixed)
        self.kernels = ks
        self.r = len(ks)
        self.tau, self.cayley_alpha = float(tau), float(cayley_alpha)
        self.eps = 1e-6
        self._ternary_exclude = True
        # The key this module's choice is recorded under by the gain hinge (`route`); set
        # by the model to the module's own path, unique within the core.
        self.route_key = ""
        C, N, k = self.d_model, self.n, self.k
        g = generator

        def _w(rows: int, fan_in: int) -> nn.Parameter:
            return nn.Parameter(torch.empty(rows, fan_in).normal_(
                0.0, init_gain / math.sqrt(fan_in), generator=g))

        # Dense read over all N streams (one softmax row) and the router over N - m.
        self.w_pre = _w(N, N * C)
        self.pre_bias = nn.Parameter(torch.zeros(N))
        self.w_route = _w(N - self.m, N * C)
        self.route_bias = nn.Parameter(torch.zeros(N - self.m))
        # The active-set maps, one fused projection of the normalised active state:
        # [H_res k*k | H_post k*k | H_aug k*R].
        n_act_out = 2 * k * k + k * self.r
        self.w_act = _w(n_act_out, k * C)
        act_bias = torch.zeros(n_act_out)
        if self.r:
            act_bias[2 * k * k:] = math.atanh(0.01)
        self.act_bias = nn.Parameter(act_bias)
        # Causal depthwise convs over the slot axis, moving-average init (no RNG).
        self.conv_w = nn.ParameterList(
            [nn.Parameter(torch.full((C, 1, kk), 1.0 / kk)) for kk in ks])

    def extra_repr(self) -> str:
        return (f"streams={self.n}, active={self.k}, fixed={self.m}, "
                f"temporal_kernels={self.kernels}")

    def route(self, h: Tensor, fixed_route: dict | None = None
              ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(idx, gate, H_pre, rms)`` of the carrier ``h`` ``[B, S, N, C]``; fp32 maps.

        ``fixed_route`` (the slot-gain hinge's finite difference ONLY, never the training
        pass): ``{"replay": False, "idx": {}}`` records this module's chosen streams under
        ``self.route_key``; ``{"replay": True, "idx": <the recorded dict>}`` reuses them and
        recomputes only the gates from the current scores. The map is discontinuous where
        a perturbation flips the TopK choice, and a difference across that jump divided by
        a small step is not a gain; holding the choice fixed measures the map within the
        piece the operating point is in. ``None`` routes as always.
        """
        B, S, N, C = h.shape
        xf = h.reshape(B, S, N * C)
        rms = xf.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()     # [B, S, 1]
        pre = F.linear(xf, self.w_pre.to(xf.dtype)).float() / rms + self.pre_bias.float()
        h_pre = torch.softmax(pre / self.tau, dim=-1)                          # [B, S, N]
        score = torch.sigmoid(
            F.linear(xf, self.w_route.to(xf.dtype)).float() / rms + self.route_bias.float())
        if fixed_route is not None and fixed_route["replay"]:
            idx = fixed_route["idx"][self.route_key]
            gate = torch.cat([score.new_ones(B, S, self.m),
                              score.gather(-1, idx[..., self.m:] - self.m)], dim=-1)
        else:
            idx, gate = xhc_select(score, self.m, self.k - self.m)
            if fixed_route is not None:
                fixed_route["idx"][self.route_key] = idx.detach()
        return idx, gate, h_pre, rms

    def augment(self, y: Tensor, valid: Tensor) -> Tensor:
        """The orthogonalised temporal components ``[B, S, R, C]`` (fp32) of ``y``.

        ``valid`` ``[B, S]`` bool: pad slots are zeroed BEFORE the convolutions, so a pad's
        output never enters a valid slot's component whatever its position in the row.
        """
        yv = y * valid.unsqueeze(-1).to(y.dtype)
        raw = [causal_slot_conv(yv, w) for w in self.conv_w]
        return gram_schmidt_components(yv, raw)

    def forward(self, h: Tensor, sublayer_fn: Callable[..., Tensor], *args,
                post_inject: Tensor | None = None, valid: Tensor | None = None,
                fixed_route: dict | None = None, **kwargs) -> Tensor:
        """``h`` ``[B, S, N, C]`` -> the updated carrier, same shape.

        ``valid`` ``[B, S]`` bool is REQUIRED under temporal augmentation (the pad mask of the
        convolutions) and must be None without it. ``post_inject`` (the prelude/coda
        carrier-engine fold) is never used by the slot loop and is refused.
        ``fixed_route``: see :meth:`route` (the gain hinge only).
        """
        if h.dim() != 4 or h.shape[2] != self.n:
            raise ValueError(
                f"XHCResidual expects a [B, S, {self.n}, C] carrier, got {tuple(h.shape)}. "
                f"The expanded core runs in `_tul_core` only.")
        if post_inject is not None:
            raise NotImplementedError(
                "XHCResidual: `post_inject` (the carrier-engine fold) is not defined on the "
                "expanded slot-loop carrier; the slot loop never passes it.")
        if (valid is None) != (self.r == 0):
            raise ValueError(
                "XHCResidual: `valid` must be given exactly when temporal augmentation is on "
                f"(kernels {self.kernels}).")
        B, S, N, C = h.shape
        k = self.k
        dt = h.dtype
        idx, gate, h_pre, _ = self.route(h, fixed_route)
        x_bar = torch.einsum("bsn,bsnc->bsc", h_pre.to(dt), h)
        y = sublayer_fn(x_bar, *args, **kwargs)                               # [B, S, C]

        gidx = idx.unsqueeze(-1).expand(B, S, k, C)
        xa = h.gather(2, gidx)                                                # [B, S, k, C]
        xaf = xa.reshape(B, S, k * C)
        rms_a = xaf.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        act = F.linear(xaf, self.w_act.to(xaf.dtype)).float() / rms_a + self.act_bias.float()
        kk = k * k
        h_res = cayley_orthogonal(act[..., :kk].reshape(B, S, k, k), alpha=self.cayley_alpha)
        h_post = torch.softmax(act[..., kk:2 * kk].reshape(B, S, k, k) / self.tau,
                               dim=-2).sum(-1)                                # [B, S, k]
        new_a = torch.einsum("bsij,bsjc->bsic", h_res.to(dt), xa)
        new_a = new_a + (gate * h_post).to(dt).unsqueeze(-1) * y.to(dt).unsqueeze(2)
        if self.r:
            h_aug = torch.tanh(act[..., 2 * kk:].reshape(B, S, k, self.r)) * gate.unsqueeze(-1)
            comps = self.augment(y, valid)                                    # [B, S, R, C]
            new_a = new_a + torch.einsum("bskr,bsrc->bskc", h_aug, comps).to(dt)
        # `scatter` needs one dtype: under autocast the einsums above return bf16 while the
        # slot carrier is fp32 (RMSNorm returns fp32, morph/model/CLAUDE.md).
        return h.scatter(2, gidx, new_a.to(dt))
