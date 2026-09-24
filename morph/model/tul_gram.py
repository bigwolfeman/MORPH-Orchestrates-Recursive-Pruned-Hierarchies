"""``tul.gram`` — LXTUL-G: the slot loop as a stochastic, contractive latent-variable model.

GRAM shape (Generative Recursive Reasoning, arXiv 2605.19376; design note
``.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md``, Proposal
and Paper refresh; prereg ``lab/experiments/failures/2026-09-23-lxtul-g-panel.md``; design
synthesis ``/home/wolfe/morph-scratch/arc/notes/2026-09-23-lxtul-g-paper-synthesis.md``,
Decisions 1-7).

What it is
----------
At every pass ``t`` of the SLOT loop (``MORPHTransformer._tul_core``) the deterministic
core update ``u_t = f(h_{t-1})`` is followed by a learned Gaussian step::

    r_t  = RMS(u_t) per slot, over (streams, channels), DETACHED
    u^_t = mean_streams(u_t) / r_t
    prior      (m_p, s_p) = heads(u^_t)                 sees only what the loop sees
    posterior  (m_q, s_q) = heads(u^_t, e_next)         also sees the NEXT span (train)
    mu = r_t * m,   sigma = r_t * (softplus(s) + 1e-4)
    h_t = u_t + mu + sigma * n,   n ~ N(0, I)           single-stream, broadcast into the
                                                        Hyper-Connection streams

``e_next`` (:class:`TULGramPool`) is a one-query attention pool over the prelude states of
the span the slot PRECEDES (``bag_id == s + 1``, the ``TULCodeEncoder`` relation), projected
to ``d_model`` and RMS-normalised. Slots with no next span (pads, the row's last slot,
whose next span is the partial tail) get ``e_next = 0`` and take the PRIOR step even at
train; their KL is masked out.

The KL per pass is the closed-form diagonal Gaussian ``KL(q || p)``, summed over channels;
the shared scale ``r_t`` cancels in it, so the loss cannot be gamed by growing the state
(Synthesis Decision 4). :func:`gram_kl_balanced` applies GRAM's KL balancing:
``a * KL(sg(q) || p) + (1 - a) * KL(q || sg(p))`` (value = KL; the gradients split).

Initialisation, and why each piece is what it is
-------------------------------------------------
* ``m``'s output layer is ZERO (weight and bias): the step starts mean-free.
* ``s``'s output weight is ZERO and its bias is ``softplus^{-1}(sigma_init - 1e-4)``, so
  ``sigma / r`` starts at EXACTLY ``sigma_init`` (default 0.1) on every slot.
* The posterior heads are COPIES of the prior heads on the shared ``u^`` input, and their
  ``e_next`` input layers are separate, zero-init ``nn.Linear``s ADDED to the copied
  pre-activations. At step 0 the posterior therefore computes the SAME ops as the prior
  plus an exact 0, so ``q == p`` bit for bit and the KL is exactly 0 (a single
  ``Linear(2d, h)`` with a zero half would reorder the accumulation and break that).
* Every draw comes from a PRIVATE generator and the global CPU RNG is snapshotted and
  restored around the build (``nn.Linear``'s own kaiming draw included), so a gram model
  shares every base weight with its same-seed ruler.

Contracts
---------
* **Off is nothing.** ``tul.gram: false`` builds no module, draws no RNG and adds no term.
* **Never quantised, never pruned.** Every leaf carries ``_ternary_exclude = True``
  (honoured by ``ternary_qat._categorize``); none is a ``MortarLinear``/``CMSBlockLinear``,
  so the CMS prune, the MORTAR carve, the ReMoE router and the deploy packer walk past.
* **The posterior is train-only.** At eval the default is a PRIOR sample; the posterior
  is an eval INSTRUMENT only under ``gram_mode="post"`` (see ``MORPHTransformer.forward``).

Refused (``TULConfig._check_gram`` and ``MORPHTransformer.__init__``): the paid loop,
``loop_reads_tokens``, ``code`` / ``code_target`` / ``loop_denoise``, ``fan_k > 0``,
``slot_cells > 1``, ``vq_codes``, a non-``exit`` prefix source, a gated pass readout,
``core_stage_cond``, ``db_loop``, ``progressive_p``, ``detach_z``, a gathered coda, SCSE, an
FM planner and ``n_core == 0``. Each bypasses or reshapes ``_tul_core``'s one-state-per-slot
write, or would make the per-pass KL mean something other than the ELBO term.

The multi-sample objective (``tul.gram_objective: "iw"``, LXTUL-GK, 2026-09-23)
------------------------------------------------------------------------------
Prereg ``lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md``. NO posterior and NO
KL: the step is built without the pool and the posterior heads (``with_posterior=False``;
the private generator still draws the pool's weights and throws them away, so the prior
heads are bit-equal to an ``"elbo"`` model's at the same seed). The training forward runs
the front ONCE, the slot loop and the coda on ``K = tul.gram_iw_k`` PRIOR rollouts per row
(batch-expanded, rollout-major), and charges

    L = - sum_s log (1/K) sum_k exp( S_k(s) ) / sum_j w_j,   S_k(s) = sum_{j in s} w_j log p_k(tok_j)

(:func:`iw_span_bound`), ``s`` a span (the token positions of one ``bag_id`` in one row,
the grouping ``lab/divergence/lxtul_g_probe.py::bayes_read`` uses) and ``w_j`` the ruler's
own CE weight of position ``j``. Summed over a span it equals the per-token Bayesian read
the probe deploys (``w_k(j) = softmax_k(sum_{i<j} log p_k(tok_i))``), so the model trains
on the object it is scored on. The gradient into ``S_k(s)`` is ``softmax_k(S(s))``: each
rollout is paid in proportion to how well it explains the span.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .tul_layout import SlotLayout

__all__ = ["TULGramStep", "TULGramPool", "gram_kl", "gram_kl_balanced", "gram_has_next",
           "iw_span_bound", "iw_span_groups", "GRAM_SIGMA_FLOOR", "GRAM_MODES",
           "GRAM_OBJECTIVES"]

_SEED_GRAM = 0x6A4A1
GRAM_SIGMA_FLOOR = 1e-4
GRAM_MODES = ("prior", "post", "mean")
# `tul.gram_objective`: "elbo" = the shipped LXTUL-G (posterior + KL); "iw" = LXTUL-GK, the
# multi-sample bound over K prior rollouts (no posterior, no KL).
GRAM_OBJECTIVES = ("elbo", "iw")


def _softplus_inv(y: float) -> float:
    """``x`` with ``softplus(x) == y`` (``y > 0``)."""
    return y + math.log(-math.expm1(-y))


def _normal_(w: Tensor, std: float, gen: torch.Generator) -> None:
    with torch.no_grad():
        w.copy_(torch.empty(w.shape, device="cpu").normal_(mean=0.0, std=std, generator=gen))


def _linear(d_in: int, d_out: int, bias: bool, gen: torch.Generator | None,
            std: float | None) -> nn.Linear:
    """An ``nn.Linear`` whose weight is ``N(0, std)`` from ``gen`` (``std=None``: zeros)."""
    lin = nn.Linear(d_in, d_out, bias=bias)
    with torch.no_grad():
        if std is None:
            lin.weight.zero_()
        else:
            _normal_(lin.weight, std, gen)
        if bias:
            lin.bias.zero_()
    lin._ternary_exclude = True
    return lin


class _GramHead(nn.Module):
    """``out = W_down( silu(W_g x [+ X_g e]) * (W_u x [+ X_u e]) )``, a SwiGLU MLP.

    ``d_extra > 0`` builds the posterior's second input (``e_next``) as two SEPARATE
    zero-init linears added to the pre-activations (see the module docstring for why a
    concatenated input would not give an exact KL of 0 at init).
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, gen: torch.Generator,
                 out_bias: float, d_extra: int = 0):
        super().__init__()
        self.w_gate = _linear(d_in, d_hidden, False, gen, d_in ** -0.5)
        self.w_up = _linear(d_in, d_hidden, False, gen, d_in ** -0.5)
        self.w_down = _linear(d_hidden, d_out, True, None, None)
        with torch.no_grad():
            self.w_down.bias.fill_(float(out_bias))
        self.x_gate = self.x_up = None
        if d_extra > 0:
            self.x_gate = _linear(d_extra, d_hidden, False, None, None)
            self.x_up = _linear(d_extra, d_hidden, False, None, None)

    def copy_shared_from(self, other: "_GramHead") -> None:
        """Copy every weight the two heads share (the ``u^`` path and the output layer)."""
        with torch.no_grad():
            self.w_gate.weight.copy_(other.w_gate.weight)
            self.w_up.weight.copy_(other.w_up.weight)
            self.w_down.weight.copy_(other.w_down.weight)
            self.w_down.bias.copy_(other.w_down.bias)

    def forward(self, x: Tensor, extra: Tensor | None = None) -> Tensor:
        """fp32 output. The hidden layers run in the autocast dtype; the OUTPUT layer runs
        in fp32 with autocast off. Measured 2026-09-23 on the 40-step smokes: under bf16
        autocast the sigma head's output is ``bias + W h`` with ``bias`` = -2.25, whose
        bf16 spacing is 0.0156, so every update smaller than that was rounded away —
        sigma/r read 0.100307 (the bf16 bias) at step 0 AND step 20, prior and posterior
        bit-equal, and the mean-free arm's KL was exactly 0 for all 40 steps."""
        a = self.w_gate(x)
        b = self.w_up(x)
        if self.x_gate is not None:
            if extra is None:
                raise RuntimeError("the posterior head needs e_next; got None")
            a = a + self.x_gate(extra)
            b = b + self.x_up(extra)
        hid = F.silu(a) * b
        with torch.autocast(device_type=hid.device.type, enabled=False):
            return F.linear(hid.float(), self.w_down.weight.float(), self.w_down.bias.float())


def gram_has_next(layout: SlotLayout) -> Tensor:
    """``[B, S]`` bool: slot ``s`` is valid AND slot ``s+1`` is valid, i.e. span ``s+1`` is
    a complete span. The ``code_target_valid`` rule, restated here so this module does not
    import the LCTUL stack: the row's last valid slot precedes the partial tail and has no
    posterior input."""
    ok = torch.zeros_like(layout.slot_valid)
    S = layout.slot_valid.shape[1]
    ok[:, :S - 1] = layout.slot_valid[:, :S - 1] & layout.slot_valid[:, 1:S]
    return ok


class TULGramPool(nn.Module):
    """``e_next``: a one-query attention pool over the NEXT span's prelude states.

    ``e_s = rmsnorm( W_o( sum_j softmax_j(<q, W_k x_j> / sqrt(d)) W_v x_j ) )`` over the
    TOKEN positions ``j`` with ``bag_id == s + 1``. Weights ``N(0, 0.02)`` from the private
    generator (the ``TULCodeEncoder`` precedent: near-uniform pooling at init, i.e. about
    the span mean, then learned). Parameter-free RMS norm, so the posterior's zero-init
    ``e_next`` path starts on a unit-scale input. Slots with no next span return exactly 0.
    """

    def __init__(self, d_model: int, gen: torch.Generator):
        super().__init__()
        self.scale = d_model ** -0.5
        self.q = nn.Parameter(torch.empty(d_model))
        _normal_(self.q, 0.02, gen)
        self.W_k = _linear(d_model, d_model, False, gen, 0.02)
        self.W_v = _linear(d_model, d_model, False, gen, 0.02)
        self.W_o = _linear(d_model, d_model, False, gen, 0.02)
        self._ternary_exclude = True

    def forward(self, xs: Tensor, layout: SlotLayout) -> tuple[Tensor, Tensor]:
        """``xs`` ``[B, L, C]`` single-stream prelude states -> ``(e [B, S, C], ok [B, S])``."""
        B, L, C = xs.shape
        S = layout.slot_index.shape[1]
        k = self.W_k(xs)
        v = self.W_v(xs)
        sc = torch.einsum("c,blc->bl", self.q.to(k.dtype), k) * self.scale         # [B, L]
        tok = ~layout.slot_mask
        nxt = (layout.bag_id.unsqueeze(1) == (torch.arange(
            S, device=xs.device) + 1).view(1, S, 1)) & tok.unsqueeze(1)          # [B, S, L]
        has = nxt.any(dim=-1)
        # An all-(-inf) row would give NaN in the forward AND the backward; open it to
        # everything and zero the result below (the TULCodeEncoder guard).
        nxt = torch.where(has.unsqueeze(-1), nxt, torch.ones_like(nxt))
        logits = sc.float().unsqueeze(1).masked_fill(~nxt, float("-inf"))        # [B, S, L]
        a = torch.softmax(logits, dim=-1).to(v.dtype)
        pooled = torch.einsum("bsl,blc->bsc", a, v)
        out = self.W_o(pooled).float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-6)
        ok = gram_has_next(layout) & has
        return (out * ok.unsqueeze(-1).float()).to(xs.dtype), ok


def gram_kl(m_q: Tensor, s_q: Tensor, m_p: Tensor, s_p: Tensor) -> Tensor:
    """Closed-form ``KL(N(m_q, s_q^2) || N(m_p, s_p^2))`` per element, fp32.

    Means and stds in units of ``r`` (the shared per-slot scale cancels).

    Written as ``0.5 * (e^{2d} - 1 - 2d) + 0.5 * (m_q - m_p)^2 / s_p^2`` with
    ``d = log s_q - log s_p``, and ``e^x - 1 - x`` by its series below ``|x| = 1e-2``. The
    textbook form ``log(s_p/s_q) + (s_q^2 + dm^2) / (2 s_p^2) - 1/2`` subtracts terms of
    order 0.5 to get a KL of order d^2, and in fp32 returned EXACTLY 0 for the whole
    40-step mean-free smoke while the heads were moving (2026-09-23): a sigma-only KL
    below ~1e-8 per channel vanishes in the cancellation."""
    m_q, s_q, m_p, s_p = m_q.float(), s_q.float(), m_p.float(), s_p.float()
    x = 2.0 * (torch.log(s_q) - torch.log(s_p))
    small = x * x * (0.5 + x * (1.0 / 6.0 + x / 24.0))
    em1 = torch.where(x.abs() < 1e-2, small, torch.expm1(x) - x)
    return 0.5 * em1 + 0.5 * (m_q - m_p).pow(2) / s_p.pow(2)


def gram_kl_balanced(m_q: Tensor, s_q: Tensor, m_p: Tensor, s_p: Tensor,
                     alpha: float) -> tuple[Tensor, Tensor]:
    """``(balanced, raw)`` per element: ``alpha * KL(sg(q) || p) + (1 - alpha) * KL(q ||
    sg(p))`` (DreamerV2 / GRAM App. B.2) and the plain KL. Both have the same VALUE; only
    the balanced one's gradient is split between the two sides."""
    kp = gram_kl(m_q.detach(), s_q.detach(), m_p, s_p)
    kq = gram_kl(m_q, s_q, m_p.detach(), s_p.detach())
    return alpha * kp + (1.0 - alpha) * kq, kq.detach()


class TULGramStep(nn.Module):
    """The two noise heads and the posterior's pool. See the module docstring."""

    def __init__(self, d_model: int, *, hidden: int, sigma_init: float, use_mean: bool,
                 with_posterior: bool = True):
        super().__init__()
        if hidden < 1:
            raise ValueError(f"TULGramStep needs hidden >= 1, got {hidden}")
        if not sigma_init > GRAM_SIGMA_FLOOR:
            raise ValueError(
                f"tul.gram_sigma_init must be > {GRAM_SIGMA_FLOOR} (the sigma floor), got "
                f"{sigma_init}")
        self.use_mean = bool(use_mean)
        self.with_posterior = bool(with_posterior)
        _rng0 = torch.random.get_rng_state()
        try:
            g = torch.Generator(device="cpu").manual_seed(_SEED_GRAM)
            b_s = _softplus_inv(float(sigma_init) - GRAM_SIGMA_FLOOR)
            # The pool draws FIRST from the private generator. Without a posterior it is
            # still constructed (and dropped), so the prior heads below take the same
            # draws and equal an "elbo" model's prior heads bit for bit.
            _pool = TULGramPool(d_model, g)
            self.pool = _pool if self.with_posterior else None
            self.prior_s = _GramHead(d_model, hidden, d_model, g, out_bias=b_s)
            self.prior_m = (_GramHead(d_model, hidden, d_model, g, out_bias=0.0)
                            if self.use_mean else None)
            self.post_s = self.post_m = None
            if self.with_posterior:
                self.post_s = _GramHead(d_model, hidden, d_model, g, out_bias=b_s,
                                        d_extra=d_model)
                self.post_s.copy_shared_from(self.prior_s)
                if self.use_mean:
                    self.post_m = _GramHead(d_model, hidden, d_model, g, out_bias=0.0,
                                            d_extra=d_model)
                    self.post_m.copy_shared_from(self.prior_m)
        finally:
            torch.random.set_rng_state(_rng0)
        self._ternary_exclude = True

    @staticmethod
    def _sigma(raw: Tensor) -> Tensor:
        return F.softplus(raw.float()) + GRAM_SIGMA_FLOOR

    def prior(self, uh: Tensor) -> tuple[Tensor, Tensor]:
        """``uh`` ``[B, S, C]`` (= ``u / r``) -> ``(m, s)`` fp32, in units of ``r``."""
        s = self._sigma(self.prior_s(uh))
        m = (self.prior_m(uh).float() if self.prior_m is not None
             else torch.zeros_like(s))
        return m, s

    def posterior(self, uh: Tensor, e_next: Tensor) -> tuple[Tensor, Tensor]:
        if not self.with_posterior:
            raise RuntimeError(
                "TULGramStep.posterior on a step built without one (tul.gram_objective="
                "'iw'): the multi-sample objective trains on prior rollouts only.")
        s = self._sigma(self.post_s(uh, e_next))
        m = (self.post_m(uh, e_next).float() if self.post_m is not None
             else torch.zeros_like(s))
        return m, s


def iw_span_groups(bag_id: Tensor, n_slots: int) -> Tensor:
    """``[B, L]`` int64 group index ``row * (n_slots + 1) + bag_id``: one group per (row,
    span), the dump bin ``bag_id == n_slots`` (tokens past the row's last slot) its own
    group. On a packed TUL row the token positions of one ``bag_id`` are one contiguous
    run, so this is the probe's ``bayes_read`` grouping (contiguous runs of one span id)."""
    B = bag_id.shape[0]
    off = torch.arange(B, device=bag_id.device).unsqueeze(1) * (n_slots + 1)
    return bag_id + off


def iw_span_bound(lp: Tensor, w: Tensor, group: Tensor, n_groups: int
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """The multi-sample bound per span, summed. ``lp`` ``[K, N]`` per-position log p of the
    label under rollout k (any value where ``w == 0``), ``w`` ``[N]`` the position's CE
    weight (0 = unscored), ``group`` ``[N]`` its span id in ``[0, n_groups)``.

    Returns ``(bound_sum, S, scored)``: ``bound_sum = sum_g [logsumexp_k S_k(g) - log K]``
    over the groups with a scored position (a scalar with the graph), ``S`` ``[K, G]`` the
    per-rollout weighted span log-likelihoods, ``scored`` ``[G]`` bool. ``K = 1`` gives
    ``sum_j w_j lp_j`` exactly (logsumexp of one element is the element; log 1 = 0).

    The gradient of ``bound_sum`` into ``S_k(g)`` is ``softmax_k(S(g))`` (the credit), so
    into ``lp_k(j)`` it is ``w_j * softmax_k(S(g_j))``. fp32 throughout."""
    K = lp.shape[0]
    wf = w.float()
    S = lp.float().new_zeros(K, n_groups).index_add_(
        1, group, lp.float() * wf.unsqueeze(0))
    scored = torch.zeros(n_groups, dtype=wf.dtype, device=wf.device).index_add_(
        0, group, (wf != 0).to(wf.dtype)) > 0
    lme = torch.logsumexp(S, dim=0)
    if K > 1:
        lme = lme - math.log(K)
    bound_sum = torch.where(scored, lme, torch.zeros_like(lme)).sum()
    return bound_sum, S, scored
