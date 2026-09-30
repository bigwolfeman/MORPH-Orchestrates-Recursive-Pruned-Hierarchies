"""Three one-factor arms on the write-all fan with no WTA term ("nowta", 2026-09-30).

nowta (``tul_slot_spandec_strict_fan4_all_fp01_nowta.yaml``) writes all M loop cells of a
slot 1:1 into its M prefix cells and lets the coda's attention pick per token. Nothing
gives one cell a job the others do not have. Each arm below adds ONE thing:

F  ``tul.fan_opf: true`` — Orthogonal Predictive Factorization on the cells (JEPA-Anything,
   arXiv 2609.20800, Eqs. 2-10). Cell ``k`` predicts factor ``k`` of an EMA-prelude code of
   the NEXT span. The factors are an orthogonal split of that code (a projector ``P`` kept
   exactly orthonormal by a QR retraction after every optimizer step), so the M cells get
   M disjoint jobs. The prediction gradient reaches the cells jointly with the coda's
   token CE (multi-task, NOT the LCTUL-J shape where the latent term was the loop's only
   trainer and the coda read detached cells).
R  ``tul.fan_route: reader`` — a top-1 router (MoE style). The coda reads ONLY the
   router's winner, scaled by its gate ``p_winner`` (Switch Transformer), so the coda's
   token CE trains the router through ``p_winner`` and credits the winning cell alone.
T  ``tul.fan_route: latent`` — the same router and the same read, but trained by a latent
   TEACHER (the target of F) instead of by the reader. The read is hard (no ``p``), so
   the reader's CE sends nothing to the router.

This file holds the modules and the pure math. The wiring (where the target is built,
where the router picks, how the pick is written) is in ``MORPHTransformer`` —
``_tul_fan_target``, ``_tul_fan_opf``, ``_tul_fan_route``, ``_fan_route_cells`` and the
post-step hook ``tul_fan_after_step``. Tests: ``tests/test_tul_fan_opf.py``,
``tests/test_tul_fan_router.py``. Note:
``.agents/notes/proposed/architecture/2026-09-30-fan-opf-and-routers.md``.

Every module here is RNG-NEUTRAL at construction (built inside a forked stream with its
own seed), so every other weight of an arm equals the same-seed nowta model; and never
ternarised (``_ternary_exclude``), and invisible to prune / carve / route (no
``MortarLinear``). All the math runs in fp32 with autocast off: the heads are tiny and
the hinge terms compare standard deviations against 0.1, which bf16 would round.
"""
from __future__ import annotations

import contextlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_OPF_SEED = 0x0F0F_2609
_ROUTER_SEED = 0x2007_E6
_TEACHER_SEED = 0x7EAC_4E6

FAN_ROUTE_MODES = ("none", "reader", "latent")


@contextlib.contextmanager
def _cpu_seeded(seed: int):
    """Draw a module's init from a fixed CPU stream and leave EVERY generator as it was.

    FOOT GUN (hit 2026-09-30): `torch.manual_seed` inside `fork_rng(devices=[])` is NOT
    neutral. `manual_seed` also reseeds every CUDA generator (lazily, if CUDA is not yet
    initialised: the call is queued and runs at CUDA init, AFTER the trainer's own seed),
    and `fork_rng(devices=[])` restores the CPU generator only. On the GPU that moved
    every dropout mask and depth draw of arms F / R / T away from nowta's (step-0 token
    CE 11.2827 vs 11.2654) while every CPU test stayed bit-equal. Only the CPU default
    generator is seeded here; the model is built on CPU, so every init draw uses it.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        yield


def _canon(name: str) -> str:
    """A parameter / buffer name with every ``torch.compile`` wrapper segment removed.

    The trainer compiles the prelude's MLPs IN PLACE (``layer.mlp = torch.compile(...)``)
    AFTER the target twin is deep-copied, so the live names carry ``mlp._orig_mod.`` and
    the twin's do not. Matching by the canonical name keeps the two sides paired."""
    return name.replace("._orig_mod.", ".").replace("_orig_mod.", "")


def _ln(x: Tensor) -> Tensor:
    """LayerNorm over the last axis, no affine, fp32."""
    return F.layer_norm(x.float(), (x.shape[-1],))


# ── the EMA target encoder (arms F and T) ────────────────────────────────────────────


class FanTargetFront(nn.Module):
    """The EMA copy of the PRELUDE the target is computed with (paper Eq. 2).

    WHAT IS COPIED. The prelude's per-layer compute: ``prelude`` (the blocks),
    ``x0_injects[:n_prelude]`` (each prelude layer's x0 injection projection) and
    ``value_embeds`` (the value-embedding projections, which feed prelude layers only).
    WHAT IS SHARED, NOT COPIED. Every lookup TABLE ``_tul_front`` reads — the token
    embedding (weight-tied to the LM head), the hash-bigram table and its per-layer
    lambdas, the value-embedding tables and the TUL slot inputs. They are read LIVE,
    under ``no_grad``. Two reasons: they are 50M-row tables each (three value-embedding
    tables plus the token and bigram tables would be ~1 GB of fp32 EMA state on a card
    with ~1 GB of slack), and the token table is trained by the LM loss through the tied
    head, so it is not the online encoder's own weight in the JEPA sense. Stated, not
    hidden: the target therefore moves with the live tables at once and with the blocks
    at EMA speed.

    NOT A REGISTERED SUBMODULE of the model (the ``tul.code_target_ref`` precedent): every
    walk in this tree — the ternary QAT pass, CMS prune / carve / route, the optimizer, the
    gradient probes — enumerates modules or parameters, and a registered twin would be
    walked, trained or pruned. Consequences, each a foot gun:
      * the model's ``.to()`` / ``.float()`` / ``.train()`` do NOT reach it. It is built
        on the live model's device and dtype and is always in eval mode (no dropout);
      * the trainer builds it AFTER quantisation (so its weights carry the same ternary
        parametrisation as the live prelude) and BEFORE torch.compile (so its blocks run
        eager), and re-syncs it after ``init_from`` / resume;
      * it rides its OWN checkpoint key (``fan_target``), so a resume keeps the EMA state
        instead of re-snapshotting the live weights (which would jump the target).
    Its parameters have ``requires_grad`` False; the forward that uses it runs under
    ``no_grad``.
    """

    def __init__(self, prelude: nn.ModuleList, x0_injects: nn.ModuleList,
                 value_embeds: nn.ModuleList):
        super().__init__()
        self.prelude = copy.deepcopy(prelude)
        self.x0_injects = nn.ModuleList([copy.deepcopy(m) for m in x0_injects])
        self.value_embeds = copy.deepcopy(value_embeds)
        for mod in self.modules():
            # Hooks installed on the live modules for instruments (saliency, probes)
            # would otherwise fire from the twin and record ITS activations as the live
            # model's. The forward of a MORPHBlock does not rely on hooks.
            mod._forward_hooks.clear()
            mod._forward_pre_hooks.clear()
            mod._backward_hooks.clear()
        for prm in self.parameters():
            prm.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """Always eval: the target is deterministic (no dropout draws, no RNG use)."""
        return super().train(False)

    @staticmethod
    def live_tensors(model: nn.Module, n_prelude: int) -> dict[str, Tensor]:
        """``{canonical name: tensor}`` of the live modules the twin mirrors, every
        parameter and buffer, in the twin's own naming."""
        out: dict[str, Tensor] = {}
        groups = (("prelude", model.prelude),
                  ("x0_injects", nn.ModuleList(list(model.x0_injects)[:n_prelude])),
                  ("value_embeds", model.value_embeds))
        for gname, mod in groups:
            for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
                out[_canon(f"{gname}.{n}")] = t
        return out

    def own_tensors(self) -> dict[str, Tensor]:
        return {_canon(n): t for n, t in
                list(self.named_parameters()) + list(self.named_buffers())}

    def _pairs(self, model: nn.Module, n_prelude: int) -> list[tuple[Tensor, Tensor]]:
        """``(twin, live)`` pairs by name, RECOMPUTED on every call: a structure change
        on the live side (a carve replaces tensors) must fail here, not leave the twin
        tracking tensors the live model no longer uses."""
        live = self.live_tensors(model, n_prelude)
        own = self.own_tensors()
        if live.keys() != own.keys():
            raise RuntimeError(
                "FanTargetFront: the live prelude and the EMA twin no longer name the same "
                f"tensors. Missing from the twin: {sorted(live.keys() - own.keys())[:6]}. "
                f"Extra in the twin: {sorted(own.keys() - live.keys())[:6]}. A carve or a "
                "re-parametrisation after the twin was built changes the live structure; "
                "the fan target arms do not compose with a schedule that does that.")
        return [(own[k], live[k]) for k in sorted(own)]

    @torch.no_grad()
    def ema_update_(self, model: nn.Module, n_prelude: int, m: float) -> None:
        """``theta_twin <- m * theta_twin + (1 - m) * theta_live`` (paper Eq. 2).

        Float tensors lerp; integer / bool buffers (masks, counters) are COPIED, they have
        no meaningful average. Called once per optimizer step, after the step."""
        if not 0.0 <= m < 1.0:
            raise ValueError(f"FanTargetFront.ema_update_: m must be in [0, 1), got {m}")
        for twin, live in self._pairs(model, n_prelude):
            if twin.is_floating_point():
                twin.lerp_(live.detach().to(twin.dtype), 1.0 - m)
            else:
                twin.copy_(live)

    @torch.no_grad()
    def sync_from_(self, model: nn.Module, n_prelude: int) -> None:
        """Copy the live weights into the twin (m = 0): the snapshot after the weights
        load, when the checkpoint carries no twin state of its own."""
        for twin, live in self._pairs(model, n_prelude):
            twin.copy_(live)


def pooled_span_states(x: Tensor, gid: Tensor, keep_tok: Tensor, g_bins: int) -> Tensor:
    """``[B, S, C]`` fp32: the mean of ``x`` over the TOKEN positions of each slot's next
    span, then LayerNorm without affine (the target's space).

    ``x`` is the prelude's output ``[B, L, (n,) C]``; an HC stream axis is reduced by its
    mean first. The bins are :func:`~morph.model.transformer.span_ce_index`'s (``gid``,
    ``keep_tok``, ``g_bins``), so index ``s`` of the result pools bin ``s + 1`` — exactly
    the span the coda's span-CE table scores for slot ``s``. A slot whose next span has no
    scored token pools nothing (zero sum, LayerNorm of 0 is 0) and must be excluded by
    the caller's ``ok`` mask."""
    xs = x.float()
    while xs.dim() > 3:
        xs = xs.mean(dim=-2)
    B, L, C = xs.shape
    w = keep_tok.reshape(-1, 1).to(xs.dtype)
    sums = xs.new_zeros(B * g_bins, C).index_add_(0, gid.reshape(-1), xs.reshape(-1, C) * w)
    cnt = xs.new_zeros(B * g_bins).index_add_(0, gid.reshape(-1),
                                              keep_tok.reshape(-1).to(xs.dtype))
    pooled = (sums / cnt.clamp_min(1.0).unsqueeze(-1)).view(B, g_bins, C)[:, 1:]
    return _ln(pooled)


def _wview(w: Tensor, x: Tensor) -> Tensor:
    return w.float().view(-1, *([1] * (x.dim() - 1)))


def batch_var(x: Tensor, w: Tensor) -> Tensor:
    """Per-coordinate POPULATION variance over the leading axis of ``x`` ``[N, ...]``,
    counting row ``n`` with weight ``w[n]`` in {0, 1} (the valid slots). fp32. Masked
    by weights, not by indexing, so it costs no host sync on the training path."""
    xf, ww = x.float(), _wview(w, x)
    n = ww.sum().clamp_min(1.0)
    mean = (xf * ww).sum(dim=0, keepdim=True) / n
    return ((xf - mean).square() * ww).sum(dim=0) / n


def batch_std(x: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    """``sqrt(Var + eps)`` per coordinate (paper Sec. 2.3), :func:`batch_var`'s mask."""
    return torch.sqrt(batch_var(x, w) + eps)


def participation_ratio(x: Tensor, w: Tensor) -> Tensor:
    """Effective rank of the batch covariance of the rows of ``x`` ``[N, C]`` with
    ``w[n] = 1``: ``(sum lambda)^2 / sum lambda^2`` = ``tr(G)^2 / ||G||_F^2`` on the
    centred Gram matrix ``G`` (the covariance's nonzero eigenvalues, ``N x N`` instead of
    ``C x C``; a zero-weight row is a zero row of ``G``). 1 = one direction,
    ``min(N-1, C)`` = isotropic."""
    xf, ww = x.float(), _wview(w, x)
    n = ww.sum().clamp_min(1.0)
    xc = (xf - (xf * ww).sum(dim=0, keepdim=True) / n) * ww
    g = xc @ xc.t()
    return g.diagonal().sum().square() / g.square().sum().clamp_min(1e-12)


# ── arm F: Orthogonal Predictive Factorization ───────────────────────────────────────


class FanOPF(nn.Module):
    """The projector ``P`` and the K = M per-cell predictors ``q_k`` (paper Eqs. 3-4).

    ``P`` ``[d, d]`` stores the ANALYSIS ROWS: rows ``k*r .. (k+1)*r - 1`` are ``P_k^T``,
    so factor ``k`` of a target ``z`` is ``z^(k) = P_k^T z`` (the paper's ``(K, r, d)``
    basis, flattened). It starts at the identity (the paper code's default, already
    orthonormal) and is trained by ``L_pred`` and ``L_fac`` — the target's stop-gradient
    sits on the EMA encoder's OUTPUT, not on ``P`` (``factor_prediction_loss``'s rule).
    After every optimizer step :meth:`retract_` replaces it by the Q of its QR
    factorisation (the paper's strict mode), so it is orthonormal to fp32 round-off at
    every forward and no ``L_orth`` term is needed (``lambda_orth`` would multiply an
    exact zero). The optimizer's moments are NOT rewritten (the paper code's choice).

    ``q_k`` (the arm's own choice, the spec left it open): LayerNorm (no affine) ->
    ``Linear(d, h)`` -> GELU -> ``Linear(h, r)``, one per cell, batched as stacked
    weights. Why an MLP and not ``d -> r``: the cell lives in the loop's carrier space,
    which the coda reads through ``W_prefix`` and attention; a linear head would force the
    target coordinates to be LITERAL linear coordinates of that carrier and compete with
    the coda's read for them, where one hidden layer lets the predictor decode. Why the
    LayerNorm: the carrier's scale is not fixed (the loop's gain moves it), and the target
    is LayerNormed. The gradient still reaches the cell through the norm.
    """

    def __init__(self, d_model: int, k: int, hidden: int):
        super().__init__()
        d, k = int(d_model), int(k)
        if k < 2 or d % k != 0:
            raise ValueError(f"FanOPF needs k >= 2 factors that divide d_model: d={d}, k={k}")
        h = int(hidden) if int(hidden) > 0 else d
        self.k, self.r, self.h = k, d // k, h
        self.P = nn.Parameter(torch.eye(d))
        with _cpu_seeded(_OPF_SEED):
            b1, b2 = 1.0 / d ** 0.5, 1.0 / h ** 0.5   # nn.Linear's default bound
            self.w1 = nn.Parameter(torch.empty(k, d, h).uniform_(-b1, b1))
            self.b1 = nn.Parameter(torch.zeros(k, h))
            self.w2 = nn.Parameter(torch.empty(k, h, self.r).uniform_(-b2, b2))
            self.b2 = nn.Parameter(torch.zeros(k, self.r))
        self._ternary_exclude = True

    def factors(self, z: Tensor) -> Tensor:
        """``[N, d]`` -> ``[N, K, r]``: ``z^(k) = P_k^T z``. Gradient reaches ``P``;
        the caller hands in a DETACHED ``z``."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            return (z.float() @ self.P.float().t()).view(z.shape[0], self.k, self.r)

    def predict(self, cells: Tensor) -> Tensor:
        """``[N, K, d]`` cell states (cell ``k`` in slot ``k``) -> ``[N, K, r]``."""
        with torch.autocast(device_type=cells.device.type, enabled=False):
            a = F.gelu(torch.einsum("nkc,kch->nkh", _ln(cells), self.w1.float())
                       + self.b1.float())
            return torch.einsum("nkh,khr->nkr", a, self.w2.float()) + self.b2.float()

    @torch.no_grad()
    def retract_(self) -> None:
        """``P <- Q`` of ``P^T = QR`` with QR's column-sign ambiguity canonicalised
        (``diag R > 0``), exactly ``opf.orthonormalize_basis``: the rows of the result are
        orthonormal and a retraction of an orthonormal ``P`` returns ``P``."""
        w = self.P.detach().float()
        q, r = torch.linalg.qr(w.t(), mode="reduced")
        sign = torch.where(torch.diagonal(r) < 0, -1.0, 1.0).to(q.dtype)
        self.P.copy_((q * sign.unsqueeze(0)).t().to(self.P.dtype))

    @torch.no_grad()
    def orth_err(self) -> Tensor:
        """``max |P P^T - I|``, fp32 — ~1e-6 after a retraction."""
        w = self.P.detach().float()
        return (w @ w.t() - torch.eye(w.shape[0], device=w.device)).abs().amax()


def opf_terms(q: Tensor, zf: Tensor, zo: Tensor, w: Tensor, gamma_fac: float,
              gamma_enc: float, eps: float = 1e-6) -> tuple[Tensor, Tensor, Tensor, dict]:
    """``(L_pred, L_fac, L_enc, stats)`` over the slots with ``w = 1`` (paper Eqs. 6, 9).

    ``q`` ``[N, K, r]`` the cells' predictions (grad to the cells and ``q_k``);
    ``zf`` ``[N, K, r]`` the projected target ``P_k^T sg(z)`` (grad to ``P`` only);
    ``zo`` ``[N, d]`` the ONLINE prelude's pooled span states in the target's space (grad
    to the online prelude); ``w`` ``[N]`` in {0, 1}, the valid slots. ``L_pred`` is the
    mean over valid slots, factors and coordinates of the squared error; ``L_fac`` /
    ``L_enc`` the mean hinge ``max(0, gamma - std)`` over each projected target coordinate
    / each online coordinate, the std over the valid slots.

    FOOT GUN: with fewer than 2 valid slots every std is ``sqrt(eps)``, the hinges read
    ``gamma`` (a false collapse) and carry zero gradient. A packed 1024-token row holds
    ~50 slots, so this does not happen on a real batch; a unit test with one slot would
    read it.

    Stats (detached): per-factor ``R^2 = 1 - MSE_k / Var_k`` (``Var_k`` the mean over the
    factor's coordinates of their batch variance; 0 predicts the batch mean, 1 is
    perfect), the online encoder's min and mean coordinate std, the smallest projected
    target coordinate std."""
    wf = w.float()
    n = wf.sum().clamp_min(1.0)
    err = (q - zf).square()                                             # [N, K, r]
    l_pred = (err.mean(dim=(1, 2)) * wf).sum() / n
    s_fac = batch_std(zf, wf, eps)                                      # [K, r]
    l_fac = torch.relu(gamma_fac - s_fac).mean()
    s_enc = batch_std(zo, wf, eps)                                      # [d]
    l_enc = torch.relu(gamma_enc - s_enc).mean()
    with torch.no_grad():
        var_k = batch_var(zf.detach(), wf).mean(dim=-1)                  # [K]
        mse_k = (err.detach().mean(dim=2) * wf.unsqueeze(-1)).sum(dim=0) / n   # [K]
        r2 = 1.0 - mse_k / var_k.clamp_min(1e-12)
        stats = {"opf_enc_std_min": s_enc.detach().amin(),
                 "opf_enc_std_mean": s_enc.detach().mean(),
                 "opf_fac_std_min": s_fac.detach().amin()}
        for i in range(int(r2.shape[0])):
            stats[f"opf_r2_k{i}"] = r2[i]
        stats["opf_r2_mean"] = r2.mean()
    return l_pred, l_fac, l_enc, stats


# ── arms R and T: the top-1 router ───────────────────────────────────────────────────


class FanRouter(nn.Module):
    """A lightning-indexer-style scorer over the M cells of a slot, a load-balance bias,
    and the gate (arms R and T).

    ``score_i = v^T ReLU(W_c LN(cell_i) + W_x LN(ctx))`` (DeepSeek-V3.2's indexer form,
    ``rank`` = the low width). ``ctx`` is the slot's loop INPUT: the normed prelude output
    at the slot's first prefix position, the state every cell of the slot is seeded from.
    Why that and not a cell's register seed: the seed differs per cell (the register adds
    a per-cell trigger), so it would hand the router each cell's identity as "context";
    the loop input is one vector per slot, so the M cells are ranked in the SAME context.
    The caller passes ``ctx`` DETACHED: the router's conditioning, not a new gradient path
    into the prelude.

    ``bias`` ``[M]`` (a persistent BUFFER, checkpointed; DeepSeek-V3's auxiliary-loss-free
    balancing): added to the scores for the SELECTION only, never to ``p``. After every
    optimizer step :meth:`balance_step_` moves ``b_i += u * sign(mean_load - load_i)``,
    ``load`` the winner share over the valid slots of the forwards of that step
    (accumulated in the non-persistent ``pending`` buffer by :meth:`record_load`, only in
    training forwards with grad enabled).

    ``v`` starts small and random (not zero): a zero ``v`` makes every score 0, every
    selection a tie that ``argmax`` breaks to cell 0, and the bias then oscillates one
    cell at a time.
    """

    def __init__(self, d_model: int, m: int, rank: int):
        super().__init__()
        d, m, rank = int(d_model), int(m), int(rank)
        if m < 2 or rank < 1:
            raise ValueError(f"FanRouter needs m >= 2 cells and rank >= 1, got {m}, {rank}")
        self.m = m
        with _cpu_seeded(_ROUTER_SEED):
            self.W_c = nn.Linear(d, rank, bias=False)
            self.W_x = nn.Linear(d, rank, bias=True)
            self.v = nn.Parameter(torch.randn(rank) / rank ** 0.5)
        self.register_buffer("bias", torch.zeros(m))
        self.register_buffer("pending", torch.zeros(m), persistent=False)
        self._ternary_exclude = True
        for mod in (self.W_c, self.W_x):
            mod._ternary_exclude = True

    def scores(self, cells: Tensor, ctx: Tensor) -> Tensor:
        """``cells`` ``[B, S, M, C]``, ``ctx`` ``[B, S, C]`` -> ``[B, S, M]`` fp32."""
        with torch.autocast(device_type=cells.device.type, enabled=False):
            a = (F.linear(_ln(cells), self.W_c.weight.float())
                 + F.linear(_ln(ctx), self.W_x.weight.float(),
                            self.W_x.bias.float()).unsqueeze(2))
            return torch.relu(a) @ self.v.float()

    def select(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        """``(winner [B, S] int64, p [B, S, M])``: ``argmax(score + bias)`` (the bias
        chooses, it never enters ``p``) and ``softmax(score)``. The SAME function at train
        and at eval: no noise, no temperature, no train-only branch."""
        winner = (scores.detach() + self.bias.to(scores.dtype)).argmax(dim=-1)
        return winner, torch.softmax(scores, dim=-1)

    @torch.no_grad()
    def record_load(self, winner: Tensor, valid: Tensor) -> None:
        """Accumulate the winner counts of the VALID slots for the next balance step."""
        oh = F.one_hot(winner, self.m).to(self.pending.dtype) * valid.unsqueeze(-1)
        self.pending.add_(oh.reshape(-1, self.m).sum(dim=0))

    @torch.no_grad()
    def balance_step_(self, u: float) -> None:
        """``bias_i += u * sign(mean_load - load_i)`` from the accumulated counts, then
        clear them. No counts (no training forward since the last step): no move."""
        total = self.pending.sum()
        load = self.pending / total.clamp_min(1.0)
        step = u * torch.sign(1.0 / self.m - load)
        self.bias.add_(torch.where(total > 0, step, torch.zeros_like(step)))
        self.pending.zero_()


class FanTeacherMap(nn.Module):
    """Arm T's map ``g``: LayerNorm (no affine) -> ``Linear(d, d)``. Reads each cell
    DETACHED and is trained by MSE onto the detached target, so it learns to map ANY cell
    into the target's space and sends no gradient into the loop."""

    def __init__(self, d_model: int):
        super().__init__()
        with _cpu_seeded(_TEACHER_SEED):
            self.g = nn.Linear(int(d_model), int(d_model))
        self._ternary_exclude = True
        self.g._ternary_exclude = True

    def forward(self, cells: Tensor) -> Tensor:
        with torch.autocast(device_type=cells.device.type, enabled=False):
            return F.linear(_ln(cells), self.g.weight.float(), self.g.bias.float())
