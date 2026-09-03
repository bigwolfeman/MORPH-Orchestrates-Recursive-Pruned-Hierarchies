"""dmorph — the no-loop MORPH's DiffusionBlocks-routed rectified-flow noisy stream.

Design: ``.agents/notes/proposed/architecture/2026-09-03-dmorph-v1.md``. Prereg:
``lab/experiments/planned/2026-09-03-dmorph-v1-panel.md``. The audits every number
below comes from live in ``lab/dmorph/research/``.

What this is, in one paragraph. The backbone is a FLAT 12-layer MORPH (``n_core == 0``)
over the packed TUL row. The CLEAN stream is the shipped forward and its head is the
metric. A SECOND stream over the same row carries a rectified-flow state
``x_t = (1 - t)·x0 + t·y`` at a flow time ``t`` drawn once per row, and is routed
through ONE block of the flat stack chosen by ``t`` (``n_blocks`` equal-width bands;
low ``t`` = mostly noise = coarse work = the EARLY block — the paper's σ→block reversal
in ``t`` coordinates, arXiv 2506.14202 §3.3 / ``model.py::estimate_target_layer``).
Inside its block the noisy stream runs the SAME layers and weights as the clean stream,
with two differences: AdaLN-Zero time conditioning on every layer (the
``CoreStageConditioning`` pattern ported from ``d9e04e6:morph/model/iter_cond.py``), and
attention whose keys/values are the CLEAN stream's, captured during the clean pass
(``cla_capture`` → ``cla_kv`` in ``morph/model/attention.py``). The clean stream never
sees the noisy stream, so the clean head cannot leak the target (the testbed's
``tests/test_no_leak.py`` perturbation proof, ported to ``tests/test_dmorph_no_leak.py``).

The two arms differ ONLY in the target ``y`` (``DmorphConfig.arm``):

* ``"tok"`` — ``y_i = normalize(E[label_i])`` at every position with a label: the
  paper's autoregressive DiffusionBlocks target, FM-parameterised. The stream covers the
  whole row.
* ``"hs"``  — ``y_s = normalize(stopgrad(h_s^final))``, the clean stream's own post-stack
  state at the SLOT positions (after ``_back_region``'s final norm, before the head): the
  "post-core carriers" cell the FM-planner rejection left open
  (``lab/dmorph/research/2026-09-03-fm-planner-arc.md`` §E). v1 runs it on full-row
  tensors with token positions masked out of both loss terms.

Rules written in blood that this module obeys (each cites its scar):

* Targets are UNIT L2, never ``SliceScaler`` — per-component-std scaling put σ* at 3.30
  and 77–98 % of training into autoencoding
  (``2026-09-03-db-in-morph-audit.md`` B.1; ``db-testbed-ladder.md`` B).
* The source is MATCHED: ``x0 ~ N(0, s²I)`` with ``s = 1/sqrt(d)`` for unit-L2 targets, so
  ``E‖v‖² = E‖y‖² + d·s² = 2`` and not ``1 + d`` (``fm_planner.py`` / DeepWeightFlow App. H;
  ``fm-planner-arc.md`` §A).
* The FM term is divided by the analytic null floor (``loss_scale: auto``) so it starts
  near 1 and does not drown a 4–11 nat CE (``tul_fm.py`` header, quoted in
  ``fm-planner-arc.md`` §A).
* The CE term through the tied head from ``D̂`` is UNWEIGHTED (EDM's ``w(σ)`` on CE
  over-weights the trivial low-noise region ~45,000× and collapses training;
  ``db-in-morph-audit.md`` B.4). It is the testbed's Option B, the only DB variant that
  ever used context (B.5, A3).
* The targets are DETACHED in both arms by default. The tied embedding is BOTH the
  regression target and a parameter, and a live target let the loss move the target
  toward the prediction (directional embedding collapse, ``db-in-morph-audit.md`` A1/
  ``diffusion_blocks.py`` ``ce_anchor_lambda`` note). ``hs`` may run LIVE targets only
  with ``sigreg_lambda > 0`` (LeJEPA's collapse guard, ``tul_fm.py::fm_sigreg_loss``).
* The inference bridge is HARD (the argmax row of the table, full norm). The softmax
  expectation shrinks ``‖D̂‖`` to 0.27 at high noise and cost an order of magnitude of
  gen-PPL (``db-testbed-ladder.md`` B: 584 → 17).
* ONE road from ``t`` to a layer group: :func:`band_of_t` → :func:`block_layers`, used by
  training, eval and the generator alike. The testbed lost a 573M-token run to two
  σ→block functions that disagreed (``db-testbed-fidelity.md``, "block reversal").

Nothing here branches on a runtime flag: :class:`DmorphStream` is built ONLY when
``MORPHConfig.dmorph`` is set (construction time, last in ``MORPHTransformer.__init__``),
and ``dmorph=None`` never reaches this module at all.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import RMSNorm
from .diffusion_blocks import AdaLNGate, SigmaConditioning
from .fused_ce import fused_linear_cross_entropy

__all__ = [
    "ARMS",
    "DmorphConfig",
    "DmorphStream",
    "DmCtx",
    "argmax_head",
    "band_bounds",
    "band_of_t",
    "block_layers",
    "fm_euler_step",
    "hard_bridge",
    "ladder",
    "noisy_stream",
    "sample_blocks",
    "sample_t_in_band",
    "training_terms",
]

ARMS = ("tok", "hs")
LOSS_SCALES = ("auto", "none")


@dataclass
class DmorphConfig:
    """Construction-time dmorph settings. Mirrors the Hydra ``dmorph:`` block.

    Every float here is RESOLVED (``morph/training/dmorph_setup.py`` turns ``"matched"``
    / ``"auto"`` into numbers and puts both the raw and the resolved value in the wandb
    manifest), so the model never sees a string it has to interpret.
    """

    arm: str = "tok"                 # "tok" | "hs"
    n_blocks: int = 4                # B equal-width bands on t, B contiguous layer blocks
    gamma: float = 0.1               # band overlap, fraction of the band width each side (paper App. C)
    lambda_fm: float = 1.0           # weight of the flow-matching term
    lambda_ce: float = 1.0           # weight of the CE-through-D̂ term (Option B)
    source_std: float = 0.03125      # s in x0 ~ N(0, s²I); the matched value is 1/sqrt(d)
    detach_ctx: bool = False         # True: the FM gradient never reaches the clean stream
    t_per_position: bool = False     # one t per row (default) or one per position
    sigreg_lambda: float = 0.0       # hs only; > 0 keeps the target LIVE and adds SIGReg
    sigreg_slices: int = 1024
    infer_steps: int = 0             # Euler steps of the ladder; 0 → n_blocks
    cond_dim: int = 256              # width of the t embedding fed to AdaLN
    t_embed_scale: float = 16.0      # t·scale enters the sinusoidal basis (fm_planner.py note)
    in_gain: float = 32.0            # x_t·in_gain enters the block; the resolved sqrt(d)
    loss_scale: str = "auto"         # "auto": divide the FM term by the analytic null floor
    block_visit: tuple[float, ...] | None = None   # training visit distribution; None = uniform

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"dmorph.arm must be one of {ARMS}, got {self.arm!r}")
        if self.n_blocks < 1:
            raise ValueError(f"dmorph.n_blocks must be ≥ 1, got {self.n_blocks}")
        if not 0.0 <= self.gamma < 0.5:
            raise ValueError(f"dmorph.gamma must be in [0, 0.5), got {self.gamma}")
        if self.lambda_fm < 0.0 or self.lambda_ce < 0.0:
            raise ValueError("dmorph.lambda_fm / lambda_ce must be ≥ 0")
        if self.source_std <= 0.0:
            raise ValueError(f"dmorph.source_std must be > 0, got {self.source_std}")
        if self.sigreg_lambda < 0.0:
            raise ValueError(f"dmorph.sigreg_lambda must be ≥ 0, got {self.sigreg_lambda}")
        if self.sigreg_lambda > 0.0 and self.arm != "hs":
            raise ValueError(
                "dmorph.sigreg_lambda > 0 is an hs-only switch (a live hidden-state target "
                "under LeJEPA's guard); the tok target is the embedding table and stays "
                "detached (the 2026-08-19 directional-collapse finding).")
        if self.infer_steps < 0:
            raise ValueError(f"dmorph.infer_steps must be ≥ 0, got {self.infer_steps}")
        if self.cond_dim < 8:
            raise ValueError(f"dmorph.cond_dim must be ≥ 8, got {self.cond_dim}")
        if self.t_embed_scale <= 0.0:
            raise ValueError(f"dmorph.t_embed_scale must be > 0, got {self.t_embed_scale}")
        if self.in_gain <= 0.0:
            raise ValueError(f"dmorph.in_gain must be > 0, got {self.in_gain}")
        if self.loss_scale not in LOSS_SCALES:
            raise ValueError(
                f"dmorph.loss_scale must be one of {LOSS_SCALES}, got {self.loss_scale!r}")
        if self.block_visit is not None:
            v = tuple(float(x) for x in self.block_visit)
            if len(v) != self.n_blocks or any(x <= 0.0 for x in v):
                raise ValueError(
                    f"dmorph.block_visit needs {self.n_blocks} positive entries, got {v}")
            s = sum(v)
            object.__setattr__(self, "block_visit", tuple(x / s for x in v))

    @property
    def n_infer_steps(self) -> int:
        return self.infer_steps if self.infer_steps > 0 else self.n_blocks


# ── the ONE road from t to a layer group ─────────────────────────────────────

def band_of_t(t: Tensor, n_blocks: int) -> Tensor:
    """``t ∈ [0, 1]`` → band index in ``[0, n_blocks)``; equal-width bands.

    Uniform ``t`` makes equal-width equal-mass, which is the paper's equi-probability
    partition in ``t`` coordinates. Band 0 is ``t ∈ [0, 1/B)`` — mostly noise, the
    coarsest work — and runs the EARLIEST layers (:func:`block_layers`). ``t == 1``
    lands in the last band.
    """
    return (t.float() * n_blocks).floor().long().clamp_(0, n_blocks - 1)


def block_layers(block: int, layers_per_block: int) -> range:
    """Global layer indices of block ``b``: ``[b·k, (b+1)·k)``, ``k = n_layers / B``.

    Index ``gi < n_prelude`` is ``prelude[gi]``, otherwise ``coda[gi - n_prelude]``
    (the flat stack has no core). Used by :func:`noisy_stream` and asserted by the
    layer-call spy in ``tests/test_dmorph_band_block.py``.
    """
    return range(block * layers_per_block, (block + 1) * layers_per_block)


def band_bounds(block: int, n_blocks: int, gamma: float) -> tuple[float, float]:
    """``(lo, hi)`` of band ``b`` on ``t``, widened by ``gamma`` of its width each side
    and clipped to ``[0, 1]`` — the paper's overlap (App. C, γ = 0.1 for text), in ``t``.
    """
    w = 1.0 / n_blocks
    lo = max(0.0, block * w - gamma * w)
    hi = min(1.0, (block + 1) * w + gamma * w)
    return lo, hi


def sample_blocks(shape, n_blocks: int, visit: tuple[float, ...] | None, device,
                  generator: torch.Generator | None = None) -> Tensor:
    """Block indices of ``shape`` drawn from the visit distribution (uniform by default).

    Block FIRST, then ``t`` inside the block's widened band
    (:func:`sample_t_in_band`) — the authors' order (``model.py::get_sigmas``,
    ``db-testbed-fidelity.md``), which is what gives the γ overlap a meaning.
    """
    n = 1
    for s in shape:
        n *= int(s)
    if visit is None:
        idx = torch.randint(0, n_blocks, (n,), device=device, generator=generator)
    else:
        p = torch.tensor(visit, dtype=torch.float32, device=device)
        idx = torch.multinomial(p, n, replacement=True, generator=generator)
    return idx.reshape(*shape)


def sample_t_in_band(band: Tensor, n_blocks: int, gamma: float,
                     generator: torch.Generator | None = None) -> Tensor:
    """``t`` drawn uniformly inside each entry's γ-widened band (same shape, fp32)."""
    w = 1.0 / n_blocks
    b = band.float()
    lo = (b * w - gamma * w).clamp(min=0.0)
    hi = ((b + 1.0) * w + gamma * w).clamp(max=1.0)
    u = torch.rand(band.shape, device=band.device, dtype=torch.float32, generator=generator)
    return lo + (hi - lo) * u


def stratified_t(shape, device, phase: float = 0.0) -> Tensor:
    """Deterministic eval ``t``: the midpoints ``(i + 0.5)/n + phase (mod 1)`` along
    the LAST axis.

    Eval must be bit-reproducible on the same rows (``test_eval_forward_is_deterministic``
    is the property every same-rows sweep in ``lab/`` rests on), so ``t`` is a grid, not
    a draw. A FIXED grid, though, pins each stratum to one band for every eval batch: at
    batch 2 and 4 blocks the midpoints 0.25 / 0.75 never visit bands 0 and 2 (the tok
    smoke read ``dm_n_band0 = 0``), and at batch 6 bands 1 and 3 get two rows to bands
    0 and 2's one. ``phase`` rotates the grid per batch (``eval_phase``), so the eval SET
    covers every band evenly while the same rows still get the same ``t``.
    """
    n = int(shape[-1])
    grid = ((torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n + phase) % 1.0
    return grid.expand(*shape).contiguous()


_PHASE_MOD = 4093     # prime: the rotation does not alias the batch or grid size


def eval_phase(input_ids: Tensor) -> float:
    """A deterministic phase in ``[0, 1)`` from the rows themselves: the same rows give
    the same phase (reproducible eval), different eval batches give different ones."""
    s = int(input_ids[:, :64].to(torch.int64).sum().item())
    return (s % _PHASE_MOD) / _PHASE_MOD


def eval_weight_key(key: str) -> str | None:
    """The ``dm_n_*`` count that weights ``key`` when eval batches are averaged.

    A per-batch mean over batches is wrong for every masked term: a batch with no row
    in band ``b`` emits ``dm_fm_band{b} = 0`` and would pull the band's mean toward 0.
    Returns None for terms that are a plain mean (``dm_head_scale``, the counts).
    """
    if key.startswith("dm_n_") or key == "dm_head_scale":
        return None
    m = re.fullmatch(r"dm_(fm|ce)_band(\d+)", key)
    if m:
        return f"dm_n_band{m.group(2)}" if m.group(1) == "fm" else f"dm_n_ce_band{m.group(2)}"
    if key.startswith(("dm_fm", "dm_sigreg")) or key == "dm_cos":
        return "dm_n_fm"
    return "dm_n_ce"      # dm_ce*, dm_ladder_*, dm_target_ce, dm_clean_acc, dm_worth_*


def aggregate_eval(acc: dict[str, list[float]], prefix: str = "val/") -> dict[str, float]:
    """Count-weighted means of the ``dm_*`` lists an eval loop collected (one entry per
    batch). A term whose weights sum to zero over the whole eval is NaN, never 0."""
    out: dict[str, float] = {}
    for k, vals in acc.items():
        if not k.startswith(prefix + "dm_") or not vals:
            continue
        wk = eval_weight_key(k[len(prefix):])
        if wk is None:
            out[k] = sum(vals) / len(vals)
            continue
        w = acc.get(prefix + wk)
        if w is None or len(w) != len(vals):
            raise KeyError(f"{k} needs its count {prefix}{wk} on every batch")
        tot = sum(w)
        out[k] = sum(v * wi for v, wi in zip(vals, w)) / tot if tot > 0 else float("nan")
    return out


# ── flow arithmetic ──────────────────────────────────────────────────────────

def fm_euler_step(x_t: Tensor, d_hat: Tensor, t: Tensor, t_next: Tensor) -> Tensor:
    """One Euler step of the rectified flow, written in ``D̂`` (the ``x1`` estimate).

    From ``x_t = (1 - t)·x0 + t·D̂`` recover ``x̂0 = (x_t - t·D̂)/(1 - t)``, then
    re-interpolate at ``t'``: ``x_{t'} = (1 - t')·x̂0 + t'·D̂``. With an UNBRIDGED
    ``D̂ = x_t + (1 - t)·v̂`` this is exactly ``x_t + (t' - t)·v̂`` (Euler on the
    velocity); with a bridged ``D̂`` it is the FM analogue of the EDM step
    ``z ← α·z + (1 - α)·D`` (:func:`morph.model.diffusion_blocks.euler_step`). At
    ``t' = 1`` the state IS ``D̂``.
    """
    t = t.float().view(-1, *([1] * (x_t.dim() - 1))) if t.dim() == 1 else t.float()[..., None]
    tn = (t_next.float().view(-1, *([1] * (x_t.dim() - 1))) if t_next.dim() == 1
          else t_next.float()[..., None])
    xf, df = x_t.float(), d_hat.float()
    x0_hat = (xf - t * df) / (1.0 - t).clamp(min=1e-6)
    return (1.0 - tn) * x0_hat + tn * df


def argmax_head(x: Tensor, w_head: Tensor, mask_id: int, chunk: int = 1024) -> Tensor:
    """``argmax_v (x @ w_headᵀ)`` over rows of ``x`` ``[N, d]`` WITHOUT materialising
    ``[N, V]``: vocab logits are formed ``chunk`` rows at a time. ``mask_id`` (the
    structural ``slot_id``) can never win."""
    N = x.shape[0]
    out = torch.empty(N, dtype=torch.long, device=x.device)
    w = w_head.to(x.dtype)
    for s in range(0, N, chunk):
        lg = x[s:s + chunk] @ w.t()
        if mask_id >= 0:
            lg[:, mask_id] = float("-inf")
        out[s:s + chunk] = lg.argmax(dim=-1)
    return out


def readout_state(state: Tensor, head_scale: Tensor) -> Tensor:
    """``D̂`` → the vector the tied head reads: ``normalize(D̂) · head_scale``, fp32.

    The targets are unit L2, but ``D̂ = x_t + (1 - t)·v̂`` is NOT: at low ``t`` it is
    dominated by the source ``x0 ~ N(0, s² I)`` whose norm is ``s·sqrt(d)`` — 32 at
    ``source_std 1.0, d 1024``. Reading the raw ``D̂`` through ``head_scale`` (itself
    ~sqrt(d)) put logit gaps in the hundreds and the CE-through-D̂ term at 40+ nats on
    the first panel run (dmorph-tok-s1-5k, 2026-09-03: dm_ce 41.7 at step 250, the
    shared weights dragged by it, the guard tripped at step 2040). The testbed reads
    its bridge through the table in TARGET space for the same reason. Normalising
    first makes the readout depend on D̂'s direction only, so the term starts at the
    calibrated ``head_scale`` temperature whatever the source scale is."""
    return F.normalize(state.float(), dim=-1) * head_scale.float()


def hard_bridge(d_hat: Tensor, w_head: Tensor, head_scale: Tensor, mask_id: int,
                chunk: int = 1024) -> Tensor:
    """The HARD bridge: replace ``D̂`` by the unit-normalised embedding row of its argmax
    token. Always full norm, always on the target manifold — the soft
    ``softmax @ E`` bridge shrinks ``‖D̂‖`` to 0.27 at high noise
    (``db-testbed-ladder.md`` B, gen-PPL 584 → 17 when switched). ``[..., d]`` in,
    same shape out, fp32."""
    shp = d_hat.shape
    flat = readout_state(d_hat, head_scale).reshape(-1, shp[-1])
    idx = argmax_head(flat, w_head.float(), mask_id, chunk)
    rows = F.normalize(w_head.float()[idx], dim=-1)
    return rows.reshape(shp)


# ── the module ───────────────────────────────────────────────────────────────

class DmorphStream(nn.Module):
    """The noisy stream's OWN parameters: time conditioning, per-layer AdaLN-Zero gates,
    the velocity head and the head scale. The layers themselves are the backbone's.

    Zero-init bit-identity, inherited from ``CoreStageConditioning``: every
    :class:`AdaLNGate` and ``W_v`` are zero-initialised, so at construction the block
    modulation is the identity and the velocity is exactly 0 — the FM term starts at
    the analytic null floor (``tests/test_dmorph_losses.py``).

    RNG discipline: ``SigmaConditioning`` draws two Linear inits from the GLOBAL stream,
    so this module is built LAST in ``MORPHTransformer.__init__`` (after SCSE) and a
    dmorph model's base weights match a control built with the same seed
    (``tests/test_dmorph_off_is_bit_identical.py``).

    Ternary QAT: every Linear here is a CONTROL path (AdaLN modulation, the t MLP, the
    velocity read-out) and is excluded from ``ternary_scope`` by path prefix
    (``morph/model/ternary_qat.py::_categorize``), the same rule as the HC ``W_fused``
    projection and the ``DiagonalInjection`` matrices.
    """

    def __init__(self, d_model: int, n_layers: int, cfg: DmorphConfig):
        super().__init__()
        if n_layers < 1 or n_layers % cfg.n_blocks != 0:
            raise ValueError(
                f"dmorph needs n_prelude + n_coda ({n_layers}) divisible by n_blocks "
                f"({cfg.n_blocks}) — one block is one contiguous, equal-size layer group")
        self.cfg = cfg
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.layers_per_block = self.n_layers // cfg.n_blocks
        self.cond = SigmaConditioning(cond_dim=cfg.cond_dim)
        self.gates = nn.ModuleList([AdaLNGate(cfg.cond_dim, d_model) for _ in range(n_layers)])
        self.v_norm = RMSNorm(d_model)
        self.v_gate = AdaLNGate(cfg.cond_dim, d_model)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.W_v.weight)
        # Scalar read-out gain on normalize(D̂) before the tied head (see
        # :func:`readout_state`). The clean head reads a state of norm ~sqrt(d)
        # (final_norm's per-component RMS is 1), so a UNIT vector needs ~sqrt(d) to reach
        # the same logit temperature; without it the CE term cannot leave ln V. Learnable
        # so the ladder head can set its own temperature; the testbed solved the same
        # constant in closed form (``db-testbed-ladder.md`` A, "Readout scale").
        self.head_scale = nn.Parameter(torch.tensor(math.sqrt(float(d_model))))
        # E‖v*‖² of the zero network: E‖y‖² + d·s² with unit-L2 targets.
        self.null_floor = 1.0 + float(d_model) * float(cfg.source_std) ** 2

    # -- conditioning -------------------------------------------------------
    def embed_t(self, t: Tensor) -> Tensor:
        """``t`` of any shape → ``[..., cond_dim]``."""
        flat = self.cond(t.reshape(-1).float() * self.cfg.t_embed_scale)
        return flat.reshape(*t.shape, -1)

    @staticmethod
    def _apply_gate(gate: AdaLNGate, x: Tensor, cond: Tensor) -> Tensor:
        """AdaLN-Zero with a per-ROW ``[B, c]`` or per-POSITION ``[B, L, c]`` condition on
        a ``[B, L, C]`` or HC ``[B, L, n, C]`` carrier. :class:`AdaLNGate.forward`
        broadcasts a ``[B, c]`` condition only, so the broadcast is done here."""
        shift, scale = gate.to_mod(cond.to(x.dtype)).chunk(2, dim=-1)
        if cond.dim() == 2:                       # [B, C] → [B, 1, (1,) C]
            view = (x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
        else:                                     # [B, L, C] → [B, L, (1,) C]
            view = x.shape[:2] + (1,) * (x.dim() - 3) + (x.shape[-1],)
        return x * (1.0 + scale.reshape(view)) + shift.reshape(view)

    def modulate(self, x: Tensor, cond: Tensor, layer_idx: int) -> Tensor:
        return self._apply_gate(self.gates[layer_idx], x, cond)

    def in_gain_value(self) -> float:
        """The fixed input gain on ``x_t``: ``sqrt(d)`` by default, so the noisy state
        enters its block at per-component RMS ~1, the scale ``input_norm`` hands the
        clean coda (a unit-L2 ``x_t`` would otherwise sit ~sqrt(d) below the block's own
        residual writes and be drowned by depth 3)."""
        return float(self.cfg.in_gain)

    def velocity(self, h: Tensor, cond: Tensor) -> Tensor:
        """``[B, L, C]`` block output → ``[B, L, C]`` velocity, fp32.

        DiT's final layer: norm → AdaLN-Zero → zero-init linear. Exactly 0 at init."""
        return self.W_v(self._apply_gate(self.v_gate, self.v_norm(h), cond)).float()

    def manifest(self) -> dict:
        c = self.cfg
        return {
            "dmorph/arm": c.arm,
            "dmorph/n_blocks": c.n_blocks,
            "dmorph/layers_per_block": self.layers_per_block,
            "dmorph/block_layers": [list(block_layers(b, self.layers_per_block))
                                    for b in range(c.n_blocks)],
            "dmorph/band_bounds_widened": [list(band_bounds(b, c.n_blocks, c.gamma))
                                           for b in range(c.n_blocks)],
            "dmorph/gamma": c.gamma,
            "dmorph/lambda_fm": c.lambda_fm,
            "dmorph/lambda_ce": c.lambda_ce,
            "dmorph/source_std": c.source_std,
            "dmorph/null_floor": self.null_floor,
            "dmorph/loss_scale": c.loss_scale,
            "dmorph/detach_ctx": c.detach_ctx,
            "dmorph/t_per_position": c.t_per_position,
            "dmorph/sigreg_lambda": c.sigreg_lambda,
            "dmorph/sigreg_slices": c.sigreg_slices,
            "dmorph/infer_steps": c.n_infer_steps,
            "dmorph/cond_dim": c.cond_dim,
            "dmorph/t_embed_scale": c.t_embed_scale,
            "dmorph/in_gain": c.in_gain,
            "dmorph/head_scale_init": math.sqrt(float(self.d_model)),
            "dmorph/block_visit": (list(c.block_visit) if c.block_visit is not None
                                   else [1.0 / c.n_blocks] * c.n_blocks),
            "dmorph/n_params": int(sum(p.numel() for p in self.parameters())),
        }


# ── the stream over the backbone ─────────────────────────────────────────────

@dataclass
class DmCtx:
    """The clean stream's per-position inputs the noisy stream shares.

    ``x0`` ``[B, L, C]`` is the front's single-stream skip signal (the token embedding,
    or the slot seed at slot positions); ``bigram`` ``[B, L, C]`` or None; ``input_ids``
    ``[B, L]``; ``ve_bagged`` the value-embed signals with the slot bag-means applied.
    These feed :meth:`MORPHTransformer._build_injection_term` at every layer, exactly
    as they feed the clean stream. All are CLEAN context (the current token, never the
    label), so sharing them leaks nothing — ``diffusion_blocks.py::DBConfig`` spells
    out why unshifted ``x0`` is causally correct.
    """

    x0: Tensor
    bigram: Tensor | None
    input_ids: Tensor
    ve_bagged: list[Tensor] | None

    def rows(self, idx: Tensor | None) -> "DmCtx":
        if idx is None:
            return self
        sel = lambda t: t.index_select(0, idx)  # noqa: E731
        return DmCtx(
            x0=sel(self.x0),
            bigram=None if self.bigram is None else sel(self.bigram),
            input_ids=sel(self.input_ids),
            ve_bagged=None if self.ve_bagged is None else [sel(v) for v in self.ve_bagged],
        )

    def detach(self) -> "DmCtx":
        return DmCtx(
            x0=self.x0.detach(),
            bigram=None if self.bigram is None else self.bigram.detach(),
            input_ids=self.input_ids,
            ve_bagged=None if self.ve_bagged is None else [v.detach() for v in self.ve_bagged],
        )


def _layer_of(model, gi: int) -> nn.Module:
    n_pre = model.cfg.n_prelude
    return model.prelude[gi] if gi < n_pre else model.coda[gi - n_pre]


def _select_kv(kv: dict, idx: Tensor | None, detach: bool) -> dict:
    out = {}
    for k, v in kv.items():
        t = v if idx is None else v.index_select(0, idx)
        out[k] = t.detach() if detach else t
    return out


def run_block(model, block: int, x_in: Tensor, cond: Tensor, kv: list[dict],
              ctx: DmCtx, rows: Tensor | None) -> Tensor:
    """Run block ``b`` of the flat stack on the noisy input ``x_in`` ``[n, L, C]``.

    Per layer ``gi`` of the block: the clean stream's additive injection for that layer
    (x0 / value-embed / bigram, the same ``_build_injection_term`` call), the AdaLN-Zero
    time gate, then the layer with ``cla_kv = kv[gi]`` so every attention head reads the
    CLEAN keys and values (rows ``j < i`` in the window, blocks strictly before ``i`` in
    the compressed branch — ``tests/test_dmorph_no_leak.py``). Returns the stream-mean
    ``[n, L, C]`` block output.

    ``rows`` selects the batch rows this block owns (None = all). ``kv`` and ``ctx`` are
    the FULL-batch captures; they are row-selected here so the attention's own
    ``[:bsz]`` prefix slice is the identity.
    """
    dm: DmorphStream = model.dmorph
    detach = dm.cfg.detach_ctx
    c = ctx.rows(rows)
    if detach:
        c = c.detach()
    h = x_in
    if model._is_hc:
        n, L, C = h.shape
        h = h.unsqueeze(2).expand(n, L, model._n_streams, C).contiguous()
    for gi in block_layers(block, dm.layers_per_block):
        term = model._build_injection_term(
            gi, model.x0_injects[gi].precompute(c.x0), c.input_ids, c.bigram, h.dtype,
            ve_bagged=c.ve_bagged)
        h = model._apply_injection(h, term)
        h = dm.modulate(h, cond, gi)
        h = _layer_of(model, gi)(h, attn_kwargs={"cla_kv": _select_kv(kv[gi], rows, detach)})
    return h.mean(dim=2) if model._is_hc else h


def noisy_stream(model, x_t: Tensor, t: Tensor, band: Tensor, kv: list[dict],
                 ctx: DmCtx) -> Tensor:
    """``v̂`` ``[B, L, d]`` (fp32) for the noisy state ``x_t`` at flow time ``t``.

    ``t`` and ``band`` are ``[B]`` (one per row, the default) or ``[B, L]``. Each block
    runs ONCE, on the rows that have at least one position in its band; a position's
    velocity is read from its own band's block. In the per-row mode every row belongs
    to exactly one block, so the total work is one block per row — the 1.25× of the
    design note's FLOP table. In the per-position mode every block runs on every row.
    """
    dm: DmorphStream = model.dmorph
    B, L, d = x_t.shape
    cond = dm.embed_t(t)                                   # [B, c] or [B, L, c]
    x_in = x_t.float() * dm.in_gain_value()
    v_hat = x_t.new_zeros(B, L, d, dtype=torch.float32)
    per_row = band.dim() == 1
    for b in range(dm.cfg.n_blocks):
        if per_row:
            rows = (band == b).nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            sel = None if rows.numel() == B else rows
            h = run_block(model, b, x_in if sel is None else x_in.index_select(0, sel),
                          cond if sel is None else cond.index_select(0, sel), kv, ctx, sel)
            contrib = dm.velocity(h, cond if sel is None else cond.index_select(0, sel))
            v_hat = contrib if sel is None else v_hat.index_add(0, rows, contrib)
        else:
            has = (band == b).any(dim=1)
            rows = has.nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            sel = None if rows.numel() == B else rows
            cond_r = cond if sel is None else cond.index_select(0, sel)
            h = run_block(model, b, x_in if sel is None else x_in.index_select(0, sel),
                          cond_r, kv, ctx, sel)
            v_b = dm.velocity(h, cond_r)
            m = ((band if sel is None else band.index_select(0, sel)) == b).unsqueeze(-1)
            contrib = torch.where(m, v_b, torch.zeros_like(v_b))
            v_hat = v_hat + contrib if sel is None else v_hat.index_add(0, rows, contrib)
    return v_hat


def _bcast(t: Tensor, like: Tensor) -> Tensor:
    """``[B]`` or ``[B, L]`` → broadcastable against ``[B, L, d]``."""
    return t.float().view(-1, 1, 1) if t.dim() == 1 else t.float().unsqueeze(-1)


# ── targets and losses ───────────────────────────────────────────────────────

def targets(model, xh: Tensor, labels: Tensor, layout, row_w: Tensor):
    """``(y, fm_valid, ce_labels, ce_weights)`` for the configured arm.

    ``y`` ``[B, L, d]`` fp32 unit-L2 (zero where invalid); ``fm_valid`` ``[B, L]`` bool;
    ``ce_labels`` ``[B, L]`` with ``-100`` outside the CE term; ``ce_weights`` ``[B·L]``.

    tok: ``y = normalize(E[label])`` at every position whose clean-head weight is
    positive — the same ``row_w`` (plast 1.0 / emit 0.0) the clean CE trains with, so
    the two heads are supervised on the same positions. DETACHED.

    hs: ``y = normalize(xh)`` at REAL slot positions (tail pads excluded); the CE term
    lives at the slot positions that carry a label (the emit position: the first token
    of the next span). Detached unless ``sigreg_lambda > 0``.
    """
    dm: DmorphStream = model.dmorph
    B, L = labels.shape
    if dm.cfg.arm == "tok":
        w_head = model.embed.lm_weight().detach().float()
        valid = (labels != -100) & (row_w.view(B, L) > 0)
        y = F.normalize(w_head[labels.clamp(min=0)], dim=-1) * valid.unsqueeze(-1)
        ce_labels = torch.where(valid, labels, torch.full_like(labels, -100))
        return y, valid, ce_labels, row_w
    real_slot = layout.slot_mask & (layout.bag_id != layout.max_slots)
    src = xh.float() if dm.cfg.sigreg_lambda > 0.0 else xh.detach().float()
    y = F.normalize(src, dim=-1) * real_slot.unsqueeze(-1)
    ce_valid = real_slot & (labels != -100)
    ce_labels = torch.where(ce_valid, labels, torch.full_like(labels, -100))
    return y, real_slot, ce_labels, labels.new_ones(B * L, dtype=torch.float32)


def _masked_mean(x: Tensor, m: Tensor) -> Tensor:
    mf = m.to(x.dtype)
    return (x * mf).sum() / mf.sum().clamp(min=1.0)


def _ce_from(model, state: Tensor, ce_labels: Tensor, ce_w: Tensor, w_head: Tensor) -> Tensor:
    dm: DmorphStream = model.dmorph
    d = state.shape[-1]
    flat = readout_state(state, dm.head_scale).reshape(-1, d)
    return fused_linear_cross_entropy(
        flat, w_head, ce_labels.reshape(-1), ignore_index=-100,
        chunk_size=model.cfg.ce_chunk_size, mask_token_id=model.cfg.tul.slot_id,
        weights=ce_w)


def _acc_from(model, state: Tensor, ce_labels: Tensor, ce_w: Tensor, w_head: Tensor) -> Tensor:
    dm: DmorphStream = model.dmorph
    d = state.shape[-1]
    lab = ce_labels.reshape(-1)
    m = (lab != -100) & (ce_w > 0)
    if int(m.sum()) == 0:
        return state.new_zeros(())
    flat = readout_state(state, dm.head_scale).reshape(-1, d)[m]
    pred = argmax_head(flat, w_head.float(), model.cfg.tul.slot_id, model.cfg.ce_chunk_size)
    return (pred == lab[m]).float().mean()


def training_terms(model, *, xh: Tensor, labels: Tensor, layout, kv: list[dict],
                   ctx: DmCtx, row_w: Tensor, want_eval: bool) -> tuple[Tensor, dict]:
    """The dmorph loss terms for one forward. Returns ``(add_to_loss, groups)``.

    ``groups`` carries every term detached, keyed ``dm_*`` (the trainer logs them under
    ``dm/`` and ``val/``). With ``want_eval`` (eval mode) it adds the per-band losses,
    the ladder metrics and, for hs, the cosine and the four-condition worth.
    """
    dm: DmorphStream = model.dmorph
    cfg = dm.cfg
    B, L, d = xh.shape
    dev = xh.device
    y, fm_valid, ce_labels, ce_w = targets(model, xh, labels, layout, row_w)

    # ── flow time and block, block FIRST then t in its widened band ────────────
    if model.training:
        shape = (B, L) if cfg.t_per_position else (B,)
        band = sample_blocks(shape, cfg.n_blocks, cfg.block_visit, dev)
        t = sample_t_in_band(band, cfg.n_blocks, cfg.gamma)
    else:
        t = stratified_t((B, L) if cfg.t_per_position else (B,), dev,
                         phase=eval_phase(ctx.input_ids))
        band = band_of_t(t, cfg.n_blocks)

    if model.training:
        x0 = torch.randn(B, L, d, device=dev, dtype=torch.float32) * cfg.source_std
    else:
        # Eval draws its source from a FIXED generator: two eval forwards on the same
        # rows are bit-identical (the same-rows sweep property), and the global stream
        # is not advanced by an instrument.
        g = torch.Generator(device=dev).manual_seed(0)
        x0 = torch.randn(B, L, d, device=dev, dtype=torch.float32, generator=g) * cfg.source_std
    tb = _bcast(t, y)
    x_t = (1.0 - tb) * x0 + tb * y
    v_star = y - x0
    v_hat = noisy_stream(model, x_t, t, band, kv, ctx)
    d_hat = x_t + (1.0 - tb) * v_hat

    sq = (v_hat - v_star).pow(2).sum(-1)                   # [B, L]
    fm_raw = _masked_mean(sq, fm_valid)
    fm = fm_raw / dm.null_floor if cfg.loss_scale == "auto" else fm_raw
    w_head = model.embed.lm_weight()
    ce = _ce_from(model, d_hat, ce_labels, ce_w, w_head)

    add = cfg.lambda_fm * fm + cfg.lambda_ce * ce
    groups = {
        "dm_fm": fm.detach(),
        "dm_fm_raw": fm_raw.detach(),
        "dm_fm_rel": (fm_raw / dm.null_floor).detach(),
        "dm_ce": ce.detach(),
        "dm_fm_weighted": (cfg.lambda_fm * fm).detach(),
        "dm_ce_weighted": (cfg.lambda_ce * ce).detach(),
        "dm_head_scale": dm.head_scale.detach(),
        "dm_n_fm": fm_valid.sum().float(),
        "dm_n_ce": ((ce_labels != -100) & (ce_w.view(B, L) > 0)).sum().float(),
    }
    if cfg.arm == "hs" and cfg.sigreg_lambda > 0.0:
        from .tul_fm import fm_sigreg_loss
        sig = fm_sigreg_loss(y, fm_valid, cfg.sigreg_slices)
        add = add + cfg.sigreg_lambda * sig
        groups["dm_sigreg"] = sig.detach()
        groups["dm_sigreg_weighted"] = (cfg.sigreg_lambda * sig).detach()

    if want_eval:
        groups.update(eval_terms(model, xh=xh, y=y, fm_valid=fm_valid, ce_labels=ce_labels,
                                 ce_w=ce_w, band=band, sq=sq, d_hat=d_hat, kv=kv, ctx=ctx,
                                 layout=layout, w_head=w_head))
    return add, groups


@torch.no_grad()
def eval_terms(model, *, xh, y, fm_valid, ce_labels, ce_w, band, sq, d_hat, kv, ctx,
               layout, w_head) -> dict:
    """Eval-only instruments (no gradient): per-band losses, the ladder, cosine, worth.

    Per-band: the FM loss and the CE-through-D̂ restricted to positions whose row (or
    position) sat in band ``b`` this forward. Band 0 is the "σ_max" read — the only
    band a model cannot win by autoencoding (``db-testbed-ladder.md`` B, "σ_max is the
    metric"): trust no grid mean that band 0 disagrees with.
    """
    dm: DmorphStream = model.dmorph
    cfg = dm.cfg
    B, L, d = xh.shape
    out: dict = {}
    band_pos = band if band.dim() == 2 else band.view(B, 1).expand(B, L)
    for b in range(cfg.n_blocks):
        m = fm_valid & (band_pos == b)
        out[f"dm_fm_band{b}"] = _masked_mean(sq, m) / dm.null_floor
        out[f"dm_n_band{b}"] = m.sum().float()
        wb = ce_w * (band_pos == b).reshape(-1).float()
        out[f"dm_ce_band{b}"] = _ce_from(model, d_hat, ce_labels, wb, w_head)
        out[f"dm_n_ce_band{b}"] = ((ce_labels != -100) & (wb.view(B, L) > 0)).sum().float()

    d_last, _x_final = ladder(model, kv, ctx, (B, L, d), bridge=(cfg.arm == "tok"),
                              w_head=w_head)
    out["dm_ladder_ce"] = _ce_from(model, d_last, ce_labels, ce_w, w_head)
    out["dm_ladder_acc"] = _acc_from(model, d_last, ce_labels, ce_w, w_head)
    out["dm_target_ce"] = _ce_from(model, y, ce_labels, ce_w, w_head)
    cos = F.cosine_similarity(d_last, y, dim=-1)
    out["dm_cos"] = _masked_mean(cos, fm_valid)
    if cfg.arm == "tok":
        # The clean head's greedy accuracy on the same positions (prereg P3).
        out["dm_clean_acc"] = (
            (argmax_head(xh.reshape(-1, d).float(), w_head.float(), model.cfg.tul.slot_id,
                         model.cfg.ce_chunk_size).view(B, L) == ce_labels)
            .float().mul((ce_labels != -100) & (ce_w.view(B, L) > 0)).sum()
            / ((ce_labels != -100) & (ce_w.view(B, L) > 0)).sum().clamp(min=1).float())
    else:
        # The four-condition worth, on the one reader a post-stack slot state has in a
        # flat stack: the head at the emit position (docs/tul-fm-probing.md §4 rule 1 —
        # report the COST, never a fraction). `clean` is the target itself through the
        # same read-out (what a perfect ladder would score); `zero` removes the state;
        # `shuffle` swaps in another row's slot state at the same slot index.
        shuf = _shuffle_rows_at_slots(y, layout)
        c_clean = out["dm_target_ce"]
        c_lad = out["dm_ladder_ce"]
        c_zero = _ce_from(model, torch.zeros_like(y), ce_labels, ce_w, w_head)
        c_shuf = _ce_from(model, shuf, ce_labels, ce_w, w_head)
        out.update({
            "dm_worth_clean": c_clean, "dm_worth_ladder": c_lad,
            "dm_worth_zero": c_zero, "dm_worth_shuffle": c_shuf,
            "dm_worth_cost_ladder": c_lad - c_clean,
            "dm_worth_cost_zero": c_zero - c_clean,
            "dm_worth_cost_shuffle": c_shuf - c_clean,
        })
    return out


def _shuffle_rows_at_slots(y: Tensor, layout) -> Tensor:
    """Row ``b``'s slot ``s`` receives row ``(b+1) mod B``'s slot ``s`` state (zero when
    that row has no valid slot ``s``); token positions are untouched."""
    B, S = layout.slot_index.shape
    K = layout.prefix_k
    d = y.shape[-1]
    offs = torch.arange(K, device=y.device)
    pos = (layout.slot_index.unsqueeze(-1) + offs).reshape(B, S * K)          # [B, S·K]
    valid = layout.slot_valid.unsqueeze(-1).expand(B, S, K).reshape(B, S * K)
    safe = torch.where(valid, pos, torch.zeros_like(pos))
    g = torch.gather(y, 1, safe.unsqueeze(-1).expand(B, S * K, d))
    g = g * valid.unsqueeze(-1).to(g.dtype)
    g_roll = torch.roll(g, shifts=1, dims=0)
    v_roll = torch.roll(valid, shifts=1, dims=0)
    put = torch.where((valid & v_roll).unsqueeze(-1), g_roll, torch.zeros_like(g_roll))
    out = y.clone()
    rows_idx = torch.arange(B, device=y.device).unsqueeze(1).expand(B, S * K)[valid]
    out[rows_idx, pos[valid]] = put[valid]
    return out


@torch.no_grad()
def ladder(model, kv: list[dict], ctx: DmCtx, shape, *, bridge: bool, w_head: Tensor,
           seed: int = 0) -> tuple[Tensor, Tensor]:
    """The ``K``-step Euler ladder from pure noise, one block per step in order.

    Step ``k`` runs at ``t_k = k/K`` through block :func:`band_of_t` ``(t_k)`` — with
    ``K == n_blocks`` that is block ``k`` exactly, each block once, in order 0 → B−1
    (``tests/test_dmorph_ladder.py``). ``D̂_k = x + (1 - t_k)·v̂`` is read; for the tok
    arm it is then replaced by the HARD bridge before the Euler step to ``t_{k+1}``.
    Returns ``(D̂_last_prebridge, x_final)`` — the last block's UNBRIDGED estimate is
    what the head scores (a bridged one is one-hot by construction).

    Noise is drawn from a fixed-seed generator so an eval is reproducible on its rows.
    """
    dm: DmorphStream = model.dmorph
    cfg = dm.cfg
    B, L, d = shape
    dev = ctx.x0.device
    g = torch.Generator(device=dev).manual_seed(int(seed))
    x = torch.randn(B, L, d, device=dev, dtype=torch.float32, generator=g) * cfg.source_std
    K = cfg.n_infer_steps
    d_hat = x
    for k in range(K):
        t_k = torch.full((B,), k / K, device=dev, dtype=torch.float32)
        band = band_of_t(t_k, cfg.n_blocks)
        v = noisy_stream(model, x, t_k, band, kv, ctx)
        d_hat = x + (1.0 - k / K) * v
        d_step = hard_bridge(d_hat, w_head, dm.head_scale, model.cfg.tul.slot_id,
                             model.cfg.ce_chunk_size) if bridge else d_hat
        t_next = torch.full((B,), (k + 1) / K, device=dev, dtype=torch.float32)
        x = fm_euler_step(x, d_step, t_k, t_next)
    return d_hat, x
