"""Analytic FLOP model and the throughput / FLOP-efficiency / VRAM metric keys.

Gate A3 of ``docs/diffusionblocks-plan-of-action.md``. Metric contract:
``docs/diffusionblocks-experiment-sheet.md`` §1.

Why this file exists. Before it, MORPH could not report FLOP efficiency at all:
``perf/mfu`` did not exist and ``layer_passes_per_token`` was logged only when TUL was on.
Comparing a DiffusionBlocks arm to A0 on tok/s alone is misleading, because MORPH is
launch-bound — A0's step is ~16 % fixed overhead and a DB arm's is ~50-60 %. Every claim
of "faster" has to cite tok/s AND flop_proxy AND peak memory together.

Why not ``torch.utils.flop_counter.FlopCounterMode``. It is blind to Triton. MORPH's fused
attention, HC, GLA, decode and CE kernels are custom, so FlopCounterMode silently
undercounts the majority of the model. It is still useful as a CROSS-CHECK on the aten
half; it cannot be the logged number. ``ncu`` (installed) gives real hardware counters but
serialises kernels — a one-off validation, never per-step. ``nvidia-smi`` has no FLOP
counter at all.

The approach here, and its honesty boundary:

* **Exact.** Every weight GEMM is counted from the REAL module shapes by walking the model
  tree once (``nn.Linear``, ``MortarLinear``, anything exposing ``in_features``/
  ``out_features``). No guessed dimensions.
* **Modelled.** The attention score/value matmuls scale with sequence length, not with a
  weight shape, so they are computed from config with the formulas in
  :func:`attention_flops_per_token`. These are the approximate terms. They are reported
  separately (``perf/flops_attn_frac``) so a reader can see how much of the total rests on
  a model rather than a measurement.
* **Not counted.** Norms, activations, elementwise adds, softmax, the HC Cayley closed
  form, RoPE/CoPE. Together these are a few percent of FLOPs but a large share of KERNEL
  LAUNCHES — which is exactly why ``flop_proxy`` and not ``mfu`` is the primary
  cross-arm number.

NOMINAL vs REALIZED — the distinction that makes the A3 gate self-consistent. Core depth is
``clamp(Poisson(mean_depth), 1, max_depth)`` (``transformer.py::_sample_depths``). At
mean 6 capped 8 the realized mean is **5.688**, not 6. So:

* ``perf/flop_proxy`` is NOMINAL: derived from config at ``T = mean_depth``. A0 = 44.0
  exactly (4 + 6·6 + 4). This is the pre-registered, run-to-run comparable number.
* ``perf/layer_passes_per_token`` is REALIZED: from the depths actually sampled. A0 ≈ 42.1.
  The TUL anchor of 10.68 in the ablation ledger is a realized number.

Mixing the two is the bug this docstring exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = [
    "FLOP_MODEL_VERSION",
    "FlopModel",
    "expected_clamped_poisson",
    "build_flop_model",
]

# Bump on ANY change to a formula here. Logged next to every perf/mfu value: an MFU from
# model v1 is not comparable to one from v2, and without the version stamped in the run
# nobody can tell them apart six weeks later.
FLOP_MODEL_VERSION = "1.0.0"


def expected_clamped_poisson(lam: float, lo: int, hi: int) -> float:
    """``E[clamp(Poisson(lam), lo, hi)]`` — the REALIZED mean core depth.

    ``transformer.py::_sample_depths`` clamps, and clamping is not mean-preserving: at
    ``lam=6, lo=1, hi=8`` the answer is 5.688, not 6. The A3 gate asserts on this value,
    so it is computed exactly here rather than assumed.
    """
    if lo > hi:
        raise ValueError(f"lo={lo} > hi={hi}")
    if lo == hi:
        # Degenerate window: every draw clamps to the single allowed value. Special-cased
        # because the decomposition below splits the mass into P(X≤lo) + middle + P(X≥hi),
        # and those two tails OVERLAP at lo==hi — which double-counted P(X=lo) and returned
        # 3.27 for (6, 3, 3) instead of 3.0. Caught by
        # test_expected_clamped_poisson_degenerate_bounds.
        return float(lo)
    # Poisson pmf by stable recurrence: p_k = p_{k-1} · lam/k.
    pmf = []
    p = math.exp(-lam)
    for k in range(hi + 1):
        if k > 0:
            p = p * lam / k
        pmf.append(p)
    mass_below = sum(pmf[: lo + 1])           # everything clamped UP to lo
    mass_at_or_above_hi = 1.0 - sum(pmf[:hi])  # everything clamped DOWN to hi
    total = lo * mass_below
    for k in range(lo + 1, hi):
        total += k * pmf[k]
    total += hi * mass_at_or_above_hi
    return total


def _linear_flops(mod: nn.Module) -> int:
    """2·in·out FLOPs per token for one linear-like module, or 0 if it is not one."""
    inf = getattr(mod, "in_features", None)
    outf = getattr(mod, "out_features", None)
    if inf is None or outf is None:
        return 0
    return 2 * int(inf) * int(outf)


def _region_of(path: str, n_prelude: int, n_core: int) -> str:
    """Attribute a module path to prelude / core / coda / other.

    MORPH names its stacks ``prelude.<i>``, ``core.<i>``, ``coda.<i>``. The per-layer
    injection modules live in flat ``ModuleList``s (``x0_injects.<global_idx>``) indexed by
    GLOBAL layer index, so those are split by index against the section boundaries.
    """
    if path.startswith("prelude."):
        return "prelude"
    if path.startswith("core."):
        return "core"
    if path.startswith("coda."):
        return "coda"
    for flat in ("x0_injects.", "value_embeds.", "injection."):
        if path.startswith(flat):
            tail = path[len(flat):].split(".", 1)[0]
            if tail.isdigit():
                gi = int(tail)
                if gi < n_prelude:
                    return "prelude"
                if gi < n_prelude + n_core:
                    return "core"
                return "coda"
    return "other"


@dataclass
class FlopModel:
    """Per-token FLOP accounting for one MORPH configuration.

    All ``*_per_token`` values are FLOPs for ONE position through ONE pass of that region.
    The core value is multiplied by the depth at call time, so the same object serves both
    the nominal and the realized number.
    """

    version: str
    d_model: int
    n_prelude: int
    n_core: int
    n_coda: int
    mean_depth: int
    max_depth: int
    seq_len: int

    gemm_prelude: int = 0          # weight-GEMM FLOPs/token, EXACT (from module shapes)
    gemm_core: int = 0
    gemm_coda: int = 0
    gemm_other: int = 0            # embeddings-adjacent, LM head, mixers
    attn_prelude: int = 0          # attention score/value terms, MODELLED
    attn_core: int = 0
    attn_coda: int = 0
    assumptions: dict = field(default_factory=dict)

    # ── layer-pass proxy (exact, config-derived) ─────────────────────────────
    # Units: layer applications per INPUT TOKEN. Position inflation is already inside the
    # number, so DO NOT multiply by positions again afterwards — an earlier version of this
    # file did exactly that and double-counted the concat conditioning.

    ALL_SECTIONS = ("prelude", "core", "coda")

    def layer_passes_per_token(self, depth: float,
                               positions_per_token: float = 1.0,
                               core_position_frac: float | None = None,
                               sections: tuple[str, ...] | None = None) -> float:
        """Layer applications per INPUT token.

        Args:
            depth: core iterations. ``mean_depth`` for the baseline's looped core;
                **1.0** for a DiffusionBlocks training step, because the core is applied
                ONCE — that single pass is the entire compute win.
            positions_per_token: ``L_total / (REAL TOKENS PER ROW)``. The prelude and coda
                run on every position, so their cost scales with this.
                **The denominator is real tokens, NOT ``seq_len``.** Without TUL they are
                equal and the distinction is invisible. Under TUL they are not: slot
                positions eat row budget, so a 144-position row may carry only 49 real
                tokens, and using ``seq_len`` understates the ratio ~2.6×. MORPH's own live
                metric divides by ``out["n_tokens"]``, so match it or the analytic and live
                numbers disagree (verified: analytic 29.1 vs live 29.39 with the right
                denominator, 11.1 vs 29.39 with the wrong one).
            core_position_frac: core positions ÷ REAL TOKENS PER ROW. Defaults to
                ``positions_per_token`` (the core sees everything the rest does). Under TUL
                the core runs on slots ONLY, so it is ``n_slots / tokens_per_row`` — same
                denominator caveat as above.
            sections: which sections run. All three for the baseline and for ``mode="b1"``.
                Exactly one under ``mode="b3"`` — use :meth:`db_expected_passes` to take the
                expectation over the block-visit distribution instead of calling this once
                per block.

        Checks against the recorded anchors (ablation-ledger.md):
            A0  → 4·1 + 6·6·1 + 4·1                        = 44.0  (nominal, T̄=6)
            A1  → 8·(1152/1033) + 6·5.688·(64/1033)        ≈ 11.0  (ledger measured 10.68,
                  i.e. 3.4 % agreement, using tokens/row = 1033 not seq_len = 1024)
        """
        if core_position_frac is None:
            core_position_frac = positions_per_token
        sec = self.ALL_SECTIONS if sections is None else sections
        n = 0.0
        if "prelude" in sec:
            n += self.n_prelude * positions_per_token
        if "coda" in sec:
            n += self.n_coda * positions_per_token
        if "core" in sec:
            n += self.n_core * depth * core_position_frac
        return n

    def db_expected_passes(self, visit_probs: list[float], depth: float = 1.0,
                           positions_per_token: float = 1.0,
                           core_position_frac: float | None = None) -> float:
        """Expected layer applications per token for a DiffusionBlocks B=3 step.

        One block runs per step (the authors sample one block per BATCH), so the per-step
        cost is a random variable and the comparable number is its expectation over the
        visit distribution.

        With uniform visits, ``depth=1``, MORPH's 4:6:4 and no position inflation:
        ``(4 + 6 + 4)/3 = 4.67``. With the concat conditioning (``positions_per_token=2``):
        ``(8 + 12 + 8)/3 = 9.33``.
        """
        if len(visit_probs) != 3:
            raise ValueError(
                f"db_expected_passes is the B=3 form; got {len(visit_probs)} visit probs. "
                f"For mode='b1' call layer_passes_per_token with all sections and depth=1.")
        per_section = [
            self.layer_passes_per_token(depth, positions_per_token, core_position_frac,
                                        sections=(name,))
            for name in self.ALL_SECTIONS
        ]
        return sum(p * c for p, c in zip(visit_probs, per_section))

    def flop_proxy(self, positions_per_token: float = 1.0,
                   core_position_frac: float | None = None) -> float:
        """The NOMINAL baseline proxy: the full net at ``T = mean_depth``.

        A0 (4:6:4, T=6, no TUL, no concat) = 44.0 exactly. This is the pre-registered
        cross-arm anchor. For a DB arm use :meth:`db_expected_passes` (B=3) or
        :meth:`layer_passes_per_token` with ``depth=1`` (B=1) — a DB step does not run a
        looped core, so feeding ``mean_depth`` here would overstate it ~6×.
        """
        return self.layer_passes_per_token(
            float(self.mean_depth), positions_per_token, core_position_frac)

    # ── absolute FLOPs ───────────────────────────────────────────────────────

    def step_flops(self, batch: int, seq_len: int, depth: float,
                   positions_per_token: float = 1.0,
                   core_position_frac: float | None = None,
                   backward_multiplier: float = 3.0,
                   density: float = 1.0,
                   sections: tuple[str, ...] | None = None) -> tuple[int, int]:
        """``(total_flops, attn_flops)`` for one optimizer step.

        ``sections`` restricts the count to the sections that actually ran — required for
        ``mode="b3"``, where one block runs per step. Without it b1 and b3 reported the
        SAME TFLOPs, which would have made the B=3 arm look like it saved nothing.

        ``backward_multiplier=3.0`` is the standard forward+backward convention (1 forward
        + ~2 for the backward). Gradient checkpointing pushes the effective figure toward
        4.0; the plan says to report which was used next to any MFU. ``density`` scales the
        MORTAR-eligible GEMMs — **1.0 for this whole campaign**, which runs dense (plan O3).
        """
        # `None` means "the core sees whatever the rest sees" — same default as
        # layer_passes_per_token. Without this, perf_metrics(core_position_frac=None) hit
        # `int * NoneType` at the first logging tick of every non-TUL run.
        if core_position_frac is None:
            core_position_frac = positions_per_token
        pos = int(round(seq_len * positions_per_token))
        core_pos = int(round(seq_len * core_position_frac))
        sec = self.ALL_SECTIONS if sections is None else sections

        # `other` (embeddings, LM head, mixers) is the shared readout: it runs for EVERY
        # block, which is why it is not gated on `sections`. The authors' head is shared the
        # same way (audit §3).
        gemm = self.gemm_other * pos
        attn = 0.0
        if "prelude" in sec:
            gemm += self.gemm_prelude * pos
            attn += self.attn_prelude * pos
        if "coda" in sec:
            gemm += self.gemm_coda * pos
            attn += self.attn_coda * pos
        if "core" in sec:
            gemm += self.gemm_core * core_pos * depth
            attn += self.attn_core * core_pos * depth
        gemm *= density
        total = (gemm + attn) * batch * backward_multiplier
        return int(total), int(attn * batch * backward_multiplier)

    def manifest(self) -> dict:
        """Full config of the FLOP model itself, for the wandb config dict."""
        return {
            "flops/version": self.version,
            "flops/gemm_prelude_per_token": self.gemm_prelude,
            "flops/gemm_core_per_token": self.gemm_core,
            "flops/gemm_coda_per_token": self.gemm_coda,
            "flops/gemm_other_per_token": self.gemm_other,
            "flops/attn_prelude_per_token": self.attn_prelude,
            "flops/attn_core_per_token": self.attn_core,
            "flops/attn_coda_per_token": self.attn_coda,
            "flops/nominal_depth": self.mean_depth,
            "flops/realized_depth": expected_clamped_poisson(
                float(self.mean_depth), 1, self.max_depth),
            "flops/nominal_proxy_a0": self.flop_proxy(),
            **{f"flops/assume_{k}": v for k, v in self.assumptions.items()},
        }


def attention_flops_per_token(d_model: int, n_heads: int, n_kv_heads: int, seq_len: int,
                              compression: int, csa_ratio: int, hca_ratio: int,
                              top_k: int, window: int) -> tuple[int, dict]:
    """MODELLED attention score/value FLOPs per token per layer.

    These are the terms that scale with sequence length rather than with a weight shape,
    so they cannot be read off the module tree. Returned with the assumption dict so the
    approximation is visible in every run's config rather than buried here.

    Model: CCA compresses the channel axis by ``compression`` before the score matmul.
    MORPH alternates a sparse-global branch (CSA: each query attends ``top_k`` compressed
    keys) with a dense-compressed branch (HCA: ``seq_len/hca_ratio`` keys), plus a local
    window. Score and value matmuls are each ``2·d_eff`` per (query, key) pair.
    """
    d_eff = max(d_model // max(compression, 1), 1)
    keys_csa = min(top_k, max(seq_len // max(csa_ratio, 1), 1))
    keys_hca = max(seq_len // max(hca_ratio, 1), 1)
    keys_local = min(window, seq_len)
    keys = keys_csa + keys_hca + keys_local
    # score (q·k) + value (p·v), 2 FLOPs per MAC each.
    flops = 2 * 2 * d_eff * keys
    return int(flops), {
        "attn_d_eff": d_eff,
        "attn_keys_csa": keys_csa,
        "attn_keys_hca": keys_hca,
        "attn_keys_local": keys_local,
        "attn_keys_total": keys,
        "attn_note": "MODELLED, not measured; validate against ncu once (plan A3/G2)",
    }


def build_flop_model(model: nn.Module, cfg, seq_len: int) -> FlopModel:
    """Walk the real module tree and build the FLOP model for this configuration.

    The weight-GEMM half is EXACT: every linear-like submodule's ``in_features`` and
    ``out_features`` are read off the constructed model, so a config change, a different
    ``d_ff``, or an extra coda layer are all picked up with no edit here.
    """
    mc = cfg.model
    n_prelude, n_core, n_coda = int(mc.n_prelude), int(mc.n_core), int(mc.n_coda)

    buckets = {"prelude": 0, "core": 0, "coda": 0, "other": 0}
    for path, mod in model.named_modules():
        f = _linear_flops(mod)
        if f:
            buckets[_region_of(path, n_prelude, n_core)] += f

    attn_per_layer, assumptions = attention_flops_per_token(
        d_model=int(mc.d_model),
        n_heads=int(mc.n_heads),
        n_kv_heads=int(getattr(mc, "n_kv_heads", mc.n_heads)),
        seq_len=seq_len,
        compression=int(getattr(mc, "compression", 1)),
        csa_ratio=int(getattr(mc, "csa_compress_ratio", 1)),
        hca_ratio=int(getattr(mc, "hca_compress_ratio", 1)),
        top_k=int(getattr(mc, "top_k", seq_len)),
        window=int(getattr(mc, "window_size", seq_len)),
    )
    assumptions["backward_multiplier_default"] = 3.0
    assumptions["counted"] = "weight GEMMs (exact) + attention score/value (modelled)"
    assumptions["not_counted"] = "norms, activations, softmax, HC Cayley, RoPE/CoPE"

    return FlopModel(
        version=FLOP_MODEL_VERSION,
        d_model=int(mc.d_model),
        n_prelude=n_prelude,
        n_core=n_core,
        n_coda=n_coda,
        mean_depth=int(mc.mean_depth),
        max_depth=int(mc.max_depth),
        seq_len=seq_len,
        gemm_prelude=buckets["prelude"],
        gemm_core=buckets["core"],
        gemm_coda=buckets["coda"],
        gemm_other=buckets["other"],
        attn_prelude=attn_per_layer * n_prelude,
        attn_core=attn_per_layer * n_core,
        attn_coda=attn_per_layer * n_coda,
        assumptions=assumptions,
    )


def perf_metrics(fm: FlopModel, *, batch: int, seq_len: int, step_time_s: float,
                 realized_depth: float | None = None,
                 positions_per_token: float = 1.0,
                 core_position_frac: float | None = None,
                 ceiling_tflops: float | None = None,
                 backward_multiplier: float = 3.0,
                 density: float = 1.0,
                 db_mode: str | None = None,
                 db_visit_probs: list[float] | None = None,
                 cum_tokens: float = 0.0,
                 cum_layer_passes: float = 0.0) -> dict:
    """The ``perf/*`` dict for one logging tick.

    ``ceiling_tflops`` must be a MEASURED dense-bf16 GEMM ceiling for this GPU at MORPH's
    shapes (gate G2), never a spec-sheet figure — quoting marketing TFLOPS would make every
    MFU in the campaign wrong in the same direction. ``None`` omits ``perf/mfu`` rather than
    inventing a denominator.
    """
    if realized_depth is None:
        realized_depth = expected_clamped_poisson(float(fm.mean_depth), 1, fm.max_depth)

    # A DiffusionBlocks step applies the core ONCE and (under b3) runs one section, so the
    # baseline proxy would overstate it several-fold. Branch on the declared mode rather
    # than inferring it, so a mislabelled run is a loud error and not a quiet wrong number.
    if db_mode == "b3":
        if not db_visit_probs:
            raise ValueError("db_mode='b3' needs db_visit_probs")
        proxy = fm.db_expected_passes(db_visit_probs, depth=1.0,
                                      positions_per_token=positions_per_token,
                                      core_position_frac=core_position_frac)
        realized_passes = proxy          # depth is 1 by construction: nothing to realize
        depth_for_flops = 1.0
    elif db_mode == "b1":
        proxy = fm.layer_passes_per_token(1.0, positions_per_token, core_position_frac)
        realized_passes = proxy
        depth_for_flops = 1.0
    else:
        proxy = fm.flop_proxy(positions_per_token, core_position_frac)
        realized_passes = fm.layer_passes_per_token(
            realized_depth, positions_per_token, core_position_frac)
        depth_for_flops = realized_depth

    _sf = dict(batch=batch, seq_len=seq_len, depth=depth_for_flops,
               positions_per_token=positions_per_token,
               core_position_frac=core_position_frac,
               backward_multiplier=backward_multiplier, density=density)
    if db_mode == "b3":
        # One section runs per step, so the comparable figure is the expectation over the
        # visit distribution — same reasoning as db_expected_passes.
        parts = [fm.step_flops(sections=(name,), **_sf) for name in fm.ALL_SECTIONS]
        total = int(sum(p * t for p, (t, _) in zip(db_visit_probs, parts)))
        attn = int(sum(p * a for p, (_, a) in zip(db_visit_probs, parts)))
    else:
        total, attn = fm.step_flops(**_sf)
    tflops = (total / step_time_s) / 1e12 if step_time_s > 0 else 0.0

    out = {
        # NOMINAL — the pre-registered, cross-arm comparable proxy. A0 = 44.0.
        "perf/flop_proxy": proxy,
        # ── Cross-arm alignment axes ────────────────────────────────────────
        # db_b1 and db_b3 cost DIFFERENT amounts per step (proxy 14.00 vs 4.67), so a loss
        # curve plotted against STEP compares equal DATA at unequal COMPUTE, and a curve
        # against compute compares equal compute at unequal data. Neither alone is a fair
        # read, so log both cumulative axes and report on both:
        #   step-matched    -> same cum_tokens     (equal data seen)
        #   compute-matched -> same cum_layer_passes (equal work done)
        # At proxy 4.67 vs 14.00, db_b3 needs 3.0x the steps of db_b1 for equal compute;
        # against A0's 44.0 it needs 9.4x. Without these two counters that arithmetic has to
        # be redone by hand for every comparison, which is how unfair tables get published.
        "perf/cum_tokens": cum_tokens,
        "perf/cum_layer_passes": cum_layer_passes,
        # REALIZED — what the sampled depths actually produced. A0 ≈ 42.1. Equal to the
        # nominal proxy for a DB arm, where the core depth is 1 by construction.
        "perf/layer_passes_per_token": realized_passes,
        "perf/db_mode": db_mode or "off",
        "perf/positions_per_token": positions_per_token,
        "perf/core_position_frac": core_position_frac,
        "perf/realized_depth": realized_depth,
        "perf/model_tflops": tflops,
        "perf/flops_attn_frac": (attn / total) if total else 0.0,
        "perf/flop_model_version": fm.version,
        "perf/backward_multiplier": backward_multiplier,
    }
    if ceiling_tflops and ceiling_tflops > 0:
        out["perf/mfu"] = tflops / ceiling_tflops
        out["perf/ceiling_tflops"] = ceiling_tflops
    return out
