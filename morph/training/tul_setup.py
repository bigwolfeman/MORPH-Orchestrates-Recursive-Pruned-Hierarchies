"""Resolve the Hydra ``tul:`` block into the objects the training loop needs.

Spec: ``docs/tul-spec.md`` §8 (config keys), §5 (schedule), §4 (data).

Kept out of ``train.py`` so the tokenizer work (resolving the boundary id set and the
slot token) is one testable unit, and so ``train.py``'s TUL seam is three lines. The
whole block resolves to ONE object; ``None`` means plain MORPH and nothing downstream
changes — no TUL parameters are constructed and the forward never sees a layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataclasses import replace as _dc_replace

from morph.model.tul import TULConfig, TULGateConfig
from morph.model.tul_layout import (
    BOUNDARY_SUBSTRINGS,
    BOUNDARY_SUFFIX_CHARS,
    BoundaryRule,
    TulDataConfig,
    TulGateSpec,
    boundary_lut_from_tokenizer,
)

__all__ = ["TulRuntime", "build_tul_runtime", "build_boundary_rule"]

NEVER = "never"


def build_boundary_rule(cfg, cache_dir: str = "ignore/tul_cache"):
    """``(rule, lut, eos_id, substrings)`` — THE span rule, from ``cfg.tul`` + the tokenizer.

    Extracted from :func:`build_tul_runtime` so the rule has ONE construction site. It is
    a property of the DATA, not of whether the slot apparatus is built, so it resolves
    even when ``tul.activate_at: never`` (arm A3) — which is exactly the case TUL-FM P1
    needs: a frozen A3 backbone has no slots, and the planner still has to segment rows
    with the same ``.;!?`` + newline/dash rule, the same ``min_span``, the same
    ``span_cap``, and the same EOS handling as every TUL arm.
    """
    tokenizer_name = str(cfg.data.tokenizer)
    vocab_size = int(cfg.model.vocab_size)
    tc = getattr(cfg, "tul", None)
    if tc is None:
        tc = {}

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = int(tok.eos_token_id if tok.eos_token_id is not None else 0)
    substrings = tuple(tc.get("boundary_substrings", BOUNDARY_SUBSTRINGS))
    lut = boundary_lut_from_tokenizer(
        tokenizer_name, vocab_size, eos_id,
        cache_dir=cache_dir,
        suffix_chars=str(tc.get("boundary_chars", BOUNDARY_SUFFIX_CHARS)),
        substrings=substrings,
    )
    rule = BoundaryRule(
        is_boundary=lut,
        min_span=int(tc.get("min_span", 4)),
        span_cap=int(tc.get("span_cap", 32)),
        eos_id=eos_id,
        fixed_stride=int(tc.get("fixed_stride", 0)),
    )
    return rule, lut, eos_id, substrings


@dataclass
class TulRuntime:
    """Everything TUL needs at runtime, resolved once at train start."""

    model_cfg: TULConfig
    data_cfg: TulDataConfig
    activate_at: float
    manifest: dict = field(default_factory=dict)

    @property
    def val_data_cfg(self) -> TulDataConfig:
        """The val loader's layout: the same segmentation with the gate augmentation OFF.

        docs/tul-gate-spec.md §3.2 truncates spans with OUR rng. A val CE measured over
        rng-truncated spans is not comparable to the reference arm's, and it would move
        with the seed. Val therefore always scores the data's own segmentation — which is
        also the segmentation the generation metrics are checked against.
        """
        if self.data_cfg.gate is None:
            return self.data_cfg
        return _dc_replace(self.data_cfg,
                           gate=_dc_replace(self.data_cfg.gate, truncate_p=0.0))

    def activation_step(self, total_steps: int) -> int:
        """Step at which the layout switches on (spec §5). 0 → active from step 0."""
        return int(self.activate_at * total_steps)


def build_tul_runtime(cfg, cache_dir: str = "ignore/tul_cache") -> TulRuntime | None:
    """Build the TUL runtime from ``cfg.tul``; ``None`` when TUL is off.

    ``tul.activate_at: never`` (the base.yaml default, arm A0) returns None, which is
    what makes the default recipe bit-identical to plain MORPH — no parameters, no
    layout, no branch (runtime-invariants §6b).
    """
    tc = getattr(cfg, "tul", None)
    if tc is None:
        return None
    raw = tc.get("activate_at", NEVER)
    if raw is None or (isinstance(raw, str) and str(raw).lower() == NEVER):
        return None
    activate_at = float(raw)
    if not 0.0 <= activate_at < 1.0:
        raise ValueError(f"tul.activate_at must be in [0,1) or 'never', got {raw!r}")

    tokenizer_name = str(cfg.data.tokenizer)
    vocab_size = int(cfg.model.vocab_size)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    slot_token = str(tc.get("slot_token", "<fim_pad>"))
    slot_id = tok.convert_tokens_to_ids(slot_token)
    if slot_id is None or slot_id < 0 or slot_id >= vocab_size:
        raise ValueError(
            f"tul.slot_token {slot_token!r} does not resolve to a valid id for "
            f"{tokenizer_name} (got {slot_id}, vocab {vocab_size})")

    rule, lut, eos_id, substrings = build_boundary_rule(cfg, cache_dir=cache_dir)
    if slot_id == eos_id:
        raise ValueError("tul.slot_token resolves to EOS — pick an unused special token")
    if bool(lut[slot_id]):
        raise ValueError(
            f"tul.slot_token {slot_token!r} (id {slot_id}) is itself a boundary token — "
            f"it would cut spans it is only supposed to mark")

    prefix_k = int(tc.get("prefix_k", 2))

    # ── the span-length gate (docs/tul-gate-spec.md §1, §3, §12) ──────────────
    # `tul.gate: false` ⇒ gate_cfg and gate_spec are both None ⇒ no parameter is built,
    # the packer draws no random number, and the arm IS arm A1 (§9 invariant 1).
    gate_cfg = gate_spec = None
    if bool(tc.get("gate", False)):
        gate_cfg = TULGateConfig(
            k_max=int(tc.get("gate_k_max", 40)),
            k_decode_max=rule.span_cap,      # never ask for more than the rule can give
            train_zeros=bool(tc.get("gate_train_zeros", False)),
            lam=float(tc.get("gate_lambda", 1.0)),
            budget_cond=bool(tc.get("gate_budget_cond", True)),
            huber_beta=float(tc.get("gate_huber_beta", 1.0)),
            drives_depth=bool(tc.get("gate_drives_depth", False)),
            scheduled_sampling=float(tc.get("gate_scheduled_sampling", 0.0)),
            stop_head=bool(tc.get("gate_stop_head", False)),
            ponder_lambda=float(tc.get("gate_ponder_lambda", 0.0)),
        )
        gate_spec = TulGateSpec(k_max=gate_cfg.k_max,
                                truncate_p=float(tc.get("gate_truncate_p", 0.15)))
        if rule.span_cap > gate_cfg.k_max:
            raise ValueError(
                f"tul.span_cap={rule.span_cap} > tul.gate_k_max={gate_cfg.k_max}: the "
                f"length label span_len/k_max would saturate on the longest spans.")

    data_cfg = TulDataConfig(rule=rule, prefix_k=prefix_k, slot_id=int(slot_id),
                             max_slots=int(tc.get("max_slots", 0)),
                             gate=gate_spec, seed=int(tc.get("gate_seed", 0)))
    model_cfg = TULConfig(
        gate=gate_cfg,
        prefix_k=prefix_k,
        slot_id=int(slot_id),
        token_state_dropout=float(tc.get("token_state_dropout", 0.15)),
        slot_mean_depth=int(tc.get("slot_mean_depth", 0)),
        slot_max_depth=int(tc.get("slot_max_depth", 0)),
        coda_sees_slots=bool(tc.get("coda_sees_slots", True)),
        tokens_through_core=bool(tc.get("tokens_through_core", False)),
        stp_lambda=float(tc.get("stp_lambda", 0.0)),
        set_lambda=float(tc.get("set_lambda", 0.0)),
        carry=bool(tc.get("carry", False)),
        xattn=bool(tc.get("xattn", False)),
        bcast=bool(tc.get("bcast", False)),
        coda_token_cut=int(tc.get("coda_token_cut", 0)),
        emit_weight=float(tc.get("emit_weight", 0.5)),
        plast_weight=float(tc.get("plast_weight", 0.5)),
        mux_beta=float(tc.get("mux_beta", 0.0)),
        mux_rho=float(tc.get("mux_rho", 0.9)),
        mux_tau=float(tc.get("mux_tau", 1.0)),
        mux_detach_head=bool(tc.get("mux_detach_head", True)),
        mux_target=str(tc.get("mux_target", "next")),
        db_loop=bool(tc.get("db_loop", False)),
        db_mux_iters=int(tc.get("db_mux_iters", 4)),
        core_stage_cond=str(tc.get("core_stage_cond", "none")),
        recur_gate=str(tc.get("recur_gate", "none")),
        recur_gate_bias=float(tc.get("recur_gate_bias", 4.0)),
        recur_gate_noise=float(tc.get("recur_gate_noise", 0.1)),
        recur_gate_tau=float(tc.get("recur_gate_tau", 1.0)),
        db1_cond_dim=int(tc.get("db1_cond_dim", 256)),
        db1_sigma_min=float(tc.get("db1_sigma_min", 0.002)),
        db1_sigma_max=float(tc.get("db1_sigma_max", 80.0)),
        db1_p_mean=float(tc.get("db1_p_mean", -1.2)),
        db1_p_std=float(tc.get("db1_p_std", 1.2)),
        db1_sigma_data=float(tc.get("db1_sigma_data", 0.5)),
        db1_w_sigma=bool(tc.get("db1_w_sigma", False)),
        db1_ladder_steps=int(tc.get("db1_ladder_steps", 0)),
        center_bag_mean=bool(tc.get("center_bag_mean", False)),
        mux_activate_at=float(tc.get("mux_activate_at", 0.0)),
        sigreg_lambda=float(tc.get("sigreg_lambda", 0.0)),
        sigreg_slices=int(tc.get("sigreg_slices", 256)),
        sigreg_activate_at=float(tc.get("sigreg_activate_at", 0.0)),
        tg_restrict=bool(tc.get("tg_restrict", False)),
        tg_span_comp=bool(tc.get("tg_span_comp", False)),
        tg_span_gate=bool(tc.get("tg_span_gate", False)),
        tg_soft_prev_span=bool(tc.get("tg_soft_prev_span", False)),
        slot_seed=str(tc.get("slot_seed", "bag_mean")),
        # Sized from the DATA's own span_cap, never a separate config key: `bound_R`
        # (built only when slot_seed=="bound") must never silently disagree with what
        # the loader can actually produce — see TULConfig.bound_span_cap's docstring.
        bound_span_cap=rule.span_cap,
        eval_ablations=bool(tc.get("eval_ablations", False)),
    )
    seq_len = int(cfg.data.seq_len)
    spec = data_cfg.spec_for(seq_len)
    # Per-slot input embedding: `tul.per_slot_embed: true` sizes it from the DERIVED slot
    # budget, so it cannot silently disagree with the layout's max_slots. An int is honoured
    # as-is for the odd case where someone wants a different number.
    _pse = tc.get("per_slot_embed", 0)
    model_cfg.per_slot_embed = (spec.max_slots if _pse is True
                                else 0 if _pse is False else int(_pse))
    model_cfg.per_slot_embed_std = float(tc.get("per_slot_embed_std", 0.0))
    # Everything that is DERIVED rather than typed goes into the wandb config, so a run
    # is reproducible from its config alone (no re-deriving ids from a tokenizer version).
    manifest = {
        "activate_at": activate_at,
        "slot_token": slot_token,
        "slot_id": int(slot_id),
        "eos_id": eos_id,
        "n_boundary_ids": int(lut.sum()),
        "emit_weight": model_cfg.emit_weight,
        "plast_weight": model_cfg.plast_weight,
        "mux_beta": model_cfg.mux_beta,
        "db_loop": model_cfg.db_loop,
        "db_mux_iters": model_cfg.db_mux_iters,
        "core_stage_cond": model_cfg.core_stage_cond,
        "recur_gate": model_cfg.recur_gate,
        "db1_cond_dim": model_cfg.db1_cond_dim,
        "db1_sigma_min": model_cfg.db1_sigma_min,
        "db1_sigma_max": model_cfg.db1_sigma_max,
        "db1_p_mean": model_cfg.db1_p_mean,
        "db1_p_std": model_cfg.db1_p_std,
        "db1_sigma_data": model_cfg.db1_sigma_data,
        "db1_w_sigma": model_cfg.db1_w_sigma,
        "db1_ladder_steps": model_cfg.db1_ladder_steps,
        "mux_rho": model_cfg.mux_rho,
        "mux_tau": model_cfg.mux_tau,
        "mux_detach_head": model_cfg.mux_detach_head,
        "mux_target": model_cfg.mux_target,
        "center_bag_mean": model_cfg.center_bag_mean,
        "mux_activate_at": model_cfg.mux_activate_at,
        "sigreg_lambda": model_cfg.sigreg_lambda,
        "sigreg_slices": model_cfg.sigreg_slices,
        "sigreg_activate_at": model_cfg.sigreg_activate_at,
        "tg_restrict": model_cfg.tg_restrict,
        "tg_span_comp": model_cfg.tg_span_comp,
        "tg_span_gate": model_cfg.tg_span_gate,
        "tg_soft_prev_span": model_cfg.tg_soft_prev_span,
        "slot_seed": model_cfg.slot_seed,
        "bound_span_cap": model_cfg.bound_span_cap,
        "eval_ablations": model_cfg.eval_ablations,
        "boundary_chars": str(tc.get("boundary_chars", BOUNDARY_SUFFIX_CHARS)),
        "boundary_substrings": list(substrings),
        "min_span": rule.min_span,
        "span_cap": rule.span_cap,
        "fixed_stride": rule.fixed_stride,
        "prefix_k": prefix_k,
        "seq_len": seq_len,
        "max_slots": spec.max_slots,
        "l_total": spec.l_total,
        "token_state_dropout": model_cfg.token_state_dropout,
        "coda_sees_slots": model_cfg.coda_sees_slots,
        "tokens_through_core": model_cfg.tokens_through_core,
        "coda_token_cut": model_cfg.coda_token_cut,
        "per_slot_embed": model_cfg.per_slot_embed,
        "per_slot_embed_std": model_cfg.per_slot_embed_std,
        "slot_mean_depth": model_cfg.slot_mean_depth or int(cfg.model.mean_depth),
        "slot_max_depth": model_cfg.slot_max_depth or int(cfg.model.max_depth),
        "gate": gate_cfg is not None,
        "gate_k_max": gate_cfg.k_max if gate_cfg else None,
        "gate_k_decode_max": gate_cfg.k_decode_max if gate_cfg else None,
        "gate_train_zeros": gate_cfg.train_zeros if gate_cfg else None,
        "gate_lambda": gate_cfg.lam if gate_cfg else None,
        "gate_budget_cond": gate_cfg.budget_cond if gate_cfg else None,
        "gate_huber_beta": gate_cfg.huber_beta if gate_cfg else None,
        "gate_drives_depth": gate_cfg.drives_depth if gate_cfg else None,
        "gate_truncate_p": gate_spec.truncate_p if gate_spec else None,
        "gate_seed": data_cfg.seed,
    }
    print(f"  TUL ON: activate_at={activate_at} slot_token={slot_token!r}(id {slot_id}) "
          f"|B|={int(lut.sum())} min_span={rule.min_span} span_cap={rule.span_cap} "
          f"prefix_k={prefix_k} max_slots={spec.max_slots} L_total={spec.l_total} "
          f"p_drop={model_cfg.token_state_dropout} slot_seed={model_cfg.slot_seed!r}"
          + (f" fixed_stride={rule.fixed_stride}" if rule.fixed_stride else "")
          + (f" coda_token_cut={model_cfg.coda_token_cut}" if model_cfg.coda_token_cut else ""),
          flush=True)
    if model_cfg.recur_gate != "none":
        print(f"  TUL RECUR GATE ON: {model_cfg.recur_gate} "
              f"bias={model_cfg.recur_gate_bias} sigma_g={model_cfg.recur_gate_noise} "
              f"tau={model_cfg.recur_gate_tau} (morph/model/recur_gate.py)", flush=True)
    if gate_cfg is not None:
        print(f"  TUL GATE ON: k_max={gate_cfg.k_max}"
              f"(decode≤{gate_cfg.k_decode_max}) lambda={gate_cfg.lam} "
              f"budget_cond={gate_cfg.budget_cond} truncate_p={gate_spec.truncate_p} "
              f"huber_beta={gate_cfg.huber_beta} drives_depth={gate_cfg.drives_depth} "
              f"seed={data_cfg.seed} (docs/tul-gate-spec.md)", flush=True)
    if model_cfg.tg_restrict:
        print(f"  TUL TG-RESTRICT ON: soft_prev_span={model_cfg.tg_soft_prev_span} "
              f"— window branch restricted to same-span-or-slot, compressed branch "
              f"restricted to slot positions (docs/tul-tg-spec.md); model.use_kernels "
              f"must be false", flush=True)
    if model_cfg.tg_span_comp:
        print("  TUL TG SPAN-COMP ON: compressed branch = per-span mean-pooled "
              "live K/V (E-SAC; zero new params; span-granular causality)", flush=True)
    if model_cfg.tg_span_gate:
        print("  TUL TG SPAN-GATE ON: learned per-head gated softmax span pool "
              "(E-SAC-G; one zero-init [H,D] gate per attn layer — mean pool at "
              "init)", flush=True)
    return TulRuntime(model_cfg=model_cfg, data_cfg=data_cfg,
                      activate_at=activate_at, manifest=manifest)
