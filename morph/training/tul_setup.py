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

__all__ = ["TulRuntime", "build_tul_runtime", "build_boundary_rule",
           "KNOWN_TUL_KEYS", "reject_unknown_tul_keys"]

# Every `tul:` key this module reads. A key outside this set is a typo or a knob from a
# branch this tree does not carry, and either one must RAISE rather than run a model under a
# name that promises something else (runtime-invariants §6b). Built from the `tc.get(...)`
# reads below; tests/test_tul_setup_keys.py checks every shipped config against it.
KNOWN_TUL_KEYS = frozenset({
    "activate_at", "bcast", "boundary_chars", "boundary_substrings", "carry",
    "center_bag_mean", "coda_sees_slots", "coda_span_heads", "coda_span_source",
    "coda_span_weight", "coda_token_cut", "coda_token_input", "cond_layers",
    "core_stage_cond", "core_token_aux", "core_token_aux_weight",
    "critic_every", "critic_eps", "critic_replay_groups", "critic_weight",
    "db1_cond_dim", "db1_ladder_steps", "db1_p_mean", "db1_p_std", "db1_sigma_data",
    "db1_sigma_max", "db1_sigma_min", "db1_w_sigma", "db_loop", "db_mux_iters",
    "detach_z", "emit_weight", "eval_ablations", "fixed_stride", "gate",
    "gate_budget_cond", "gate_drives_depth", "gate_huber_beta", "gate_k_max", "gate_lambda",
    "gate_ponder_lambda", "gate_scheduled_sampling", "gate_seed", "gate_stop_head", "gate_train_zeros",
    "gate_truncate_p", "egrad_disc_hidden", "egrad_heads", "egrad_layers", "egrad_max_tokens",
    "egrad_soft_labels", "egrad_soft_mix", "egrad_weight",
    "grad_pass", "grad_pass_energy", "grad_pass_norm", "grad_pass_scale",
    "loop_reach", "loop_reads_tokens", "max_slots", "min_span",
    "mux_activate_at", "mux_beta",
    "mux_detach_head", "mux_every_pass", "mux_readout", "mux_rho", "mux_stage_all",
    "mux_stage_own_iters",
    "mux_target",
    "mux_tau",
    "oracle_z", "oracle_z_lr", "oracle_z_max_tokens", "oracle_z_steps",
    "oracle_z_weight",
    "per_slot_embed",
    "per_slot_embed_std", "pass_lora_rank", "pass_lora_targets",
    "pass_residual_lambda", "plast_weight", "prefix_k", "prefix_source", "progressive_p",
    "reinject_seed_every_pass", "recur_gate", "recur_gate_bias",
    "recur_gate_noise", "recur_gate_tau", "set_lambda", "sigreg_activate_at", "sigreg_lambda",
    "sigreg_slices", "slot_chain", "slot_chain_detach",
    "slot_depth_fixed", "slot_max_depth", "slot_mean_depth", "slot_seed", "slot_token",
    "spandec", "spandec_heads", "spandec_horizon", "spandec_layers", "spandec_max_tokens",
    "spandec_pass_horizon_max", "spandec_pass_tokens", "spandec_pass_weight",
    "spandec_per_pass", "spandec_weight",
    "reread", "reread_heads", "reread_scope", "span_cap", "stp_lambda",
    "tg_coda_prefix_reach", "tg_geometry",
    "tg_restrict", "tg_restrict_scope", "tg_soft_prev_span", "tg_span_comp",
    "tg_span_gate", "token_state_dropout", "tokens_through_core", "xattn",
})


def reject_unknown_tul_keys(tc) -> None:
    """Raise ``ValueError`` naming every ``tul:`` key outside :data:`KNOWN_TUL_KEYS`."""
    unknown = sorted(str(k) for k in tc.keys() if str(k) not in KNOWN_TUL_KEYS)
    if unknown:
        raise ValueError(
            f"tul: has unknown key(s) {unknown}. Known keys: {sorted(KNOWN_TUL_KEYS)}.")

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
    if tc is not None and hasattr(tc, "keys"):
        reject_unknown_tul_keys(tc)
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
    if tc is not None and hasattr(tc, "keys"):
        reject_unknown_tul_keys(tc)
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
        slot_depth_fixed=int(tc.get("slot_depth_fixed", 0)),
        coda_sees_slots=bool(tc.get("coda_sees_slots", True)),
        tokens_through_core=bool(tc.get("tokens_through_core", False)),
        stp_lambda=float(tc.get("stp_lambda", 0.0)),
        set_lambda=float(tc.get("set_lambda", 0.0)),
        carry=bool(tc.get("carry", False)),
        xattn=bool(tc.get("xattn", False)),
        bcast=bool(tc.get("bcast", False)),
        reread=bool(tc.get("reread", False)),
        reread_heads=int(tc.get("reread_heads", 8)),
        reread_scope=str(tc.get("reread_scope", "causal")),
        progressive_p=float(tc.get("progressive_p", 0.0)),
        pass_lora_rank=int(tc.get("pass_lora_rank", 0)),
        pass_lora_targets=tuple(tc.get("pass_lora_targets", ("attn", "mlp"))),
        coda_token_cut=int(tc.get("coda_token_cut", 0)),
        emit_weight=float(tc.get("emit_weight", 0.5)),
        plast_weight=float(tc.get("plast_weight", 0.5)),
        mux_beta=float(tc.get("mux_beta", 0.0)),
        mux_rho=float(tc.get("mux_rho", 0.9)),
        mux_tau=float(tc.get("mux_tau", 1.0)),
        mux_detach_head=bool(tc.get("mux_detach_head", True)),
        mux_target=str(tc.get("mux_target", "next")),
        mux_readout=str(tc.get("mux_readout", "mean")),
        mux_stage_own_iters=int(tc.get("mux_stage_own_iters", 0)),
        mux_stage_all=bool(tc.get("mux_stage_all", False)),
        mux_every_pass=bool(tc.get("mux_every_pass", False)),
        spandec=bool(tc.get("spandec", False)),
        spandec_layers=int(tc.get("spandec_layers", 2)),
        spandec_heads=int(tc.get("spandec_heads", 0)),
        spandec_weight=float(tc.get("spandec_weight", 1.0)),
        spandec_max_tokens=int(tc.get("spandec_max_tokens", 0)),
        spandec_horizon=int(tc.get("spandec_horizon", 1)),
        spandec_per_pass=bool(tc.get("spandec_per_pass", False)),
        spandec_pass_horizon_max=int(tc.get("spandec_pass_horizon_max", 6)),
        spandec_pass_weight=float(tc.get("spandec_pass_weight", 1.0)),
        spandec_pass_tokens=int(tc.get("spandec_pass_tokens", 8)),
        coda_span_heads=int(tc.get("coda_span_heads", 0)),
        coda_span_weight=float(tc.get("coda_span_weight", 1.0)),
        coda_span_source=str(tc.get("coda_span_source", "cell")),
        critic_weight=float(tc.get("critic_weight", 1.0)),
        critic_every=int(tc.get("critic_every", 1)),
        critic_eps=float(tc.get("critic_eps", 0.1)),
        critic_replay_groups=int(tc.get("critic_replay_groups", 1)),
        prefix_source=str(tc.get("prefix_source", "exit")),
        loop_reads_tokens=bool(tc.get("loop_reads_tokens", False)),
        core_token_aux=bool(tc.get("core_token_aux", False)),
        core_token_aux_weight=float(tc.get("core_token_aux_weight", 1.0)),
        slot_chain=bool(tc.get("slot_chain", False)),
        slot_chain_detach=bool(tc.get("slot_chain_detach", False)),
        grad_pass=bool(tc.get("grad_pass", False)),
        grad_pass_scale=float(tc.get("grad_pass_scale", 0.1)),
        grad_pass_norm=str(tc.get("grad_pass_norm", "rms")),
        grad_pass_energy=str(tc.get("grad_pass_energy", "own_mux")),
        egrad_weight=float(tc.get("egrad_weight", 1.0)),
        egrad_layers=int(tc.get("egrad_layers", 2)),
        egrad_heads=int(tc.get("egrad_heads", 0)),
        egrad_max_tokens=int(tc.get("egrad_max_tokens", 8)),
        egrad_soft_labels=bool(tc.get("egrad_soft_labels", False)),
        egrad_soft_mix=float(tc.get("egrad_soft_mix", 0.5)),
        egrad_disc_hidden=int(tc.get("egrad_disc_hidden", 0)),
        pass_residual_lambda=float(tc.get("pass_residual_lambda", 0.0)),
        reinject_seed_every_pass=bool(tc.get("reinject_seed_every_pass", False)),
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
        tg_restrict_scope=str(tc.get("tg_restrict_scope", "all")),
        tg_geometry=str(tc.get("tg_geometry", "restrict")),
        tg_coda_prefix_reach=str(tc.get("tg_coda_prefix_reach", "all")),
        loop_reach=int(tc.get("loop_reach", 0)),
        oracle_z=bool(tc.get("oracle_z", False)),
        oracle_z_steps=int(tc.get("oracle_z_steps", 6)),
        oracle_z_lr=float(tc.get("oracle_z_lr", 0.1)),
        oracle_z_weight=float(tc.get("oracle_z_weight", 1.0)),
        oracle_z_max_tokens=int(tc.get("oracle_z_max_tokens", 8)),
        coda_token_input=str(tc.get("coda_token_input", "prelude")),
        tg_span_comp=bool(tc.get("tg_span_comp", False)),
        tg_span_gate=bool(tc.get("tg_span_gate", False)),
        tg_soft_prev_span=bool(tc.get("tg_soft_prev_span", False)),
        slot_seed=str(tc.get("slot_seed", "bag_mean")),
        # Sized from the DATA's own span_cap, never a separate config key: `bound_R`
        # (built only when slot_seed=="bound") must never silently disagree with what
        # the loader can actually produce — see TULConfig.bound_span_cap's docstring.
        bound_span_cap=rule.span_cap,
        eval_ablations=bool(tc.get("eval_ablations", False)),
        cond_layers=int(tc.get("cond_layers", 0)),
        detach_z=bool(tc.get("detach_z", False)),
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
        "mux_readout": model_cfg.mux_readout,
        "mux_stage_own_iters": model_cfg.mux_stage_own_iters,
        "mux_stage_all": model_cfg.mux_stage_all,
        "mux_every_pass": model_cfg.mux_every_pass,
        "spandec": model_cfg.spandec,
        "spandec_layers": model_cfg.spandec_layers,
        "spandec_heads": model_cfg.spandec_heads,
        "spandec_weight": model_cfg.spandec_weight,
        # DERIVED, so the run is reproducible from its wandb config alone: 0 means "the
        # data's span_cap", which is itself derived from the boundary rule.
        "spandec_max_tokens": (model_cfg.spandec_max_tokens or model_cfg.bound_span_cap),
        "spandec_horizon": model_cfg.spandec_horizon,
        "spandec_per_pass": model_cfg.spandec_per_pass,
        "spandec_pass_horizon_max": model_cfg.spandec_pass_horizon_max,
        "spandec_pass_weight": model_cfg.spandec_pass_weight,
        "spandec_pass_tokens": model_cfg.spandec_pass_tokens,
        "coda_span_heads": model_cfg.coda_span_heads,
        "coda_span_weight": model_cfg.coda_span_weight,
        "coda_span_source": model_cfg.coda_span_source,
        "grad_pass_energy": model_cfg.grad_pass_energy,
        "egrad_weight": model_cfg.egrad_weight,
        "egrad_max_tokens": model_cfg.egrad_max_tokens,
        "egrad_disc_hidden": model_cfg.egrad_disc_hidden,
        "critic_weight": model_cfg.critic_weight,
        "critic_every": model_cfg.critic_every,
        "critic_eps": model_cfg.critic_eps,
        "critic_replay_groups": model_cfg.critic_replay_groups,
        "pass_residual_lambda": model_cfg.pass_residual_lambda,
        "prefix_source": model_cfg.prefix_source,
        "loop_reads_tokens": model_cfg.loop_reads_tokens,
        "core_token_aux": model_cfg.core_token_aux,
        "core_token_aux_weight": model_cfg.core_token_aux_weight,
        "slot_chain": model_cfg.slot_chain,
        "slot_chain_detach": model_cfg.slot_chain_detach,
        "grad_pass": model_cfg.grad_pass,
        "grad_pass_scale": model_cfg.grad_pass_scale,
        "grad_pass_norm": model_cfg.grad_pass_norm,
        "center_bag_mean": model_cfg.center_bag_mean,
        "mux_activate_at": model_cfg.mux_activate_at,
        "sigreg_lambda": model_cfg.sigreg_lambda,
        "sigreg_slices": model_cfg.sigreg_slices,
        "sigreg_activate_at": model_cfg.sigreg_activate_at,
        "tg_restrict": model_cfg.tg_restrict,
        "tg_restrict_scope": model_cfg.tg_restrict_scope,
        "tg_geometry": model_cfg.tg_geometry,
        "tg_coda_prefix_reach": model_cfg.tg_coda_prefix_reach,
        "loop_reach": model_cfg.loop_reach,
        "oracle_z": model_cfg.oracle_z,
        "oracle_z_steps": model_cfg.oracle_z_steps,
        "oracle_z_lr": model_cfg.oracle_z_lr,
        "oracle_z_weight": model_cfg.oracle_z_weight,
        "oracle_z_max_tokens": model_cfg.oracle_z_max_tokens,
        "coda_token_input": model_cfg.coda_token_input,
        "bcast": model_cfg.bcast,
        "reread": model_cfg.reread,
        "reread_heads": model_cfg.reread_heads,
        "reread_scope": model_cfg.reread_scope,
        "progressive_p": model_cfg.progressive_p,
        "pass_lora_rank": model_cfg.pass_lora_rank,
        "pass_lora_targets": list(model_cfg.pass_lora_targets),
        "tg_span_comp": model_cfg.tg_span_comp,
        "tg_span_gate": model_cfg.tg_span_gate,
        "tg_soft_prev_span": model_cfg.tg_soft_prev_span,
        "slot_seed": model_cfg.slot_seed,
        "bound_span_cap": model_cfg.bound_span_cap,
        "eval_ablations": model_cfg.eval_ablations,
        "cond_layers": model_cfg.cond_layers,
        "detach_z": model_cfg.detach_z,
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
        "slot_depth_fixed": int(model_cfg.slot_depth_fixed),
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
    if model_cfg.cond_layers > 0 or model_cfg.detach_z:
        print(f"  TUL THINK-ONCE: cond_layers={model_cfg.cond_layers} "
              f"detach_z={model_cfg.detach_z} (arms R7/R8; the coda reads the "
              f"conditioning stack's output{', with stop-gradient' if model_cfg.detach_z else ''})",
              flush=True)
    if model_cfg.spandec:
        print(f"  TUL SPAN DECODER ON: layers={model_cfg.spandec_layers} "
              f"heads={model_cfg.spandec_heads or int(cfg.model.n_heads)} "
              f"J={model_cfg.spandec_max_tokens or model_cfg.bound_span_cap}"
              f"x{model_cfg.spandec_horizon} "
              f"weight={model_cfg.spandec_weight} — the next span is decoded from z with a "
              f"teacher-forced token path (morph/model/tul_spandec.py)", flush=True)
    if model_cfg.spandec_per_pass:
        _cap = model_cfg.spandec_pass_horizon_max
        _pt = model_cfg.spandec_pass_tokens
        print(f"  TUL SPANDEC PER-PASS ON: pass t is graded on spans s+1..s+min(t,{_cap}) "
              f"at {_pt} tokens per span, through the SAME decoder, weight "
              f"{model_cfg.spandec_pass_weight} — the exit term stays H=1 (the next "
              f"thought). Cost: sum_t min(t,{_cap})*{_pt} decoded positions per slot per "
              f"step ({sum(min(t, _cap) * _pt for t in range(1, _cap + 1))} at depth "
              f"{_cap}) on its OWN zero-init position table "
              f"(lab/experiments/planned/2026-09-12-arc-objective-arms.md)", flush=True)
    if model_cfg.coda_span_heads > 0:
        print(f"  TUL CODA SPAN HEADS ON: {model_cfg.coda_span_heads} parallel offset heads "
              f"on the coda readout at each slot's "
              f"{'last prefix cell' if model_cfg.coda_span_source == 'cell' else 'boundary TOKEN position (a CONTROL: that state never saw its own slot z)'}"
              f", predicting the next span's tokens 1..{model_cfg.coda_span_heads} "
              f"NON-autoregressively, weight {model_cfg.coda_span_weight} "
              f"(lab/experiments/planned/2026-09-12-arc-objective-arms.md)", flush=True)
    if model_cfg.grad_pass_energy == "critic":
        print(f"  TUL CRITIC ENERGY ON (weight {model_cfg.critic_weight}, every "
              f"{model_cfg.critic_every} forward(s), eps {model_cfg.critic_eps}, "
              f"{model_cfg.critic_replay_groups} replay group(s)): every pass is "
              f"conditioned on the gradient of a scorer trained by a PAIRWISE comparison "
              f"of two candidate slot states through a REPLAY of the real coda "
              f"(h_t-1 vs h_t, and h_t vs h_t + eps*rms*n). Every replay is no_grad and "
              f"the whole label sits in an RNG save/restore. "
              f"{3 * model_cfg.critic_replay_groups} extra coda forwards per step, no "
              f"backward. Read `critic_agree` (0.5 = the energy carries nothing) and "
              f"`critic_gap_traj` (the MEASURED worth of one pass) "
              f"(lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md)",
              flush=True)
    if model_cfg.prefix_source != "exit":
        print(f"  TUL PREFIX SOURCE = {model_cfg.prefix_source!r} (prefix_k "
              f"{model_cfg.prefix_k}): 'trajectory' gives cell k of a slot the state AFTER "
              f"PASS k+1 and the LAST cell the EXIT state, so every written pass has its "
              f"own reader and its own gradient edge; a cell no pass reached is a PAD "
              f"(zero carrier, cut out of the coda's key set, no label). 'exit_repeat' is "
              f"the matched-count CONTROL: every cell holds the exit. Both add the "
              f"zero-init per-cell embedding `tul.E_pass`, so trajectory minus exit_repeat "
              f"isolates the cell CONTENT "
              f"(lab/experiments/planned/2026-09-13-arc-trajectory-prefix.md)",
              flush=True)
    if model_cfg.loop_reads_tokens:
        print(f"  TUL LOOP READS TOKENS ON: the SHIPPED core stage runs over EVERY "
              f"position (tokens + slot cells, one sequence, `_core_region`'s per-SAMPLE "
              f"Poisson depth) under `causal AND (same span OR a slot cell)`, so a token "
              f"reaches an earlier span ONLY through a cell and the cells are still the "
              f"whole cross-span channel. NOT the paid loop, whose core is unrestricted. "
              f"No prefix write (the cell's state is already at its position); z is "
              f"gathered at the slot's first cell. Forced-depth eval goes through "
              f"model.cfg.mean_depth "
              f"(lab/experiments/planned/2026-09-13-arc-loop-reads-tokens.md)",
              flush=True)
    if model_cfg.core_token_aux:
        print(f"  TUL CORE-TOKEN AUX ON (weight {model_cfg.core_token_aux_weight}): a "
              f"TRAINING-ONLY second pass sends EVERY position (tokens + cells) through "
              f"the per-sample Poisson-depth core and the coda, and charges the ordinary "
              f"weighted token CE as `core_token_aux`. The SHIPPED forward is unchanged — "
              f"eval, the sweeps and inference still run the slot loop with tokens OUTSIDE "
              f"the core. The aux core is masked to `causal AND (same span OR a slot "
              f"cell)` with the segment reset, so tokens do NOT become a cross-span "
              f"channel. COST: ~45 extra core+coda block-passes per real token, roughly "
              f"one extra paid-loop forward+backward per step "
              f"(lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md)",
              flush=True)
    if model_cfg.slot_chain:
        print(f"  TUL SLOT CHAIN ON: detach={model_cfg.slot_chain_detach} — slot k takes "
              f"W(z_k-1) at every pass (zero-init; the wavefront form of the chain)",
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
    if model_cfg.tg_geometry == "strict":
        print(f"  TUL STRICT GEOMETRY ON (coda_prefix_reach={model_cfg.tg_coda_prefix_reach}): "
              f"the prelude is same-span ONLY, a coda prefix cell reads itself alone, the "
              f"conv/value-shift and the retention carry reset at every segment, and the "
              f"coda's per-layer injections at the slot cells are zeroed — the slot LOOP is "
              f"the only cross-span channel "
              f"(lab/experiments/planned/2026-09-12-arc-strict-geometry.md)", flush=True)
    if model_cfg.oracle_z:
        print(f"  TUL ORACLE-Z ON: T={model_cfg.oracle_z_steps} lr={model_cfg.oracle_z_lr} "
              f"weight={model_cfg.oracle_z_weight} J={model_cfg.oracle_z_max_tokens} — each "
              f"pass is REGRESSED onto a detached descent trajectory of the span decoder's "
              f"loss. THIS BREAKS THE STANDING 'never regress onto the slot state' RULE "
              f"(LCM/CoCoMix/BT); it is a TEST of whether a per-pass target produces a "
              f"per-pass K-curve, not a shipped design", flush=True)
    if model_cfg.loop_reach > 0:
        print(f"  TUL LOOP REACH {model_cfg.loop_reach}: inside the loop a slot attends "
              f"slots k-{model_cfg.loop_reach}..k only, the compressed branch takes the "
              f"same relation and the CCA conv / value shift reset PER CELL — a slot m "
              f"spans back needs ceil(m/{model_cfg.loop_reach}) passes to reach k", flush=True)
    if model_cfg.tg_span_comp:
        print("  TUL TG SPAN-COMP ON: compressed branch = per-span mean-pooled "
              "live K/V (E-SAC; zero new params; span-granular causality)", flush=True)
    if model_cfg.tg_span_gate:
        print("  TUL TG SPAN-GATE ON: learned per-head gated softmax span pool "
              "(E-SAC-G; one zero-init [H,D] gate per attn layer — mean pool at "
              "init)", flush=True)
    return TulRuntime(model_cfg=model_cfg, data_cfg=data_cfg,
                      activate_at=activate_at, manifest=manifest)
