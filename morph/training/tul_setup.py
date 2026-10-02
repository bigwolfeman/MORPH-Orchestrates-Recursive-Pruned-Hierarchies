"""Resolve the Hydra ``tul:`` block into the objects the training loop needs.

Spec: ``docs/tul-spec.md`` §8 (config keys), §5 (schedule), §4 (data).

Kept out of ``train.py`` so the tokenizer work (resolving the boundary id set and the
slot token) is one testable unit, and so ``train.py``'s TUL seam is three lines. The
whole block resolves to ONE object; ``None`` means plain MORPH and nothing downstream
changes — no TUL parameters are constructed and the forward never sees a layout.
"""

from __future__ import annotations

import math
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
    "activate_at", "bcast", "bcast_layers", "boundary_chars", "boundary_substrings", "carry",
    "center_bag_mean", "center_exit", "coda_logit_l2", "coda_sees_slots",
    "coda_span_heads", "coda_span_source",
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
    "horizon_free_first", "horizon_tokens", "horizon_weight",
    "loop_carry", "loop_reach", "loop_reads_tokens", "max_slots", "min_span",
    "mux_activate_at", "mux_beta",
    "mux_detach_head", "mux_every_pass", "mux_readout", "mux_rho", "mux_stage_all",
    "mux_stage_own_iters",
    "mux_target",
    "mux_tau",
    "oracle_z", "oracle_z_lr", "oracle_z_max_tokens", "oracle_z_steps",
    "oracle_z_weight",
    "per_slot_embed",
    "per_slot_embed_std", "pass_lora_rank", "pass_lora_targets",
    "pass_readout",
    "pass_residual_lambda", "plast_weight", "prefix_k", "prefix_source", "progressive_p",
    "reinject_seed_every_pass", "recur_gate", "recur_gate_bias",
    "row_contrast_lambda", "row_contrast_tau",
    "fan_k", "fan_mix", "fan_repel_lambda", "fan_repel_passes",
    "fan_repel_mode", "fan_epi_features", "fan_epi_ridge", "fan_epi_eta",
    "fan_select_eps", "fan_select_gate_lambda", "fan_all_wta_lambda",
    "fan_all_wta_winner", "fan_all_wta_grad_rollouts", "fan_all_wta_latent_temp",
    "fan_all_wta_grader",
    "fan_opf", "fan_opf_lambda_pred", "fan_opf_lambda_fac", "fan_opf_lambda_enc",
    "fan_opf_gamma_fac", "fan_opf_gamma_enc", "fan_opf_pred_hidden", "fan_target_ema",
    "fan_route", "fan_route_rank", "fan_route_bias_u", "fan_rlat_lambda",
    "fan_loop_select", "fan_lsel_lambda", "fan_lsel_eps", "fan_lsel_enc_lambda",
    "fan_lsel_enc_gamma", "fan_lsel_router_lambda", "fan_lsel_router_rank",
    "fan_lsel_hidden", "fan_lsel_train_follow", "fan_lsel_read",
    "fan_lsel_head_input",
    "fan_select_write", "fan_select_write_anneal",
    "fan_trigger_every_pass", "fan_seed_noise", "fan_lineage", "fan_history_streams",
    "recur_gate_noise", "recur_gate_tau", "set_lambda", "sigreg_activate_at", "sigreg_lambda",
    "sigreg_slices", "slot_cells", "slot_cell_init", "slot_chain", "slot_chain_detach",
    "vq_beta", "vq_codebook", "vq_codes", "vq_dim", "vq_groups", "vq_reset_after",
    "vq_weight",
    "code", "code_noise", "code_noise_renorm", "code_norm", "code_fm_weight", "code_source_std",
    "code_t_embed_scale", "code_t_logit_mean", "code_t_logit_std",
    "code_phase2_at", "code_phase3_at", "code_rollout_p",
    "code_rollout_steps", "code_infer_steps", "code_seed_detach", "code_marginal_k",
    "code_cfg_drop", "code_cfg_scale", "code_target_lambda", "code_rank_abort",
    "code_xm_k", "code_xm_select", "code_xm_mode", "code_tape_rollout_p", "code_sigreg_lambda",
    "code_discrete", "code_vq_codebook", "code_vq_groups", "code_vq_dim", "code_vq_beta",
    "code_vq_weight", "code_sub_p", "code_mask_schedule",
    "code_target_source", "code_sonar_cache",
    "code_target", "code_target_weight", "code_target_detach", "code_target_skip_coda",
    "code_target_loss", "code_target_tau", "code_target_ref",
    "code_target_ema", "code_enc_var_lambda", "code_enc_var_gamma",
    "code_grade", "code_grade_k", "code_grade_tokens", "code_grade_rows",
    "code_grade_every", "code_grade_loss", "code_grade_grader", "code_grade_weight",
    "code_grade_temp", "code_grade_tau", "code_grade_min_distinct2",
    "loop_denoise", "loop_denoise_grid", "loop_denoise_weight",
    # LXTUL-G (tul.gram, 2026-09-23): the stochastic slot loop, morph/model/tul_gram.py
    "gram", "gram_beta", "gram_kl_balance", "gram_free_bits", "gram_mean",
    "gram_sigma_init", "gram_hidden",
    # LXTUL-GK (tul.gram_objective, 2026-09-23): the multi-sample bound over K prior rollouts
    "gram_objective", "gram_iw_k",
    "slot_depth_fixed", "slot_max_depth", "slot_mean_depth", "slot_seed", "slot_token",
    "spandec", "spandec_heads", "spandec_horizon", "spandec_layers", "spandec_max_tokens",
    "spandec_pass_horizon_max", "spandec_pass_tokens", "spandec_pass_weight",
    "spandec_per_pass", "spandec_reads_cells", "spandec_target_offset",
    "spandec_weight",
    # LXTUL-E (tul.spandec_parallel, 2026-09-23): the committed product reader
    "spandec_parallel", "spandec_parallel_k", "spandec_parallel_weight",
    "spandec_parallel_code_init",
    # LXTUL-E Stage 2 (2026-09-24): the head's target cap and the probe mode
    "spandec_parallel_span_cap", "spandec_parallel_detach",
    # LXTUL-E Stage 1 (tul.code_enum_k, 2026-09-24): the enumerated loop code
    "code_enum_k", "code_enum_ratio",
    # LX hard credit (2026-09-26): MCL over the K rollouts at train
    "code_enum_credit", "code_enum_hard_eps",
    # map-cause I-2 (2026-09-24): the slot loop's source enters at pass 0 only
    "slot_source_once",
    # span-level NextLat (arXiv 2511.05963, 2026-09-25)
    "nextlat_weight", "nextlat_beta",
    # arm B, the code policy (2026-09-29): morph/model/tul_code_policy.py
    "code_policy_k", "code_policy_ratio", "code_policy_lambda", "code_policy_entropy",
    "code_policy_value_lambda", "code_policy_eval",
    # LX efficient-exploration knobs (2026-09-29): morph/model/tul_explore.py
    "coda_fuse_layer", "hyp_score_head",
    "latent_set_loss", "latent_set_weight", "enum_decode_k",
    "hyp_merge", "hyp_merge_weight",
    # expanded hyper-connections in the slot loop (xHC, plan C, 2026-09-26)
    "xhc_streams", "xhc_active", "xhc_fixed", "xhc_temporal_kernels",
    "reread", "reread_heads", "reread_scope", "span_cap", "stp_lambda",
    "tg_coda_prefix_reach", "tg_geometry",
    # the looser strict coda (2026-09-26): previous spans' tokens in the coda relation
    "tg_coda_token_reach",
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

    # ── LXTUL: `tul.fan_k` ALIASES `tul.slot_cells` (morph/model/tul_fan.py) ──────
    # The fan's K streams ARE the Thought Register's cells: the same per-stream learned
    # trigger, the same within-slot loop relation, the same cell-level layout in
    # `_tul_core`. Resolving the alias HERE, once, is what keeps that machinery
    # un-duplicated — everything downstream (the model build, the cost accounting, the
    # probes, the refusal list) reads `slot_cells` and needs no knowledge of the fan.
    # Setting BOTH raises: one of them would otherwise win silently, and which one is
    # exactly the kind of thing a reader cannot check from a config.
    fan_k = int(tc.get("fan_k", 0))
    slot_cells = int(tc.get("slot_cells", 1))
    if fan_k > 0:
        if slot_cells != 1:
            raise ValueError(
                f"tul.fan_k={fan_k} with tul.slot_cells={slot_cells}: fan_k IS the cell "
                f"count (the fan's K streams are the Thought Register's cells). Set one "
                f"of them, not both.")
        slot_cells = fan_k

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
        bcast_layers=str(tc.get("bcast_layers", "entry")),
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
        spandec_target_offset=int(tc.get("spandec_target_offset", 1)),
        spandec_reads_cells=bool(tc.get("spandec_reads_cells", False)),
        spandec_per_pass=bool(tc.get("spandec_per_pass", False)),
        spandec_pass_horizon_max=int(tc.get("spandec_pass_horizon_max", 6)),
        spandec_pass_weight=float(tc.get("spandec_pass_weight", 1.0)),
        spandec_pass_tokens=int(tc.get("spandec_pass_tokens", 8)),
        spandec_parallel=bool(tc.get("spandec_parallel", False)),
        spandec_parallel_k=int(tc.get("spandec_parallel_k", 1)),
        spandec_parallel_weight=float(tc.get("spandec_parallel_weight", 1.0)),
        spandec_parallel_code_init=float(tc.get("spandec_parallel_code_init", 0.1)),
        spandec_parallel_span_cap=int(tc.get("spandec_parallel_span_cap", 0)),
        spandec_parallel_detach=bool(tc.get("spandec_parallel_detach", False)),
        code_enum_k=int(tc.get("code_enum_k", 1)),
        code_enum_ratio=float(tc.get("code_enum_ratio", 0.1)),
        code_enum_credit=str(tc.get("code_enum_credit", "soft")),
        code_enum_hard_eps=float(tc.get("code_enum_hard_eps", 0.05)),
        slot_source_once=bool(tc.get("slot_source_once", False)),
        nextlat_weight=float(tc.get("nextlat_weight", 0.0)),
        nextlat_beta=float(tc.get("nextlat_beta", 1.0)),
        code_policy_k=int(tc.get("code_policy_k", 0)),
        code_policy_ratio=float(tc.get("code_policy_ratio", 0.1)),
        code_policy_lambda=float(tc.get("code_policy_lambda", 1.0)),
        code_policy_entropy=float(tc.get("code_policy_entropy", 0.01)),
        code_policy_value_lambda=float(tc.get("code_policy_value_lambda", 1.0)),
        code_policy_eval=str(tc.get("code_policy_eval", "argmax")),
        coda_fuse_layer=int(tc.get("coda_fuse_layer", 0)),
        hyp_score_head=bool(tc.get("hyp_score_head", False)),
        latent_set_loss=str(tc.get("latent_set_loss", "none")),
        latent_set_weight=float(tc.get("latent_set_weight", 0.0)),
        enum_decode_k=int(tc.get("enum_decode_k", 0)),
        hyp_merge=str(tc.get("hyp_merge", "none")),
        hyp_merge_weight=float(tc.get("hyp_merge_weight", 0.0)),
        xhc_streams=int(tc.get("xhc_streams", 0)),
        xhc_active=int(tc.get("xhc_active", 4)),
        xhc_fixed=int(tc.get("xhc_fixed", 2)),
        xhc_temporal_kernels=tuple(int(k) for k in tc.get("xhc_temporal_kernels", ())),
        horizon_weight=float(tc.get("horizon_weight", 0.0)),
        horizon_free_first=bool(tc.get("horizon_free_first", True)),
        horizon_tokens=int(tc.get("horizon_tokens", 0)),
        pass_readout=str(tc.get("pass_readout", "last")),
        coda_span_heads=int(tc.get("coda_span_heads", 0)),
        coda_span_weight=float(tc.get("coda_span_weight", 1.0)),
        coda_span_source=str(tc.get("coda_span_source", "cell")),
        critic_weight=float(tc.get("critic_weight", 1.0)),
        critic_every=int(tc.get("critic_every", 1)),
        critic_eps=float(tc.get("critic_eps", 0.1)),
        critic_replay_groups=int(tc.get("critic_replay_groups", 1)),
        prefix_source=str(tc.get("prefix_source", "exit")),
        slot_cells=slot_cells,
        slot_cell_init=str(tc.get("slot_cell_init", "distinct")),
        fan_k=fan_k,
        fan_repel_lambda=float(tc.get("fan_repel_lambda", 0.0)),
        fan_repel_passes=int(tc.get("fan_repel_passes", 2)),
        fan_mix=str(tc.get("fan_mix", "mean")),
        fan_repel_mode=str(tc.get("fan_repel_mode", "cos")),
        fan_epi_features=int(tc.get("fan_epi_features", 64)),
        fan_epi_ridge=float(tc.get("fan_epi_ridge", 3.0)),
        fan_epi_eta=float(tc.get("fan_epi_eta", 30.0)),
        fan_select_eps=float(tc.get("fan_select_eps", 0.05)),
        fan_select_gate_lambda=float(tc.get("fan_select_gate_lambda", 1.0)),
        fan_select_write=str(tc.get("fan_select_write", "oracle")),
        fan_select_write_anneal=int(tc.get("fan_select_write_anneal", 1500)),
        fan_all_wta_lambda=float(tc.get("fan_all_wta_lambda", 1.0)),
        fan_all_wta_winner=str(tc.get("fan_all_wta_winner", "per_rollout")),
        fan_all_wta_grad_rollouts=str(tc.get("fan_all_wta_grad_rollouts", "all")),
        fan_all_wta_latent_temp=float(tc.get("fan_all_wta_latent_temp", 0.1)),
        fan_all_wta_grader=str(tc.get("fan_all_wta_grader", "coda")),
        fan_opf=bool(tc.get("fan_opf", False)),
        fan_opf_lambda_pred=float(tc.get("fan_opf_lambda_pred", 1.0)),
        fan_opf_lambda_fac=float(tc.get("fan_opf_lambda_fac", 0.05)),
        fan_opf_lambda_enc=float(tc.get("fan_opf_lambda_enc", 0.02)),
        fan_opf_gamma_fac=float(tc.get("fan_opf_gamma_fac", 0.1)),
        fan_opf_gamma_enc=float(tc.get("fan_opf_gamma_enc", 0.1)),
        fan_opf_pred_hidden=int(tc.get("fan_opf_pred_hidden", 0)),
        fan_target_ema=float(tc.get("fan_target_ema", 0.996)),
        fan_route=str(tc.get("fan_route", "none")),
        fan_route_rank=int(tc.get("fan_route_rank", 64)),
        fan_route_bias_u=float(tc.get("fan_route_bias_u", 1e-3)),
        fan_rlat_lambda=float(tc.get("fan_rlat_lambda", 1.0)),
        fan_loop_select=str(tc.get("fan_loop_select", "off")),
        fan_lsel_lambda=float(tc.get("fan_lsel_lambda", 10.0)),
        fan_lsel_eps=float(tc.get("fan_lsel_eps", 0.05)),
        fan_lsel_enc_lambda=float(tc.get("fan_lsel_enc_lambda", 0.2)),
        fan_lsel_enc_gamma=float(tc.get("fan_lsel_enc_gamma", 0.1)),
        fan_lsel_router_lambda=float(tc.get("fan_lsel_router_lambda", 1.0)),
        fan_lsel_router_rank=int(tc.get("fan_lsel_router_rank", 64)),
        fan_lsel_hidden=int(tc.get("fan_lsel_hidden", 0)),
        fan_lsel_train_follow=str(tc.get("fan_lsel_train_follow", "teacher")),
        fan_lsel_read=str(tc.get("fan_lsel_read", "winner")),
        fan_lsel_head_input=str(tc.get("fan_lsel_head_input", "live")),
        fan_trigger_every_pass=bool(tc.get("fan_trigger_every_pass", False)),
        fan_seed_noise=float(tc.get("fan_seed_noise", 0.0)),
        fan_lineage=str(tc.get("fan_lineage", "off")),
        fan_history_streams=int(tc.get("fan_history_streams", 0)),
        vq_codes=int(tc.get("vq_codes", 0)),
        vq_codebook=int(tc.get("vq_codebook", 512)),
        vq_dim=int(tc.get("vq_dim", 0)),
        vq_groups=int(tc.get("vq_groups", 1)),
        vq_beta=float(tc.get("vq_beta", 0.25)),
        vq_weight=float(tc.get("vq_weight", 1.0)),
        vq_reset_after=int(tc.get("vq_reset_after", 0)),
        center_exit=bool(tc.get("center_exit", False)),
        row_contrast_lambda=float(tc.get("row_contrast_lambda", 0.0)),
        row_contrast_tau=float(tc.get("row_contrast_tau", 0.1)),
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
        tg_coda_token_reach=int(tc.get("tg_coda_token_reach", 0)),
        loop_reach=int(tc.get("loop_reach", 0)),
        loop_carry=str(tc.get("loop_carry", "none")),
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
        code=bool(tc.get("code", False)),
        code_noise=float(tc.get("code_noise", 0.5)),
        code_noise_renorm=bool(tc.get("code_noise_renorm", False)),
        code_norm=str(tc.get("code_norm", "rms")),
        code_fm_weight=float(tc.get("code_fm_weight", 1.0)),
        code_source_std=float(tc.get("code_source_std", 1.0)),
        code_t_embed_scale=float(tc.get("code_t_embed_scale", 1.0)),
        # `null` stays None (the uniform draw); anything else is a float.
        code_t_logit_mean=(None if tc.get("code_t_logit_mean", None) is None
                           else float(tc.get("code_t_logit_mean"))),
        code_t_logit_std=float(tc.get("code_t_logit_std", 1.0)),
        code_phase2_at=float(tc.get("code_phase2_at", 0.10)),
        code_phase3_at=float(tc.get("code_phase3_at", 0.50)),
        code_rollout_p=float(tc.get("code_rollout_p", 0.5)),
        code_rollout_steps=int(tc.get("code_rollout_steps", 8)),
        code_infer_steps=int(tc.get("code_infer_steps", 8)),
        code_seed_detach=bool(tc.get("code_seed_detach", False)),
        code_marginal_k=int(tc.get("code_marginal_k", 8)),
        code_cfg_drop=float(tc.get("code_cfg_drop", 0.0)),
        code_cfg_scale=float(tc.get("code_cfg_scale", 1.0)),
        code_target_lambda=float(tc.get("code_target_lambda", 0.0)),
        code_rank_abort=float(tc.get("code_rank_abort", 0.0)),
        code_xm_k=int(tc.get("code_xm_k", 1)),
        code_xm_select=str(tc.get("code_xm_select", "l2")),
        code_xm_mode=str(tc.get("code_xm_mode", "sample")),
        code_tape_rollout_p=float(tc.get("code_tape_rollout_p", 0.0)),
        code_sigreg_lambda=float(tc.get("code_sigreg_lambda", 0.0)),
        code_discrete=bool(tc.get("code_discrete", False)),
        code_vq_codebook=int(tc.get("code_vq_codebook", 512)),
        code_vq_groups=int(tc.get("code_vq_groups", 4)),
        code_vq_dim=int(tc.get("code_vq_dim", 0)),
        code_vq_beta=float(tc.get("code_vq_beta", 0.25)),
        code_vq_weight=float(tc.get("code_vq_weight", 1.0)),
        code_sub_p=float(tc.get("code_sub_p", 0.3)),
        code_mask_schedule=str(tc.get("code_mask_schedule", "linear")),
        code_target_source=str(tc.get("code_target_source", "e")),
        code_sonar_cache=(None if tc.get("code_sonar_cache", None) is None
                          else str(tc.get("code_sonar_cache"))),
        code_target=bool(tc.get("code_target", False)),
        code_target_weight=float(tc.get("code_target_weight", 1.0)),
        code_target_detach=bool(tc.get("code_target_detach", True)),
        code_target_skip_coda=bool(tc.get("code_target_skip_coda", False)),
        code_target_loss=str(tc.get("code_target_loss", "l2")),
        code_target_tau=float(tc.get("code_target_tau", 0.1)),
        code_target_ref=bool(tc.get("code_target_ref", False)),
        code_target_ema=float(tc.get("code_target_ema", 0.0)),
        code_enc_var_lambda=float(tc.get("code_enc_var_lambda", 0.0)),
        code_enc_var_gamma=float(tc.get("code_enc_var_gamma", 1.0)),
        loop_denoise=bool(tc.get("loop_denoise", False)),
        loop_denoise_grid=str(tc.get("loop_denoise_grid", "linear")),
        loop_denoise_weight=float(tc.get("loop_denoise_weight", 1.0)),
        gram=bool(tc.get("gram", False)),
        gram_beta=float(tc.get("gram_beta", 0.1)),
        gram_kl_balance=float(tc.get("gram_kl_balance", 0.8)),
        gram_free_bits=float(tc.get("gram_free_bits", 0.0)),
        gram_mean=bool(tc.get("gram_mean", True)),
        gram_sigma_init=float(tc.get("gram_sigma_init", 0.1)),
        gram_hidden=int(tc.get("gram_hidden", 256)),
        gram_objective=str(tc.get("gram_objective", "elbo")),
        gram_iw_k=int(tc.get("gram_iw_k", 4)),
        coda_logit_l2=float(tc.get("coda_logit_l2", 0.0)),
        code_grade=bool(tc.get("code_grade", False)),
        code_grade_k=int(tc.get("code_grade_k", 4)),
        code_grade_tokens=int(tc.get("code_grade_tokens", 16)),
        code_grade_rows=int(tc.get("code_grade_rows", 1)),
        code_grade_every=int(tc.get("code_grade_every", 8)),
        code_grade_loss=str(tc.get("code_grade_loss", "pref")),
        code_grade_grader=str(tc.get("code_grade_grader", "coda_past")),
        code_grade_weight=float(tc.get("code_grade_weight", 1.0)),
        code_grade_temp=float(tc.get("code_grade_temp", 1.0)),
        code_grade_tau=float(tc.get("code_grade_tau", 0.1)),
        code_grade_min_distinct2=float(tc.get("code_grade_min_distinct2", 0.5)),
    )
    seq_len = int(cfg.data.seq_len)
    spec = data_cfg.spec_for(seq_len)
    # `tul.spandec_target_offset` against the DERIVED slot budget. Slot s is graded on span
    # s+k, so k at or past max_slots leaves every slot in every row unsupervised and the
    # term is identically zero. `span_slots` raises on the same condition at the layout,
    # which is the one place the rule lives; this is the same check moved to config time so
    # the run dies at startup and not five minutes into a queue slot.
    if model_cfg.spandec and model_cfg.spandec_target_offset >= spec.max_slots:
        raise ValueError(
            f"tul.spandec_target_offset={model_cfg.spandec_target_offset} is at or past "
            f"the derived slot budget max_slots={spec.max_slots} at seq_len={seq_len}: "
            f"slot s is graded on span s+{model_cfg.spandec_target_offset}, so no slot in "
            f"a row would have a target and the span-decoder term would be identically "
            f"zero.")
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
        "spandec_target_offset": model_cfg.spandec_target_offset,
        "spandec_reads_cells": model_cfg.spandec_reads_cells,
        "spandec_per_pass": model_cfg.spandec_per_pass,
        "spandec_pass_horizon_max": model_cfg.spandec_pass_horizon_max,
        "spandec_pass_weight": model_cfg.spandec_pass_weight,
        "spandec_pass_tokens": model_cfg.spandec_pass_tokens,
        "spandec_parallel": model_cfg.spandec_parallel,
        "spandec_parallel_k": model_cfg.spandec_parallel_k,
        "spandec_parallel_weight": model_cfg.spandec_parallel_weight,
        "spandec_parallel_code_init": model_cfg.spandec_parallel_code_init,
        "spandec_parallel_span_cap": model_cfg.spandec_parallel_span_cap,
        "spandec_parallel_detach": model_cfg.spandec_parallel_detach,
        "code_enum_k": model_cfg.code_enum_k,
        "code_enum_ratio": model_cfg.code_enum_ratio,
        "code_enum_credit": model_cfg.code_enum_credit,
        "code_enum_hard_eps": model_cfg.code_enum_hard_eps,
        "slot_source_once": model_cfg.slot_source_once,
        "nextlat_weight": model_cfg.nextlat_weight,
        "nextlat_beta": model_cfg.nextlat_beta,
        "code_policy_k": model_cfg.code_policy_k,
        "code_policy_ratio": model_cfg.code_policy_ratio,
        "code_policy_lambda": model_cfg.code_policy_lambda,
        "code_policy_entropy": model_cfg.code_policy_entropy,
        "code_policy_value_lambda": model_cfg.code_policy_value_lambda,
        "code_policy_eval": model_cfg.code_policy_eval,
        "coda_fuse_layer": model_cfg.coda_fuse_layer,
        "hyp_score_head": model_cfg.hyp_score_head,
        "latent_set_loss": model_cfg.latent_set_loss,
        "latent_set_weight": model_cfg.latent_set_weight,
        "enum_decode_k": model_cfg.enum_decode_k,
        "hyp_merge": model_cfg.hyp_merge,
        "hyp_merge_weight": model_cfg.hyp_merge_weight,
        "xhc_streams": model_cfg.xhc_streams,
        "xhc_active": model_cfg.xhc_active,
        "xhc_fixed": model_cfg.xhc_fixed,
        "xhc_temporal_kernels": list(model_cfg.xhc_temporal_kernels),
        "horizon_weight": model_cfg.horizon_weight,
        "horizon_free_first": model_cfg.horizon_free_first,
        "horizon_tokens": (model_cfg.horizon_tokens or model_cfg.bound_span_cap),
        "pass_readout": model_cfg.pass_readout,
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
        "slot_cells": model_cfg.slot_cells,
        "fan_k": model_cfg.fan_k,
        "fan_mix": model_cfg.fan_mix,
        "fan_repel_lambda": model_cfg.fan_repel_lambda,
        "fan_repel_passes": model_cfg.fan_repel_passes,
        "fan_repel_mode": model_cfg.fan_repel_mode,
        "fan_select_eps": model_cfg.fan_select_eps,
        "fan_select_gate_lambda": model_cfg.fan_select_gate_lambda,
        "fan_select_write": model_cfg.fan_select_write,
        "fan_select_write_anneal": model_cfg.fan_select_write_anneal,
        "fan_all_wta_lambda": model_cfg.fan_all_wta_lambda,
        "fan_all_wta_winner": model_cfg.fan_all_wta_winner,
        "fan_all_wta_grad_rollouts": model_cfg.fan_all_wta_grad_rollouts,
        "fan_all_wta_latent_temp": model_cfg.fan_all_wta_latent_temp,
        "fan_all_wta_grader": model_cfg.fan_all_wta_grader,
        "fan_opf": model_cfg.fan_opf,
        "fan_opf_lambda_pred": model_cfg.fan_opf_lambda_pred,
        "fan_opf_lambda_fac": model_cfg.fan_opf_lambda_fac,
        "fan_opf_lambda_enc": model_cfg.fan_opf_lambda_enc,
        "fan_opf_gamma_fac": model_cfg.fan_opf_gamma_fac,
        "fan_opf_gamma_enc": model_cfg.fan_opf_gamma_enc,
        "fan_opf_pred_hidden": model_cfg.fan_opf_pred_hidden,
        "fan_target_ema": model_cfg.fan_target_ema,
        "fan_route": model_cfg.fan_route,
        "fan_route_rank": model_cfg.fan_route_rank,
        "fan_route_bias_u": model_cfg.fan_route_bias_u,
        "fan_rlat_lambda": model_cfg.fan_rlat_lambda,
        "fan_loop_select": model_cfg.fan_loop_select,
        "fan_lsel_lambda": model_cfg.fan_lsel_lambda,
        "fan_lsel_eps": model_cfg.fan_lsel_eps,
        "fan_lsel_enc_lambda": model_cfg.fan_lsel_enc_lambda,
        "fan_lsel_enc_gamma": model_cfg.fan_lsel_enc_gamma,
        "fan_lsel_router_lambda": model_cfg.fan_lsel_router_lambda,
        "fan_lsel_router_rank": model_cfg.fan_lsel_router_rank,
        "fan_lsel_hidden": model_cfg.fan_lsel_hidden,
        "fan_lsel_train_follow": model_cfg.fan_lsel_train_follow,
        "fan_lsel_read": model_cfg.fan_lsel_read,
        "fan_lsel_head_input": model_cfg.fan_lsel_head_input,
        "fan_trigger_every_pass": model_cfg.fan_trigger_every_pass,
        "fan_seed_noise": model_cfg.fan_seed_noise,
        "fan_lineage": model_cfg.fan_lineage,
        "fan_history_streams": model_cfg.fan_history_streams,
        "fan_epi_features": model_cfg.fan_epi_features,
        "fan_epi_ridge": model_cfg.fan_epi_ridge,
        "fan_epi_eta": model_cfg.fan_epi_eta,
        "center_exit": model_cfg.center_exit,
        "row_contrast_lambda": model_cfg.row_contrast_lambda,
        "row_contrast_tau": model_cfg.row_contrast_tau,
        "slot_cell_init": model_cfg.slot_cell_init,
        "vq_codes": model_cfg.vq_codes,
        "vq_codebook": model_cfg.vq_codebook,
        "vq_dim": model_cfg.vq_dim,
        "vq_groups": model_cfg.vq_groups,
        "vq_beta": model_cfg.vq_beta,
        "vq_weight": model_cfg.vq_weight,
        "vq_reset_after": model_cfg.vq_reset_after,
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
        "tg_coda_token_reach": model_cfg.tg_coda_token_reach,
        "loop_reach": model_cfg.loop_reach,
        "loop_carry": model_cfg.loop_carry,
        "oracle_z": model_cfg.oracle_z,
        "oracle_z_steps": model_cfg.oracle_z_steps,
        "oracle_z_lr": model_cfg.oracle_z_lr,
        "oracle_z_weight": model_cfg.oracle_z_weight,
        "oracle_z_max_tokens": model_cfg.oracle_z_max_tokens,
        "coda_token_input": model_cfg.coda_token_input,
        "bcast": model_cfg.bcast,
        "bcast_layers": model_cfg.bcast_layers,
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
        "code": model_cfg.code,
        "code_noise": model_cfg.code_noise,
        "code_noise_renorm": model_cfg.code_noise_renorm,
        "code_norm": model_cfg.code_norm,
        "code_fm_weight": model_cfg.code_fm_weight,
        "code_source_std": model_cfg.code_source_std,
        "code_t_embed_scale": model_cfg.code_t_embed_scale,
        "code_t_logit_mean": model_cfg.code_t_logit_mean,
        "code_t_logit_std": model_cfg.code_t_logit_std,
        "code_phase2_at": model_cfg.code_phase2_at,
        "code_phase3_at": model_cfg.code_phase3_at,
        "code_rollout_p": model_cfg.code_rollout_p,
        "code_rollout_steps": model_cfg.code_rollout_steps,
        "code_infer_steps": model_cfg.code_infer_steps,
        "code_seed_detach": model_cfg.code_seed_detach,
        "code_marginal_k": model_cfg.code_marginal_k,
        "code_cfg_drop": model_cfg.code_cfg_drop,
        "code_cfg_scale": model_cfg.code_cfg_scale,
        "code_target_lambda": model_cfg.code_target_lambda,
        "code_rank_abort": model_cfg.code_rank_abort,
        "code_xm_k": model_cfg.code_xm_k,
        "code_xm_select": model_cfg.code_xm_select,
        "code_xm_mode": model_cfg.code_xm_mode,
        "code_tape_rollout_p": model_cfg.code_tape_rollout_p,
        "code_sigreg_lambda": model_cfg.code_sigreg_lambda,
        "code_discrete": model_cfg.code_discrete,
        "code_vq_codebook": model_cfg.code_vq_codebook,
        "code_vq_groups": model_cfg.code_vq_groups,
        "code_vq_dim": model_cfg.code_vq_dim,
        "code_vq_beta": model_cfg.code_vq_beta,
        "code_vq_weight": model_cfg.code_vq_weight,
        "code_sub_p": model_cfg.code_sub_p,
        "code_mask_schedule": model_cfg.code_mask_schedule,
        "code_target_source": model_cfg.code_target_source,
        "code_sonar_cache": model_cfg.code_sonar_cache,
        "code_target": model_cfg.code_target,
        "code_target_weight": model_cfg.code_target_weight,
        "code_target_detach": model_cfg.code_target_detach,
        "code_target_skip_coda": model_cfg.code_target_skip_coda,
        "code_target_loss": model_cfg.code_target_loss,
        "code_target_tau": model_cfg.code_target_tau,
        "code_target_ref": model_cfg.code_target_ref,
        "code_target_ema": model_cfg.code_target_ema,
        "code_enc_var_lambda": model_cfg.code_enc_var_lambda,
        "code_enc_var_gamma": model_cfg.code_enc_var_gamma,
        "loop_denoise": model_cfg.loop_denoise,
        "loop_denoise_grid": model_cfg.loop_denoise_grid,
        "loop_denoise_weight": model_cfg.loop_denoise_weight,
        "gram": model_cfg.gram,
        "gram_beta": model_cfg.gram_beta,
        "gram_kl_balance": model_cfg.gram_kl_balance,
        "gram_free_bits": model_cfg.gram_free_bits,
        "gram_mean": model_cfg.gram_mean,
        "gram_sigma_init": model_cfg.gram_sigma_init,
        "gram_hidden": model_cfg.gram_hidden,
        "gram_objective": model_cfg.gram_objective,
        "gram_iw_k": model_cfg.gram_iw_k,
        "coda_logit_l2": model_cfg.coda_logit_l2,
        "code_grade": model_cfg.code_grade,
        "code_grade_k": model_cfg.code_grade_k,
        "code_grade_tokens": model_cfg.code_grade_tokens,
        "code_grade_rows": model_cfg.code_grade_rows,
        "code_grade_every": model_cfg.code_grade_every,
        "code_grade_loss": model_cfg.code_grade_loss,
        "code_grade_grader": model_cfg.code_grade_grader,
        "code_grade_weight": model_cfg.code_grade_weight,
        "code_grade_temp": model_cfg.code_grade_temp,
        "code_grade_tau": model_cfg.code_grade_tau,
        "code_grade_min_distinct2": model_cfg.code_grade_min_distinct2,
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
        _m = int(model_cfg.slot_cells)
        _reads = ("the loop's S*%d compact CELL axis under the register's own in-loop "
                  "relation (a cell reads its whole slot, later siblings INCLUDED, and "
                  "every earlier slot; delivered as `tg_relation`, which REPLACES the "
                  "branches' causal term instead of narrowing it — the same delivery "
                  "the loop's core stage uses)"
                  % _m if _m > 1 else
                  "the loop's S compact SLOT axis, plain causal")
        print(f"  TUL THINK-ONCE: cond_layers={model_cfg.cond_layers} "
              f"detach_z={model_cfg.detach_z} (arms R7/R8) — the stack runs ONCE over "
              f"{_reads}, straight out of the loop and BEFORE any mean, and every reader "
              f"of z (the span decoder, SIGReg, the coda's prefix write"
              f"{', with stop-gradient' if model_cfg.detach_z else ''}) reads ITS output",
              flush=True)
    if model_cfg.spandec:
        _k = model_cfg.spandec_target_offset
        _h = model_cfg.spandec_horizon
        _tgt = (f"span s+{_k}" if _h == 1 else f"spans s+{_k}..s+{_k + _h - 1}")
        print(f"  TUL SPAN DECODER ON: layers={model_cfg.spandec_layers} "
              f"heads={model_cfg.spandec_heads or int(cfg.model.n_heads)} "
              f"J={model_cfg.spandec_max_tokens or model_cfg.bound_span_cap}"
              f"x{_h} offset={_k} "
              f"weight={model_cfg.spandec_weight} — {_tgt} "
              f"{'is' if _h == 1 else 'are'} decoded from z with a "
              f"teacher-forced token path (morph/model/tul_spandec.py)", flush=True)
        if _k != 1:
            print(f"  TUL SPANDEC TARGET OFFSET {_k}: slot s is graded on span s+{_k} and "
                  f"ONLY that span — the NEXT span gets no direct z target, the last "
                  f"{_k - 1} slot(s) of every row are masked to -100, and `spandec_ce` is "
                  f"NOT comparable with an offset-1 arm's (it grades a harder span). "
                  f"This is not spandec_horizon, which WIDENS the target instead of "
                  f"moving it.", flush=True)
        if model_cfg.spandec_reads_cells:
            print(f"  TUL SPANDEC READS CELLS: the decoder cross-attends, per layer, to "
                  f"the slot's {model_cfg.slot_cells} register cells — the same states "
                  f"prefix_project writes into the coda — instead of grading only their "
                  f"mean. The z conditioning stays the MEAN and the cross output "
                  f"projection is zero-init, so step 0 is exactly the mean-only decoder.",
                  flush=True)
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
    if model_cfg.horizon_weight > 0.0:
        _hj = model_cfg.horizon_tokens or model_cfg.bound_span_cap
        _h0 = "2" if model_cfg.horizon_free_first else "1"
        print(f"  TUL HORIZON (LoopMTP) ON: pass t (t={_h0}..T) aligned by cosine loss "
              f"to the mean tied-embedding of span s+t ({_hj} tokens/span, weight "
              f"{model_cfg.horizon_weight}) — no decoder, arXiv 2608.03624 Eq 12-14. "
              f"pass_readout={model_cfg.pass_readout!r}.", flush=True)
    elif model_cfg.pass_readout == "gated":
        print(f"  TUL PASS-GATED READOUT ON (LoopMTP Eq 9-11): z is a content-conditional "
              f"softmax mixture of every one of the loop's {model_cfg.slot_depth_fixed} "
              f"realised passes, not only the last.", flush=True)
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
    if model_cfg.slot_cells > 1:
        print(f"  TUL THOUGHT REGISTER ON: slot_cells={model_cfg.slot_cells} "
              f"init={model_cfg.slot_cell_init!r} - each span gets M MUTABLE looped cells "
              "instead of one, seeded apart by M learned queries pooling the span's own "
              "prelude states (W_o zero-init: step 0 IS the ruler). They loop together at "
              "one per-slot depth; inside the loop a cell reads every cell of earlier "
              "slots AND every cell of its own slot, LATER siblings included (the "
              "relation travels as `tg_relation` and REPLACES the attention branches' "
              "causal term; through `tg_allow` it would only narrow and would execute "
              "as plain flattened causal - measured and fixed 2026-09-13). "
              + ("The M cells are LXTUL's K streams and every one of them is written 1:1 "
                 "into ITS prefix cell (fan_mix=all; see the LXTUL FAN line below). "
                 if (model_cfg.fan_k > 0 and model_cfg.fan_mix == "all") else
                 "The M cells are LXTUL's K streams: they are MIXED to one state and "
                 "written through the ordinary single-source prefix_project (see the "
                 "LXTUL FAN line below), NOT 1:1 into the prefix cells. "
                 if model_cfg.fan_k > 0 else
                 "Cell i is written 1:1 into prefix "
                 "cell i (prefix_k == slot_cells), ")
              + "the coda is unchanged, and the span "
              "decoder grades the MEAN of the M cells. Built against the measured rank "
              "collapse (slot_eff_rank 5.7-7.3 in 1024 dims, pairwise cos 0.72-0.77). "
              "Read `val/slot_cell_eff_rank` - the rank WITHIN a slot - beside "
              "`val/slot_eff_rank` "
              "(lab/experiments/planned/2026-09-13-arc-thought-register.md)",
              flush=True)
    if model_cfg.loop_carry == "persist":
        print(f"  LOOP CARRY ON: loop_carry='persist' loop_reach={model_cfg.loop_reach} "
              f"slot_cells={model_cfg.slot_cells} - a per-cell state that KEEPS what the "
              "cell read from its neighbours, written ONCE at the loop's EXIT and NEVER "
              "handed back into the loop. At pass t core layer 0's WINDOW branch is the "
              f"cell's read of cells k-{model_cfg.loop_reach}..k-1 (its XSA excludes the "
              "self token, so the read is purely cross-cell); the accumulator sums it "
              "(plain addition, no parameter) EVERY pass, exactly as 'sum' does, but the "
              "sum is NEVER handed to a pass as its `carry=` argument — the map every "
              "later reader sees (the gain hinge, the terminal fixed-point term, "
              "slot_state_renorm, every forced-depth sweep) is therefore the SAME map "
              "loop_carry='none' runs, pass for pass. AFTER the last pass the full "
              "accumulator is RMS-matched to the exit carrier and ADDED ONCE, masked to "
              "valid cells. WHY: 'sum' / 'gate' re-inject at the entry of every later "
              "pass, and that re-supply REPLACES the pass-1 read and dilutes the far hops "
              "instead of adding to them (carry RMS 0.5 -> 75 over a run, "
              "lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md) — 'persist' "
              "keeps the direct read and bounds the state by construction (the add is one "
              "carrier-RMS worth of the accumulated direction, however long the "
              "accumulator has been summing). Measured on the Thought Register "
              f"(tul.slot_cells > 1 / tul.fan_k > 0), where 'sum' / 'gate' stay refused. "
              "READ `carry/rms_t{t}` (the accumulator's own per-pass growth) and "
              "`carry/persist_ratio` (mean, over valid cells, of the exit add's RMS over "
              "the raw exit's RMS) "
              "(.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-"
              "composition.md, Step 2)",
              flush=True)
    elif model_cfg.loop_carry != "none":
        print(f"  LOOP CARRY ON: loop_carry={model_cfg.loop_carry!r} "
              f"loop_reach={model_cfg.loop_reach} - a per-cell state that KEEPS what the "
              f"cell read from its neighbours. At pass t core layer 0's WINDOW branch is "
              f"the cell's read of cells k-{model_cfg.loop_reach}..k-1 (its XSA excludes "
              f"the self token, so the read is purely cross-cell); the carry accumulates "
              f"it ("
              + ("plain sum" if model_cfg.loop_carry == "sum" else
                 "sigmoid(W_g [r ; c]) * r, W_g zero-init so the gate opens at 1/2")
              + ") and re-injects it at the entry of every LATER pass, RMS-matched to "
              "the carrier. WHY: the hop probe measured that carried content decays "
              "under the cell's own passes (planted g=2: 0.148 -> 0.035 nats from depth "
              "1 to 6 with arrivals cut) while own-span content, re-supplied every pass "
              "by the x0/bigram injection, is refined (0.181 -> 0.291). READ "
              "`carry/rms_t{t}` and, on gate, `carry/gate_mean_t{t}`; "
              "`carry/inject_ratio_t{t}` is 1.0 BY CONSTRUCTION and is the check that "
              "the RMS match is live, not a result. NOT a no-op at init: the RMS match "
              "cancels any constant in front of the carry, so pass 1 injects a full "
              "carrier-RMS term on both modes "
              "(lab/experiments/planned/2026-09-19-loop-carry-prev-reach1.md)",
              flush=True)
    if model_cfg.fan_k > 0:
        print(f"  LXTUL FAN ON: fan_k={model_cfg.fan_k} mix={model_cfg.fan_mix!r} "
              f"repel_mode={model_cfg.fan_repel_mode!r} repel_lambda={model_cfg.fan_repel_lambda} "
              f"repel_passes={model_cfg.fan_repel_passes} prefix_k={model_cfg.prefix_k} "
              + (f"SELECT eps={model_cfg.fan_select_eps} gate_lambda={model_cfg.fan_select_gate_lambda} "
                 f"(no mixture: K no-grad coda passes pick each slot's winner at train, the "
                 f"winner is written ALONE, the gate learns to predict it, the eval write is "
                 f"the gate's argmax stream; train WRITE={model_cfg.fan_select_write!r}"
                 + (f" over {model_cfg.fan_select_write_anneal} steps"
                    if model_cfg.fan_select_write == "anneal" else "")
                 + (": the coda trains on the gate's own pick, the same forward as eval; the "
                    f"table is the gate's label only" if model_cfg.fan_select_write != "oracle"
                    else ": the coda trains on the table's winner, a stream it never gets at "
                    "eval") + ") " if model_cfg.fan_mix == "select" else "")
              + (f"ALL eps={model_cfg.fan_select_eps} wta_lambda={model_cfg.fan_all_wta_lambda} "
                 f"(no mixture, no gate: every stream is written into ITS prefix cell "
                 f"through W_prefix[i], the register's 1:1 route, and the coda reads all "
                 f"K; "
                 + ("WTA grader='head': NO extra coda pass; the parallel span head reads "
                    "each cell ALONE and grades it by its NLL of the true next span (the "
                    "coda table's own tokens), and the relaxed WTA over the cells ((1-eps) "
                    "on the head's argmin, eps/(M-1) on each other) trains the head and "
                    "the cells; the coda reads all cells in its one ordinary pass. Read "
                    "fan/head_coda_agree (chance 1/K) and fan/head_pick_regret vs "
                    "fan/rand_pick_regret at val) "
                    if model_cfg.fan_all_wta_lambda > 0.0
                    and model_cfg.fan_all_wta_grader == "head" else
                    "WTA grader='coda': at train K no-grad passes with stream i alone in "
                    "its cell pick each slot's winner and one more pass with grad charges "
                    "the winner-alone span CE, the responsibility term) "
                    if model_cfg.fan_all_wta_lambda > 0.0 else
                    "wta_lambda 0: NO responsibility term and NO extra coda pass) ")
                 + (f"OPF (arm F): each cell k predicts factor k of the EMA-prelude code of "
                    f"the next span (orthonormal P, QR-retracted every step; lambda_pred="
                    f"{model_cfg.fan_opf_lambda_pred} lambda_fac={model_cfg.fan_opf_lambda_fac}"
                    f" lambda_enc={model_cfg.fan_opf_lambda_enc} gamma_fac="
                    f"{model_cfg.fan_opf_gamma_fac} gamma_enc={model_cfg.fan_opf_gamma_enc}, "
                    f"EMA m={model_cfg.fan_target_ema}); the gradient reaches the cells "
                    f"beside the coda's CE. Read fan/opf_r2_k*, fan/opf_target_rank "
                    f"(collapse), fan/opf_enc_std_min, fan/opf_orth_err "
                    if model_cfg.fan_opf else "")
                 + (f"ROUTER '{model_cfg.fan_route}' (rank {model_cfg.fan_route_rank}, balance "
                    f"bias u={model_cfg.fan_route_bias_u}): the coda reads ONLY the router's "
                    f"pick, "
                    + ("scaled by its gate p (the coda's CE trains the router) "
                       if model_cfg.fan_route == "reader" else
                       f"hard, trained by a latent teacher (argmin of g(cell) to the "
                       f"EMA-prelude target, weight {model_cfg.fan_rlat_lambda}, EMA m="
                       f"{model_cfg.fan_target_ema}) ")
                    + "- read fan/router_coda_agree (chance 1/K) and fan/router_pick_regret "
                      "vs fan/rand_pick_regret at val "
                    if model_cfg.fan_route != "none" else "")
                 + (f"LATENT-SELECTED LOOP '{model_cfg.fan_loop_select}': after every pass "
                    f"the M cells of a continuing slot are reset to the winner (train: "
                    f"follow the {model_cfg.fan_lsel_train_follow.upper()}; the teacher is the "
                    f"argmin ||g(cell) - z||^2 on the EMA-prelude target, EMA m="
                    f"{model_cfg.fan_target_ema}; eval and slots without a target: the "
                    f"router, rank {model_cfg.fan_lsel_router_rank}, CE onto the teacher "
                    f"at weight {model_cfg.fan_lsel_router_lambda}); ONE latent loss at "
                    f"the exit (relaxed WTA eps={model_cfg.fan_lsel_eps}, weight "
                    f"{model_cfg.fan_lsel_lambda}, "
                    + ("WITH grad into the loop" if model_cfg.fan_lsel_head_input == "live"
                       else "head reads DETACHED cells: ranks only, no pull on the loop")
                    + f"; online floor "
                    f"{model_cfg.fan_lsel_enc_lambda} @ {model_cfg.fan_lsel_enc_gamma}); "
                    f"the coda reads "
                    + ("the final winner alone" if model_cfg.fan_lsel_read == "winner"
                       else "ALL M final candidates (the write-all fan's write)")
                    + (" DETACHED (no token CE reaches the loop) "
                       if model_cfg.fan_loop_select == "detached" else
                       " (the coda's CE reaches the loop through the winner) ")
                    + "- read fan/lsel_r2, fan/lsel_teacher_router_agree (chance 1/K), "
                      "fan/lsel_switch_rate, fan/lsel_cell_spread, and at val "
                      "fan/lsel_teacher_pick_gap "
                    if model_cfg.fan_loop_select != "off" else "")
                 if model_cfg.fan_mix == "all" else "")
              + ("TRIGGER EVERY PASS (the per-stream trigger W_o(pooled)+P_cell, the "
                 "SAME tensor the seed adds once, is added to the cell carrier at the "
                 "start of passes 2..T, so stream identity is re-supplied to the map "
                 "instead of living only in the initial condition; zero at step 0, so "
                 "this arm starts bit-identical to its fan partner. NOTE the seed "
                 "ALREADY reaches every pass through DiagonalInjection - at "
                 f"injection_channels={getattr(cfg.model, 'injection_channels', 'ctx')!r} "
                 f"that route is "
                 + ("the CONTEXT channel slice only, decayed, so this add is a NEW "
                    "full-width route"
                    if str(getattr(cfg.model, "injection_channels", "ctx")) == "ctx" else
                    "every channel, so read this add as a GAIN on the existing route")
                 + ") " if model_cfg.fan_trigger_every_pass else "")
              + (f"SEED NOISE std={model_cfg.fan_seed_noise} (the K streams are K "
                 "SAMPLES: an independent Gaussian draw per stream, per slot and per "
                 "forward is added to the ENTRY STATE core_init(e) - NOT to `e`, which "
                 "every pass re-injects - at train AND at eval, pads zero. The seed "
                 "lives in the input_norm'd field, so std 1.0 IS unit per-channel RMS "
                 "there and two streams differ by RMS sqrt(2). "
                 + ("The diversity term is OFF (fan_repel_lambda 0): the noise replaces "
                    "it and fan_vol_t/fan_epi_t stay INSTRUMENTS. "
                    if model_cfg.fan_repel_lambda == 0.0 else
                    f"NOTE the diversity term is STILL CHARGED "
                    f"(fan_repel_lambda={model_cfg.fan_repel_lambda}): this arm is noise "
                    f"AND a term, which is two factors. ")
                 + ") " if model_cfg.fan_seed_noise > 0.0 else "")
              + ("LINEAGE RELATION (tul.fan_lineage='relation': across slots a cell reads "
                 "only its OWN stream index, so the K streams are K channels along the "
                 "slot axis; within a slot the register's all-to-all relation is "
                 "unchanged. The REWEIGHTING half of rung P4 is NOT built - the "
                 "per-stream span CE that would set the weights comes from the coda AFTER "
                 "the loop, so gating pass 1 with it is circular here) "
                 if model_cfg.fan_lineage != "off" else "")
              + (f"HISTORY STREAMS h={model_cfg.fan_history_streams} of "
                 f"{model_cfg.fan_k} (loop_reach={model_cfg.loop_reach}: h streams relay "
                 "across slots, the remaining K-h PLAN streams are slot-local and take "
                 "the diversity term ALONE, so the relay and the repulsion stop sharing "
                 "cells) " if model_cfg.fan_history_streams > 0 else "")
              +
              "- K latent STREAMS per span through the ONE shared core. The streams ARE "
              "the Thought Register's cells (fan_k aliases slot_cells, so the message "
              "above is this arm's loop); what is NEW is the three things the register "
              + (f"did not have. (1) a DIVERSITY term ({model_cfg.fan_repel_mode!r}) "
                 f"charged after passes 1..{model_cfg.fan_repel_passes} only, because PLR "
                 f"Thm 4.4 makes the collapse exponential in depth (pass 1 fights L^2, "
                 f"pass 6 fights L^12). "
                 if model_cfg.fan_repel_lambda > 0.0 else
                 f"did not have. (1) a DIVERSITY term ({model_cfg.fan_repel_mode!r}) - NOT "
                 "CHARGED on this arm (fan_repel_lambda 0): `fan/vol_t`, `fan/epi_t` and "
                 "`fan/stream_cos_t` are still computed every pass and reported, as "
                 "instruments. ")
              + "(2) the "
              + (f"exit WRITE: every stream in its own prefix cell, the register's 1:1 "
                 f"route, so the coda's width is prefix_k={model_cfg.prefix_k} and the width "
                 f"partner is the ruler at the same prefix_k. " if model_cfg.fan_mix == "all" else
                 f"exit MIXTURE: the K streams become ONE state at the register's mean seam "
                 f"and go through the ordinary SINGLE-SOURCE prefix_project, NOT the "
                 f"register's 1:1 cell write - so this arm's coda width is the strict "
                 f"ruler's and the 2026-09-13 width confound is closed by construction. ")
              +
              f"(3) the ORACLE: at every val the coda is re-run once per stream and "
              f"`fan/oracle_ce` is the per-span minimum. READ `fan/oracle_ce` AGAINST "
              f"`fan/single_ce` FIRST - if the gap is inside a width control's own CE "
              f"gain, the streams are copies and the width branch closes, whatever "
              f"`fan/mixed_ce` says. Then `fan/stream_cos_t{{t}}` (1.0 = collapsed; the "
              f"register read 0.94) and `fan/stream_rank_t{{t}}` (the register read 1.24 "
              f"of 4) "
              f"(lab/experiments/planned/2026-09-19-lxtul-fan4.md)",
              flush=True)
    if model_cfg.code:
        # LCTUL (tul.code, docs/tul-code-spec.md). No banner existed before 2026-09-21;
        # this one prints the two things an arm is read by — WHICH thinker was built and
        # WHICH noise levels its loss was trained on.
        _sched = ("t ~ U(0, 1) (lambda = 2*logit(t) uniform in t: HALF of training sits "
                  "above SNR 1, where the noisy input carries the answer and the context "
                  "is worth nothing to the velocity - LCM Table 5's peaked schedule, "
                  "'akin to a Base-LCM')"
                  if model_cfg.code_t_logit_mean is None else
                  f"t ~ logit-normal(mu={model_cfg.code_t_logit_mean}, "
                  f"sigma={model_cfg.code_t_logit_std}): median t = "
                  f"{1.0 / (1.0 + math.exp(-float(model_cfg.code_t_logit_mean))):.4f} "
                  f"(lambda = {2.0 * float(model_cfg.code_t_logit_mean):.2f}), the flow "
                  "LOSS is reweighted toward HIGH NOISE (LCM's wide schedule: CA 80.3 % "
                  "against the peaked schedule's 70.6 %). The SAMPLER's Euler grid stays "
                  "UNIFORM - this knob moves training only")
        # Rung P3 (`tul.code_target_source`): WHAT the thinker aims at. The cache is
        # opened here, at config time, so a missing or malformed one kills the run at
        # startup instead of five minutes into a queue slot — and so the banner's row
        # count is the file's, not the config's claim about it.
        _tgt = "TARGET = E's own code of the next span (verbatim reconstruction)"
        if model_cfg.code_target_source == "sonar":
            from morph.model.sonar_cache import SONAR_DIM, SonarSpanCache
            _sc = SonarSpanCache(model_cfg.code_sonar_cache)
            _mc = int(model_cfg.prefix_k) * int(cfg.model.d_model)
            _tgt = (f"TARGET = SONAR cache {_sc.path}, {_sc.n_rows} rows, frozen "
                    f"{SONAR_DIM}->{_mc} expansion "
                    f"(M={model_cfg.prefix_k} cells x C={int(cfg.model.d_model)}); E is "
                    f"built and RUNS for its validity mask but receives NO GRADIENT")
        print(f"  LCTUL ON: prefix_k={model_cfg.prefix_k} "
              + (f"DISCRETE code ({model_cfg.code_vq_groups} symbols/cell of "
                 f"{model_cfg.code_vq_codebook}), masked denoiser, "
                 f"{model_cfg.code_mask_schedule} unmasking "
                 if model_cfg.code_discrete else
                 f"CONTINUOUS code, flow thinker (source_std="
                 f"{model_cfg.code_source_std}), {model_cfg.code_infer_steps} Euler steps "
                 "at eval ")
              + "- the slot holds the CODE of the span it precedes: E makes it from that "
              "span at train, the core body samples it at eval. Train schedule: "
              + ("the mask rate t ~ U(0, 1) with the ELBO's 1/t weight (NOT a tunable "
                 "noise schedule)" if model_cfg.code_discrete else _sched)
              + ". " + _tgt
              + ". READ `val/code_ca` AGAINST `val/code_ca_chance` FIRST (LCM's "
              "contrastive accuracy: does the SAMPLED code retrieve its own span's true "
              "code out of the batch, neighbours excluded) - a code that wins "
              "`code_fm_rel` and reads CA at chance is a regression to the conditional "
              "mean, whatever the CE says. Then "
              + ("" if model_cfg.code_discrete else
                 "`train/code_t_mean` (the schedule, as drawn) and ")
              + "the four `code_fm_band{b}_rel` ratios "
              "(.agents/notes/proposed/architecture/"
              "2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md)",
              flush=True)
    if model_cfg.code_target and (model_cfg.code_target_ema > 0.0
                                  or model_cfg.code_enc_var_lambda > 0.0):
        # LCTUL-J Stage 1 (tul.code_target_ema / tul.code_enc_var_lambda, 2026-09-22).
        _ema_bit = (f"EMA target, momentum m={model_cfg.code_target_ema} (the twin lerps "
                   f"toward the live model after every optimizer step, weight "
                   f"1-m={1.0 - model_cfg.code_target_ema:.4f})"
                   if model_cfg.code_target_ema > 0.0 else "FROZEN target (m=0)")
        _floor_bit = (f"variance floor L_enc: lambda={model_cfg.code_enc_var_lambda}, "
                     f"gamma={model_cfg.code_enc_var_gamma} (own weight, NOT scaled by "
                     f"code_target_weight)"
                     if model_cfg.code_enc_var_lambda > 0.0 else "no variance floor")
        print(f"  LCTUL-J ON: {_ema_bit}; {_floor_bit}. Read `tul/code_tgt_std` (the "
              f"target-side collapse instrument, always on with code_target) beside "
              f"`tul/code_enc_std` and `tul/code_enc_active`.", flush=True)
    if model_cfg.loop_denoise:
        # LXTUL-P change 1 (tul.loop_denoise). The banner prints the two things that make
        # this arm unreadable if they are forgotten: train and eval run DIFFERENT
        # functions, and every per-pass loop instrument now reads a noised entry.
        print(f"  LOOP DENOISE ON: grid {model_cfg.loop_denoise_grid}, teacher-forced at "
              f"train, rolled out at eval - each pass of the slot loop gets a JOB, a "
              f"noise level. Pass i of a slot whose realised depth is T_s enters at "
              f"t_i = (i-1)/T_s on the straight line z_t = (1-t) z0 + t x0, with x0 the "
              f"FROZEN reference encoder's code of the NEXT span (prefix_k="
              f"{model_cfg.prefix_k} cells) and ONE z0 per slot. AT TRAIN the entry is "
              f"built from the TRUE x0 and NOT from the previous pass (LCM's teacher "
              f"forcing, Eq. 16 x0-prediction, omega(t)=1): the passes are INDEPENDENT, "
              f"which is what gives each one a job. AT EVAL the loop RUNS - pass i+1 "
              f"enters at the re-noised prediction of pass i on the same line with the "
              f"same z0 - so the exit is a SAMPLE and two forwards differ. "
              f"weight={model_cfg.loop_denoise_weight}, code_target_weight="
              f"{model_cfg.code_target_weight} (0 by construction: a slot's LAST pass IS "
              f"the exit term). READ `train/loop_denoise_l2_t{{t}}` ACROSS t FIRST - the "
              f"per-pass term is what this arm exists to move - and score the arm on the "
              f"READER's paired CE against its rung below, never on the K-curve alone: a "
              f"denoiser's K-curve rises trivially because the early passes are noisy "
              f"(the theatre risk the note names). NOT COMPARABLE with any other arm: "
              f"gain_est, loop/core_gain*, loop/delta_*, loop/eff_rank, loop/in_norm, "
              f"fixed_point, pass_residual and tul/code_target_cos_l{{t}} all read an "
              f"independently noised entry at train. Train and eval differ BY DESIGN "
              f"(teacher forcing vs rollout); LCM 2.3.2's epsilon-scaling against "
              f"exposure bias is NOT built "
              f"(.agents/notes/proposed/architecture/"
              f"2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md, Part 2 change 1)",
              flush=True)
    if model_cfg.spandec_parallel:
        _kp = model_cfg.spandec_parallel_k
        _ke = model_cfg.code_enum_k
        _readers = (f"head codes K={_kp}, enumerated under the exact mixture" if _kp > 1
                    else f"the K={_ke} loop rollouts' exit states under the exact mixture"
                    if _ke > 1 else "one reader (no code)")
        _cap = model_cfg.spandec_parallel_span_cap
        _span = (f"the FIRST {_cap} tokens of span s+{model_cfg.spandec_target_offset}"
                 if _cap else f"span s+{model_cfg.spandec_target_offset}")
        if model_cfg.fan_all_wta_grader == "head":
            _readers = (f"the fan's M={model_cfg.fan_k} cells, EACH READ ALONE, as the "
                        f"write-all fan's WTA GRADER (its only term is the relaxed WTA, "
                        f"weight fan_all_wta_lambda={model_cfg.fan_all_wta_lambda}; its "
                        f"mean-of-cells term is NOT run; target = the labels the coda's "
                        f"WTA table scores)")
            _span = "each cell's next span (the coda table's labels)"
        print(f"  TUL PARALLEL SPAN HEAD ON (LXTUL-E): {_readers}, "
              f"weight={model_cfg.spandec_parallel_weight}, "
              f"{'BESIDE' if model_cfg.spandec else 'INSTEAD OF'} the teacher-forced span "
              f"decoder — {_span} decoded AT ONCE from z with NO token input. "
              f"TRAINING-ONLY target and scorer, never a decoder "
              f"(morph/model/tul_spandec_parallel.py)", flush=True)
        if model_cfg.spandec_parallel_detach:
            print("  TUL PARALLEL SPAN HEAD IS A PROBE: it reads the exit state DETACHED; "
                  "its term trains the head alone and sends NO gradient to the loop, the "
                  "codes, the front or the tied table", flush=True)
    if model_cfg.code_enum_k > 1:
        print(f"  LXTUL-E CODE ON: K={model_cfg.code_enum_k} enumerated codes, re-added at "
              f"the end of EVERY slot-loop pass, h <- f(h) + {model_cfg.code_enum_ratio} * "
              f"rms(f(h)).detach() * u_k (u_k: a learned regular simplex, unit RMS, sum "
              f"zero, no learned scale). The front runs ONCE; the slot loop and the coda run "
              f"on K rollouts per row at train AND eval (shared depth draw and dropout "
              f"masks; coda checkpointed per block). Coda loss: the exact per-span mixture "
              f"over the K rollouts. A label-free forward returns the per-span sequential "
              f"Bayes read. Read `tul/enum_width_gain_best`, `tul/enum_w_entropy`, "
              f"`tul/enum_exit_sep` (morph/model/tul_code_enum.py)", flush=True)
    if model_cfg.code_policy_k > 1:
        print(f"  CODE POLICY ON (arm B): C={model_cfg.code_policy_k} LX codes, ONE rollout, "
              f"one code PER SLOT re-added at the end of EVERY slot-loop pass, h <- f(h) + "
              f"{model_cfg.code_policy_ratio} * rms(f(h)).detach() * u_c. A linear policy "
              f"head on the slot's DETACHED loop-entry state picks c: SAMPLED at train "
              f"(global RNG), {model_cfg.code_policy_eval.upper()} at eval. REINFORCE on "
              f"r = -mean token CE of the span the slot feeds, baseline = leave-one-out "
              f"batch mean + value head; lambda={model_cfg.code_policy_lambda}, "
              f"entropy={model_cfg.code_policy_entropy}, "
              f"value_lambda={model_cfg.code_policy_value_lambda}, folded as "
              f"`code_policy_weighted` (train only). The coda runs ONCE. Read "
              f"`val/code_policy_vs_random` FIRST (val-only extra pass: CE with random "
              f"codes minus CE with the policy's argmax), then `tul/code_policy_entropy` and "
              f"`tul/code_policy_share_k*` (morph/model/tul_code_policy.py)", flush=True)
    if model_cfg.code_enum_k > 1 and model_cfg.code_enum_credit == "hard":
        _ke, _eh = model_cfg.code_enum_k, model_cfg.code_enum_hard_eps
        print(f"  LX HARD CREDIT ON: the TRAINING token loss is sum_k c_k CE_k(span), c = "
              f"stop_grad({1.0 - _eh:g} on the span's best rollout, {_eh / (_ke - 1):g} on "
              f"each other) instead of the mixture's posterior credit. Eval, the deploy "
              f"read and every val metric stay the mixture; train/loss stays the mixture "
              f"NLL (`tul/enum_ce_mix`), the objective is `tul/enum_hard_obj`, the wins "
              f"`tul/enum_code_win{{k}}` / `tul/enum_win_entropy` "
              f"(.agents/notes/proposed/architecture/2026-09-26-lx-hard-credit.md)",
              flush=True)
    if model_cfg.code_enum_k > 1 and model_cfg.fan_k > 0:
        print(f"  LX-FAN ON: K={model_cfg.code_enum_k} code rollouts x M={model_cfg.fan_k} "
              f"write-all cells per slot. Rollout k re-adds u_k to EVERY cell at the end of "
              f"every pass (sized by that cell's own rms); each cell is written 1:1 into its "
              f"prefix cell; the coda reads all M under each code and the token loss is the "
              f"exact per-span mixture over the K rollouts. The register pools the prelude "
              f"ONCE (base rows) and is tiled. "
              + (f"WTA over the cells ON (lambda {model_cfg.fan_all_wta_lambda}, eps "
                 f"{model_cfg.fan_select_eps}, winner={model_cfg.fan_all_wta_winner!r}, "
                 f"grad_rollouts={model_cfg.fan_all_wta_grad_rollouts!r}): "
                 + (f"per (rollout, slot) winner, M no-grad coda passes (one per stream, "
                    f"K-fold batch) + one grad pass over all K*B0 rows. "
                    if model_cfg.fan_all_wta_winner == "per_rollout" else
                    f"per (rollout, slot) winner picked by an InfoNCE score (temp="
                    f"{model_cfg.fan_all_wta_latent_temp}) against the true next span "
                    f"(lab/divergence/latent_wta_probe.py), NO coda pick passes at all; "
                    f"+ one grad pass over all K*B0 rows. "
                    if model_cfg.fan_all_wta_winner == "latent" else
                    f"ONE winner per slot shared by every rollout, ranked by the TRUE "
                    f"mixture posterior from the model's OWN deployed coda pass (built "
                    f"here, cached, reused below instead of recomputed); then M no-grad "
                    f"coda passes (one per stream, BASE-row batch, a quarter the rows at "
                    f"K={model_cfg.code_enum_k}) pick the stream, + one grad pass over "
                    + (f"a BASE-row batch (each slot's own MAP rollout — a different "
                       f"objective from the mean over K, not merely cheaper). "
                       if model_cfg.fan_all_wta_grad_rollouts == "map" else
                       f"all K*B0 rows (the mean over rollouts, unchanged). "))
                 if model_cfg.fan_all_wta_lambda > 0 else "No WTA term. ")
              + "No parallel head (refused). "
              "`fan/*_ce` oracle readings are per-span MIXTURES over the K rollouts; "
              "`tul/enum_exit_sep` reads the written cells "
              "(.agents/notes/proposed/architecture/2026-09-26-lx-fan.md)", flush=True)
    if model_cfg.gram and model_cfg.gram_objective == "iw":
        # LXTUL-GK (tul.gram_objective="iw", 2026-09-23). NOT the LXTUL-G banner below:
        # there is no posterior and no KL, and train and eval draw from the SAME prior.
        print(f"  LXTUL-GK ON: the LXTUL-G Gaussian step after every slot-loop pass "
              f"({'mean + variance heads' if model_cfg.gram_mean else 'MEAN-FREE (m = 0)'}, "
              f"sigma/r at init {model_cfg.gram_sigma_init}), NO posterior, NO KL. TRAIN "
              f"runs the front once and the slot loop + coda on K={model_cfg.gram_iw_k} "
              f"PRIOR rollouts per row (shared depth draw and token dropout; coda "
              f"checkpointed per block at K > 1); the loss is the multi-sample bound "
              f"-sum_span log mean_k exp(sum_j w_j log p_k) / sum_j w_j, which summed over "
              f"a span equals lab/divergence/lxtul_g_probe.py's per-token Bayesian read. "
              f"EVAL is unchanged: one seeded prior sample. Read `tul/gk_width_gain` "
              f"(ce_single - ce_iw), `tul/gk_w_entropy`, `tul/gram_sigma_ratio_prior` "
              f"(prereg lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)",
              flush=True)
    elif model_cfg.gram:
        # LXTUL-G (tul.gram, 2026-09-23). The banner names the two things that make the
        # arm unreadable if forgotten: train and eval draw from DIFFERENT distributions,
        # and the K-curve is read on ONE seeded prior sample.
        print(f"  LXTUL-G ON: a learned Gaussian step after every slot-loop pass, "
              f"h_t = u_t + r_t*(m + s*n) with r_t the slot's detached RMS; "
              f"{'mean + variance heads' if model_cfg.gram_mean else 'MEAN-FREE (m = 0)'}, "
              f"sigma/r at init {model_cfg.gram_sigma_init}, head width "
              f"{model_cfg.gram_hidden}. TRAIN draws the POSTERIOR (it sees the next "
              f"span's prelude states through a one-query pool); EVAL draws the PRIOR from "
              f"a seeded generator (gram_eval_seed), so a forced-depth sweep is paired per "
              f"pass. Loss += beta={model_cfg.gram_beta} * sum KL_bal / n_tokens, KL "
              f"balancing {model_cfg.gram_kl_balance}, free bits "
              f"{model_cfg.gram_free_bits} nats per slot. The fixed-point term reads the "
              f"DETERMINISTIC u_T. Read `tul/gram_kl` (collapse below 0.5 nats per slot), "
              f"then the exposure gap ce_prior@1 - ce_post from "
              f"lab/divergence/lxtul_g_probe.py (prereg "
              f"lab/experiments/failures/2026-09-23-lxtul-g-panel.md)", flush=True)
    if model_cfg.vq_codes > 0:
        _dc = model_cfg.vq_dim or (int(cfg.model.d_model) // model_cfg.vq_codes)
        print(f"  TUL DISCRETE THOUGHT ON: vq_codes={model_cfg.vq_codes} "
              f"codebook={model_cfg.vq_codebook} d_c={_dc} groups={model_cfg.vq_groups} "
              f"beta={model_cfg.vq_beta} weight={model_cfg.vq_weight} "
              f"reset_after={model_cfg.vq_reset_after} - the loop's EXIT state goes "
              f"through W_vq to {model_cfg.vq_codes} sub-vectors, each snapped to the "
              f"nearest entry of ONE shared codebook (cosine: both sides l2-normalised, "
              f"so the encoder's scale cannot decide the match), and the quantized "
              f"sub-vectors are lifted back to {model_cfg.vq_codes} full-width cells "
              f"written 1:1 into the prefix cells (prefix_k == vq_codes). The CODA is "
              f"unchanged. Rank is given BY CONSTRUCTION against the measured collapse "
              f"(slot_eff_rank 5.76 in 1024 dims, pairwise cos 0.71). A straight-through "
              f"estimator is the ONLY edge the token CE has into the loop here. "
              f"STEP 0 IS NOT THE RULER: a quantizer cannot be zero-init, so the coda "
              f"starts on a near-arbitrary code and this arm's early CE is worse by "
              f"construction. Read `tul/vq_perplexity` (codebook usage, in "
              f"(1, {model_cfg.vq_codebook}]) FIRST - at 1 every span picked the same "
              f"symbol and the channel carries nothing, whatever the CE says "
              f"(lab/experiments/planned/2026-09-13-arc-discrete-thought-vq.md)",
              flush=True)
    if model_cfg.center_exit:
        print("  TUL ROW-CENTERED EXIT ON (tul.center_exit): after the loop and the "
              "think-once stack, every VALID slot's exit state has the ROW's mean over "
              "its valid slots subtracted and ONE learned bias `b_center` (zero-init) "
              "added back. Pads are untouched; at slot_cells > 1 each CELL INDEX is "
              "centered separately across the row. Every reader sees it - the MUX, the "
              "span decoder, SIGReg, the energy and the coda's prefix write - because it "
              "is applied to the CARRIER at the one seam upstream of all five. Aimed at "
              "`val/slot_pairwise_cos` 0.7104: READ THAT, not `val/slot_eff_rank`, which "
              "already centers its covariance globally and so cannot see a shared offset "
              "at all. PRECEDENT: `tul.center_bag_mean` on the SEED (arm tul_center, "
              "2026-08-27) FAILED its < 0.20 cosine prediction (0.337-0.578 vs a control's "
              "0.485-0.589) and bought +0.0015 nats of loop worth "
              "(lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md)",
              flush=True)
    if model_cfg.row_contrast_lambda > 0.0:
        print(f"  TUL ROW CONTRAST ON: row_contrast_lambda="
              f"{model_cfg.row_contrast_lambda} tau={model_cfg.row_contrast_tau} - an "
              "InfoNCE term WITHIN each row: span i+1's pooled prelude states must pick "
              "slot i's exit state out of the row's other valid slots. Negatives are the "
              "row's OTHER slots only; pads are never keys; the pooled targets are "
              "DETACHED, so the term trains the WRITE and `W_contrast`, never the prelude "
              "or the embedding table. It has an ABSOLUTE reference: with the row's states "
              "indistinguishable the term reads log(n_valid) and `tul/row_contrast_acc` "
              "reads 1/n_valid. READ `tul/row_contrast_acc` - it is the direct "
              "distinctness number the rank-collapse lane has never had "
              "(lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md)",
              flush=True)
    if model_cfg.prefix_source != "exit":
        print(f"  TUL PREFIX SOURCE = {model_cfg.prefix_source!r} (prefix_k "
              f"{model_cfg.prefix_k}): 'trajectory' gives cell k of a slot the state AFTER "
              f"PASS k+1 and the LAST cell the EXIT state, so every written pass has its "
              f"own reader and its own gradient edge; a cell no pass reached is a PAD "
              f"(zero carrier, cut out of the coda's key set, no label). 'exit_repeat' is "
              f"the matched-count CONTROL: every cell holds the exit. 'entry_exit' holds "
              f"the loop's ENTRY (z1 = core_init(e)) in cell 0 and the exit in the LAST "
              f"cell, the rest exit copies: the BINDING control, because "
              f"I((z1..zT);Y) = I(z1;Y) says the whole trajectory carries exactly what the "
              f"ENTRY carries (.agents/notes/proposed/architecture/"
              f"2026-09-13-information-view-of-the-slot-loop.md). Both add the "
              f"zero-init per-cell embedding `tul.E_pass`, so trajectory minus exit_repeat "
              f"isolates the cell CONTENT "
              f"(lab/experiments/planned/2026-09-13-arc-trajectory-prefix.md)",
              flush=True)
    if model_cfg.loop_reads_tokens:
        print("  TUL LOOP READS TOKENS ON: the SHIPPED core stage runs over EVERY "
              "position (tokens + slot cells, one sequence, `_core_region`'s per-SAMPLE "
              "Poisson depth) under `causal AND (same span OR a slot cell)`, so a token "
              "reaches an earlier span ONLY through a cell and the cells are still the "
              "whole cross-span channel. NOT the paid loop, whose core is unrestricted. "
              "No prefix write (the cell's state is already at its position); z is "
              "gathered at the slot's first cell. Forced-depth eval goes through "
              "model.cfg.mean_depth "
              "(lab/experiments/planned/2026-09-13-arc-loop-reads-tokens.md)",
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
        if model_cfg.tg_coda_token_reach:
            print(f"  TUL STRICT CODA TOKEN REACH {model_cfg.tg_coda_token_reach}: a coda "
                  f"token ALSO reads the tokens of the {model_cfg.tg_coda_token_reach} "
                  f"previous span(s) directly; conv/value-shift/retention resets unchanged; "
                  f"token-path receptive field = reach x n_coda spans, the loop beyond "
                  f"(.agents/notes/proposed/architecture/"
                  f"2026-09-26-slot-channel-width-and-reach.md)", flush=True)
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
