"""Resolve the Hydra ``tul:`` block into the objects the training loop needs.

Spec: ``docs/tul-spec.md`` §8 (config keys), §5 (schedule), §4 (data);
the shipped arm is the paid loop of ``docs/tul-paid-loop-recipe.md``.

Kept out of ``train.py`` so the tokenizer work (resolving the boundary id set and the
slot token) is one testable unit, and so ``train.py``'s TUL seam is three lines. The
whole block resolves to ONE object; ``None`` means plain MORPH and nothing downstream
changes — no TUL parameters are constructed and the forward never sees a layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from morph.model.tul import TULConfig
from morph.model.tul_layout import (
    BOUNDARY_SUBSTRINGS,
    BOUNDARY_SUFFIX_CHARS,
    BoundaryRule,
    TulDataConfig,
    boundary_lut_from_tokenizer,
)

__all__ = ["TulRuntime", "build_tul_runtime", "build_boundary_rule",
           "KNOWN_TUL_KEYS", "reject_unknown_tul_keys"]

NEVER = "never"

# Every key `tul:` may carry. Anything else is a retired arm (gate, tg_*, coda_sees_slots,
# tokens_through_core, per_slot_embed, fixed_stride, stp_lambda, …) or a typo, and either
# one must RAISE rather than run the shipped model under a name that promises something
# else (runtime-invariants §6b).
KNOWN_TUL_KEYS = frozenset({
    "activate_at", "slot_token", "boundary_chars", "boundary_substrings", "min_span",
    "span_cap", "prefix_k", "max_slots", "token_state_dropout", "emit_weight",
    "plast_weight", "slot_seed",
})


def reject_unknown_tul_keys(tc) -> None:
    """Raise ``ValueError`` naming every ``tul:`` key outside :data:`KNOWN_TUL_KEYS`."""
    unknown = sorted(str(k) for k in tc.keys() if str(k) not in KNOWN_TUL_KEYS)
    if unknown:
        raise ValueError(
            f"tul: has unknown key(s) {unknown}. Known keys: {sorted(KNOWN_TUL_KEYS)}. "
            f"Retired arm keys (gate, tg_*, coda_sees_slots, tokens_through_core, "
            f"per_slot_embed, fixed_stride, …) left the tree on 2026-09-03 with the "
            f"slot-only core — docs/tul-paid-loop-recipe.md.")


def build_boundary_rule(cfg, cache_dir: str = "ignore/tul_cache"):
    """``(rule, lut, eos_id, substrings)`` — THE span rule, from ``cfg.tul`` + the tokenizer.

    Extracted from :func:`build_tul_runtime` so the rule has ONE construction site. It is
    a property of the DATA, not of whether the slot apparatus is built, so it resolves
    even when ``tul.activate_at: never`` — which is exactly the case the FM planner's
    frozen-backbone spike (``lab/tulfm/``) needs: a backbone with no slots still has to
    segment rows with the same ``.;!?`` + newline/dash rule, the same ``min_span``, the
    same ``span_cap``, and the same EOS handling as every TUL arm.
    """
    tokenizer_name = str(cfg.data.tokenizer)
    vocab_size = int(cfg.model.vocab_size)
    tc = getattr(cfg, "tul", None)
    if tc is None:
        tc = {}
    reject_unknown_tul_keys(tc)

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
    )
    return rule, lut, eos_id, substrings


@dataclass
class TulRuntime:
    """Everything TUL needs at runtime, resolved once at train start."""

    model_cfg: TULConfig
    data_cfg: TulDataConfig
    activate_at: float
    manifest: dict = field(default_factory=dict)

    def activation_step(self, total_steps: int) -> int:
        """Step at which the layout switches on (spec §5). 0 → active from step 0."""
        return int(self.activate_at * total_steps)


def build_tul_runtime(cfg, cache_dir: str = "ignore/tul_cache") -> TulRuntime | None:
    """Build the TUL runtime from ``cfg.tul``; ``None`` when TUL is off.

    ``tul.activate_at: never`` returns None, which is what makes the plain recipe
    bit-identical to plain MORPH — no parameters, no layout, no branch
    (runtime-invariants §6b).
    """
    tc = getattr(cfg, "tul", None)
    if tc is None:
        return None
    reject_unknown_tul_keys(tc)
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
    data_cfg = TulDataConfig(rule=rule, prefix_k=prefix_k, slot_id=int(slot_id),
                             max_slots=int(tc.get("max_slots", 0)))
    model_cfg = TULConfig(
        prefix_k=prefix_k,
        slot_id=int(slot_id),
        token_state_dropout=float(tc.get("token_state_dropout", 0.15)),
        emit_weight=float(tc.get("emit_weight", 0.0)),
        plast_weight=float(tc.get("plast_weight", 1.0)),
        slot_seed=str(tc.get("slot_seed", "boundary")),
    )
    seq_len = int(cfg.data.seq_len)
    spec = data_cfg.spec_for(seq_len)
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
        "slot_seed": model_cfg.slot_seed,
        "boundary_chars": str(tc.get("boundary_chars", BOUNDARY_SUFFIX_CHARS)),
        "boundary_substrings": list(substrings),
        "min_span": rule.min_span,
        "span_cap": rule.span_cap,
        "prefix_k": prefix_k,
        "seq_len": seq_len,
        "max_slots": spec.max_slots,
        "l_total": spec.l_total,
        "token_state_dropout": model_cfg.token_state_dropout,
    }
    print(f"  TUL ON: activate_at={activate_at} slot_token={slot_token!r}(id {slot_id}) "
          f"|B|={int(lut.sum())} min_span={rule.min_span} span_cap={rule.span_cap} "
          f"prefix_k={prefix_k} max_slots={spec.max_slots} L_total={spec.l_total} "
          f"p_drop={model_cfg.token_state_dropout} slot_seed={model_cfg.slot_seed!r} "
          f"emit_weight={model_cfg.emit_weight} plast_weight={model_cfg.plast_weight}",
          flush=True)
    return TulRuntime(model_cfg=model_cfg, data_cfg=data_cfg,
                      activate_at=activate_at, manifest=manifest)
