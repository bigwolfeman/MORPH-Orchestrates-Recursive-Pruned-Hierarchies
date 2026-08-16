"""Resolve the Hydra ``tul:`` block into the objects the training loop needs.

Spec: ``docs/tul-spec.md`` §8 (config keys), §5 (schedule), §4 (data).

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

__all__ = ["TulRuntime", "build_tul_runtime"]

NEVER = "never"


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
    eos_id = int(tok.eos_token_id if tok.eos_token_id is not None else 0)
    if slot_id == eos_id:
        raise ValueError("tul.slot_token resolves to EOS — pick an unused special token")

    substrings = tuple(tc.get("boundary_substrings", BOUNDARY_SUBSTRINGS))
    lut = boundary_lut_from_tokenizer(
        tokenizer_name, vocab_size, eos_id,
        cache_dir=cache_dir,
        suffix_chars=str(tc.get("boundary_chars", BOUNDARY_SUFFIX_CHARS)),
        substrings=substrings,
    )
    if bool(lut[slot_id]):
        raise ValueError(
            f"tul.slot_token {slot_token!r} (id {slot_id}) is itself a boundary token — "
            f"it would cut spans it is only supposed to mark")

    rule = BoundaryRule(
        is_boundary=lut,
        min_span=int(tc.get("min_span", 4)),
        span_cap=int(tc.get("span_cap", 32)),
        eos_id=eos_id,
        fixed_stride=int(tc.get("fixed_stride", 0)),
    )
    prefix_k = int(tc.get("prefix_k", 2))
    data_cfg = TulDataConfig(rule=rule, prefix_k=prefix_k, slot_id=int(slot_id),
                             max_slots=int(tc.get("max_slots", 0)))
    model_cfg = TULConfig(
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
        "slot_mean_depth": model_cfg.slot_mean_depth or int(cfg.model.mean_depth),
        "slot_max_depth": model_cfg.slot_max_depth or int(cfg.model.max_depth),
    }
    print(f"  TUL ON: activate_at={activate_at} slot_token={slot_token!r}(id {slot_id}) "
          f"|B|={int(lut.sum())} min_span={rule.min_span} span_cap={rule.span_cap} "
          f"prefix_k={prefix_k} max_slots={spec.max_slots} L_total={spec.l_total} "
          f"p_drop={model_cfg.token_state_dropout}"
          + (f" fixed_stride={rule.fixed_stride}" if rule.fixed_stride else ""), flush=True)
    return TulRuntime(model_cfg=model_cfg, data_cfg=data_cfg,
                      activate_at=activate_at, manifest=manifest)
