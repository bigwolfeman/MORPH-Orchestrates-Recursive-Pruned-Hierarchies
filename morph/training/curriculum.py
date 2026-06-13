"""Curriculum scheduler — maps a global training step to a curriculum stage.

A stage = (seq_len, RoPE context_len, micro-batch). The scheduler owns only the *when*
(step → stage index); the train loop owns the *what* (checkpoint → RoPE set_context →
loader.set_stage → micro-batch/grad-accum swap) when ``stage_at`` advances.

Stage step-counts can be given directly, or derived from per-stage TOKEN shares via
``from_token_shares`` (steps_k = share_k · total_tokens / (eff_batch · seq_len_k)), which
honors data proportions even though tokens-per-step changes as seq_len ramps.
"""
from __future__ import annotations
import numpy as np

__all__ = ["CurriculumScheduler"]


class CurriculumScheduler:
    def __init__(self, stage_steps: list[int]):
        self.stage_steps = [int(s) for s in stage_steps]
        if any(s <= 0 for s in self.stage_steps):
            raise ValueError(f"stage_steps must be positive: {self.stage_steps}")
        self.bounds = [int(x) for x in np.cumsum(self.stage_steps)]   # exclusive end step
        self.n = len(self.stage_steps)
        self.total_steps = self.bounds[-1]

    def stage_at(self, step: int) -> int:
        for k, b in enumerate(self.bounds):
            if step < b:
                return k
        return self.n - 1

    def transitions(self) -> list[int]:
        """Steps at which the stage advances (the first step of each stage after stage 0)."""
        return self.bounds[:-1]

    @classmethod
    def from_token_shares(cls, shares: list[float], total_tokens: int,
                          seq_lens: list[int], eff_batch: int) -> "CurriculumScheduler":
        assert len(shares) == len(seq_lens)
        ssum = float(sum(shares))
        steps = []
        for sh, L in zip(shares, seq_lens):
            toks = (sh / ssum) * total_tokens
            steps.append(max(1, int(round(toks / (eff_batch * L)))))
        return cls(steps)
