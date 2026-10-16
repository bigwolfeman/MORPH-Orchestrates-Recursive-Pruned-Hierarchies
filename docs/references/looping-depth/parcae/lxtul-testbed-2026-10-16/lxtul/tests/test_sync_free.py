"""`sync_free_loop` on plain Parcae (per-sequence depths) against the original bookkeeping.

EXACT: same loop bounds, same per-pass `step` / `total_steps` values and dtype, same loss.
CPU only, so it covers the host-side logic; the pinned-copy branch (`input_embeds.is_cuda`) and
the sync count are checked by the GPU steps in gpu_todo.sh.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_sync_free.py
"""
from __future__ import annotations

import pytest
import torch

import parcae_lm

V, SEQ = 256, 32


def _model():
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ,
        mean_recurrence=6, mean_backprop_depth=3)
    m = cfg.construct_model(gradient_checkpointing=False)
    assert m.recurrent_iteration_method == "per-sequence"
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)
    return m.train()


def _run(sync_free: bool, step: int):
    m = _model()
    m.sync_free_loop = sync_free
    m.step = step
    seen = []
    orig = m.update_recurrent_state

    def spy(x, emb, freqs, mask, step_t, total):
        seen.append((int(step_t), step_t.dtype, int(total), total.dtype))
        return orig(x, emb, freqs, mask, step_t, total)
    m.update_recurrent_state = spy
    ids = torch.randint(0, V, (4, SEQ), generator=torch.Generator().manual_seed(1))
    out = m(ids, labels=ids)
    out["loss"].backward()
    grads = [p.grad.clone() for p in m.parameters() if p.grad is not None]
    return out["loss"].detach(), seen, grads


@pytest.mark.parametrize("step", [0, 1, 2, 3, 4])
def test_same_loop_same_loss(step):
    l0, s0, g0 = _run(False, step)
    l1, s1, g1 = _run(True, step)
    assert len(s0) >= 1 and s0 == s1, "per-pass step and total_steps values and dtypes"
    assert torch.equal(l0, l1)
    assert len(g0) == len(g1) and all(torch.equal(a, b) for a, b in zip(g0, g1))


def test_depths_vary_across_steps():
    """The comparison above means something only if the drawn depth changes with the step."""
    totals = {_run(False, s)[1][-1][2] for s in range(5)}
    assert len(totals) > 1, totals
