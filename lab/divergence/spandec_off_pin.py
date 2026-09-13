"""OFF-state pin for `tul.spandec_reads_cells` / `tul.spandec_target_offset`.

Runs the SAME four fixtures on the tree before those knobs existed and on the tree after,
and prints loss, logit sum, grad sum and `state_dict` key count to sixteen digits. The two
runs must agree to the last printed digit, which is the "OFF is bit-identical" proof for
this change (the `TULSlotRegister` precedent: the register note proved its own OFF state
by running `d778845` and the new tree side by side, not by reading the diff).

The script deliberately uses ONLY the pre-change API — no `spandec_reads_cells`, no
`spandec_target_offset`, no `cells=` argument — so it imports and runs on both trees.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python lab/divergence/spandec_off_pin.py

The numbers it prints are pinned as literals in
``tests/test_tul_spandec_reads_cells.py`` and ``tests/test_tul_spandec_target_offset.py``,
so a later edit that moves the OFF forward fails a test rather than a memory.
"""
from __future__ import annotations

import numpy as np
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(5, V, size=(B, n))
    x[x == 4] = 5
    x[:, ::8] = DOT
    return x.astype(np.int64)


def batch(M: int, seed: int):
    spec = TulLayoutSpec(seq_len=64, prefix_k=max(M, 2), max_slots=10, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(ids(seed=seed), rule(), spec)
    return inp, lab, layout


def model(M: int, seed: int = 99):
    kw = dict(prefix_k=max(M, 2), slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
              mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
              tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict")
    if M > 1:
        kw.update(slot_cells=M, slot_cell_init="distinct")
    torch.manual_seed(seed)
    m = MORPHTransformer(tiny(tul=TULConfig(**kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


def run(M: int, seed: int) -> tuple[float, float, float, int]:
    torch.manual_seed(1234)
    m = model(M)
    x, y, layout = batch(M, seed)
    torch.manual_seed(7)
    out = m(x, labels=y, slot_layout=layout)
    loss = out["loss"]
    # A SECOND number from the same forward, so a change that leaves the total loss alone
    # and moves the span-decoder term still fails. `tul/spandec_ce` is the decoder's own
    # per-token conditional CE over its target span.
    sce = float(out["spandec_ce"])
    loss.backward()
    gsum = sum(float(p.grad.double().sum()) for p in m.parameters() if p.grad is not None)
    # LOGITS come from a second forward: with `labels` the model runs the fused CE and
    # returns `logits: None` by design. Same seed, so the per-slot depth draw is the same.
    torch.manual_seed(7)
    with torch.no_grad():
        lg = m(x, slot_layout=layout)["logits"].double()
        # The `slot_id` column is masked to -inf by design (spec 3.1), so the plain sum is
        # -inf and carries no information. Zero the non-finite entries and sum the rest.
        lsum = float(lg.masked_fill(~torch.isfinite(lg), 0.0).sum())
    return float(loss), lsum, sce, gsum, len(m.state_dict())


if __name__ == "__main__":
    for M in (1, 4):
        for seed in (0, 1):
            lo, ls, sce, gs, nk = run(M, seed)
            print(f"M={M} seed={seed} loss={lo!r} logit_sum={ls!r} spandec_ce={sce!r} "
                  f"grad_sum={gs!r} keys={nk}", flush=True)
