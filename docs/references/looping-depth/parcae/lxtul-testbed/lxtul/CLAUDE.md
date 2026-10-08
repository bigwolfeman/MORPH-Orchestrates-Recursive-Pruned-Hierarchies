# lxtul/ — MORPH's LXTUL slot loop on a Parcae backbone (testbed)

Branch `lxtul-testbed`. A clean backbone to test whether the slot loop contributes,
without MORPH's Hyper-Connections, ternary weights, CCA/CSA/HCA attention or AdEMAMix.

- `SPEC.md` — the LXTUL forward and losses read off MORPH master 3f9aae2, with file:line
  citations. `model.py` cites its section numbers. Read it before changing the model.
- `model.py` — `LXTULParcae(Parcae)`: Parcae's own blocks, injection, value embeddings and
  init, plus the LXTUL mechanism. The module docstring lists every backbone deviation
  from MORPH (one stream, full-width injection with floor gain 0.447 instead of 0.865,
  no x0/bigram re-injection, no dropout, kv heads = heads).
- `attention.py` — `FlexSelfAttention` (class-swapped onto Parcae blocks, same params) and
  the three strict masks. Parcae's stock attention IGNORES its mask argument.
- `ce.py` — chunked linear CE, slot-id logit masked, grads computed in the forward.
- `data.py` — MORPH's tokenizer, OWT stream, boundary rule and packer, unchanged (so CE is
  comparable to MORPH ledger CE). Needs MORPH_ROOT on sys.path (set inside).
- `build.py` / `bench.py` / `configs/bench.yaml` — plain Parcae vs Parcae-LXTUL throughput.
- `tests/test_geometry.py` — causality and "loop is the only cross-span channel" on the
  forward (GPU). Run it after ANY change to masks, the write or the layout.

Run with the MORPH venv: `PYTHONPATH=/home/wolfe/parcae:$MORPH_ROOT
/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python -m lxtul.bench`, under
`flock /home/wolfe/morph-scratch/gpu.lock`, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Gotcha: the bench cycles 16 batches, so its CE is memorization (trained batch 0.92 vs
unseen 6.51 after 40 steps, 2026-10-06). Read only tok/s and memory from it.
