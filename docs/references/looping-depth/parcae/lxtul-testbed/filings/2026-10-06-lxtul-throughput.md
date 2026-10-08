# 2026-10-06 LXTUL throughput: Parcae backbone vs MORPH

Not preregistered. Wolfe's gate ("if either is slower than our current throughput we will
need to address it before doing a bunch of arms") was the only criterion stated before
the run. Read this as a measurement record, not an experiment verdict.

Same GPU (RTX 5090, power limit 575 W), same day, same MORPH rows (StarCoder2 tokenizer,
OWT train stream, MORPH packer, seq 1024, batch 6). tok/s = steps/s x 6 x 1024, MORPH's
formula. Shape: d 1024, 8 heads, 4 prelude / 6 core / 4 coda, mean depth 6.

| model | tok/s | step | peak alloc | source |
| --- | --- | --- | --- | --- |
| Parcae-LXTUL (`lxtul/model.py`, commit of this file) | 23,528 | 0.261 s median, 40 steps | 14.5 GiB | `parcae-results.json`, wandb `adew-me/parcae-lxtul/wud5r9v0` |
| plain Parcae (stock, Poisson(6), BPTT 6) | 23,019 | 0.267 s | 21.1 GiB | same |
| MORPH LXTUL (`lxtul.yaml`, master 3f9aae2) | 12,250 | steps 100-180 log lines | 19.8 GB | `morph-lxtul-log.txt` |
| MORPH plain (`notul_panel_norm_match`) | 10,300 | steps 100-180 | 10.2 GB | `morph-plain-log.txt` |

Both Parcae arms clear the gate: Parcae-LXTUL is 1.92x MORPH LXTUL.

Method differences, stated so they are not hidden:

- Parcae numbers: median wall time of 40 timed steps after 10 warmup steps, cycling 16
  batches; each step is forward, backward, clip 1.0, MuonAdamW, EMA twin update. MORPH
  numbers: its trainer's logged `tok/s` (20-step windows, data prefetch on, AdEMAMix).
- Optimizer differs (MuonAdamW vs AdEMAMix fused fp32); weights differ (bf16 autocast vs
  ternary STE). These are backbone differences, part of what is being compared.
- Parcae kv heads 8 (Parcae's core blocks force kv = heads); MORPH uses 4.
- The Parcae bench CE is memorization of the 16 cycled batches (trained batch 0.92, unseen
  6.51 after 40 steps). It says nothing about learning.

Correctness checks run the same day (`lxtul/tests/test_geometry.py`, 3 passed): future
tokens move no earlier position (max |dh| = 0); span s sees span s-1 only through the loop
write (0 with `W_prefix` zeroed); loser cells enter the coda as exact zeros. Sabotage: a
plain causal coda mask fails the second test.

Not measured: training CE on a real (non-cycled) stream, the K-sweep, memory headroom at a
larger batch, and torch.compile of the whole loop (only per-block compile is used).
