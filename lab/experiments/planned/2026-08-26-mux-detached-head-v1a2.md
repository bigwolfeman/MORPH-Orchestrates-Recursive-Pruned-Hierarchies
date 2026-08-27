# Experiment v1a-2: was the MUX failure the gradient PATH, or the head itself?

Status: **planned, code implemented and unit-tested, runs NOT started.**
Predictions frozen 2026-08-26 before either run.

## Question

[Arm v1a](../failures/2026-08-25-mux-head-arm-v1a.md) aborted at step 2800 while
its control ran healthy past 3250 — the MUX head did not just fail to help, it
INDUCED a divergence. Post-mortem found a mechanism in the code, not in the idea:

`_tul_mux_loss` read the readout matrix with no `detach`, and
`embed.lm_weight()` in MORPH is **weight-tied to the input embeddings** (a concat
of `euc_embed.weight` and the Lorentz log-map). So the auxiliary gradient trained
the embedding table itself — the same table (a) every token representation reads
and (b) the slot input is a bag-mean OF (`E_slot + mean(embed(span))`). That is a
feedback loop into the plan's own input, and a dense corruption of the LM head:
the softmax term puts gradient on EVERY vocabulary row every step.

MUX's own protocol never has this problem — they LoRA-finetune a PRETRAINED model
and use W as a fixed readout for supervision (their Table 9 / §11.1). Training
from scratch under both objectives is our deviation, not theirs.

Does removing that path rescue the arm?

## Arms

Both are `tul_v1a` plus one change each, same protocol as v1a and the control
(batch 6, 3500 steps, seed 1, `ademamix_alpha_cap=3.5`, `use_kernels=false`,
`eval_every=250`, `ckpt_every=500`). Control remains `seedsweep-s1`.

- **v1a-2a** (`tul_v1a2a.yaml`): `mux_detach_head: true`, `mux_beta: 1.0`.
  Isolates the gradient path alone against v1a.
- **v1a-2b** (`tul_v1a2b.yaml`): `mux_detach_head: true`, `mux_beta: 0.1`.
  Adds the weight reduction. In v1a the auxiliary (6.4-8.0 nats) outweighed the
  LM objective (~5 nats), which is not the regime the paper's beta=1.0 sits in.

Run 2a FIRST. Sequential only (UPS).

Implementation: `TULConfig.mux_detach_head` (default **true** — the corrected
default; `false` is kept only so v1a's failure stays reproducible as an
ablation). Contract test `test_detached_head_leaves_absent_vocab_rows_untouched`
uses vocabulary rows absent from the batch as an exact discriminator: the dense
readout term reaches them, the legitimate mux → z → core → slot-input path
cannot. Suite: 8 tests in `tests/test_tul_mux.py`.

## Predictions (frozen 2026-08-26, before either run)

Measured baselines: v1a aborted at 2800, val ppl_tok 973 at 2750, `mux_local`
11.20 → 6.36 (step 1320) → 7.63 (2640). Control `seedsweep-s1`: no abort, val
ppl_tok 117 at 2750, ~94-106 at 3000-3250.

- **Q1 (the path was the cause):** v1a-2a reaches step 3500 with no divergence
  abort. If it aborts too, the gradient path was NOT the cause and the head
  itself is the problem.
- **Q2 (no LM tax):** v1a-2a `val/ce_tokens` at step 3000 is within **+0.10
  nats** of `seedsweep-s1` at the same step. (Looser than v1a's +0.05: the
  auxiliary still shapes shared readout weights `lm_mixer`/`final_norm`, so a
  small tax is expected and is not by itself a failure.)
- **Q3 (the head still learns):** v1a-2a `tul/mux_local` reaches **≤ 6.36** (its
  undetached best) by step 3500 and does NOT turn back up before 3000. A
  detached head that learns WORSE would mean the corruption was load-bearing for
  the auxiliary — the head was training the table to make its own job easy.
- **Q4 (plan worth):** plan-off ablation cost at the step-3000 checkpoint
  ≥ **0.04 nats**, against the 0.0191 baseline. This is v1a's P2, unchanged.
- **Q5 (beta, only if 2a survives):** v1a-2b's val CE at step 3000 is no worse
  than v1a-2a's. If 2a survives and 2b is better, beta was a second, independent
  problem; if 2a survives and 2b is worse, beta=1.0 was fine once detached.

Failure reading: 2a aborts → the head's TARGET is wrong for this data (next-span
bag prediction in OpenWebText may be near-irreducible), not its wiring, and the
next question is the target, not another weight.

## Method notes

- Sequential runs only (UPS). wandb on; the tul manifest logs `mux_detach_head`.
- `train/loss`, val loss and the divergence guard all report the MODEL's CE with
  `mux_weighted` subtracted; the console step line prints the raw objective.
