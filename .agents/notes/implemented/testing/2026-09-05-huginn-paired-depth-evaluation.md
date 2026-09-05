# Agent Note: Huginn paired depth evaluation

Status: implemented

## Problem

The initial Huginn sweep loses all result rows when interrupted before its last
depth. Its independent random initial states confound paired comparisons. Its
profile labels input positions while the losses predict the next token.

## Decision

[The runner](../../../../lab/huginn/huginn_depth_sweep.py) pins the cached model and
tokenizer revision. It pairs initial RNG state by batch across depths and measures
the target token's offset. Atomic snapshots preserve each completed depth. Resume
rejects changed config, source code, or tokenized rows. The supervisor records the
actual child's PID and exit status. Online W&B includes the resolved Hydra config,
full model config, runtime versions, source hashes, and row hashes.

Effective model configuration uses canonical JSON key types before W&B compares
it with a resumed configuration. Real value changes remain errors. The one-off
`migrate_wandb_resume.py` helper accepts only the exact logging repair against its
pinned old source blob. It backs up the original checkpoint and records the source
transition. Every measured field and all other source checks remain unchanged.

[The original preregistration](../../../../lab/experiments/planned/2026-09-04-huginn-loop-contribution.md)
retains its frozen H1-H6 predictions. Its dated method amendment records the known
earlier partial output, sampling correction, and runtime gate predictions.

## Alternatives considered

- Reuse the initial runner unchanged. This loses interrupted results and mixes
  changes in random initialization with changes in depth.
- Downgrade the shared Transformers installation. This changes another project's
  environment. The local in-process tied-weight compatibility mapping is sufficient
  for the required loading contract, subject to the real CUDA smoke.
- Call the shard prefix validation data. There is no independent holdout selection
  here. The output instead names the actual train-shard prefix sampling.
- Permit arbitrary W&B configuration changes on resume. This would hide genuine
  changes. Canonical serialization fixes only the observed representation mismatch.

## Consequences

The profile is not numerically comparable with older profiles that bin input
positions. Cross-tokenizer row alignment and nats-per-token comparisons remain
invalid. The run answers depth effects within Huginn on fixed rows. It cannot
isolate the cause of MORPH's behavior. Contract tests cover RNG pairing, shifted
CE, target offsets, restored counts, and adjacent comparisons. GPU memory and
remote-model execution require the preregistered real smoke.
