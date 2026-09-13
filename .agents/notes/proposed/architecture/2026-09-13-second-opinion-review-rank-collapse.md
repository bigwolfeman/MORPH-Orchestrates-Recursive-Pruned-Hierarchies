# Agent Note: second-opinion review of the slot loop, checked point by point

Status: proposed

## Problem

The slot loop reads token K1−K6 at or below 0.002 nats across roughly 45 arms
([`lab/experiments/planned/2026-09-04-loop-contribution-arc.md`](../../../../lab/experiments/planned/2026-09-04-loop-contribution-arc.md)).
Wolfe asked an outside model for a second opinion on why. Wolfe named it a suspected early
RSI model and said not to trust it fully, so every claim in its review needs to be checked
against a file before it moves the build queue. The verdict on the one point he wanted
kept: the review's Proposal 1 says the loop keeps collapsing the span's state into a rank so
low that the vector handed to the coda does not carry enough to do the job, and he called
this the real problem.

## Proposal

The review is a document, not a source. Each of its proposals is checked here against filed
numbers, not against its own restatement of them.

| Proposal | What the review claimed | Checked against | Verdict |
|---|---|---|---|
| Rank collapse (the reviewer's Proposal 1 body, "dimensional & rank starvation") | The slot state's effective rank is 6.3-7.1 in 1024 dims, too low to carry an iterative computation | [`2026-09-13-information-view-of-the-slot-loop.md`](2026-09-13-information-view-of-the-slot-loop.md) and the arc ledger: per row effective rank 5.7-7.3, pairwise cosine 0.72-0.77 (ruler 6.34 / 0.74); geometry audit 1.7-4.8; across 4,906 slots from many rows, rank 18-23 | **Adopted.** The number is real and is the basis of the Thought Register build below. |
| DPI framing (deterministic passes add no Shannon information; loop value is relay or extractability) | A theoretical account for why the slot loop is flat while the plain loop earns | `lab/theory/tul_information/` (Lean 4, `Joint.mi_iterate_le`, `Joint.mi_traj`, `Joint.mi_relay_redundant`; build verified sorry-free, exit 0) | **Adopted independently.** The theory was already built and verified in this session before the review arrived; the review's account and the Lean account agree, and the note above treats them as two readings of one fact rather than crediting the review with the proof. |
| "Reader insensitivity" (coda ignores a perturbed exit state) | 10% RMS perturbation of the exit moves coda loss <= 0.0005 nats; a critic cannot tell pass t from pass t-1 | [`lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md`](../../../../lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md) | **Adopted**, already filed before the review. |
| Build A (`prefix_source: trajectory`) is dead on arrival by `mi_traj` | The trajectory carries exactly what the entry carries, so it cannot beat the exit-repeat control | Same information note, "Correction to the brief" section: `mi_traj` says the trajectory equals the ENTRY, and the entry is measured 0.0185 nats worse than the exit, not better; a map can THROW information away, so trajectory can beat exit-repeat | **Rejected as stated.** The review's "dead on arrival" verdict conflates entry and exit. The arm stays queued, split into two controls (`trajectory` vs `exit_repeat`, plus an `entry+exit` control) instead of being dropped. |
| Falsification arm: unshared 3-stage unrolled slot stack | Cheapest experiment to separate a DPI ceiling from an optimization (weight-sharing) failure | No file gives it a clean forced-depth readout; an unrolled stack changes width, depth and sharing at once with no arm that isolates which one moved a result | **Not queued.** Rejected as a build, not as an idea: no clean instrument exists for it yet. |
| Olympiad cells "inject noise" (zeroing slot cells improves the model 2.2 nats at offset 0) | The 1024-d bottleneck actively hurts autoregression, not just fails to help it | `lab/experiments/planned/2026-09-04-loop-contribution-arc.md`, Lane 6 (Olympiad math), same number cited by the reviewer | **Unverified claim, marked as a hypothesis.** The number is real; the causal reading ("injects noise") is not established. It is filed as a candidate train/holdout shift, since the Olympiad holdout is known contaminated
([`../bug-fix/2026-09-08-olympiad-holdout-contamination.md`](../bug-fix/2026-09-08-olympiad-holdout-contamination.md)),
and the data on hand cannot separate the two accounts yet. |
| Halt Build B (`loop_reads_tokens`), it costs 44 block-passes/token and abandons "decode cheap" | The token-reading loop training arm should stop | Cost figure matches the design (tokens run through the shared core across the loop's passes); "decode cheap" is TUL's stated intent (`tul-intent-think-once-decode-cheap` in project memory) | **Adopted for the training arm.** Its free evaluation probe on an existing checkpoint stays, since it costs nothing further. |
| Proposal 1: the Thought Register (M looped cells per span, distinct learned seeds, one-to-one write into M prefix cells) | Raises effective rank and gives the loop's attention something relational to iterate on | Configs already staged: `morph/configs/tul_slot_register_m4.yaml`, `tul_slot_register_m4_sameinit.yaml`, `tul_slot_register_m8.yaml` | **Adopted, now the main build.** `m4_sameinit` is the control that isolates capacity (four cells, identical seeds) from distinct seeding. |
| Proposal 2: the ultralight coda (n_coda 1, blocks moved to the prelude) | Forces the coda to depend on the loop's output by removing its own depth | No arm run yet; the design is a clean control on reader headroom | **Adopted as a queued control**, cheap and separable from the register. |
| Build C (memory rescoring at matched inference cost) | Correct framing regardless of the loop's fate | Already the standing instruction (`tul-goal-is-loop-contribution-not-matched-compute` in project memory: CE against controls is context, not a verdict) | **Adopted, unchanged**, it was already the framing before the review. |

## Alternatives considered

**(a) Ignore the review because the source is suspect.** Rejected. Wolfe's own read of
Proposal 1 (the rank number) matched a fact already sitting in the arc ledger and the
drawing-board note; throwing the whole document out would have meant re-deriving that match
later at GPU cost.

**(b) Adopt every point in the review as written.** Rejected. The review's "dead on arrival"
call on the trajectory prefix conflates the Lean theorem's own entry/exit distinction (its
own cited theorem, misapplied), and its falsification arm names no instrument that separates
the effect it is meant to isolate. Taking the review at face value would have dropped a cheap,
already-built arm and queued an ambiguous one.

**(c) Adopt only the points independently checked against a file in this repo, chosen.** Each
row above cites the file that confirms or fails to confirm the claim. Points with no file
backing are marked unverified rather than accepted or rejected outright.

## Acceptance criteria

* `tul_slot_register_m4.yaml`: `val/slot_eff_rank` within a row above 12 (roughly double the
  single-cell ruler's 6.34-7.12), read against `tul_slot_register_m4_sameinit.yaml` on the
  same instrument.
* Any token K3−K6 above 0.002 nats with a paired bootstrap CI clear of zero on `m4` is the
  first result in this lineage that is not a ceiling.
* If `m4` matches `m4_sameinit` within noise, capacity (more cells) was not the lever and
  distinct seeding is the next factor to isolate, not width.
* Ultralight coda (n_coda 1): if token K1−K6 stays at or below 0.005, a bounded reader is not
  the limit either, and the register result stands on its own.

## Risks

* **Over-trusting a model review.** The reviewer's diagnosis (Proposal 1) and its concrete
  recommendation (falsify with an unrolled unshared stack) are treated separately in this
  note precisely because the diagnosis checked out and the recommended experiment did not.
  Future reviews from the same or a similar source get the same per-claim treatment, not a
  blanket accept or reject.
* **The register adds parameters.** M cells at M distinct seeds is more capacity than one
  cell; the `sameinit` control exists so that a rank gain is not credited to the loop when it
  is really credited to extra parameters. Any read of `m4` against the single-cell ruler that
  skips `sameinit` is not a clean result.
* **Effective rank is a correlational instrument.** A higher rank is necessary for the coda to
  have more to read; it is not proof the passes are doing useful iterative work. The K-curve
  and the CI, not the rank number, are what closes or keeps this lane open.
