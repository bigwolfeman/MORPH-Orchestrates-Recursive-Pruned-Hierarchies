# Agent Note: LXTUL CE test queue, closing the gap to plain

Status: proposed

## Problem

LXTUL ([decision record](../../implemented/architecture/2026-10-04-lxtul-primary-candidate.md))
trails plain at the same step by +0.272 nats at 5k and +0.356 at 10k. The best account so
far:

- Most of the gap is cross-span information that never reaches the coda. With no slots
  the strict mask costs 0.40 nats at 5k. LXTUL's cells win back about 0.13 to 0.18.
- The latent head costs about 0.02 at most (it is detached). The short token path
  (8 blocks) leaves little of the 5k gap unexplained.
- Cells' worth plus gap is 0.448 at 5k and 0.589 at 10k, and the cells hold a constant
  39-40 % of it. If that sum tracks the budget, the gap grows because plain learns to use
  the past better with training, not because LXTUL gets worse.

Wolfe (2026-10-04): "You may be reading too much in to how these act over longer
training, but we should be diligent and test and probe for it." The long-training account
rests on one direct budget measure (5k) and a sum of two instruments at 10k. Treat it as a
hypothesis.

## Proposal

The queue. Each item gets its own prereg before it runs. Status is kept here.

| # | test | what it changes | why | cost | status |
| --- | --- | --- | --- | --- | --- |
| 1 | **More cells** | `fan_k` / `prefix_k` 4 -> 8 | is the cells' share limited by channel width? | loop 2x cells; ~17 % slower at 5k | RUNNING 2026-10-04 (`lxtul_fan8`, prereg `2026-10-04-lxtul-eight-cells`) |
| 2 | **Wider writes** | each of the 4 fan cells is written into m = 2 coda positions through different projections (`tul.prefix_per_cell`, copy 0 identity, copies >= 1 random orthogonal); the loop is unchanged | isolates more coda POSITIONS from more loop STREAMS against item 1 at the same 8 positions per slot. Correction 2026-10-05: the write already passes the full 4-stream HC carrier, so a per-HC-stream write adds no information; width here means addressable positions | coda reads 2x the prefix positions; loop unchanged | FRONT OF THE QUEUE (Wolfe 2026-10-05): building (`lxtul_fan4x2`), trains right after the fan8 readouts |
| 3 | **Stronger reader** | coda 4 -> 6 blocks (or extra cross-attention to the cells) | 2026-09-18: the same cell went from -4.6 to +0.18 nats when only the reader trained; the reader may cap what the cells can deliver | +2 blocks on every token | to design |
| 4 | **Raw local window for the coda** (was: previous-span window) | coda tokens read the raw tokens of the last W positions across span boundaries (W up to plain's `window_size` 256); the loop carries everything farther back | **TOP PRIORITY after item 6**: 65 % of the 10k gap is bigram repeats 33-256 tokens back, inside plain's exact window; past 256 the gap on repeats is +0.17 | coda attention over W more keys; mask code; decode KV for W positions only | to design; risk: every arm whose coda read raw tokens across spans had K1-K6 <= 0.002, so try W 64 and 256 |
| 5 | **Shorter spans** | `span_cap` 32 -> 16 | twice the slots, so more channel per token | 2x slots in the loop | low prior: prior art and our runs say shorter spans hurt; test for completeness |
| 6 | **Exact-recall probe** (offline) | none; buckets LXTUL 10k vs plain 10k per-token CE by whether the target bigram already appeared more than 32 tokens back in the row | is the missing 60 % mostly copying that attention does and a compressed cell cannot? If yes, items 2 and 4 are the levers; if the gap is flat across buckets, cell content is | CPU only | DONE 2026-10-04, success ([filing](../../../../lab/experiments/successes/2026-10-04-exact-recall-gap-probe.md)): bigram repeats 33-256 back are 15 % of tokens, 65 % of the gap; the gap drops to +0.17 past 256 |
| 7 | **Budget curve to 10k** | train the span-mask budget pair (`budget_web_full`, `budget_web_span`) to 10k and read the budget at 2.5k, 5k, 7.5k and 10k | the direct test of the long-training account: does the budget itself grow from 0.40? | 2 x 10k steps (the 5k checkpoints were purged) | to queue |
| 8 | **LXTUL to 20k** | resume `lxtul` 10k -> 20k, against a plain 20k at the same recipe | does the gap keep widening, hold or close (the paid loop's closed 0.132 -> 0.012 by 20k)? | 10k + plain 10k steps | to queue |
| 9 | **Multi-head span pool** (pack more INTO the cell) | the register's seed pool is ONE single-head softmax over the span's prelude states per cell (`TULSlotRegister`): a weighted average. Make it H heads (e.g. 16), each with its own query, concatenated into the cell | the only way span tokens enter the loop is that pool, once at entry; a weighted average blurs exact tokens, and copying needs them | small: H queries per cell | proposed 2026-10-05 (Wolfe: "our issue is the amount of data contained within a cell") |
| 10 | **Own-span reconstruction** (make the cell KEEP it) | an auxiliary head decodes the slot's OWN span from the written cell (teacher-forced, like `spandec`, which decodes the NEXT span) | copying later needs the current span's exact tokens in the cell; the shipped targets (coda CE through the winner, next-span decoder, latent rank) do not ask for them | one small decoder per slot at train only | proposed 2026-10-05 |
| 11 | **Wider read-out, m = 4 / 8** (get it OUT of the cell) | item 2 at larger m | if item 2 helps, find where positions stop paying | coda reads m positions per slot | after item 2 |

Readouts for every training arm: the depth sweep (K1-K6), the gap to plain at the SAME
step, the cells' zero-ablation worth and its share, the mean-ablation of the shared
direction, the exploration ledger, the repetition eval, tok/s. Loop contribution is
scored first: an arm that closes the gap and kills K1-K6 is not a win.

## Alternatives considered

- **Accept the gap and score at matched decode cost.** Kept as a parallel track (the
  cached-decode port of the fan), not a replacement: Wolfe asked for the CE levers.
- **Relax strict geometry fully** (coda reads all earlier tokens). Measured: every
  coda-reach-all arm earned K1-K6 <= 0.0016. Item 4 is the bounded version.
- **Lower the latent head's weight or drop it.** Its cost is about 0.02 nats at most and
  it is what makes the loop's earning directional. Not queued.

## Acceptance criteria

- Each item is filed as an experiment (success or failure) or struck here with a reason.
- This table's status column is kept current in the same change as each filing.

## Risks

- Items 1-3 and 5 raise cost per step. Each is priced in tok/s beside its CE.
- The share instrument adds two readings (zero-ablation worth and the gap) that are not
  the same kind of measure. Item 7 is the direct check.
- n = 1 per arm. LXTUL's seed spread on the gap is about 0.01 at 5k; a difference smaller
  than that needs a seed twin before it counts.
