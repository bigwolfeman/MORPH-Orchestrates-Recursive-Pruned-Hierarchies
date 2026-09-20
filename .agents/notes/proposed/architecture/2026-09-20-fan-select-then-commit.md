# Agent Note: select-then-commit for the fan — every stream gets a reader

Status: proposed

## Problem

`slot-spandec-strict-fan4-epivol` (`lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md`)
is the first fan arm whose diversity term was not gamed: four live streams per slot, rank
2.82 of 4, input-dependent directions, balanced norms. And the coda read one of them. The
softmax mixture concentrated on stream 0 (entropy 0.41 of ln 4, `oracle_pick0` 0.71);
stream 0 alone sits 0.005 above the mixture, streams 1 to 3 sit 0.14 to 0.42 above it; the
oracle over the four beats the mixture by 0.018. Wolfe's reading (2026-09-20): this is not a
closing, it is the thing to work on.

Read again, the mechanism is a reader problem. The coda is trained on ONE thing, the
mixture. A shared reader plus a soft gate is a rich-get-richer loop: the stream the coda
reads best gets more mixture weight, hence more gradient, hence becomes more readable, and
the others get less of each. The `stream_ce_k` instrument then writes a stream alone into a
coda that never saw it, and reads a reader that never trained on that stream, not a stream
that carries nothing (`the-reader-was-the-limit`: the same cell read −4.6 nats to a frozen
reader and +0.18 to an adapted one). The diversity term keeps the streams apart in geometry
but gives no stream a reason to be readable on its own.

## Proposal

`tul.fan_mix: select` (`morph/model/tul_fan.py`, `MORPHTransformer._tul_fan_select`). No
mixture. One stream per slot is written alone.

- **Train.** Before the write, K coda passes under `no_grad`, stream `i` written alone
  into every slot through the same `prefix_project` (the oracle instrument's own table,
  moved to the training step; no activations saved). The per-slot winner is the argmin of
  the span CE after the slot. With probability `fan_select_eps` (0.05) a valid slot writes
  a uniformly random stream instead (`select_winners`): the winner-takes-all guard, since
  a stream that never wins is never read, and never being read is why it never wins. The
  training pass writes the winner alone (`select_streams`, a hard gather: the CE's
  gradient reaches that stream and no other). The gate, the same `d_model → 1` linear the
  softmax arm has, is trained to PREDICT the winner (`select_gate_loss`, CE of its logits
  against the detached winner, weight `fan_select_gate_lambda` 1.0, folded like the
  repulsion as `fan_select_gate_weighted` so train/loss stays the model's CE).
- **Eval.** The gate's argmax stream is written alone; no extra passes. The oracle
  instrument after the coda reports `fan/gate_agree` (argmax == argmin) beside
  `fan/oracle_ce`, `fan/stream_ce_k{i}` and `fan/mixed_ce`, which under `select` IS the
  shipped write (the chosen stream alone).
- **Seam.** The choice is made at the fan's mixture seam, so every reader after it (the
  span decoder, the write) grades the state the coda gets. The selection passes read the
  coda's carrier before token dropout (the winner is chosen on clean inputs) with the slot
  cells' own injections cut as the strict coda cuts them.
- Composes the epivol diversity term unchanged.

## Alternatives considered

- **Winner-takes-all through the mixture (K graded passes, min loss).** Trains every
  stream where it wins but costs K coda passes WITH backward and K saved `[V, d]` CE
  accumulators; the no-grad selection plus one training pass reaches the same per-stream
  signal at one backward.
- **Straight-through hard routing (sample one stream from the gate's softmax).** One pass,
  but the choice is the gate's, not the coda's, so it inherits the rich-get-richer loop it
  is meant to break unless load-balanced; the coda's own span CE is the better teacher for
  which stream to read. Kept as the fallback if the K no-grad passes cost too much.
- **Load-balancing the softmax mixture (an entropy floor on the gate).** Keeps the streams
  in the mixture but a stream can be in the mixture at weight 1/K and still never be read
  ALONE; the instrument that matters writes one stream alone, so the training write should.
- **Train the coda on every stream separately (K CE terms).** Makes every stream readable
  but trains the coda to read all four as equals and gives the gate nothing to select on;
  the question is whether there is anything to select.

## Acceptance criteria

The prereg `lab/experiments/planned/2026-09-20-lxtul-fan4-select.md` freezes them. In
short: every stream within 0.05 of the best (readable); the oracle below the best single
stream by more than 0.022 (the falsifier); the gate's write below the best single stream
by 0.005 with `gate_agree` above 0.40. Tests: `tests/test_tul_fan_select.py`.

## Risks

- The K no-grad coda passes cost step time (an estimate of 1.4x; measured in the smoke and
  read by P-8). The per-row `[L, V]` logit products are the part that scales with vocab.
- The eps guard is a knob; too small and a loser starves, too large and the coda trains on
  noise. 0.05 is the standard relaxed winner-takes-all value and P-6 reads the outcome.
- The selection passes run the coda in train mode (its own dropout draws RNG), so a select
  arm is not RNG-aligned with a softmax arm of the same seed. Documented in the method.
