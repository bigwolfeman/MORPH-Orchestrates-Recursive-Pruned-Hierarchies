# Planned: where does the information go in snap? A stage-by-stage data-flow probe

Status: planned

Date: 2026-10-05 13:27 CDT, before the probe is built. Wolfe (2026-10-05): "the loop should be driving almost
all the contribution I feel like. These small contributions are so bizarre to me. I feel
like we are missing something that will be obvious when looking at the data moving through
the model."

## Question

Snap (`lxtul_snap` 5k) and LXTUL (5k seed 1) earn K1-K6 of +0.012 / +0.022 and their cells
are worth about 0.18 nats, against a cross-span budget of 0.40+. Follow one span's
information through every stage and find where it is lost, ignored or bypassed.

## Stages and readings (snap and LXTUL, same rows)

1. **Seed pool**: each cell's attention over its span's prelude tokens: entropy, the share on
   the boundary token and on the span's first token, the token types it picks.
2. **Loop, per pass**: relative change ||h_t - h_(t-1)|| / ||h_t||; cosine to pass 1; router
   winner switches between passes; cell spread before each reset.
3. **Loop, cross-slot reads**: per core layer, the attention mass a slot's cells put on
   earlier slots' cells versus their own slot's cells.
4. **Write**: norm of the winner's written vector and of the pseudo tokens against the coda's
   token-position states; the learned gate g; pseudo vertex mass and which tokens are picked
   (and whether they are copied later).
5. **Coda reads**: per coda layer and head, the attention mass a token puts on slot positions
   (winner vs pseudo vs zero positions) against its own span's tokens, by slot distance; and
   the value-weighted output norm from slot positions vs tokens.
6. **Depth at the read**: the same coda readings at forced depth 1 vs 6.
7. **Worth split**: zero the winner only, the pseudo tokens only, both; per recall bucket.

## Predictions (mine, orchestrator)

- **D-1** coda attention mass on ALL slot positions, averaged over tokens, layers, heads: under
  10 % (the coda mostly reads its own span). 70 %.
- **D-2** the seed pools put more than 30 % of their mass on one position (the boundary token
  or the span's first token) in most cells: the pool is near a sink. 45 %.
- **D-3** per-pass relative change after pass 2 under 5 %: the loop is near its fixed point
  after two passes, so K1-K6 can only come from passes 1-2. 60 %.
- **D-4** the loop's cells put under 20 % of their attention on earlier slots: the loop mostly
  re-reads its own span's seed, so it adds little cross-span computation. 50 %.
- **D-5** snap's pseudo tokens get more coda attention mass than its winner position. 55 %.

No pass rule: this is a diagnostic. It is filed as success if the readings identify a stage
that loses more than half of what enters it, and as a failure if nothing stands out.
