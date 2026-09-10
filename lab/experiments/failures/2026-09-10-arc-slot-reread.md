# Planned: the reread — the looping slot re-reads its span each pass

Status: failure
Date: 2026-09-10 (frozen before launch, 02:00; the structural arm named in
`2026-09-10-arc-slot-map-levers.md`, given its own file so its predictions are written
before it runs). Arc: `2026-09-04-loop-contribution-arc.md`.

## Question

Inside `_tul_core` the compact sequence holds only slot cells. A pass can mix slots and
apply the MLP to a summary the prelude already wrote; it cannot look at a token. The paid
loop, the one arm on this tree whose loop earns depth (K1−K6 0.120 under norm_match),
re-reads every token state every pass with a refined query. Is the missing per-pass
computation the read? `tul.reread` (commit below) gives the looping slot a cross-attention
read of the FROZEN prelude token states (K/V built once before the loop; scope `causal`:
its own span and every earlier one) at the start of every pass, inside the map the hinge
measures. W_o is zero-init, so step 0 is the unpack forward bit for bit
(`tests/test_tul_reread.py`: mask exactness, zero-init identity, liveness, causality with
a fixture-sensitivity check).

## Hypothesis

H-rr-1: the read is the missing computation; with it the slot state moves with depth and
the tokens read the difference (K3−K6 above zero with the CI above 0). H-rr-1′: the read
helps the first pass only (the slot gets a second look at its span) and the curve stays
flat past pass 2: the map converges as fast as before, from a better first state.
H-rr-0: the read changes nothing the coda uses; the block is not what the loop sees.

## Method

`tul_slot_unpack_reread` = `tul_slot_unpack_norm_match` + `tul.reread: true`
(`reread_heads` 8, `reread_scope` causal), 4.2M new parameters (302.4M total), on the same
recipe and runner as the levers panel (`arc/run_slotloop3.sh`, arm file line pinned to the
commit that carries the code), the same readout (sweeps at 2,500 and 5,000 with the token
and forecast curves, worth profile, slot-state probe at 5,000). Queued behind the three
knob arms and ahead of the deferred controls, unconditionally: it is the structural lever
and the GPU is free overnight; the levers panel's binding decides what follows it, not
whether it runs. Scored against the unpack arm's own readout. Verified before launch: 128
CPU contract tests pass (`pytest tests/test_tul_*.py tests/test_slot_*.py
tests/test_checkpoint_compat.py tests/test_onset_capture.py`: 107 + 21), one real-size CPU
training step at seq 256 runs with a non-zero gradient on W_q and W_o. Not verified: the
GPU path (the runner's smoke is the first), the rate (the read is S×L attention per pass,
small; the rate rule guards), the hinge's reading with the read inside the map.

## Predictions (frozen)

- **P-rr-a (survival).** HEALTHY to 5,000: **70 %**. Rate above the paid loop's 8,086
  tok/s at step 200: **85 %**. Peak under 16 GB: **75 %**.
- **P-rr-b (movement).** Slot-state movement per pass (probe, passes 2–6 mean) above 5 %:
  **50 %** (unpack arm: 1–3 % on the no-MUX sibling).
- **P-rr-c (the curve).** Tokens K1−K6 above 0.005 with the CI above 0: **45 %**; K3−K6
  above 0.002 with the CI above 0: **30 %**. Forecast K1−K6 above 0.010: **45 %**; forecast
  K3−K6 above 0.003: **30 %**.
- **P-rr-d (the price).** Val CE at 5,000 within 0.05 of the unpack arm's 4.5791: **55 %**;
  better by more than 0.05: **30 %**; worse by more than 0.05: **15 %**.
- **P-rr-e (worth).** Zeroing z costs more than the unpack arm's 0.811 at offset 0:
  **55 %**.

## Binding

- P-rr-c TRUE on K3−K6 ⇒ the read is the missing computation; the follow-ups are the
  scope (`span` vs `causal`) and the heads, and the slot loop's depth story is re-opened.
- P-rr-c TRUE on K1−K6 only ⇒ H-rr-1′: one extra look helps, iteration does not; the
  next read is a per-pass instrument on WHAT the read attends (entropy of the read's
  attention over passes).
- P-rr-c FALSE with movement above 5 % ⇒ the state moves and the coda does not use the
  movement: the unpack (W_bcast at offset-indexed linears) is the next suspect.
- P-rr-c FALSE with movement under 5 % ⇒ H-rr-0; the slot loop's per-pass computation is
  not rescued by what it sees, and the arc goes back to the map (the plain ruler's
  anatomy against the slot anatomy).
- NO run beyond 5,000 steps from this experiment.

## Results

Filed 2026-09-10 06:10. `slot-unpack-reread`: HEALTHY, 13,136 tok/s at step 200, peak
12.96 GB (smoke), val 4.5707 at 5,000. Tokens K1−K6 +0.0003 [+0.0001, +0.0006], K3−K6
−0.0000; forecast K1−K6 +0.0039 [+0.0030, +0.0050], K3−K6 +0.0001; CE@6 4.4955 (unpack
4.4971); worth zero 0.817, shuffle 1.577. The read is trained: W_o's norm 0 → 5.2 at 2,500
→ 9.3 at 5,000, and it adds a constant 5.0 % of the state on every pass (anatomy spy:
0.056, 0.048, 0.050, … 0.050); the core blocks stay at 1.4–5.5 % MLP out/in. Full-carrier
movement 13 % then 5.5/2.7/1.7/1.3/1.2/1.1/1.1 % per pass (unpack: 8.7 % then 3.8/2.2/…).

## Verdict

P-rr-a TRUE (healthy, rate above the bar, peak under 16 GB). P-rr-b FALSE (movement under
5 % from pass 3). P-rr-c FALSE on every clause. P-rr-d TRUE (within 0.05, better by
0.008). P-rr-e FALSE (0.817 vs 0.811, within noise). H-rr-0: the read is used and changes
nothing the coda reads across depths.

## Updated hypothesis

The read gives every pass the same 5 % term because the query, the slot state, barely
changes between passes; the read cannot start an iteration that the blocks do not
continue. What the loop sees is not the block. See the levers panel's updated hypothesis.
