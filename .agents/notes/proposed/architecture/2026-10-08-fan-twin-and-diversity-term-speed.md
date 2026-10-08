# Agent Note: the fan's EMA twin and diversity terms on the FAST step

Status: proposed

## Problem

On the FAST graph-replayed step of `lxtul_pointer` (390 ms, `perf/flame/LEDGER.md`), the fan's
training-only terms cost 30 ms for 0.8 TFLOP. The EMA twin prelude took 19.2 ms at 37 TFLOPS
(the online prelude forward runs the same blocks in 10.4 ms). 4.3 ms of the twin was its own
read of the token, bigram and value-embedding tables, and each read re-quantises the whole
table (`embed_quant`). The rest is eager blocks. The epi and vol diversity terms took 7.2 ms:
fp64 products, one LU per (pass, stream) and a Householder ridge map. The latent-selected
loop's per-pass teacher pick ran its head as fp32 SIMT GEMMs.

## Proposal

One exact change replaces the old path. Four keys default off:

- Exact. The twin reuses the live front's table reads (`_tul_front_lookups`, handed through
  `_tul_front(lookups=)` and `_tul_fan_target(lookups=)`). The EMA update is one
  `torch._foreach_lerp_`. A 45-step deterministic FAST gate is bit-identical to `014c5aad`.
- `model.fan_twin_compile`: the twin's prelude blocks go through the trainer's block compile
  (`FanTargetFront.compile_blocks_`). The state_dict names stay the same. Not exact.
- `model.fan_div_fast`: the epi and vol terms are batched over every (pass, stream) in fp32.
  A Cholesky gives the log-det and the ridge map uses the normal equations
  (`tul_fan.py`, `*_fast`). Not exact. The fp64 reference test puts the error 2 to 3 orders
  under the change that bf16 rounding of the inputs makes.
- `model.fan_lsel_pick_bf16`: the per-pass pick runs its head GEMMs in bf16
  (`FanLatentHead.forward_bf16`). Not exact. Only near ties can flip.
- `model.fan_target_online`: the target is the online prelude's pooling under stop-gradient,
  so the twin forward is skipped. This changes the objective. It is refused unless the arm
  is the latent-selected loop with `fan_lsel_train_follow: router` (the analysed case,
  `perf/lean/PROOFS.md` rank 8).

## Alternatives considered

- A token-only twin. The strict span prelude makes token outputs exactly independent of slot
  rows (this was measured bit for bit, and `perf/lean` Lemma 4 holds under
  `tg_strict_prelude: span` only). The packer fills `tokens + 4 * slots = 1280`, so a row
  holds 1024 to about 1137 tokens. A fixed-shape compaction would save at most 11 % of the
  twin's block work, about 1 ms, and would move RoPE angles away from the exact bits. Not built.
- Keep fp64 for the epi term. The fp64 rate on the 5090 is 1/64 of the fp32 rate, and the
  reference test shows fp64 buys nothing that bf16 inputs do not already erase.

## Acceptance criteria

- Every key leaves CE@6 and K1-K6 within 5 % of the winner over a paired training run.
  A fixed-weight sweep cannot see these keys, because they change only training-time terms.
- tests/test_tul_fan_twin_speed.py and tests/test_tul_fan_div_fast.py pass, and each fails
  under its listed sabotage.

## Risks

- `fan_target_online` drops the EMA (BYOL without a momentum encoder). Only the `L_enc`
  floor guards against collapse. Do not adopt it without a training run.
- `fan_lsel_pick_bf16` flips near-tie picks. Under router-follow they only change the router's
  CE target.
