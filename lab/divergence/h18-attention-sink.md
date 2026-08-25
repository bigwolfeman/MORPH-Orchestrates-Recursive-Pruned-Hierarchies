# H18 — a POSITIONAL attention sink in the core

Working document. It is the plan, the running log, and the place the numbers land.
Updated as each phase closes. The ledger entry it resolves is `takeover-campaign.md` H18.

Status: **CLOSED 2026-08-25 — H18 NOT SUPPORTED.** The follow-on, H24, is screened and
awaiting its arm; see the bottom.
Opened 2026-08-25.

## Why this hypothesis

From `docs/experiments/failures/2026-08-24-tul-takeover-cure.md`:

- H4 CONFIRMED: the loop's effect on slot-state effective rank FLIPS SIGN between step
  1750 and 1800. Healthy rungs diversify (rank ratio 1.23–1.48), sick rungs do not
  (1.01, 0.67, 0.87). It is a FORWARD quantity and it is the earliest indicator in the
  whole campaign.
- The cotangent sits on a stable sink: the same top-3 slots at every one of the 6 core
  blocks (agreement 1.0 at five of six rungs), top slot's share rising 0.18 -> 0.54.
- H17 REFUTED: the LEARNABLE sink parameter never engages (`sink_logits` 0.0036 ->
  0.0053, sigmoid 0.5009 -> 0.5013).

So a sink exists in the BACKWARD, the learnable sink is not it, and the FORWARD attention
has never been measured. H18 is that measurement.

Every weight-spectrum cure failed (H2, H3, and four more in `morph-takeover-is-state-collapse`).
That is consistent with a forward-state disease. H18 is the last cheap forward lead.

## The hypothesis, sharpened

- **H18a** the forward attention mass over slot positions concentrates, and the
  concentration GROWS with loop iteration `t`.
- **H18b** the concentration is POSITIONAL — the same slot index, not the same content.
- **H18c** the concentration flips sign between rungs 1750 and 1800, with the state-rank
  flip.

## Ladder and build

`checkpoints/morph/onset-capture` — `tul_a1`, seed 0, batch 6, `alpha_cap` 3.5,
`deterministic=true`, `use_kernels=false`. `ROLL_step_{1625..1850}` at 25-step spacing plus
`TAKEOVER_step_1866`. Replay-verified: 0 of 111 series differ on resume from 1750.

Model built through `lab/divergence/_build.py` (QAT before load; skipping it silently
drops every MLP tensor).

## Phase 0 — geometry audit

READ FROM SOURCE, NOT YET RUN. The core runs on S = 64 slot positions
(`tul.max_slots: 64`, ~52 valid). Against `base.yaml` (`window_size: 256`,
`csa_compress_ratio: 8`, `hca_compress_ratio: 256`, `top_k: 256`, `d_model 1024`,
`n_heads 8`, `compression 2` -> `d_head 64`, `n_core 6`):

| core layer | branch | expected at S=64 |
|---|---|---|
| CSA (even, 3 of 6) | window | dense causal over all 64 slots (`window_size` 256 > 64) |
| CSA | compressed | `n_blocks = 64//8 = 8`, `tk = min(256,8) = 8` -> every block selected, NO sparsity |
| HCA (odd, 3 of 6) | window | dense causal over all 64 slots |
| HCA | compressed | `n_blocks = 64//256 = 0` -> `GatedPoolCompressor` `n_blocks == 0` branch returns `[B, 0, c]` -> branch EMPTY |

If that holds, almost all positional mixing in the core runs through ONE dense causal
window, and a sink must live there.

Phase 0 measures, it does not assume:

- P0.1 `n_blocks` per core layer at S=64
- P0.2 is `out_comp` identically zero on the 3 HCA core layers?
- P0.3 does the CSA `top_idx` cover all 8 blocks for every query?
- P0.4 does the window mask reach all 64 slots?

### Phase 0 results — MEASURED 2026-08-25

    PYTHONPATH=$PWD python lab/divergence/attn_sink_probe.py --geometry --token-path

Model built at random init (geometry does not depend on the weights), `tul_a1`,
batch 6, `use_kernels=false`, 8 loop iterations, `L_total = 1152`, `max_slots = 64`.

SLOT path (S = 64):

| blk | kind | m | n_blocks | window | \|out_comp\| | \|out_win\| | g_comp | g_win | self-test |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | CSA | 8 | 8 | 256 | 275.13 | 226.84 | 0.502 | 0.490 | 6.6e-3 |
| 1 | HCA | 256 | **0** | 256 | **0.0000** | 220.40 | 0.501 | 0.500 | 6.8e-3 |
| 2 | CSA | 8 | 8 | 256 | 276.97 | 223.97 | 0.497 | 0.497 | 7.4e-3 |
| 3 | HCA | 256 | **0** | 256 | **0.0000** | 229.24 | 0.493 | 0.503 | 7.4e-3 |
| 4 | CSA | 8 | 8 | 256 | 271.60 | 222.01 | 0.494 | 0.504 | 7.4e-3 |
| 5 | HCA | 256 | **0** | 256 | **0.0000** | 225.90 | 0.503 | 0.497 | 7.6e-3 |

TOKEN path (S = 1152), same weights: HCA gets `n_blocks = 4` and `|out_comp|` ~ 1030,
so the branch is alive there.

**P0.1 held.** `n_blocks` is 8 for CSA and **0** for HCA on the slot path.

**P0.2 held, and it is the finding.** `|out_comp|` is EXACTLY 0.0000 on the three HCA
core blocks. `hca_compress_ratio: 256` against 64 slot positions puts
`GatedPoolCompressor` in its `n_blocks == 0` branch, which returns `[B, 0, c]`, so the
compressed attention has nothing to attend to. **Three of the six core blocks run with
half of their attention output identically zero**, and the gate still spends
`g_comp ~ 0.50` of its mixture on that zero tensor. Those blocks deliver about half the
attention magnitude they were built for, and all of their positional mixing comes from
the window branch alone. This is a slot-path-only defect: the token path is unaffected.

**P0.3 held, and it is BIGGER than the slot path.** `tk == n_blocks` and
`distinct_top_idx == n_blocks` at EVERY recorded call: 8 of 8 on the slot path and
**144 of 144 on the token path**. `top_k: 256` in `base.yaml` exceeds `n_blocks` at
`seq_len 1024` (1152 // 8 = 144), so CSA's sparse selection NEVER fires anywhere in this
config. CSA is running as dense pooled attention. This is independent of TUL.

**P0.4 held.** `window_size` 256 > 64, so the window branch is dense causal over all
slots. Combined with XSA (`dist != 0`, self excluded) this makes query `i` attend keys
`0..i-1` and nothing else. Query 0 has NO valid key.

**Self-test.** Max relative error between the recomputed `A @ v` and the shipped
`out_win` is 7.6e-3 across all blocks and both paths, with zero non-finite outputs. That
is the bf16 noise floor (bf16 eps is 7.8e-3), the probe runs under the same
`autocast(bfloat16)` as training, so the recomputed weights ARE the shipped path's.

### What Phase 0 changes about H18

On the slot path essentially ALL positional mixing runs through one dense causal window
with the self token excluded. Under that mask, query `i` sees keys `0..i-1`, so early
slots receive mass from many queries by CONSTRUCTION — a raw "slot 0 is the top key"
reading would be an artifact, not a finding. The probe must therefore compare mass
concentration ACROSS rungs and ACROSS loop iterations, never against a uniform baseline.

The CSA compressed branch is dense over 8 pooled blocks and carries a slightly LARGER
norm than the window (275 vs 227), so it is half the signal and Phase 1 measures it too.

## Phase 1 — the probe

`lab/divergence/attn_sink_probe.py`.

Attention weights are NEVER materialized: `fused_window_attention` and `_window_fallback`
both go straight to the output, and the CSA/HCA compressed kernels are flash online-softmax
by design. So the probe wraps `_CCABase._window_attn`, recomputes
`A = softmax(q k^T * scale + mask + sink)` explicitly, and SELF-TESTS that `A @ v` matches
the shipped `out_win`. A probe measuring a tensor the model never uses is worthless, so the
self-test is a gate, not a comment. (Same discipline as `subspace_probe.py`'s
"every curve must reach exactly 1.0 at k = in_dim".)

Recorded per (rung, core block, loop iteration `t`), meaned over heads and batch:

| metric | reads |
|---|---|
| `mass_j` = mean over queries of `A[:,:,j]` | mass received by key slot `j` |
| top-1 share, top-3 share | the sink's size |
| participation ratio `(sum m)^2 / sum m^2` | 1 = one sink, n = uniform |
| argmax slot index; top-3 set agreement across blocks and across `t` | is it the SAME slot |
| per-slot carrier norm | separates "one slot is big" from "everyone reads one slot" |

Plus a CONTENT-SHUFFLE run: same slot count, different text. Sink stays on the same INDEX
-> positional. Sink moves -> content-driven, and H18 as written is wrong.

## Phase 2 — pre-registration

`docs/experiments/planned/2026-08-25-h18-positional-attention-sink.md`, committed BEFORE
the run. Thresholds live in `lab/divergence/score_h18.py`, also committed first, so they
cannot be fitted to the data.

Predictions are written in that file. Do not restate them here.

## Phase 3 — run and score — DONE 2026-08-25

    PYTHONPATH=$PWD python lab/divergence/attn_sink_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --out attn_sink.json
    python lab/divergence/score_h18.py --json attn_sink.json

Under 60 s for all 11 rungs x 2 batches. Validity gate PASSED. **1 of 5 predictions
held**, and the one that held (P1) does not discriminate, because the healthy rungs
concentrate at 4 of 6 blocks too.

Full numbers and the verdict:
[`docs/experiments/failures/2026-08-25-h18-positional-attention-sink.md`](../../docs/experiments/failures/2026-08-25-h18-positional-attention-sink.md).

The short version: **the core's forward attention is DIFFUSE.** Window participation ratio
0.59-0.68 of 57 valid slots; the top key holds 6.4-8.4 % of the mass; `argmax` is slot 0
or 1 at every rung, which is the causal+XSA artifact Phase 0 predicted, and the rows of a
batch agree on it only 0.47-0.81 of the time. A sink is `pr -> 1` and `top1 -> 1`. This is
not within an order of magnitude of one.

The one real signal points the right way and is about half the pre-registered size: the
loop compresses attention entropy by -2 % at rung 1625 and by -13 % at 1800 and 1850, then
recovers to -5 % at 1866 — the same shape and the same non-monotone tail as H4's state
rank. A weak correlate inside a diffuse regime.

The pre-registered refuter did NOT fire (44 of 48 cells within 10 %, it needed 48), so the
honest verdict is NOT SUPPORTED rather than REFUTED. The campaign's own next-step table
said "if attention is diffuse, the last cheap forward hypothesis dies". It is diffuse.

## What this cost and what it bought

About three hours, no training, ~60 s of GPU. It killed the campaign's top open lead and
it turned up H24 on the way, which is a better lead than the one it killed.

## The follow-on: H24

**The HCA compressed branch is identically zero on the slot path, and that is a measured
architectural difference between the SICK arm and the HEALTHY one.**

A1 loops the core over 64 slots; `hca_compress_ratio: 256` gives `n_blocks = 0`, so core
blocks 1, 3 and 5 output exactly 0.0000 from their compressed branch while the gate still
spends `g_comp ~ 0.50` on it. A0 loops over 1152 token positions; the same weights give
`n_blocks = 4` and `|out_comp| ~ 1030`. A1 diverges. A0 does not.

Nothing here shows the dead branch CAUSES the takeover. It is a hypothesis this run
generated, not one it tested.

### The no-training screen — RUN 2026-08-25, and it says "lever, not cure"

[`docs/experiments/failures/2026-08-25-h24-hca-branch-screen.md`](../../docs/experiments/failures/2026-08-25-h24-hca-branch-screen.md).
Instrument: `lab/divergence/h24_screen.py`.

The first attempt failed to load: the plan claimed `m` is in no weight shape, and that was
WRONG — `GatedPoolCompressor` carries `B_a` of shape `[m, c]`. Method Amendment 1 records
it; `B_a` is sliced to 16 rows and the surgery is scoped to the CORE blocks only.

Validity gate passed cleanly: the control reproduces the published H4 unit-rank ratios to
**0.004**, and iteration-0 rank is identical between the arms to **0.000e+00**, so the
surgery touches the loop and nothing upstream of it.

`ratio = eff_rank_unit(last) / eff_rank_unit(first)`, above 1 = the loop diversifies:

| step | class | control | revived | delta |
|---:|---|---:|---:|---:|
| 1625 | HEALTHY | 1.408 | 1.498 | +0.090 |
| 1700 | HEALTHY | 1.277 | 1.445 | +0.168 |
| 1750 | HEALTHY | 1.469 | 1.584 | +0.115 |
| 1775 | HEALTHY | 1.009 | 1.075 | +0.066 |
| 1800 | SICK | 0.923 | **1.146** | +0.224 |
| 1825 | ambig | 0.954 | **1.144** | +0.189 |
| 1850 | SICK | 0.714 | 0.780 | +0.066 |
| 1866 | SICK | 1.059 | 1.054 | −0.006 |

**1 of 4 predictions held.** The deciding number:

    mean delta over HEALTHY rungs = +0.0939
    mean delta over SICK    rungs = +0.0947

Identical. The revived branch lifts the loop's rank ratio by ~+0.09 EVERYWHERE. It is a
uniform capacity effect, not a targeted repair. The two sign crossings at 1800 and 1825 are
that uniform shift crossing a threshold those rungs happen to sit near; rung 1850 gets the
same lift and stays at 0.780.

Two of the three substantive predictions keyed on rung 1850 alone, called "the cleanest
sick rung" because the campaign's two rank measures agreed there. It turned out to be the
hardest one. A calibration error, recorded rather than corrected away.

### What is left to do on H24

**The arm.** `hca_compress_ratio: 16` from step 0, paired seeds against a control, ~5 h per
arm. The screen says to expect a LEVER of the H5 `per_slot_embed` class — buys time, does
not cure — and to size the arm for time-to-failure and CE rather than for a rescue. It also
fixes a real defect whatever it does to the divergence.

Confound to declare in that arm: `B_a` shrinks from `[256, 64]` to `[16, 64]`, so the arm
has 15 360 FEWER parameters per HCA block. It is not iso-parameter, in the direction that
makes an improvement harder to explain away as capacity.

Second Phase 0 finding, separate and smaller: **CSA's sparse selection never fires on the
short schedule.** `top_k: 256` exceeds `n_blocks` at `seq_len: 1024` (144 blocks), so
`tk == n_blocks` and CSA runs as dense pooled attention. At the deploy `seq_len: 4096`
there are 512 blocks and selection does fire, so this is a short-schedule artifact and the
deploy recipe is unaffected. It still means every TUL arm measured to date ran a CSA that
was never sparse.

## Declared not verified

- fused vs eager window paths are not compared; the probe runs eager, matching how the
  ladder was produced (`model.use_kernels=false`)
- the ladder is seed 0 only, so nothing here is shown to generalize
- H24 is untested

## Log

- 2026-08-25 — opened. Phase 0 pending. Geometry table above is READ, not measured.
- 2026-08-25 — Phase 0 RUN and CLOSED. P0.1-P0.4 all held. Two defects found that were
  not part of H18: the HCA compressed branch is identically zero on the slot path, and
  CSA's sparse selection never fires on the short schedule.
- 2026-08-25 — probe, pre-registration and scorer committed (21b1753) BEFORE any ladder
  run. Smoke-tested on the random-init model only.
- 2026-08-25 — ladder run and scored. 1/5 held, refuter did not fire, H18 NOT SUPPORTED.
  Ledger updated: H18 closed, H24 opened as the new top lead.
- 2026-08-25 — H24 no-training screen pre-registered, corrected (`B_a` IS `[m, c]`), run
  and scored. Validity gate exact. 1/4 held. The lift is UNIFORM (+0.0939 healthy vs
  +0.0947 sick), so the defect is confirmed and the mechanism is not. Expect a lever.
- 2026-08-25 — 11 contract tests added for the probe, all four sabotage passes caught by
  the intended test; one strict-xfail added that guards the dead-branch defect itself.
