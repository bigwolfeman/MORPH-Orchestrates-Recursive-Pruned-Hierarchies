# LXTUL credit panel: what each credit path is worth

Status: failure

## Question

The credit matrix (2026-10-10) says the loop's weights learn mostly from the span decoder (0.69,
through the mean of 4 cells), a fifth from the token CE (through the winner cell), and nothing from
the next-span latent z, which only selects. The prelude learns 91 % from the token CE and only
6.5 % of that through the loop. Wolfe is not confident this is the right credit structure. Which
parts carry the result? Each arm changes one credit path on `lxtul_pointer_fast` at 5k.

Arms (the span-decoder winner read has its own file,
`2026-10-10-spandec-reads-winner.md`):

- **one-cell loop** (`tul_onecell_pointer_fast`, already on master): no fan, no selection, no z.
  What do the 4 cells and the selection buy at all?
- **random selection** (`lxtul_pointer_fast_randpick`, `tul.fan_lsel_pick: random`): the same 4
  cells and per-pass reset, but the followed cell is uniform random at train and eval. The router
  still trains on z as an instrument. What is the z-trained selection worth?
- **prelude serves the loop** (`lxtul_pointer_fast_preldet`, `tul.coda_prelude_detach: true`):
  the coda reads the same prelude states, but the token CE does not train the prelude through the
  coda's token path. The prelude then learns only through the loop, the span decoder and the latent
  floor. Does a loop-only prelude raise the loop's contribution, and what does it cost in CE?

## Hypothesis

The selection and the 4 cells carry part of the result: both the one-cell loop and random
selection lose CE and K1-K6. The prelude arm trades CE for loop contribution: the coda loses
features tuned for its tokens, but the loop gets an encoder of its own.

## Predictions

References, `lxtul_pointer_fast` from init at 5k, seed 1, 480 rows: CE@6 3.9248 / 3.9229 (mean
3.9239), K1-K6 +0.0104 / +0.0091 (mean +0.0098), final val 3.9161 / 3.9129. Spread between
independent 5k runs: about 0.006 in K1-K6, 0.002 in CE@6. One run per arm; a K1-K6 difference
under 0.006 is inside the spread and reads as no effect.

| Arm | CE@6 best guess | CE@6 effect threshold | K1-K6 best guess |
|---|---|---|---|
| one-cell loop | 3.950 | worse if >= 3.930 | +0.008 |
| random selection | 3.931 | worse if >= 3.930 | +0.007 |
| prelude serves the loop | 3.975 | worse if >= 3.930 | +0.013 |

Readings beside these, from each run's final val line: `val/plan_worth_zero` and
`val/plan_worth_shuffle` (what the loop's write is worth), `fan/oracle_gap`, `fan/router_pick_regret`
against `fan/rand_pick_regret`, and the trainer's late tok/s.

What each result changes:
- random selection no worse than the references (CE@6 below 3.930 and K1-K6 within the spread): the
  z machinery (EMA twin, g head, router, teacher) is not earning its complexity, and the next step
  is to remove it.
- random selection worse: the selection is real credit, and the next arms improve the selector's
  feedback (for example, a teacher built from the text outcome).
- one-cell loop within the spread on both readings: the fan is not earning its cost either.
- prelude serves the loop with K1-K6 above +0.016: the geometry objection is measured; the next arm
  gives the coda its own front layers.

## Method

Each arm: `--config-name <arm config>` from a frozen `git archive` of a commit that contains this
file and the arm's knob, 5000 steps, seed 1, WANDB offline, run one at a time under `gpu.lock` by
`/home/wolfe/morph-scratch/abq/runner.sh` (queue file `queue.txt`, log `log.txt`). Readout per run:
`core_depth_sweep.py --ckpt <name>=<config>=<step_5000> --depths 1,6 --rows 480 --batch 3`, the
references' rows and command.

## Results

Run 2026-10-10, 5000 steps, seed 1, fast recipe, 480 rows, one run per arm. Artifacts:
`lab/experiments/results/2026-10-10-lxtul-credit-panel/` (`sweep-*.json|log`, `train-*-tail.txt`,
`queue-log.txt`, `extrapolate_val_gap.py|txt`). Later rounds that reuse these arms:
`results/2026-10-10-credit-panel-round-2/`, `results/2026-10-10-credit-panel-round-3/`.

| Arm | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|
| LXTUL + pointer references (4 runs) | | 3.919-3.932 | +0.0088 to +0.0104 | 0.154-0.168 | 0.95 | 29.8k |
| one-cell loop (r1) | 3.9019 | 3.8997 | +0.0022 [+0.0019, +0.0026] | 0.193 | | 25,931 |
| random pick | 3.9404 | 3.9344 | +0.0060 [+0.0055, +0.0066] | 0.159 | 0.83 | 22,269 |
| prelude detach | 4.2372 | 4.2237 | +0.0135 [+0.0126, +0.0143] | | | 23,985 |

Span-decoder CE K1-K6: one-cell loop +0.0065, random pick +0.0145, prelude detach +0.0087.

Predictions:
- One-cell loop CE@6 best guess 3.950, worse if at or above 3.930: measured 3.8997. Failed. The
  one-cell loop is 0.019 to 0.032 better than LXTUL, not worse.
- One-cell loop K1-K6 best guess +0.008: measured +0.0022. Failed. The loop earns 0.0066 to
  0.0082 less than LXTUL (about the noise size 0.008, but the CI is tight and the sign matches the
  hypothesis).
- Random pick CE@6 best guess 3.931, worse if at or above 3.930: measured 3.9344. Held by the
  line. Inside noise: it is 0.010 above the reference mean 3.9245, under the 0.015 to 0.020 that a
  single run can resolve.
- Random pick K1-K6 best guess +0.007: measured +0.0060. Held, inside the 0.006 spread.
- Prelude detach CE@6 best guess 3.975: measured 4.2237. Failed. It is 0.29 worse than LXTUL, a
  large real effect: the coda needs the token gradient through the prelude.
- Prelude detach K1-K6 best guess +0.013: measured +0.0135. Held as a number. The "above +0.016"
  line for "geometry objection measured" is not reached.

Readings beside: write worth (val/plan_worth_zero) is 0.193 for the one-cell loop, higher than
LXTUL's 0.154 to 0.168. For the ungraded fan single-cell val CE is 4.68 against 4.537 for all four
cells (`fan/stream_ce` against `fan/mixed_ce`); the oracle best single cell per span is 4.538.
The extrapolation of the 5k val curves (`extrapolate_val_gap.txt`) shows no crossover between the
one-cell loop and LXTUL: the gap is about -0.02, flat or slowly widening.

Wolfe's position, recorded and not argued: he expects 4 cells to win after about 2B tokens. The
knee costs too many tokens to see at 20k. "Keep this in mind moving forward."

## Verdict

Failure. The hypothesis said the one-cell loop loses CE and K1-K6. It gained 0.019 to 0.032 CE@6
and lost K1-K6. A headline CE prediction missed, and the random pick result sits inside noise.

Per reading:
- Random pick versus LXTUL: inside noise on CE@6 (+0.010) and K1-K6. The panel cannot say whether
  the z-trained selection is worth anything. The rule "remove the z machinery" needs CE@6 below
  3.930 AND K1-K6 within spread: K1-K6 holds, CE@6 is inside noise either way. Not decided here.
- One-cell loop versus LXTUL: the CE@6 gap (0.025) is above the 0.020 single-run floor, but the
  two-run one-cell result (round 3) is 3.9013, so the lead holds at about 0.02 at 5k.
- Prelude detach: large, not noise.

What the method could not distinguish: whether the selection is worth anything (random pick,
one run, 0.010 from the references) and whether the 5k lead lasts.

## Updated hypothesis

At 5k the cells and the z selection buy no CE. The loop's contribution (K1-K6) is smaller without
the fan, and write worth is higher. The prelude must keep its token gradient. Next experiments
filed: `failures/2026-10-10-credit-panel-round-2.md` (ungraded fan, one-cell loop without the
decoder) and `failures/2026-10-10-credit-panel-round-3.md` (second runs and noise).
