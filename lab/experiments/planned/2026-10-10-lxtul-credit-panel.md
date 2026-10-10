# LXTUL credit panel: what each credit path is worth

Status: planned

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
