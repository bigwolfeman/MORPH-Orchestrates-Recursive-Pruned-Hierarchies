# Result: the core takeover is an unstable BACKWARD operator, compounded by the loop

Status: measured 2026-08-23, on four trajectories, two of them bit-reproducible.

Diagnostic work, not a pre-registered ablation. The detection rule reported here was fixed
before it was applied to the trajectories it is scored on, and the parameter sweep behind it
is shown rather than summarised.

## The question

Phase 1 established an ordering — the core takes the gradient, block 0 first, the LM head
never moves — and a forward observation: the realized per-iteration gain runs away at loop
iteration 0 and nowhere else. That was a localisation, not a mechanism, and it turned out
not even to be a discriminator.

## The forward gain is a correlate, not the cause

`repl-det-a` and `repl-det-b` are byte-identical runs at the same seed. One took over and
one recovered. The forward gain at iteration 0 climbs in **both**:

| | median `core_gain_t0`, last 200 steps | outcome |
|---|---:|---|
| `repl-det-b` | 2.79 | took over |
| `repl-det-a` | 2.06 | **recovered** |

A criterion of `core_gain_t0 > 2` sustained fires on the survivor too, at step 3809. The
forward gain ramps in healthy runs. **Phase 1's Finding 1 stands as a description of where
the forward gain moves and must not be read as the cause.**

## What actually separates them: the per-block backward gain

The core is `n_core` blocks applied in sequence, and the backward runs core.5 → core.0. If
each block multiplies the incoming gradient by a factor `g`, block 0 ends up a factor
`g^(N-1)` above the last block. So fit

    log ‖grad_i‖ = a + b·i   over blocks i = 0..N-1,   g = exp(-b)

and report `g` with the fit's `r²`. `g ≤ 1` is a contracting, healthy backward.

The `r²` is not decoration. A healthy profile is flat and noisy, so the fit explains nothing
and `g` is meaningless there; a sick one is cleanly geometric. That difference is itself the
signal:

| state | per-block gain `g` | `r²` | core share |
|---|---:|---:|---:|
| healthy | 0.93 – 1.04 | **0.01 – 0.55** | 0.01 |
| taken over | 1.43 – 1.88 | **0.89 – 0.99** | 0.99 |

Median `r²` while the core holds more than half the gradient: **0.971** (n = 1493 steps) on
the control, **0.942** on the reproducible control. The profile really is geometric, so this
is a uniform operator gain and not a spike confined to block 0.

## It separates all four trajectories, and it leads

Guard rule as shipped: `g > 1.0`, counting only steps with `r² ≥ 0.5`, on more than 30 % of
the last 200 probed steps.

| run | outcome | block gain fires | core share fires | forward gain > 2 |
|---|---|---:|---:|---:|
| `phase1-onset-s0` | took over | **1434** | 2033 | 2063 |
| `repl-det-b` | took over | **3368** | 3874 | 3328 |
| `repl-det-a` | **recovered** | **never** | never | 3809 |
| `repro-ctrl` (deterministic) | took over | **857** ¹ | 1093 | — |

¹ measured with a stricter consecutive-100 variant during analysis; the run was aborted at
1093 by the share criterion before the shipped windowed rule could be scored on it.

The block gain leads the share by **599 steps** on the control and **506** on `repl-det-b`,
fires on every run that died, and stays silent on the one that lived. 16 of the 36
(min_r², window, fraction) settings swept separate all four the same way, so this is a
plateau rather than a tuned edge (`ignore/perf/phase1/tune_guard.py`, `block_gain.py`).

The `r²` floor is load-bearing and was tested, not assumed: at `min_r2 = 0.0` the guard
fires on `repl-det-a` — the survivor — at around step 250.

## Why a gain barely above 1 destroys the run

The core is weight-shared across the loop, so the unrolled backward is `n_core × T_grad`
blocks deep. With `n_core = 6` and `bptt_depth = 4`, that is 24 blocks. At the measured
`g ≈ 1.88`:

    1.88^24 ≈ 3e6

against an observed pre-clip core gradient rising from ~0.015 healthy to ~1e6 — a ratio of
about 7e7. Same order of magnitude, from one measured number and two config constants. A
per-block gain of 1.9 is unremarkable in a feed-forward stack; it is fatal in a loop.

This is also why the single global clip starves everything else: the core's gradient is not
merely largest, it is larger by `g^24`.

## What this reframes

- **The disease is contractivity of the BACKWARD operator**, which is what the project's own
  iterative-map note argued for on theoretical grounds and what the forward-side probes kept
  missing. `ρ_eff` measured 1.9–3.2 on healthy checkpoints because it was measuring the
  forward map.
- **`core_gain_clip` clamps the forward magnitude**, and the forward magnitude is the
  correlate. That is consistent with it "curing" runs without anyone being able to say why,
  and with it binding only at iteration 0 (40.9 % of pre-onset steps at t0, 0.0 % at t2–t7).
- **The right intervention targets the backward gain** — a per-block spectral or Lipschitz
  bound on the core — not the forward carrier's size.

## Not verified

- **That the block gain is causal.** It is an ordering and a separation on four
  trajectories, three of which are the same architecture at one seed. The intervention test
  is running (`lab/experiments/planned/2026-08-23-tul-iteration0-mediation.md`).
- **The `g^24` arithmetic is an estimate**, not a derivation. It assumes the per-block gain
  applies uniformly across iterations and ignores the HC residual's contribution, so treat
  the agreement as consistency, not proof.
- Whether the same signature appears in non-TUL MORPH runs, or in the looped core without
  the slot layout. Only TUL arms were measured.
- The reproducible control was aborted at step 1093, so its post-onset behaviour under the
  shipped windowed rule is unmeasured.
