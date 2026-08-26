# Exact Shapley credit for prelude / core loop / coda

Measured 2026-08-25. **NOT a pre-registered experiment.** It is a measurement, run to test a
hypothesis raised in conversation: that the TUL core's problem is CREDIT ASSIGNMENT rather
than stability. Any intervention built on it must be pre-registered separately.

Tool: `lab/divergence/region_shapley.py`. Three players means exact Shapley — 8 coalitions,
no sampling. A region is ablated by making it the identity. Value of a coalition is nats
SAVED against the empty coalition. The Shapley values sum exactly to the full model's value
over the empty one at every checkpoint below; that identity is the arithmetic check.

Eval set: 8 fixed validation batches, `tul_a1`, batch 6, eager kernels.

## Result

| checkpoint | state | Shapley(total loss) | Shapley(`ce_main`) | Shapley(`ce_emit`) |
|---|---|---|---|---|
| `ROLL_step_1625` | healthy | prelude 2.6253 / **core 0.0065** / coda 2.4568 | **core 0.0015** | **core 0.2274** |
| `ROLL_step_1750` | healthy | prelude 2.6206 / **core 0.0080** / coda 2.5215 | **core 0.0007** | **core 0.3296** |
| `ROLL_step_1850` | taken over | prelude 2.6941 / **core −0.0015** / coda 2.4997 | **core −0.0010** | **core −0.0314** |

`ce_main` is the ordinary token positions — the actual language-modelling objective.
`ce_emit` is the slot's own target, the next span's first token predicted WITH the plan.

## What it says

**1. The core's entire value is in its own private target.** At `ROLL_step_1750` the core is
worth **0.3296** nats on `ce_emit` and **0.0007** nats on `ce_main` — a factor of 470. The
looped core does almost nothing for the objective the model is trained to serve.

**2. Redundancy is refuted.** A leave-one-out margin cannot separate "useless" from
"redundant, and the coda covers for it". Shapley averages over every coalition, so a
redundant-but-useful core would still score highly. It scores 0.0007. The core is not
covered for. It is not doing the job.

**3. At takeover the core goes NEGATIVE on every metric**, including the one target it had:
`ce_emit` −0.0314. It stops being worth anything and becomes a cost.

**4. The core loses its own target to a free competing path.** `_tul_half_weights`
(`transformer.py:1941`) puts the slot between a span's last token and the next span's first
token, so that token is predicted TWICE — once from `t_last` by the plain token path with no
plan (`ce_plast`), once from the slot's emit position with the plan (`ce_emit`) — each at
weight 0.5, so the target is not double-counted. Measured `ce_plast − ce_emit` in the full
model: **−0.1972** at 1625, **−0.2191** at 1750, **−0.4842** at 1850. Negative throughout:
the plan is BEATEN at its own job, and the gap widens as the takeover proceeds.

The same quantity appears as `cf` in every run's VAL line and has been negative all along:
−0.103 to −1.047 across the H24 arms.

## Why this reframes the campaign

Put beside two existing numbers:

- H14 measured the slot's own label at **2.8 % of loss weight but ~50 % of the gradient**.
- At takeover the core holds **>90 %** of the pre-clip gradient.

So the core draws half the gradient — and eventually nearly all of it — while contributing
0.014 % of the model's value. **Gradient share and value are decoupled by about three orders
of magnitude.** That is a credit-assignment failure, not a stability failure, and it explains
why every stability-targeting cure in this campaign has failed: there was nothing holding
those parameters anywhere in the first place.

## What it does not say

- Measured on ONE checkpoint family, `onset-capture`, a TUL A1 arm at 1625-1850 steps, eager
  kernels, batch 6, 8 batches. Not known for a plain non-TUL run, the shipped recipe, or a
  well-trained model.
- Ablation-by-identity is one choice of counterfactual. A region set to identity still passes
  its input through, which is the closest thing to "absent" for a residual stack, but it is
  not the only defensible definition.
- Nothing here says which intervention fixes it.

---

## Follow-up 2026-08-25: it is the PLAN that is empty, not just the loop

Raised in conversation: 0.0007 nats reads like the core is SKIPPED, not like it is competing
for credit. That was right, and it exposed a flaw in the measurement above.

**Ablating the core by identity removes the LOOP, not the SLOT.** The slot position still
carries `E_slot + mean(embed(span))` through the prelude, so the coda still gets a span
summary. `lab/divergence/slot_path_worth.py` separates them by zeroing what
`prefix_project` writes into the shared sequence — positions and validity untouched, only
the plan's content removed:

| removed | total | `ce_main` | `ce_emit` |
|---|---:|---:|---:|
| the LOOP (slot keeps its bag-mean) | 0.0169 | **0.0051** | 0.5420 |
| the WHOLE PLAN (slot values zeroed) | 0.0777 | **0.0191** | 2.6564 |

"no-plan" and "neither" agree to four decimals — once the plan is zeroed the loop cannot
matter. That is the consistency check.

The plan path is **connected**: zeroing it costs 0.0191 on `ce_main`, not 0.0000. If tokens
could not attend the slots it would be exactly zero. `window_size` is 256 against
`span_cap` 32, so reach was never the problem.

### Is the plan empty, or capable but never asked?

`lab/divergence/token_tax_sweep.py` sweeps `token_state_dropout` at EVAL — the tax whose
stated purpose is to force the coda to decode from the plan. Deterministically seeded, so
every condition masks the same positions and the comparison is paired.

| token drop `p` | `ce_main` with plan | without plan | plan worth |
|---:|---:|---:|---:|
| 0.00 | 4.8093 | 4.8284 | 0.0191 |
| 0.15 (the shipped value) | 5.0934 | 5.1278 | 0.0344 |
| 0.50 | 5.8454 | 5.9029 | 0.0575 |
| 0.90 | 7.0219 | 7.0794 | 0.0574 |
| 1.00 | 7.5327 | 7.6026 | **0.0699** |

**Mask every token state and the plan is still worth 0.0699 nats**, while `ce_main`
collapses to 7.5327. The coda cannot recover the span from the plan because the plan does
not contain it.

### The causal chain, end to end

1. The core's only DIRECT supervision is `ce_emit` — predict ONE token, the next span's
   first. Worth 2.6564 nats to that target.
2. Its INDIRECT route, helping the coda predict the span, is worth **0.0191** nats, so
   almost no gradient arrives that way.
3. So the plan learns to be a one-token predictor. That is the only thing it is asked for.
4. Hence it carries ~0.07 nats about the span even with the token path fully removed.
5. So for actual token decoding the core is, as suspected, close to skipped.
6. And it still loses its one job to the free token path: `ce_plast − ce_emit` is −0.2191.

**TUL's plan is trained to predict one token and is then expected to carry a thought.**
Nothing in the loss asks it to summarise the span, so it does not.

### What this changes about the fix

Raising `token_state_dropout` is the spec's own "collapse tax" and is the obvious lever, but
this sweep is EVAL-ONLY on a model trained at `p = 0.15`. It shows the plan is empty NOW. It
does NOT show the plan cannot be FILLED if training ran at a high `p`, where the coda's
`ce_main` would depend on the plan and real gradient would reach the core. That is the
experiment, and it must be pre-registered.

The `p = 1.0` row is out of distribution — the model never trained there — so read the trend
(0.0191 → 0.0344 → 0.0575) rather than the endpoint.
