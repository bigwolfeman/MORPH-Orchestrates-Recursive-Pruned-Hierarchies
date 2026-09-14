# TUL-Code — the span code, the tape and the sampler: specification v0.1

Date: 2026-09-14. Status: PROPOSED, not built. Owner: Wolfe. Drafted from the
2026-09-14 brainstorm (drawing board image, three paper readings) and the tree's record.

Decision note: [`.agents/notes/proposed/architecture/2026-09-14-tul-span-code.md`](../.agents/notes/proposed/architecture/2026-09-14-tul-span-code.md).
Paper readings behind it: LaDiR (arXiv 2510.04573), Latent Thought Models (arXiv 2502.01567),
Diffusion Forcing (arXiv 2407.01392); readings filed under
`docs/references/tul-latent-emission/{ladir,ltm,diffusion-forcing}/`.

Terms used in one sense only:

| term | meaning |
| --- | --- |
| span | a run of tokens cut by the boundary rule (`morph/model/tul_layout.py::BoundaryRule`), unchanged |
| slot s | the `M` cell positions that sit AFTER span s and BEFORE span s+1, unchanged from today's prefix cells |
| code z_s | the content of slot s's cells: the code OF SPAN s+1, the span the mouth is about to speak |
| encoder E | training-time only; makes z_s from span s+1's own prelude states |
| tape | the codes of all earlier slots, z_{<s}; the ONLY cross-span channel (strict geometry) |
| seed | slot s's own-span summary, today's `slot_seed` (`E_slot + W_sent·embed(t_last)`), unchanged |
| thinker | the core loop body (the 6 shared core blocks), used as a velocity field over codes |
| mouth | the coda (3 blocks), unchanged in structure |
| k | the number of thinker passes used to sample a code at inference; a deploy dial, not a training regime |

## 0. One paragraph

Every slot-loop arm since 2026-09-04 asked the loop to produce a vector that helps predict
the next span, and every one read a flat loop, a collapsed state, and a probe that could
score the state at ENTRY as well as at exit. The reason is the target: a vector trained
only through "help predict what comes next" is the mean of every continuation, and a mean
is reached in one pass. TUL-Code changes what the slot holds. The slot holds the CODE of
the span it precedes, and the code is defined by reconstruction: at training time an
encoder makes it from that span's own tokens, and the mouth must speak the span reading
that code plus the span's own token path. The thinker is trained, separately and with its
own loss, to SAMPLE that code from the tape by conditional flow matching: one pass per span
at training time, k passes per span at inference. The mouth speaks per token from prelude
to coda with no core pass at all. The loop earns depth only through k, which is a sampling
step count and is in distribution for every k by construction.

## 1. Why this shape: six measured facts

1. **The slot state is scoreable before the loop runs and not after.** A linear probe for
   "next-span CE below median" reads AUC 0.60–0.64 at the loop's entry against a null of
   0.51, and the exit is no higher (`lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md`,
   Step 0). Twelve arms read token K1−K6 inside [−0.0001, +0.0033].
2. **The state is a mean.** Effective rank 5.7–7.3 in 1024 dimensions, pairwise cosine
   0.72–0.77 across a row's slots (`val/slot_eff_rank`, every slot arm; the Thought Register
   at rank 1.24 of 4, `lab/experiments/failures/2026-09-13-arc-thought-register.md`).
3. **The mouth reads the cells when the geometry makes it.** Under strict geometry the
   loop's write is the whole cross-span channel, 0.1865 nats, at CE parity with the bypass
   arm (`lab/experiments/failures/2026-09-12-arc-strict-geometry.md`, verdict 1). The
   reader problem that killed FM1/FM1-CW/FM2 (worth 0.0000 with an additive, detached
   prefix and tokens in reach) does not exist on the strict geometry.
4. **There is 0.40 nats to carry and it is flat across the span.** Cutting every cross-span
   route with no slots costs 0.3994 [0.3838, 0.4162] at 5k, 0.3149 of it flat at offsets
   8+ (`lab/experiments/failures/2026-09-11-arc-span-budget.md`). That is the ceiling on
   what any code channel can be worth on this data at this horizon.
5. **A per-pass target is met in one step; a sampling step count is not.** Seven per-pass
   targets all read ≤ 0.002 on passes 2–6 (`per-pass-targets-met-in-one-step`); LaDiR's
   sampler improves with steps (5 → 10 steps +11.7 points, 10 → 30 +4.8, Figure 5).
6. **Fixed depth is a chain and sampled depth is not** (Wolfe 2026-09-14; `d3fixed` reads
   3.71/3.55/3.44/3.48 at eval depths 1/2/3/6). Flow matching trains no depth at all: the
   velocity field is supervised at one random time t, and every k is valid at inference.

## 2. The diagram

Training. Codes are teacher-forced; the thinker learns beside the mouth.

```
 tokens    [ span 1 ][ slot 1 ][ span 2 ][ slot 2 ][ span 3 ] ...
               |                   |                  |
            prelude(3)          prelude(3)         prelude(3)      per token, strict: own span only
               |                   |                  |
               |            E(span 2) = z_1    E(span 3) = z_2       E pools a span's OWN prelude
               |                   |                  |              states; cells := code (+ noise)
            mouth(3) ---------- mouth(3) ---------- mouth(3)         a token reads its own span's
               |                   |                  |              tokens + the cells of EARLIER
             CE                  CE                  CE              slots (tape); a cell reads itself

 thinker (ONE pass per slot, beside the above, on a doubled slot sequence):
      context copies:  [ z_1 ][ z_2 ]        clean codes, detached (the tape)
      noisy copies:    [ z_t(1) ][ z_t(2) ]  z_t = (1-t) z_0 + t z_s, t ~ U(0,1), z_0 ~ N(0, s^2)
      noisy copy s attends: clean copies < s, its own seed (injection), its own cells
      output v(s) ;  loss  || v - (z_s - z_0) ||^2      z_s detached
```

Inference. E does not exist. The thinker samples; the mouth speaks.

```
 at boundary s:   tape [z_1 .. z_{s-1}] , seed(s)  --thinker x k Euler steps from z_0--> z_s ; append
 per token:       prelude(3) -> mouth(3) reads tape ∪ {z_s} -> next token, until the boundary rule fires
```

Compute per token: prelude + coda = 6 block passes, plus `6·k/⟨span⟩` for the thinker.
The shipped looped model runs 3 + 6·6 + 3 = 42 per token.

## 3. Objects

### 3.1 Sequence layout — unchanged

`SlotLayout`, `pack_tul_row`, the boundary rule, `prefix_k`, `max_slots`, `span_cap`, pad
slots at −100: all as today (`docs/tul-spec.md` §3.1, committed copy). `tokens_through_core`
is `false` and the thinker never touches token positions. Slot s's cells are its `M` prefix
positions (`tul.prefix_k`, today 2), read by the coda exactly as the strict geometry reads
them today (`tg_geometry: strict`, `tg_coda_prefix_reach: all`).

### 3.2 The code z_s — what the cells hold

`z_s ∈ R^{M×C}`, `M = prefix_k`, `C = d_model`. One cell per prefix position. It is the code
of span s+1. Nothing writes through `W_prefix`; the cells ARE the code (the Thought
Register's shape, `tul.slot_cells`, with the register's pooling replaced by E).

Normalisation: each cell is RMS-normalised to unit per-component scale before it is used
anywhere (the FM doctrine §7: the source scale is load-bearing; with unit-scale codes,
`source_std 1.0` is matched). `tul.code_norm: rms` is the only mode in v0.1.

Rate control (what keeps the code coarse enough to guess): Gaussian noise added to every
cell at training time, `z ← z + code_noise · ε`, ε ~ N(0, I), the LaDiR augmentation
(their k = 3 in their space). `tul.code_noise` default 0.5, UNTUNED. A β-VAE head
(`code_kl_beta`) is NOT in v0.1 (see §10).

### 3.3 The encoder E — training-time only

`TULCodeEncoder(d_model, M)`: `M` learned queries pool `M` vectors out of the prelude
states of span s+1's TOKEN positions, single-head softmax over that span's own tokens only
(the `TULSlotRegister` mechanism, `morph/model/tul.py`, pointed at the NEXT span instead of
the own span), then a per-cell `W_o` and per-cell embedding `P_cell`, then RMS norm. Private
RNG stream (the `W_sent` precedent) so a code model's base weights are byte-identical to
the strict ruler's.

E sees the whole of span s+1 (its prelude states are causal within the span, and the pool
covers every position, so the last state alone already sees the span). This is the ONE
non-causal path in the training forward, and it is the point: the code is made from the
span it codes. Pad slots and the row's last slot (no span s+1) get a zero code and are
masked from every loss.

Gradient: E trains through the mouth's CE only. The FM loss never reaches E (§5).

### 3.4 The seed — unchanged

`tul.slot_seed: boundary` as today. The seed enters the thinker as the slot's injection
term (`x0` for slot positions, `_build_injection_term`), not as the state.

### 3.5 The thinker — the core body as a velocity field

The 6 shared core blocks (`n_core`) run ONE pass over a slot sequence of `2·S·M` positions:

* context copies: the clean codes `z_{s'}` for every slot s' (detached), one position per
  cell, attending causally to earlier context copies only;
* noisy copies: `z_t(s)` for every slot s, attending to context copies of slots `< s` and
  to the noisy copies of its own slot (its M cells see each other).

State entry: the carrier at a noisy copy is `z_t(s)` itself (`core_init` is the identity on
this path; the Parcae noise entry, `core_state_init: noise`, does not apply — the noise IS
the source `z_0`). Time conditioning: a sinusoidal embedding of `t` through a small MLP
(the audited `SigmaConditioning` basis in `morph/model/fm_planner.py`, `t_embed_scale`),
added to the slot's injection term. Output: `v(s) = W_v · _readout(h_noisy(s))`, `W_v`
zero-init so the first velocity estimate is 0.

Objective: conditional flow matching exactly as `fm_planner._cfm_loss` defines it —
`z_0 ~ N(0, source_std²·I)`, `t ~ U(0,1)` per slot, `z_t = (1−t)·z_0 + t·z_s`,
`loss = mean_valid ‖v̂ − (z_s − z_0)‖²`, divided by the analytic null floor
(`loss_scale: auto`) so it starts near 1.0 beside a CE of 4–11 nats. Per-t-band `rel`
stats are logged as they are today.

No BPTT through any loop exists at training time. There is no depth draw. `mean_depth`,
`max_depth`, `slot_depth_fixed`, `bptt_depth` are not read by this path.

### 3.5a Bootstrapping from a blank model — there is no oracle

LaDiR's "oracle latents" are its trained VAE encoder run on the real sentences. Ours are E
run on the real span, in the same forward. Phase 1 IS the VAE stage: E and the mouth
co-train through the mouth's CE with the code teacher-forced, from a blank model. What
makes the mouth read the code rather than ignore it is not a frozen decoder (LaDiR) but the
strict geometry: the cells are the only cross-span source, and a deterministic write in
them already carries 0.1865 nats on the strict ruler. The code will hold what the mouth
cannot get from the span's own token path, not the whole span; G1 measures how much.

LaDiR's `<BOT>`/`<EOT>` delimiters and its special-token head ("another thought or the
answer?") have no analogue here: the boundary rule cuts spans from the tokens the mouth
writes, every span gets one code, and the model never chooses how many codes to think.

### 3.6 The mouth — the coda, unchanged in structure

Strict geometry as shipped: a token sees its own span's tokens plus the cells of earlier
slots; a cell sees itself alone; the CCA conv and the retention carry reset at every
segment; the coda's per-layer injections at the cells are zeroed
(`tests/test_tul_strict_geometry.py`). Bowman token-state dropout on the coda input stays
at `tul.token_state_dropout`.

The cells' content at training time is `z_s` (+ noise) in phases 1–2, and a mix of `z_s`
and sampled `ẑ_s` in phase 3 (§6). Phase 3 codes are detached before the coda.

## 4. Training forward, step by step

1. `_tul_front`: embeddings, prelude over the packed row under the strict masks (as today).
   Output `xn` (normalised prelude states).
2. `E(xn, layout) → z [B, S, M, C]` (§3.3), zero on pad and last slots.
3. Cells for the coda: `z_coda = rmsnorm(z) + code_noise·ε` (phases 1–2). Phase 3: for a
   fraction `code_rollout_p` of valid slots, replace with `ẑ` sampled by the thinker at
   `code_rollout_steps` Euler steps, under `no_grad`.
4. Thinker pass (from phase 2 on): build the doubled slot sequence (§3.5) from
   `z.detach()`, sample `t`, `z_0`; one core pass; `v̂`; `L_fm`.
5. `_back_region` (coda) with the cells scattered into the prefix positions;
   `_tul_group_losses` → `L_ce` (`emit_weight 0.0`, `plast_weight 1.0`, `slot_id` logit
   masked, pad slots −100, as today).
6. `L = L_ce + code_fm_weight · L_fm`.

Shapes at the panel scale (seq 1024, `max_slots` 128, M 2): E adds one pooling attention
over 1024 keys per cell; the thinker's pass is 512 positions through 6 blocks, once.

## 5. Losses and gradient routes

| term | reaches | never reaches |
| --- | --- | --- |
| `L_ce` (mouth) | prelude, coda, embeddings, `E_slot`/`E_mask`, **E** | the thinker (phase 3 codes detached), `W_v`, the time MLP |
| `L_fm` (thinker) | the 6 core blocks, `W_v`, the time MLP, the seed path (`W_sent`, and through it the prelude of span s) | **E** (target detached), the tape (context copies detached), the coda |

Why the split, and why it is the opposite of FM1's: FM1 detached the thing the coda read
(the plan) and let nothing align the writer's basis with the reader's; the coda treated the
plan as noise and the oracle probe measured even the true target at 0.0000
(`lab/experiments/successes/2026-08-28-oracle-prefix-probe.md`). Here the coda reads a code
it co-trains through E, and only the GUESS is outside the CE graph. The JEPA trap (the
target learns to be easy to guess) is closed by the detach on the target; the LTM trap
(an amortised encoder collapses against a strong AR decoder with a KL) is closed by having
no KL and by the strict geometry, under which the coda cannot bypass the cells.

`code_fm_weight` default 1.0 with `loss_scale: auto` (FM1's setting). UNTUNED.

## 6. Phases — one run, three phases, fractions of `training.steps`

| phase | starts at | what changes |
| --- | --- | --- |
| 1 define the code | 0 | `L_ce` only. E and the mouth co-train; the thinker is built and idle (its loss weight is 0) |
| 2 learn to guess it | `tul.code_phase2_at` (default 0.10) | `L_fm` on. The code is a slow target: E keeps training through CE |
| 3 rollout | `tul.code_phase3_at` (default 0.50) | `code_rollout_p` (default 0.5) of valid slots hand the coda a sampled `ẑ` at `code_rollout_steps` (default 8) instead of the code |

LaDiR's stage 2 is load-bearing there (MATH 30.7 without, 45.2 with). The defaults above
are placeholders; the smoke and the first arm set them. The phase switches are step-based
like `mux_activate_at` / `sigreg_activate_at` (`morph/training/train.py`), resolved once
and written to the wandb manifest.

A phase-1 checkpoint cannot generate (E needs the span it has not written). The first
checkpoint that is a model is the first one past `code_phase2_at`.

## 7. Inference

`morph/inference/tul_generate.py` gains one branch. At a boundary, with tape
`[ẑ_1 .. ẑ_{s-1}]` and the seed of span s: `z ← z_0 ~ N(0, source_std²)`; for
`j = 0 .. k−1`: `z ← z + (1/k)·v(z, t_j, tape, seed)` with `t_j = j/k` (Euler, the same
integrator `fm_planner` uses); append `rmsnorm(z)` to the tape. Then tokens as today.

`k` is `tul.code_infer_steps` at generation and a per-call argument in the instruments.
Cost of k, per token at ⟨span⟩ ≈ 12 and M = 2: `6·k·2/12` core block passes, so k = 8 is 8
against the plain loop's 36 and the sequential break-even is near k = 72. The noise
augmentation (§3.2) is what keeps k small: the mouth tolerates a code within one noise
radius of the true one, so the sampler only has to land inside that ball. LaDiR's
batch-parallel diversity guidance (§3.4 there) is a best-of-N reasoning device and is not
used: one sample per code from the source noise is what an LM needs.
The K-curve of §8 chooses it. No halting, no gate, no PonderNet.

Wolfe's 2026-08-28 veto on loop-depth variation for the FM planner ("fixed inference
constant") was written for a planner beside a core loop. Here k is the only depth there
is. Recorded as an open decision (§13).

## 8. Evaluation — what is measured, and the gates

All on paired rows (480, `span_budget_profile.py`), never on the runner's Final val_loss.

| instrument | reads | what it is |
| --- | --- | --- |
| `val/ce_tf` | CE with the encoder's codes in the cells (teacher-forced, noise off) | the CEILING. NOT an LM number: the cells contain the span being scored |
| `val/ce_k{K}`, K ∈ {1, 2, 4, 8, 16} | CE with sampled codes, teacher-forced tape (earlier cells = encoder codes), single sample | the honest number and the K-CURVE. `ce_k1 − ce_k16` is the loop contribution |
| `val/code_gap` | `ce_k16 − ce_tf` | how much of a span's code is not guessable from the past |
| `val/code_eff_rank`, `val/code_pairwise_cos` | the codes of a row | the collapse instruments, same names as the slot family's |
| `fm/rel` per t-band | `L_fm / null` | the thinker's honesty instrument (`fm_planner._finish`) |
| `worth_profile` shuffle cost on the cells | `ce(shuffled cells) − ce` at K 8 | the reader instrument; must be positive (FM doctrine rule 1: cost, not fraction) |
| generation at K ∈ {1, 8}: gen-PPL with rep4 / distinct-3 | the diversity guard, doctrine rule 11 | |

Controls, one seed each in the first panel, two before any claim (doctrine rule 6):

* `slot-spandec-strict` at the same commit (the deterministic loop write on the same
  geometry; its cells are a mean);
* the `d1` rung (`notul_norm_match_20k_d1`, 12 block passes per token) — the compute bar;
* `notul` plain looped (42 per token) — the ceiling on CE.

Pre-registered gates, to be frozen in `lab/experiments/planned/2026-09-14-arc-tul-code.md`
BEFORE the first run (numbers here are the proposal; the prereg owns them):

* **G1 (the code is a code).** `ce_tf` at 5k at least 0.15 nats below the strict ruler on
  paired rows, and `code_eff_rank ≥ 16` of 2·1024 per row. Fails ⇒ E or the geometry is
  broken; nothing downstream is readable.
* **G2 (the loop earns by sampling).** `ce_k1 − ce_k8 ≥ 0.02` at 5k with `fm/rel` below 0.9
  in every band. Fails ⇒ the thinker is not integrating anything; the line stops.
* **G3 (a sample beats a mean).** `ce_k8` below the strict ruler's CE on paired rows at 20k,
  CI excluding 0. Fails ⇒ multimodality does not pay on this data; the line stops.
* **G4 (ship bar).** `ce_k8` within 0.02 of the `d1` rung at 20k, at fewer block passes per
  token than `d1`.

A 5k CE ranks nothing between arms within 0.01 (standing rule); G1 and G2 are within-run
readings and are readable at 5k; G3 and G4 are 20k readings.

## 9. Config — the `tul.code_*` block (all construction-time; unknown keys raise)

```yaml
tul:
  code: false                # true builds E, W_v, the time MLP; false is bit-identical to the ruler
  code_noise: 0.5            # UNTUNED. Gaussian noise on the cells at train (LaDiR k)
  code_norm: rms             # the only mode in v0.1
  code_fm_weight: 1.0        # with loss_scale auto (FM1's setting)
  code_source_std: 1.0       # matched to unit-scale codes (doctrine §7)
  code_t_embed_scale: 1.0    # fm_planner's knob, same meaning
  code_phase2_at: 0.10       # UNTUNED fraction of training.steps
  code_phase3_at: 0.50       # UNTUNED
  code_rollout_p: 0.5        # UNTUNED fraction of valid slots given a sampled code in phase 3
  code_rollout_steps: 8      # Euler steps for phase-3 samples (LaDiR: 50 -> 10)
  code_infer_steps: 8        # k at generation; the instruments sweep it
```

Required by construction, refused otherwise: `tokens_through_core: false`,
`tg_geometry: strict`, `spandec: false`, `slot_cells: 1` (the cells come from E, not the
register), `vq_codes: 0`, `oracle_z: false`, `grad_pass: false`, `fm: null` (the FM1
planner). `model.core_fixed_point_lambda` and `model.slot_gain_lambda` print INERT.

Config lineage: `tul_code.yaml` composes `tul_slot_spandec_strict` and sets `spandec: false`,
`code: true`; the panel arms are one-factor against it. Wandb project `morph-tul`.
`training.ademamix_t_beta3` pinned (doctrine rule 5).

## 10. Not in v0.1 (each is a named follow-on, not a silent omission)

* **Diffusion-Forcing tape noise** — per-cell noise levels on OLDER tape cells with a level
  embedding, so the tape decays by design. The mechanism is exact (DF Theorem 3.1); the
  decay law over span age is not given by the paper. v0.2 knob `code_tape_noise`.
* **A KL bottleneck** (`code_kl_beta`, a σ head on E). Rate control in v0.1 is the noise
  augmentation alone; the KL brings LTM's collapse dynamics and is a second arm.
* **Seed as the source** (`z_0 = seed + noise`, Wolfe's "seed FM from the mux tokens").
  A warm start needs fewer steps; it also changes what `ce_k1` means. Second arm.
* **More cells per span** (`M` 4, 8). Width is worth 0.03 nats on its own
  (`trajectory-prefix-is-width-plus-pad-artefact`); a width sweep is confounded until G1–G3 read.
* **Diversity guidance at sampling** (LaDiR §3.4). A generation-quality knob; not a CE knob.
* **The span decoder as a second reconstruction head.** The mouth is the decoder here.
* **Tokens through the core, any core loop on tokens, any per-token thinker pass**
  (the drawing board's blue "once through the loop body per token" variant). Recorded in the
  note's alternatives; not built.

## 11. Invariants and tests (one test per row, before any GPU run)

| id | invariant | test |
| --- | --- | --- |
| C1 | `code: false` ⇒ bit-identical loss, logits and parameter set to `slot-spandec-strict` at the same commit | `test_tul_code.py::test_off_is_the_strict_ruler` |
| C2 | at train, a token id in span j moves nothing at any position BEFORE slot j−1's cells (the only non-causal path is span j → its own code → span j and later) | `test_train_forward_is_causal_up_to_the_own_code` |
| C3 | at eval with sampled codes, a token id in span j moves nothing at any position ≤ its own (plain causality) | `test_sampled_forward_is_causal` |
| C4 | `L_fm` gradient on every parameter of E and on the prelude states of the target span is exactly zero | `test_fm_loss_never_reaches_the_encoder` |
| C5 | in phase 3 the coda's CE gradient on the core blocks, `W_v` and the time MLP is exactly zero | `test_ce_never_reaches_the_thinker` |
| C6 | k Euler steps from the same `z_0` with a zero `W_v` return `z_0` for every k; one step with `W_v` set reproduces `fm_planner`'s integrator to fp32 tolerance | `test_sampler_matches_fm_planner_euler` |
| C7 | the doubled slot sequence's masks: a noisy copy never attends a context copy of its own or a later slot; a context copy never attends a noisy copy | `test_thinker_masks` (two-sided, with sabotages, the strict-geometry precedent) |
| C8 | pad slots and the row's last slot contribute 0 to `L_fm` and hold a zero code | `test_pad_and_last_slots_are_masked` |
| C9 | unknown `tul.code_*` keys raise; every required-by-construction pair in §9 raises on violation | `test_tul_setup_keys.py` extension |
| C10 | every panel config composes through Hydra + `tul_setup` at the queued commit AND runs a 12-step GPU smoke on the Spark before it is queued (`compose-every-config-before-queueing`, `mean-eq-max-is-not-fixed-depth`) | `tul_smoke`-style config `tul_code_smoke.yaml` |

## 12. Compute (block passes; measured rates come from the smoke)

| model | per token | per span (⟨span⟩ ≈ 12 at seq 1024) |
| --- | --- | --- |
| plain looped, mean 6 | 3 + 36 + 3 = 42 | — |
| `d1` rung | 12 | — |
| strict slot loop, mean 6 | 6 (+ 36 on the slot positions only) | 36 |
| TUL-Code train | 6 (+ E) | 6 on 2M positions, once |
| TUL-Code infer, k | 6 | 6·k |

Memory: the thinker's pass is 512 positions, no checkpointing needed; the doubled slot
sequence is the only new activation. Expect below the strict ruler's resident figure. The
first smoke measures it; nothing here is a claim.

## 13. Open decisions for Wolfe (the spec proceeds under the first option of each)

1. **k as the only depth.** The 2026-08-28 veto on depth variation for the FM planner does
   not transfer (no core loop beside the sampler exists). Proceed with k as a deploy dial
   chosen by the K-curve. Alternative: fix k at 8 everywhere and read only `ce_k8`.
2. **Rate control.** Noise augmentation only (v0.1). Alternative: β-VAE head from the start.
3. **The FM gradient into the seed path.** Wolfe 2026-09-14: an ABLATION, not a decision.
   Arm `tul_code` allows it (the prelude of span s may learn to make span s+1's code easier
   to guess; E's input is span s+1, which `L_fm` never reaches); arm `tul_code_seeddetach`
   detaches the seed (FM1's choice). One factor, same prereg.
4. **Where the tape's context copies come from at inference.** Sampled codes (§7).
   Alternative: keep the mean code (`k`→∞) for old cells and sample only the current one.

## 14. Risks, honestly

* **LCM.** This is LaDiR's shape on web text, which is LCM's setting, and LCM lost to token
  AR. What differs: the AR token path inside every span, a coarse code (noise), a code
  co-trained with the mouth, and cross-span content bounded at 0.40 nats by measurement.
  G3 is the test and it can fail.
* **The binding rule on FM (2026-08-30).** "Any future FM-flavored proposal must target
  post-core carriers or it is pre-refuted." Every P1 design regressed PRE-core prelude
  pooled features, defined by nothing decodable, into a detached prefix. This design's
  target is a code defined by what the mouth can decode, co-trained through the mouth, on
  a geometry where the mouth must read it. **Waived by Wolfe, 2026-09-14**; the note keeps
  the argument.
* **The moving target.** LaDiR freezes the VAE; from scratch E keeps moving through CE.
  Stop-grad plus the phase-2 delay is the mitigation; `fm/rel` falling is the check.
* **Scoring.** `ce_k{K}` is a single-sample bound, not a likelihood; an importance-weighted
  estimate is a follow-on instrument. Ship claims on this model are partly generation
  claims (doctrine rule 11).
* **Three of six facts in §1 are one-seed numbers.** Two seeds before any claim.

## 15. Provenance — decision → source

| decision | source |
| --- | --- |
| the code is of the NEXT span and made from its own tokens | LaDiR §3.1 (VAE on the sentence), LTM (z fit on the sequence) |
| noise augmentation as rate control | LaDiR §3.2.1 (latent noise k = 3, token substitution p = 0.3) |
| stage 2 rollout | LaDiR Table 1 ablation (30.7 → 45.2) |
| no KL, no amortised-encoder-plus-KL | LTM §3.4 (VAE baseline collapsed with annealing) |
| tape as decaying history (v0.2) | Diffusion Forcing Theorem 3.1, App. B.7, stabilisation §3.3 |
| strict geometry, cells read by the coda | `2026-09-12-strict-slot-geometry.md` |
| CFM, `loss_scale auto`, source scale rule | `morph/model/fm_planner.py`, `docs/tul-fm-probing.md` §7 (committed copy) |
| no BPTT through any loop, guess outside the CE graph | FM1's non-negotiable, `morph/model/tul_fm.py` header |
| never decode a span from one vector with no token path | CLAUDE.md (Huginn 2026-08-16, MegaByte T7, Bowman T2, Hourglass T6) |
