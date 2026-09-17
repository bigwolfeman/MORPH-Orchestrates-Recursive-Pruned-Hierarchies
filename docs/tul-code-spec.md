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

Compute, counted as block passes PER POSITION: prelude + coda = 6 per token, plus the
thinker's `6·k·M` per span, i.e. `6·k·M/⟨span⟩` per token (M = 2, ⟨span⟩ ≈ 12: k per token).
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
of span s+1. `W_prefix` is still BUILT (so the parameter set matches the strict ruler,
invariant C1) and never applied on a code model: the cells are scattered into the prefix
positions directly through the same index `prefix_project` computes (lifted into a helper),
and `W_prefix.requires_grad` is False so it draws no decay. The offset, drawn once because
every reader gets it wrong first:

```
 [ span 1 ][ slot 1 = z_1 ][ span 2 ][ slot 2 = z_2 ][ span 3 ] ...
   z_1 = E(span 2)  seed(1) = summary of span 1   tape at slot 3 = {z_1, z_2} = codes of spans 2, 3
``` Nothing writes through `W_prefix`; the cells ARE the code (the Thought
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
added to the slot's injection term. Output: `v(s) = W_v · RMSNorm_v(mean over HC streams
of h_noisy(s))` with a PRIVATE norm inside the velocity head, never `_readout` (whose
`lm_mixer` and `final_norm` are the LM head's own parameters and must not receive the flow
gradient). `W_v` zero-init so the first velocity estimate is 0.

Objective: conditional flow matching exactly as `fm_planner._cfm_loss` defines it —
`z_0 ~ N(0, source_std²·I)`, `t ~ U(0,1)` per slot, `z_t = (1−t)·z_0 + t·z_s`,
`loss = mean_valid ‖v̂ − (z_s − z_0)‖²`, divided by the analytic null floor
(`loss_scale: auto`) so it starts near 1.0 beside a CE of 4–11 nats. For RMS-normed codes
`E‖z‖² = C`, so the CFM floor is `C + C·source_std²` = 2048 at C 1024, NOT FM1's
`1 + d·s²` (its targets were unit-L2). Passing FM1's constant would put `fm/rel` off by
about 1000x; `_cfm_loss` and `_finish` are copied as arithmetic, not called (they take
an `FMPlanner`). Per-t-band `rel`
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

Implementation shape (from the 2026-09-14 map, `ignore/notes/2026-09-14-tul-code-impl-map.md`):
the thinker is a NEW sibling method `_tul_code_core` beside `_tul_core_db1` and a new
branch in `_forward_tul` before the slot-loop `else:`, never an if-branch inside
`_tul_core`'s depth loop. The doubled sequence's mask is delivered as `tg_relation` (the
one kwarg that REPLACES the causal relation; `tg_allow` can only narrow it), following
`clean_noisy_mask` in `morph/model/diffusion_blocks.py`. E is a new `TULCodeEncoder`
(the register's pooling with the bag index shifted by one and `next_span_pool`'s validity
mask), not a re-pointed `TULSlotRegister`.

1. `_tul_front`: embeddings, prelude over the packed row under the strict masks (as today).
   Output `xn` (normalised prelude states).
2. `E(xn, layout) → z [B, S, M, C]` (§3.3), zero on pad and last slots.
3. Cells for the coda: `z_coda = rmsnorm(z) + code_noise·ε` (phases 1–2), rms-renormed
   again when `code_noise_renorm` is on. Phase 3: for a fraction `code_rollout_p` of valid
   slots, replace with `rmsnorm(ẑ)` sampled by the thinker at `code_rollout_steps` Euler
   steps, under `no_grad`. **Statistic rule (2026-09-15):** without the renorm a truth cell
   has RMS `sqrt(1 + code_noise²)` and a sampled cell RMS 1, and the phase-3 coda learns
   that norm as the "sample" flag; eval's `encoder` mode and the generator's closed-slot
   tape therefore feed `z · sqrt(1 + code_noise²)` (the trained statistic), never bare `z`.
   Measured on tul-code-20k at 20k: bare z reads 1.25 nats, the trained statistic 0.35
   (`lab/experiments/results/2026-09-14-arc-tul-code-20k/code_norm_flag_*`).
4. Thinker pass (from phase 2 on): build the doubled slot sequence (§3.5) from
   `z.detach()`, sample `t`, `z_0`; one core pass; `v̂`; `L_fm`.
5. `_back_region` (coda) with the cells scattered into the prefix positions;
   `_tul_group_losses` → `L_ce` (`emit_weight 0.0`, `plast_weight 1.0`, `slot_id` logit
   masked, pad slots −100, as today).
6. `L = L_ce + code_fm_weight · L_fm`.

   **Explorative Modeling (2026-09-15, `code_xm_k` > 1, arXiv 2607.27372 Forward XM):** from
   phase 2 on, each step draws K samples per slot (`code_rollout_steps` Euler steps each,
   no grad), scores each against the data — `l2`: squared error of the rms-normed sample
   to E's code; `coda`: the coda's summed CE on the true next span with that sample in the
   cells, dropout off — and keeps the best. The flow loss then runs on the pair (z_0 of the
   kept sample, E's code): the paper's rule, the standard loss on the seed that produced
   the best generation, so the field commits to a mode instead of the mean over seeds. In
   phase 3 the kept sample is what the coda reads at the rollout slots; no second draw.
   Inference is unchanged (one sample, `code_infer_steps`). Stats: `code_xm_score_mean`,
   `code_xm_score_best`. `_code_last_passes` counts K × `code_rollout_steps`.

   **Tape rollout (2026-09-16, `code_tape_rollout_p`, LaDiR's reasoning-model stage 2):**
   LaDiR's second stage does not hand samples to the decoder; it conditions the reasoning
   model on its OWN generated latents for the earlier blocks while the flow target stays
   the oracle latent. Here: on a fraction `code_tape_rollout_p` of rows the thinker's
   clean context tape is replaced by the thinker's sampled codes for every valid slot
   (one parallel `code_rollout_steps`-step draw conditioned on the truth tape, no grad;
   the paper's sequential draw would cost S × k passes) and the flow pair is unchanged.
   Stat `code_tape_rollout_frac`; `_code_last_passes` adds `code_rollout_steps`. This is
   the third stage of the LaDiR chain (`tul_code_vae` → `tul_code_ladir_tf` →
   `tul_code_ladir_ro`), in which the coda is frozen from stage 2 on and never reads a
   sample in training; `code_rollout_p` stays 0 there.

   **The noise-search form (2026-09-16, `code_xm_mode: noise`, the paper's Diffusion/Flow
   hybrid, App. C):** no generation. The flow block draws its t and condition once, then K
   corruption noises z_0 for the same target; each candidate is ONE velocity prediction
   at that t, scored by the flow loss itself (the paper's rule: the selection criterion is
   the training loss); the lowest pair is re-run with grad (the memory-saving mode). K
   no-grad thinker passes per step, `_code_last_passes` = 1 + K + the rollout passes; the
   phase-3 rollout is a fresh sample as on the parent. `code_xm_select` must be `l2`
   (nothing is generated for the coda to score). The paper calls this a coupling search
   over the noise rather than a best-of-K against a fixed target.

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
| 1 define the code | 0 | `L_ce` only. E and the mouth co-train; the thinker's pass does NOT run (no forward, no loss, no cost) |
| 2 learn to guess it | `tul.code_phase2_at` (default 0.10) | `L_fm` on. The code is a slow target: E keeps training through CE |
| 3 rollout | `tul.code_phase3_at` (default 0.50) | `code_rollout_p` (default 0.5) of valid slots hand the coda a sampled `ẑ` at `code_rollout_steps` (default 8) instead of the code |

LaDiR's stage 2 is load-bearing there (MATH 30.7 without, 45.2 with). The defaults above
are placeholders; the smoke and the first arm set them. The phase switches are step-based
like `mux_activate_at` / `sigreg_activate_at` (`morph/training/train.py`), resolved once
and written to the wandb manifest.

A phase-1 checkpoint cannot generate (E needs the span it has not written). The first
checkpoint that is a model is the first one past `code_phase2_at`.

The phase switches are Python-level branches read at trace time (phase 1 runs no thinker
pass at all; phase 3 runs a `no_grad` sampler), NOT the `mux_gate` multiply-by-zero
buffers: a gate would pay the full thinker and sampler cost in every phase. Cost: two
`torch.compile` recompiles per run, at the two switches. Named here so nobody "fixes" it.

## 7. Inference

`morph/inference/tul_generate.py` gains one branch (BUILT 2026-09-14, `code_mode:
"generate"`). At a boundary the open span's slot gets ONE sample: `z ← z_0 ~ N(0,
source_std²)`; for `j = 0 .. k−1`: `z ← z + (1/k)·v(z, t_j, tape, seed)` with `t_j = j/k`
(Euler, the same integrator `fm_planner` uses); the generator caches `rmsnorm(z)` and
hands it back through `code_given` on every recompute until the next boundary, so a span
is written from one code. The TAPE at generation is the RE-ENCODED past: a slot whose span
is already written holds E's code of that text, not the sample it was written from. This
is the third option of §13.4 and the one built, because the text exists and E is cheap;
the fully sampled tape remains the `ce_k8_rolled` instrument. Then tokens as today.

`k` is `tul.code_infer_steps` at generation and a per-call argument in the instruments.
Cost of k, per token at ⟨span⟩ ≈ 12 and M = 2: `6·k·2/12` core block passes per position,
so k = 8 is 8 against the plain loop's 36 and the break-even in FLOPs is k = 36. Counted as
SEQUENTIAL block launches (decode latency), the plain loop pays 36 per token and the thinker
6·k per span, so that break-even is k = 72. The noise
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
| `val/code_eff_rank`, `val/code_pairwise_cos` | the S·M cells of a row in C = 1024 dims, the SAME computation as `val/slot_eff_rank` (slot family: 5.7–7.3, cos 0.72–0.77) | the collapse instruments |
| `val/ce_k8_rolled` | CE with a FULLY sampled tape: codes sampled slot by slot down the row, each reading the sampled ones before it | the generation regime; the compounding instrument that `ce_k{K}` cannot read |
| `val/ce_marginal`, `val/ce_single_mean` | the K-sample marginal (K = `tul.code_marginal_k`, default 8): −Σ_spans [logsumexp_k LL_k(span) − log K] / n_tokens with LL_k the span's log-likelihood under sampled code k (seed k, `code_infer_steps` passes, the encoder's codes elsewhere), and the K-average of one-sample CE beside it | a latent-variable LM is owed the marginal over its draws, not one draw's CE (added 2026-09-14 after the 5k panel: one draw is mostly a different sentence). A lower bound that tightens with K; `ce_single_mean − ce_marginal` says how much the draws differ. `morph/training/code_eval.py` |
| `val/ce_k8` by span length (buckets 4–7, 8–15, 16–32) | the same number, split | whether M = 2 cells serve a 32-token span as well as an 8-token one |
| `fm/rel` per t-band | `L_fm / null` | the thinker's honesty instrument (`fm_planner._finish`) |
| `val/layer_passes_per_token` | code-aware branch of `_tul_layer_passes` | the compute column of §12, measured not tabulated |
| `worth_profile` shuffle cost on the cells | `ce(shuffled cells) − ce` at K 8 | the reader instrument; must be positive (FM doctrine rule 1: cost, not fraction) |
| generation at K ∈ {1, 8}: gen-PPL with rep4 / distinct-3 | the diversity guard, doctrine rule 11 | |

Instrument plumbing the map requires: `code_fm_weighted` joins the `*_weighted`
subtraction list in `evaluate()` so `val/loss` stays the CE; `tul_slot_state_probe` gains
a code branch reading E's cells (it runs `_tul_core` today, which a code model does not
have); `plan_mode="wrong_seed"` is REFUSED on a code model (the seed feeds the thinker,
not the cells, so the reading would mean something else); `zero` / `shuffle` /
`all_slots` act on the cells as today.

Controls, one seed each in the first panel, two before any claim (doctrine rule 6):

* `slot-spandec-strict` at the same commit (the deterministic loop write on the same
  geometry; its cells are a mean);
* the `d1` rung (`notul_norm_match_20k_d1`, 12 block passes per token) — the compute bar;
* `notul` plain looped (42 per token) — the ceiling on CE.

Pre-registered gates, to be frozen in `lab/experiments/planned/2026-09-14-arc-tul-code.md`
BEFORE the first run (numbers here are the proposal; the prereg owns them):

* **G1 (the code is a code, and it is read).** A SANITY gate, expected to pass: `ce_tf`
  contains the span being scored, so it should sit well below the strict ruler; the gate is
  `ce_tf` at least 0.15 nats below the ruler at 5k on paired rows, the shuffle cost of the
  cells (cells permuted across a row's slots, `ce_tf` regime) at least 0.15, and
  `code_eff_rank ≥ 16` of 1024 (the slot family reads 5.7–7.3). Fails ⇒ the mouth is not
  reading the cells or E writes a constant; nothing downstream is readable.
* **G2 (the loop earns by sampling).** `ce_k1 − ce_k8 ≥ 0.02` at 5k with `fm/rel` below 0.9
  in every band. Fails ⇒ the thinker is not integrating anything; the line stops.
* **G3 (a sample beats a mean).** `ce_k8` (teacher-forced tape) below the strict ruler's CE
  on paired rows at 20k, CI excluding 0; `ce_k8_rolled` reported beside it and within 0.05
  of `ce_k8`, or the compounding is the next problem. Fails ⇒ multimodality does not pay on this data; the line stops.
* **G4 (ship bar).** `ce_k8` within 0.02 of the `d1` rung at 20k, at fewer block passes per
  token than `d1`.

A 5k CE ranks nothing between arms within 0.01 (standing rule); G1 and G2 are within-run
readings and are readable at 5k; G3 and G4 are 20k readings.

## 9. Config — the `tul.code_*` block (all construction-time; unknown keys raise)

```yaml
tul:
  code: false                # true builds E, W_v, the time MLP; false is bit-identical to the ruler
  code_noise: 0.5            # UNTUNED. Gaussian noise on the cells at train (LaDiR k)
  code_cfg_drop: 0.0         # CFG: per-row probability of the null condition on the thinker pass
  code_cfg_scale: 1.0        # CFG guidance at sampling (needs code_cfg_drop > 0); 1 = off
  code_target_lambda: 0.0    # the flow gradient reaches E at this weight (0 = C4 stop-gradient)
  code_rank_abort: 0.0       # trainer raises when val/code_eff_rank < this; 0 = off
  code_xm_k: 1               # Explorative Modeling: K samples per slot per step, train on the
                             # nearest; 1 = off (one draw, no selection)
  code_xm_select: l2         # "l2" (nearest to E's code, the paper) | "coda" (lowest coda CE
                             # on the true next span; K extra no-grad coda passes)
  code_tape_rollout_p: 0.0   # LaDiR reasoning stage 2: fraction of rows whose thinker CONTEXT is
                             # its own sampled tape (code_rollout_steps, no grad); target = E's code
  code_xm_mode: sample       # "sample": K full generations, select the endpoint (Algorithm 1)
                             # | "noise": the Diffusion/Flow hybrid, K corruption noises at one
                             # t scored by the flow loss (K thinker passes, no generation)
  code_sigreg_lambda: 0.0    # LeJEPA SIGReg on E's code cells (per cell, valid slots), weight in
                             # the loss; with code_target_lambda 1.0 this is LeJEPA on the code
                             # (arXiv 2511.08544: lambda 0.05, sigreg_slices 1024); 0 = off
  code_noise_renorm: false   # rms-renorm the noisy truth cell (RMS 1, like a sampled cell).
                             # false = legacy: truth RMS sqrt(1+noise²); eval's encoder mode
                             # and the generator's tape feed z at that scale (2026-09-15)
  code_norm: rms             # the only mode in v0.1
  code_fm_weight: 1.0        # with loss_scale auto (FM1's setting)
  code_source_std: 1.0       # matched to unit-scale codes (doctrine §7)
  code_t_embed_scale: 1.0    # fm_planner's knob, same meaning
  code_phase2_at: 0.10       # UNTUNED fraction of training.steps
  code_phase3_at: 0.50       # UNTUNED
  code_rollout_p: 0.5        # UNTUNED fraction of valid slots given a sampled code in phase 3
  code_rollout_steps: 8      # Euler steps for phase-3 samples (LaDiR: 50 -> 10)
  code_infer_steps: 8        # k at generation; the instruments sweep it
  code_marginal_k: 8         # K draws behind val/ce_marginal (0 = off)
```

Required by construction, refused otherwise: `tokens_through_core: false`,
`tg_geometry: strict`, `spandec: false`, `slot_cells: 1` (`slot_cells` is the REGISTER's knob and stays 1; the
code's cell count M is `prefix_k`, and E writes the cells, not the register), `vq_codes: 0`, `oracle_z: false`, `grad_pass: false`, `fm: null` (the FM1
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

## 12. Compute (block passes counted per position; measured rates come from the smoke)

| model | per token | per span (⟨span⟩ ≈ 12 at seq 1024) |
| --- | --- | --- |
| plain looped, mean 6 | 3 + 36 + 3 = 42 | — |
| `d1` rung | 12 | — |
| strict slot loop, mean 6 | 6 (+ 36 on the slot positions only) | 36 |
| TUL-Code train | 6 (+ E) | 6·2M, once (M = 2: 24) |
| TUL-Code infer, k | 6 | 6·k·M (M = 2: 12k) |

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
4. **Where the tape's context copies come from at inference.** BUILT: the re-encoded
   past (E on the text already written) with one sample for the open span (§7).
   Alternatives: a tape of the samples themselves (the `ce_k8_rolled` instrument reads
   that regime), or the mean code for old cells.

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


## 16. LCTUL-D — the discrete code and the masked denoiser (2026-09-16)

Why (measured, `lab/experiments/results/2026-09-16-lctul-context-blind-probe/`): on
`tul-code-cfg` at 20k the flow loss moves 0.3125 → 0.3161 of the null floor when the past
is removed (the trained CFG null condition on every row). The velocity field's loss is
99 % noise removal and the marginal's shape; the conditional `p(code | past)` gets 1 % of
the gradient. And (proved, `lab/theory/lctul_euler_depth/`): k = 1 Euler is the
conditional mean, and on a Gaussian-like conditional every further step changes one
scalar. Neither the sampler nor the coda schedule can move that. §16 changes the code.

`tul.code_discrete: true` keeps §1–§15 and replaces three objects:

* **The code is N = `prefix_k` · `code_vq_groups` symbols** from one cosine codebook of
  `code_vq_codebook` rows (`TULThoughtVQ`, `morph/model/tul_vq.py`, on E's pooled
  vector, `codes = prefix_k`, `groups = code_vq_groups`). The coda reads
  `rmsnorm(lift(symbols))` in the M cells (the quantiser's lift, one map per cell), with the
  straight-through gradient into E and the VQ-VAE terms (`vq`, `vq_weighted` at
  `code_vq_weight`, `vq_perplexity`) beside the CE. Rate control is LaDiR's token
  substitution: at train `code_sub_p` of the valid symbols the coda reads are replaced by
  a uniform random symbol (gradient cut at those symbols; the denoiser's target is always
  E's own symbols). `code_noise` must be 0.
* **The thinker is a masked denoiser.** Same doubled slot sequence and relation with
  `M := N` symbol positions per copy: a noisy copy holds the symbol embedding or the MASK
  embedding (`tul_code_sym`, row C = MASK, unit per-component scale), plus the
  per-position embedding and the time embedding of the slot's masked fraction; a clean
  copy the (context) symbol embedding plus the clean marker. Head `tul_code_sym_head`:
  private RMSNorm, zero-init linear to C logits (uniform at step 0). Loss = the
  masked-diffusion ELBO (MDLM / LLaDA): `t ~ U(1e-3, 1)` per slot, each symbol masked
  with probability `t`, `(1/t) · Σ_masked CE` in nats per span, an upper bound on
  `−log p(code | past)`; a slot that draws no mask contributes 0 (no forced mask: unbiased).
  Reported through the flow keys in units of the uniform floor `N · log C`
  (`code_fm_rel` = 1.0 at the zero head) plus `code_mdm_nats` and `code_mask_frac`. The
  loss IS the conditional entropy of the code given the past: 100 % of its gradient is
  about the past.
* **The sampler is k rounds of parallel unmasking** (MaskGIT without the Gumbel term):
  start all-MASK, predict every masked symbol, sample it, commit the most confident so
  that `mdm_unmask_counts(N, k, code_mask_schedule)` symbols are set after round j.
  `k = 1` is one-shot sampling (a REAL sample, each symbol from its marginal given the
  past); `k = N` commits one symbol per round; `k > N` is refused. Depth here resolves
  the dependence among a span's symbols, and the k-curve reads it directly. `code_infer_steps`
  / `code_rollout_steps` are rounds. CFG: the null condition MASKs the tape, guidance
  acts on the logits.

Everything else is unchanged in interface: `_tul_code_sample` takes the SYMBOL tape
(`[B, S, N]` long) and returns lifted cells at the coda's statistic, so `code_marginal_ce`,
the eval modes (`encoder` / `sampled` / `rolled` / `generate`), the generator's `code_given`
cache and the probes run as they are. Not defined on a discrete code (refused): `code_xm_k >
1`, `code_sigreg_lambda`, `code_target_lambda`, `code_noise_renorm`, `code_source_std ≠ 1`.

Tests: `tests/test_tul_code_d.py` (D1 build and RNG-neutrality; D2 truth cells =
rmsnorm(lift(index)), STE into E; D3 substitution touches the coda input only and cuts the
gradient; D4 the ELBO scores masked symbols at 1/t and the zero head reads the floor; D5
the schedule and the sampler, seeded, k = 1 = one-shot; D6 causality; D7 the losses share
nothing; D8 null condition and guidance; D9 refusals; D10 the arm composes; D11/D12 eval
modes, pass counts, generation, the marginal). Config: `tul_code_d.yaml`. Note:
`.agents/notes/proposed/architecture/2026-09-16-lctul-d-discrete-code-masked-denoiser.md`.

## 17. The code TARGET — LaDiR with the slot loop as the latent generator (2026-09-17)

Why (measured, `lab/experiments/failures/2026-09-16-lctul-{d-first-arm,dplan-semantic,ladir-chain}.md`):
every SAMPLED next-span code reads as a foreign sample on generation (own − foreign +0.008
to −0.006, CI spanning 0, on the flow, masked-denoiser and LaDiR thinkers), while every
reader decodes the TRUE code well (the LaDiR VAE stage: 1.34 nats teacher-forced, oracle
cosine 0.64). Wolfe, 2026-09-17: "we can train TUL latent to match oracle. Its impossible
for it to carry nothing about the span." A latent REGRESSED onto the code carries its
predictable part, at least the 0.40 nats/token cross-span budget; the earlier
future-prediction targets that read flat were span-mean embeddings sitting within 0.01
cosine of each other (`horizon-targets-degenerate-on-web-text`). The VAE code is rank 57.

`tul.code_target: true` keeps the ORDINARY strict slot loop (`_tul_core`, every loop knob
live, the gain constraint on) and changes three things:

- **The write.** `TULCodeProj` (`morph/model/tul_code.py`) maps the loop's readout of a
  slot's exit state to M = `prefix_k` unit-RMS cells (`W_code` identity-init, `b_code`
  zero, no RNG draw, ternary-excluded), masked by E's `ok`. THOSE cells are scattered into
  the prefix positions (`TULSlots.prefix_positions`); `W_prefix` is built and inert (§3.2's
  precedent). `_forward_tul` → `_tul_code_target_write`.
- **The target.** E (`TULCodeEncoder`) is built FROZEN (`requires_grad` off at build, no
  `tul_code_enc.` prefix in `training.train_only`), loaded from the VAE stage's checkpoint
  (`tul-code-vae/step_10000.pt`) with the thinker's tensors dropped loudly
  (`train.py::drop_code_thinker_keys`), and run under `no_grad` on the forward's own prelude
  output. The term: `mean over valid cells of ||pred − z||² / C = 2 (1 − cos)`
  (`code_target_regression`), weight `code_target_weight`, exposed as `code_target` /
  `code_target_weighted` (train.py subtracts the weighted term from the reported loss, the
  `spandec_weighted` contract).
- **The read.** `code_target_detach: true` (arm A, `tul_code_target.yaml`): the coda reads
  the predicted cells with stop-gradient, so the loop learns from the regression ALONE
  (LaDiR: the decoder never trains on a generated latent). `false` (arm B,
  `tul_code_target_ce.yaml`): the token CE through the frozen coda trains the loop too.

Instruments. `code_target_cos` (exit cell cosine to the code, train and eval);
`code_target_cos_l{t}` (train only, no_grad): the entry state `core_init(e)` at `l0` and
the state after every realised pass, projected and read against the SAME code — whether
the passes MOVE toward the code. Eval: `code_mode="encoder"` hands the coda E's own code
(`val/ce_tf`, the ceiling); `code_mode="generate"` + `code_given` overrides slots (the
generator's per-span cache; the probe's SHUF / ZERO / ORACLE); `plan_mode` zero / shuffle
act on the CELLS; `slot_depths` forces the loop depth (the K-curve); `code_steps` /
`code_seed` / the sampler modes are refused (no sampler). `code_semantic_probe.py --kind
target` scores OWN / SHUF / ZERO / ORACLE on the 120 cuts.

Refusals (`TULConfig._check_code_target`): with `tul.code`; the paid loop /
`loop_reads_tokens`; non-strict geometry; a cut coda; `slot_cells > 1` / `vq_codes` /
`prefix_source != exit`; `detach_z` (one knob: `code_target_detach`); `bcast` /
`spandec_reads_cells`; `code_target_*` set with `code_target: false`. Model: `n_core == 0`;
an FM planner.

THE RULE. This regresses the slot state onto an external vector, which root `CLAUDE.md`
forbids (LCM T3/4, CoCoMix §6b, BT §4.2); `oracle_z` carved the one earlier exception.
Wolfe directed this arm on 2026-09-17, and the note
`.agents/notes/proposed/architecture/2026-09-17-lctul-target-slot-loop.md` records the
exception. Tests: `tests/test_tul_code_target.py` (twelve, one per invariant above).

### 17.1 The code-ONLY arm: no coda at train, everything trains, and a discriminative term (2026-09-17)

Arm A's reading at 10k (`lab/experiments/results/2026-09-17-lctul-target/`): the exit
cosine reached 0.14 by step 1500 and 0.17 at 10k; pass 1 does all of it (l0 0.06, l1
0.135, l6 0.132); the predicted cell is a real conditional prediction (own 0.145 vs
shuffled 0.021, centred own 0.135 vs 0.002) that leans on the corpus-mean direction (cos
0.33-0.53 vs the codes' 0.06-0.09) and lives in an effective rank of 16 against the codes'
75 (`lab/divergence/code_target_mean_probe.py`); the frozen coda reads it at 9.1 nats,
worse than no cell (7.4). Wolfe: "Poisson loop, the loop guesses the code. We never even
run the coda." Three knobs, one arm family (`tul_code_only.yaml`, `tul_code_only_nce.yaml`):

- **`code_target_skip_coda`.** At TRAIN (`self.training` and labels given) `_forward_tul`
  ends at `_tul_code_target_write`: no token dropout, no `_back_region`, no token CE.
  `groups["loss"]` starts at an exact 0, so the loss is the code term plus the slot-loop
  constraint (`gain_reg_weighted`) and nothing else; train.py's subtraction of
  `code_target_weighted` reports the constraint alone as `train/loss`. The EVAL forward is
  the ordinary one (the frozen-at-VAE coda reads the cells: `val/loss`, `val/ce_tf`, the
  probes' OWN / SHUF / ZERO / ORACLE). Refused with `code_target_detach: false` (the CE
  route needs a coda) and with weight 0 (nothing would train).
- **`training.train_only: []`** on these configs: the prelude, embeddings and the whole
  loop train. Arm A froze the prelude at the VAE stage, where it learned to serve a coda
  reading noised TRUE codes and never to represent the past for prediction; the
  predictable subspace the loop can find is bounded by what that prelude hands it. E stays
  frozen at build (the target); the coda and the untied heads get no gradient because no
  coda runs at train.
- **`code_target_loss: "infonce"`** (`code_target_tau`, default 0.1): per cell, the
  predicted cells of the batch's valid slots score against EVERY valid slot's code
  (`<p_i, z_j> / C / tau`) with the own code as the class (`code_target_infonce`). L2's
  optimum is the conditional MEAN, which hedges toward the corpus mean; InfoNCE asks the
  loop to pick its span out of the batch (~380 codes at batch 6) and cannot hedge.
  Readings: `code_target_acc` (top-1; chance ~1/n_valid) beside the L2 `code_target_mse`.

Instrument added on every code-target arm: `code_target_cos_shuf`, the predicted cells
against the valid slots' codes rolled by half the valid count (a deterministic cross-row
pairing). Own minus this is what the cell knows about ITS span; the offline twin is the
corpus-mean probe. Prereg `lab/experiments/planned/2026-09-17-lctul-code-only.md`.

**THE TARGET COLLAPSED, and `tul.code_target_ref` is the fix (2026-09-17, measured).** The
first code-only draw was killed at step 3500 (wandb `38naddpq`). `E` is frozen; its INPUT
is not. `E` pools the LIVE prelude's states of the next span, so with `train_only: []` the
prelude drifted under the loop's gradient and `E`'s codes collapsed onto one direction:
train own cosine 0.61 / shuffled 0.56 at steps 500-1000, 0.99 / 0.98 by 2500-3000 (val
0.989 / 0.984), regression loss 0.016, the frozen coda's oracle CE 13.1 nats against arm
A's 1.4. No gradient ever reached `E` — it runs under `no_grad` — so this is drift of its
INPUT, not an optimised collapse. Either way the target measures nothing and the draw is
void, not a result.

`tul.code_target_ref: true` holds a FROZEN DEEP COPY of the whole model, snapshotted right
after `training.init_from` / `training.resume` loads weights, and EVERY reading of the VAE
stage comes from it: the prelude states `E` pools (the twin runs its own `_tul_front` with
its own TG kwargs — `instruments-must-use-the-models-tg-kwargs`), `E` itself, the tied head
the grader scores through, and on §17.2 the coda that samples the continuations and the
coda-with-zero-cell that grades them. The live model contributes the loop's predicted cell
and nothing else.

The twin is deliberately NOT a registered submodule. Every walk in this tree enumerates
modules or parameters — the ternary QAT pass, the embedding QAT, the CMS prune / carve /
route walk, the optimizer, the gradient probes — and a registered 270M-parameter twin would
silently double all of them. It lives on `MORPHTransformer._code_ref` (read through
`model.code_ref`), is built by `tul_code_ref_snapshot()` AFTER quantisation and AFTER the
weights load so it copies the parametrised model exactly, and rides its own `code_ref` key
in the checkpoint. On a RESUME that key wins: re-snapshotting the live weights at step N
would move the target mid-run, which is the failure the mechanism exists to prevent. A
checkpoint without the key (every `init_from` seed, the VAE stage included) is the snapshot
case and says so in the log. Cost: one extra fp32 copy of the model (~1.1 GB at 270M) plus
one `no_grad` prelude forward per step.

The trainer REFUSES a `tul.code_target` model whose front trains without the twin
(`assert_code_target_front_frozen`: any trainable `embed.` / `prelude.` / `input_norm.` /
`value_embed*` / prelude-range `x0_injects.`), with the collapse numbers in the message, so
this cannot happen again. `tul_code_only.yaml` and `tul_code_only_nce.yaml` carry
`code_target_ref: true` and keep `train_only: []`. Tests: `tests/test_tul_code_ref.py`.

### 17.2 The GRADED-continuation target: the loop proposes, a blind grader ranks (2026-09-17)

Every target in §17 / §17.1 is a deterministic function of the past, and every one is met
in ONE pass (arm A at 10k: `cos_l0` 0.06, `l1` 0.135, `l6` 0.132; the exit cosine flat at
0.17 since step 1500; a rank-16 cell against the codes' 75). A conditional mean is a POINT,
the map's first pass sets the scale and the rest rotate
(`morph-loop-is-a-power-iteration`), so nothing about a point pays for a second pass. The
code-only prereg names this exit itself: "P-C6 fails on both → the depth question moves off
predicted targets entirely (a target that is computed, not predicted)". Wolfe, 2026-09-17:
"didn't we also have an arm that was grading plausible continuations with a grader?"

`tul.code_grade: true` keeps everything §17 sets up (strict slot loop, `TULCodeProj` cells,
frozen `E`, frozen VAE coda) and adds a term whose target is COMPUTED. It requires
`tul.code_target_ref` in practice and its configs set it: the sampler's coda, the grader's
coda, the tied head both read through and `E` on the candidate rows all come from the frozen
twin (§17.1), so nothing the live front does can move the target or the judge.

- **The proposal.** The loop's predicted cells, DETACHED, condition the frozen coda, which
  samples `code_grade_k` candidate continuations of span `s+1` at `code_grade_temp`. The
  candidate fills the TRUE span's token window: at most `code_grade_tokens` tokens, and a
  slot whose span is longer than that is not graded (`E` pools the whole span, so a partly
  substituted span would mix the candidate with the truth). The first candidate token comes
  from the boundary TOKEN position — the only trained emit head at `emit_weight: 0` — and
  is cell-blind by the strict geometry; the rest are cell-conditioned.
- **The grade.** Mean log-probability per candidate token under a grader that CANNOT see
  span `s+1`. `code_grade_grader: "coda_past"` (default) is the frozen coda reading `E`'s
  true codes at every slot with the graded slot's OWN cell zeroed, run in two passes over
  the slot-index parity (one pass cannot both zero cell `s` and keep it as context for span
  `s+2`). `"coda_zero"` zeroes every cell, one pass, context blind under the strict
  geometry: the fluency control. A candidate whose distinct-2 falls below
  `code_grade_min_distinct2` is given the worst grade (`genppl-needs-a-diversity-guard`).
  The grader is NOT the proposing coda with its own cell: that argmax is the cell's own
  greedy decode, a fixed point that teaches nothing — `disc`'s failure in a new costume.
- **The term.** `code_grade_loss: "best"` regresses the predicted cells onto `E(best)`
  (`2(1 - cos)` per cell, `code_target_regression`); `"pref"` (default) is an InfoNCE over
  the `K` candidate codes with the winner as the class at `code_grade_tau`, which pushes
  toward the winner and away from the losers. Weight `code_grade_weight`, exposed as
  `code_grade` / `code_grade_weighted` with the `code_target_weighted` subtraction
  contract. The ordinary `code_target` term may stay on at its own weight.

**Why this is affordable, and it only is under STRICT geometry.** `tg_strict_allow` makes
the prelude span-local and lets a coda token read its own span plus EARLIER prefix cells,
with the conv, the value shift and the retention carry reset per segment. Substituting
candidate tokens into span `s+1` therefore changes the model only at span `s+1`'s own
positions, so ONE forward decodes a candidate token for EVERY span of a row at once and the
spans cannot contaminate each other. The graded-slot count is FREE; the cost levers are
`code_grade_k`, `code_grade_tokens` and `code_grade_rows`, amortised by
`code_grade_every`. Arithmetic in block-position units at `B` 6, `L` 1024, 64 slots: a
code-only training step is ~115,200 u with its backward; one sampling forward over one row
is 11,264 u; a graded step is `(J + 2) · rows · K` such passes = 811,008 u at `J` 16,
`rows` 1, `K` 4 — 7.0 training steps, or 0.53× the rate at `code_grade_every: 8`. The real
generator (recompute-per-token over the whole row) would be ~36× a training step and is not
an option at any setting.

**Instruments** (`no_grad`, graded steps only, through the `code_target*` log whitelists):
`code_grade_best` / `_mean` / `_worst` (mean log-prob per token of the best / all / worst
candidates), `code_grade_true` and `code_grade_true_rank` (the TRUE next span under the SAME
grader, and the fraction of non-degenerate candidates it outscores — the grader's sanity
reading. Read the RANK: a self-grading design, where the candidates are drawn and scored
under the same cell, puts the truth below the sample mean by Jensen whatever the grader is
worth, so the difference means nothing there. This grader zeroes the cell the sampler used,
the two distributions differ, and the 2026-09-17 Spark smoke reads truth −4.09 against best
−5.00 / mean −6.13 / worst −7.40, rank 0.89, at the VAE checkpoint), `code_grade_cos_best_true` and
`code_grade_cos_worst_true` (does winning the grade mean being nearer the truth's code?),
`code_grade_cos_l{t}` (the per-pass cosines to `E(best)`, beside `code_target_cos_l{t}` to
`E(true)` — the depth question), `code_grade_degen` (flagged fraction), `code_grade_n`
(graded slots) and `code_grade_slot_frac` (graded / valid, the coverage of the length
filter).

**Refusals** (`TULConfig._check_code_grade`): `code_grade_*` set with `code_grade: false`;
`code_grade` without `code_target`; `code_grade_k < 2`; `code_grade_tokens < 2`;
`code_grade_rows < 1`; `code_grade_every < 1`; `code_grade_weight <= 0`;
`code_grade_temp <= 0`; `code_grade_tau <= 0`; an unknown `code_grade_loss` or
`code_grade_grader`; `code_grade_detach: false` (the proposal must not be a gradient path —
sampling is not differentiable and the whole term runs under `no_grad` up to the target).
Configs `tul_code_grade.yaml` (the `code_target` L2 term OFF) and `tul_code_grade_l2.yaml`
(it ON at 1.0), both composing `tul_code_only`. Note
`.agents/notes/proposed/architecture/2026-09-17-lctul-graded-continuation-target.md`;
prereg `lab/experiments/planned/2026-09-17-lctul-graded-target.md`; tests
`tests/test_tul_code_grade.py`.
