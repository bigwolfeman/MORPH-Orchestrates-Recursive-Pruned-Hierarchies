# Agent Note: Stp Prediction 2

Status: implemented

Origin: Ai-notes/06-23-2026/STP-AB-Results/PREDICTION2.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# STP-in-SFT crossover (prediction 2) — Results

**Date:** 2026-06-23 ~02:30 CDT
**Question (Wolfe's prediction 2):** "[STP pretraining] will be more repetitive than without.
However if we SFT on just a few hundred instruction-follow prompts I bet that **totally
changes** [it]." Paired aside: "even minor instruction following won't save it from how dumb it is."
**Design:** both pretrained bases (prediction-1 ckpts `tst_stp_off_50k`, `tst_stp_on_50k`,
both at step 50000) → **STP-SFT** (the paper's intended STP use) on a ~512-example Dolly-15k
subset, identical config except the init ckpt. Then probe generation on (a) the same OWT
continuation benchmark as prediction 1 and (b) held-out instruction prompts.

## Verdict — prediction 2 REFUTED (for this recipe); pessimistic aside CONFIRMED
**Light instruction SFT did not "totally change" the repetition for the better — it made free
generation categorically WORSE.** Both arms went from "dumb but coherent English" (base) to
**degenerate token loops** (`"...."`, `"------"`, `"CCCC"`, `"ResourcesResources"`) on BOTH
OWT continuations AND held-out instructions, in BOTH greedy and sampled decode. The un-tuned
base models generate *more* coherently and stop *more* often than the SFT'd models.

This is a real over-specialization effect, not a harness bug — it is **dose-responsive**: the
arm that fit the SFT data harder collapsed harder (see below).

## SFT training (both arms clean, paired)
| arm | init STP loss | final SFT loss | final ppl (TF) |
|---|---|---|---|
| base_off → STP-SFT | **1.467** | 1.40 | 4.1 |
| base_on → STP-SFT  | **0.644** | 2.08 | 8.0 |

- 42 optimizer steps (2 epochs × 512 examples, packed seq_len=1024, batch 4), flat LR 2e-5,
  AdamW8bit, stp_lambda=0.02 both arms, dropout 0.1, reseeded for paired RNG.
- **Mechanistic signal:** the STP-pretrained base enters SFT with a **2.3× geodesically smoother**
  hidden trajectory (STP loss 0.644 vs 1.467) and the gap persists through SFT — STP pretraining
  "took" and is not erased by a light fine-tune.
- Teacher-forced loss descended cleanly (10.3 → 1.4/2.1). **Training worked; free generation broke.**

## (a) OWT continuation — apples-to-apples vs prediction 1 (sampled, temp0.8/top50)
| metric | off-base | off-SFT | on-base | on-SFT |
|---|---|---|---|---|
| rep-2 (↑ = worse) | 0.138 | **0.951** | 0.214 | **0.895** |
| rep-3 | 0.070 | 0.938 | 0.138 | 0.862 |
| distinct-2 (↓ = worse) | 0.862 | 0.049 | 0.786 | 0.105 |
| entropy (↓ = worse) | 2.072 | 0.202 | 1.946 | 0.412 |
| loop_frac (↑ = worse) | 0.067 | 0.913 | 0.074 | 0.803 |
| ARC-easy acc | 0.39 | 0.29 | 0.34 | 0.29 |

SFT degenerates the OWT continuation ~7× on rep-2. Verified from raw text: off-SFT greedy on a
recipe prompt emits `"lowerlowerlower…"` (a single repeated token); the base emits flawed-but-real
prose. This is **format specialization**: 42 steps on the Alpaca template made non-instruction
prompts out-of-distribution → collapse.

## (b) Held-out instruction prompts (5 hand-written + 8 deep-shuffle Dolly, NOT in train set)
**EOS-stop rate (does the model stop?) — higher = better:**
| | greedy | sampled |
|---|---|---|
| off-base | 2/13 | **8/13** |
| on-base  | 0/13 | 3/13 |
| off-SFT  | 0/13 | 0/13 |
| on-SFT   | 1/13 | 3/13 |

**Response content (sampled):**
- **base** models: dumb, off-topic, repetitive — but real English. e.g. "What is the capital of
  France?" → off-base: *"From France To France … From France North"* (stops); on-base: *"…a detailed
  overview of the capital of France…"*.
- **SFT** models: non-language token loops in every case. e.g. → off-SFT: `"...................."`;
  on-SFT: `".....CCCCCCCCCC…"`. Same on greedy.

The SFT'd models are **more degenerate than the un-tuned bases** on the very instruction task SFT
was meant to help, and stop less often. Sampling does not rescue them (it rescued the bases).

## Dose-response (why this is over-specialization, not a bug)
A separate epochs=8 probe (discarded, but logged) drove TF loss to **0.077 / ppl 1.1** (verbatim
memorization of the 512 responses). At the kept epochs=2 point: the arm that fit harder
(**off-SFT, ppl 4.1**) is *more* collapsed in free generation than the less-fit arm
(**on-SFT, ppl 8.0**): 0/13 vs 3/13 sampled EOS, OWT rep-2 0.951 vs 0.895. **More SFT fitting →
more free-generation collapse.** A bug would not scale monotonically with fit; over-specialization
on tiny data does.

## Paired off-vs-on (the literal prediction-2 question: does the gap flip/vanish?)
Prediction 1 ordering was off **better** than on (less repetitive). After STP-SFT, **both collapse**;
the gap neither cleanly persists nor flips — SFT-induced degeneration dominates, with the
STP-pretrained arm (on) **marginally more robust** (OWT rep-2 0.895 < 0.951; 3/13 vs 0/13 sampled
EOS), weakly consistent with its smoother hidden trajectory resisting collapse a little better.

## Calibration / what was NOT tested (no-theater)
- **Single recipe.** 42 steps, packed stream, flat LR 2e-5, AdamW8bit, 512 examples. Gentler
  (lower LR, fewer effective updates), longer-with-more-data, or **unpacked** SFT are UNTESTED.
  The collapse is plausibly avoidable with a more careful recipe; this run shows the *light*
  recipe Wolfe described destabilizes this model, NOT that "SFT can never help MORPH."
- **No-STP control NOT run.** Both arms had stp_lambda=0.02; a plain-SFT (stp=0) control would
  isolate whether STP-in-SFT specifically contributes to the collapse vs SFT alone. (Cheap follow-up.)
- **Model is genuinely weak** (276M, ternary, 0.25 density, looped) — even the bases are barely
  coherent. Wolfe's "how dumb it is" prior is well supported; this caps what any SFT can show.
- EOS-stop is low even for bases — the model never had strong stopping behavior to begin with.
- TF loss healthy ⇒ the failure is a **free-generation / exposure-bias** effect, not optimization.

## Suggested next steps (for Wolfe, gated)
1. **Gentler SFT recipe** (lr 5e-6, ~20–30 effective updates, or unpacked one-example-per-seq with
   a proper attention-mask path) to test whether instruction-following emerges without collapse.
2. **Plain-SFT (stp=0) control** on base_off to separate STP-in-SFT from SFT-alone.
3. If the goal is deployable instruction-following, the model likely needs to be **bigger/denser**
   before SFT can land — or SFT folded into pretraining (instruction-mix) rather than as a fragile
   post-hoc pass.

## Artifacts (MORPH repo)
- Harness: `morph/training/sft.py`, `morph/training/sft_data.py`, `morph/configs/sft.yaml`.
- Gate: `ignore/gate_sft_load.py` (GATE_SFT_LOAD_PASS).
- SFT ckpts: `checkpoints/morph/sft_off_stp/step_42.pt`, `checkpoints/morph/sft_on_stp/step_42.pt`.
- OWT bench: `ignore/bench_sft_off.json`, `ignore/bench_sft_on.json` (vs pred-1 `bench_{off,on}_50k.json`).
- Instruction probe: `ignore/sft_instruct_gen.py`, `ignore/ig_{off,on}_{sft,base}.json`.
- wandb: `sft_off_stp`, `sft_on_stp` (project morph, entity adew-me).
