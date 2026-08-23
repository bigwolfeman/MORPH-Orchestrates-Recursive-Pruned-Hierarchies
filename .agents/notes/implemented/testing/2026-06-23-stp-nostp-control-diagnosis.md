# Agent Note: Stp Nostp Control Diagnosis

Status: implemented

Origin: Ai-notes/06-23-2026/STP-AB-Results/NOSTP_CONTROL_DIAGNOSIS.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# No-STP control + STP-recipe diagnosis (follow-up to PREDICTION2)

**Date:** 2026-06-23 ~07:45 CDT
**Trigger :** "kick off a run without STP on both, see if it collapses. While that runs
read the STP paper/code to see what we are doing wrong during SFT. We may not have enough data,
or we may have genuinely lobotomized the model with the pruning and looping… Could also be
related to how you are handling the QAT. Probably it's in the STP algo though?"

## Verdict — STP is EXONERATED. The collapse is SFT-ALONE.

The no-STP control (`model.stp_lambda=0.02→0.0`, every other knob identical) collapses
**identically** to the STP arm. Removing the STP regularizer entirely does NOT prevent the
free-generation collapse. 's lead hypothesis ("probably it's in the STP algo") is **refuted**.

### Single-variable control design
`ignore/run_sft_nostp_pipeline.sh`: same bases (`tst_stp_off_50k`, `tst_stp_on_50k`), same Dolly-512
packed data, same seed/RNG-reseed, same flat-LR 2e-5 / AdamW8bit / 42-step recipe, same reconstructed
routed+carved topology. ONLY `stp_lambda` flips 0.02→0.0. With λ=0 the forward's `stp_lambda*stp_loss`
term contributes exactly zero (raw `stp_loss` still logged, stays flat ~1.44 — confirms it is truly off).

### The control collapses identically (OWT continuation, greedy)
| arm | rep-2 | loop_frac | distinct-2 |
|---|---|---|---|
| off-base | 0.7240 | 0.116 | 0.276 |
| off-STP (λ=.02) | **0.98904132** | 0.991 | 0.0110 |
| **off-NOSTP (λ=0)** | **0.98904132** | 0.991 | 0.0110 |
| on-base | 0.7762 | 0.132 | 0.224 |
| on-STP (λ=.02) | 0.986119 | 0.974 | 0.0139 |
| **on-NOSTP (λ=0)** | 0.986200 | 0.974 | 0.0138 |

STP vs NO-STP match to **~7 significant figures** on the off arm. The λ=0.02 STP term changed
essentially nothing about the collapse.

### Held-out instruction probe — EOS-stop rate (does it stop?)
| arm | greedy /13 | sampled /13 |
|---|---|---|
| off-STP | 0 | 0 |
| **off-NOSTP** | 0 | 0 |
| on-STP | 1 | 3 |
| **on-NOSTP** | 1 | 4 |

Identical pattern with and without STP.

### Raw text (no-theater: verified from generated tokens, not metrics)
Same OWT prompt, only the weights differ (42 steps plain SFT, λ=0):
- **BASE (off):** `" and a bacon-like flavor … but it's not a perfect recipe f"` — flawed but real English.
- **NO-STP SFT (off):** `"lowerlowerlowerlower…"` / `", and, and, and,,,,,,,,,,,"` — pure single-token loop.

Instruction probe (off-NOSTP): every prompt → `"\n...................."` (greedy AND sampled),
`"ResourcesResourcesResources…"`, `"…ccccccccc"`. **EOS-failure single-token looping** — the SAME
degeneration mode as task #164 (LLM-JEPA gen degeneration). NOT a packing "### Instruction" artifact
(the model never spams the template); it collapses to the highest-frequency response token and
cannot emit EOS.

## What we're doing wrong during SFT (paper read: arXiv:2602.22617)

Paper: **"Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA"**, Huang (Atlassian),
LeCun (NYU), Balestriero (Brown), Feb 2026. Code: `github.com/galilai-group/llm-jepa#stp`.

1. **Our recipe is far outside the paper's empirical envelope.** Paper fine-tunes **off-the-shelf
 DENSE full-precision 1B–8B instruct models** (Llama-3.2-1B/3B, Llama-3.1-8B, Gemma-2-2B, Qwen3-1.7B,
 OLMo-1B, …) on **NARROW tasks** (regex synthesis, GSM8K, Spider-SQL, NQ, HellaSwag), ~4 epochs,
 and evaluates **task accuracy only** — no repetition/distinct-n/MAUVE/free-gen metric at all.
 Ours: **276M, ternary-QAT, 0.25-density carved, whole-body-routed, weight-tied looped**, broad
 Dolly instructions, evaluated on **free generation**. They tested ZERO quantized/sparse/looped
 models; their smallest base (~1B dense FP) is ~4× our params and far less compressed. → strongly
 supports 's "too dumb / lobotomized" prior. The collapse lives in a regime the paper never
 probed.

2. **Collapse mechanism = generation-stability / EOS-failure**, not STP and not packing-format.
 Confirmed from raw text above. A short plain-SFT update on this weak/compressed base destabilizes
 the stopping/diversity behavior the base barely had.

3. **Our STP implementation DIVERGES from the paper (moot here, matters later).** Paper STP =
 a SINGLE random triplet `s<r<t` per loss eval: `1 − cos(h_t−h_r, h_r−h_s)`, with `τ` = the
 sequence-length window bound (`|t−s|≤τ`), constant small `λ≈0.01` (sweet spot 0.01–0.08; SYNTH
 used exactly 0.02). Optional **two-view anchoring**: put `s` at query-start, `t` at answer-end,
 `s>0` to skip the system prompt. Paper explicitly: too-high λ → "precipitous drop in accuracy +
 drastic std increase", and "λ ≪ 1 preferred" because real geodesics curve. **Ours** (`prediction.py`):
 a multi-stride SUM over k∈{1,2,4,…,τ=64} of `1−cos(h[t+2k]−h[t+k], h[t+k]−h[t])` averaged over
 ALL positions, on post-`final_norm` states, no two-view anchoring. That is a **much denser/stronger**
 regularizer at the same nominal λ — so our effective STP strength at λ=0.02 is well above the paper's.
 IRRELEVANT to this collapse (λ=0 collapses too) but **must be fixed if we ever want STP-SFT to help**.

## 's four hypotheses, scored
| hypothesis | verdict |
|---|---|
| "probably it's in the STP algo" | ❌ **REFUTED** — λ=0 control collapses identically |
| not enough data (512 ex) | ✅ **likely contributor** — tiny + broad task |
| lobotomized base (prune+loop) | ✅ **likely contributor** — sub-1B, ternary, 0.25-density, looped = below paper's weakest |
| QAT handling | ⚠️ **untested in isolation** — plausibly adds fragility; not separated yet |

## Recommended next steps (GATED — operator rejoins for inference work; do NOT auto-launch)
1. **Discriminate base-weakness vs recipe** with a gentler **unpacked** SFT (one example per
 sequence, lr 5e-6, ~15–25 updates, no packing). If it STILL collapses → the base is too weak
 (lobotomy confirmed). If it does NOT → the packed/aggressive recipe was the problem.
2. **If pursuing STP-SFT specifically:** rewrite `STPLoss` to the paper form (single random triplet,
 `|t−s|≤τ` window) + two-view anchoring (s@query-start, t@answer-end), keep λ≈0.01 constant.
3. **Honest strategic read:** deployable instruction-following on this model likely needs SFT
 **folded into pretraining** (instruction-mix) or a **bigger/denser** base — a fragile post-hoc
 42-step pass on a 276M ternary-sparse-looped model is the wrong tool. Generation-stability
 (EOS/looping) is the real failure to fix, independent of STP.

## Artifacts
- Control pipeline: `ignore/run_sft_nostp_pipeline.sh`, log `ignore/sft_nostp_pipeline.log`
 (`SFT_NOSTP_PIPELINE_DONE … rc=0` all stages).
- SFT ckpts: `checkpoints/morph/sft_off_nostp/step_42.pt`, `checkpoints/morph/sft_on_nostp/step_42.pt`.
- OWT bench: `ignore/bench_sft_off_nostp.json`, `ignore/bench_sft_on_nostp.json`.
- Instruction probe: `ignore/ig_off_sft_nostp.json`, `ignore/ig_on_sft_nostp.json`.
- wandb: `sft_off_nostp` (yrgnh15z), `sft_on_nostp` (project morph, entity adew-me).
- STP paper: arXiv:2602.22617 (HTML/PDF read by research subagent).
