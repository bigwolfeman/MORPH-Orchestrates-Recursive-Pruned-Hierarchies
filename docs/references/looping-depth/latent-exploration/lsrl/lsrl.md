# LSRL (process supervision at every latent depth): reading and local cache

Read 2026-09-18.

## Citation (verified)

Hangliang Ren (Khoury College of Computer Sciences, Northeastern University). *LSRL:
Process-Supervised GRPO on Latent Recurrent States Improves Mathematical Reasoning*.
Findings of the Association for Computational Linguistics: EMNLP 2025, pages
12534-12545, Suzhou, China.
<https://aclanthology.org/2025.findings-emnlp.669/>

**Not on arXiv.** A title search, an abstract search for "LSRL", and two topic searches
on the arXiv API all returned nothing relevant (the one "LSRL" hit is an unrelated 2024
NeurIPS paper's code repository name). The ACL Anthology PDF is the source of record.

**Single author.** The brief implied a group. It is one author.

Read in full: 12 pages, all of it.

## Local cache

- [PDF](../../../../../ignore/papers/2025.findings-emnlp.669-lsrl-process-supervised-grpo.pdf)
- [Extracted text](../../../../../ignore/papers/2025.findings-emnlp.669-lsrl-process-supervised-grpo.txt)
- 375,096 bytes, 12 pages, 712 text lines.
- SHA256: `c3e60c781440f0899b15d9a834b7e042fe487bfccb66c24ce0700cacecffc8e0`.

## What it actually does

Huginn-3.5B is a prelude / core / coda recurrent model: the core is looped r times and
only the final state reaches the LM head. LSRL's claim is that the bottleneck is sparse
credit assignment, because one reward arrives after all r iterations.

The method DECODES EVERY DEPTH. Each intermediate latent state s_k is pushed through the
coda and LM head and autoregressively decoded into a textual snapshot. A GPT-4.1-nano
grader, called twice and averaged, scores each snapshot on two axes: internal quality
(logical consistency, clarity) and mathematical progress (does it reduce unknowns, apply
a correct operation). Scores are min-max normalised within the GRPO group.

    R_k    = 0.5 * IQS_k + 0.5 * PS_k
    R_proc = sum_k gamma^(k-1) * R_k                  gamma = 0.99
    R_tot  = 0.7 * [answer correct] + 0.3 * R_proc

GRPO with G = 8 trajectories per prompt, rank-8 LoRA on the core's 16 projection
matrices only (0.17 % of parameters trainable), int8 QLoRA, one L40S GPU, 500 GSM8K
training problems. A one-pass cache collects all r states in a single core unrolling
instead of re-running from s_0 per depth.

    prelude -> s_1 -> s_2 -> ... -> s_r -> coda -> answer
                |      |             |
                v      v             v
              decode decode        decode        -> grader -> R_k

## The key numbers

    model          r    GSM8K    MATH    MathQA   FLOPs/token
    SFT-r8         8    13.49     5.61   24.07     1.0x
    RL-Outcome     8    14.54     6.32   24.62     1.0x
    LSRL           8    17.76     6.94   26.13     1.0x
    SFT-r32       32    24.87    11.24   27.97     4.0x

LSRL gains +4.27 on GSM8K over the supervised depth-8 baseline. Outcome-only RL gains
+1.05, so process supervision contributes about 75 % of the lift. LSRL-r8 recovers
roughly 75 % of the depth-32 score at 25 % of the recurrent compute.

The shallow-recurrence ablation is the number MORPH should keep. Rerunning the whole
recipe at r = 4:

    model             GSM8K   MathQA
    SFT-r4             8.36    22.93
    RL-Outcome-r4      8.01    22.47
    LSRL-r4            8.59    23.14

Flat. Every variant clusters. The author's reading is that there is a minimum recurrent
depth below which intermediate states carry too little to grade, and that further work
is needed to locate the boundary.

The qualitative trajectory table is worth reading once. On the baseline, depth 1 is
boilerplate and depth 3 is arithmetically wrong. On LSRL, depth 1 already contains the
correct plank count and later depths refine it. Residual errors persist at depths 2 to 3
(prompt echo, irrelevant phrases, a unit mix-up giving an off-by-factor answer).

## Where it sits in MORPH's 2x2

Deterministic, one stream. It is the only paper in this batch that attacks per-depth
credit directly, which is the axis MORPH has hammered hardest.

## What in MORPH already tested this

Seven per-pass targets, all flat. Staged, oracle, gradpass, per-pass horizon, critic,
recon and disc all read at most 0.002 nats on passes 2 to 6
(`per-pass-targets-met-in-one-step`). The per-pass arm's own ladder RISES with t
(t1 4.51 to t6 4.71), which is harder targets, not progress.

The depth-summing warning is ours and applies to their reading too.
`progressive_p: 0.5` doubled the per-pass gain AND degraded pass 1 by about the same
amount, so an instrument at the sampled depth saw nothing
(`depth-summing-instruments-hide-pass-trades`). LSRL reports only end-task accuracy, so
a trade of that shape would be invisible in their tables.

The decode-every-depth machinery has a local analogue that already exists and already
read flat: `tul.spandec_per_pass` grades every pass with a span decoder, and
`tul.mux_every_pass` reads the MUX at every pass. The `perpass` arm scored K1-K6 +0.0008
at 2.31x wall clock.

## What a MORPH arm implementing it would need

Exists: `tul.spandec_per_pass` with `spandec_pass_tokens`, `spandec_pass_weight` and
`spandec_pass_horizon_max` (a per-pass decoder), `tul.critic_weight` and
`tul.critic_every` (a per-pass critic), `tul.progressive_p` (Deep Thinking detach).
The coda already accepts a cell, so decoding a pass-t cell to text is a probe we can run
today on any checkpoint.

Does not exist, and mostly should not be built:

- Any RL. MORPH trains by CE.
- Any external grader. Their whole signal comes from an LLM judging a decoded snapshot
  for mathematical progress. Web text has no analogue of "reduces unknowns"; the nearest
  is CE against the true continuation, which is the per-pass target we have run seven
  times.
- A depth-threshold experiment. This IS buildable and is the useful piece: rerun the
  standing per-pass arm at `slot_depth_fixed` 4, 8 and 16 and check whether per-pass
  targets stay flat at every depth. Note the standing rule that a fixed rung must not be
  ranked on single-depth CE (`fixed-depth-is-not-sampled-depth`); the reading is the
  per-pass profile, not the end CE.

## The instrument it implies

Decode the cell at every pass through the coda and READ THE TEXT. LSRL's Figure 1 is a
qualitative instrument and MORPH has never run it on a slot cell. We have measured cosine
to a code, effective rank, worth profiles and CE, but nobody has looked at what pass 3's
cell says. It is one forward pass on an existing checkpoint. If pass 1 and pass 6 decode
to the same string, that settles the per-pass question in a way no scalar has.

The second instrument is their depth threshold: the per-pass gain as a function of the
loop's trained depth, with the pass-1 level and the per-pass slope reported separately
per the `depth-summing-instruments-hide-pass-trades` rule.

## What it does NOT establish for us

It does not establish that per-depth supervision works in general. Its own r = 4 run is
flat, which is a two-point comparison with no seeds and no confidence interval.

It does not separate process supervision from the grader's knowledge. GPT-4.1-nano is a
far stronger model than Huginn-3.5B, so part of the +3.22 could be distillation.

It does not control for the LoRA. Every RL variant carries rank-8 LoRA on the core and
the SFT baselines do not.

It reports single runs. No seeds, no error bars, 500 training problems.

Its MATH gain is +1.33 points from a 5.61 baseline, and the author attributes the
shortfall partly to the reward model not seeing higher-order steps at all.
