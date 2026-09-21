# Planned: LXTUL-P rung P3, the target is SEMANTIC (`tul.code_target_source: sonar`)

Status: planned

Date: 2026-09-21 (frozen before any GPU step of the arm and before the cache check
below is run). Arc: the LXTUL-P note
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
rung P3, and the LCM reading
[`2026-09-21-lcm-reading.md`](../../../docs/references/tul-latent-emission/lcm/2026-09-21-lcm-reading.md)
§1–2. Partner: `tul-code-cfg-tlow`
([`2026-09-21-lctul-cfg-tlow.md`](2026-09-21-lctul-cfg-tlow.md)), queued first on the
5090; this arm pairs against it key for key at 20k.

## Question

The LCTUL flow thinker aims at E's VERBATIM reconstruction code of the next span, of
which the context explains 5–15 %, and it reads the past at 1 % of its loss. In SONAR's
sentence space LCM finds the true next sentence retrievable among in-batch alternatives
75–80 % of the time. If the thinker's target is SONAR's 1024-d embedding of the next
span's text (lifted to the cells by a frozen orthogonal map and rms-normed) instead of
E's code, does the field start reading the context, and does the coda's read of the
sample get cheaper?

## Precondition: does SONAR carry sequential structure on MORPH's spans?

LCM's spans are sentences of up to ~250 characters. MORPH's spans are the packer's
fragments (min 4, cap 32 tokens, boundaries at `.;!?` newline and dashes), so a span is
often a clause or a sentence fragment. If SONAR embeddings of adjacent fragments are no
closer than embeddings of random fragments, the target has no context to find and the
arm should not be queued. The check runs on the 2k cache
(`/home/wolfe/sonar-cache/tul_code_cfg_tlow`, 590,446 unique spans, `spans.keys.npy`
in stream order, 2,000 train steps at batch 6) BEFORE the run and AFTER this file is
committed (`lab/divergence/sonar_cache_structure.py`, first 20,000 stream keys, 1,000
bootstrap resamples):

- **C-1.** Mean cosine between a span and the NEXT stream key minus the mean cosine
  between a span and a random key from the same 20,000: at or above **+0.10**. **70 %.**
  SONAR is anisotropic (random same-language pairs sit near 0.3–0.5), so the excess, not
  the raw cosine, is the reading. Row boundaries (~1 in 30 adjacent pairs) count against
  the excess.
- **C-2.** Successor retrieval with the zero-parameter predictor "next ≈ current":
  among 1,000 candidates (the true successor plus 999 random keys), the true successor
  ranks first by cosine to the current span at or above **5 %** of queries (chance
  0.1 %). **65 %.**

If **C-1 fails** the arm is not queued and this file moves to `failures/` with the
reading: SONAR on 32-token fragments is the wrong space, and the next target is a span
encoder trained on MORPH's own spans (E with a contrastive objective), not a borrowed
sentence encoder. If **C-1 holds and C-2 fails** the arm is queued; C-2 is a floor on
how much an UNCONDITIONAL "copy the last span" predictor gets, and the thinker reads more
than the last span.

**Precondition reading (2026-09-21 03:45, run after commit 0ee368b, artifact
`../results/2026-09-21-lctul-tlow-sonar/sonar_cache_structure_2k.{json,txt}`):** over
19,999 adjacent pairs of the first 20,000 stream keys, cos(next) 0.1860, cos(random)
0.0907, **C-1 excess +0.0953 [+0.0935, +0.0972]: FAILS its 0.10 threshold**, the interval
clear of it by 0.003. **C-2 top-1 0.0955 [0.0835, 0.1085] among 1,000 candidates: HOLDS**
(95 × chance 0.001). The letter of the binding says the arm is not queued. The structure
the check was written to detect is present (a 95-fold retrieval excess on a
zero-parameter predictor) and the threshold I froze on C-1 was missed by 5 % of itself.
The rule against post-hoc rescue is the reason this file stays in `planned/` with the arm
HELD rather than queued or filed: the go/no-go on a 0.005 miss is Wolfe's, stated here
before anyone reads the run. The SONAR anisotropy assumption in C-1 was wrong (random
pairs sit at 0.09, not 0.3–0.5), which does not change the excess.

## Hypothesis

A target whose predictable-from-context fraction is large gives the field a reason to
read the past. LCM's Table 3 says that fraction is large for sentences in SONAR space.
The lift (`1024 → M·C = 2048`, orthogonal, frozen) and the per-cell rms-norm keep the
target's geometry; the coda has to learn to read a semantic vector it cannot invert to
tokens (LCM's decoder loses ~30 % AutoBLEU on a round trip), so the teacher-forced CE
`val/ce_tf` will be far ABOVE the parent's 0.32 by construction. The rung's reading is
the field's context dependence and the reader's paired CE on the SAMPLE, not `ce_tf`.

## Method

`morph/configs/tul_code_cfg_tlow_sonar.yaml`, one factor over `tul_code_cfg_tlow`:
`code_target_source: sonar`, `code_sonar_cache` pointing at the FULL cache
(`/home/wolfe/sonar-cache/tul_code_cfg_tlow_full`, being encoded on the 3070 at 03:40:
1.31 M of 5.74 M spans; the config must be repointed from the 2k cache before the queue
line is appended, and a miss RAISES). Same schedule (μ −1, σ 1), guidance (drop 0.1,
scale 2), 20,000 steps, `ademamix_t_beta3` 20000, checkpoints every 5000, batch 6, the
same seed and rows as the parent. E runs for its validity mask and gets no gradient.
Build: `morph/model/sonar_cache.py`, `_sonar_target` in `transformer.py`,
`scripts/sonar_span_cache.py`, `tests/test_tul_code_sonar_target.py` (15). Committed
2ef117c.

Readouts as the parent's prereg: `val/code_ca` vs `val/code_ca_chance`, the
context-blind flow probe on the 20k checkpoint (cfg_drop 0.999 vs 0.001), the 8-draw
k-curve marginal, the paired one-draw CE at k = 8 against the parent on the same rows,
`val/ce_tf`, phase-3 rate.

## Predictions (frozen)

Parent numbers are the tlow arm's own at 20k, unknown at freeze; where a number is
needed the `tul-code-cfg` figures from the tlow prereg stand in: context-blind gap
0.0036 (1 %), marginal k = 1 → 16 −0.005, one draw vs the strict ruler +0.648, ce_tf
0.32, phase-3 rate 19.7k tok/s. `code_ca_chance` about 0.003. Probabilities are mine.

- **P-1 (healthy).** 20,000 steps, tripwire HEALTHY, phase-3 rate at or above **13,000**
  tok/s. **80 %.** The cache lookup is one host round trip per forward under
  `torch.compiler.disable`.
- **P-2 (zero misses).** The run completes without the cache-miss raise. **85 %.** The
  cache was cut by the trainer's own loader over the same stream; the risk is a row the
  20k-step stream reaches that the 20k cut did not (the cut ran `--steps` over the
  same seed; the val batches are included).
- **P-3 (THE RUNG'S REASON: the field reads the context).** The context-blind flow probe
  on the 20k checkpoint moves by **more than 5 %** of the with-past loss when the past
  is removed, AND by more than the tlow arm's own reading. **50 %.**
- **P-4 (CA above chance).** `val/code_ca` at 20k at or above **5 × `val/code_ca_chance`**
  and at or above **1.5 ×** the tlow arm's. **50 %.**
- **P-5 (the reader cannot invert the target).** `val/ce_tf` at 20k inside **[0.8, 4.0]**
  nats (parent 0.32). **75 %.** A 1024-d sentence embedding of a fragment does not
  carry its tokens; the coda speaks a paraphrase. Below 0.8 would say the lift leaks the
  tokens (check the cache key: it is the token-id hash, so a leak is impossible by
  construction, but a reading below 0.8 says look).
- **P-6 (the sample's read).** Paired one-draw CE at k = 8 on the same rows, sonar −
  tlow, inside **[−0.10, +0.30]**. **55 %.** A semantic sample makes the coda's token
  read WORSE per token even when the sample is right, because the coda no longer has
  the verbatim code; the rung is scored on P-3/P-4 and this clause bounds the price.
  Residual: 35 % worse than +0.30, 10 % better than −0.10.
- **P-7 (the marginal earns steps).** 8-draw marginal k = 16 − k = 1 at or below
  **−0.020**. **40 %.** A field that reads the context is the precondition for guidance
  to curve it.
- **P-8 (the sample is not the mean).** `val/code_ca` at 20k above chance by the P-4
  margin AND the sampled code's cosine to its own truth (the round-trip instrument
  `lab/divergence/code_roundtrip_probe.py`, k = 8) above **2 × the chance cosine**
  between two spans' truths in SONAR-lift space. **45 %.** The tul-code-20k sampler sat
  AT chance (0.098 vs 0.102).

## Binding

If **P-3 and P-4 hold**: the target was the reason the thinker was context-blind. Next:
the SONAR target on the DENOISER (rung P2 + P3), which is a build at the code-target
seam, and the late-training token-path unfreeze Wolfe named (the coda never sees text is
not a hard rule).

If **P-3 fails and P-5 holds**: a semantic target in a borrowed space does not make the
field read the context either; the conditional-mean reading is not the target's fault.
The remaining lever is the thinker's job structure (rung P2), and this arm's rung
closes on the flow thinker.

If **P-2 fails**: a cache-coverage fault; rebuild the cache over the missed rows and
re-queue with a dated Method amendment. Nothing about the idea is read.

## Not verified before launch

- The full cache is not finished (ETA ~07:30 on the 3070); this arm cannot start before
  it is rsynced and the config repointed, and that repoint is a config commit after this
  file.
- The cache check (C-1, C-2) has not run; its predictions above are frozen with the
  rest.
- No GPU step of this arm on the full cache. The 2k-cache smoke ran 0 misses over its
  own 2,000 steps (from the build report, not re-run by me).
- The parent's own 20k readings do not exist yet; every "vs tlow" clause waits on them.
- `torch.compile` with the `compiler.disable`d seam: the tests are eager.
