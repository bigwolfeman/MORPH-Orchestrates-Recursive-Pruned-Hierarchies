# Experiment: does the thinker's sample carry the meaning of the context? (LCTUL semantic probe)

Status: planned
Date: 2026-09-16
Owner: Claude (session f9558148), for Wolfe ("You can't actually determine which is
better accurately [from the L2 residual] ... if you actually decode it and read it, you
can tell ... What we're looking for is how semantically correct it is.")

## Question

Every LCTUL verdict so far rests on the L2 distance between the thinker's sample and E's
code (the residual probe), the flow loss, and the coda's CE. All three are blind to a
sample that carries the right topic in the wrong words. The coda always writes a
plausible sentence; the question is whether what it writes from the thinker's sample is
ABOUT the right thing more often than what it writes from a sample made for another
context. `lab/divergence/code_semantic_probe.py`: for N = 120 cuts, the open slot is
handed OWN (this context's sample), SHUF (another cut's sample), ZERO and ORACLE (E's
code of the true span), the coda writes the span greedily, and each text is scored
against the true span and the context tail by MiniLM sentence-embedding cosine and by
content-word overlap. The reading is the paired OWN − SHUF difference with a bootstrap
CI over cuts.

Arms: the parent at 20k (`tul-code-20k`), the plain continuation at 50k
(`tul-code-20k-50k`), the jepa continuation at 50k (`tul-code-jepa-50k`).

## Hypothesis

H-S1 (Wolfe's reading): the sample is semantically conditioned on the past even where
its L2 residual reads as a random draw; OWN lands nearer its own true span than SHUF
does, and the margin grows from 20k to 50k (the flow model is data-starved, not wrong).
H-S0: the sample carries no more of the context than another cut's sample; OWN − SHUF
sits at 0 on both metrics, at 20k and at 50k, while ORACLE − SHUF is large.

## Predictions (frozen, before any full run; a 6-cut smoke of the script ran first)

- P-S1. ORACLE − SHUF cosine to the true span > +0.10 with the CI excluding 0 on every
  arm (the ceiling is real: the probe can see meaning). 90 %.
- P-S2. OWN − SHUF cosine to the true span > 0 with the CI excluding 0 on the plain 50k
  arm. 35 %.
- P-S3. OWN − SHUF cosine to the true span on the plain 50k arm exceeds the parent's
  20k value by ≥ 0.02 (training moves the semantic reading). 30 %.
- P-S4. OWN − ZERO cosine to the true span > 0 with the CI excluding 0 on at least one
  arm. 40 %.
- P-S5. Content-word overlap with the true span: OWN − SHUF > 0 with the CI excluding
  0 on the plain 50k arm. 30 %.

## Binding

- P-S2 holds → the L2 instruments were the wrong ruler; the semantic reading becomes a
  standing probe on every arm and the 50k continuations are re-read on it.
- P-S1 holds and P-S2, P-S4, P-S5 fail → the coda can read meaning from a code (the
  oracle) and the thinker's sample carries none of it, at 20k and at 50k; H-S0.
- P-S1 fails → the probe cannot see meaning at this span length; find a better judge
  before reading anything else.

## Method

Spark, `HF_HUB_OFFLINE=1`, MiniLM from the local HF cache; greedy decoding, span length
= true length + 2; sampler steps 8; the SHUF partner is the next cut's sample; the past
cells hold E's codes of the written spans in every condition (the generation regime).
Output: `semantic_<arm>_<step>.json` plus a `.txt` of every decoded line for reading.

## Not verified before launch

- MiniLM cosine on 5–32-token spans is noisy; the N = 120 bootstrap is the guard.
- The greedy decode under the ORACLE code returns the true span nearly verbatim on the
  copy arms, so P-S1 is close to a tautology; it is the sanity gate, not a finding.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
