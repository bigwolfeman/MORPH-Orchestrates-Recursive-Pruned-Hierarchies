# Paper note: When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning

- **Title:** When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning (Ayushi Chadha; ICML 2026 CompLearn workshop)
- **arXiv or URL:** 2606.03741 (v1 2026-06-02, v2 2026-06-30)
- **Local paper text:** docs/references/looping-depth/when-to-replan-2606-03741/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20
- **Triage score:** 1. Persistence sweep on ARC training loss; effect about one seed std; aux cosine loss is on our banned list.

## 1. Why we reference it

- 2026-10-20: It sweeps how many fast steps a slow state's intent lives, which is the question "how long should a span-level state persist while token-level states update". Our span-CALM and LXTUL x spiral panels have a slow level over a fast one, and our spiral uses fixed chunk sizes 16/8/4/2.
- 2026-10-20: The test bed is ARC grids with a puzzle-augmented HRM, and the metric is training-set LM loss. Weak link to web-text LMs.

## 2. What the paper shows

Read in full (main text, Sections 3 to 5; Section 6 and 7 skimmed). 

- 2026-10-20 (Sec 3, Eq 3 to 7): every P low-level steps the high state emits a unit-norm direction g; it is added to the low update through a learned projection; a cosine loss 1 - cos(low-level displacement, g) with weight lambda is added to the HRM loss.
- 2026-10-20 (Sec 4.3, Fig 2a, lambda 0.05): training LM loss, baseline HRM 1.640; P=1 1.674 (worse than baseline); P=2 1.638; P=3 1.544; P=4 about 1.56; P=3 to 8 in 1.544 to 1.590. The 5-seed replicas: P=3 mean 1.595 (std 0.045, min 1.540, max 1.662), P=4 mean 1.601 (std 0.048). The baseline 1.640 has no stated seed spread.
- 2026-10-20 (Sec 4.4): lambda sweep at P=4 has an optimum near 0.05; 0.10 is near baseline; 0.20 and above are worse.
- 2026-10-20 (Sec 5, Tables 1 and 2, one seed, batch 64, L4 GPU, a different data scale): at lambda 0.10, P=4: full 1.327, no subgoal 1.227, random unit-vector directions 1.230. So learned direction content, not the extra module or loss, produces the 0.10 loss increase.
- 2026-10-20 (Sec 4.5): ConceptARC-mini 2.316 baseline vs 2.308, about 0.4 %.
- Failure modes reported: gains are modest; evaluation is ARC only; a representation-level analysis is deferred.

## 3. Our speculation

- 2026-10-20 Speculation: the 5-seed std (0.045) equals the gap between the best-P mean (1.595) and the single baseline value (1.640). With one baseline number the "sweet spot" is about one standard deviation. The P=1 vs P=3 gap (0.13) is the only effect clearly larger than seed noise. This matches our memory that n=1 comparisons are unreadable.
- 2026-10-20 Speculation: P=1 harming is a mild version of our finding that a slow level which resets every step has no identity to compose. It fits "particles collapse unless each has a persistent identity". It does not give a depth result.
- 2026-10-20 Speculation: the cosine alignment loss is an auxiliary term on the state path. Our rules ban pass-indexed auxiliary depth losses, and the paper's own lambda curve shows it turns harmful above 0.05. Do not port the loss. The persistence knob alone is a possible control: it is the ratio of fast steps per slow update, which our spiral levels already fix by chunk size.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper claim: a persistent direction biases the fast loop usefully. Our measurement: coarse/span states carry context in one pass, and depth was bought only by bounding a pass's reach. The paper has no test of whether the gain comes from a longer reach or from a smoother optimization target; it reports training loss only.
