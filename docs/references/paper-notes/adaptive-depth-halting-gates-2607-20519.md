# Paper note: Adaptive depth in looped transformers (halting gates and trajectory readouts)

- **Title:** Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts (Popescu, Saez de Ocariz Borde, Lio)
- **arXiv or URL:** 2607.20519 (v1, 2026-07-08, cs.LG). Type: paper.
- **Local paper text:** docs/references/looping-depth/adaptive-depth-halting-gates-2607-20519/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It separates two roles of a halting distribution: it weights the per-depth training loss, so it shapes the trajectory, and it picks the exit. We use a fixed Poisson depth draw, which is a fixed prior in their terms. Triage score 1: halting and decode speed are not our priority, and it adds no training-speed or credit-assignment result.

## 2. What the paper shows

Read: Sections 1 to 6 text and Tables 1 to 4. Not read: Appendices B to I (parity, OOD, Ouro-Thinking).

- 2026-10-20 (Sec 3.3, Table 1, MANO modular arithmetic, 4-layer core, T=6, 3 seeds): fixed geometric prior (lambda 0.3) with a confidence readout reaches 99 % at 1.50 to 1.55 average loops. Uniform prior reaches 95 % at 2.19 +/- 0.51 loops and never reaches 99 %. Fixed depth needs 2.50 (geometric, 95 %) and 4.50 (uniform).
- 2026-10-20 (Sec 5.2, Fig 4, Table 2): fixed-prior trajectories with simple confidence readouts beat jointly trained linear gates, and MLP gates still do not match the best fixed-prior frontier.
- 2026-10-20 (Sec 5.4, Fig 6): gates fitted on a FROZEN trajectory can match simple readouts on the fixed-prior trajectory; on a trajectory made by joint gate training they cannot improve it. Conclusion: the failure sits in the trajectory, not the gate class.
- 2026-10-20 (Sec 5.5, Table 3, Ouro-1.4B/2.6B, 12 model-benchmark pairs, single runs): post-hoc readouts beat the pretrained gate on held-out accuracy in 5, use fewer loops in 7, both in 3. Latency (RTX 5070 Ti, Table 4): Ouro gate 1.24x to 1.58x; MANO top-1 confidence 1.36x at 6.0 -> 1.26 loops.
- 2026-10-20 Failure modes: synthetic tasks plus no retrain of Ouro; Ouro differences are small and single-run; hidden-displacement readout exits late and adds overhead.

## 3. Our speculation

- 2026-10-20 Speculation: a geometric prior moving loss weight to early depths gives early states that are already good, so it makes K1 small and K1-K6 small too. That matches our "skip flattens K-curve, CE improves" memory. A depth-weighted loss therefore trades the metric we use for go/no-go against exit quality. This is why it does not fit our "only main CE buys depth" rule.
- 2026-10-20 Speculation: Poisson(6) draws are a fixed prior with weight on deeper passes than their geometric prior. Their uniform-versus-geometric gap says the draw shape sets how early the trajectory commits. A mean-depth sweep is already in our ledger (mean-depth dial); this paper adds nothing numeric for it.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None.
