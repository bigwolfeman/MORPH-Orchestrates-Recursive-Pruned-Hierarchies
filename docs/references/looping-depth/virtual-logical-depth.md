# Beyond Parameters: reading and local cache

Read 2026-09-05. Ruike Zhu et al., *Beyond Parameters: Exploring Virtual Logic
Depth for Scaling Laws*. [arXiv 2506.18233v3](https://arxiv.org/abs/2506.18233v3),
revision dated 2025-10-12. The PDF identifies itself as under review for ICLR 2026.
This record does not assert a later publication outcome.

## Cache and reading coverage

- [Local PDF](../../../ignore/papers/2506.18233v3-virtual-logical-depth.pdf)
- [Full extracted text](../../../ignore/papers/2506.18233v3-virtual-logical-depth.txt)
- PDF size: 1,442,847 bytes, 22 pages.
- SHA256: `cd1a9e7e36edb45e2626f2a0f6dbda5cfc345a2807d06c0736c560b8c8984180`.

The main agent reads the entire extracted text, lines 1 through 1365, including
references and appendices A through I. Pages 1-10 contain the main argument and
experiments; pages 10-13 contain references and the appendix map; pages 14-22
contain methods, datasets, configurations, evaluation, placement ablation, and
limitations. Figure 3 and Figure 4 on page 9 also receive visual inspection.

## What changes our plan

The paper tests weight reuse patterns and separates synthetic memorization from
exact-answer reasoning. Appendix H compares repeating early versus later layers.
The latter performs better in that small experiment. That motivates testing
recurrence on contextual states, but does not establish that compressed slots
retain the required information. Section 4 and Appendix I also show or discuss
non-monotonic depth effects. The work supplies neither an OWT optimum nor a TUL
result. Its parameter-efficiency evidence is not an equal-compute comparison.

Our inference is to retain cycle recurrence, start thoughts from contextual
prelude states, and measure OWT CE and reasoning separately. Do not transplant
the paper's depth labels directly into Parcae or MORPH.

## Reasons for caution

Section 3.1 and Appendix G.1 use output-distribution entropy as a memory proxy.
This is not target-token cross-entropy. A confidently wrong predictor has low
output entropy without having remembered the correct sequence. That counterexample
limits the interpretation of the proxy, not the separate reasoning measurements.

The prose and figures also conflict on repetition-factor semantics. The main
reasoning recipe states LR 2e-5, while Appendix F.2 lists 5e-4. Table 1 averages
nearby checkpoints, not independent seeds. These issues require source-code
resolution before any faithful reproduction. We do not copy that recipe.

## Relation to local results

The [Parcae result](/home/wolfe/parcae/docs/experiments/successes/2026-09-05-parcae-owt-loop-contribution.md)
is the local CE reference. The
[action plan](../../../.agents/notes/proposed/architecture/2026-09-05-parcae-tul-depth-transfer-plan.md)
keeps its positive-control result separate from the proposed slot mechanism.
