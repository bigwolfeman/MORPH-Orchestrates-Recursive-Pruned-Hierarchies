# Parcae: does epivol or do the stabilisers make the 4-cell slot loop depend on depth?

Status: success (all 5 predictions hold; filed 2026-10-14)

## Question

Every Parcae slot-loop arm that earned K1-K6 (+0.029 to +0.133) had 4 cells and all of: fixed-point term
0.1, gain hinge 100, cotangent clip 4, and epivol 0.1. Every arm without them is flat: one cell with the
stabilisers (`bare`), the minimal fan (3.7446, 0.0000), and the minimal fan with latent selection
(3.7563, 0.0000; `results/2026-10-14-parcae-minimal-fan-5k/hypothesis-result.md`). No run has 4 cells with
only some of the four on (`morph-scratch/abq/drawing-board/parcae-arm-ledger.md` section 3).

Epivol (SPEC 3.7) rewards the spread of a slot's cells at passes 1 and 2 only, and the coda reads the last
pass. A term that pushes the early passes apart would make CE depend on depth by construction, which fits
the ledger's pattern: among earning arms K1-K6 rises as CE@1 gets worse while CE@6 stays put. Which of the
two groups produces the depth dependence, and does either one buy final CE?

## Arms (5k, pointer head, all minimal-fan keys; one seed each)

| arm | overrides on `arm=minimal_fan` |
| --- | --- |
| minimal fan + epivol | `arm.lxtul.repel_lambda=0.1` |
| minimal fan + stabilisers | `arm.lxtul.fixed_point_lambda=0.1 arm.lxtul.gain_lambda=100.0 arm.lxtul.cot_clip=4.0` |

References: minimal fan s1 3.7446 / 0.0000 (s2 running); strict LXTUL + pointer 3.7608 / +0.0468.

## Predictions

1. Epivol arm: K1-K6 at least +0.02 (65 %).
2. Epivol arm: CE@1 at least 0.02 worse than minimal fan's 3.7447 (60 %).
3. Epivol arm: CE@6 not better than minimal fan's 3.7446 (70 %).
4. Stabilisers arm: K1-K6 within +-0.005 of 0 (65 %).
5. No non-finite loss in either arm (90 %).

## What each outcome changes

- 1 and 4 hold: the Parcae loop contribution of the LXTUL family comes from a pass-indexed diversity term.
  It is the "tax refund" pattern, and the drawing board starts without it.
- Both flat: the depth dependence needs epivol and the stabilisers together; the next run is the stack.
- Stabilisers arm earns: the hinge or the fixed-point term is the source; split it next.

## Method

Code: testbed (cc53940 plus docs). Queue `morph-scratch/abq/pz-panel/split_5k.sh`, started after the flat
references; one job at a time under the GPU lock. Amendments get a date and a reason here.

## Results

Both runs finished at 0764c76 with exit 0 (`morph-scratch/abq/pz-panel/panel.log`). Same 480 held-out rows
as the minimal fan. Sweep CE by eval depth:

| arm | CE@1 | CE@2 | CE@3 | CE@6 | CE@8 | K1-K6 [95 % CI] |
| --- | --- | --- | --- | --- | --- | --- |
| minimal fan s1 (reference) | 3.7447 | | | 3.7446 | | 0.0000 |
| minimal fan + epivol | 3.8020 | 3.7861 | 3.7574 | 3.7557 | 3.7557 | +0.0462 [+0.0447, +0.0478] |
| minimal fan + stabilisers | 3.7419 | 3.7419 | 3.7418 | 3.7419 | 3.7418 | +0.0000 [-0.0000, +0.0001] |

Prediction check:
1. Epivol K1-K6 +0.0462, at least +0.02. HOLDS.
2. Epivol CE@1 3.8020, 0.057 worse than the fan's 3.7447. HOLDS.
3. Epivol CE@6 3.7557, 0.011 worse than the fan's 3.7446. HOLDS.
4. Stabilisers K1-K6 +0.0000, inside +-0.005. HOLDS.
5. No non-finite loss in either run. HOLDS.

## Verdict

Epivol alone makes the 4-cell slot loop depend on depth: +0.046, as large as strict LXTUL + pointer
(+0.047). It does so by making the depth-1 read worse (+0.057) and letting passes 2 and 3 win most of it
back. Final CE is 0.011 worse than without it. The fixed-point term, the gain hinge and the cotangent clip
together do nothing to depth (0.0000) and nothing outside seed spread to CE (-0.003; spread about 0.010).

So the loop contribution of the LXTUL family on Parcae is the tax-refund pattern of a pass-indexed
diversity term, not depth the main CE buys. This is a direct breach of the "only the main CE buys depth"
rule, measured.

## Updated hypothesis

No Parcae slot-loop arm has yet earned depth that the main CE pays for. The redesign starts with no epivol
and no stabilisers (they are inert on the minimal fan). The decode loop (`lxtul/decode-loop`) and the
LCM reference are the next readings.
