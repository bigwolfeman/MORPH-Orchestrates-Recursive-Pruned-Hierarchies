# Repo note: jacobfa/Attractor (code for "Solve the Loop", 2605.12466)

- **Title:** Attractor, "Solve the Loop: Attractor Models for Language and Reasoning" (Jacob Fein-Ashley, Paria Rashidinejad; USC). Type: GitHub repo (MIT), a fork of Parcae training code plus TinyRecursiveModels.
- **arXiv or URL:** https://github.com/jacobfa/Attractor ; paper arXiv 2605.12466 (2026-05-12), abstract only.
- **Local paper text:** docs/references/looping-depth/attractor-solve-the-loop-repo/ (paper.txt = README + file list + key files, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It replaces the Parcae recurrent core with a fixed-point solver and a one-step implicit gradient. We hold a fixed-point term and a gain story for our core, and Parcae is our base. The repo shows what a fixed-point core costs in code and what it claims.

## 2. What the repo shows

Read: README, `attractor/models/attractor/attractor.py` and `config.py`, the 140M model config, the Parcae 140M config, the 140M launch yaml. NOT read: `solvers.py` (only its first lines), the sudoku/ARC code, the HF checkpoints, the paper body. The README reports NO numbers. The numbers below come from the arXiv abstract page only.

- 2026-10-20 (attractor.py): prelude (7 blocks at 140M) encodes tokens to context c. One weight-tied block (`n_fp_blocks=1`, LayerScale gates init 0.75, gamma_max 0.75) is the "attractor head" f(y,c) = FPHead(y+c) - y. Forward finds the fixed point y* with Anderson acceleration under `no_grad` (max_iter 64, min_iter 6, tol 3e-4, m=5). Backward is the implicit-function-theorem gradient with the one-step Neumann adjoint u = v + J^T v (`backward_type: onestep`). No coda (`n_layers_in_coda = 0`); logits read LN(y*).
- 2026-10-20 (config): depth is set by convergence, not by a Poisson draw; `partial_depth_eval: [1, 2, 4]` evaluates the solver cut at 1, 2, 4 iterations plus the full run. 11.2B-token budget at 140M, Muon+AdamW, bf16, `compile_model: False` (the implicit-gradient hook cannot be traced by torch.compile).
- 2026-10-20 (code comment, monitor): `fp_head_contribution = |y*-c|/|c|`, with the note "near zero means the head is a near-identity and the backbone is doing all the work". The model logs this.
- 2026-10-20 (abstract only): perplexity better by up to 46.6 %, downstream up to 19.7 %; 770M beats a 1.3B Transformer trained on 2x tokens; 27M TRM-DEQ reaches 91.4 % Sudoku-Extreme and 93.1 % Maze-Hard; "equilibrium internalization": the solver can be removed at inference with little loss.
- Failure modes: not reported in what we read. The code adds an adjoint clip option (`adjoint_grad_clip`, 1.0 default, None in the 140M config) "if the head drifts toward a less contractive regime". The forward solver runs up to 64 iterations per step, so training FLOPs per step are not lower than Parcae's.

## 3. Our speculation

- 2026-10-20 Speculation: "equilibrium internalization" is the claim that training with the loop moves the one-pass output to the fixed point. That is the same reading as our flat K1-K6 on slot arms: the loop did not earn because the first pass already lands on the answer. It supports the "necessary is not deep" line and gives a name to the failure.
- 2026-10-20 Speculation: concrete test, training speed: replace full BPTT through the Poisson core with the one-step adjoint on the last pass (our `bptt_depth` knob already exists), and compare ms/step and K1-K6 at 5k against bptt_depth 8. Control: the same config at bptt_depth 8. Predicted behaviour: ms/step falls; K1-K6 falls or holds; a fall says full-depth credit is what builds the K-curve. Per the 2026-09 notes, full BPTT is the recipe whose loop earns, so expect a fall.
- 2026-10-20 Speculation: their LayerScale gate at init 0.75 plays the role of our 0.87 injection floor. Same quiet-block risk: the head starts near an identity map.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Their depth is chosen by a solver tolerance (convergence). Our rule is a Poisson draw with the main loss only; a convergence stop would hide the K-curve we measure. The claim "770M beats 1.3B at 2x tokens" is from the abstract and untested by us.
