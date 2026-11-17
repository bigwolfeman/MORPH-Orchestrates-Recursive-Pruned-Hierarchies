# Paper note: CRAM (classical energies for reasoning)

- **Title:** Classical Energies Learn to Reason 1,000x Faster than Looped Models. The model is called CRAM in the text.
- **arXiv or URL:** 2611.17309 (v1). Read: HTML text characters 0 to 137k (appendices D to F included).
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it
- 2026-11-17: Wolfe asked about the energy-based reasoning claim. It is the energy-form loop we measured at chance on graph search above 6 hops, so we check what it does on graphs.

## 2. What the paper shows
- 2026-11-17 (Eq 1-3, Sec 3): the energy is a Hopfield-Tank style mean-field free energy: belief entropy + learned pairwise costs between cells + per-cell input biases. Inference is an Euler relaxation then focused Gibbs search over single-cell moves, clues clamped, stop at an energy threshold or when no move lowers energy.
- 2026-11-17 (Eq 5, Fig 4): training is pseudolikelihood on solved boards. It never differentiates through the relaxation. PL reaches 99 % Sudoku in 48 ms; full BPTT on the same energy reaches 48.8 % after 5 h.
- 2026-11-17 (Table 2): Sudoku-Extreme 99.996 % (the table prints 100.00), Maze-Unique 100 %. Fit time 0.04 s and 0.13 s vs 1.2 h to 15.6 h for looped baselines. Parameters: Sudoku 7.7K nonzero, Maze 18 nonzero. The "1000x" is fit wall clock, CRAM warm median of 7 vs baseline cold single runs.
- 2026-11-17 (Sec 5, App F): tasks are constraint puzzles a pairwise energy represents exactly (Sudoku, Maze, N-Queens, graph coloring). Maze-Hard: the energy alone solves 0 of the test mazes because it has nowhere to store distances. Adding a hidden layer (11 more parameters) that settles to a distance field solves 999 of 1000.
- 2026-11-17: No language-model result. No multi-hop or general graph benchmark. The Maze-Hard distance field is the only hop-like result and it needed new storage.
- 2026-11-17 (why many steps): not argued. Search runs 10k to 20k moves; descent does not guarantee a correct answer (App E).
- Authors' caveats: N-Queens recipe and maze start were chosen after seeing test or dev data; baselines reproduced with gaps (GRAM 93.61 vs 96.1 published); priors (locality, augmentation) are given.

## 3. Our speculation
- 2026-11-17 Speculation: the win is that the task rules ARE a pairwise energy, so a convex-ish local fit replaces a deep net. Web text has no such exact energy. Hence no transfer claim.
- 2026-11-17 Speculation: the Maze-Hard failure (no storage for distance) agrees with our item 6: reach needs per-pass state that carries hops.

## 4. Corrections
- None recorded yet.

## 5. Where our findings disagree
- None recorded yet. Agrees with our chance-level energy loop on graph search: the paper only fixed a maze after adding hidden state.

## 6. Triage
Score 0. Puzzle-only, no text, no loop contribution measure.
