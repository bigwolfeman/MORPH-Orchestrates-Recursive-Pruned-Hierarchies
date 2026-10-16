# Experiment records

One file per experiment, named `YYYY-MM-DD-slug.md`, filed by outcome. The date is the
date the experiment RAN, not the date the note was edited.

| Folder | Holds |
|---|---|
| `planned/` | Pre-registered question, method and predictions, written BEFORE the run. |
| `successes/` | Runs whose main prediction was "supported". |
| `results/` | Runs that produced a usable comparison, plus the data behind their figures. |
| `failures/` | Runs whose main prediction was "not supported" or "can't tell". A "can't tell" record names the next planned experiment. A record here is still a result: it is where the campaign's real cost is written down. |
| `mixed/` | Closed to new records (historical). One main prediction makes a mixed verdict impossible. Existing records stay as written. |
| `figures/` | Rendered PNGs for the records above. |

## How to write and grade an experiment

This is the method children learn, used the same way here. Write one planned file per question. Commit it before the run.

1. **Ask a question.** One sentence: "Does changing X change Y?" One question per experiment.
2. **Make a guess (hypothesis).** "I think X will make Y better / worse / no different, because ___." The "because" names the mechanism.
3. **Say what you expect to see (prediction).** A behavior against a control, not a number to hit. Allowed forms: direction (A better / worse / same as B), ordering (A < B < C), shape (the K-curve slopes down / is flat), sign. A number is allowed only when it is a fixed anchor (uniform ln vocab, 0, an oracle ceiling). "The same" means inside the seed spread of the controls. You measure that spread from repeated control runs. You do not choose it. Write at most three predictions. ONE is the main prediction, and only it answers the question.
4. **Make it a fair test.** Change one thing. The control is the same model without the change, on the same rows. Run enough seeds to know what "no change" looks like. If no spread has been measured, measuring it is the first job.
5. **Write what each answer makes you do.** "If supported, we do A. If not, we do B." If both answers lead to the same next step, do not run it.
6. **Observe.** Report paired differences with their intervals, and the whole curve, not one cut point.
7. **Conclude,** with one of three words for the main prediction:
   - **supported**: the result matches it, outside noise.
   - **not supported**: the result is the opposite, or there is no effect where one was expected.
   - **can't tell**: the result is inside noise. The test was too small. This is not a verdict on the idea. Write the bigger test that would answer it.
8. **Next question.** What the result taught, and what you try next.

### Gates are not predictions

Finite loss, peak memory, head-off checks and token counts are GATES. If a gate fails, the run is broken: fix it and rerun. Gates never count toward a verdict. Speed is a measurement. Report it. It is not a pass/fail line unless speed is the question.

### Rules

- The verdict comes from the main prediction, never from a count such as "5 of 11".
- No invented thresholds. A number in a prediction that is not a fixed anchor or a measured noise band gets deleted.
- No probabilities on predictions. An audit of 51 predictions found stated probabilities 10 to 25 points too high in the middle range.
- Read the earlier filings on the same arm before you write a prediction.
- Never edit predictions after the run starts. Method changes go in a dated amendment.

### Filing

| Verdict on the main prediction | Folder |
|---|---|
| supported | `successes/` |
| not supported | `failures/` |
| can't tell | `failures/`, with the next planned experiment named |

A filing leads with the main prediction's word and the paired number behind it.

### Example (Block-Parcae, rewritten)

- **Question.** Can a loop that runs over chunks in series replace the token loop?
- **Hypothesis.** I think the coda can decode tokens from the chunk states alone, because the spiral's chunk levels carry most of its gain.
- **Main prediction.** Block-4 CE is the same as plain Parcae + pointer (inside seed spread), and its K-curve slopes down.
- **What each answer does.** If supported, build the series design out. If not, keep the token stream in the loop.
- **Result.** 0.25 nats worse, far outside noise. Not supported. (The old filing said "failure 5/11" and hid this.)

A `planned/` file moves to `successes/` or `failures/` when it has an answer. It keeps its
original name and its predictions are never edited after the fact; corrections go in a
numbered `## Amendment N` section with the date.

This tree is separate from `.agents/notes/`, which records DECISIONS (why we chose X,
what we gave up). An experiment record says what happened when we measured. If a
measurement changes a decision, the note links the record.

## Figures

`figures/` holds matplotlib PNGs regenerated from wandb, not TikZ diagrams — the
architecture diagrams and their LaTeX sources live in `docs/figures/` and use a
different pipeline (see `docs/figures/MANIFEST.md`). Do not mix the two.

Regenerate everything:

```bash
python scripts/plot_tul_arms.py              # all four
python scripts/plot_tul_arms.py --only ce    # ce | divergence | efficiency | order | bakeoff | decode
```

The script pulls run history from the `morph-tul` wandb project by run id, so a figure
cannot drift from the run it describes. The one exception is the order-parameter probe,
which never went to wandb: its numbers live in `results/tul_order_parameter.csv`, next to
the figure that reads them.

**Colour:** the palette is Okabe-Ito, which is safe for red-green colour blindness, and
no figure uses colour as its only channel — every series also carries its own line style
and marker. Keep it that way.

| Figure | Shows |
|---|---|
| `tul_arms_val_ce.png` | Validation CE for all six arms, plus a tight-axis panel where the 0.007-0.056 nat differences between survivors are actually visible. |
| `tul_arms_divergence.png` | The detonation: CE turning upward, and the gradient norm reaching 3.0e11 while the capped arms stay near 1. |
| `tul_arms_efficiency.png` | Final CE against throughput — the A0 / A1c / A3 trade at equal tokens. |
| `tul_decode_modes.png` | rep4 by decode mode for every arm, with CE beside each name and a real-text line. The figure that exists because one decode mode hid a greedy repetition loop. |
| `tul_bakeoff.png` | The gate bake-off live: val CE, the degeneration watch (rep4 / distinct3), and the halting policy's mean chosen depth. Arms appear as they start; stale wandb ids are dropped by name and listed in the caption. |
| `tul_order_parameter.png` | The core-map order parameter across three estimator passes, against the diverged band. |


## Generation samples

`val CE cannot see degeneration.` A repetition loop scores an excellent perplexity — a
measured 1.46 against real text's 32.44 — so every fluency number needs a diversity
number beside it. `rep4` and `distinct3` are that pair and are logged by the training
loop as `gen/*`.

The training loop's own generation test decodes at **one** setting, `temperature 0.8 /
top-k 50`. That is not enough. Greedy is where a repetition loop actually appears, and
pure ancestral sampling is where a collapsed readout shows up as "sampling is identical
to greedy". For the full table:

```bash
python scripts/tul_samples.py \
  --ckpt gate_20k=tul_gate=checkpoints/morph/tul-gate/step_20000.pt --halt
```

It scores greedy, top-k 50 at t=0.8, and pure ancestral t=1.0, each with rep4 /
distinct3 / span geometry, plus a REAL TEXT row scored by the same code as the anchor.
Rank nothing against a row whose distinct3 is far from the real-text value.

**Run it on finished checkpoints, not during a campaign.** `gate_bakeoff.sh` launches
each arm as its own process, so editing the training loop mid-campaign makes later arms
differ from earlier ones by more than the variable under test — and a second model on
the GPU can OOM a training arm that is already sitting on ~3.8 GB of margin.
