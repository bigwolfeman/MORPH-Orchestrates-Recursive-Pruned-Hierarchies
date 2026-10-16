# Parcae testbed snapshot, 2026-10-16: the coarse-to-fine (spiral) arms

A reference copy of the Parcae testbed code and filings as of 2026-10-16. The earlier snapshot
([`../lxtul-testbed/`](../lxtul-testbed/README.md), commit `5d630a2`, 2026-10-07) holds the LXTUL slot-loop work. This
one holds the work since the 2026-10-14 redesign: the SpiralFormer-style coarse-to-fine loop, the hierarchy repeated
per loop ("cycling spiral"), and the span-concept arms (NCM, LCM, best-of-K plans).

This folder is a snapshot. It does not run from here, and MORPH does not import it.

## Source

- Repository: the Parcae codebase (`parcae_lm`, upstream https://github.com/sandyresearch/parcae, MIT). The testbed
  branches are local and are not pushed upstream.
- `lxtul/`: the whole `lxtul/` tree of branch `lxtul/cycle` at `b892109` (2026-10-16), from `git archive`, unchanged.
  It has the plain + pointer arm, flat references, every spiral arm, the cycling spiral and plain Parcae with deep
  supervision.
- `extras/<branch>/`: the files each other branch ADDS on top of `lxtul/cycle` (NCM, LCM, XM code and configs). These
  files also need that branch's versions of shared files (`train.py`, `evaluate.py`, `spiral.py`), which are not
  copied. Commits: [`extras/COMMITS.txt`](extras/COMMITS.txt).
- `filings/`: the newest filed copy of every 2026-10-14 to 2026-10-16 experiment across the branches, unchanged.
  [`filings/SOURCES.txt`](filings/SOURCES.txt) names the branch and path of each. The cycling-spiral prereg is
  included as `.PLANNED.md`: its runs were in flight when this snapshot was made.
- Run (from `lxtul/CLAUDE.md`): `PYTHONPATH=<parcae>:<MORPH> python -m lxtul.train arm=<arm>`. The data path uses
  MORPH's tokenizer (StarCoder2, vocab 49152), OpenWebText stream and row packer.

## Testbed

d 1024; 4 prelude, 6 core and 4 coda layers (Parcae blocks); the core is looped at a per-row Poisson depth (mean 6)
with Parcae's diagonal injection; seq 1024, batch 6, 5000 steps; MuonAdamW. Every arm has a 4-head output-only
pointer (copy) head. Scoring: CE on 480 fixed validation rows (510,003 tokens), paired across runs
(`lxtul/paired_eval.py`). Loop contribution is K1-K6 = CE at eval depth 1 minus CE at eval depth 6.

## Top variants (CE@6 paired, 2-seed means unless named; speed from `lxtul/bench_all.py`, batch 6)

| arm (config) | what it is | CE@6 | K1-K6 | train ms/step | TFLOP fwd+bwd |
| --- | --- | --- | --- | --- | --- |
| `plain_pointer` | plain Parcae loop + pointer head (1 seed) | 3.7256 | +0.0377 | 283.0 | 28.50 |
| `spiral_published` | SpiralFormer: one core pass per level, chunks 16, 8, 4, 2 | 3.6912 | 0 (fixed depth) | 138.0 | 8.73 |
| `spiral_pub_poisson2` | published spiral, finest level (chunks of 2) at Poisson depth | 3.6907 | +0.0071 | 234.2 | about 17 |
| `spiral_cycle_*` | the descent repeated per loop (Poisson cycle count, sampled start level, skips; end or HRM deep credit) | runs in flight | | 222.6 (end) | |
| `gpt8_pointer` | flat 8-layer reference | 3.6996 | | | |
| `flat14_pointer` | flat 14-layer reference | 3.6731 | | | |

Flat models are references only: at 5k steps a loop is before its knee, and the bar is loops against loops.

## What the filings say, in one line each

- Span-level loops do not use depth when the coda can read tokens: the spiral's span level was flat (+0.0001,
  worth 0.0005 nats); spans, slots and plans read the same way.
- The chunk size is a depth lever: chunk-2 loop +0.0063, chunk-4 loop +0.0029; Poisson at chunks of 4 +0.0008.
  Finer chunks use more depth.
- Block-Parcae (a series cut between blocks) costs +0.227; a loop memory closes 23 % of that.
- NCM span codes: the code costs more to predict than it saves (v1 0.226 per token cost vs 0.118 saved; v2 0.260),
  but the K-curve grows with the number of levels (+0.0003 to +0.0185 for 1 to 4 levels).
- LCM: the old readout was broken (scores above uniform); a variational LCM scores 4.41 (bound) but its concept
  collapses (KL 0.018 per token). Coda token dropout raises KL in order (0.018, 0.042, 0.109) and worsens the bound;
  depth stays flat.
- Best-of-8 span plans (XM): the coda uses the plan (a random plan costs 0.13 to 0.17), the loop's K-curve is
  +0.0012 vs +0.0002, and the bound is worse.
