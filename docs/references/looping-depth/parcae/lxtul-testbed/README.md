# LXTUL on a Parcae backbone: the testbed snapshot

A reference copy of a small testbed, built in October 2026 to test MORPH's LXTUL slot loop
without MORPH's own machinery (Hyper-Connections, ternary weights, CCA/CSA/HCA attention,
AdEMAMix). It ports the LXTUL forward and losses onto Parcae's blocks. The model and its
tools are about 2,300 lines of Python (`lxtul/*.py`, 2,332 lines), plus 700 lines of tests.

This folder is a snapshot. It does not run from here, and MORPH does not import it.

## Source

- Repository: the Parcae codebase (`parcae_lm`), local clone at `/home/wolfe/parcae`.
- Branch `lxtul-testbed`, snapshot at commit `5d630a2` (2026-10-07, "File pointer
  generation: ..."). The testbed was started at `4ba962e` (2026-10-06).
- Copied: `lxtul/*.py`, `lxtul/SPEC.md`, `lxtul/CLAUDE.md`, `lxtul/configs/` and
  `lxtul/tests/`, unchanged. Not copied: `__pycache__`, checkpoints, run outputs.
- Copied for the citations below: nine filings from the branch's `docs/experiments/`, into
  [`filings/`](filings/), unchanged.

## What it depends on

- **Parcae** (`parcae_lm`): the testbed subclasses `parcae_lm.models.parcae.parcae.Parcae`
  (`lxtul/model.py`) and reuses Parcae's attention, mixer and config code
  (`lxtul/attention.py`, `lxtul/build.py`). Parcae is MIT-licensed, (c) 2026 Hayden Prairie.
  Its license notice is in [`LICENSE-parcae`](LICENSE-parcae). The paper is in
  [`../parcae.md`](../parcae.md).
- **MORPH**: `lxtul/data.py` puts `MORPH_ROOT` on `sys.path` and uses MORPH's own tokenizer,
  OpenWebText stream, boundary rule and row packer, so its CE is comparable to MORPH's.
- How it was run (from `lxtul/CLAUDE.md`): `PYTHONPATH=/home/wolfe/parcae:$MORPH_ROOT
  python -m lxtul.train arm=<arm>`.
- `conftest.py` here (added in MORPH) stops a bare `pytest` at the MORPH root from
  collecting these tests. They need `parcae_lm` and a GPU.

## What it measured (5k steps, 480 paired rows; one seed unless named)

| reading | number | filing |
| --- | --- | --- |
| training speed, LXTUL | 23,528 tok/s (MORPH LXTUL 12,250 on the same rows) | [throughput](filings/2026-10-06-lxtul-throughput.md) |
| strict LXTUL vs plain Parcae | gap +0.294, K1-K6 +0.0585 | [5k panel](filings/2026-10-06-parcae-lxtul-5k-panel.md) |
| where the gap is | the strict geometry, and inside it copying | [gap decomposition](filings/2026-10-06-parcae-gap-decomposition.md) |
| + exact copy cache (output only) | gap +0.022 (two seeds: +0.022 / +0.023), K1-K6 +0.049 | [copy cache](filings/2026-10-06-parcae-copy-cache.md), [seed 2](filings/2026-10-06-parcae-copy-cache-seed2.md) |
| + learned pointer head | 0.026 ahead of plain (plain has no head), K1-K6 kept | [copy heads](filings/2026-10-06-parcae-copy-heads.md) |
| like for like: strict + pointer vs plain + pointer | +0.098 | [plain + pointer](filings/2026-10-06-parcae-plain-pointer.md) |
| the same, with the normalised (null-key) head | +0.045; cell keys -0.0074 more | [pointer cell keys](filings/2026-10-06-parcae-pointer-cellkey.md) |
| generation with the head | no repetition beyond plain at T 1; greedy loops like plain | [pointer generation](filings/2026-10-06-parcae-pointer-generation.md) |

The pointer head went to MORPH next
([`lab/experiments/mixed/2026-10-06-lxtul-pointer.md`](../../../../../lab/experiments/mixed/2026-10-06-lxtul-pointer.md)),
and became part of the 2026-10-07 TUL recipe
([winner note](../../../../../.agents/notes/implemented/architecture/2026-10-07-lxtul-pointer-ditto-winner.md)).
On the same rows and GPU, this testbed's LXTUL + pointer trains at 22.2k tok/s and MORPH's at 11.8k
([speed filing](../../../../../lab/experiments/mixed/2026-10-07-morph-vs-parcae-speed.md)).

## Known deviations from MORPH LXTUL

`lxtul/SPEC.md` was read off MORPH master `3f9aae2`. Its §0.2 and the `lxtul/model.py`
docstring list the backbone differences:

- one residual stream (MORPH: Hyper-Connections, 4 streams);
- Parcae's pre-norm blocks (ReLU² MLP, QK-norm, RoPE, value embeddings) and FlexAttention
  strict masks (MORPH: CCA + CSA/HCA attention);
- bf16 weights (MORPH: ternary STE with `norm_match` scales);
- the diagonal injection covers ALL channels at decay 0.447, so a do-nothing pass has gain
  0.447 (MORPH injects only its 320 ctx channels: floor 0.865);
- no per-layer x0 / bigram re-injection, no dropout, kv heads = heads (MORPH: 4 kv heads);
- the span decoder uses Parcae blocks with RoPE (MORPH: a learned position table);
- Parcae's optimizer and schedule (MuonAdamW, LR cooled to 0 at the end) (MORPH: AdEMAMix,
  flat LR after a 1000-step ramp).

One more, found 2026-10-07: **the gain hinge's pass draw.** Parcae draws a fresh pass every
step (`lxtul/model.py:413`, `torch.randint` on the global CPU stream). MORPH draws it with the
CPU RNG saved and restored (`morph/model/transformer.py:6431-6436` at `5e913cad`), so on the
GPU the same pass is drawn on every step of a run. That MORPH behaviour is an open bug
(owner decision pending).
