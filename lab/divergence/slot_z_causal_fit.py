"""What is a slot state worth to a reader that CANNOT see the future?

`slot_z_optimize.py` fits `z` by gradient descent on the tokens the coda is about to
predict and scores it on those same tokens. Its own docstring says what that is: an ORACLE
upper bound on the coda's capacity to use `z`, not a bound on what a causal inferencer
could deliver. Every "the reader has capacity" reading in this arc (0.9-2.6 nats) comes
from that number, and it has no causal twin. This is the twin.

THE PROCEDURE, per slot `s`:

  1. take the history up to and INCLUDING span `s`'s boundary token — exactly what the
     loop is given, no more;
  2. sample K continuations from a frozen full-context TEACHER (a plain MORPH checkpoint),
     temperature 1.0, each exactly as many tokens as the real next span has;
  3. optimise `z` against the coda CE of the FIRST `--fit-samples` of them, placed in span
     `s+1`'s row positions (inputs AND labels), with the real continuation nowhere in the
     objective;
  4. score that `z` on (a) the held-out sample(s) and (b) the UNTOUCHED real continuation.

WHAT LEAKS, named. The span LENGTH: a sample is drawn to exactly the real span's token
count, because the row's position layout is fixed and a shorter or longer continuation
would not occupy the same cells. Nothing else about the real continuation enters the fit —
`tests/test_slot_z_causal_fit.py` asserts it on the input ids AND on the label tensor the
loss receives.

WHAT IS COUNTERFACTUAL, also named, because it is the instrument's real weakness. With
`--fit-groups 1` (the cheap default) every graded span in the fit row is replaced by its
own sample at once, so a slot's fitted `z` is conditioned on a history whose EARLIER spans
are teacher samples rather than the real text. `--fit-groups G` splits the slots into G
interleaved groups and runs the fit once per group, replacing only that group's spans, so
the nearest counterfactual span sits G spans back: G=1 is "contaminated at distance 1",
G=4 is "contaminated at distance 4", at G times the cost. The SCORING rows are clean
either way — the real-continuation column runs on the untouched row with only `z`
substituted.

THE COLUMNS (all on the same positions: every token of a slot's next span, by
`_next_span.py`'s map, so offset 0 is predicted at the previous span's LAST token):

  ce_entry           the slot's prelude-entry state (`core_init`), no loop
  ce_loop            the loop's own exit state — the shipped forward
  ce_hindsight       `z` fitted on the REAL next span, same mask, same optimiser: the
                     ORACLE. `slot_z_optimize`'s number, restricted to these positions.
  ce_hindsight_full  `slot_z_optimize.optimise_z` verbatim (the FULL-row objective it
                     publishes), scored on these positions — the link back to the
                     existing readings.
  ce_causal_real     the causal fit, scored on the real continuation
  ce_causal_heldout  the causal fit, scored on the held-out sample(s)

`ce_loop - ce_causal_real` is what a better causal inferencer could still win at this
frozen coda and this frozen write. `ce_causal_real - ce_hindsight` is the part of the
oracle's advantage that is hindsight and can never be won.

WHAT IT CANNOT SAY. The teacher's samples are a MODEL's futures, not the world's: a weak
teacher makes the causal fit look easy and a teacher that is better than the student makes
it look hard, so the teacher's own CE on the same rows is reported beside every column and
the number is meaningless without it. It is still an UPPER bound on a causal producer —
gradient descent on K samples is not a procedure the loop could run at inference. It says
nothing about span 0, the row's open tail, or slots whose next span is incomplete, and
nothing about whether a DIFFERENT coda or a different write could use `z` better.

THE TEACHER'S DECODE is `morph.inference.plain_generate.generate_plain_batch` — eager,
one full recompute per step, ragged (every (slot, sample) prompt in a row is one batch
element with its own cursor), and already gated token-for-token against single-row greedy
in `tests/test_generation_sampling.py`. The KV-cache engine (`morph/inference/kv_cache.py`)
is NOT used: `decode_step` requires "all batch elements at the same absolute position", so
a batch of prefixes of different lengths cannot be prefilled together, and re-prefilling
per slot is O(L) decode steps a slot with no reuse. The design that WOULD win — walk the
row once with a batch of K, clone the cache at every boundary, decode the span from the
clone — needs a cache-clone helper the module does not have and a parity gate of its own;
it is named here and not built.

COST, arithmetic. One `generate_plain_batch` call per row over `n_slots * K` prompts for
`max_s len(span s+1)` steps, each step a full forward of that batch. At seq 1024, ~51
slots, K=4 and a 32-token cap that is ~204 prompts x 32 steps x ~1150 positions = 7.5M
position-forwards a row, about 6,500 single-row forwards. The z fit adds
`--steps` x (`--fit-groups` x `--fit-samples`) coda replays a batch. Run it at `--rows 8`
first and let the CI say whether more are needed.

Usage:
  python lab/divergence/slot_z_causal_fit.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --teacher plain=notul_panel_norm_match=/path/plain/step_5000.pt \
      --rows 8 --batch 2 --depth 6 --samples 4 --fit-samples 3 \
      --steps 200 --out .../causal_fit.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _next_span import next_span_positions  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402
from slot_z_optimize import ZSplit, guard_split_point, optimise_z  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

OFFSET_BUCKETS = [str(k) for k in range(8)] + ["8+"]
ARMS = ("entry", "loop", "hindsight", "hindsight_full", "causal")


# ── the teacher ───────────────────────────────────────────────────────────────


def row_stream(inp: torch.Tensor, layout, b: int) -> list[int]:
    """The row's TOKEN ids in order — what a plain model would have been fed.

    The packer interleaves `prefix_k` slot cells after every boundary token; a plain
    teacher has never seen `tul.slot_id` at any position, so the cells come out and what
    is left is exactly the stream segment this row was cut from (`_rows.pack_rows`
    verifies that identity when it packs).
    """
    return inp[b][~layout.slot_mask[b]].tolist()


def token_ordinals(layout) -> torch.Tensor:
    """`[B, L]` — the index of each TOKEN position within the row's token stream.

    Slot positions get -1. This is what turns a row position into a prompt length.
    """
    tok = (~layout.slot_mask).long()
    return torch.where(tok.bool(), tok.cumsum(1) - tok, torch.full_like(tok, -1))


@torch.no_grad()
def sample_continuations(teacher, stream: list[int], prompt_len: list[int],
                         n_tokens: list[int], k: int, temperature: float, seed: int,
                         chunk: int, slot_id: int, device) -> list[list[list[int]]]:
    """`out[i][j]` = the j-th of `k` sampled continuations of `stream[:prompt_len[i]]`.

    Every prompt is a PREFIX of the same row, so `generate_plain_batch`'s ragged batching
    puts all `len(prompt_len) * k` of them in one call: each element keeps its own cursor
    and reads its logits at its own last real position. Each continuation is truncated to
    `n_tokens[i]` — the real span's length, the one thing this instrument leaks.

    A draw of the structural `slot_id` RAISES. The teacher never saw that token in its
    corpus, so any mass on it is a bug or a broken checkpoint, and feeding it back into a
    TUL row would put a structural id at a token position the student has never seen one
    at. Loud, not silently substituted.
    """
    from morph.inference.plain_generate import generate_plain_batch

    prompts, want = [], []
    for i, (pl, nt) in enumerate(zip(prompt_len, n_tokens)):
        for j in range(k):
            prompts.append(stream[:pl])
            want.append((i, j, nt))
    out: list[list[list[int]]] = [[[] for _ in range(k)] for _ in prompt_len]
    for c0 in range(0, len(prompts), max(1, chunk)):
        sl = slice(c0, c0 + max(1, chunk))
        sub, sub_want = prompts[sl], want[sl]
        gen = generate_plain_batch(
            teacher, sub, max_new_tokens=max(nt for _i, _j, nt in sub_want),
            temperature=temperature, top_k=0,
            seeds=[seed + c0 + t for t in range(len(sub))], device=device)
        for (i, j, nt), toks in zip(sub_want, gen):
            out[i][j] = [int(t) for t in toks[:nt]]
    bad = sum(t == slot_id for row in out for s in row for t in s)
    if bad:
        raise SystemExit(
            f"the teacher sampled the structural slot id {slot_id} {bad} times. It never "
            "occurs in the corpus, so this is a broken checkpoint or a wrong slot_token — "
            "not something to substitute away.")
    return out


# ── counterfactual rows ───────────────────────────────────────────────────────


def _scatter(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor,
             keep: torch.Tensor, L: int) -> torch.Tensor:
    """`dst` with `src` written at `index` wherever `keep`, via a dump column.

    Masked-out entries address column `L` of a padded copy and are thrown away, so no
    `index_put_` with a ragged selection and no Python loop over slots.
    """
    B = dst.shape[0]
    pad = torch.cat([dst, dst.new_zeros(B, 1)], dim=1)
    idx = torch.where(keep, index, torch.full_like(index, L))
    pad.scatter_(1, idx.reshape(B, -1), src.reshape(B, -1))
    return pad[:, :L].contiguous()


def scored_mask(pos: torch.Tensor, valid: torch.Tensor, group_mask: torch.Tensor,
                L: int) -> torch.Tensor:
    """`[B, L]` bool — the positions this group's slots are answerable for."""
    B = pos.shape[0]
    keep = valid & group_mask.unsqueeze(-1)
    acc = torch.zeros((B, L + 1), dtype=torch.bool, device=pos.device)
    idx = torch.where(keep, pos, torch.full_like(pos, L))
    acc.scatter_(1, idx.reshape(B, -1), keep.reshape(B, -1))
    return acc[:, :L].contiguous()


def build_variant(inp, labels, layout, tok_pos, pos, valid, samples, group_mask):
    """`(cf_ids, cf_labels)` — the row with the graded spans replaced by sampled tokens.

    `samples` is `[B, S, J]` int64 (anything where `valid` is False is ignored) and
    `group_mask` `[B, S]` picks WHICH slots' spans are replaced. Three writes, all over
    positions the layout already owns:

      * span `s+1`'s token positions get the sample (`tok_pos`);
      * the positions that PREDICT them get it as their LABEL (`pos`) — span `s`'s
        boundary token for offset 0, the span's own tokens for the rest;
      * the slot's emitting cell, whose label is also span `s+1`'s first token, so the row
        stays internally consistent. It carries `tul.emit_weight` (0 on every arm here)
        and is never scored, but a stale label there would be a lie in the tensor.

    Everything outside the replaced spans is the real row, byte for byte.
    """
    L = inp.shape[1]
    keep = valid & group_mask.unsqueeze(-1)
    cf_ids = _scatter(inp, tok_pos, samples, keep, L)
    cf_lab = _scatter(labels, pos, samples, keep, L)
    emit = (layout.slot_index + layout.prefix_k - 1).unsqueeze(-1)         # [B, S, 1]
    cf_lab = _scatter(cf_lab, emit, samples[:, :, :1], keep[:, :, :1], L)
    return cf_ids, cf_lab


def masked(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, labels, torch.full_like(labels, -100))


# ── the fit ───────────────────────────────────────────────────────────────────


def optimise_z_multi(split, variants, layout, z0, lr: float, steps: int, grad_mask,
                     every: int = 20):
    """Adam on `z` against the MEAN CE over several counterfactual rows.

    The single-variant case is `slot_z_optimize.optimise_z` — same fp32 master cast to the
    core's dtype every call, same Adam, same "freeze a slot by zeroing its gradient"
    (Adam's update of a coordinate whose gradient is exactly 0 at every step is exactly 0).
    `tests/test_slot_z_causal_fit.py::test_one_variant_reproduces_slot_z_optimize` pins
    that equality bit for bit rather than asserting it here.

    `variants` is `[(snapshot, inp, labels)]`; `grad_mask` `[B, S]` is the slots this fit
    is allowed to move — pad slots, and under `--fit-groups` every slot outside the group.
    """
    dtype = z0.dtype
    m = grad_mask.reshape(*grad_mask.shape, *([1] * (z0.dim() - 2))).to(torch.float32)
    zm = z0.detach().float().clone().requires_grad_(True)
    opt = torch.optim.Adam([zm], lr=lr)
    curve = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for snap, inp, lab in variants:
            use_snapshot(split, snap)
            split.replay(inp, lab, layout, zm.to(dtype), want_groups=False, grad=True)
            ce = split.token_ce / len(variants)
            ce.backward()
            total += float(ce.detach())
            del ce
        if s % every == 0:
            curve.append((s, total))
        zm.grad.mul_(m)
        opt.step()
    return zm.detach().to(dtype), curve


def snapshot(split) -> tuple:
    """The (front, core, h0) the split just recorded — one cached row."""
    return (split.st["front"], split.st["core"], split.st["h0"])


def use_snapshot(split, snap) -> None:
    split.st["front"], split.st["core"], split.st["h0"] = snap


@contextlib.contextmanager
def quiet_aux(model):
    """Silence the auxiliary heads for the duration of a fit.

    The span decoder and the MUX are NOT in this instrument's objective —
    `_tul_group_losses` returns before `_forward_tul` adds them, which is exactly why
    `ZSplit.token_ce` is the trainer's token CE alone. But they are still COMPUTED on
    every replay, and the span decoder's readout is `[B, S, J, V]`: paying it `--steps`
    times a fit would dominate the run for a quantity nothing reads.
    """
    dec, beta = model.tul_spandec, float(model.cfg.tul.mux_beta)
    model.tul_spandec = None
    model.cfg.tul.mux_beta = 0.0
    try:
        yield
    finally:
        model.tul_spandec = dec
        model.cfg.tul.mux_beta = beta


# ── reading the coda ──────────────────────────────────────────────────────────


@torch.no_grad()
def ce_positions(model, xh: torch.Tensor, labels: torch.Tensor,
                 chunk: int = 1024) -> torch.Tensor:
    """`[B, L]` per-position CE from the coda states, with the slot id masked out.

    The aggregate columns come from the model's own `_tul_group_losses`; this exists for
    the per-slot and per-offset breakdowns, which need the CE before it is reduced. The
    two are CHECKED against each other once per batch (`--no-check` turns that off), so
    this is a view of the shipped number rather than a second definition of it.
    """
    w = model.embed.lm_weight()
    B, L, C = xh.shape
    flat = xh.reshape(-1, C)
    lab = labels.reshape(-1)
    out = torch.empty(B * L, dtype=torch.float32, device=xh.device)
    mask_id = int(model.cfg.tul.slot_id)
    for i in range(0, B * L, chunk):
        logits = (flat[i:i + chunk] @ w.t()).float()
        logits[:, mask_id] = torch.finfo(torch.float32).min
        out[i:i + chunk] = F.cross_entropy(logits, lab[i:i + chunk].clamp(min=0),
                                           reduction="none")
    return out.reshape(B, L)


def summarise(row_sum, row_cnt, off_sum, off_cnt, slot_sum, slot_cnt, heldout,
              teacher_ce, n_boot: int = 2000) -> dict:
    """The JSON `results` block: one entry per arm, plus the two scale readings.

    Every arm carries `ce` (the token-weighted mean over the graded next-span positions),
    `vs_loop` (a paired bootstrap over ROWS against the loop's own state — the arm is the
    first term, so a NEGATIVE point estimate is better than the loop), a `by_offset` table
    on `span_budget_profile.py`'s buckets and a `by_slot` vector indexed by slot position
    in the row. `teacher_ce` is the teacher's own CE on the same positions: without it the
    causal columns have no scale, because they are bounded by how good the teacher's
    futures were.
    """
    cnt = np.asarray(row_cnt, dtype=np.float64)
    ocnt = np.concatenate(off_cnt, axis=0)
    res: dict = {"n_scored": float(cnt.sum()),
                 "teacher_ce": teacher_ce["sum"] / max(teacher_ce["n"], 1.0),
                 "n_teacher_scored": teacher_ce["n"],
                 "ce_heldout_causal": heldout["sum"] / max(heldout["n"], 1.0),
                 "n_heldout": heldout["n"], "arms": {}}
    for name in ARMS:
        s = np.asarray(row_sum[name], dtype=np.float64)
        osum = np.concatenate(off_sum[name], axis=0)
        res["arms"][name] = {
            "ce": float(s.sum() / cnt.sum()),
            "vs_loop": paired_bootstrap_ci(s, np.asarray(row_sum["loop"], dtype=np.float64),
                                           cnt, n_boot=n_boot),
            "by_offset": {OFFSET_BUCKETS[i]:
                          {"n": float(ocnt[:, i].sum()),
                           "ce": float(osum[:, i].sum() / ocnt[:, i].sum())}
                          for i in range(len(OFFSET_BUCKETS)) if ocnt[:, i].sum() > 0},
            "by_slot": [float(slot_sum[name][i] / slot_cnt[i]) if slot_cnt[i] > 0 else None
                        for i in range(len(slot_cnt))]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH (the SLOT-LOOP arm)")
    ap.add_argument("--teacher", required=True,
                    help="LABEL=CONFIG=PATH (a PLAIN checkpoint, tul.activate_at never)")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--samples", type=int, default=4, help="K continuations per slot")
    ap.add_argument("--fit-samples", type=int, default=3,
                    help="how many of them the fit sees; the rest are held out")
    ap.add_argument("--fit-groups", type=int, default=1,
                    help="split the slots into G interleaved groups and fit each on a row "
                         "where ONLY that group's spans are replaced (G x the cost, "
                         "counterfactual history pushed G spans back)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen-chunk", type=int, default=64,
                    help="prompts per generate_plain_batch call (memory, not FLOPs)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the per-batch check that the per-position CE map "
                         "reproduces _tul_group_losses on the scored set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if not (1 <= a.fit_samples < a.samples):
        raise SystemExit("--fit-samples must be at least 1 and leave at least one "
                         "held-out sample (it is the only honest generalisation column)")

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    s_label, s_config, s_path = a.ckpt.split("=", 2)
    t_label, t_config, t_path = a.teacher.split("=", 2)

    cfg = build_cfg(s_config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_z_causal_fit needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false)")
    model, step = load_ckpt(cfg, s_path, "cuda", tul_rt.model_cfg)
    guard_split_point(model)
    tc = model.cfg.tul
    if float(tc.plast_weight) != 1.0 or float(tc.emit_weight) != 0.0:
        raise SystemExit(
            f"this instrument reports a PLAIN per-token CE over the next span, which is "
            f"the model's weighted CE only at plast_weight 1.0 / emit_weight 0.0 (got "
            f"{float(tc.plast_weight)} / {float(tc.emit_weight)}). Refusing rather than "
            "reporting a weighted mean under an unweighted name.")

    t_cfg = build_cfg(t_config, ["model.use_kernels=false"])
    t_rt = build_tul_runtime(t_cfg)
    if t_rt is not None:
        raise SystemExit(
            f"--teacher must be a PLAIN checkpoint (tul.activate_at: never); {t_config} "
            "builds a TUL runtime, and a teacher with slot cells does not generate the "
            "token stream this instrument substitutes.")
    teacher, t_step = load_ckpt(t_cfg, t_path, "cuda", None)
    teacher.eval()
    teacher.requires_grad_(False)

    notes = {
        "slot_depth_fixed": [int(tc.slot_depth_fixed), a.depth],
        "slot_gain_lambda": [float(model.cfg.slot_gain_lambda), 0.0],
        "leak": "the span LENGTH only — a sample is drawn to exactly the real span's "
                "token count because the row's cells are fixed",
        "counterfactual_history": f"--fit-groups {a.fit_groups}: in a fit row the nearest "
                                  f"replaced span other than the graded one sits "
                                  f"{a.fit_groups} spans back",
        "objective": "_tul_group_losses(...)['loss'] masked to the graded next-span "
                     "positions, BEFORE the sigreg / gain / mux / spandec terms",
        "aux": "the span decoder and the MUX are switched off for every record, fit and "
               "replay: they are not in the objective (_tul_group_losses returns before "
               "_forward_tul adds them) and the decoder's [B, S, J, V] readout would "
               "dominate the run",
        "teacher_decode": "generate_plain_batch (eager, ragged, one full recompute per "
                          "step); the KV engine's decode_step needs one absolute position "
                          "for the whole batch, so a batch of prefixes cannot be prefilled",
        "upper_bound": "gradient descent on K sampled futures is not a procedure the loop "
                       "could run at inference: this is still an UPPER bound on a causal "
                       "producer, a tighter one than slot_z_optimize's oracle",
    }
    tc.slot_depth_fixed = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth or model.cfg.max_depth))
    model.cfg.slot_gain_lambda = 0.0
    model.eval()
    model.requires_grad_(False)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    # per-ROW sums / counts, the bootstrap unit
    row_sum: dict[str, list[float]] = {k: [] for k in ARMS}
    row_cnt: list[float] = []
    off_sum: dict[str, list[np.ndarray]] = {k: [] for k in ARMS}
    off_cnt: list[np.ndarray] = []
    slot_sum: dict[str, np.ndarray] = {}
    slot_cnt: np.ndarray | None = None
    heldout: dict[str, float] = {"sum": 0.0, "n": 0.0}
    teacher_ce: dict[str, float] = {"sum": 0.0, "n": 0.0}
    check_err = 0.0
    curves: dict[str, list] = {}
    n_rows = 0

    with ZSplit(model) as split:
        for bi, (inp, labels, layout, _idx) in enumerate(batches):
            inp, labels = inp.cuda(), labels.cuda()
            layout = layout.to("cuda")
            B, L = inp.shape
            pos, valid = next_span_positions(layout, labels, inp)
            S, J = pos.shape[1], pos.shape[2]
            tok_pos = (layout.slot_index + layout.prefix_k).unsqueeze(-1) + torch.arange(
                J, device=inp.device).view(1, 1, J)
            tok_pos = torch.where(valid, tok_pos, torch.zeros_like(tok_pos))
            all_mask = scored_mask(pos, valid, layout.slot_valid, L)
            if slot_cnt is None:
                slot_cnt = np.zeros(S)
                slot_sum = {k: np.zeros(S) for k in ARMS}

            # ── the teacher's futures ────────────────────────────────────────
            ords = token_ordinals(layout)
            samples = torch.zeros((B, S, J, a.samples), dtype=torch.long,
                                  device=inp.device)
            t_sum = t_n = 0.0
            for b in range(B):
                live = [s for s in range(S) if bool(valid[b, s].any())]
                if not live:
                    continue
                st = row_stream(inp, layout, b)
                p_len = [int(ords[b, pos[b, s, 0]]) + 1 for s in live]
                n_tok = [int(valid[b, s].sum()) for s in live]
                got = sample_continuations(teacher, st, p_len, n_tok, a.samples,
                                           a.temperature, a.seed + 1000 * bi + b,
                                           a.gen_chunk, int(tc.slot_id), "cuda")
                for i, s in enumerate(live):
                    for j in range(a.samples):
                        samples[b, s, :n_tok[i], j] = torch.tensor(
                            got[i][j], dtype=torch.long, device=inp.device)
                # the teacher's OWN CE on the graded positions of this row — the scale
                # every column below has to be read against
                with torch.no_grad():
                    tl = teacher(torch.tensor(st, device=inp.device)[None])["logits"]
                    ce_t = F.cross_entropy(
                        tl[0, :-1].float(),
                        torch.tensor(st[1:], device=inp.device), reduction="none")
                    # the row's LAST token predicts a token outside the row, so it has no
                    # label here and is dropped from the teacher's reference number only
                    sel = ords[b][all_mask[b]]
                    sel = sel[sel < len(st) - 1]
                    t_sum += float(ce_t[sel].sum())
                    t_n += float(sel.numel())
            teacher_ce["sum"] += t_sum
            teacher_ce["n"] += t_n

            # ── cache one forward per row variant ────────────────────────────
            groups = [((torch.arange(S, device=inp.device) % a.fit_groups == g)
                       .unsqueeze(0).expand(B, S) & layout.slot_valid)
                      for g in range(a.fit_groups)]
            real_lab = masked(labels, all_mask)
            torch.manual_seed(a.seed + bi)
            with quiet_aux(model):
                out, z_loop, h0 = split.record(inp, real_lab, layout, want_groups=False)
                snap_real = snapshot(split)
                ce_loop_chk = float(split.token_ce)
                del out
                var: dict[tuple[int, int], tuple] = {}
                for g, gm in enumerate(groups):
                    gmask = scored_mask(pos, valid, gm, L)
                    for k in range(a.samples):
                        cf_ids, cf_lab = build_variant(inp, labels, layout, tok_pos, pos,
                                                       valid, samples[..., k], gm)
                        cf_lab = masked(cf_lab, gmask)
                        split.record(cf_ids, cf_lab, layout, want_groups=False)
                        var[(g, k)] = (snapshot(split), cf_ids, cf_lab)

                # ── the fits ─────────────────────────────────────────────────
                z_causal = z_loop.clone()
                for g, gm in enumerate(groups):
                    zg, curve = optimise_z_multi(
                        split, [var[(g, k)] for k in range(a.fit_samples)], layout,
                        z_loop, a.lr, a.steps, gm)
                    curves.setdefault(f"causal_g{g}", []).append(curve)
                    z_causal = torch.where(
                        gm.reshape(*gm.shape, *([1] * (z_loop.dim() - 2))), zg, z_causal)
                z_hind, curve = optimise_z_multi(
                    split, [(snap_real, inp, real_lab)], layout, z_loop, a.lr, a.steps,
                    layout.slot_valid)
                curves.setdefault("hindsight", []).append(curve)
                use_snapshot(split, snap_real)
                z_full, curve = optimise_z(split, inp, labels, layout, z_loop, a.lr,
                                           a.steps, layout.slot_valid)
                curves.setdefault("hindsight_full", []).append(curve)

                # ── the columns, all on the REAL row ─────────────────────────
                use_snapshot(split, snap_real)
                zs = {"entry": h0, "loop": z_loop, "hindsight": z_hind,
                      "hindsight_full": z_full, "causal": z_causal}
                maps = {}
                for name, z in zs.items():
                    split.replay(inp, real_lab, layout, z, want_groups=False)
                    maps[name] = ce_positions(model, split.xh, labels)
                    if not a.no_check:
                        mm = float((maps[name] * all_mask).sum() / all_mask.sum())
                        check_err = max(check_err, abs(mm - float(split.token_ce)))

                # ── the held-out column ──────────────────────────────────────
                for g, gm in enumerate(groups):
                    for k in range(a.fit_samples, a.samples):
                        snap, cf_ids, cf_lab = var[(g, k)]
                        use_snapshot(split, snap)
                        split.replay(cf_ids, cf_lab, layout, z_causal, want_groups=False)
                        n = float((cf_lab != -100).sum())
                        heldout["sum"] += float(split.token_ce) * n
                        heldout["n"] += n

            # ── per row / per slot / per offset ──────────────────────────────
            am = all_mask.float()
            row_cnt.extend(am.sum(1).tolist())
            off_b = np.minimum(np.arange(J), 8)
            oc = np.zeros((B, len(OFFSET_BUCKETS)))
            vn = (valid & layout.slot_valid.unsqueeze(-1)).cpu().numpy()
            for b in range(B):
                np.add.at(oc[b], off_b[np.where(vn[b])[1]], 1.0)
            off_cnt.append(oc)
            slot_cnt += vn.sum(axis=(0, 2))
            for name in ARMS:
                cm = maps[name] * am
                row_sum[name].extend(cm.sum(1).tolist())
                g_ce = (maps[name].gather(1, pos.reshape(B, -1)).reshape(B, S, J)
                        * torch.as_tensor(vn, device=inp.device).to(
                            maps[name].dtype))
                gn = g_ce.cpu().numpy()
                os_ = np.zeros((B, len(OFFSET_BUCKETS)))
                for b in range(B):
                    idx = np.where(vn[b])
                    np.add.at(os_[b], off_b[idx[1]], gn[b][idx])
                off_sum[name].append(os_)
                slot_sum[name] += gn.sum(axis=(0, 2))
            n_rows += B
            print(f"  batch {bi + 1}/{len(batches)}: loop CE {ce_loop_chk:.4f}", flush=True)
            torch.cuda.empty_cache()

    if check_err > 1e-3:
        raise SystemExit(
            f"the per-position CE map and _tul_group_losses disagree by {check_err:.3e} "
            "on the scored set — the breakdown is not a view of the shipped number")

    res = summarise(row_sum, row_cnt, off_sum, off_cnt, slot_sum, slot_cnt,
                    heldout, teacher_ce, a.boot)

    rec = {"student": s_label, "student_config": s_config, "step": step,
           "teacher": t_label, "teacher_config": t_config, "teacher_step": t_step,
           "rows": n_rows, "batch": a.batch, "depth": a.depth, "samples": a.samples,
           "fit_samples": a.fit_samples, "fit_groups": a.fit_groups, "steps": a.steps,
           "lr": a.lr, "temperature": a.temperature,
           "map_vs_group_losses_abs_err": check_err,
           "notes": notes, "results": res, "curves": curves}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)

    print(f"\n{s_label} step {step}  x  teacher {t_label} step {t_step} — "
          f"{n_rows} rows, depth {a.depth}, K {a.samples} (fit {a.fit_samples}), "
          f"{a.fit_groups} group(s)")
    print(f"  CE-map vs _tul_group_losses |delta| {check_err:.3e}")
    print(f"  teacher CE on the same positions {res['teacher_ce']:.4f}")
    print("  " + "arm".ljust(18) + "CE".rjust(9) + "vs loop".rjust(11) + "95% CI".rjust(22))
    for name in ARMS:
        v = res["arms"][name]
        ci = v["vs_loop"]
        print(f"  {name:18s}{v['ce']:9.4f}{ci['point']:+11.4f}"
              f"   [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
    print(f"  {'causal (held-out)':18s}{res['ce_heldout_causal']:9.4f}"
          f"   over {res['n_heldout']:.0f} positions")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
