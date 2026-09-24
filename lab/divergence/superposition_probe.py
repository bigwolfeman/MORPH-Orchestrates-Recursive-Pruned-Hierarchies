"""Superposition, committed point, or blur: what does a deterministic slot state hold?

Spec: the 3c instrument of the 2026-09-23 superposition report (Reasoning by
Superposition re-read; decision rule R2). No training. The model's OWN eval forward
(`tul_forward_ablated`, which builds the arm's TG / strict kwargs itself through
`_tul_tg_kwargs`) on the SAME validation stream, packer and row cut `core_depth_sweep.py`
reads, so every per-token array pairs with a sweep's by stream index.

TERMS (one meaning each):

  span s        the token positions with ``bag_id == s``. Scored spans are
                ``1 <= s < max_slots``: span 0 reads no cell and the dump bin is not a span.
  own cell      the prefix cells of slot ``s - 1``, the slot that closes span ``s - 1`` and
                sits right before span s. Under ``tg_geometry: strict`` a token of span s
                reads these cells (and, at ``tg_coda_prefix_reach: all``, every older
                slot's cells too). On a fan arm (``tul.fan_mix: all``) the own cell is ALL
                K prefix cells of slot ``s - 1``, stream i in cell i.
  offset        a token position's index among span s's token positions (0-based, the
                ``worth_profile.py`` cumcount). Offset 0 is the FIRST position that reads
                the own cell. Its input is the span's first token and its label is the
                token the ``span_first`` column of ``core_depth_sweep.py`` scores. This
                file calls that label "the first token" (``t0``). The span's opening token
                itself is predicted at the previous span's boundary position, which under
                strict cannot read the own cell, so it is not scored here.
  bucket        the rank of the true ``t0`` under the coda's distribution at offset 0 with
                the own cell: ``top1``, ``top2`` or ``other``.
  worth         ``CE(control) - CE(own)`` at one token position, same rows, same forward
                apart from the cells. Positive = the own cell helped.
  controls      ``xrow``  every slot's cells replaced by the same-index slot of ANOTHER
                          row (a row derangement; slot s takes slot ``s mod n_valid`` of the
                          source row). Removes the content correspondence and the
                          document. THE PRIMARY CONTROL (the spec's "shuffled from another
                          row").
                ``xrow2`` another cross-row map that differs from ``xrow`` at every slot
                          whose source row has >= 2 slots: a fresh row derangement, and slot
                          s takes the source row's slot ``(s + k) mod n_valid`` with a
                          random ``k != 0``. (With 2 rows there is ONE row derangement, so a
                          second derangement alone would copy ``xrow``.) Used only as the
                          null of the decode clustering test (below).
                ``row``   slots permuted WITHIN the row by a random cyclic permutation (no
                          fixed slot when the row has >= 2 slots). The existing
                          ``plan_mode="shuffle"`` semantics without its fixed points.
                ``zero``  the written cells zeroed (``plan_mode="zero"``).
                Every shuffle runs through the model's own ``plan_mode="shuffle"`` seam
                (``_tul_plan_ablate``, the one point every slot path's cells pass on the
                way to ``prefix_project``); this file replaces only the PERMUTATION, on the
                model instance, for the duration of one forward. Whole slots move, so a fan
                slot's K cells move together. The override counts its calls and the probe
                raises if a shuffled forward never reached it.

THE MEASUREMENT (spec 3c.1):

  1. At offset 0 read the own-cell distribution; bucket the span by the rank of ``t0``.
  2. On offsets 1..n (after ``t0`` is seen) the per-bucket mean worth, with a 95 %
     bootstrap CI over 1,024-token stream blocks (``sweep_score.py``'s unit), and the
     same at offset 0 and per offset bin (``_earning.BINS``: the sharpening read).
  3. The ratio ``worth(top2) / worth(top1)`` on offsets 1..n under ``xrow``, with its CI
     (the same block resamples, so the two buckets stay paired).

DECISION RULE (``classify``, on the xrow worth at offsets 1..n):

  blur           every bucket's worth CI lies inside ``[-eps, +eps]`` (default 0.01 nats)
  inconclusive   not blur, but top1's CI does not exclude 0 (no denominator to take a
                 ratio against)
  superposition  ratio >= 0.5
  committed      ratio < 0.25, or top2's worth < 0
  intermediate   0.25 <= ratio < 0.5
  ``ci_supported`` says whether the ratio's CI lies wholly on the verdict's side.

KNOWN CONFOUND of the spec's bucket: it is the own cell's own ranking, so a span whose
cell is "good" lands in top1 more often and its later tokens also profit more. That
inflates worth(top1) against worth(top2) for reasons that are about cell quality, not
branching, and it biases the rule toward ``committed``. THE CONTEXT BUCKET removes it:
``bucket_ctx`` ranks ``t0`` under the ``xrow`` forward (the same context with another
row's cells), which cannot know how good the own cell is. Under it, top1 / top2 are the
branches the context alone favours. The same worth, ratio and rule are reported on it
(``*_ctx`` keys, ``decision_ctx``). A superposed cell helps the continuation whichever
context branch came true; a cell committed to the context's favourite helps top1 only.

DECODE CLUSTERING (spec 3c.2, first-token form): per span draw 16 tokens from the own
distribution at offset 0 (seeded); a cluster is one distinct drawn token; its gain is
``log p_own(c) - log p_xrow(c)``. The spec's statistic is "two or more clusters with a
positive gain". Sampling from p_own biases that gain positive BY CONSTRUCTION (its mean
is ``KL(p_own || p_xrow) >= 0``), so the same statistic is computed on the matched null:
draw from ``p_xrow2`` and score ``log p_xrow2(c) - log p_xrow(c)``. Report the difference.
NOT implemented: the spec's 8-token continuations clustered by first bigram. That needs an
autoregressive decode through the packed strict layout, and a sampled token at offset 1
changes span s's own slot state, which the next span reads.

Usage (GPU):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/superposition_probe.py \\
      --ckpt /home/wolfe/morph-to/checkpoints/morph/slot-spandec-strict/step_5000.pt \\
      --config tul_slot_spandec_strict --rows 192 --batch 3 --out .../superpos_ruler.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _earning import BINS  # noqa: E402  (ONE home for the offset bins)

BUCKETS = ("top1", "top2", "other")
CONTROLS = ("xrow", "xrow2", "row", "zero")
BLOCK = 1024           # stream tokens per bootstrap unit (sweep_score.BLOCK)
N_BOOT = 2000
N_DRAW = 16            # decode-clustering draws per span
MIN_BLOCKS = 50        # below this the block bootstrap is flagged, not trusted


# ── layout bookkeeping ──────────────────────────────────────────────────────────────

def span_offsets(layout, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(bag [B, L], offset [B, L], scored [B, L])`` on CPU.

    ``offset`` is the position's index among its span's TOKEN positions (``-1`` at slot
    positions). ``scored`` = token position, label >= 0, ``1 <= bag < max_slots``, and the
    span's offset-0 label is valid (a span with no readable ``t0`` has no bucket, so none
    of its tokens is scored)."""
    bag = layout.bag_id.cpu()
    tok = ~layout.slot_mask.cpu()
    B, L = bag.shape
    S = int(layout.slot_valid.shape[1])
    off = torch.full((B, L), -1, dtype=torch.long)
    scored = torch.zeros((B, L), dtype=torch.bool)
    lab = labels.cpu()
    for b in range(B):
        counts: dict[int, int] = {}
        first_ok: dict[int, bool] = {}
        for p in torch.nonzero(tok[b]).flatten().tolist():
            s = int(bag[b, p])
            o = counts.get(s, 0)
            counts[s] = o + 1
            off[b, p] = o
            if o == 0:
                first_ok[s] = bool(lab[b, p] >= 0)
            if 1 <= s < S and first_ok.get(s, False) and bool(lab[b, p] >= 0):
                scored[b, p] = True
    return bag, off, scored


def rank_bucket(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``[n]`` bucket index (0 = top1, 1 = top2, 2 = other) of ``target`` under
    ``logits [n, V]``. Ties go to the lower index (``torch.topk`` order)."""
    top2 = logits.topk(2, dim=-1).indices                          # [n, 2]
    t = target.view(-1, 1)
    out = torch.full_like(target, 2)
    out = torch.where(top2[:, 1:2].eq(t).squeeze(1), torch.ones_like(out), out)
    out = torch.where(top2[:, 0:1].eq(t).squeeze(1), torch.zeros_like(out), out)
    return out


# ── the shuffles ────────────────────────────────────────────────────────────────────

def _sattolo(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random n-CYCLE (Sattolo): a permutation with no fixed point for n >= 2."""
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        p[i], p[j] = p[j], p[i]
    return p


def make_shuffle(slot_valid: torch.Tensor, kind: str, rng: np.random.Generator
                 ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(src_row [B, S], src_slot [B, S], n_fixed)``: slot ``(b, s)`` takes the cells of
    slot ``(src_row, src_slot)``. Pad slots map to themselves and are not counted.

    ``xrow``: a row derangement (Sattolo over B rows); slot s takes the source row's slot
    ``s mod n_valid``. Raises on B < 2. ``xrow2``: a row derangement, and slot s takes the
    source row's slot ``(s + k) mod n_valid`` with ``k`` drawn from ``1..n_valid-1`` per
    row, so it differs from ANY ``xrow`` map wherever the source row has >= 2 slots.
    ``row``: a random cycle over the row's valid slots.
    ``n_fixed`` counts valid slots that map to themselves (a row with one slot under
    ``row``; a source row with no slot under ``xrow``)."""
    valid = slot_valid.cpu().numpy()
    B, S = valid.shape
    src_row = np.repeat(np.arange(B)[:, None], S, axis=1)
    src_slot = np.repeat(np.arange(S)[None, :], B, axis=0)
    n_valid = valid.sum(1)
    n_fixed = 0
    if kind in ("xrow", "xrow2"):
        if B < 2:
            raise ValueError(f"the {kind} shuffle needs >= 2 rows per batch")
        rp = _sattolo(B, rng)
        for b in range(B):
            nv = int(n_valid[b])
            ns = int(n_valid[rp[b]])
            if ns == 0:
                n_fixed += nv
                continue
            k = int(rng.integers(1, ns)) if (kind == "xrow2" and ns >= 2) else 0
            src_row[b, :nv] = rp[b]
            src_slot[b, :nv] = (np.arange(nv) + k) % ns
    elif kind == "row":
        for b in range(B):
            nv = int(n_valid[b])
            if nv < 2:
                n_fixed += nv
                continue
            src_slot[b, :nv] = _sattolo(nv, rng)
    else:
        raise ValueError(f"shuffle kind must be xrow|xrow2|row, got {kind!r}")
    return torch.from_numpy(src_row), torch.from_numpy(src_slot), n_fixed


def apply_shuffle(h: torch.Tensor, src_row: torch.Tensor, src_slot: torch.Tensor) -> torch.Tensor:
    """``h [B, S, ...]`` gathered at ``(src_row, src_slot)``: every trailing dim (a fan's K
    cells, the HC streams, the channels) moves with its slot."""
    sr = src_row.to(h.device)
    ss = src_slot.to(h.device)
    return h[sr, ss]


@contextlib.contextmanager
def controlled_shuffle(model, src_row: torch.Tensor, src_slot: torch.Tensor):
    """For the duration of the block, ``plan_mode="shuffle"`` on ``model`` applies THIS
    permutation instead of the model's own random one. Every other mode is the model's.
    Yields ``{"calls": n, "shapes": [...]}``: how often the seam was reached and the cell
    tensor shapes it moved."""
    real = model._tul_plan_ablate
    calls: dict = {"calls": 0, "shapes": []}

    def _ablate(h_slots, layout, mode):
        if mode != "shuffle":
            return real(h_slots, layout, mode)
        B, S = layout.slot_valid.shape
        if tuple(h_slots.shape[:2]) != (B, S):
            raise RuntimeError(f"shuffle seam got cells {tuple(h_slots.shape)}, expected "
                               f"[B={B}, S={S}, ...]")
        calls["calls"] += 1
        calls["shapes"].append(tuple(h_slots.shape))
        return apply_shuffle(h_slots, src_row, src_slot)

    model._tul_plan_ablate = _ablate
    try:
        yield calls
    finally:
        del model._tul_plan_ablate          # back to the class method


# ── one forward ─────────────────────────────────────────────────────────────────────

def _fwd_kwargs(model, seed: int) -> dict:
    """A gram arm reads the PRIOR at a fixed seed (its eval noise is a function of the
    seed and the pass only), so every control sees the same noise draw."""
    if getattr(model, "tul_gram", None) is not None:
        return {"gram_mode": "prior", "gram_sample_seed": int(seed)}
    return {}


@torch.no_grad()
def forward_ce(model, inp, layout, labels, device: str, plan_mode: str, fwd_kw: dict,
               first_pos: tuple[torch.Tensor, torch.Tensor]
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ce [B, L] float32 CPU, logp_first [n_spans, V] float32 CPU)``: the per-position
    CE of the label, and the full log-distribution at the offset-0 positions."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode=plan_mode,
                                        **fwd_kw)
    logits = out["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    fb, fp = first_pos
    lp = F.log_softmax(logits[fb.to(device), fp.to(device)], dim=-1)
    return ce.cpu(), lp.cpu()


# ── statistics ──────────────────────────────────────────────────────────────────────

def _boot_weights(blocks: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(inv [n], W [n_boot, n_blocks])``: ``inv`` maps an item to its block, ``W`` holds
    how many times each block is drawn in each resample."""
    ub, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ub), size=(n_boot, len(ub)))
    W = np.zeros((n_boot, len(ub)))
    np.add.at(W, (np.arange(n_boot)[:, None], idx), 1.0)
    return inv, W


def _mean_ci(val: np.ndarray, sel: np.ndarray, inv: np.ndarray, W: np.ndarray
             ) -> tuple[dict, np.ndarray]:
    """Point mean of ``val[sel]`` and its block-bootstrap percentile CI. Returns the dict
    and the per-resample means (nan where a resample drew no selected item)."""
    nb = W.shape[1]
    s = np.bincount(inv[sel], weights=val[sel], minlength=nb)
    c = np.bincount(inv[sel], minlength=nb).astype(np.float64)
    n = int(sel.sum())
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}, np.full(W.shape[0], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (W @ s) / (W @ c)
    fin = boots[np.isfinite(boots)]
    lo, hi = (np.quantile(fin, [0.025, 0.975]) if fin.size else (np.nan, np.nan))
    return {"mean": float(val[sel].mean()), "lo": float(lo), "hi": float(hi), "n": n}, boots


def classify(w: dict[str, dict], ratio: dict, eps: float) -> dict:
    """The R2 decision rule on per-bucket worth ``w[bucket] = {mean, lo, hi}`` and the
    ``top2 / top1`` ratio ``{point, lo, hi}``. See the module docstring."""
    present = [b for b in BUCKETS if w.get(b, {}).get("n", 0)]
    if present and all(w[b]["lo"] >= -eps and w[b]["hi"] <= eps for b in present):
        return {"verdict": "blur", "ci_supported": True}
    t1, t2 = w.get("top1", {}), w.get("top2", {})
    if not t1.get("n") or not t2.get("n") or t1["lo"] <= 0.0:
        return {"verdict": "inconclusive", "ci_supported": False}
    r, lo, hi = ratio["point"], ratio["lo"], ratio["hi"]
    if t2["mean"] < 0.0 or r < 0.25:
        return {"verdict": "committed", "ci_supported": bool(hi < 0.25)}
    if r >= 0.5:
        return {"verdict": "superposition", "ci_supported": bool(lo >= 0.5)}
    return {"verdict": "intermediate", "ci_supported": bool(lo >= 0.25 and hi < 0.5)}


def _clusters(lp_draw: torch.Tensor, lp_ref: torch.Tensor, n_draw: int, seed: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """Per span: ``(n_clusters, n_positive)`` for ``n_draw`` tokens drawn from
    ``exp(lp_draw)``; a cluster's gain is ``lp_draw[c] - lp_ref[c]``."""
    g = torch.Generator().manual_seed(int(seed))
    draws = torch.multinomial(lp_draw.exp(), n_draw, replacement=True, generator=g)
    n_cl = np.zeros(lp_draw.shape[0], dtype=np.int64)
    n_pos = np.zeros(lp_draw.shape[0], dtype=np.int64)
    for i in range(lp_draw.shape[0]):
        c = torch.unique(draws[i])
        gain = lp_draw[i, c] - lp_ref[i, c]
        n_cl[i] = int(c.numel())
        n_pos[i] = int((gain > 0).sum())
    return n_cl, n_pos


def _ratio(w: dict[str, dict], boots: dict[str, np.ndarray]) -> dict:
    """``worth(top2) / worth(top1)`` with its CI from the paired block resamples."""
    t1, t2 = w["top1"], w["top2"]
    if not (t1["n"] and t2["n"] and t1["mean"] != 0.0):
        return {"point": None, "lo": None, "hi": None, "frac_boot_top1_nonpos": None}
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = boots["top2"] / boots["top1"]
    ok = np.isfinite(rb) & (boots["top1"] > 0)
    lo, hi = (np.quantile(rb[ok], [0.025, 0.975]) if ok.any() else (np.nan, np.nan))
    return {"point": t2["mean"] / t1["mean"], "lo": float(lo), "hi": float(hi),
            "frac_boot_top1_nonpos": float(1.0 - ok.mean())}


# ── the probe ───────────────────────────────────────────────────────────────────────

def superposition_probe(model, batches, device: str, *, seed: int = 0,
                        controls: tuple[str, ...] = CONTROLS, eps: float = 0.01,
                        n_boot: int = N_BOOT) -> tuple[dict, dict]:
    """The instrument on packed batches ``[(inp, labels, layout, idx)]`` (``idx [B, L]``
    the stream index per position, ``-1`` off the stream). Returns ``(result JSON,
    per-token arrays)``."""
    if "xrow" not in controls:
        raise ValueError("xrow is the primary control and cannot be dropped")
    bad = [c for c in controls if c not in CONTROLS]
    if bad:
        raise ValueError(f"unknown controls {bad}; known: {CONTROLS}")
    controls = tuple(sorted(set(controls), key=CONTROLS.index))   # xrow before xrow2
    model.eval()
    fwd_kw = _fwd_kwargs(model, seed)
    tok: dict[str, list[np.ndarray]] = {k: [] for k in
                                        ("tok_index", "offset", "bucket", "bucket_ctx",
                                         "span_key",
                                         "ce_own", *(f"ce_{c}" for c in controls))}
    span: dict[str, list[np.ndarray]] = {k: [] for k in
                                         ("bucket", "bucket_ctx", "p_true", "p_top1",
                                          "p_top2",
                                          "entropy", "p_true_xrow")}
    clus: dict[str, list[np.ndarray]] = {k: [] for k in ("own_n", "own_pos", "null_n",
                                                         "null_pos")}
    shuffle_calls: dict[str, int] = {c: 0 for c in controls if c != "zero"}
    n_fixed: dict[str, int] = {c: 0 for c in controls if c != "zero"}
    n_xrow2_same = 0          # valid slots where xrow2 reads the same source as xrow
    n_slots_valid = 0
    span_base = 0
    for bi, (inp, labels, layout, idx) in enumerate(batches):
        layout = layout.to(device)
        bag, off, scored = span_offsets(layout, labels)
        first = scored & (off == 0)
        fb, fp = torch.nonzero(first, as_tuple=True)
        n_sp = int(fb.numel())
        n_slots_valid += int(layout.slot_valid.sum())
        # one span key per (row, bag); tokens carry their span's key
        key = torch.full_like(bag, -1)
        key[fb, fp] = torch.arange(n_sp) + span_base
        key_of = {(int(b), int(bag[b, p])): int(key[b, p]) for b, p in zip(fb, fp)}
        ce_own, lp_own = forward_ce(model, inp, layout, labels, device, "normal", fwd_kw,
                                    (fb, fp))
        ce_c: dict[str, torch.Tensor] = {}
        lp_c: dict[str, torch.Tensor] = {}
        rng = np.random.default_rng([int(seed), bi])
        for c in controls:
            if c == "zero":
                ce_c[c], lp_c[c] = forward_ce(model, inp, layout, labels, device, "zero",
                                              fwd_kw, (fb, fp))
                continue
            sr, ss, nf = make_shuffle(layout.slot_valid, c, rng)
            n_fixed[c] += nf
            if c == "xrow":
                xmap = (sr, ss)
            elif c == "xrow2" and "xrow" in controls:
                same = (sr == xmap[0]) & (ss == xmap[1]) & layout.slot_valid.cpu()
                n_xrow2_same += int(same.sum())
            with controlled_shuffle(model, sr, ss) as calls:
                ce_c[c], lp_c[c] = forward_ce(model, inp, layout, labels, device,
                                              "shuffle", fwd_kw, (fb, fp))
            if calls["calls"] == 0:
                raise RuntimeError(f"control {c!r}: the forward never reached "
                                   "_tul_plan_ablate, so nothing was shuffled")
            shuffle_calls[c] += calls["calls"]
        # spans: bucket and offset-0 diagnostics
        t0 = labels[fb, fp]
        bk = rank_bucket(lp_own, t0)
        bk_ctx = rank_bucket(lp_c["xrow"], t0)
        p_own = lp_own.exp()
        top2 = p_own.topk(2, dim=-1).values
        ent = -(p_own * torch.where(p_own > 0, lp_own, torch.zeros_like(lp_own))).sum(-1)
        span["bucket"].append(bk.numpy())
        span["bucket_ctx"].append(bk_ctx.numpy())
        span["p_true"].append(p_own.gather(1, t0.view(-1, 1)).squeeze(1).numpy())
        span["p_top1"].append(top2[:, 0].numpy())
        span["p_top2"].append(top2[:, 1].numpy())
        span["entropy"].append(ent.numpy())
        span["p_true_xrow"].append(lp_c["xrow"].exp().gather(1, t0.view(-1, 1)).squeeze(1)
                                   .numpy())
        if n_sp:
            n, p = _clusters(lp_own, lp_c["xrow"], N_DRAW, seed * 1000003 + bi)
            clus["own_n"].append(n)
            clus["own_pos"].append(p)
            if "xrow2" in lp_c:
                n, p = _clusters(lp_c["xrow2"], lp_c["xrow"], N_DRAW, seed * 1000003 + bi)
                clus["null_n"].append(n)
                clus["null_pos"].append(p)
        # tokens
        sb, sp_ = torch.nonzero(scored, as_tuple=True)
        tkey = np.array([key_of[(int(b), int(bag[b, p]))] for b, p in zip(sb, sp_)],
                        dtype=np.int64)
        tok["span_key"].append(tkey)
        for name, b_ in (("bucket", bk), ("bucket_ctx", bk_ctx)):
            tok[name].append(b_.numpy()[tkey - span_base] if tkey.size else
                             np.zeros(0, dtype=np.int64))
        tok["offset"].append(off[sb, sp_].numpy())
        tok["tok_index"].append(idx[sb, sp_].numpy() if idx is not None
                                else np.full(tkey.size, -1))
        tok["ce_own"].append(ce_own[sb, sp_].numpy())
        for c in controls:
            tok[f"ce_{c}"].append(ce_c[c][sb, sp_].numpy())
        span_base += n_sp

    arr = {k: np.concatenate(v) if v else np.zeros(0) for k, v in tok.items()}
    sp = {k: np.concatenate(v) if v else np.zeros(0) for k, v in span.items()}
    n_spans = int(sp["bucket"].size)
    if n_spans == 0:
        raise RuntimeError("no scored span: every span lacked a readable offset-0 label")
    # blocks: stream index // BLOCK (a token with no stream index is its own span's block)
    blocks = np.where(arr["tok_index"] >= 0, arr["tok_index"] // BLOCK,
                      -1 - arr["span_key"])
    inv, W = _boot_weights(blocks, n_boot, seed)
    bk_t = arr["bucket"].astype(np.int64)
    pos1 = arr["offset"] >= 1
    pos0 = arr["offset"] == 0
    res: dict = {"n_spans": n_spans, "n_tokens_scored": int(arr["offset"].size),
                 "n_slots_valid": n_slots_valid, "seed": seed, "controls": list(controls),
                 "eps": eps, "n_boot": n_boot, "block": BLOCK,
                 "shuffle_calls": shuffle_calls, "shuffle_fixed_slots": n_fixed,
                 "xrow2_same_as_xrow_slots": n_xrow2_same,
                 "n_blocks": int(W.shape[1]),
                 # A percentile CI over a handful of blocks has few distinct resamples and
                 # its edges sit on the point (the 2-row smoke: 3 blocks). Not a verdict.
                 "ci_warning": (f"only {W.shape[1]} stream blocks: CIs and ci_supported "
                                "are not meaningful" if W.shape[1] < MIN_BLOCKS else None),
                 "gram_prior_seed": fwd_kw.get("gram_sample_seed"),
                 "buckets": {}, "worth": {}, "ratio_top2_top1": {}, "decision": None}
    for i, b in enumerate(BUCKETS):
        m = sp["bucket"] == i
        res["buckets"][b] = {
            "n_spans": int(m.sum()),
            "n_tokens_pos1plus": int((pos1 & (bk_t == i)).sum()),
            "p_true_own": float(sp["p_true"][m].mean()) if m.any() else None,
            "p_true_xrow": float(sp["p_true_xrow"][m].mean()) if m.any() else None,
            "p_top1": float(sp["p_top1"][m].mean()) if m.any() else None,
            "p_top2": float(sp["p_top2"][m].mean()) if m.any() else None,
            "entropy": float(sp["entropy"][m].mean()) if m.any() else None,
        }
    bk_ctx_t = arr["bucket_ctx"].astype(np.int64)
    res["buckets_ctx"] = {b: {"n_spans": int((sp["bucket_ctx"] == i).sum()),
                              "n_tokens_pos1plus": int((pos1 & (bk_ctx_t == i)).sum()),
                              "own_top1_share": (float((sp["bucket"][sp["bucket_ctx"] == i]
                                                        == 0).mean())
                                                 if (sp["bucket_ctx"] == i).any() else None)}
                          for i, b in enumerate(BUCKETS)}
    res["ratio_top2_top1_ctx"] = {}
    for c in controls:
        w = arr[f"ce_{c}"] - arr["ce_own"]
        wc: dict = {"pos1plus": {}, "pos0": {}, "by_offset": {}, "pos1plus_ctx": {}}
        boots1: dict[str, np.ndarray] = {}
        boots_ctx: dict[str, np.ndarray] = {}
        for i, b in enumerate(BUCKETS):
            d1, boots1[b] = _mean_ci(w, pos1 & (bk_t == i), inv, W)
            wc["pos1plus"][b] = d1
            wc["pos1plus_ctx"][b], boots_ctx[b] = _mean_ci(w, pos1 & (bk_ctx_t == i), inv, W)
            wc["pos0"][b], _ = _mean_ci(w, pos0 & (bk_t == i), inv, W)
            wc["by_offset"][b] = []
            for lo_, hi_ in BINS:
                sel = (bk_t == i) & (arr["offset"] >= lo_) & (arr["offset"] <= hi_)
                d, _ = _mean_ci(w, sel, inv, W)
                wc["by_offset"][b].append({"offsets": [lo_, min(hi_, 10 ** 6)], **d})
        wc["pos1plus"]["all"], _ = _mean_ci(w, pos1, inv, W)
        wc["pos0"]["all"], _ = _mean_ci(w, pos0, inv, W)
        res["worth"][c] = wc
        res["ratio_top2_top1"][c] = _ratio(wc["pos1plus"], boots1)
        res["ratio_top2_top1_ctx"][c] = _ratio(wc["pos1plus_ctx"], boots_ctx)
    for key, wkey, rkey in (("decision", "pos1plus", "ratio_top2_top1"),
                            ("decision_ctx", "pos1plus_ctx", "ratio_top2_top1_ctx")):
        rx = res[rkey]["xrow"]
        res[key] = {"control": "xrow", "positions": "offsets 1..n",
                    "bucket": "own-cell rank" if key == "decision" else "xrow-cell rank",
                    **classify(res["worth"]["xrow"][wkey],
                               rx if rx["point"] is not None else
                               {"point": float("nan"), "lo": float("nan"),
                                "hi": float("nan")}, eps)}
    if clus["own_n"]:
        on, op = np.concatenate(clus["own_n"]), np.concatenate(clus["own_pos"])
        dec = {"n_draw": N_DRAW, "unit": "first token (offset-0 label), not a bigram",
               "own": {"mean_clusters": float(on.mean()),
                       "frac_ge2_positive": float((op >= 2).mean())},
               "by_bucket": {}}
        if clus["null_n"]:
            nn_, np_ = np.concatenate(clus["null_n"]), np.concatenate(clus["null_pos"])
            dec["null_xrow2"] = {"mean_clusters": float(nn_.mean()),
                                 "frac_ge2_positive": float((np_ >= 2).mean())}
            dec["excess_frac_ge2_positive"] = (dec["own"]["frac_ge2_positive"]
                                               - dec["null_xrow2"]["frac_ge2_positive"])
        for i, b in enumerate(BUCKETS):
            m = sp["bucket"] == i
            if m.any():
                dec["by_bucket"][b] = {"frac_ge2_positive_own": float((op[m] >= 2).mean()),
                                       **({"frac_ge2_positive_null":
                                           float((np_[m] >= 2).mean())}
                                          if clus["null_n"] else {})}
        res["decode_clusters"] = dec
    return res, arr


def _fmt(d: dict) -> str:
    if d.get("mean") is None:
        return "   n/a"
    return f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['n']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint path (relative = repo root)")
    ap.add_argument("--config", required=True, help="Hydra config name the arm trained with")
    ap.add_argument("--ovr", action="append", default=[],
                    help="Hydra override the arm trained with (repeatable)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--rows", type=int, default=192)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle / draw seed, and the gram prior's sample seed")
    ap.add_argument("--controls", default=",".join(CONTROLS))
    ap.add_argument("--eps", type=float, default=0.01, help="blur band, nats")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_start = time.time()

    from _build import ROOT, build_cfg
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    path = a.ckpt if a.ckpt.startswith("/") else os.path.join(ROOT, a.ckpt)
    label = a.label or os.path.basename(os.path.dirname(path))
    cfg = build_cfg(a.config, ["model.use_kernels=false", *a.ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None:
        raise SystemExit(f"{a.config} is not a TUL arm")
    tc = tul_rt.model_cfg
    if tc.tokens_through_core or getattr(tc, "loop_reads_tokens", False):
        raise SystemExit(f"{a.config}: no separate slot write to shuffle (paid loop / "
                         "loop_reads_tokens)")
    model, step = load_ckpt(cfg, path, a.device, tc)
    model.eval()
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]
    rows = sum(b[0].shape[0] for b in batches)
    if rows < a.rows:
        raise SystemExit(f"packed {rows} rows, asked for {a.rows}")
    if any(b[0].shape[0] < 2 for b in batches):
        raise SystemExit("a batch has one row: the xrow control needs >= 2. Pick --rows "
                         "and --batch so the last batch holds >= 2 rows.")
    controls = tuple(c.strip() for c in a.controls.split(",") if c.strip())
    t_probe = time.time()
    res, arr = superposition_probe(model, batches, a.device, seed=a.seed,
                                   controls=controls, eps=a.eps)
    res.update({"label": label, "config": a.config, "ovr": a.ovr, "ckpt": path,
                "step": step, "rows": rows, "batch": a.batch,
                "geometry": {"tg_geometry": tc.tg_geometry,
                             "coda_prefix_reach": tc.tg_coda_prefix_reach,
                             "prefix_k": tc.prefix_k, "slot_cells": tc.slot_cells,
                             "fan_mix": tc.fan_mix if getattr(tc, "fan_k", 0) else None,
                             "gram": bool(tc.gram)},
                "own_cell": ("all K prefix cells of slot s-1, moved together"
                             if tc.slot_cells > 1 else "the prefix cells of slot s-1"),
                "wall_s": {"load": round(t_probe - t_start, 1),
                           "probe": round(time.time() - t_probe, 1)}})
    npz = a.out.rsplit(".", 1)[0] + f".{label}.tokens.npz"
    np.savez_compressed(npz, tok_index=arr["tok_index"].astype(np.int32),
                        offset=arr["offset"].astype(np.int16),
                        bucket=arr["bucket"].astype(np.int8),
                        bucket_ctx=arr["bucket_ctx"].astype(np.int8),
                        **{k: v.astype(np.float32) for k, v in arr.items()
                           if k.startswith("ce_")})
    res["tokens_npz"] = npz
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{label} step {step}: {res['n_spans']} spans, {res['n_tokens_scored']} tokens, "
          f"gram_prior_seed={res['gram_prior_seed']}")
    for b in BUCKETS:
        bb = res["buckets"][b]
        print(f"  {b:5s} spans={bb['n_spans']:5d}  p_true={bb['p_true_own']}")
    for c in controls:
        for b in (*BUCKETS, "all"):
            print(f"  worth[{c:5s}] {b:5s} pos1+ {_fmt(res['worth'][c]['pos1plus'][b])}"
                  f"   pos0 {_fmt(res['worth'][c]['pos0'][b])}")
        r = res["ratio_top2_top1"][c]
        if r["point"] is not None:
            print(f"  ratio[{c}] top2/top1 = {r['point']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    for b in BUCKETS:
        print(f"  ctx {b:5s} spans={res['buckets_ctx'][b]['n_spans']:5d}  worth[xrow] pos1+ "
              f"{_fmt(res['worth']['xrow']['pos1plus_ctx'][b])}")
    print(f"  DECISION_CTX: {res['decision_ctx']}")
    print(f"  DECISION: {res['decision']}  n_blocks={res['n_blocks']}"
          + (f"  WARNING: {res['ci_warning']}" if res["ci_warning"] else ""))
    if "decode_clusters" in res:
        print(f"  decode clusters: {json.dumps(res['decode_clusters'])}")
    print(f"  wall: {res['wall_s']}")
    print(f"wrote {a.out} and {npz}")


if __name__ == "__main__":
    main()
