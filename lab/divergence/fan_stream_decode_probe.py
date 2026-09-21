"""Do the fan's K streams SAY different things, or only spell the same thing differently?

`fan_mixture_probe.py` closed the selector question in CE: the deployed all-cell read
already beats every branch-blind mixture, and the oracle's remaining 0.043 nats is mostly
the ``log K`` a reader pays for not being told the branch. What CE cannot say is whether
the four streams are four HYPOTHESES or one hypothesis in four spellings — the question
LCM's future-work paragraph names ("sample multiple embeddings and associate a score",
§8; the reading note §8 bullet 9) and the one a lineage design stands on.

This probe reads it in the only place a hypothesis is visible: the TEXT. For every
eligible slot of N validation rows, the strict coda writes the next span GREEDILY four
times, once with each stream alone in the cells (``_tul_fan_stream_write``'s write — cell
``i`` through ``W_prefix[i]`` with the other K−1 cells blank, the deployed geometry with
the losers removed), and the probe reads:

  ``first_agree``    the fraction of slots where all K decodes share their FIRST token
                     (see the note below: that token is written blind to the cells);
  ``distinct_all``   the fraction of slots where all K decodes are pairwise different;
  ``pair_distinct``  the fraction of the K(K−1)/2 PAIRS that differ, per slot;
  ``between_dist``   the mean pairwise ``1 − cos`` of the K decodes under an embedding;
  ``within_dist``    the SAME statistic over K decodes from ONE stream with the Bowman
                     token-state dropout (``tul.token_state_dropout``) forced ON — the
                     surface-variation control;
  ``ratio``          ``between_dist / within_dist``, with a row-block bootstrap.

A ratio near 1 means the streams differ no more than one stream differs from itself under
a nuisance perturbation: four spellings, not four hypotheses.

THE EMBEDDING. A fan arm has no ``TULCodeEncoder`` (``tul.code`` is false on the
spandec-strict family), so there is no frozen E to re-encode a decoded span with. The
embedding used here is the model's OWN mean-pooled PRELUDE states over the span's tokens,
which under the strict geometry depend on that span's tokens alone — a function of the
decoded text and nothing else, but the model's own representation, not an independent
one. Said plainly rather than hidden: this is a weaker semantic probe than SONAR is for
LCM, and the token-level ``distinct`` columns are the reading that does not depend on it.

THE FIRST TOKEN OF A SPAN IS WRITTEN BLIND TO ITS OWN SLOT'S CELLS, and that is the
architecture, not this probe. Token 0 of span ``s+1`` is emitted at the BOUNDARY token of
span ``s`` (the only position trained to emit it at ``tul.emit_weight: 0``), and that token
sits in bag ``s``, so under the strict geometry it reads span ``s``'s tokens and the prefix
cells of slots BEFORE ``s`` — never slot ``s``'s own cells. It can still differ between
streams, because the single-stream write blanks the losers in EVERY slot, the earlier ones
included; ``first_agree`` is what that leaves, and on fan4-all @ 5000 it reads 0.72 against
0.66 for four dropout draws of one stream. Reported rather than worked around; the
``distinct`` columns are unaffected (they ask whether ANY position differs).

TWO CAVEATS, stated rather than buried:
  * the between-stream decodes are dropout FREE (the deployed path) while the within-
    stream control carries dropout, so the control is not a symmetric twin — it bounds
    the nuisance variation from ABOVE, which makes the ratio conservative in the
    direction that matters (a ratio above 1 is evidence the streams differ);
  * the decode is length matched to the true span (see ``_span_decode.py``).

Usage:

    python lab/divergence/fan_stream_decode_probe.py \
        --ckpt fan4-all=tul_slot_spandec_strict_fan4_all=/path/step_5000.pt \
        --rows 24 --batch 2 --out fan_stream_decode_fan4-all_5000.json

The pure functions (``pair_distinct``, ``mean_pairwise_dist``) are tested in
``tests/test_lcm_instruments.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fan_mixture_probe import block_bootstrap  # noqa: E402

__all__ = ["pair_distinct", "first_token_agree", "mean_pairwise_dist", "span_pool",
           "block_bootstrap"]


# ────────────────────────── pure readings ──────────────────────────

def pair_distinct(tokens: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[S, K, J]`` decodes + ``[S]`` lengths -> ``([S] all-different, [S] pair rate)``.

    Two decodes are "different" when they disagree at any position inside the span's own
    length; positions past ``lens`` are padding and never counted. ``all-different`` is
    true when every one of the ``K(K−1)/2`` pairs differs.
    """
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be [S, K, J], got {tuple(tokens.shape)}")
    S, K, J = tokens.shape
    if lens.shape != (S,):
        raise ValueError(f"lens must be [S] = ({S},), got {tuple(lens.shape)}")
    if K < 2:
        raise ValueError("need at least two decodes to compare")
    m = (torch.arange(J, device=tokens.device).view(1, J) < lens.view(S, 1))      # [S, J]
    diff = []
    for i in range(K):
        for j in range(i + 1, K):
            diff.append(((tokens[:, i] != tokens[:, j]) & m).any(dim=1))
    d = torch.stack(diff, dim=1)                                                  # [S, P]
    return d.all(dim=1), d.float().mean(dim=1)


def first_token_agree(tokens: torch.Tensor) -> torch.Tensor:
    """``[S, K, J]`` -> ``[S]`` 1.0 where all K decodes share token 0, else 0.0.

    Token 0 of a span is emitted at the previous span's boundary token, which under the
    strict geometry cannot read the slot's cells, so this column is expected to be near 1
    on ANY arm and is here to say so with a number rather than an anecdote.
    """
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be [S, K, J], got {tuple(tokens.shape)}")
    f = tokens[:, :, 0]
    return (f == f[:, :1]).all(dim=1).float()


def mean_pairwise_dist(emb: torch.Tensor) -> torch.Tensor:
    """``[S, K, D]`` -> ``[S]`` mean ``1 − cos`` over the ``K(K−1)/2`` unordered pairs."""
    if emb.dim() != 3:
        raise ValueError(f"emb must be [S, K, D], got {tuple(emb.shape)}")
    S, K, D = emb.shape
    if K < 2:
        raise ValueError("need at least two embeddings to compare")
    x = torch.nn.functional.normalize(emb.double(), dim=-1)
    c = torch.matmul(x, x.transpose(1, 2))                                        # [S, K, K]
    iu = torch.triu_indices(K, K, offset=1, device=emb.device)
    return (1.0 - c[:, iu[0], iu[1]]).mean(dim=1)


def span_pool(xn: torch.Tensor, bag_id: torch.Tensor, slot_mask: torch.Tensor,
              n_slots: int) -> torch.Tensor:
    """``[B, L, C]`` prelude states -> ``[B, S, C]``, the MEAN over span ``s+1``'s tokens.

    Bin ``s`` holds span ``s+1`` (``bag_id == s + 1``), the span the slot ``s`` precedes —
    the same convention ``TULCodeEncoder`` pools on. A slot whose next span has no token
    gets exactly zero.
    """
    B, L, C = xn.shape
    s_ar = torch.arange(n_slots, device=xn.device).view(1, n_slots, 1)
    sel = ((bag_id.unsqueeze(1) == (s_ar + 1)) & (~slot_mask).unsqueeze(1)).to(xn.dtype)
    num = torch.einsum("bsl,blc->bsc", sel, xn)
    den = sel.sum(-1, keepdim=True).clamp_min(1.0)
    return num / den


# ────────────────────────── the run ──────────────────────────

class ForceTokenDropout:
    """Force ``TULSlots.apply_token_dropout`` ON at eval with a SEEDED, repeatable mask.

    The decode re-runs the whole forward once per token, so an unseeded mask would be a
    different dropout on every step and the "same stream, one nuisance draw" control would
    not exist. Seeding ``torch.manual_seed`` immediately before the shipped draw makes the
    mask a function of (seed, shape) alone, so one draw index is one fixed mask for the
    whole decode. The shipped method does the drop; only the flag and the seed change.
    """

    def __init__(self, model):
        self.model = model
        self.seed: int | None = None
        self._restore = None

    def __enter__(self):
        from morph.model.tul import TULSlots
        real = TULSlots.apply_token_dropout
        box = self

        def patched(self, x, layout, training):
            if box.seed is None:
                return real(self, x, layout, training)
            torch.manual_seed(int(box.seed))
            if x.device.type == "cuda":
                torch.cuda.manual_seed_all(int(box.seed))
            return real(self, x, layout, True)

        TULSlots.apply_token_dropout = patched
        self._restore = lambda: setattr(TULSlots, "apply_token_dropout", real)
        return self

    def __exit__(self, *exc):
        self._restore()
        return False


def _rowsum(vals: np.ndarray, row_ix: np.ndarray, n_rows: int) -> np.ndarray:
    out = np.zeros(n_rows, dtype=np.float64)
    np.add.at(out, row_ix, vals.astype(np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=ovr1,ovr2]")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--depth", type=int, default=6, help="forced slot depth, as the sweep")
    ap.add_argument("--max-span-tokens", type=int, default=32)
    ap.add_argument("--within-stream", type=int, default=0,
                    help="which stream the dropout control re-decodes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed; it does NOT draw rows")
    ap.add_argument("--check-parity", action="store_true")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    from _build import ROOT, build_cfg, parse_ckpt_spec
    from _rows import pack_rows, stream_from_loader
    from _span_decode import (PinnedCells, check_span_layout, greedy_decode, row_logits,
                              span_slots)
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    import morph

    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or int(getattr(tul_rt.model_cfg, "fan_k", 0)) < 2:
        raise SystemExit(f"{label}: not a fan arm (tul.fan_k < 2)")
    if str(getattr(tul_rt.model_cfg, "tg_geometry", "")) != "strict":
        raise SystemExit(f"{label}: the parallel span decode needs tul.tg_geometry=strict")
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    K = int(tc.fan_k)
    p_drop = float(tc.token_state_dropout)
    if p_drop <= 0.0:
        raise SystemExit(f"{label}: tul.token_state_dropout is {p_drop}; the within-stream "
                         "control has no nuisance to draw and the ratio would be 0/0")
    tc.slot_mean_depth = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth))
    if int(getattr(tc, "slot_depth_fixed", 0)):
        tc.slot_depth_fixed = a.depth
    J = int(a.max_span_tokens)
    has_e = getattr(model, "tul_code_enc", None) is not None
    emb_source = "frozen encoder E" if has_e else "mean-pooled prelude states (no E on this arm)"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]

    rec = {"row": [], "ntok": [], "between_dist": [], "within_dist": [],
           "distinct_all": [], "pair_distinct": [], "within_distinct_all": [],
           "within_pair_distinct": [], "first_agree": [], "within_first_agree": []}
    n_rows_seen, identity_max, parity_bad = 0, 0.0, None
    ex_lines: list[str] = []

    with PinnedCells(model, "loop") as pin, ForceTokenDropout(model) as drop:
        for bi, (inp, labels, layout, _idx) in enumerate(batches):
            ids = inp.to(a.device)
            lay = layout.to(a.device)
            elig, lens, first, src0 = span_slots(lay, J)
            check_span_layout(lay, elig, first, src0)
            if not bool(elig.any()):
                n_rows_seen += ids.shape[0]
                continue
            B, S = elig.shape
            ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda")
            with ac:
                pin.pin, drop.seed = None, None
                lg_ref = row_logits(model, ids, lay).float()
                cells_flat = pin.last.clone()                    # [B, S*K, (n,) C]
                pin.pin = cells_flat
                identity_max = max(identity_max,
                                   float((row_logits(model, ids, lay).float()
                                          - lg_ref).abs().max()))
                del lg_ref
            cells = cells_flat.reshape(B, S, K, *cells_flat.shape[2:])

            def _decode(pin_cells, seed):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=a.device == "cuda"):
                    pin.pin, drop.seed = pin_cells, seed
                    ids_d, cand = greedy_decode(model, ids, lay, elig, lens, first, src0, J)
                    pin.pin, drop.seed = None, None
                    row_logits(model, ids_d, lay)
                    xn = pin.last_xn
                    xn = xn.mean(dim=2) if xn.dim() == 4 else xn
                return cand, span_pool(xn.float(), lay.bag_id, lay.slot_mask, S)

            # ── between streams: stream i ALONE in its cell, the others blank ──
            bt_tok, bt_emb = [], []
            for i in range(K):
                blank = torch.zeros_like(cells)
                blank[:, :, i] = cells[:, :, i]
                cand, emb = _decode(blank.reshape(cells_flat.shape), None)
                bt_tok.append(cand)
                bt_emb.append(emb)
                if bi == 0 and len(ex_lines) < a.samples * (K + 1):
                    r0, s0 = elig.nonzero(as_tuple=True)
                    for q in range(min(a.samples, r0.numel())):
                        b_i, s_i = int(r0[q]), int(s0[q])
                        n = int(lens[b_i, s_i])
                        ex_lines.append(
                            f"  [stream {i}] row {b_i} slot {s_i}: "
                            f"{tok.decode(cand[b_i, s_i, :n].tolist())!r}")
            if a.check_parity and bi == 0:
                blank = torch.zeros_like(cells)
                blank[:, :, 0] = cells[:, :, 0]
                s_ar = torch.arange(S, device=elig.device)
                even = elig & ((s_ar % 2) == 0).view(1, -1)
                with ac:
                    pin.pin, drop.seed = blank.reshape(cells_flat.shape), None
                    _ids_e, cand_e = greedy_decode(model, ids, lay, even, lens, first,
                                                   src0, J)
                    pin.pin = None
                parity_bad = int((cand_e[even] != bt_tok[0][even]).sum())

            # ── within one stream: K dropout draws, same cells ──────────────
            wi = int(a.within_stream)
            blank = torch.zeros_like(cells)
            blank[:, :, wi] = cells[:, :, wi]
            wt_tok, wt_emb = [], []
            for d in range(K):
                cand, emb = _decode(blank.reshape(cells_flat.shape), 10_000 + 97 * d + bi)
                wt_tok.append(cand)
                wt_emb.append(emb)

            r, s = elig.nonzero(as_tuple=True)
            bt = torch.stack(bt_tok, dim=2)[r, s]                  # [n, K, J]
            wt = torch.stack(wt_tok, dim=2)[r, s]
            be = torch.stack(bt_emb, dim=2)[r, s]                  # [n, K, C]
            we = torch.stack(wt_emb, dim=2)[r, s]
            ln = lens[r, s]
            da, pd = pair_distinct(bt, ln)
            wa, wp = pair_distinct(wt, ln)
            rec["row"].append((r + n_rows_seen).cpu().numpy())
            rec["ntok"].append(ln.cpu().numpy().astype(np.float64))
            rec["between_dist"].append(mean_pairwise_dist(be).cpu().numpy())
            rec["within_dist"].append(mean_pairwise_dist(we).cpu().numpy())
            rec["distinct_all"].append(da.float().cpu().numpy())
            rec["pair_distinct"].append(pd.cpu().numpy())
            rec["within_distinct_all"].append(wa.float().cpu().numpy())
            rec["within_pair_distinct"].append(wp.cpu().numpy())
            rec["first_agree"].append(first_token_agree(bt).cpu().numpy())
            rec["within_first_agree"].append(first_token_agree(wt).cpu().numpy())
            n_rows_seen += ids.shape[0]
            print(f"  batch {bi + 1}/{len(batches)} done ({time.time() - t0:.0f}s)",
                  flush=True)

    cat = {k: np.concatenate(v) for k, v in rec.items()}
    uniq, ix = np.unique(cat["row"], return_inverse=True)
    n_rows = int(uniq.shape[0])
    names = ["between_dist", "within_dist", "distinct_all", "pair_distinct",
             "within_distinct_all", "within_pair_distinct", "first_agree",
             "within_first_agree"]
    # every column is a PER-SLOT statistic; the blocks are rows and the weight is 1 per
    # slot (a span's length does not make its four decodes more or less alike)
    row_sum = {k: _rowsum(cat[k], ix, n_rows) for k in names}
    row_n = _rowsum(np.ones_like(cat["ntok"]), ix, n_rows)
    boot = block_bootstrap(row_sum, row_n, [("between_dist", "within_dist")],
                           n_boot=a.boot, seed=a.seed)
    # the ratio needs its own resample (a difference of means is not a ratio of means)
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, n_rows, size=(a.boot, n_rows))
    rb = row_sum["between_dist"][idx].sum(1) / row_sum["within_dist"][idx].sum(1)
    lo, hi = np.quantile(rb, [0.025, 0.975])
    ratio = {"point": float(row_sum["between_dist"].sum() / row_sum["within_dist"].sum()),
             "lo": float(lo), "hi": float(hi), "n_units": n_rows, "n_boot": int(a.boot),
             "level": 0.95}

    res = {"arm": label, "config": config, "ckpt": path, "step": step, "rows": a.rows,
           "batch": a.batch, "depth": a.depth, "fan_k": K, "within_stream": a.within_stream,
           "token_state_dropout": p_drop, "embedding": emb_source,
           "n_spans": int(cat["row"].shape[0]), "n_rows": n_rows,
           "n_tokens": float(cat["ntok"].sum()),
           "morph_module": os.path.dirname(morph.__file__),
           "identity_check_max_abs": identity_max,
           "parity_check_mismatched_tokens": parity_bad,
           "bootstrap": boot, "between_over_within": ratio,
           "wall_clock_s": time.time() - t0}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    L = [f"{label} step {step}  rows={a.rows} batch={a.batch} depth={a.depth} K={K} "
         f"spans={res['n_spans']} tokens={res['n_tokens']:.0f}",
         f"embedding: {emb_source}",
         f"within-stream control: stream {a.within_stream}, "
         f"tul.token_state_dropout={p_drop} forced on, {K} seeded draws",
         f"morph from {res['morph_module']}",
         f"identity check (pinned == unpinned logits) max|d| = {identity_max:.3e}",
         f"parity check mismatched tokens = "
         f"{parity_bad if parity_bad is not None else 'not run'}", "",
         "NOTE: a span's FIRST token is emitted at the previous span's boundary token,",
         "which under the strict geometry cannot read THIS slot's cells. It can still read",
         "EARLIER slots' cells, which the single-stream write also blanks, so the streams",
         "are not forced to agree there. `first_agree` is what is left; read it beside",
         "`within_first_agree`, the same statistic under the nuisance draw.", "",
         "reading                     value   95 % row-block bootstrap"]
    for k in names:
        v = boot[k]
        L.append(f"  {k:<24} {v['point']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    v = boot["between_dist-within_dist"]
    L.append(f"  {'between - within':<24} {v['point']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    L.append(f"  {'between / within':<24} {ratio['point']:+.4f}  "
             f"[{ratio['lo']:+.4f}, {ratio['hi']:+.4f}]")
    L += ["", "decoded spans (first batch):"] + ex_lines
    L += ["", f"wall clock {res['wall_clock_s']:.0f} s"]
    with open(txt, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.out} and {txt}")
    if identity_max > 1e-2:
        raise AssertionError(
            f"the cell pin is not an identity at the captured value: max|d| "
            f"{identity_max:.3e} on the logits; nothing above is a reading of this model")
    if parity_bad:
        raise AssertionError(
            f"the parallel span decode is not independent: {parity_bad} tokens differ "
            f"when only the even slots decode")


if __name__ == "__main__":
    main()
