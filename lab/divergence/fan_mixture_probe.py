"""Fan mixture read: what a READER can recover from a fan arm's K streams, on a checkpoint.

The trainer's oracle instrument (``MORPHTransformer._tul_fan_oracle``) reports three
token-weighted numbers per val: ``fan/single_ce`` (stream 0 alone), ``fan/mixed_ce`` (the
deployed all-cell read) and ``fan/oracle_ce`` (the per-span minimum over the K
single-stream coda passes). The gap ``mixed_ce − oracle_ce`` is called selector regret,
and on ``slot-spandec-strict-fan4-all`` @ 5000 it is 0.043 nats per token. Part of that
gap is NOT a selector failure: the oracle is told which branch happened and a deployed
reader is not. The bound is ``log K`` nats per SPAN — a reader that carries all K
hypotheses forward and marginalises over them pays at most ``log K / span_len`` per token
over the oracle, and no selector can do better than that without the answer.

This probe reads the whole ladder on the same rows and the same tokens:

  ``oracle``       Σ_s min_k ℓ_{s,k} / N_tok        (the model's own ``fan/oracle_ce``)
  ``single_k``     Σ_s ℓ_{s,k} / N_tok, each k      (the model's own ``fan/stream_ce_k{i}``)
  ``best_single``  the smallest of those
  ``deployed``     the all-cell read                (the model's own ``fan/mixed_ce``)
  ``uniform_mix``  Σ_s −log((1/K) Σ_k exp(−ℓ_{s,k})) / N_tok
  ``prefix_mix``   the SEQUENTIAL mixture: at token j of span s the stream weights are
                   ``w_{j,k} ∝ (1/K) Π_{i<j} p_k(y_i)``, i.e. ``softmax_k(−Σ_{i<j}
                   nll_{s,k,i})``, and the token pays ``−log Σ_k w_{j,k} p_k(y_j)``.
                   It reads only tokens already decoded, so it is deployable.

``prefix_mix`` telescopes to ``uniform_mix`` exactly per span (the weight denominators
cancel), so the two TOTALS are the same number by construction and the probe asserts it
to 1e-9 in float64. What they do not share is the per-offset profile: the prefix mixture
pays nearly the whole ``log K`` at offset 0 and converges on the oracle as the prefix
identifies the branch. That profile is the reading, and it is reported in the same
offset-in-span bins ``worth_profile.py`` uses (``_earning.BINS``).

``winner_persistence`` is the second reading: with the per-span winner ``k*_s``, the
probability that ``k*_{s+1} = k*_s`` along a row against the chance level ``Σ_k share_k²``.
Stream identities that persist across spans are the precondition for lineages
(``.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md``,
Part 2 stage 4).

HOW THE PER-TOKEN NLLs ARE TAKEN. The probe does not re-derive the coda call. It replaces
``MORPHTransformer._tul_fan_oracle`` with a wrapper that (1) calls the REAL instrument
first, so the model's own ``fan/*`` scalars are produced by the shipped code and can be
checked against the probe's, and (2) then repeats the instrument's own loop with the
arguments the forward handed it — the same ``base`` carrier, the same ``_tul_fan_stream_write``
(so the same ``TULSlots.prefix_project`` and, under ``fan_mix="all"``, the same blanking of
the other K−1 cells), the same ``_back_region(inject_keep=keep, attn_kwargs=coda_kw,
ret_reset_mask=tg_reset)``, the same tied head and the same ``span_ce_index`` scoring mask
— stopping one step earlier, at the per-TOKEN cross entropy instead of the per-span
``index_add``. Summing the probe's per-token NLL over a span's tokens therefore
reproduces ``accumulate_span_ce`` exactly, and the probe asserts that against the
instrument's own ``oracle_ce`` and ``mixed_ce``. Both run inside the forward's autocast
context, under ``no_grad``, so the replay differs from the shipped instrument in nothing.

The span/slot bookkeeping is the instrument's, kept deliberately: bin ``g`` of
``span_ce_index`` is slot ``g − 1``'s next span, a span is scored only when its slot is
valid and it has scored tokens, and bin ``S`` doubles as the packer's dump bin (the model
attributes it to slot ``S − 1``; the probe does the same so the totals match).

Usage (a probe host, never the trainer's GPU):

    python lab/divergence/fan_mixture_probe.py \
        --ckpt slot-spandec-strict-fan4-all=tul_slot_spandec_strict_fan4_all=/path/step_5000.pt \
        --rows 48 --batch 3 --depth 6 --out fan_mixture.json

The pure functions (``span_stream_nll``, ``oracle_per_span``, ``uniform_mix_nll``,
``prefix_mix_token_nll``, ``offset_bin_index``, ``winner_persistence``,
``block_bootstrap``) take a ``[S, K, Lmax]`` NLL tensor with a ``[S, Lmax]`` length mask
and are tested in ``tests/test_fan_mixture_probe.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _earning import BINS  # noqa: E402  (ONE home for the offset bins)

__all__ = ["span_stream_nll", "oracle_per_span", "uniform_mix_nll", "prefix_mix_token_nll",
           "offset_bin_index", "winner_persistence", "block_bootstrap", "bin_labels",
           "per_token_nll", "BINS"]


# ────────────────────────── pure readings ──────────────────────────

def _check(nll: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if nll.dim() != 3:
        raise ValueError(f"nll must be [S, K, Lmax], got {tuple(nll.shape)}")
    if mask.shape != (nll.shape[0], nll.shape[2]):
        raise ValueError(f"mask must be [S, Lmax] = {(nll.shape[0], nll.shape[2])}, "
                         f"got {tuple(mask.shape)}")
    if nll.shape[1] < 1:
        raise ValueError("need at least one stream")
    m = mask.bool()
    # A masked entry must not reach any sum: zero it once, here.
    return nll.double() * m.unsqueeze(1).double(), m


def span_stream_nll(nll: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``[S, K, Lmax]`` + ``[S, Lmax]`` -> ``[S, K]`` summed NLL of each stream per span."""
    x, _ = _check(nll, mask)
    return x.sum(dim=-1)


def oracle_per_span(nll: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``([S] best summed NLL, [S] argmin stream)`` — the instrument's per-span minimum."""
    ell = span_stream_nll(nll, mask)
    best, arg = ell.min(dim=-1)
    return best, arg


def uniform_mix_nll(nll: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``[S]``: ``−log((1/K) Σ_k exp(−ℓ_{s,k}))``, in float64 through ``logsumexp``.

    The reader that keeps all K hypotheses with equal prior. Bounded above by
    ``min_k ℓ_{s,k} + log K`` and below by ``min_k ℓ_{s,k}``.
    """
    ell = span_stream_nll(nll, mask)
    k = int(ell.shape[1])
    return -(torch.logsumexp(-ell, dim=1) - math.log(float(k)))


def prefix_mix_token_nll(nll: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``[S, Lmax]``: the sequential (prefix-weighted) mixture's per-token NLL, 0 at pads.

    ``w_{j,k} = softmax_k(−Σ_{i<j} nll_{s,k,i})`` and the token pays
    ``−log Σ_k w_{j,k} exp(−nll_{s,k,j})``. Computed in the explicit weight form, NOT
    through the telescoping shortcut, so that ``Σ_j == uniform_mix_nll`` stays a real
    numerical check rather than an identity of the code.
    """
    x, m = _check(nll, mask)
    c = torch.cumsum(x, dim=2) - x                       # exclusive prefix sums [S, K, Lmax]
    logw = torch.log_softmax(-c, dim=1)                  # [S, K, Lmax]
    tok = -torch.logsumexp(logw - x, dim=1)              # [S, Lmax]
    return tok * m.double()


def bin_labels() -> list[str]:
    return [f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}")
            for lo, hi in BINS]


def offset_bin_index(mask: torch.Tensor) -> torch.Tensor:
    """``[S, Lmax]`` -> ``[S, Lmax]`` int64 bin index of the token's offset in its span, −1 at pads.

    The offset is the rank of the token among its span's scored tokens (0 = the span's
    first scored token), which for a prefix mask is the column index. The bins are
    ``_earning.BINS``, the ones ``worth_profile.py`` reports.
    """
    m = mask.bool()
    off = m.long().cumsum(dim=1) - 1
    out = torch.full_like(off, -1)
    for i, (lo, hi) in enumerate(BINS):
        out = torch.where(m & (off >= lo) & (off <= hi), torch.full_like(off, i), out)
    if bool((m & (out < 0)).any()):
        raise AssertionError("the offset bins do not cover every unmasked token")
    return out


def winner_persistence(winners: np.ndarray, row_ids: np.ndarray, k: int,
                       n_boot: int = 0, seed: int = 0, level: float = 0.95) -> dict:
    """P(winner of span n+1 == winner of span n) along each row, against chance.

    ``winners`` and ``row_ids`` are parallel ``[S]`` arrays with the spans of a row in
    ROW ORDER. Consecutive pairs are taken inside a row only. ``chance`` is
    ``Σ_k share_k²``, the repeat rate of independent draws from the observed marginal.

    ``n_boot > 0`` adds the same ROW-block bootstrap the CE table uses, on ``p_repeat``,
    ``chance`` and their difference — the excess over chance is the reading that gates
    stream lineages, and the pairs inside a row are not independent, so a binomial
    standard error would overstate it.
    """
    w = np.asarray(winners, dtype=np.int64)
    r = np.asarray(row_ids, dtype=np.int64)
    if w.shape != r.shape or w.ndim != 1:
        raise ValueError("winners and row_ids must be 1-D arrays of the same length")
    shares = [float((w == i).mean()) if w.size else 0.0 for i in range(k)]
    same_row = r[1:] == r[:-1]
    pairs = int(same_row.sum())
    rep = float((w[1:][same_row] == w[:-1][same_row]).mean()) if pairs else float("nan")
    out = {"p_repeat": rep, "chance": float(sum(s * s for s in shares)),
           "shares": shares, "n_pairs": pairs, "n_spans": int(w.size)}
    out["excess"] = rep - out["chance"]
    if n_boot <= 0:
        return out
    uniq, inv = np.unique(r, return_inverse=True)
    n_rows = int(uniq.shape[0])
    r_pairs = np.zeros(n_rows)
    r_rep = np.zeros(n_rows)
    np.add.at(r_pairs, inv[1:][same_row], 1.0)
    np.add.at(r_rep, inv[1:][same_row], (w[1:][same_row] == w[:-1][same_row]).astype(float))
    r_cnt = np.zeros((n_rows, k))
    np.add.at(r_cnt, (inv, w), 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=(n_boot, n_rows))
    bp = r_rep[idx].sum(1) / np.maximum(r_pairs[idx].sum(1), 1.0)
    cnt = r_cnt[idx].sum(1)                                    # [n_boot, k]
    sh = cnt / np.maximum(cnt.sum(1, keepdims=True), 1.0)
    bc = (sh * sh).sum(1)
    alpha = (1.0 - level) / 2.0
    for name, draw in (("p_repeat", bp), ("chance", bc), ("excess", bp - bc)):
        lo, hi = np.quantile(draw, [alpha, 1.0 - alpha])
        out[f"{name}_ci"] = {"lo": float(lo), "hi": float(hi), "n_units": n_rows,
                             "n_boot": int(n_boot), "level": float(level)}
    return out


def block_bootstrap(row_sum: dict[str, np.ndarray], row_tok: np.ndarray,
                    diffs: list[tuple[str, str]], n_boot: int = 1000, seed: int = 0,
                    level: float = 0.95) -> dict[str, dict]:
    """Rows as blocks: resample ROWS with replacement, recompute ``Σsum / Σtok``.

    ``row_sum[name]`` is the per-row SUM of that metric's per-span NLL and ``row_tok`` the
    per-row scored-token count, so a resample's point estimate is the same token-weighted
    mean the table prints. ``diffs`` names pairs ``(a, b)`` reported as ``a − b`` on the
    SAME resample (paired). Deterministic in ``seed``.
    """
    tok = np.asarray(row_tok, dtype=np.float64)
    n = tok.shape[0]
    if n == 0 or tok.sum() <= 0:
        raise ValueError("bootstrap needs at least one row with a positive token count")
    for name, v in row_sum.items():
        if np.asarray(v).shape != tok.shape:
            raise ValueError(f"row_sum[{name!r}] must match row_tok's shape")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ctok = tok[idx].sum(axis=1)
    draws = {k: np.asarray(v, dtype=np.float64)[idx].sum(axis=1) / ctok
             for k, v in row_sum.items()}
    alpha = (1.0 - level) / 2.0
    out: dict[str, dict] = {}
    for k, v in row_sum.items():
        lo, hi = np.quantile(draws[k], [alpha, 1.0 - alpha])
        out[k] = {"point": float(np.asarray(v, dtype=np.float64).sum() / tok.sum()),
                  "lo": float(lo), "hi": float(hi)}
    for a, b in diffs:
        d = draws[a] - draws[b]
        lo, hi = np.quantile(d, [alpha, 1.0 - alpha])
        out[f"{a}-{b}"] = {"point": out[a]["point"] - out[b]["point"],
                           "lo": float(lo), "hi": float(hi)}
    for v in out.values():
        v.update({"n_units": int(n), "n_boot": int(n_boot), "level": float(level)})
    return out


def per_token_nll(xh: torch.Tensor, w_head: torch.Tensor, lab: torch.Tensor,
                  keep_tok: torch.Tensor) -> torch.Tensor:
    """``[B, L]`` per-token CE from a coda state and the tied head, 0 at unscored positions.

    This is ``morph.model.transformer.accumulate_span_ce`` with the final ``index_add``
    removed: same one-row-at-a-time ``[L, V]`` logits (``[B, L, V]`` at V ~ 49k does not
    fit), same dtype cast, same ``keep_tok`` gate. Summing the output by
    ``span_ce_index``'s ``gid`` reproduces ``accumulate_span_ce`` bit for bit.
    """
    B = lab.shape[0]
    out = torch.zeros(lab.shape, device=lab.device, dtype=torch.float32)
    for b in range(B):
        logits = (xh[b].to(w_head.dtype) @ w_head.t()).float()          # [L, V]
        out[b] = F.cross_entropy(logits, lab[b], reduction="none") * keep_tok[b]
    return out


# ────────────────────────── the capture ──────────────────────────

def _collect_spans(nll_k: torch.Tensor, nll_dep: torch.Tensor, bag: torch.Tensor,
                   keep_tok: torch.Tensor, ok: torch.Tensor, row_base: int) -> list[dict]:
    """Ragged per-span records from a batch's per-token tables.

    ``nll_k`` ``[K, B, L]``, ``nll_dep`` ``[B, L]``, ``bag`` ``[B, L]`` clamped bag ids,
    ``keep_tok`` ``[B, L]`` bool, ``ok`` ``[B, S]`` the instrument's scored-span mask.
    Bin ``g`` is slot ``g − 1``'s next span (``span_ce_index``'s convention).
    """
    b_n, s_n = ok.shape
    out: list[dict] = []
    for b in range(b_n):
        for g in range(1, s_n + 1):
            if not bool(ok[b, g - 1]):
                continue
            pos = ((bag[b] == g) & keep_tok[b]).nonzero().flatten()
            if pos.numel() == 0:                     # ok already excludes these
                continue
            out.append({"row": row_base + b, "slot": g - 1,
                        "nll": nll_k[:, b, :].index_select(1, pos).double().numpy(),
                        "dep": nll_dep[b].index_select(0, pos).double().numpy()})
    return out


def _install_capture(store: list[dict]):
    """Swap ``MORPHTransformer._tul_fan_oracle`` for the real one + the per-token replay."""
    from morph.model.transformer import (MORPHTransformer, scatter_positions, span_ce_index,
                                         span_token_counts)

    real = MORPHTransformer._tul_fan_oracle

    @torch.no_grad()
    def capture(self, cells, xh, base, x0, bigram_emb, input_ids, labels, layout, L,
                keep, coda_kw, tg_reset, stats):
        real(self, cells, xh, base, x0, bigram_emb, input_ids, labels, layout, L,
             keep, coda_kw, tg_reset, stats)
        k = int(cells.shape[2])
        gid, keep_tok, lab, g_bins = span_ce_index(labels, layout)
        w_head = self.embed.lm_weight()
        n_tok = span_token_counts(gid, keep_tok, g_bins)[:, 1:]
        ok = layout.slot_valid & (n_tok > 0)
        per_stream = []
        for i in range(k):
            # IDENTICAL to the instrument's loop, one step short of the span index_add.
            values, pos = self._tul_fan_stream_write(cells, i, layout, L)
            x_i = scatter_positions(base, pos, values)
            xh_i = self._back_region(x_i, x0, bigram_emb, input_ids, inject_keep=keep,
                                     attn_kwargs=coda_kw, ret_reset_mask=tg_reset)
            per_stream.append(per_token_nll(xh_i, w_head, lab, keep_tok).cpu())
        store.append({
            "nll_k": torch.stack(per_stream, dim=0),                     # [K, B, L]
            "dep": per_token_nll(xh, w_head, lab, keep_tok).cpu(),       # [B, L]
            "bag": layout.bag_id.clamp(0, int(layout.slot_valid.shape[1])).cpu(),
            "keep_tok": keep_tok.cpu(),
            "ok": ok.cpu(),
            "stats": {kk: float(vv) for kk, vv in stats.items()},
        })

    MORPHTransformer._tul_fan_oracle = capture
    return lambda: setattr(MORPHTransformer, "_tul_fan_oracle", real)


# ────────────────────────── the table ──────────────────────────

def _fmt(v: dict) -> str:
    return f"{v['point']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"


def _print_table(res: dict) -> None:
    b = res["bootstrap"]
    print(f"\n{res['arm']} step {res['step']}  rows={res['rows']} depth={res['depth']} "
          f"K={res['fan_k']}  spans={res['n_spans']} tokens={res['n_tokens']} "
          f"mean_span_len={res['mean_span_len']:.2f}")
    print(f"  model fan/oracle_ce {res['model']['oracle_ce']:.6f}  "
          f"probe oracle {res['point']['oracle']:.6f}  "
          f"|d| {abs(res['model']['oracle_ce'] - res['point']['oracle']):.2e}")
    print(f"  model fan/mixed_ce  {res['model']['mixed_ce']:.6f}  "
          f"probe deployed {res['point']['deployed']:.6f}  "
          f"|d| {abs(res['model']['mixed_ce'] - res['point']['deployed']):.2e}")
    print("\n  reading         nats/token   95 % block bootstrap (rows)")
    for name in ("oracle", "best_single", "deployed", "uniform_mix", "prefix_mix"):
        print(f"  {name:<14} {_fmt(b[name])}")
    for i in range(res["fan_k"]):
        n = f"single_k{i}"
        print(f"  {n:<14} {res['point'][n]:+.4f}"
              f"   (model fan/stream_ce_k{i} {res['model'][f'stream_ce_k{i}']:+.4f})")
    print(f"  log(K)/mean_span_len = {res['log_k_over_span']:.4f} "
          f"(the per-token bound on uniform_mix − oracle)")
    print("\n  difference               nats/token   95 %")
    for a, c in (("deployed", "prefix_mix"), ("deployed", "oracle"),
                 ("uniform_mix", "oracle")):
        print(f"  {a + ' - ' + c:<24} {_fmt(b[f'{a}-{c}'])}")

    print("\n  offset bin   n_tok   prefix_mix   deployed     oracle   best_single")
    for i, lab in enumerate(res["bins"]):
        o = res["by_bin"]
        if o["n_tok"][i] == 0:
            continue
        print(f"  {lab:<11} {o['n_tok'][i]:>6}  {o['prefix_mix'][i]:>10.4f} "
              f"{o['deployed'][i]:>10.4f} {o['oracle'][i]:>10.4f} "
              f"{o['best_single'][i]:>12.4f}")
    w = res["winner_persistence"]
    def _ci(name: str) -> str:
        c = w.get(f"{name}_ci")
        return "" if c is None else f" [{c['lo']:+.4f}, {c['hi']:+.4f}]"
    print(f"\n  winner persistence P(k*_(s+1) = k*_s) = {w['p_repeat']:+.4f}{_ci('p_repeat')}"
          f"  vs chance {w['chance']:+.4f}{_ci('chance')}")
    print(f"  excess over chance {w['excess']:+.4f}{_ci('excess')} over {w['n_pairs']} "
          f"in-row pairs in {res['n_rows']} rows")
    print(f"  winner shares: {' '.join(f'{s:.3f}' for s in w['shares'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="NAME=CONFIG=PATH[=ovr1,ovr2]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6, help="forced slot depth, as the sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed; it does NOT draw rows")
    ap.add_argument("--tol", type=float, default=2e-4,
                    help="max |probe − instrument| on oracle and deployed, nats/token")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from _build import ROOT, build_cfg, parse_ckpt_spec
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or int(getattr(tul_rt.model_cfg, "fan_k", 0)) < 2:
        raise SystemExit(f"{label}: not a fan arm (tul.fan_k < 2)")
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    k_streams = int(tc.fan_k)
    tc.slot_mean_depth = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth))
    if int(getattr(tc, "slot_depth_fixed", 0)):
        tc.slot_depth_fixed = a.depth

    # The SAME row source as fan_stream_probe.py / core_depth_sweep.py, so the spans of
    # this probe are the spans of the stream probe.
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]

    store: list[dict] = []
    restore = _install_capture(store)
    try:
        with torch.no_grad():
            for inp, labels, layout, _idx in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                    model.tul_forward_ablated(inp.to(a.device), labels.to(a.device),
                                              layout.to(a.device), plan_mode="normal")
    finally:
        restore()
    if not store:
        raise SystemExit("no fan passes captured: the eval path did not call _tul_fan_oracle "
                         "(labels must be given, the model must be in eval, plan_mode normal)")

    # ── assemble the ragged spans ───────────────────────────────────────────
    spans: list[dict] = []
    row_base = 0
    model_sum = {kk: 0.0 for kk in ("oracle_ce", "single_ce", "mixed_ce")}
    model_sum.update({f"stream_ce_k{i}": 0.0 for i in range(k_streams)})
    model_tok = 0.0
    for rec in store:
        spans.extend(_collect_spans(rec["nll_k"], rec["dep"], rec["bag"], rec["keep_tok"],
                                    rec["ok"], row_base))
        row_base += int(rec["ok"].shape[0])
        nt = rec["stats"]["oracle_n_tokens"]
        model_tok += nt
        for kk in model_sum:
            model_sum[kk] += rec["stats"][kk] * nt
    model_ce = {kk: v / model_tok for kk, v in model_sum.items()}

    lmax = max(int(s["dep"].shape[0]) for s in spans)
    n_s = len(spans)
    nll = torch.zeros(n_s, k_streams, lmax, dtype=torch.float64)
    dep = torch.zeros(n_s, lmax, dtype=torch.float64)
    mask = torch.zeros(n_s, lmax, dtype=torch.bool)
    rows = np.zeros(n_s, dtype=np.int64)
    for i, s in enumerate(spans):
        n = int(s["dep"].shape[0])
        nll[i, :, :n] = torch.from_numpy(s["nll"])
        dep[i, :n] = torch.from_numpy(s["dep"])
        mask[i, :n] = True
        rows[i] = s["row"]

    # ── the readings ────────────────────────────────────────────────────────
    ell = span_stream_nll(nll, mask)                                 # [S, K]
    best, winner = oracle_per_span(nll, mask)                        # [S], [S]
    umix = uniform_mix_nll(nll, mask)                                # [S]
    ptok = prefix_mix_token_nll(nll, mask)                           # [S, Lmax]
    pmix = ptok.sum(dim=1)                                           # [S]
    dsum = (dep * mask.double()).sum(dim=1)                          # [S]
    ntok = mask.sum(dim=1).double()                                  # [S]
    n_tokens = float(ntok.sum())

    tel = float((pmix - umix).abs().max())
    if tel > 1e-9:
        raise AssertionError(f"prefix_mix does not telescope to uniform_mix: max |d| {tel:.3e}")

    single_tot = (ell.sum(dim=0) / n_tokens)                         # [K]
    k_best = int(single_tot.argmin())
    point = {
        "oracle": float(best.sum() / n_tokens),
        "deployed": float(dsum.sum() / n_tokens),
        "uniform_mix": float(umix.sum() / n_tokens),
        "prefix_mix": float(pmix.sum() / n_tokens),
        "best_single": float(single_tot[k_best]),
    }
    point.update({f"single_k{i}": float(single_tot[i]) for i in range(k_streams)})

    d_or = abs(point["oracle"] - model_ce["oracle_ce"])
    d_mx = abs(point["deployed"] - model_ce["mixed_ce"])

    # ── per-offset-bin profile ──────────────────────────────────────────────
    bidx = offset_bin_index(mask)                                    # [S, Lmax]
    orc_tok = nll.gather(1, winner.view(-1, 1, 1).expand(-1, 1, lmax)).squeeze(1)
    bst_tok = nll[:, k_best, :]
    by_bin = {"n_tok": [], "prefix_mix": [], "deployed": [], "oracle": [], "best_single": []}
    for i in range(len(BINS)):
        sel = mask & (bidx == i)
        n = float(sel.sum())
        by_bin["n_tok"].append(int(n))
        for name, t in (("prefix_mix", ptok), ("deployed", dep),
                        ("oracle", orc_tok), ("best_single", bst_tok)):
            by_bin[name].append(float((t * sel.double()).sum() / n) if n else float("nan"))
    if sum(by_bin["n_tok"]) != int(n_tokens):
        raise AssertionError("the offset bins do not partition the scored tokens")

    # ── row blocks for the bootstrap ────────────────────────────────────────
    uniq = np.unique(rows)
    r_of = {int(r): i for i, r in enumerate(uniq)}
    idx = np.array([r_of[int(r)] for r in rows])
    n_rows = len(uniq)

    def _rs(v: torch.Tensor) -> np.ndarray:
        out = np.zeros(n_rows, dtype=np.float64)
        np.add.at(out, idx, v.numpy().astype(np.float64))
        return out

    row_sum = {"oracle": _rs(best), "deployed": _rs(dsum), "uniform_mix": _rs(umix),
               "prefix_mix": _rs(pmix), "best_single": _rs(nll[:, k_best, :].sum(dim=1))}
    row_tok = _rs(ntok)
    boot = block_bootstrap(row_sum, row_tok,
                           [("deployed", "prefix_mix"), ("deployed", "oracle"),
                            ("uniform_mix", "oracle")],
                           n_boot=a.boot, seed=a.seed)

    wp = winner_persistence(winner.numpy(), rows, k_streams,
                            n_boot=a.boot, seed=a.seed)
    mean_len = n_tokens / float(n_s)
    res = {
        "arm": label, "config": config, "ckpt": path, "step": step, "rows": a.rows,
        "batch": a.batch, "depth": a.depth, "fan_k": k_streams,
        "n_spans": n_s, "n_tokens": int(n_tokens), "n_rows": n_rows,
        "mean_span_len": mean_len,
        "log_k_over_span": math.log(float(k_streams)) / mean_len,
        "k_best": k_best,
        "point": point,
        "model": model_ce,
        "agreement": {"oracle_vs_fan_oracle_ce": d_or, "deployed_vs_fan_mixed_ce": d_mx,
                      "telescope_max_abs": tel, "tol": a.tol,
                      "ok": bool(d_or <= a.tol and d_mx <= a.tol)},
        "bootstrap": boot,
        "bins": bin_labels(), "by_bin": by_bin,
        "winner_persistence": wp,
        # the raw per-span winner and its row, so a later reading (lineage length, a
        # different pairing) does not need the GPU again.
        "span_winner": winner.numpy().tolist(), "span_row": rows.tolist(),
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    _print_table(res)
    print(f"\nwrote {a.out}")
    # Last, so the table and the JSON exist whatever the verdict: the probe's per-token
    # replay must reproduce the shipped instrument on the same rows, or nothing above is
    # a reading of this model.
    if not res["agreement"]["ok"]:
        raise AssertionError(
            f"the probe's per-token replay does not reproduce the instrument: "
            f"oracle {point['oracle']:.6f} vs fan/oracle_ce {model_ce['oracle_ce']:.6f} "
            f"(|d| {d_or:.2e}); deployed {point['deployed']:.6f} vs fan/mixed_ce "
            f"{model_ce['mixed_ce']:.6f} (|d| {d_mx:.2e}); tol {a.tol:.1e}")


if __name__ == "__main__":
    main()
