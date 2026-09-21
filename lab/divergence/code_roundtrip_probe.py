"""LCM's round-trip gap on the LCTUL thinker: does the sampled code live on the manifold?

LCM's instrument (arXiv 2412.08821 §2.4.1, and
`docs/references/tul-latent-emission/lcm/2026-09-21-lcm-reading.md` §2): regress a
next-sentence vector, DECODE it to text, RE-ENCODE that text, and read how far the
re-encoded vector sits from the truth against how far the prediction did.

    l2   = d(x_hat, x_true)          the training loss
    l2-r = d(encode(decode(x_hat)), x_true)
    gap  = l2-r − l2

Base-LCM (pure MSE regression) posts +0.060 / +0.057 / +0.054 / +0.057 on ROC / C4 /
Wikipedia-en / Gutenberg — the decoder pulls its prediction about a quarter of its own
l2 onto the nearest real sentence, which is the direct evidence that a conditional MEAN
is off the manifold. The diffusion LCMs post −0.002 / −0.004 / −0.010 / 0.000: their
sample is already a point the decoder can say.

THE PORT. The LCTUL slot is the concept vector, ``TULCodeEncoder`` (E) is SONAR's encoder
and the strict coda is SONAR's decoder. ``d = 1 − cos`` rather than l2, because every code
on this path is RMS-normalised per cell (``code_rmsnorm``) and a norm difference is not a
content difference. For every eligible slot of N validation rows:

  z_true   E's code of the row's own next span — the thinker's flow TARGET.
  z_hat    the sampled code, ``code_mode="sampled"`` at k = ``tul.code_infer_steps``,
           and again at k = 1 (the one-step sample, the Euler-depth control).
  decode   the strict coda writing the next span GREEDILY from z_hat, length matched to
           the true span (`_span_decode.py` explains why length matched).
  z_re     E's code of that decoded row — same E, same positions, same context.

and the table reads ``cos(z_hat, z_true)``, ``cos(z_re, z_hat)`` (the round trip itself),
``cos(z_re, z_true)`` and ``gap = d(z_re, z_true) − d(z_hat, z_true)``, token weighted,
with a ROW-block bootstrap.

THE FLOOR IS NOT OPTIONAL. The third condition decodes and re-encodes the TRUTH code
``z_true`` itself. Its gap is what this encoder and this coda cost on a round trip when
the code is exactly right, and every other row of the table is read against it — a +0.05
gap means nothing if the truth's own round trip already loses 0.20.

ONE ARCHITECTURAL FACT THAT LIMITS EVERY DECODE HERE. Token 0 of span ``s+1`` is emitted at
the BOUNDARY token of span ``s`` (the only position trained to emit it at
``tul.emit_weight: 0``), and that token sits in bag ``s``, so under the strict geometry it
reads span ``s``'s tokens and the prefix cells of slots BEFORE ``s`` — never slot ``s``'s own
cells. A span's first token is therefore written blind to the code the rest of the span is
written from. The shipped grader (``_tul_code_grade``) scores the same positions; this probe
does not work around it.

TWO DIFFERENCES FROM LCM, both structural and neither fixable here:
  * SONAR re-encodes a sentence in isolation; E pools the span's PRELUDE states. Under
    the strict geometry a span's prelude depends on that span's tokens alone, so the
    re-encode is still a function of the decoded text only — but it is this model's
    encoder, not an independent one.
  * LCM's decoder is free running; this decode is length matched (see `_span_decode.py`).

Usage (a probe host, never the trainer's GPU):

    python lab/divergence/code_roundtrip_probe.py \
        --ckpt tul-code-20k=tul_code=/path/step_20000.pt \
        --rows 48 --batch 3 --out code_roundtrip_tul-code-20k.json

The pure functions (``cos_flat``, ``roundtrip_readings``) are tested on synthetic vectors
in ``tests/test_lcm_instruments.py``.
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
from fan_mixture_probe import block_bootstrap  # noqa: E402  (ONE home for the bootstrap)

__all__ = ["cos_flat", "roundtrip_readings", "block_bootstrap"]


# ────────────────────────── pure readings ──────────────────────────

def cos_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine of two cell stacks, flattened over every axis after the first.

    ``[N, M, C]`` (or ``[N, D]``) -> ``[N]``, in float64. A slot's code is M cells and the
    coda reads all of them, so the comparison is on the whole stack, not per cell.
    """
    if a.shape != b.shape:
        raise ValueError(f"shapes must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.dim() < 2:
        raise ValueError(f"need at least [N, D], got {tuple(a.shape)}")
    x = a.reshape(a.shape[0], -1).double()
    y = b.reshape(b.shape[0], -1).double()
    num = (x * y).sum(dim=1)
    den = x.norm(dim=1) * y.norm(dim=1)
    return num / den.clamp_min(1e-30)


def roundtrip_readings(z_hat: torch.Tensor, z_re: torch.Tensor,
                       z_true: torch.Tensor) -> dict[str, torch.Tensor]:
    """LCM's round-trip battery per slot, with ``d = 1 − cos``.

    ``[N, M, C]`` each -> ``[N]`` each of:

      ``cos_hat_true``  how good the prediction is;
      ``cos_re_hat``    the round trip: how far the decode+re-encode MOVED the prediction;
      ``cos_re_true``   where the round-tripped point lands;
      ``d_hat``         ``1 − cos_hat_true``  (LCM's l2);
      ``d_re``          ``1 − cos_re_true``   (LCM's l2-r);
      ``move``          ``1 − cos_re_hat``;
      ``gap``           ``d_re − d_hat``      (LCM's ``l2-r − l2``).

    A prediction that is already a real span round trips to itself: ``move`` 0 and
    ``gap`` 0. A prediction that is an average of several real spans is pulled onto one
    of them, so ``move`` is large and ``gap`` is positive.
    """
    ch = cos_flat(z_hat, z_true)
    cr = cos_flat(z_re, z_true)
    cm = cos_flat(z_re, z_hat)
    d_hat, d_re = 1.0 - ch, 1.0 - cr
    return {"cos_hat_true": ch, "cos_re_hat": cm, "cos_re_true": cr,
            "d_hat": d_hat, "d_re": d_re, "move": 1.0 - cm, "gap": d_re - d_hat}


# ────────────────────────── the run ──────────────────────────

_COND_HELP = "the sampler's k for the second (one-step) condition"


def _rowsum(vals: np.ndarray, weights: np.ndarray, row_ix: np.ndarray, n_rows: int):
    out = np.zeros(n_rows, dtype=np.float64)
    np.add.at(out, row_ix, vals.astype(np.float64) * weights.astype(np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=ovr1,ovr2]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0,
                    help="sampler k for the main condition; 0 = tul.code_infer_steps")
    ap.add_argument("--one-step", type=int, default=1, help=_COND_HELP)
    ap.add_argument("--max-span-tokens", type=int, default=32)
    ap.add_argument("--code-seed", type=int, default=0, help="the sampler's eval stream")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed; it does NOT draw rows")
    ap.add_argument("--check-parity", action="store_true",
                    help="falsify the parallel decode on the first batch (2x its cost)")
    ap.add_argument("--samples", type=int, default=6, help="decoded spans to print")
    ap.add_argument("--out", required=True, help="the .json path; the .txt sits beside it")
    a = ap.parse_args()
    t0 = time.time()

    from _build import ROOT, build_cfg, parse_ckpt_spec
    from _rows import pack_rows, stream_from_loader
    from _span_decode import (PinnedCells, check_span_layout, greedy_decode,
                              row_logits, span_slots)
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    import morph

    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or not bool(getattr(tul_rt.model_cfg, "code", False)):
        raise SystemExit(f"{label}: not an LCTUL arm (tul.code is false)")
    if not bool(getattr(tul_rt.model_cfg, "tg_geometry", "") == "strict"):
        raise SystemExit(f"{label}: the parallel span decode needs tul.tg_geometry=strict")
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    k_main = int(a.steps or tc.code_infer_steps)
    k_one = int(a.one_step)
    J = int(a.max_span_tokens)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]

    conds = [(f"k{k_main}", k_main), (f"k{k_one}", k_one), ("truth", 0)]
    acc: dict[str, list] = {c: [] for c, _ in conds}
    n_rows_seen = 0
    ex_lines: list[str] = []
    identity_max = 0.0
    parity_bad = None

    with PinnedCells(model, "code") as pin:
        for bi, (inp, labels, layout, _idx) in enumerate(batches):
            ids = inp.to(a.device)
            lay = layout.to(a.device)
            elig, lens, first, src0 = span_slots(lay, J)
            check_span_layout(lay, elig, first, src0)
            if not bool(elig.any()):
                n_rows_seen += ids.shape[0]
                continue
            ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda")

            # ── the three codes, each captured off the SHIPPED forward ──────────
            with ac:
                pin.pin = None
                lg_enc = row_logits(model, ids, lay, "encoder").float()
                z_true = pin.last.float().clone()
                model(ids, slot_layout=lay, code_mode="sampled", code_steps=k_main,
                      code_seed=a.code_seed)
                z_k = pin.last.float().clone()
                model(ids, slot_layout=lay, code_mode="sampled", code_steps=k_one,
                      code_seed=a.code_seed)
                z_1 = pin.last.float().clone()

                # The pin is an override, not a second write: a pinned forward at the
                # captured value must reproduce the unpinned one elementwise.
                pin.pin = z_true
                lg_pin = row_logits(model, ids, lay, "encoder").float()
                identity_max = max(identity_max,
                                   float((lg_pin - lg_enc).abs().max()))
                del lg_pin, lg_enc

            for cname, z_used in (("truth", z_true), (f"k{k_main}", z_k), (f"k{k_one}", z_1)):
                with ac:
                    pin.pin = z_used
                    ids_d, cand = greedy_decode(model, ids, lay, elig, lens, first, src0,
                                                J, code_mode="encoder")
                    if a.check_parity and bi == 0 and parity_bad is None:
                        s_ar = torch.arange(elig.shape[1], device=elig.device)
                        even = elig & ((s_ar % 2) == 0).view(1, -1)
                        ids_e, cand_e = greedy_decode(model, ids, lay, even, lens, first,
                                                      src0, J, code_mode="encoder")
                        d = (cand_e[even] != cand[even]).sum().item()
                        parity_bad = int(d)
                    # the re-encode: E on the DECODED row, cells NOT pinned
                    pin.pin = None
                    row_logits(model, ids_d, lay, "encoder")
                    z_re = pin.last.float().clone()
                r, s = elig.nonzero(as_tuple=True)
                # the CHANCE level: another scored slot's true code, the batch rolled by
                # one. Without it a cosine of 0.30 cannot be read — LCM's CA exists for
                # the same reason (an l2 is not interpretable without in-batch negatives).
                zt = z_true[r, s]
                zt_other = torch.roll(zt, 1, dims=0)
                rd = roundtrip_readings(z_used[r, s], z_re[r, s], zt)
                rd["cos_true_other"] = cos_flat(zt, zt_other)
                rd["cos_re_other"] = cos_flat(z_re[r, s], zt_other)
                acc[cname].append({
                    "row": (r + n_rows_seen).cpu().numpy(),
                    "ntok": lens[r, s].cpu().numpy().astype(np.float64),
                    **{k: v.cpu().numpy() for k, v in rd.items()}})
                if bi == 0 and len(ex_lines) < a.samples * 4:
                    for i in range(min(a.samples, r.numel())):
                        b_i, s_i = int(r[i]), int(s[i])
                        n = int(lens[b_i, s_i])
                        ex_lines.append(
                            f"  [{cname}] slot {s_i} row {b_i}: "
                            f"{tok.decode(cand[b_i, s_i, :n].tolist())!r}")
            n_rows_seen += ids.shape[0]
            print(f"  batch {bi + 1}/{len(batches)} done  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        pin.pin = None

    # ── the table ────────────────────────────────────────────────────────────
    res = {"arm": label, "config": config, "ckpt": path, "step": step, "rows": a.rows,
           "batch": a.batch, "k_main": k_main, "k_one": k_one, "code_seed": a.code_seed,
           "max_span_tokens": J, "morph_module": os.path.dirname(morph.__file__),
           "identity_check_max_abs": identity_max,
           "parity_check_mismatched_tokens": parity_bad,
           "conditions": {}}
    keys = ["cos_hat_true", "cos_re_hat", "cos_re_true", "d_hat", "d_re", "move", "gap",
            "cos_true_other", "cos_re_other"]
    for cname, _ in conds:
        parts = acc[cname]
        if not parts:
            raise SystemExit("no eligible slot was scored: raise --rows")
        row = np.concatenate([p["row"] for p in parts])
        ntok = np.concatenate([p["ntok"] for p in parts])
        uniq, ix = np.unique(row, return_inverse=True)
        n_rows = int(uniq.shape[0])
        row_sum = {k: _rowsum(np.concatenate([p[k] for p in parts]), ntok, ix, n_rows)
                   for k in keys}
        row_tok = _rowsum(np.ones_like(ntok), ntok, ix, n_rows)
        boot = block_bootstrap(row_sum, row_tok, [("d_re", "d_hat")],
                               n_boot=a.boot, seed=a.seed)
        res["conditions"][cname] = {
            "n_slots": int(row.shape[0]), "n_tokens": float(ntok.sum()),
            "n_rows": n_rows, "bootstrap": boot}
    res["wall_clock_s"] = time.time() - t0

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    lines = [f"{label} step {step}  rows={a.rows} batch={a.batch} k_main={k_main} "
             f"k_one={k_one} max_span_tokens={J}",
             f"morph from {res['morph_module']}",
             f"identity check (pinned == unpinned logits) max|d| = {identity_max:.3e}",
             f"parity check (even-only decode vs all-live) mismatched tokens = "
             f"{parity_bad if parity_bad is not None else 'not run'}", ""]
    hdr = ("condition   slots   tokens   cos(zhat,ztrue)   cos(zre,zhat)   "
           "cos(zre,ztrue)   cos(zre,other)        gap = d_re - d_hat")
    lines.append(hdr)
    for cname, _ in conds:
        c = res["conditions"][cname]
        b = c["bootstrap"]
        def f(n):
            return f"{b[n]['point']:+.4f}"
        lines.append(
            f"{cname:<11} {c['n_slots']:>5} {c['n_tokens']:>8.0f}   "
            f"{f('cos_hat_true'):>15}   {f('cos_re_hat'):>13}   {f('cos_re_true'):>14}   "
            f"{f('cos_re_other'):>14}   "
            f"{b['d_re-d_hat']['point']:+.4f} "
            f"[{b['d_re-d_hat']['lo']:+.4f}, {b['d_re-d_hat']['hi']:+.4f}]")
    _ct = res["conditions"]["truth"]["bootstrap"]["cos_true_other"]
    lines += ["",
              f"CHANCE for every cosine column: cos(z_true, another scored slot's z_true) "
              f"= {_ct['point']:+.4f} [{_ct['lo']:+.4f}, {_ct['hi']:+.4f}]",
              "A cos(zre, ztrue) at that level means the round trip kept NO slot identity.",
              "", "95 % row-block bootstrap on every column:"]
    for cname, _ in conds:
        b = res["conditions"][cname]["bootstrap"]
        lines.append(f"  {cname}")
        for k in keys + ["d_re-d_hat"]:
            lines.append(f"    {k:<14} {b[k]['point']:+.4f} "
                         f"[{b[k]['lo']:+.4f}, {b[k]['hi']:+.4f}]")
    lines += ["", "decoded spans (first batch):"] + ex_lines
    lines += ["", f"wall clock {res['wall_clock_s']:.0f} s"]
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {a.out} and {txt}")
    if identity_max > 1e-2:
        raise AssertionError(
            f"the cell pin is not an identity at the captured value: max|d| "
            f"{identity_max:.3e} on the logits; nothing above is a reading of this model")
    if parity_bad:
        raise AssertionError(
            f"the parallel span decode is not independent: {parity_bad} tokens differ "
            f"when only the even slots decode; the strict geometry assumption fails here")


if __name__ == "__main__":
    main()
