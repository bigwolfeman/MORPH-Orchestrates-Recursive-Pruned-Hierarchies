"""P5 refuter: does the MUX local head beat the corpus UNIGRAM prior?

The v1a pre-registration (lab/experiments/planned/2026-08-25-mux-head-arm-v1a.md)
says the head only "bites" if its local CE beats 0.8x the CE of the best
span-INDEPENDENT predictor against the same targets. That predictor is the corpus
unigram distribution: it knows the marginal token frequencies and nothing about
which span it is being asked to summarise. A head that cannot beat it carries no
span-specific content, however low its loss looks in isolation.

The baseline is `MORPHTransformer._tul_mux_loss` with the model's log-probs swapped
for `log p_uni`, and NOTHING else changed — same targets from
`mux_span_targets(input_ids, layout, rho)`, same weighted sum, same denominator
`slot_supervised.sum().clamp(min=1)`. Mirroring that reduction exactly is the whole
point; a different normaliser makes the two numbers incomparable.

p_uni is estimated from an INDEPENDENT sample of TRAINING tokens (the same stream the
run trained on, plain NTP, no slot insertion) with add-one smoothing, never from the
val batches it is scored on.

Eval batches are the standard probe set: the loader args of `region_shapley.py` /
`slot_path_worth.py` / `readout_jacobian.py` (validation, skip_samples=60_000,
batch 6, 8 batches), materialised once.

    PYTHONPATH=. python lab/divergence/mux_unigram_baseline.py \
        --ckpts checkpoints/morph/tul-v1a-s1/step_2500.pt --out out.json

NOT pre-registered — forensic probe on a failed arm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.model.tul import mux_span_targets                # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402
from morph.training.tul_setup import build_tul_runtime      # noqa: E402
from morph.training.train import load_checkpoint            # noqa: E402


def unigram_from_train(cfg, vocab_size: int, n_batches: int) -> tuple[torch.Tensor, int]:
    """Token counts over `n_batches` PLAIN (no-TUL) training batches.

    No slot layout: these are raw corpus tokens, which is what a corpus unigram prior
    is. The stream is the deterministic unshuffled train split the run itself read.
    """
    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="train", tul=None))
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    total = 0
    for i in range(n_batches):
        x, _y = next(loader)
        counts += torch.bincount(x.reshape(-1), minlength=vocab_size).to(torch.float64)
        total += int(x.numel())
        if (i + 1) % 50 == 0:
            print(f"    unigram: {i + 1}/{n_batches} batches, {total:,} tokens", flush=True)
    return counts, total


def weighted_ce(logp_at_ids: torch.Tensor, pos_valid: torch.Tensor,
                alpha: torch.Tensor, sup: torch.Tensor) -> float:
    """The EXACT reduction of `_tul_mux_loss`, given log-probs already gathered at ids."""
    ce = -torch.where(pos_valid, alpha * logp_at_ids,
                      torch.zeros_like(logp_at_ids)).sum()
    return float(ce / sup.sum().to(ce.dtype).clamp(min=1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tul_v1a")
    ap.add_argument("--ckpts", nargs="*", default=[],
                    help="optional: also report the MODEL's mux_local on the same batches")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--unigram-batches", type=int, default=400,
                    help="train batches for p_uni (400 x 6 x 1024 = 2.46 M tokens)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = build_cfg(a.config, ["training.batch_size=6", "model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None, "config has no TUL runtime — wrong config for this probe"
    tc = tul_rt.model_cfg
    V = int(cfg.model.vocab_size)
    slot_id = int(tc.slot_id)
    rho = float(tc.mux_rho)
    print(f"vocab {V}  slot_id {slot_id}  mux_rho {rho}  mux_tau {tc.mux_tau}  "
          f"mux_beta {tc.mux_beta}")

    # ── fixed eval set (the standard probe set) ───────────────────────────
    vloader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg))
    batches = []
    for _ in range(a.batches):
        bx, by, bl = next(vloader)
        batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
    print(f"fixed eval set: {len(batches)} batches of {batches[0][0].shape[0]}\n")

    # ── p_uni from an INDEPENDENT training sample ─────────────────────────
    print(f"estimating p_uni from {a.unigram_batches} train batches…", flush=True)
    counts, n_tok = unigram_from_train(cfg, V, a.unigram_batches)
    nz = int((counts > 0).sum())
    print(f"  {n_tok:,} training tokens, {nz}/{V} vocab entries observed")

    # add-one smoothing over the full vocab.
    p_all = (counts + 1.0) / (counts.sum() + float(V))
    # `_tul_mux_loss` masks the slot id to -inf and re-normalises over the rest, so the
    # comparable prior does the same. (The targets never ARE slot_id — pos_valid
    # excludes slot positions — so this only changes the normaliser, by ~5%.)
    p_masked = p_all.clone()
    p_masked[slot_id] = 0.0
    p_masked = p_masked / p_masked.sum()
    logp_masked = torch.log(p_masked).cuda()
    logp_all = torch.log(p_all).cuda()
    print(f"  p_uni[slot_id] before masking = {float(p_all[slot_id]):.6e} "
          f"(mass removed by the mask)")

    # ── the baseline, batch by batch ──────────────────────────────────────
    per_batch_masked, per_batch_all, sup_counts, pos_counts = [], [], [], []
    for x, _y, layout in batches:
        pos_valid, alpha, tgt_slot, sup = mux_span_targets(x, layout, rho)
        per_batch_masked.append(weighted_ce(logp_masked[x], pos_valid, alpha, sup))
        per_batch_all.append(weighted_ce(logp_all[x], pos_valid, alpha, sup))
        sup_counts.append(int(sup.sum()))
        pos_counts.append(int(pos_valid.sum()))

    uni_masked = sum(per_batch_masked) / len(per_batch_masked)
    uni_all = sum(per_batch_all) / len(per_batch_all)
    ln_V = math.log(V)
    print(f"\nunigram baseline CE (slot_id masked, the comparable one) = {uni_masked:.4f}")
    print(f"unigram baseline CE (no mask)                            = {uni_all:.4f}")
    print(f"uniform prior ln(V)                                      = {ln_V:.4f}")
    print(f"0.8 x unigram threshold                                  = {0.8 * uni_masked:.4f}")
    print(f"per-batch (masked): " + " ".join(f"{v:.4f}" for v in per_batch_masked))
    print(f"supervised slots/batch: {sup_counts}   supervising positions: {pos_counts}")

    # ── optional: the MODEL's own mux_local on the SAME batches ───────────
    model_mux = {}
    if a.ckpts:
        model, _ = build_model(cfg, device="cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for ck in a.ckpts:
            load_checkpoint(ck, model, scaler, torch.device("cuda"))
            model.eval()
            vals = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for x, y, layout in batches:
                    out = model(x, labels=y, slot_layout=layout)
                    assert "mux_local" in out, "mux_local missing — mux_beta is 0?"
                    vals.append(float(out["mux_local"]))
            m = sum(vals) / len(vals)
            model_mux[ck] = {"mux_local_mean": m, "per_batch": vals,
                             "ratio_to_unigram": m / uni_masked,
                             "beats_0.8x": bool(m < 0.8 * uni_masked)}
            print(f"\n{ck}\n  mux_local (this probe set) = {m:.4f}   "
                  f"ratio to unigram = {m / uni_masked:.4f}   "
                  f"beats 0.8x: {m < 0.8 * uni_masked}")

    res = {
        "config": a.config, "vocab_size": V, "slot_id": slot_id, "mux_rho": rho,
        "eval_batches": a.batches, "batch_size": int(cfg.training.batch_size),
        "unigram_train_tokens": n_tok, "unigram_vocab_observed": nz,
        "unigram_ce_slotid_masked": uni_masked,
        "unigram_ce_unmasked": uni_all,
        "unigram_ce_per_batch_masked": per_batch_masked,
        "threshold_0.8x": 0.8 * uni_masked,
        "uniform_ln_V": ln_V,
        "supervised_slots_per_batch": sup_counts,
        "supervising_positions_per_batch": pos_counts,
        "model_mux_local": model_mux,
    }
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
