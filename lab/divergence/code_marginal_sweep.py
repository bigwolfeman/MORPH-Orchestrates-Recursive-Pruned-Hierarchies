"""The inference-depth curve of a TUL-Code model under the RIGHT metric.

For a code model the inference depth is the sampler's Euler step count k. The runner's
sweep (core_depth_sweep.py) scores each k by one-draw CE, which cannot reward a better
draw of the right distribution. This sweep scores each k by the K-draw marginal
(morph/training/code_eval.py: log of the mean over K sampled codes of the span
likelihood, a lower bound that tightens with K) beside the one-draw CE, on the same rows,
plus k = 0 = the encoder's code (the ceiling).

    python lab/divergence/code_marginal_sweep.py \
        --ckpt tul-code=tul_code=/path/step_5000.pt --ks 1,2,4,8,16 --draws 8 --rows 96 \
        --out .../marginal_sweep.json

Read-only. Same validation stream and packer as core_depth_sweep.py.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.training.code_eval import code_marginal_ce  # noqa: E402


@torch.no_grad()
def _encoder_ce(model, inp, labels, layout, device) -> tuple[float, int]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal",
                                        code_mode="encoder")
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    keep = (lab >= 0) & (~layout.slot_mask)
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    return float(ce[keep].sum()), int(keep.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    ks = [int(x) for x in a.ks.split(",")]

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path = a.ckpt.split("=", 2)
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(getattr(tul_rt.model_cfg, "code", False)), \
        "a TUL-Code checkpoint is required"
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    batches = [(inp, labels.to(device), lay.to(device)) for inp, labels, lay, _ in batches]
    n_rows = sum(inp.shape[0] for inp, _, _ in batches)
    print(f"{label}: step {step}, {n_rows} rows, K={a.draws} draws, ks={ks}", flush=True)

    # per batch: token count and, per k, the summed nats of both metrics (bootstrap unit = batch)
    _enc = [_encoder_ce(model, inp, lab, lay, device) for inp, lab, lay in batches]
    enc = np.array([e for e, _ in _enc])
    n_tok = np.array([n for _, n in _enc], dtype=np.float64)
    out = {"label": label, "step": step, "rows": n_rows, "draws": a.draws, "ks": ks,
           "ce_encoder": float(enc.sum() / n_tok.sum()), "per_k": {}}
    print(f"  k=0 (encoder)   ce={out['ce_encoder']:.4f}", flush=True)
    marg = {}
    for k in ks:
        m_sum, s_sum = [], []
        for inp, lab, lay in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                r = code_marginal_ce(model, inp.to(device), lab, lay, a.draws, k)
            nt = float(((lab >= 0) & (~lay.slot_mask)).sum())
            m_sum.append(r["ce_marginal"] * nt)
            s_sum.append(r["ce_single_mean"] * nt)
        m_sum, s_sum = np.array(m_sum), np.array(s_sum)
        marg[k] = m_sum
        out["per_k"][k] = {"ce_marginal": float(m_sum.sum() / n_tok.sum()),
                           "ce_single_mean": float(s_sum.sum() / n_tok.sum())}
        print(f"  k={k:<3d}          ce_marginal={out['per_k'][k]['ce_marginal']:.4f}  "
              f"ce_single_mean={out['per_k'][k]['ce_single_mean']:.4f}", flush=True)
    # paired bootstrap over batches: marginal at k_max minus at k_min
    k0, k1 = ks[0], ks[-1]
    d = (marg[k1] - marg[k0])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        idx = rng.integers(0, len(d), len(d))
        bs.append(d[idx].sum() / n_tok[idx].sum())
    out["marginal_k_last_minus_k_first"] = [float(d.sum() / n_tok.sum()),
                                            float(np.percentile(bs, 2.5)),
                                            float(np.percentile(bs, 97.5))]
    print(f"  marginal k={k1} minus k={k0}: {out['marginal_k_last_minus_k_first'][0]:+.4f} "
          f"[{out['marginal_k_last_minus_k_first'][1]:+.4f}, "
          f"{out['marginal_k_last_minus_k_first'][2]:+.4f}] over {len(d)} batches", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
