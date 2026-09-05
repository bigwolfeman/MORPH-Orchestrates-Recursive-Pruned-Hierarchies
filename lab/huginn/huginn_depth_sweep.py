"""Loop contribution on Huginn-3.5B (tomg-group-umd/huginn-0125), the arc's instruments.

Prereg: lab/experiments/planned/2026-09-04-huginn-loop-contribution.md. Token CE as a
function of the recurrence count `num_steps` on identical OpenWebText rows (Huginn's own
tokenizer; the same documents in the same order as the MORPH sweeps), per-row and
per-offset-in-span CE sums (`lab/divergence/_earning.EarningProfile`, spans cut by MORPH's
boundary rule rebuilt on Huginn's tokenizer), paired-bootstrap CIs over rows. The JSON is
`score_arc_e0.py`-compatible (`--sweep huginn=path --lo 1 --hi 6`).

  python lab/huginn/huginn_depth_sweep.py --rows 480 --batch 3 --seq 1024 \
      --steps 1,2,3,4,6,8,12,16,24,32,48,64 --out lab/experiments/results/<dir>/sweep_huginn.json

transformers 5 vs the model's remote code (written for 4.x): `_tied_weights_keys` must be a
mapping, and the tie is asserted after the load (the checkpoint stores wte only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/lab/divergence")
from _earning import EarningProfile, offsets_from_ids  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402

MODEL = "tomg-group-umd/huginn-0125"


def load_huginn(device: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    base = get_class_from_dynamic_module("raven_modeling_minimal.RavenPreTrainedModel", MODEL)
    if isinstance(getattr(base, "_tied_weights_keys", None), list):
        base._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    cfg.tie_word_embeddings = True
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()
    head, wte = model.lm_head.weight, model.transformer.wte.weight
    if head.data_ptr() != wte.data_ptr():
        if not torch.equal(head, wte):
            raise RuntimeError("lm_head is neither tied to nor equal to wte after the load")
        model.lm_head.weight = wte
    return model, tok, cfg


def build_rule(tok, vocab: int):
    from morph.model.tul_layout import BoundaryRule, boundary_lut_from_tokenizer
    from morph.training.tul_setup import BOUNDARY_SUBSTRINGS, BOUNDARY_SUFFIX_CHARS
    eos = int(tok.eos_token_id if tok.eos_token_id is not None else 0)
    lut = boundary_lut_from_tokenizer(MODEL, vocab, eos, cache_dir=f"{ROOT}/ignore/tul_cache",
                                      suffix_chars=BOUNDARY_SUFFIX_CHARS,
                                      substrings=tuple(BOUNDARY_SUBSTRINGS))
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=eos)


@torch.no_grad()
def ce_map(model, x, y, n_steps: int, device: str) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(input_ids=x.to(device), num_steps=int(n_steps))
    logits = out.logits.float()
    B, L, V = logits.shape
    return F.cross_entropy(logits.reshape(B * L, V), y.to(device).reshape(B * L),
                           reduction="none").reshape(B, L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", default="1,2,3,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--dataset", default=os.path.expanduser(
        "~/.cache/huggingface/datasets/openwebtext/**/openwebtext-train-*.arrow"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")]
    device = a.device

    from morph.training.data import create_dataloader
    model, tok, cfg = load_huginn(device)
    vocab = int(getattr(cfg, "padded_vocab_size", cfg.vocab_size))
    rule = build_rule(tok, vocab)
    loader = create_dataloader(MODEL, a.dataset, a.seq, a.batch, split="validation",
                               skip_samples=0, bag_size=0, tul=None)
    batches = []
    while len(batches) * a.batch < a.rows:
        x, y = next(loader)[:2]
        batches.append((x, y))
    n_rows = len(batches) * a.batch
    offs = [[offsets_from_ids(x[b].numpy(), rule) for b in range(x.shape[0])] for x, _ in batches]
    prof = EarningProfile(steps, n_rows)
    arm = {"model": MODEL, "rows": n_rows, "seq": a.seq, "batch": a.batch,
           "mean_recurrence": int(getattr(cfg, "mean_recurrence", 0)), "depths": {}}
    row_sum: dict[int, np.ndarray] = {}
    for d in steps:
        t0 = time.time()
        tot = tot_n = 0.0
        rs: list[float] = []
        for i, (x, y) in enumerate(batches):
            ce = ce_map(model, x, y, d, device)
            valid = (y >= 0)
            for b in range(x.shape[0]):
                prof.add(d, i * a.batch + b, ce[b], valid[b], offs[i][b])
            ce_c = ce.cpu()
            tot += float(ce_c[valid].sum())
            tot_n += float(valid.sum())
            rs.extend((ce_c * valid).sum(dim=1).tolist())
        row_sum[d] = np.asarray(rs)
        arm["depths"][d] = {"ce_tokens": tot / tot_n, "n_tokens": tot_n,
                            "seconds": round(time.time() - t0, 1)}
        print(f"huginn steps={d:<3d} ce={tot/tot_n:.4f}  ({time.time()-t0:.0f}s, "
              f"mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB)", flush=True)
    row_cnt = np.asarray(prof.row_n)
    arm["ci_ce_tokens"] = {}
    for lo, hi in [(1, 3), (3, 6), (6, 16), (16, 32), (32, 64), (1, 6), (3, 32)]:
        if lo in row_sum and hi in row_sum:
            arm["ci_ce_tokens"][f"K{lo}-K{hi}"] = paired_bootstrap_ci(row_sum[lo], row_sum[hi], row_cnt)
    for k, v in arm["ci_ce_tokens"].items():
        print(f"huginn ce_tokens {k}: {v['point']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] over {v['n_units']} rows", flush=True)
    arm["profile"] = prof.to_json()
    arm["row_ce_sum"] = {str(d): row_sum[d].tolist() for d in steps}
    arm["row_n_tokens"] = row_cnt.tolist()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"huginn": arm}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
