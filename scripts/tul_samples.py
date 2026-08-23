#!/usr/bin/env python
"""Generation samples and degeneration metrics for TUL checkpoints.

    python scripts/tul_samples.py --ckpt gate_5k=tul_gate=<path>.pt [--ckpt ...]

Why this is a standalone script and not more logging inside the training loop:
`gate_bakeoff.sh` launches each arm as its OWN `python -m morph.training.train`, so a
mid-campaign edit to `train.py` would silently make arms 2 and 3 differ from arm 1 by
more than the variable under test. This reads finished checkpoints instead and cannot
perturb a running arm.

THREE DECODE MODES, NEVER ONE. A val CE cannot see degeneration and neither can a single
decode setting:

  * greedy (temperature 0) is the deterministic floor. This is where a repetition loop
    shows up, and a loop scores an EXCELLENT perplexity -- 1.46 measured against real
    text's 32.44 -- so fluency without a diversity number beside it is worthless.
  * top-k 50 at t=0.8 is what the training loop's own generation test uses; it keeps
    this table comparable to `gen/*` in wandb.
  * pure ancestral (t=1.0, no truncation) is the full-entropy end. If a model's readout
    has collapsed to a point mass, this is identical to greedy -- that is exactly how
    the DiffusionBlocks readout was caught, and it is invisible if you only sample with
    a truncation on.

Every row carries rep4 and distinct3 from `morph.inference.gen_metrics`, and a REAL TEXT
row is scored by the same code as the anchor. Rank nothing against a model row whose
distinct3 is far from the real-text value; a degenerate row is not a better model.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from morph.inference.gen_metrics import generation_metrics, ngram_stats  # noqa: E402
from morph.inference.tul_generate import generate_tul  # noqa: E402
from morph.training.sft import build_model_with_quant  # noqa: E402
from morph.training.tul_setup import build_tul_runtime  # noqa: E402

PROMPTS = [
    "The theory of relativity states that",
    "Once upon a time in a distant land, there lived a",
    "In machine learning, the key insight is that",
    "The capital of France is",
    "def quicksort(arr):",
    "Yesterday the committee announced that",
    "Water boils at a temperature of",
    "She opened the letter and read",
]

# (label, temperature, top_k)
DECODES = [
    ("greedy", 0.0, 0),
    ("topk50_t0.8", 0.8, 50),
    ("sample_t1", 1.0, 0),
]


def load_cfg(name):
    from hydra import compose, initialize_config_dir
    name = os.path.basename(name).replace(".yaml", "")
    with initialize_config_dir(config_dir=os.path.abspath("morph/configs"),
                               version_base=None):
        return compose(config_name=name)


def load_ckpt(cfg, path, device, tul_cfg):
    # tul_cfg is REQUIRED here: without it MORPHTransformer builds no E_slot/E_mask/
    # W_prefix, the checkpoint's TUL tensors load as "unexpected" (silently dropped),
    # and the first slot_layout= forward raises. That is exactly how the first run of
    # this script died.
    model = build_model_with_quant(cfg, device, tul=tul_cfg)
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    mat = [k for k in missing
           if not any(s in k.lower() for s in ("rope", "cache", "freqs"))]
    if mat:
        # Same rule the order-parameter probe uses: sampling from partly-random weights
        # produces a table that looks fine and means nothing.
        print(f"LOAD_FAIL {path}: {len(mat)} material-missing, first 5 {mat[:5]}")
        sys.exit(1)
    print(f"    load ok: {len(unexpected)} unexpected, step {ck.get('step')}")
    return model.eval(), int(ck.get("step", -1))


def real_text_anchor(cfg, tokenizer, n_tokens, device):
    """rep4/distinct3 of held-out text, scored by the SAME code as the model rows."""
    from morph.training.data import create_dataloader
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 512, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    ids = next(loader)[0]
    rows = [ids[i, :n_tokens].tolist() for i in range(min(len(PROMPTS), ids.shape[0]))]
    r4 = [ngram_stats(r, 4)[0] for r in rows]
    d3 = [ngram_stats(r, 3)[1] for r in rows]
    return {"rep4": sum(r4) / len(r4), "distinct3": sum(d3) / len(d3)}


def run_one(model, tul_rt, tokenizer, seq_len, n_tokens, device, halt, seed):
    spec = tul_rt.data_cfg.spec_for(seq_len)
    rule = tul_rt.data_cfg.rule
    out = {}
    for label, temp, topk in DECODES:
        per, texts = [], []
        for pi, prompt in enumerate(PROMPTS):
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            new, builder = generate_tul(model, ids, rule, spec,
                                        max_new_tokens=n_tokens, temperature=temp,
                                        top_k=topk, seed=seed + pi, device=device,
                                        halt=halt)
            per.append(generation_metrics(new, builder, rule))
            texts.append(tokenizer.decode(ids + new, skip_special_tokens=True))
        agg = {k: float(sum(d[k] for d in per) / len(per)) for k in per[0]}
        out[label] = {"metrics": agg, "samples": texts}
        print(f"    {label:14s} rep4={agg['rep4']:.3f} distinct3={agg['distinct3']:.3f} "
              f"mean_span={agg['mean_span']:.1f} on_boundary={agg['boundary_frac']:.2f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH")
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--halt", action="store_true",
                    help="also score the gate-driven depth policy (arm TUL-halt)")
    ap.add_argument("--out", default="docs/experiments/results/tul_samples.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    device = torch.device("cuda")
    results, anchor = {}, None

    for spec_s in a.ckpt:
        parts = spec_s.split("=", 2)
        if len(parts) != 3:
            sys.exit(f"--ckpt needs LABEL=CONFIG=PATH, got {spec_s!r}")
        label, cfg_name, path = parts
        print(f"\n=== {label}  [{cfg_name}]  {path} ===", flush=True)
        cfg = load_cfg(cfg_name)
        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is None:
            print("    SKIP: this arm builds no TUL layout"); continue
        model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
        if anchor is None:
            anchor = real_text_anchor(cfg, tok, a.n_tokens, device)
            print(f"    REAL TEXT anchor rep4={anchor['rep4']:.3f} "
                  f"distinct3={anchor['distinct3']:.3f}")
        results[label] = {"step": step, "config": cfg_name, "path": path,
                          "fixed": run_one(model, tul_rt, tok, a.seq, a.n_tokens,
                                           device, False, a.seed)}
        # --halt is a REQUEST, not a promise: the halting policy is the gate choosing
        # each slot's depth, so an arm built without tul.gate has nothing to halt with
        # and generate_tul raises. Applying the flag globally is what killed the first
        # full run of this script, after the gate arm had already been scored.
        gated = getattr(tul_rt.model_cfg, "gate", None) is not None
        if a.halt and not gated:
            print("    -- halt policy SKIPPED: this arm has no gate --")
        if a.halt and gated:
            print("    -- halt policy --")
            results[label]["halt"] = run_one(model, tul_rt, tok, a.seq, a.n_tokens,
                                             device, True, a.seed)
        del model
        torch.cuda.empty_cache()

    results["_real_text"] = anchor
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
