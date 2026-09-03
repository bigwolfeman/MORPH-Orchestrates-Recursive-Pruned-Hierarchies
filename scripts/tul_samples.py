#!/usr/bin/env python
"""Generation samples and degeneration metrics for MORPH checkpoints, TUL or not.

    python scripts/tul_samples.py --ckpt plain=notul=<path>.pt --ckpt a2=tul_a2=<path>.pt

Why this is a standalone script and not more logging inside the training loop:
`gate_bakeoff.sh` launches each arm as its OWN `python -m morph.training.train`, so a
mid-campaign edit to `train.py` would silently make arms 2 and 3 differ from arm 1 by
more than the variable under test. This reads finished checkpoints instead and cannot
perturb a running arm.

THE BASELINE IS NOT OPTIONAL. Arm A0 sets `tul.activate_at: never` and therefore has no
TUL runtime; version 1 of this script printed "SKIP: this arm builds no TUL layout" and
produced a table of TUL against TUL. The question the table exists to answer — does the
slot loop repeat itself less than a plain model — was not in it. An arm with no TUL
runtime now decodes through `generate_plain`, which is the same eager recompute-per-step
loop as `generate_tul` and shares its sampling step (`morph.inference.sampling`), so a
difference between the arms comes from the weights and not from the decoder.

RANK ON SAMPLED DECODING. Greedy is reported last and is a DIAGNOSTIC: it says whether an
argmax loop exists, not how good the model is. Measured on this tree, greedy rep4 runs
0.5-0.9 for every arm including a diverged one, which is a statement about argmax basins.

LENGTH IS PART OF THE METRIC. rep_n is not comparable across lengths, and at 128 tokens
held-out OpenWebText scores rep4 = 0.015 with 54 % of rows at exactly 0.000 (256 rows) --
the reference is on the floor, and so was every sampled model row. At 512 tokens the same
text gives 0.037 with 1 % at 0. Default length is 512 for that reason; the real-text
anchor is scored at the SAME length, over `--anchor-rows` rows, with a standard deviation.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from morph.inference.gen_metrics import generation_metrics, ngram_stats  # noqa: E402
from morph.inference.plain_generate import generate_plain  # noqa: E402
from morph.inference.tul_generate import generate_tul  # noqa: E402
from morph.training.sft import build_model_with_quant  # noqa: E402
from morph.training.train import drop_retired_tul_keys  # noqa: E402
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
    "The main advantage of this approach is",
    "After the war ended, the government",
    "import numpy as np\n\ndef",
    "According to the report published last week,",
]

# (label, temperature, top_k). Sampled modes FIRST — they are what the arms are ranked
# on. `greedy` is last because it is a diagnostic, not a ranking.
DECODES = [
    ("topk50_t0.8", 0.8, 50),
    ("sample_t1", 1.0, 0),
    ("greedy", 0.0, 0),
]


def load_cfg(name):
    from hydra import compose, initialize_config_dir
    name = os.path.basename(name).replace(".yaml", "")
    with initialize_config_dir(config_dir=os.path.abspath("morph/configs"),
                               version_base=None):
        return compose(config_name=name)


def load_ckpt(cfg, path, device, tul_cfg):
    # tul_cfg is REQUIRED for a TUL arm: without it MORPHTransformer builds no E_slot/
    # E_mask/W_sent, the checkpoint's TUL tensors load as "unexpected" (silently
    # dropped), and the first slot_layout= forward raises. That is exactly how the first
    # run of this script died. For a non-TUL arm it is correctly None.
    model = build_model_with_quant(cfg, device, tul=tul_cfg)
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    drop_retired_tul_keys(state, model, path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    mat = [k for k in missing
           if not any(s in k.lower() for s in ("rope", "cache", "freqs"))]
    if mat:
        # Same rule the order-parameter probe uses: sampling from partly-random weights
        # produces a table that looks fine and means nothing.
        print(f"LOAD_FAIL {path}: {len(mat)} material-missing, first 5 {mat[:5]}")
        sys.exit(1)
    unexp = [k for k in unexpected if "rope" not in k.lower()]
    if unexp:
        # A TUL tensor landing here means the model was built without the parameter the
        # checkpoint trained. Silently dropping it is how you sample a half-loaded model.
        print(f"LOAD_FAIL {path}: {len(unexp)} unexpected, first 5 {unexp[:5]}")
        sys.exit(1)
    print(f"    load ok: step {ck.get('step')}, 0 unexpected, "
          f"{len(missing) - len(mat)} rope/cache-missing")
    return model.eval(), int(ck.get("step", -1))


def real_text_anchor(cfg, n_tokens, n_rows):
    """rep4/distinct3 of held-out text, scored by the SAME code as the model rows.

    Reported with a standard deviation and percentiles over `n_rows` rows. Version 1 of
    this script used ONE batch of 8 rows and reported rep4 = 0.003 with no spread; the
    same corpus over 256 rows at the same length gives 0.015 +- 0.039 with a median of
    0.000, so that figure was a low draw from a floored distribution and it anchored the
    whole table too low.
    """
    from morph.training.data import create_dataloader
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                               max(512, n_tokens), 8, split="validation",
                               skip_samples=0, bag_size=0, tul=None)
    rows = []
    while len(rows) < n_rows:
        ids = next(loader)[0]
        rows.extend(ids[i, :n_tokens].tolist() for i in range(ids.shape[0]))
    rows = rows[:n_rows]
    r4 = np.array([ngram_stats(r, 4, window=n_tokens)[0] for r in rows])
    d3 = np.array([ngram_stats(r, 3, window=n_tokens)[1] for r in rows])
    return {
        "n_rows": int(n_rows), "n_tokens": int(n_tokens),
        "rep4": float(r4.mean()), "rep4_std": float(r4.std()),
        "rep4_p10": float(np.percentile(r4, 10)), "rep4_p50": float(np.percentile(r4, 50)),
        "rep4_p90": float(np.percentile(r4, 90)),
        "rep4_frac_zero": float((r4 == 0).mean()),
        "distinct3": float(d3.mean()), "distinct3_std": float(d3.std()),
    }


def slot_invariance_check(model, tul_rt, tokenizer, seq_len, n_tokens, device, wide):
    """Widening the slot budget must not change a greedy sample. If it does, the layout
    padding is reaching the model and every number below is measuring the padding."""
    base = tul_rt.data_cfg.spec_for(seq_len)
    rule = tul_rt.data_cfg.rule
    import dataclasses
    wide_spec = dataclasses.replace(base, max_slots=wide)
    ids = tokenizer(PROMPTS[0], add_special_tokens=False)["input_ids"]
    src = emit_source_for(tul_rt)
    a, _ = generate_tul(model, ids, rule, base, max_new_tokens=n_tokens,
                        temperature=0.0, top_k=0, seed=0, device=device, emit_source=src)
    b, _ = generate_tul(model, ids, rule, wide_spec, max_new_tokens=n_tokens,
                        temperature=0.0, top_k=0, seed=0, device=device, emit_source=src)
    if a != b:
        n = sum(1 for x, y in zip(a, b) if x != y)
        sys.exit(f"SLOT_INVARIANCE_FAIL: max_slots {base.max_slots} vs {wide} changed "
                 f"{n}/{len(a)} greedy tokens; first divergence at "
                 f"{next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)}")
    print(f"    slot-invariance ok: max_slots {base.max_slots} vs {wide}, "
          f"{len(a)} greedy tokens identical")


def emit_source_for(tul_rt) -> str:
    """Read span-first tokens from the position training actually supervised.

    emit_weight == 0 arms (the whole GL line) never train the slot readout; sampling
    from it is the missing-space-after-period artifact (emit_space_probe.py). The
    boundary-token position is trained at weight 1 in every arm."""
    return "token" if float(tul_rt.model_cfg.emit_weight) == 0.0 else "slot"


def run_one(model, tul_rt, tokenizer, seq_len, n_tokens, device, seed,
            max_slots=0, reps=1, only=()):
    """One arm, every decode mode. Per-prompt values are KEPT: the A1-minus-A0 gap is a
    PAIRED difference over the same prompts, and a mean with no spread cannot say whether
    a 0.01 gap is real."""
    if tul_rt is not None:
        spec = tul_rt.data_cfg.spec_for(seq_len)
        rule = tul_rt.data_cfg.rule
        import dataclasses
        if max_slots:
            spec = dataclasses.replace(spec, max_slots=max_slots)
        else:
            # Degenerate decodes (greedy loops on '.'/newline) can cut a boundary on
            # nearly every token and overflow the trained budget of 64 — which crashed
            # the whole arm instead of recording the diagnostic. Each appended token
            # adds at most one slot, so prompt-budget + n_tokens can never overflow;
            # the slot-invariance check shows a wider budget is behavior-preserving.
            spec = dataclasses.replace(spec, max_slots=spec.max_slots + n_tokens)
    out = {}
    for label, temp, topk in DECODES:
        if only and label not in only:
            continue
        # Greedy is deterministic: repeating it would inflate n with duplicate rows and
        # shrink the standard error of a quantity that has no sampling variance at all.
        n_rep = 1 if temp <= 0.0 else max(1, reps)
        per, texts, t0 = [], [], time.time()
        for pi, prompt in enumerate(PROMPTS):
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for ri in range(n_rep):
                sd = seed + pi + 100003 * ri
                if tul_rt is None:
                    new = generate_plain(model, ids, max_new_tokens=n_tokens,
                                         temperature=temp, top_k=topk, seed=sd,
                                         device=device)
                    m = generation_metrics(new, window=n_tokens)
                else:
                    new, builder = generate_tul(model, ids, rule, spec,
                                                max_new_tokens=n_tokens, temperature=temp,
                                                top_k=topk, seed=sd, device=device,
                                                emit_source=emit_source_for(tul_rt))
                    m = generation_metrics(new, builder, rule, window=n_tokens)
                m["prompt_index"] = float(pi)
                m["draw"] = float(ri)
                per.append(m)
                if ri == 0:
                    texts.append(tokenizer.decode(ids + new, skip_special_tokens=True))
        # prompt_index / draw are bookkeeping for the paired analysis, not metrics: a
        # mean over them is meaningless and would sit in the table looking like one.
        keys = sorted(set().union(*(d.keys() for d in per)) - {"prompt_index", "draw"})
        agg = {k: float(np.mean([d[k] for d in per if k in d])) for k in keys}
        sd = {k + "_sd": float(np.std([d[k] for d in per if k in d])) for k in ("rep4", "distinct3")}
        agg.update(sd)
        out[label] = {"metrics": agg, "per_prompt": per, "samples": texts}
        print(f"    {label:14s} rep4={agg['rep4']:.4f}+-{agg['rep4_sd']:.4f} "
              f"distinct3={agg['distinct3']:.4f} n={len(per)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--n-tokens", type=int, default=512,
                    help="512 by default: at 128 the real-text reference is on the floor")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--max-slots", type=int, default=0,
                    help="widen the layout's fixed slot budget for generation. The "
                         "builder RAISES when a row needs more slots than the budget, "
                         "and a 512-token sample at the rule's min_span of 4 can need "
                         "128. 0 keeps the config value. Widening pads the layout with "
                         "more INVALID slots; slot_valid masks them, so the emitted "
                         "text must be unchanged -- asserted by --assert-slot-invariance.")
    ap.add_argument("--assert-slot-invariance", action="store_true",
                    help="greedy-decode one prompt at the config budget and at "
                         "--max-slots and require identical token ids before doing "
                         "anything else")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--anchor-rows", type=int, default=256)
    ap.add_argument("--samples-per-prompt", type=int, default=1,
                    help="independent draws per prompt, pooled into per_prompt. rep4 has "
                         "a per-sample paired sd of ~0.34 at top-k, so n=12 can only "
                         "resolve an effect of 0.27 and the A1-A0 gap is ~0.03. More "
                         "draws is the only cheap way to buy power; greedy is "
                         "deterministic so it is decoded ONCE whatever this is set to.")
    ap.add_argument("--decodes", default="",
                    help="comma-separated subset of the decode labels, e.g. "
                         "topk50_t0.8,sample_t1. Empty = all three.")
    ap.add_argument("--out", default="lab/experiments/results/tul_rep_ab.json")
    a = ap.parse_args()

    only = tuple(x for x in a.decodes.split(",") if x)
    bad = [x for x in only if x not in {d[0] for d in DECODES}]
    if bad:
        sys.exit(f"--decodes: unknown {bad}; valid are {[d[0] for d in DECODES]}")

    from transformers import AutoTokenizer
    device = torch.device("cuda")
    results, anchor = {}, None
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush(res, anch, args):
        """Write the table after every arm. Partial output beats a lost sweep."""
        payload = dict(res)
        payload["_real_text"] = anch
        payload["_meta"] = {"n_tokens": args.n_tokens, "seed": args.seed,
                            "max_slots": args.max_slots, "n_prompts": len(PROMPTS),
                            "decodes": list(only) or [d[0] for d in DECODES],
                            "samples_per_prompt": args.samples_per_prompt,
                            "arms_done": [k for k in res if not k.startswith("_")]}
        out.write_text(json.dumps(payload, indent=2))
        print(f"    [flush] {out} now holds {len(payload['_meta']['arms_done'])} arm(s)",
              flush=True)

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
            # NOT a skip. This is the baseline arm, decoded by the matched plain loop.
            print("    no TUL runtime -> plain (baseline) decoding", flush=True)
        model, step = load_ckpt(cfg, path, device, None if tul_rt is None else tul_rt.model_cfg)
        if a.assert_slot_invariance and tul_rt is not None and a.max_slots:
            slot_invariance_check(model, tul_rt, tok, a.seq, 48, device, a.max_slots)
        if anchor is None:
            anchor = real_text_anchor(cfg, a.n_tokens, a.anchor_rows)
            print(f"    REAL TEXT anchor n={anchor['n_rows']} L={anchor['n_tokens']} "
                  f"rep4={anchor['rep4']:.4f}+-{anchor['rep4_std']:.4f} "
                  f"(p50 {anchor['rep4_p50']:.4f}, {100 * anchor['rep4_frac_zero']:.0f}% zero) "
                  f"distinct3={anchor['distinct3']:.4f}", flush=True)
        results[label] = {"step": step, "config": cfg_name, "path": path,
                          "tul": tul_rt is not None,
                          "fixed": run_one(model, tul_rt, tok, a.seq, a.n_tokens,
                                           device, a.seed, a.max_slots,
                                           a.samples_per_prompt, only)}
        del model
        torch.cuda.empty_cache()
        flush(results, anchor, a)

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
