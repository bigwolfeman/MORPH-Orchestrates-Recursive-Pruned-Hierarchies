"""What does a TUL-Code model SAY when it guesses the next span?

For validation rows, cut after span b, and show side by side:
  TRUE      the corpus's span b+1;
  ORACLE    the coda writing span b+1 from the ENCODER's code of the true span (greedy);
  GREEDY    the coda writing from the thinker's sampled code, greedy tokens;
  SAMPLE s  the same with token sampling (temperature 0.8, top-k 50), one line per seed.

The oracle line is the ceiling (the answer is in the cells); the others are honest
generation (`code_mode="generate"`: one code sample per open span, re-encoded past).

    python lab/divergence/code_span_samples.py \
        --ckpt tul-code=tul_code=/path/step_5000.pt --n 10 --out .../samples.txt
"""
from __future__ import annotations

import argparse
import sys

import torch

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.inference.tul_generate import generate_tul  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--n", type=int, default=10, help="examples")
    ap.add_argument("--context_spans", type=int, default=6, help="spans of context before the cut")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--extra_tokens", type=int, default=6, help="generate len(true)+this tokens")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    seeds = [int(s) for s in a.seeds.split(",") if s]

    from transformers import AutoTokenizer
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path = a.ckpt.split("=", 2)
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(cfg.tul.code), "a TUL-Code checkpoint is required"
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    rule, spec = tul_rt.data_cfg.rule, tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    emit = "token" if tul_rt.model_cfg.emit_weight == 0.0 else "slot"

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, (a.n + 4) * (spec.l_total + 1))
    rows = pack_rows(stream, tul_rt, cfg, 1, False)

    orig_forward = model.forward
    lines = [f"{label} step {step}: {a.n} cuts after span {a.context_spans}, emit_source={emit}", ""]

    def gen(prefix, n_new, temperature, seed):
        new, _ = generate_tul(model, prefix, rule, spec, max_new_tokens=n_new,
                              temperature=temperature, top_k=(50 if temperature > 0 else 0),
                              seed=seed, device=device, emit_source=emit)
        return tok.decode(new, skip_special_tokens=True).replace("\n", "\\n")

    done = 0
    for inp, labels, layout, _ in rows:
        if done >= a.n:
            break
        b = a.context_spans - 1
        lay = layout.to(device)
        if not (bool(layout.slot_valid[0, b]) and bool(layout.slot_valid[0, b + 1])):
            continue
        tokpos = ~layout.slot_mask[0]
        ids = inp[0][tokpos].tolist()
        bags = layout.bag_id[0][tokpos].tolist()
        prefix = [t for t, g in zip(ids, bags) if g <= b]
        true = [t for t, g in zip(ids, bags) if g == b + 1]
        if not prefix or not true:
            continue
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device == "cuda"):
            res = model.tul_forward_ablated(inp.to(device), None, lay, plan_mode="normal",
                                            code_mode="encoder")
        oracle = res["code_cells"][0, b].float().clone()  # code of the TRUE span b+1
        n_new = len(true) + a.extra_tokens

        def oracle_forward(*args, **kw):
            g, gm = kw.get("code_given"), kw.get("code_given_mask")
            if g is not None:
                s_open = int(kw["slot_layout"].slot_valid[0].sum()) - 1
                if s_open == b:  # only the first generated span gets the answer
                    g[0, b] = oracle.to(g.dtype)
                    gm[0, b] = True
            return orig_forward(*args, **kw)

        ctx = tok.decode(prefix[-60:], skip_special_tokens=True).replace("\n", "\\n")
        lines.append(f"=== example {done + 1}  (context tail, {len(true)} true tokens)")
        lines.append(f"CONTEXT ...{ctx}")
        lines.append(f"TRUE     {tok.decode(true, skip_special_tokens=True)!r}")
        model.forward = oracle_forward
        lines.append(f"ORACLE   {gen(prefix, n_new, 0.0, 0)!r}")
        model.forward = orig_forward
        lines.append(f"GREEDY   {gen(prefix, n_new, 0.0, 0)!r}")
        for s in seeds:
            lines.append(f"SAMPLE{s}  {gen(prefix, n_new, 0.8, s)!r}")
        lines.append("")
        done += 1
        print("\n".join(lines[-7:]), flush=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {a.out} ({done} examples)")


if __name__ == "__main__":
    main()
