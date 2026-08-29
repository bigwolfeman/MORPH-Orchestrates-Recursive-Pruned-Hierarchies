"""Where does the missing-space-after-period artifact come from, per arm?

The generator samples each span's FIRST token from the slot's emit position
(tul_generate.py line 16), but GL arms train that position at emit_weight=0
(tul_gl1.yaml). This probe teacher-forces packed val rows through each arm's
step_4500 checkpoint and measures, at every real slot:

  - space_mass_emit:  P(next token is space/newline-prefixed) read at the slot's
                      EMIT position (what the generator actually samples from)
  - space_mass_tok:   the same mass read at the boundary TOKEN position
                      (the position training actually supervised, weight 1)
  - top1_space_*:     fraction of slots whose argmax token is space-prefixed

Prediction (vlt tul-span-jepa/123): space_mass_emit ranks inversely with the
measured missing-space rate (ctrl 44.6% > gl1b 30.9% > gl1c 12.7% > gl1 7.6%),
while space_mass_tok is high and flat across arms.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402  (handles the _orig_mod. compile prefix)


def space_vocab_mask(tok, vocab: int, device) -> torch.Tensor:
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    m = torch.zeros(vocab, dtype=torch.bool)
    for i, p in enumerate(pieces):
        if p is not None and p.startswith(("Ġ", "Ċ", "ĉ")):
            m[i] = True
    return m.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH, e.g. gl1b=tul_gl1b=checkpoints/morph/gl1b-s1/step_4500.pt")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--batch", type=int, default=6)
    a = ap.parse_args()
    device = "cuda"

    from transformers import AutoTokenizer
    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader

    print(f"{'arm':10s} {'slots':>5s}  {'space_mass_emit':>15s} {'space_mass_tok':>14s}  "
          f"{'top1_space_emit':>15s} {'top1_space_tok':>14s}")
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        cfg = build_cfg(config, ["model.use_kernels=false"])
        from morph.training.tul_setup import build_tul_runtime
        tul_rt = build_tul_runtime(cfg)
        model, _step = load_ckpt(cfg, f"{ROOT}/{path}" if not path.startswith("/") else path,
                                 device, tul_rt.model_cfg if tul_rt else None)

        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        rule = tul_rt.data_cfg.rule
        K = spec.prefix_k
        smask = space_vocab_mask(tok, cfg.model.vocab_size, device)

        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        buf: list[int] = []
        need = a.batch * (spec.l_total + 1)
        me, mt, t1e, t1t, n_slots = [], [], [], [], 0
        rows_done = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            while rows_done < a.rows:
                while len(buf) < need:
                    ids = next(loader)[0]
                    buf.extend(ids.reshape(-1).tolist())
                inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
                layout = layout.to(device)
                res = model(inp.to(device), slot_layout=layout)
                logits = res["logits"] if isinstance(res, dict) else res
                probs = torch.softmax(logits.float(), dim=-1)
                for b in range(a.batch):
                    for s in range(spec.max_slots):
                        if not bool(layout.slot_valid[b, s]):
                            continue
                        first = int(layout.slot_index[b, s])
                        emit_pos, tok_pos = first + K - 1, first - 1
                        if tok_pos < 0:
                            continue
                        pe, pt = probs[b, emit_pos], probs[b, tok_pos]
                        me.append(float(pe[smask].sum())); mt.append(float(pt[smask].sum()))
                        t1e.append(bool(smask[int(pe.argmax())])); t1t.append(bool(smask[int(pt.argmax())]))
                        n_slots += 1
                rows_done += a.batch
        print(f"{label:10s} {n_slots:5d}  {np.mean(me):15.4f} {np.mean(mt):14.4f}  "
              f"{np.mean(t1e):15.4f} {np.mean(t1t):14.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
