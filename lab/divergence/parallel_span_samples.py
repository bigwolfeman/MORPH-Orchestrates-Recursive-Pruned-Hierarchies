"""What does the parallel span head WRITE? Greedy whole-span decodes, one per reader.

The parallel span head (``morph/model/tul_spandec_parallel.py``, LXTUL-E) predicts every
token of the next span at once from a slot's exit state, with no token input. Its CE says
how well it scores; this prints what it produces. For a few validation rows it shows the
context tail, the TRUE next span, and the head's greedy decode of that span from each
reader:

  * Stage 0 heads (``tul.spandec_parallel_k`` K > 1 on a frozen ruler): one decode per
    code, ``z + rms(z) u_k``; K = 1 is the head alone.
  * Stage 1 arms (``tul.code_enum_k`` K > 1): one decode per loop ROLLOUT, the code
    already inside the exit state; e1 has one reader.

Decodes run to the true span's length (the head is trained on that many positions), and a
reader's log-likelihood of the true span is printed beside it, so a code's decode and its
fit can be read together. Greedy is a DIAGNOSTIC of the head's argmax, not a sampler.

Usage:
  python lab/divergence/parallel_span_samples.py --ckpt LABEL=CONFIG=PATH [--ckpt ...]
      --rows 2 --spans 4 [--device cpu --ovr model.tg_scoped_kernels=false
      --ovr model.hc_use_kernel=false] --out samples.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lxtul_e_stage1_score import _Capture, _one  # noqa: E402  (ONE home of the seam capture)


@torch.no_grad()
def span_decodes(m, inp, labels, layout, device: str) -> list[dict]:
    """Per supervised slot of row 0.. : the true next span's ids and each reader's greedy
    decode plus its log-likelihood of the true span."""
    head = m.tul_spandec_par
    if head is None:
        raise SystemExit("this model has no parallel span head (tul.spandec_parallel)")
    K_loop = max(int(m._code_enum_k), 1)
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    with ac, _Capture(m, "_tul_spandec_par_loss") as hc:
        m(inp.to(device), labels=labels.to(device), slot_layout=layout)
    h_slots = _one(hc)[0][0]                                           # [K*B, S, (n,) C]
    B = inp.shape[0]
    ids, valid = head.targets(inp.to(device), layout)                   # [B, S, J]
    w = m.embed.lm_weight().detach().float()
    with ac:
        z = m._readout(h_slots)                                         # [K*B, S, C]
    if K_loop > 1:
        zr = z.view(K_loop, B, *z.shape[1:])
    else:
        Bz, S, C = z.shape
        zr = head.with_codes(z.reshape(Bz * S, C)).view(-1, Bz, S, C)
    R = zr.shape[0]
    J = int(ids.shape[-1])
    out = []
    for b in range(B):
        for s in range(ids.shape[1]):
            if not bool(valid[b, s].any()):
                continue
            n = int(valid[b, s].sum())
            with ac:
                st = head.states(zr[:, b, s].unsqueeze(1), J)[:, 0]     # [R, J, C]
            logits = st.float() @ w.t()                                 # [R, J, V]
            logits[..., m.cfg.tul.slot_id] = float("-inf")
            lp = torch.log_softmax(logits, dim=-1)
            true = ids[b, s, :n]
            ll = lp[:, torch.arange(n), true].sum(-1)                   # [R]
            out.append({"row": b, "slot": s, "true": true.tolist(),
                        "decodes": lp[:, :n].argmax(-1).tolist(), "ll": ll.tolist(),
                        "readers": R})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--spans", type=int, default=4, help="spans printed per row")
    ap.add_argument("--skip-spans", type=int, default=6,
                    help="skip a row's first spans (short contexts)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ovr", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from _build import ROOT, build_cfg
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from transformers import AutoTokenizer

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    lines: list[str] = []
    for spec in a.ckpt:
        label, config, path = spec.split("=", 2)
        cfg = build_cfg(config, ["model.use_kernels=false", *a.ovr])
        tul_rt = build_tul_runtime(cfg)
        model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
        model.eval()
        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        inp, labels, layout, _idx = pack_rows(stream, tul_rt, cfg, a.rows, False)[0]
        layout = layout.to(a.device)
        res = span_decodes(model, inp, labels, layout, a.device)
        lines.append(f"\n==== {label} (step {step}, {res[0]['readers']} reader(s)) ====")
        for b in range(inp.shape[0]):
            row = [r for r in res if r["row"] == b]
            by_slot = {r["slot"]: r for r in row}
            for r in row[a.skip_spans:a.skip_spans + a.spans]:
                prev = by_slot.get(r["slot"] - 1)
                lines.append(f"-- row {b} slot {r['slot']} --")
                if prev is not None:
                    lines.append(f"  PREV : {tok.decode(prev['true'])!r}")
                lines.append(f"  TRUE : {tok.decode(r['true'])!r}")
                for k, (d, ll) in enumerate(zip(r["decodes"], r["ll"])):
                    lines.append(f"  r{k} ll={ll:8.2f}: {tok.decode(d)!r}")
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
    text = "\n".join(lines)
    with open(a.out, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
