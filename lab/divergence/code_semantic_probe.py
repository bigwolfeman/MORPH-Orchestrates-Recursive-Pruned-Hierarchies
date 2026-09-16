"""Does the thinker's sample carry the MEANING of the context, whatever its L2 distance?

The residual probe (code_subspace_probe.py) reads the sample's distance from the true
code, which is blind to a sample that carries the right topic in other words. This probe
scores what the coda SAYS from the sample. For N cuts after span b, the open slot b is
handed four codes and the coda writes span b+1 greedily under each, everything else
(the past cells, the token path) held fixed:

  OWN     the thinker's sample for THIS context (the generation regime);
  SHUF    the thinker's sample for a DIFFERENT cut (same statistic, wrong context);
  ZERO    an all-zero code (no information);
  ORACLE  E's code of the true span b+1 (the ceiling).

Each generated span is scored against the true span and the context tail by sentence
embedding cosine (sentence-transformers/all-MiniLM-L6-v2 through transformers, mean
pooling) and by content-word overlap. The number that answers the question is the paired
difference OWN − SHUF over cuts with a bootstrap CI: a sample that carries the context
lands nearer its own true span than a sample made for another context does.

    python lab/divergence/code_semantic_probe.py \
        --ckpt tul-code=tul_code=/path/step_20000.pt --n 120 --out .../semantic.json

Read-only. The model's forward is wrapped (never edited) to hand the open slot its code.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import torch

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.inference.tul_generate import generate_tul  # noqa: E402
from morph.inference.tul_generate import TulRowBuilder  # noqa: E402

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those he she they them his her their we you i our your me my
not no so than then there here has have had do does did will would can could should may
might also just about into over after before more most such very said says say one two
""".split())


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP}


class Embedder:
    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer
        name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        b = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        b = {k: v.to(self.device) for k, v in b.items()}
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return torch.nn.functional.normalize(e, dim=-1).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--n", type=int, default=120, help="cuts")
    ap.add_argument("--context_spans", default="3,4,5,6,7,8", help="cycled per cut")
    ap.add_argument("--extra_tokens", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8, help="sampler Euler steps for OWN/SHUF")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_scale", default="truth",
                    help="scale of the OWN/SHUF cells handed to the coda: 'truth' = the model's "
                         "truth-cell statistic (sqrt(1+noise^2) on a non-renorm arm, 1 on a renorm "
                         "arm), or a number (1.0 = the generator's current behaviour)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    ctx_list = [int(x) for x in a.context_spans.split(",")]

    from transformers import AutoTokenizer
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path = a.ckpt.split("=", 2)
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(getattr(tul_rt.model_cfg, "code", False)), \
        "a TUL-Code checkpoint is required"
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    rule, spec = tul_rt.data_cfg.rule, tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    emit = "token" if tul_rt.model_cfg.emit_weight == 0.0 else "slot"
    M, C = spec.prefix_k, model.cfg.d_model
    s_scale = float(model._code_truth_scale) if a.sample_scale == "truth" else float(a.sample_scale)
    print(f"sample cells handed to the coda at RMS {s_scale:.4f} (truth statistic "
          f"{float(model._code_truth_scale):.4f})", flush=True)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, (a.n + 8) * (spec.l_total + 1))
    rows = pack_rows(stream, tul_rt, cfg, 1, False)

    # ── pass 1: the cuts, their true spans, E's oracle code and the thinker's OWN sample ──
    cuts = []
    for i, (inp, labels, layout, _) in enumerate(rows):
        if len(cuts) >= a.n:
            break
        b = ctx_list[len(cuts) % len(ctx_list)] - 1
        if not (bool(layout.slot_valid[0, b]) and bool(layout.slot_valid[0, b + 1])):
            continue
        tokpos = ~layout.slot_mask[0]
        ids = inp[0][tokpos].tolist()
        bags = layout.bag_id[0][tokpos].tolist()
        prefix = [t for t, g in zip(ids, bags) if g <= b]
        true = [t for t, g in zip(ids, bags) if g == b + 1]
        if not prefix or len(true) < 3:
            continue
        lay = layout.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            res = model.tul_forward_ablated(inp.to(device), None, lay, plan_mode="normal",
                                            code_mode="encoder")
            oracle = res["code_cells"][0, b].float().clone()
            # the OWN sample: the generation regime on the prefix row (open slot = b)
            builder = TulRowBuilder(rule=rule, spec=spec)
            for t in prefix:
                builder.append(int(t))
            pids, play = builder.tensors(device)
            assert builder.n_slots - 1 == b, f"open slot {builder.n_slots - 1} != {b}"
            given = torch.zeros(1, spec.max_slots, M, C, device=device, dtype=torch.float32)
            gmask = torch.zeros(1, spec.max_slots, dtype=torch.bool, device=device)
            r2 = model(pids, slot_layout=play, code_mode="generate", code_given=given,
                       code_given_mask=gmask, code_steps=a.steps,
                       code_seed=(a.seed * 1_000_003 + len(cuts)) % (2 ** 31))
            own = r2["code_cells"][0, b].float().clone()
        cuts.append({"b": b, "prefix": prefix, "true": true, "oracle": oracle, "own": own})
    N = len(cuts)
    print(f"{label}: step {step}, {N} cuts, sampler steps {a.steps}", flush=True)

    # ── pass 2: the coda writes span b+1 under each code (greedy) ────────────────────
    orig_forward = model.forward
    state = {"code": None, "b": -1}

    def inject_forward(*args, **kw):
        g, gm = kw.get("code_given"), kw.get("code_given_mask")
        if g is not None and state["code"] is not None:
            s_open = int(kw["slot_layout"].slot_valid[0].sum()) - 1
            if s_open == state["b"]:  # only the first generated span gets the code
                g[0, s_open] = state["code"].to(g.dtype)
                gm[0, s_open] = True
        return orig_forward(*args, **kw)

    def gen(prefix, n_new):
        new, _ = generate_tul(model, prefix, rule, spec, max_new_tokens=n_new, temperature=0.0,
                              top_k=0, seed=0, device=device, emit_source=emit)
        return tok.decode(new, skip_special_tokens=True)

    conds = ("OWN", "SHUF", "ZERO", "ORACLE")
    texts = {c: [] for c in conds}
    trues, ctxs = [], []
    model.forward = inject_forward
    try:
        for i, cut in enumerate(cuts):
            j = (i + 1) % N                                   # a different cut's sample
            codes = {"OWN": cut["own"] * s_scale, "SHUF": cuts[j]["own"] * s_scale,
                     "ZERO": torch.zeros_like(cut["own"]), "ORACLE": cut["oracle"]}
            n_new = len(cut["true"]) + a.extra_tokens
            trues.append(tok.decode(cut["true"], skip_special_tokens=True))
            ctxs.append(tok.decode(cut["prefix"][-80:], skip_special_tokens=True))
            for c in conds:
                state["code"], state["b"] = codes[c], cut["b"]
                texts[c].append(gen(cut["prefix"], n_new))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{N} cuts written", flush=True)
    finally:
        model.forward = orig_forward

    # ── scoring ──────────────────────────────────────────────────────────────────────
    emb = Embedder(device)
    e_true, e_ctx = emb(trues), emb(ctxs)
    out = {"label": label, "step": step, "cuts": N, "steps": a.steps, "sample_scale": s_scale,
           "per_condition": {}, "paired": {}}
    cos_true, cos_ctx, ov_true, ov_ctx = {}, {}, {}, {}
    for c in conds:
        e = emb(texts[c])
        cos_true[c] = (e * e_true).sum(-1).numpy()
        cos_ctx[c] = (e * e_ctx).sum(-1).numpy()
        ov_true[c] = np.array([len(_content(g) & _content(t)) / max(len(_content(t)), 1)
                               for g, t in zip(texts[c], trues)])
        ov_ctx[c] = np.array([len(_content(g) & _content(x)) / max(len(_content(g)), 1)
                              for g, x in zip(texts[c], ctxs)])
        out["per_condition"][c] = {"cos_true": float(cos_true[c].mean()),
                                   "cos_ctx": float(cos_ctx[c].mean()),
                                   "overlap_true": float(ov_true[c].mean()),
                                   "overlap_ctx": float(ov_ctx[c].mean())}
        print(f"  {c:6s} cos(true) {cos_true[c].mean():.4f}  cos(ctx) {cos_ctx[c].mean():.4f}  "
              f"overlap(true) {ov_true[c].mean():.4f}  overlap(ctx) {ov_ctx[c].mean():.4f}", flush=True)
    rng = np.random.default_rng(0)

    def paired(x, y):
        d = x - y
        bs = [d[rng.integers(0, N, N)].mean() for _ in range(2000)]
        return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    for name, (x, y) in {"OWN-SHUF": ("OWN", "SHUF"), "OWN-ZERO": ("OWN", "ZERO"),
                         "ORACLE-SHUF": ("ORACLE", "SHUF"), "ORACLE-OWN": ("ORACLE", "OWN")}.items():
        out["paired"][name] = {"cos_true": paired(cos_true[x], cos_true[y]),
                               "cos_ctx": paired(cos_ctx[x], cos_ctx[y]),
                               "overlap_true": paired(ov_true[x], ov_true[y]),
                               "overlap_ctx": paired(ov_ctx[x], ov_ctx[y])}
        p = out["paired"][name]
        print(f"  {name:12s} cos(true) {p['cos_true'][0]:+.4f} [{p['cos_true'][1]:+.4f}, {p['cos_true'][2]:+.4f}]"
              f"  cos(ctx) {p['cos_ctx'][0]:+.4f} [{p['cos_ctx'][1]:+.4f}, {p['cos_ctx'][2]:+.4f}]"
              f"  overlap(true) {p['overlap_true'][0]:+.4f} [{p['overlap_true'][1]:+.4f}, {p['overlap_true'][2]:+.4f}]",
              flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    txt = a.out.rsplit(".", 1)[0] + ".txt"
    with open(txt, "w") as f:
        for i in range(N):
            f.write(f"=== cut {i + 1} (after span {cuts[i]['b'] + 1}, {len(cuts[i]['true'])} true tokens)\n")
            f.write(f"CONTEXT ...{ctxs[i][-160:]!r}\nTRUE    {trues[i]!r}\n")
            for c in conds:
                f.write(f"{c:7s} {texts[c][i]!r}\n")
            f.write("\n")
    print(f"wrote {a.out} and {txt}", flush=True)


if __name__ == "__main__":
    main()
