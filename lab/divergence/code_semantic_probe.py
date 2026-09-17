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

from morph.inference.plain_generate import generate_plain  # noqa: E402
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


def _distinct2(seqs: list[list[int]]) -> float:
    """Distinct-2 over all generated tokens: unique bigrams / bigrams (the diversity guard:
    a repetition loop reads near 0 whatever its cosine)."""
    bg, n = set(), 0
    for q in seqs:
        for i in range(len(q) - 1):
            bg.add((q[i], q[i + 1]))
            n += 1
    return len(bg) / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--kind", default="code", choices=("code", "slot", "plain"),
                    help="code: LCTUL / LCTUL-D (OWN@k, SHUF, ZERO, ORACLE); slot: a slot-loop "
                         "model such as the strict ruler (OWN only, its deterministic write); "
                         "plain: a model with no slots (OWN only, generate_plain)")
    ap.add_argument("--cuts_config", default="tul_code_d",
                    help="kind=plain: the TUL config whose boundary rule and packer define the "
                         "cuts, so every model is scored on the SAME 120 cuts")
    ap.add_argument("--n", type=int, default=120, help="cuts")
    ap.add_argument("--context_spans", default="3,4,5,6,7,8", help="cycled per cut")
    ap.add_argument("--extra_tokens", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8, help="sampler steps (rounds) for SHUF")
    ap.add_argument("--k_list", default="", help="sampler steps for OWN@k, e.g. 1,2,4,8 "
                                                  "(default: --steps only)")
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
    k_list = [int(x) for x in a.k_list.split(",")] if a.k_list else [int(a.steps)]
    k_shuf = int(a.steps)
    if k_shuf not in k_list:
        k_list.append(k_shuf)
    k_list = sorted(set(k_list))

    from transformers import AutoTokenizer
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path = a.ckpt.split("=", 2)
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    is_code = tul_rt is not None and bool(getattr(tul_rt.model_cfg, "code", False))
    if a.kind == "code":
        assert is_code, "kind=code needs a TUL-Code checkpoint"
    elif a.kind == "slot":
        assert tul_rt is not None and not is_code, "kind=slot needs a slot-loop TUL checkpoint"
    else:
        assert tul_rt is None, "kind=plain needs a checkpoint with no TUL runtime"
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg if tul_rt is not None else None)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    if a.kind == "plain":
        cuts_cfg = build_cfg(a.cuts_config, ["model.use_kernels=false"])
        cuts_rt = build_tul_runtime(cuts_cfg)
        assert cuts_rt is not None
    else:
        cuts_cfg, cuts_rt = cfg, tul_rt
    rule, spec = cuts_rt.data_cfg.rule, cuts_rt.data_cfg.spec_for(cuts_cfg.data.seq_len)
    emit = "token"
    if tul_rt is not None:
        emit = "token" if tul_rt.model_cfg.emit_weight == 0.0 else "slot"
    M, C = spec.prefix_k, model.cfg.d_model
    s_scale = 1.0
    if a.kind == "code":
        s_scale = (float(model._code_truth_scale) if a.sample_scale == "truth"
                   else float(a.sample_scale))
        print(f"sample cells handed to the coda at RMS {s_scale:.4f} (truth statistic "
              f"{float(model._code_truth_scale):.4f}); OWN at k = {k_list}, SHUF at k = {k_shuf}",
              flush=True)

    loader = create_dataloader(cuts_cfg.data.tokenizer, cuts_cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, (a.n + 8) * (spec.l_total + 1))
    rows = pack_rows(stream, cuts_rt, cuts_cfg, 1, False)

    # ── pass 1: the cuts, their true spans; on a code model E's oracle code and OWN@k ──
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
        cut = {"b": b, "prefix": prefix, "true": true, "oracle": None, "own": {}}
        if a.kind == "code":
            lay = layout.to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                res = model.tul_forward_ablated(inp.to(device), None, lay, plan_mode="normal",
                                                code_mode="encoder")
                cut["oracle"] = res["code_cells"][0, b].float().clone()
                # the OWN sample at every k: the generation regime on the prefix row (open slot = b)
                builder = TulRowBuilder(rule=rule, spec=spec)
                for t in prefix:
                    builder.append(int(t))
                pids, play = builder.tensors(device)
                assert builder.n_slots - 1 == b, f"open slot {builder.n_slots - 1} != {b}"
                for k in k_list:
                    given = torch.zeros(1, spec.max_slots, M, C, device=device, dtype=torch.float32)
                    gmask = torch.zeros(1, spec.max_slots, dtype=torch.bool, device=device)
                    r2 = model(pids, slot_layout=play, code_mode="generate", code_given=given,
                               code_given_mask=gmask, code_steps=k,
                               code_seed=(a.seed * 1_000_003 + len(cuts)) % (2 ** 31))
                    cut["own"][k] = r2["code_cells"][0, b].float().clone()
        cuts.append(cut)
    N = len(cuts)
    print(f"{label}: step {step}, kind {a.kind}, {N} cuts", flush=True)

    # ── pass 2: the model writes span b+1 under each condition (greedy) ───────────────
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

    def gen(prefix, n_new) -> list[int]:
        if a.kind == "plain":
            return generate_plain(model, prefix, max_new_tokens=n_new, temperature=0.0,
                                  top_k=0, seed=0, device=device)
        new, _ = generate_tul(model, prefix, rule, spec, max_new_tokens=n_new, temperature=0.0,
                              top_k=0, seed=0, device=device, emit_source=emit)
        return list(new)

    if a.kind == "code":
        conds = tuple(f"OWN@{k}" for k in k_list) + ("SHUF", "ZERO", "ORACLE")
    else:
        conds = ("OWN",)
    toks_out = {c: [] for c in conds}
    trues, ctxs = [], []
    if a.kind == "code":
        model.forward = inject_forward
    try:
        for i, cut in enumerate(cuts):
            j = (i + 1) % N                                   # a different cut's sample
            n_new = len(cut["true"]) + a.extra_tokens
            trues.append(tok.decode(cut["true"], skip_special_tokens=True))
            ctxs.append(tok.decode(cut["prefix"][-80:], skip_special_tokens=True))
            if a.kind == "code":
                codes = {f"OWN@{k}": cut["own"][k] * s_scale for k in k_list}
                codes["SHUF"] = cuts[j]["own"][k_shuf] * s_scale
                codes["ZERO"] = torch.zeros_like(cut["oracle"])
                codes["ORACLE"] = cut["oracle"]
            for c in conds:
                if a.kind == "code":
                    state["code"], state["b"] = codes[c], cut["b"]
                toks_out[c].append(gen(cut["prefix"], n_new))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{N} cuts written", flush=True)
    finally:
        model.forward = orig_forward
    texts = {c: [tok.decode(q, skip_special_tokens=True) for q in toks_out[c]] for c in conds}

    # ── scoring ──────────────────────────────────────────────────────────────────────
    emb = Embedder(device)
    e_true, e_ctx = emb(trues), emb(ctxs)
    out = {"label": label, "step": step, "kind": a.kind, "cuts": N, "steps": k_shuf,
           "k_list": k_list, "sample_scale": s_scale, "per_condition": {}, "paired": {},
           "per_cut": {}}
    cos_true, cos_ctx, ov_true, ov_ctx = {}, {}, {}, {}
    for c in conds:
        e = emb(texts[c])
        cos_true[c] = (e * e_true).sum(-1).numpy()
        cos_ctx[c] = (e * e_ctx).sum(-1).numpy()
        ov_true[c] = np.array([len(_content(g) & _content(t)) / max(len(_content(t)), 1)
                               for g, t in zip(texts[c], trues)])
        ov_ctx[c] = np.array([len(_content(g) & _content(x)) / max(len(_content(g)), 1)
                              for g, x in zip(texts[c], ctxs)])
        d2 = _distinct2(toks_out[c])
        out["per_condition"][c] = {"cos_true": float(cos_true[c].mean()),
                                   "cos_ctx": float(cos_ctx[c].mean()),
                                   "overlap_true": float(ov_true[c].mean()),
                                   "overlap_ctx": float(ov_ctx[c].mean()),
                                   "distinct2": d2}
        out["per_cut"][c] = {"cos_true": cos_true[c].tolist(), "overlap_true": ov_true[c].tolist()}
        print(f"  {c:8s} cos(true) {cos_true[c].mean():.4f}  cos(ctx) {cos_ctx[c].mean():.4f}  "
              f"overlap(true) {ov_true[c].mean():.4f}  overlap(ctx) {ov_ctx[c].mean():.4f}  "
              f"distinct2 {d2:.3f}", flush=True)
    rng = np.random.default_rng(0)

    def paired(x, y):
        d = x - y
        bs = [d[rng.integers(0, N, N)].mean() for _ in range(2000)]
        return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    pairs = {}
    if a.kind == "code":
        own_hi, own_lo = f"OWN@{k_list[-1]}", f"OWN@{k_list[0]}"
        for k in k_list:
            pairs[f"OWN@{k}-SHUF"] = (f"OWN@{k}", "SHUF")
        pairs.update({f"{own_hi}-{own_lo}": (own_hi, own_lo), f"{own_hi}-ZERO": (own_hi, "ZERO"),
                      "ORACLE-SHUF": ("ORACLE", "SHUF"), f"ORACLE-{own_hi}": ("ORACLE", own_hi)})
    for name, (x, y) in pairs.items():
        out["paired"][name] = {"cos_true": paired(cos_true[x], cos_true[y]),
                               "cos_ctx": paired(cos_ctx[x], cos_ctx[y]),
                               "overlap_true": paired(ov_true[x], ov_true[y]),
                               "overlap_ctx": paired(ov_ctx[x], ov_ctx[y])}
        p = out["paired"][name]
        print(f"  {name:14s} cos(true) {p['cos_true'][0]:+.4f} [{p['cos_true'][1]:+.4f}, {p['cos_true'][2]:+.4f}]"
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
                f.write(f"{c:8s} {texts[c][i]!r}\n")
            f.write("\n")
    print(f"wrote {a.out} and {txt}", flush=True)


if __name__ == "__main__":
    main()
