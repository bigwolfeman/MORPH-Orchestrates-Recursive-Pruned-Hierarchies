"""Generation samples and repetition metrics for testbed checkpoints (2026-10-06).

Why: a copy/pointer head scores well on CE by copying earlier bigrams; in free generation the
same head can feed its own output back and loop, which CE cannot see (See et al. 2017 added a
coverage loss for exactly this). This script decodes from every arm with the SAME prompts and
reports how repetitive the continuations are against the real continuation of each prompt.

    # 1. prompts (CPU, reads MORPH's validation rows; the first --rows rows of the 480)
    python -m lxtul.generate prompts --out prompts.pt --rows 32
    # 2. decode + score (GPU)
    python -m lxtul.generate run --prompts prompts.pt --runs-root /home/wolfe/parcae-runs \
        --arms plain-5k,plain-pointer-norm-5k,lxtul-5k,lxtul-pointer-norm-5k \
        --scorer plain-5k --out gen.json --examples gen.md

Decoding: recompute per step (no KV cache). An LXTUL arm re-packs the growing token sequence
with MORPH's boundary rule and packer every step; the buffer is padded with a non-boundary
filler token so the fixed-shape row fills, and the filler sits after the last real token, so
causal attention keeps it invisible to every real position. The next token is read from the
LAST real token's position (its own slot lies after it). A pointer arm reads the full mixed
distribution there (`PointerHead.mixed_at`, the training rule at every vocabulary entry).

Metrics, per continuation, mean over prompts with a bootstrap 95 % CI:
  seq_rep_4   1 - unique 4-grams / 4-grams within the continuation (Welleck et al. 2019)
  rep_l       fraction of tokens already seen in the previous 128 continuation tokens
  ctx_copy_4  fraction of continuation 4-grams that already occur EARLIER in prompt +
              continuation (the pointer's own risk: verbatim copying from context)
  max_copy    longest continuation run that occurs verbatim earlier in prompt + continuation
  distinct_4  unique 4-grams over all continuations of the arm
  gen_ppl     exp(mean NLL of the continuation) under the --scorer model (a low gen_ppl with a
              high seq_rep_4 is a loop, not quality: read the two together)
The REAL row scores the true continuation the same way.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lxtul.data import MORPH_ROOT, morph_cfg, tul_runtime
from lxtul.gap_probe import load_run

sys.path.insert(0, str(MORPH_ROOT / "lab" / "divergence"))
from gen_diversity import bootstrap_ci, distinct_n, rep_l, seq_rep_n  # noqa: E402

PROMPT_LEN, GEN_LEN = 128, 128


def ctx_copy(prompt: list[int], cont: list[int], n: int = 4) -> tuple[float, int]:
    """(fraction of continuation n-grams that start somewhere earlier in prompt + continuation,
    longest continuation run that occurs verbatim earlier in that context)."""
    seq, P = prompt + cont, len(prompt)
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    seen, hits = set(grams[:P]), 0
    for g in grams[P:]:
        hits += g in seen
        seen.add(g)
    tot = len(grams) - P
    best = 0
    for st in range(len(cont)):
        while st + best < len(cont) and _occurs(seq[:P + st + best], cont[st:st + best + 1]):
            best += 1
    return (hits / tot if tot > 0 else 0.0), best


def _occurs(hay: list[int], needle: list[int]) -> bool:
    k = len(needle)
    return any(hay[i:i + k] == needle for i in range(len(hay) - k + 1))


def make_prompts(a) -> None:
    mcfg = morph_cfg("lxtul")
    rt = tul_runtime(mcfg)
    from lxtul.evaluate import eval_rows
    rows = eval_rows(mcfg, rt, a.rows, 8)
    prompts, refs = [], []
    for x, _y, lay in rows:
        for b in range(x.shape[0]):
            ids = x[b][~lay.slot_mask[b]].tolist()
            if len(ids) < PROMPT_LEN + GEN_LEN:
                raise ValueError(f"row holds {len(ids)} tokens, need {PROMPT_LEN + GEN_LEN}")
            prompts.append(ids[:PROMPT_LEN])
            refs.append(ids[PROMPT_LEN:PROMPT_LEN + GEN_LEN])
    torch.save({"prompts": prompts, "refs": refs}, a.out)
    print(f"wrote {len(prompts)} prompts to {a.out}")


class Packer:
    """Re-packs a growing token list into MORPH's fixed TUL row every step."""

    def __init__(self, mcfg, rt):
        self.rule = rt.val_data_cfg.rule
        self.spec = rt.val_data_cfg.spec_for(int(mcfg.data.seq_len))
        self.slot_id = int(rt.val_data_cfg.slot_id)
        lut = np.asarray(self.rule.is_boundary)
        self.filler = int(next(i for i in range(1000, len(lut)) if not lut[i]))

    def batch(self, seqs: list[list[int]], device):
        from morph.model.tul_layout import SlotLayout, pack_tul_row
        need = self.spec.l_total + 1
        rows, q_tok = [], []
        for s in seqs:
            buf = np.asarray(s + [self.filler] * (need - len(s)), dtype=np.int64)
            arrays, n_used, _st = pack_tul_row(buf, self.rule, self.spec)
            if n_used < len(s):
                raise ValueError(f"row packed {n_used} tokens of {len(s)} real ones")
            lab = arrays["labels"].copy()
            tok_pos = np.flatnonzero(~arrays["slot_mask"])
            lab[tok_pos[len(s) - 1:]] = -100            # the query and the filler: no label
            arrays["labels"] = lab
            rows.append(arrays)
            q_tok.append(len(s) - 1)
        x = torch.from_numpy(np.stack([r["input_ids"] for r in rows])).to(device)
        y = torch.from_numpy(np.stack([r["labels"] for r in rows])).to(device)
        lay = SlotLayout.from_rows(rows, self.spec.prefix_k)
        for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
            setattr(lay, f, getattr(lay, f).to(device))
        return x, y, lay, torch.tensor(q_tok, device=device)


@torch.no_grad()
def next_logprobs(model, kind, seqs, packer, depth, device):
    """[B, V] next-token log-probs after each sequence (all the same length)."""
    if kind in ("plain", "gpt"):
        x = torch.tensor(seqs, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if hasattr(model, "pointer"):
                lg = model.inner(x, return_logits=True,
                                 num_steps_pair=torch.tensor([depth, 0], device=device))["logits"]
                h = model._h
                model._h = None
            else:
                kw = {} if kind == "gpt" else {"num_steps_pair": torch.tensor([depth, 0], device=device)}
                lg = model(x, return_logits=True, **kw)["logits"]
        lp = torch.log_softmax(lg[:, -1].float(), -1)
        if hasattr(model, "pointer"):
            lab = torch.full_like(x, -100)
            lab[:, :-1] = x[:, 1:]
            q = torch.full((x.shape[0],), x.shape[1] - 1, device=device)
            lp = model.pointer.mixed_at(h.float(), lab, torch.zeros_like(x, dtype=torch.bool), lp, q)
        return lp
    x, y, lay, q_tok = packer.batch(seqs, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(x, None, lay, depth=depth)["hidden"]
    order = torch.argsort(lay.slot_mask.to(torch.int8), dim=1, stable=True)
    b = torch.arange(x.shape[0], device=device)
    hq = h[b, order[b, q_tok]].float()
    lg = (hq @ model.lm_head.weight.float().T) * model.config.init.logit_scale
    lg[:, model.tul.slot_id] = float("-inf")
    lp = torch.log_softmax(lg, -1)
    if model.tul.pointer_heads > 0:
        lp = model.pointer.mixed_at(h.float(), y, lay.slot_mask, lp, q_tok, lay)
    elif model.tul.copy_cache:
        raise NotImplementedError("copy-cache generation is not wired here")
    return lp


def sample(lp: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
    if mode == "greedy":
        return lp.argmax(-1)
    p = lp.exp()
    sp, si = torch.sort(p, -1, descending=True)
    keep = (sp.cumsum(-1) - sp) < 0.95                     # nucleus p = 0.95, T = 1
    sp = sp * keep
    pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen).squeeze(-1)
    return si.gather(1, pick[:, None]).squeeze(-1)


@torch.no_grad()
def score_nll(scorer, prompts, conts, device, depth):
    """mean NLL of each continuation under the plain scorer, given its prompt. The scorer is
    on the GPU only while it scores (an 8 GB card holds one model and its activations)."""
    scorer.to(device)
    out = []
    for i in range(0, len(prompts), 8):
        P, C = prompts[i:i + 8], conts[i:i + 8]
        x = torch.tensor([p + c for p, c in zip(P, C)], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = scorer(x[:, :-1], return_logits=True,
                        num_steps_pair=torch.tensor([depth, 0], device=device))["logits"]
        nll = F.cross_entropy(lg.float().transpose(1, 2), x[:, 1:], reduction="none")
        out += nll[:, PROMPT_LEN - 1:].mean(1).tolist()
    scorer.cpu()
    torch.cuda.empty_cache()
    return out


def metrics(prompts, conts, nlls):
    rows = [dict(seq_rep_4=seq_rep_n(c, 4), rep_l=rep_l(c, 128),
                 **dict(zip(("ctx_copy_4", "max_copy"), ctx_copy(p, c))), nll=n)
            for p, c, n in zip(prompts, conts, nlls)]
    res = {}
    for k in ("seq_rep_4", "rep_l", "ctx_copy_4", "max_copy"):
        v = [r[k] for r in rows]
        lo, hi = bootstrap_ci(v)
        res[k] = {"mean": float(np.mean(v)), "lo": lo, "hi": hi}
    res["distinct_4"] = distinct_n(conts, 4)
    res["gen_ppl"] = math.exp(float(np.mean(nlls)))
    return res


def run(a) -> None:
    dev = "cuda"
    data = torch.load(a.prompts)
    prompts, refs = data["prompts"][:a.n], data["refs"][:a.n]
    mcfg = morph_cfg("lxtul")
    packer = Packer(mcfg, tul_runtime(mcfg))
    scorer, _sk, _ = load_run(Path(a.runs_root) / a.scorer)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(mcfg.data.tokenizer))
    except Exception as e:                                    # samples print as ids then
        print(f"[gen] tokenizer unavailable ({e!r}); examples will show token ids", flush=True)
    res = {"REAL": metrics(prompts, refs, score_nll(scorer, prompts, refs, dev, a.depth))}
    examples = {"REAL": refs[:a.show]}
    print(f"[gen] REAL {json.dumps(res['REAL'])}", flush=True)
    for arm in a.arms.split(","):
        model, kind, _ = load_run(Path(a.runs_root) / arm)
        model.to(dev).eval()
        for mode in ("greedy", "sample_p95"):
            torch.manual_seed(a.seed)                          # Parcae's eval noise draw
            gen = torch.Generator(device=dev).manual_seed(a.seed)
            conts = []
            t0 = time.time()
            for i in range(0, len(prompts), a.batch):
                seqs = [list(p) for p in prompts[i:i + a.batch]]
                for _ in range(GEN_LEN):
                    lp = next_logprobs(model, kind, seqs, packer, a.depth, dev)
                    nxt = sample(lp, mode, gen).tolist()
                    for s, t in zip(seqs, nxt):
                        s.append(int(t))
                conts += [s[PROMPT_LEN:] for s in seqs]
            key = f"{arm} [{mode}]"
            res[key] = metrics(prompts, conts, score_nll(scorer, prompts, conts, dev, a.depth))
            res[key]["wall_s"] = time.time() - t0
            examples[key] = conts[:a.show]
            print(f"[gen] {key} {json.dumps(res[key])}", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
        model.cpu()
        del model
        torch.cuda.empty_cache()
    dec = (lambda ids: tok.decode(ids)) if tok is not None else (lambda ids: " ".join(map(str, ids)))
    md = ["# Generation samples", "",
          f"{a.n} prompts of {PROMPT_LEN} tokens, {GEN_LEN} new tokens, depth {a.depth}, "
          f"seed {a.seed}. greedy and nucleus p=0.95 at T=1. gen_ppl under {a.scorer}.", "",
          "| arm | seq_rep_4 | rep_l | ctx_copy_4 | max_copy | distinct_4 | gen_ppl |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in res.items():
        md.append(f"| {k} | {r['seq_rep_4']['mean']:.4f} | {r['rep_l']['mean']:.3f} | "
                  f"{r['ctx_copy_4']['mean']:.4f} | {r['max_copy']['mean']:.1f} | "
                  f"{r['distinct_4']:.4f} | {r['gen_ppl']:.2f} |")
    for j in range(a.show):
        md += ["", f"## Prompt {j}", "", "```", dec(prompts[j]), "```"]
        for k, ex in examples.items():
            md += ["", f"**{k}**", "", "```", dec(ex[j]), "```"]
    Path(a.examples).write_text("\n".join(md) + "\n")
    print(f"[gen] wrote {a.out} and {a.examples}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=32)
    r = sub.add_parser("run")
    r.add_argument("--prompts", required=True)
    r.add_argument("--runs-root", required=True)
    r.add_argument("--arms", required=True)
    r.add_argument("--scorer", default="plain-5k")
    r.add_argument("--n", type=int, default=32)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--depth", type=int, default=6)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--show", type=int, default=4)
    r.add_argument("--out", required=True)
    r.add_argument("--examples", required=True)
    a = ap.parse_args()
    make_prompts(a) if a.cmd == "prompts" else run(a)


if __name__ == "__main__":
    main()
