"""Repetition-and-diversity eval for MORPH checkpoints, with real text as the anchor
(2026-09-30 task: "Main thing we can measure at this early training is repetition and
diversity" — Wolfe).

Prompts are N real validation DOCUMENTS, read the way the TRAINER reads its own val
stream: same tokenizer, same dataset, same `split="validation"` (falls back to the
`"train"` split for openwebtext internally — `morph/training/data.py`), and the SAME
`skip_samples=50_000` offset `morph/training/train.py::_make_val_loader` uses
(`docs/…`/memory: "OWT val overlaps training" — the probe scripts that read the val
stream from its start at `skip_samples=0`, e.g. `scripts/tul_samples.py::real_text_anchor`,
are NOT reading the trainer's actual val offset; this script deliberately is). Each row
is one contiguous `prompt_len + gen_len` token chunk of that stream (the loader packs a
concatenated document stream into fixed-length chunks, so a row occasionally spans a
document boundary — the same thing that happens to the trainer's own val batches).

  prompt      = row[:prompt_len]
  continuation (model)  = prompt_len NEW tokens sampled by the arm under test
  continuation (real)   = row[prompt_len : prompt_len + gen_len]  — the anchor

Generation goes through the model's OWN forward (`generate_plain_batch` for a plain
model, `generate_tul_batch` for a TUL one — both in `morph/inference/`), never a
reimplementation, so a fan model's write-all mixture, WTA responsibility, etc. are
exercised exactly as the checkpoint trains them. `generate_tul_batch`/`generate_tul`
are the ONLY TUL generators that do not refuse `tul.fan_k > 0`: the KV-cached
generators (`tul_generate_cached.py`, `tul_generate_graphed.py`) explicitly RAISE on a
fan model (`_check_supported`, "the fan... the caches here hold ONE cell") because their
per-(pass, layer) cache has nowhere to put K streams. Read closely: both generators run
`model(ids, slot_layout=layout)` — the model's real `_forward_tul`, full slot loop,
full coda — every step; nothing about them is TUL-specific beyond building the layout.

Metrics (`lab/divergence/gen_diversity.py`, unit-tested in `tests/test_gen_diversity.py`):
  seq-rep-n (n=1..4), rep-l (window 128), distinct-n (n=1,2,4, corpus-level), and a
  gen-PPL guard — the mean per-token NLL of a continuation, teacher-forced through the
  PLAIN checkpoint (never the arm scoring itself: a degenerate repetition loop scores an
  excellent perplexity under its OWN weights, which is exactly the failure mode the
  guard exists to catch — memory: a repetition loop scored gen-PPL 1.46 vs real text's
  32.44). Reported as both the mean NLL (nats/token) and exp(mean NLL) ("gen_ppl").

Two decode settings, matching every arm and the real-text anchor:
  greedy        temperature 0 (argmax)              — diagnostic: is there a loop at all
  sample_t1_p95 temperature 1.0, nucleus top_p 0.95, fixed per-row seed — the ranking mode

TIME BUDGET (measured 2026-09-30 on the heaviest arm, `a2` = fan_k=4, eager
TG-restricted kernels): 42.5-42.7 s per row per decode setting at prompt_len=128,
gen_len=256, batch=N. The eager generators recompute the WHOLE row every step by
design (`tul_generate.py`'s docstring), and their per-step sampling is a Python `for`
loop over the batch (`generate_tul_batch`/`generate_plain_batch`), each row forcing its
own GPU->host sync through `sample_next`'s `int(...)` cast — so wall time scales
roughly LINEARLY in N, not sublinearly the way a compute-bound batched forward would.
N=64 x 2 settings extrapolates to ~91 minutes for one TUL arm, which would blow the
environment's ~15-minute GPU-hold budget outright (see `run_eval.sh`'s comment for the
arithmetic). `run_eval.sh` therefore defaults to N=8 (~11.4 minutes wall for both
settings on `a2`), NOT the requested N=64 — raise `--n` only with a relaxed hold-time
budget.

GPU discipline: ONE arm model resident at a time, plus the (small) PLAIN checkpoint kept
resident for the whole run as the gen-PPL scorer — never two ARMS at once. Each `--ckpt`
in one invocation is processed, scored and freed (`del model; torch.cuda.empty_cache()`)
before the next is built, so `run_eval.sh` can also call this script once per checkpoint
under its own `flock` for the ~15-minute GPU-hold budget.

Output is MERGED into `--out` (existing arms are kept; the arm(s) processed this call are
added/overwritten by label) so a queue of per-checkpoint invocations builds one table.
`--examples_out` (markdown) is always fully REGENERATED from the merged JSON, so it never
accumulates duplicate sections across re-runs of the same label.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "lab", "divergence"))

from gen_diversity import bootstrap_ci, bootstrap_ci_rows, distinct_n, rep_l, seq_rep_n  # noqa: E402
from morph.inference.plain_generate import generate_plain_batch  # noqa: E402
from morph.inference.tul_generate import generate_tul_batch  # noqa: E402
from morph.training.tul_setup import build_tul_runtime  # noqa: E402
from tul_samples import emit_source_for, load_cfg, load_ckpt  # noqa: E402

VAL_SKIP_SAMPLES = 50_000  # morph/training/train.py::_make_val_loader — the trainer's own offset.

# (label, temperature, top_k, top_p, is_greedy)
DECODES = [
    ("greedy", 0.0, 0, 0.0, True),
    ("sample_t1_p95", 1.0, 0, 0.95, False),
]

SEQ_REP_NS = (1, 2, 3, 4)
DISTINCT_NS = (1, 2, 4)
REP_L_WINDOW = 128


# ── val documents, read the trainer's way ──────────────────────────────────────────
def read_val_rows(tokenizer_name: str, dataset_name: str, n: int, prompt_len: int,
                  gen_len: int, batch_size: int = 8) -> list[list[int]]:
    """N rows of exactly `prompt_len + gen_len` tokens, from the TRAINER's val offset."""
    from morph.training.data import create_dataloader
    loader = create_dataloader(tokenizer_name, dataset_name, prompt_len + gen_len,
                               batch_size, split="validation",
                               skip_samples=VAL_SKIP_SAMPLES, bag_size=0, tul=None)
    rows: list[list[int]] = []
    while len(rows) < n:
        x, _y = next(loader)
        rows.extend(x[i].tolist() for i in range(x.shape[0]))
    return rows[:n]


# ── gen-PPL: teacher-forced NLL of the continuation, under the PLAIN checkpoint ─────
@torch.no_grad()
def gen_ppl_nll(plain_model, rows: list[list[int]], prompt_len: int, device,
                chunk: int = 16) -> list[float]:
    """Per-row mean NLL (nats) of the continuation tokens, teacher-forced.

    `rows[i]` is `prompt + continuation` (any source: real text or a model's own
    sample), all the SAME length. Logit at position `prompt_len-1` predicts the
    continuation's first token, ..., logit at `L-2` predicts the last — the ordinary
    next-token alignment, scored ONLY over the continuation span.
    """
    was_training = plain_model.training
    plain_model.eval()
    out: list[float] = []
    try:
        for s in range(0, len(rows), chunk):
            batch = rows[s:s + chunk]
            ids = torch.tensor(batch, dtype=torch.long, device=device)
            res = plain_model(ids)
            logits = (res["logits"] if isinstance(res, dict) else res).float()
            pred = logits[:, prompt_len - 1:-1, :]                 # [b, gen_len, V]
            target = ids[:, prompt_len:]                           # [b, gen_len]
            ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1),
                                 reduction="none").reshape(target.shape)
            out.extend(ce.mean(dim=1).tolist())
    finally:
        if was_training:
            plain_model.train()
    return out


# ── diversity metrics over one set of continuations ─────────────────────────────────
def diversity_report(conts: list[list[int]]) -> dict:
    """seq-rep-n (n=1..4) and rep-l averaged per continuation; distinct-n (1,2,4)
    pooled over the whole set. Bootstrap 95% CIs for seq-rep-4 and distinct-4 over the
    ROWS (prompts)."""
    per_row_rep = {n: [seq_rep_n(c, n) for c in conts] for n in SEQ_REP_NS}
    per_row_repl = [rep_l(c, REP_L_WINDOW) for c in conts]
    out = {f"seq_rep_{n}": float(np.mean(per_row_rep[n])) for n in SEQ_REP_NS}
    out["rep_l"] = float(np.mean(per_row_repl))
    for n in DISTINCT_NS:
        out[f"distinct_{n}"] = distinct_n(conts, n)
    out["seq_rep_4_ci"] = list(bootstrap_ci(per_row_rep[4]))
    # n_boot kept modest here: each resample rebuilds n-grams over the WHOLE pooled
    # corpus (unlike bootstrap_ci's vectorised scalar mean), so 2000 resamples over 64
    # rows of 256 tokens is a real cost, not a rounding error.
    out["distinct_4_ci"] = list(
        bootstrap_ci_rows(conts, lambda rs: distinct_n(rs, 4), n_boot=300))
    out["n_rows"] = len(conts)
    return out


def gen_ppl_report(nlls: list[float]) -> dict:
    return {
        "gen_ppl_nll_mean": float(np.mean(nlls)),
        "gen_ppl": float(np.exp(np.mean(nlls))),
        "gen_ppl_nll_ci": list(bootstrap_ci(nlls)),
    }


# ── one arm, one decode setting ─────────────────────────────────────────────────────
def run_setting(model, tul_rt, rule_spec, prompts: list[list[int]], gen_len: int,
                label: str, temp: float, top_k: int, top_p: float, seed: int,
                device, tokenizer, plain_model, prompt_len: int) -> dict:
    seeds = [seed + i for i in range(len(prompts))] if temp > 0.0 else None
    t0 = time.time()
    if tul_rt is None:
        conts = generate_plain_batch(model, prompts, max_new_tokens=gen_len,
                                     temperature=temp, top_k=top_k, top_p=top_p,
                                     seeds=seeds, device=device)
    else:
        rule, spec = rule_spec
        conts, _builders = generate_tul_batch(
            model, prompts, rule, spec, max_new_tokens=gen_len, temperature=temp,
            top_k=top_k, top_p=top_p, seeds=seeds, device=device,
            emit_source=emit_source_for(tul_rt))
    wall = time.time() - t0

    report = diversity_report(conts)
    rows = [p + c for p, c in zip(prompts, conts)]
    nlls = gen_ppl_nll(plain_model, rows, prompt_len, device)
    report.update(gen_ppl_report(nlls))
    report["wall_s"] = wall
    report["wall_s_per_row"] = wall / len(prompts)

    examples = []
    for i in range(min(3, len(prompts))):
        examples.append({
            "prompt": tokenizer.decode(prompts[i], skip_special_tokens=True),
            "continuation": tokenizer.decode(conts[i], skip_special_tokens=True),
        })
    report["examples"] = examples
    print(f"      {label:16s} seq_rep_4={report['seq_rep_4']:.4f} "
          f"distinct_4={report['distinct_4']:.4f} gen_ppl={report['gen_ppl']:.2f} "
          f"({wall:.0f}s, {report['wall_s_per_row']:.2f}s/row)", flush=True)
    return report


# ── main ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=ovr1,ovr2,...], repeatable")
    ap.add_argument("--plain_ckpt", required=True,
                    help="LABEL=CONFIG=PATH of the PLAIN (non-TUL) checkpoint used as "
                         "the gen-PPL scorer for every arm, including itself")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples_out", default="")
    a = ap.parse_args()

    device = torch.device(a.device)
    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out_path.read_text()) if out_path.exists() else {}
    payload.setdefault("arms", {})

    # ── plain scorer (resident for the whole run) ──────────────────────────────────
    p_label, p_cfg_name, p_path, p_ovr = _parse_ckpt(a.plain_ckpt)
    if p_ovr:
        raise NotImplementedError("gen_diversity_eval: --plain_ckpt overrides not supported")
    print(f"=== loading PLAIN scorer {p_label} [{p_cfg_name}] {p_path} ===", flush=True)
    plain_cfg = load_cfg(p_cfg_name)
    plain_tul_rt = build_tul_runtime(plain_cfg)
    if plain_tul_rt is not None:
        raise ValueError(f"--plain_ckpt {p_label} builds a TUL runtime; it must be the "
                          f"non-TUL control (emit_weight/activate_at never)")
    plain_model, plain_step = load_ckpt(plain_cfg, p_path, device, None)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plain_cfg.data.tokenizer)

    rows = read_val_rows(plain_cfg.data.tokenizer, plain_cfg.data.dataset, a.n,
                         a.prompt_len, a.gen_len)
    prompts = [r[:a.prompt_len] for r in rows]
    real_conts = [r[a.prompt_len:a.prompt_len + a.gen_len] for r in rows]
    print(f"    {len(rows)} val rows read (skip_samples={VAL_SKIP_SAMPLES}, "
          f"split=validation, tokenizer={plain_cfg.data.tokenizer})", flush=True)

    # Always recomputed (cheap: N rows, one forward, no generation) rather than reused
    # from a previous merge — deterministic in (n, prompt_len, gen_len, skip_samples),
    # so re-running never drifts, and a parameter change is reflected immediately.
    anchor = diversity_report(real_conts)
    real_rows = [p + c for p, c in zip(prompts, real_conts)]
    nlls = gen_ppl_nll(plain_model, real_rows, a.prompt_len, device)
    anchor.update(gen_ppl_report(nlls))
    anchor["examples"] = [tokenizer.decode(c, skip_special_tokens=True)
                          for c in real_conts[:3]]
    payload["real_anchor"] = anchor
    print(f"    REAL TEXT anchor: seq_rep_4={anchor['seq_rep_4']:.4f} "
          f"distinct_4={anchor['distinct_4']:.4f} gen_ppl={anchor['gen_ppl']:.2f}",
          flush=True)

    payload["_meta"] = {
        "n": a.n, "prompt_len": a.prompt_len, "gen_len": a.gen_len, "seed": a.seed,
        "val_skip_samples": VAL_SKIP_SAMPLES, "val_split": "validation",
        "decodes": [d[0] for d in DECODES], "plain_scorer": p_label,
        "plain_scorer_step": plain_step, "rep_l_window": REP_L_WINDOW,
    }

    # ── plain arm decodes with generate_plain (no TUL runtime) even if it IS the
    #    scorer model — a separate generation pass (temperature != 0 has its own RNG
    #    draws); reuse the resident weights, never reload.
    for spec_s in a.ckpt:
        label, cfg_name, path, ovr = _parse_ckpt(spec_s)
        print(f"\n=== {label}  [{cfg_name}]  {path} ===", flush=True)
        if label == p_label and cfg_name == p_cfg_name and path == p_path:
            model, tul_rt, step = plain_model, None, plain_step
            print("    (same checkpoint as the plain scorer — reusing resident weights)",
                  flush=True)
        else:
            cfg = load_cfg(cfg_name)
            if ovr:
                raise NotImplementedError(
                    "gen_diversity_eval: per-ckpt overrides not supported yet "
                    f"(got {ovr} for {label})")
            tul_rt = build_tul_runtime(cfg)
            model, step = load_ckpt(cfg, path, device,
                                    None if tul_rt is None else tul_rt.model_cfg)

        rule_spec = None
        if tul_rt is not None:
            spec = tul_rt.data_cfg.spec_for(a.prompt_len)
            spec = dataclasses.replace(spec, max_slots=spec.max_slots + a.gen_len)
            rule_spec = (tul_rt.data_cfg.rule, spec)

        arm = {"step": step, "config": cfg_name, "path": path, "tul": tul_rt is not None}
        for dlabel, temp, top_k, top_p, _greedy in DECODES:
            arm[dlabel] = run_setting(model, tul_rt, rule_spec, prompts, a.gen_len,
                                      dlabel, temp, top_k, top_p, a.seed, device,
                                      tokenizer, plain_model, a.prompt_len)
        payload["arms"][label] = arm

        if model is not plain_model:
            del model
            torch.cuda.empty_cache()

        out_path.write_text(json.dumps(payload, indent=2))
        print(f"    [flush] {out_path} now holds {len(payload['arms'])} arm(s)",
              flush=True)

    print_table(payload)
    if a.examples_out:
        write_examples_md(payload, pathlib.Path(a.examples_out))
    print(f"\nwrote {out_path}")


def _parse_ckpt(spec: str) -> tuple[str, str, str, list[str]]:
    parts = spec.split("=", 3)
    if len(parts) < 3 or not all(parts[:3]):
        raise ValueError(f"--ckpt wants LABEL=CONFIG=PATH[=ovr,...], got {spec!r}")
    label, cfg, path = parts[0], parts[1], parts[2]
    ovr = [o for o in parts[3].split(",") if o] if len(parts) == 4 else []
    if not path.startswith("/"):
        path = os.path.join(_ROOT, path)
    return label, cfg, path, ovr


def print_table(payload: dict) -> None:
    anchor = payload.get("real_anchor", {})
    print("\n" + "=" * 100)
    header = (f"{'arm':28s} {'decode':16s} {'seq_rep_4':>10s} {'distinct_4':>11s} "
             f"{'rep_l':>8s} {'gen_ppl':>9s} {'wall_s/row':>11s}")
    print(header)
    print("-" * 100)

    def _row(name, d):
        ci4 = d.get("seq_rep_4_ci", [None, None])
        dci = d.get("distinct_4_ci", [None, None])
        print(f"{name:28s} {'':16s} {d['seq_rep_4']:>6.4f}±{(ci4[1]-ci4[0])/2:.3f} "
              f"{d['distinct_4']:>6.4f}±{(dci[1]-dci[0])/2:.3f} "
              f"{d.get('rep_l', float('nan')):>8.4f} "
              f"{d.get('gen_ppl', float('nan')):>9.2f} "
              f"{d.get('wall_s_per_row', float('nan')):>11.3f}")

    if anchor:
        _row("REAL TEXT", anchor)
    for label, arm in payload.get("arms", {}).items():
        for dlabel, _t, _k, _p, _g in DECODES:
            if dlabel in arm:
                _row(f"{label} [{dlabel}]", arm[dlabel])
    print("=" * 100)


def write_examples_md(payload: dict, path: pathlib.Path) -> None:
    lines = ["# Generation diversity eval — example continuations", ""]
    meta = payload.get("_meta", {})
    lines.append(f"n={meta.get('n')} prompt_len={meta.get('prompt_len')} "
                f"gen_len={meta.get('gen_len')} seed={meta.get('seed')} "
                f"val_skip_samples={meta.get('val_skip_samples')}")
    lines.append("")
    anchor = payload.get("real_anchor", {})
    if anchor.get("examples"):
        lines.append("## REAL TEXT (anchor)")
        for i, text in enumerate(anchor["examples"]):
            lines.append(f"\n**continuation {i}**\n\n```\n{text}\n```")
        lines.append("")
    for label, arm in payload.get("arms", {}).items():
        lines.append(f"## {label} (step {arm.get('step')}, config `{arm.get('config')}`)")
        for dlabel, _t, _k, _p, _g in DECODES:
            if dlabel not in arm:
                continue
            lines.append(f"\n### {dlabel}\n")
            for ex in arm[dlabel].get("examples", []):
                lines.append(f"**prompt:** `{ex['prompt']}`\n")
                lines.append(f"**continuation:**\n\n```\n{ex['continuation']}\n```\n")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
