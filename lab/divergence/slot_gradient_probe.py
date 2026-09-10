"""Per-pass GRADIENT probe for a SLOT-LOOP checkpoint: who pays for which loop pass.

The forward instruments (`slot_anatomy.py`, `slot_state_probe.py`) already say the core
blocks barely move the slot state and every K-curve is flat. Nobody had read the BACKWARD.
This does, on a trained checkpoint, at a forced depth, for each loss source separately:

1. **Per-pass cotangent norm, per loss source.** `_probe_loop` + `_probe_cot` make
   `_tul_core` record `_loop_cot[t]` (PRE-clip) and, when `slot_cot_clip > 0`,
   `_loop_cot_post[t]` / `_loop_cot_bind[t]`. ONE forward, three backwards (token CE
   alone, the weighted MUX alone, the combined training loss), so every source reads the
   SAME forward realisation.

2. **Per-pass share of the SHARED core weight gradient.** The core blocks run once per
   pass, so a core weight's leaf `.grad` is a sum over passes. The probe splits that sum
   by tapping the weight: a `torch.nn.utils.parametrize` parametrization is appended to
   every core weight, its forward is an identity whose BACKWARD banks the incoming
   gradient under the pass index the forward was evaluated at. Per pass, summed over the
   core weights: Frobenius norm, share, and the cosine to the total.

   Why a tap and not `g^T x` from module hooks: the CCA / compressor / indexer
   projections and the HC residual mixers do NOT call their `nn.Linear`'s forward — they
   read `.weight` and matmul it themselves — so a forward hook sees only the MLP and two
   of the CCA linears. Measured on the tiny CPU model: 8 of 31 core linears fire. The tap
   sits on the weight access itself, so it covers all of them, including the norms.

   **Self-check, mandatory.** `sum_t tap_t` must equal the leaf `.grad` on EVERY tapped
   weight. Under ternary QAT the tap sits AFTER the STE, and `norm_match` is a pure
   straight-through identity (`w + (s*q - w).detach()`), so the leaf gradient is the
   effective weight's gradient and the comparison needs no STE correction. What this
   check does and does not prove: it proves coverage (a weight whose access escaped the
   tap fails it) and that nothing outside the loop touched a core weight; it does NOT
   independently re-derive the pass INDEX, because the tap and the pass counter share
   `state["pass"]`. The pass index is pinned separately: `core[0]` must run exactly
   `--depth` times, every tapped weight must be seen in exactly those passes with the
   same access count each, and `_tul_core`'s own loop index (via `_loop_cot`) must
   produce exactly `--depth` keys.

   **Cross-check.** On the layers whose module forward DOES fire, the first batch also
   computes `dW_t = g_t^T x_t` from forward/tensor hooks and compares it, per pass,
   against the tap's per-pass gradient. That is a second, independent derivation of the
   same quantity (`tests/test_slot_gradient_probe.py` runs it on every pass of a tiny
   model and proves it fails when the outer product is perturbed).

3. **Where the gradient goes.** Leaf-gradient norms of the core blocks vs the prelude vs
   the coda vs `tul.*` vs the embeddings, from every backward.

What the probe CHANGES about the training forward, and why (printed, and in the JSON's
`notes`):

* `tul.slot_depth_fixed = --depth` — every slot runs the same number of passes, or a
  per-pass table is an average over ragged depths.
* `model.slot_gain_lambda = 0.0` — `_slot_gain_penalty` applies the core step TWICE more
  at one iteration, which would shift the pass counter and pollute every per-pass number.
  Consequence: on an arm that trains with the hinge, the "total" source here is the
  training loss MINUS the hinge penalty.
* `model.ckpt_grad_iters = 0` — gradient checkpointing re-runs the core forward during
  backward, which would tap every weight a second time under the wrong pass index.
  `use_reentrant=False` restores the RNG on recompute, so switching it off is
  numerically identical.
* `model.train()` with dropout ON, seeded per batch. Eval mode would change the map the
  gradient is read through (dropout, and `_sample_slot_depths`' eval branch).

Usage:
  python lab/divergence/slot_gradient_probe.py \
      --ckpt LABEL=CONFIG=PATH --rows 12 --batch 3 --depth 6 --out .../LABEL.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

from morph.model.layers.block_sparse import CMSBlockLinear  # noqa: E402


# ── the tap ────────────────────────────────────────────────────────────────────


class _TapFn(torch.autograd.Function):
    """Identity forward; the backward hands the incoming gradient to `sink`.

    The output is a VIEW of the input (`view_as`), so the tap costs no memory: the
    parametrized weight the layer multiplies by is the same storage as the effective
    weight the STE produced.
    """

    @staticmethod
    def forward(ctx, w, sink, key):
        ctx.sink, ctx.key = sink, key
        return w.view_as(w)

    @staticmethod
    def backward(ctx, g):
        ctx.sink(ctx.key, g)
        return g, None, None


class Tap(nn.Module):
    """A parametrization whose only job is to label a weight access with a pass index."""

    def __init__(self, name: str, state: dict, sink):
        super().__init__()
        self.name, self.state, self.sink = name, state, sink

    def forward(self, w):
        return _TapFn.apply(w, self.sink, (self.name, self.state["pass"]))


def weight_modules(model) -> list[tuple[str, nn.Module]]:
    """Every module under `model.core` that owns a `weight`, dotted name → module.

    Includes the norms and the MLP's `CMSBlockLinear`. A carved MLP raises: MORTAR's
    leaf is `values`, not `weight`, and the tap would sit on a tensor nothing uses.
    """
    out = []
    for name, mod in model.core.named_modules():
        if isinstance(mod, CMSBlockLinear) and not getattr(mod, "_dense_mode", True):
            raise RuntimeError(
                f"core.{name} is CARVED (MORTAR BCSR): its weight leaf is `values`, not "
                "`weight`. This probe only reads a pre-carve checkpoint.")
        p = getattr(mod, "parametrizations", None)
        if (p is not None and "weight" in p) or isinstance(
                mod._parameters.get("weight"), nn.Parameter):
            out.append((f"core.{name}", mod))
    return out


def leaf_of(mod: nn.Module) -> nn.Parameter:
    """The weight LEAF of a module, through any parametrization already registered."""
    p = getattr(mod, "parametrizations", None)
    if p is not None and "weight" in p:
        return p["weight"].original
    return mod.weight


def hookable_linears(model) -> list[tuple[str, nn.Module]]:
    """Core linears whose module forward actually runs (the cross-check's subset)."""
    return [(n, m) for n, m in model.core.named_modules()
            if isinstance(m, (nn.Linear, CMSBlockLinear))]


class Bank:
    """Per-pass gradients, per loss source, accumulated on the CPU in fp32."""

    def __init__(self, names, sources, depth):
        self.depth = depth
        self.sources = list(sources)
        self.b = {s: {n: [None] * depth for n in names} for s in self.sources}
        self.hits = defaultdict(int)

    def add(self, source, name, t, g):
        cur = self.b[source][name][t]
        g = g.detach().to("cpu", torch.float32)
        self.b[source][name][t] = g if cur is None else cur + g


def install_taps(model, targets, state, bank):
    """Append a `Tap` to every target weight. Returns the name → leaf map."""
    leaves = {}
    for name, mod in targets:
        leaves[name] = leaf_of(mod)

        def sink(key, g, _bank=bank, _state=state):
            src = _state["source"]
            _bank.hits[(key[0], key[1], src)] += 1
            if src is None:
                return
            if key[1] < 0 or key[1] >= _bank.depth:
                _state["stray"].append((key, float(g.norm())))
                return
            _bank.add(src, key[0], key[1], g)
        parametrize.register_parametrization(mod, "weight", Tap(name, state, sink),
                                             unsafe=True)
    return leaves


def hook_crosscheck(model, layers, state, store):
    """Forward hooks that save `x_t`, tensor hooks that turn `g_t` into `g_t^T x_t`."""
    handles = []

    def make(name):
        def fwd(_m, args, out):
            if not state["want_x"]:
                return
            t = state["pass"]
            x = args[0].detach()
            o = out[0] if isinstance(out, tuple) else out
            if not o.requires_grad:
                return

            def back(g, name=name, t=t, x=x):
                if not state["crosscheck"]:
                    return
                gm = g.reshape(-1, g.shape[-1]).float()
                xm = x.reshape(-1, x.shape[-1]).float()
                d = (gm.t() @ xm).to("cpu", torch.float32)
                cur = store[name][t]
                store[name][t] = d if cur is None else cur + d
            o.register_hook(back)
        return fwd

    for name, mod in layers:
        handles.append(mod.register_forward_hook(make(name)))
    return handles


# ── reporting ──────────────────────────────────────────────────────────────────


def pass_stats(bank_src, names, depth, leaf_grads, tol):
    """Per-pass norms / shares / cosines summed over the core weights + the self-check.

    The aggregate treats the concatenation of every core weight as ONE vector: a pass's
    squared norm is the sum of the weights' squared norms, and the inner product with the
    total is the sum of the weights' inner products.
    """
    sq = [0.0] * depth
    dot = [0.0] * depth
    tot_sq = 0.0
    kind_sq = {"attn": [0.0] * depth, "mlp": [0.0] * depth,
               "resid": [0.0] * depth, "norm": [0.0] * depth}
    worst = (0.0, None)
    per_layer = {}
    inert = []
    for name in names:
        leaf = leaf_grads.get(name)
        parts = bank_src[name]
        if leaf is None:
            # A weight the loss never reaches (the CSA indexer's W_IQ: its selection is
            # not differentiated). Nothing banked and nothing to bank; recorded, not
            # silently dropped, and NOT counted as a checked weight.
            if any(p is not None for p in parts):
                raise RuntimeError(
                    f"{name}: the tap banked a gradient but the leaf has none — the tap "
                    "is not on the path the layer actually uses")
            inert.append(name)
            continue
        # A pass that banked nothing contributed exactly zero; the sum check decides.
        parts = [torch.zeros_like(leaf) if p is None else p for p in parts]
        total = torch.zeros_like(parts[0])
        for p in parts:
            total += p
        rel = float((total - leaf).norm() / (leaf.norm() + 1e-30))
        if rel > worst[0]:
            worst = (rel, name)
        tn = float(total.norm())
        tot_sq += tn * tn
        kind = ("mlp" if ".mlp" in name else "resid" if ".mrr_" in name
                else "norm" if "norm" in name.split(".")[-1] else "attn")
        pl_norm, pl_cos = [], []
        for t, p in enumerate(parts):
            n = float(p.norm())
            sq[t] += n * n
            kind_sq[kind][t] += n * n
            d = float((p * total).sum())
            dot[t] += d
            pl_norm.append(n)
            pl_cos.append(d / (n * tn + 1e-30))
        per_layer[name] = {"rel_err": rel, "total_norm": tn,
                           "per_pass_norm": pl_norm, "per_pass_cos_to_total": pl_cos}
    tot_norm = tot_sq ** 0.5
    norms = [s ** 0.5 for s in sq]
    ssum = sum(norms) + 1e-30
    return {
        "per_pass_norm": norms,
        "per_pass_share_of_norm_sum": [n / ssum for n in norms],
        "per_pass_cos_to_total": [dot[t] / (norms[t] * tot_norm + 1e-30)
                                  for t in range(depth)],
        "total_norm": tot_norm,
        "per_pass_norm_mlp": [s ** 0.5 for s in kind_sq["mlp"]],
        "per_pass_norm_attn": [s ** 0.5 for s in kind_sq["attn"]],
        "per_pass_norm_resid": [s ** 0.5 for s in kind_sq["resid"]],
        "per_pass_norm_norm": [s ** 0.5 for s in kind_sq["norm"]],
        "selfcheck": {"weights_checked": len(names) - len(inert),
                      "weights_without_gradient": inert,
                      "max_rel_err": worst[0], "worst_weight": worst[1], "tol": tol,
                      "pass": worst[0] <= tol and len(inert) < len(names)},
        "per_layer": per_layer,
    }


def group_of(name: str) -> str:
    head = name.split(".")[0]
    if head == "core":
        if ".mlp" in name:
            return "core.mlp"
        if ".mrr_" in name:
            return "core.residual"
        if ".attention" in name:
            return "core.attention"
        return "core.other"
    if head in ("prelude", "coda"):
        return head
    if head == "tul":
        return "tul." + name.split(".")[1]
    if head in ("embed", "value_embed_tables", "value_embeds", "lm_mixer"):
        return "embed"
    if head in ("injection", "core_init", "x0_injects"):
        return head
    return "other." + head


def group_norms(named_grads):
    sq = defaultdict(float)
    for n, g in named_grads.items():
        sq[group_of(n)] += float(g.pow(2).sum())
    return {k: v ** 0.5 for k, v in sorted(sq.items())}


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max relative error of the per-pass self-check on any weight")
    ap.add_argument("--xtol", type=float, default=1e-2,
                    help="max relative error of the g^T x cross-check. Looser than --tol "
                         "on purpose: the cross-check recomputes the outer product in "
                         "fp32 from the bf16 tensors the forward saved, while autograd "
                         "does it as a bf16 matmul, so the two disagree by bf16 rounding "
                         "(~2e-3 measured). Under --fp32 the disagreement drops to the "
                         "fp32 floor; the self-check, which compares two fp32 sums of the "
                         "SAME numbers, is unaffected and stays at 1e-3.")
    ap.add_argument("--fp32", action="store_true",
                    help="run the forward without autocast (verification only: it is not "
                         "the arithmetic the arm trained in)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    label, config, path = a.ckpt.split("=", 2)

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_gradient_probe needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false)")
    model, step = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)

    tc = model.cfg.tul
    notes = {
        "slot_depth_fixed": [tc.slot_depth_fixed, a.depth],
        "slot_gain_lambda": [float(model.cfg.slot_gain_lambda), 0.0],
        "ckpt_grad_iters": [int(model.cfg.ckpt_grad_iters), 0],
        "slot_cot_clip": float(model.cfg.slot_cot_clip),
        "core_fixed_point_lambda": float(model.cfg.core_fixed_point_lambda),
        "mux_beta": float(tc.mux_beta or 0.0),
        "mode": "train (dropout ON), seeded per batch",
        "autocast": "off (fp32)" if a.fp32 else "bf16, as the trainer",
    }
    tc.slot_depth_fixed = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth or model.cfg.max_depth))
    model.cfg.slot_gain_lambda = 0.0
    model.cfg.ckpt_grad_iters = 0
    model._probe_loop = True
    model._probe_cot = True
    model.train()

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    targets = weight_modules(model)
    names = [n for n, _ in targets]
    n_core_w = sum(leaf_of(m).numel() for _, m in targets)
    has_mux = float(tc.mux_beta or 0.0) > 0.0
    sources = ["token_ce"] + (["mux"] if has_mux else []) + ["total"]
    bank = Bank(names, sources, a.depth)
    state = {"pass": -1, "source": None, "stray": [], "crosscheck": False, "want_x": False}
    leaves = install_taps(model, targets, state, bank)
    print(f"  tapped {len(targets)} core weights, {n_core_w / 1e6:.1f}M parameters")

    xlayers = hookable_linears(model)
    xstore = {n: [None] * a.depth for n, _ in xlayers}
    xhandles = hook_crosscheck(model, xlayers, state, xstore)

    def pre_core0(_m, _a):
        state["pass"] += 1
    core0 = model.core[0].register_forward_pre_hook(pre_core0)

    real_groups = model._tul_group_losses
    cap: dict = {}

    def spy(*args, **kw):
        r = real_groups(*args, **kw)
        cap["groups"] = r
        return r
    model._tul_group_losses = spy

    leaf_grads = {s: {} for s in sources}
    cot = {s: {"pre": defaultdict(list), "post": defaultdict(list),
               "bind": defaultdict(list)} for s in sources}
    losses = defaultdict(list)
    n_rows = 0
    # cross-check: g^T x against the tap, on the FIRST batch's "total" backward, where the
    # tap bank holds that batch alone so the comparison is exact rather than directional.
    xc = {"layers": 0, "passes": 0, "max_rel_err": 0.0, "worst": None, "tol": a.xtol,
          "metric": "relative Frobenius error of g^T x against the tap, batch 0"}
    try:
        for bi, (inp, labels, layout, _) in enumerate(batches):
            layout = layout.to("cuda")
            state["pass"] = -1
            state["want_x"] = (bi == 0)
            torch.manual_seed(1234 + bi)
            torch.cuda.manual_seed_all(1234 + bi)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32):
                out = model(inp.cuda(), labels=labels.cuda(), slot_layout=layout)
            if state["pass"] + 1 != a.depth:
                raise RuntimeError(
                    f"core[0] ran {state['pass'] + 1} times, expected {a.depth}")
            n_rows += int(inp.shape[0])
            terms = {"token_ce": cap["groups"]["loss"], "total": out["loss"]}
            if has_mux:
                terms["mux"] = float(tc.mux_beta) * model.mux_gate * out["mux_local_live"]
            for k in ("loss", "mux_local", "fixed_point"):
                if k in out:
                    losses[k].append(float(out[k].detach()))
            losses["token_ce"].append(float(cap["groups"]["loss"].detach()))
            for si, s in enumerate(sources):
                model.zero_grad(set_to_none=True)
                model._loop_cot = {}
                state["source"] = s
                state["crosscheck"] = (bi == 0 and s == "total")
                terms[s].backward(retain_graph=(si < len(sources) - 1))
                state["source"] = None
                state["crosscheck"] = False
                for t, v in model._loop_cot.items():
                    cot[s]["pre"][int(t)].append(float(v))
                for t, v in (getattr(model, "_loop_cot_post", None) or {}).items():
                    cot[s]["post"][int(t)].append(float(v))
                for t, v in (getattr(model, "_loop_cot_bind", None) or {}).items():
                    cot[s]["bind"][int(t)].append(float(v))
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        g = p.grad.detach().to("cpu", torch.float32)
                        leaf_grads[s][n] = leaf_grads[s].get(n, 0) + g
                for n, m in targets:
                    lg = leaves[n].grad
                    if lg is not None:
                        cur = leaf_grads[s].get("__leaf__" + n)
                        d = lg.detach().to("cpu", torch.float32)
                        leaf_grads[s]["__leaf__" + n] = d if cur is None else cur + d
            if bi == 0:
                for n, _m in xlayers:
                    parts = xstore[n]
                    if any(p is None for p in parts):
                        continue
                    tapn = "core." + n
                    xc["layers"] += 1
                    for t, p in enumerate(parts):
                        ref = bank.b["total"][tapn][t]
                        rel = float((p - ref).norm() / (ref.norm() + 1e-30))
                        xc["passes"] += 1
                        if rel > xc["max_rel_err"]:
                            xc["max_rel_err"], xc["worst"] = rel, f"{tapn}@pass{t + 1}"
                for k in xstore:
                    xstore[k] = [None] * a.depth
            model.zero_grad(set_to_none=True)
            del out, terms
            torch.cuda.empty_cache()
            print(f"  batch {bi + 1}/{len(batches)} done", flush=True)
    finally:
        core0.remove()
        for h in xhandles:
            h.remove()
        model._tul_group_losses = real_groups

    if state["stray"]:
        raise SystemExit(f"{label}: {len(state['stray'])} core-weight accesses happened "
                         f"OUTSIDE the loop, first {state['stray'][:3]} — the per-pass "
                         "split is not trustworthy")
    # A weight the loss reaches must be reached in EVERY pass, the same number of times
    # per pass. A weight with no backward hits at all (the CSA indexer) is inert and is
    # listed by the self-check instead.
    hits = defaultdict(list)
    for (n, t, s), c in bank.hits.items():
        if s == sources[-1]:
            hits[n].append((t, c))
    bad = {n: sorted(v) for n, v in hits.items()
           if sorted(t for t, _ in v) != list(range(a.depth))
           or len({c for _, c in v}) != 1}
    if bad:
        raise SystemExit(f"{label}: uneven per-pass weight accesses, first "
                         f"{list(bad.items())[:2]}")

    rec = {
        "label": label, "config": config, "step": step, "rows": n_rows,
        "batch": a.batch, "depth": a.depth, "n_batches": len(batches),
        "sources": sources, "notes": notes,
        "tapped_weights": len(targets), "tapped_params": n_core_w,
        "losses": {k: sum(v) / len(v) for k, v in losses.items()},
        "cotangent": {}, "weight_grad": {}, "param_group_grad_norm": {},
    }
    ok = True
    for s in sources:
        rec["cotangent"][s] = {
            kind: [sum(cot[s][kind][t]) / len(cot[s][kind][t]) if cot[s][kind][t] else None
                   for t in range(a.depth)]
            for kind in ("pre", "post", "bind")}
        pre = [v for v in rec["cotangent"][s]["pre"] if v is not None]
        tot = sum(pre) or 1.0
        rec["cotangent"][s]["share_pre"] = [
            (v / tot if v is not None else None) for v in rec["cotangent"][s]["pre"]]
        w = pass_stats(bank.b[s], names, a.depth,
                       {n: leaf_grads[s].get("__leaf__" + n) for n in names}, a.tol)
        ok = ok and w["selfcheck"]["pass"]
        rec["weight_grad"][s] = w
        rec["param_group_grad_norm"][s] = group_norms(
            {k: v for k, v in leaf_grads[s].items() if not k.startswith("__leaf__")})

    xc["pass"] = xc["layers"] > 0 and xc["max_rel_err"] <= xc["tol"]
    rec["crosscheck_gTx_vs_tap"] = xc

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)

    # ── console ───────────────────────────────────────────────────────────────
    print(f"\n{label} step {step} — {n_rows} rows, batch {a.batch}, depth {a.depth}, "
          f"{len(targets)} core weights ({n_core_w / 1e6:.1f}M params)")
    print(f"  probe changes: {json.dumps(notes)}")
    print("  losses: " + " ".join(f"{k}={v:.4f}" for k, v in rec["losses"].items()))
    print(f"  cross-check g^T x vs tap on {xc['layers']} layers: "
          f"{'PASS' if xc['pass'] else 'FAIL'} max {xc['metric']} "
          f"{xc['max_rel_err']:.3e} ({xc['worst']}, tol {xc['tol']:.1e})")
    for s in sources:
        c = rec["cotangent"][s]
        print(f"\n  [{label}/{s}] per-pass cotangent at the loop state (pass 1..{a.depth})")
        print("    pre-clip  ", " ".join(f"{v:9.3e}" if v is not None else "     None"
                                         for v in c["pre"]))
        print("    share     ", " ".join(f"{v:9.3f}" if v is not None else "     None"
                                         for v in c["share_pre"]))
        if any(v is not None for v in c["post"]):
            print("    post-clip ", " ".join(f"{v:9.3e}" if v is not None else "     None"
                                             for v in c["post"]))
            print("    clip-bind ", " ".join(f"{v:9.3f}" if v is not None else "     None"
                                             for v in c["bind"]))
        w = rec["weight_grad"][s]
        print(f"  [{label}/{s}] per-pass share of the SHARED core weight gradient "
              f"(|total| {w['total_norm']:.4e})")
        print("    |dW_t|    ", " ".join(f"{v:9.3e}" for v in w["per_pass_norm"]))
        print("    share     ", " ".join(f"{v:9.3f}" for v in w["per_pass_share_of_norm_sum"]))
        print("    cos->total", " ".join(f"{v:9.4f}" for v in w["per_pass_cos_to_total"]))
        print("    |dW_t| mlp", " ".join(f"{v:9.3e}" for v in w["per_pass_norm_mlp"]))
        print("    |dW_t| att", " ".join(f"{v:9.3e}" for v in w["per_pass_norm_attn"]))
        sc = w["selfcheck"]
        print(f"    SELF-CHECK {'PASS' if sc['pass'] else 'FAIL'}: max rel err "
              f"{sc['max_rel_err']:.3e} on {sc['worst_weight']} over "
              f"{sc['weights_checked']} weights (tol {sc['tol']:.1e})")
        print(f"  [{label}/{s}] parameter-group gradient norms")
        for k, v in rec["param_group_grad_norm"][s].items():
            print(f"      {k:24s} {v:.4e}")
    print("\nwrote", a.out)
    if not (ok and xc["pass"]):
        raise SystemExit(f"{label}: bookkeeping check FAILED — do not report these numbers")


if __name__ == "__main__":
    main()
