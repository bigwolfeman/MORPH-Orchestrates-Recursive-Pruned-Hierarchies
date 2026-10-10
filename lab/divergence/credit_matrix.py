"""Credit assignment on an LXTUL checkpoint: which loss term trains which part, by which path.

Wolfe's question (2026-10-10): "What is the credit assignment for each of the prelude, the
loop, and the coda?" on LXTUL + pointer (`lxtul_pointer`). Three readings, three parts:

``--part data``   CPU only. Packs the batches once and saves them, so the GPU job does not
                  tokenise while it holds the lock. TRAIN batches are the trainer's own
                  stream (`split="train"`, `tul=data_cfg`, batch 6) fast-forwarded by
                  ``--train-skip`` batches: with the default 5000 they are the batches step
                  5001+ would have read, never seen by a step-5000 checkpoint. VAL batches
                  are the trainer's val stream (`split="validation"`, skip 50,000 docs,
                  `tul=val_data_cfg`), clear of a 5k run's training docs (memory
                  owt-val-overlaps-training).

``--part grad``   The gradient CREDIT MATRIX and the PATH numbers, train mode, bf16
                  autocast, the trainer's own loss (`out["loss"]` of the labelled forward;
                  routing and the spectral penalty are off on this config). ONE forward per
                  batch, then one `autograd.grad` per loss term on the retained graph, so
                  every term sees the same batch, dropout masks and Poisson depths.

                  The terms are not guessed. Every key of the forward's output that ends in
                  ``_weighted`` is a term the model folded into `out["loss"]`; its LIVE
                  tensor is either the value itself (`fp_weighted`) or the tensor whose
                  `.detach()` produced it, recorded by a TorchFunctionMode during the
                  forward. The token CE is `_tul_group_losses`'s own `out["loss"]`. The
                  run REFUSES (exit 1) unless ``CE + sum(terms) == out["loss"]`` to 1e-4
                  relative: a term this script did not see would otherwise be booked as CE.
                  `fan_lsel_weighted` is split into its three summands (latent WTA, online
                  floor, router CE) from the same detach records, and the split is checked
                  against the folded value.

                  THE TIED HEAD. `embed.lm_weight()` IS the input embedding table, so one
                  parameter carries both the input lookup and the LM head. The head's use
                  is read through SHADOW leaves: `lm_weight()` is rebuilt from detached
                  copies of the table's `.original` tensors (through the same int6
                  parametrisation), so the head's gradient lands on the shadows and the
                  input lookup's on the real parameters. Exact: the two sum to the real
                  table's gradient by the chain rule. Checked: the rebuilt table equals the
                  real one bit for bit before any forward.

                  THE COTANGENT CLIP. `model.slot_cot_clip` (4.0 here) rescales, per row,
                  the gradient entering each slot-loop pass relative to the exit
                  cotangent. That is NOT linear in the loss, so per-term gradients do not
                  sum to the trainer's gradient while it binds. The matrix is therefore
                  computed with the clip OFF (exactly additive, checked per column), and
                  the trainer's clipped total is read separately and compared per column
                  (`clip_effect`).

                  Entries per (term, group): the L2 norm of d(term)/d(group); the NORM share
                  (norm / sum of norms in the column, sums to 1, sign-blind); the PROJECTION
                  share (<g_term, g_total> / |g_total|^2 in the column, sums to 1 exactly,
                  signed: a negative entry is a term that pulls AGAINST the column's total).

                  PATHS. The gradient each term sends to: the prelude OUTPUT at token
                  positions and at slot positions (`_tul_front`'s return); the loop EXIT
                  cells (`_tul_core`'s return, all M cells, split winner / losers by the
                  latent-selected loop's final pick); the routed cells the coda's write
                  reads (`prefix_project(cells=)`); the span decoder's input (the cells'
                  mean). Then, for the token CE and the span decoder, the VJP from the exit
                  cells back to every parameter group, split winner / losers: how much of
                  the loop's gradient from that term enters through the exit, and through
                  which cell.

``--part worth``  Eval-mode ablations in nats on the SAME val rows: the full model; the
                  loop's write zeroed (`plan_mode="zero"`, the trainer's
                  `val/plan_worth_zero`); forced slot depths (K1-K6 and the curve); the
                  pointer head off (gate forced onto the model, core_depth_sweep's
                  `--pointer-off`); the coda's TOKEN carrier replaced by the prelude's
                  input (`tul.coda_token_input="embed"`, an existing TULConfig knob read at
                  forward time) -- an UPPER BOUND of damage on a model never trained that
                  way, not a verdict. Per-row CE sums give paired bootstrap CIs
                  (`_stats.paired_bootstrap_ci`). The full model's per-token CE is checked
                  against the labelled forward's own `ce_tokens` on the first batch.

Usage (repo root, the MORPH venv; GPU parts under the lock):
  python lab/divergence/credit_matrix.py --part data --config lxtul_pointer \
      --batches-file ignore/credit/batches.pt --train-batches 8 --val-batches 80
  python lab/divergence/credit_matrix.py --part grad --config lxtul_pointer \
      --ckpt PATH/step_5000.pt --batches-file ignore/credit/batches.pt --out grad.json
  python lab/divergence/credit_matrix.py --part worth --config lxtul_pointer \
      --ckpt PATH/step_5000.pt --batches-file ignore/credit/batches.pt --out worth.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

# ── parameter groups ─────────────────────────────────────────────────────────────────────
# Ordered; every trainable parameter must land in exactly one (checked at run time).
GROUPS = ("embed_input", "lm_head_tied", "prelude", "slot_seed_register", "loop_core",
          "cell_norm", "router_latent_head", "W_prefix", "coda",
          "pointer", "span_decoder", "readout_mixer_norm")


def group_of(name: str, n_prelude: int, n_core: int) -> str:
    """Parameter name -> group. `x0_injects.i` is layer i's per-layer injection: prelude
    layers 0..n_prelude-1, core n_prelude..n_prelude+n_core-1, coda the rest (the model's
    own indexing: `x0_injects[np_ + i]` for core layer i). `input_norm` is the prelude's
    exit norm (`xn = input_norm(x)` feeds the coda's token carrier AND the loop's entry),
    booked with the prelude. `tul.E_mask` is the vector that REPLACES a dropped token state
    on the coda's input (token-state dropout, `TULSlots.apply_token_dropout`), so it is a
    coda-input parameter, booked with the coda. The fan's epivol reservoir (`tul_fan_epi`)
    holds buffers only and never reaches this function."""
    n = name.replace("_orig_mod.", "")
    top = n.split(".")[0]
    if top == "embed":
        return "embed_input"
    if top in ("prelude", "input_norm"):
        return "prelude"
    if top in ("core", "injection"):
        return "loop_core"
    if top == "coda":
        return "coda"
    if top == "x0_injects":
        i = int(n.split(".")[1])
        return ("prelude" if i < n_prelude else
                "loop_core" if i < n_prelude + n_core else "coda")
    if n.startswith("tul.W_prefix"):
        return "W_prefix"
    if n.startswith("tul.E_mask"):
        return "coda"
    if n.startswith(("tul.E_slot", "tul.W_sent")) or top == "tul_register":
        return "slot_seed_register"
    if top == "tul_cell_norm":
        return "cell_norm"
    if top in ("tul_fan_lsel_head", "tul_fan_lsel_router"):
        return "router_latent_head"
    if top == "tul_pointer":
        return "pointer"
    if top == "tul_spandec":
        return "span_decoder"
    if top in ("lm_mixer", "final_norm"):
        return "readout_mixer_norm"
    raise KeyError(f"parameter {name!r} has no group; add it to group_of")


# ── capture: the live tensor behind every `.detach()` of a graph scalar ────────────────────
class DetachTap(TorchFunctionMode):
    """Records ``(detached, live)`` for every `.detach()` of a 0-dim tensor that requires
    grad. The model exposes each folded loss term as a detached `*_weighted` scalar; this
    recovers the live tensor it came from, with no edit to the model."""

    def __init__(self):
        super().__init__()
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func is torch.Tensor.detach:
            src = args[0]
            if isinstance(src, torch.Tensor) and src.requires_grad and src.dim() == 0:
                self.pairs.append((out, src))
        return out

    def live_of(self, t: torch.Tensor) -> torch.Tensor | None:
        for d, s in self.pairs:
            if d is t:
                return s
        return None


class Taps:
    """Method wraps on ONE model instance (instance attributes shadow the class methods):
    the first `_tul_front` (live, `twin=None`), `_tul_core`, `prefix_project`,
    `_tul_spandec_loss` and `_tul_group_losses` call of a forward. `reset()` per forward."""

    def __init__(self, model):
        self.m = model
        self.reset()
        m = model
        o_front, o_core = m._tul_front, m._tul_core
        o_pp, o_sd, o_gl = m.tul.prefix_project, m._tul_spandec_loss, m._tul_group_losses

        def front(*a, **k):
            r = o_front(*a, **k)
            if k.get("twin") is None and self.prelude_out is None:
                self.prelude_out = r[0]
            return r

        def core(*a, **k):
            r = o_core(*a, **k)
            if self.exit is None:
                self.xn, self.exit = r[0], r[1]
            return r

        def pp(*a, **k):
            r = o_pp(*a, **k)
            if self.written is None:
                self.written = k.get("cells")
            return r

        def sd(h_slots, *a, **k):
            r = o_sd(h_slots, *a, **k)
            self.sd_in.append(h_slots)
            return r

        def gl(*a, **k):
            r = o_gl(*a, **k)
            self.ce_calls.append(r["loss"])
            return r

        m._tul_front, m._tul_core = front, core
        m.tul.prefix_project = pp
        m._tul_spandec_loss, m._tul_group_losses = sd, gl

    def reset(self):
        self.prelude_out = self.xn = self.exit = self.written = None
        self.sd_in: list[torch.Tensor] = []
        self.ce_calls: list[torch.Tensor] = []


def install_shadow_head(model):
    """Rebuild `lm_weight()` from SHADOW leaves (see module docstring). Returns the shadows."""
    hyb = model.embed.hybrid
    from morph.model import embeddings as _emb
    euc = hyb.euc_embed
    if not hasattr(euc, "parametrizations"):
        raise RuntimeError("euc_embed is not parametrised; the shadow rebuild assumes the "
                           "int6 embed_quant parametrisation this config trains with")
    plist = euc.parametrizations.weight
    if len(plist) != 1:
        raise RuntimeError(f"euc_embed.weight has {len(plist)} parametrisations, expected 1")
    s_euc = plist.original.detach().clone().requires_grad_(True)
    s_lor = hyb.lor_embed.space_embed.weight.detach().clone().requires_grad_(True)
    orig_fn = hyb.lm_weight

    def shadow_lm_weight():
        euc_w = plist[0](s_euc)
        lor_w = _emb._log_map_origin(_emb._project_to_hyperboloid(s_lor))
        return torch.cat([euc_w, lor_w], dim=-1)

    with torch.no_grad():
        a, b = orig_fn(), shadow_lm_weight()
    if not torch.equal(a, b):
        raise RuntimeError(f"shadow lm_weight differs from the real one "
                           f"(max abs {(a - b).abs().max().item():.3e}); refusing")
    hyb.lm_weight = shadow_lm_weight
    return [s_euc, s_lor]


def load_model(config: str, ckpt: str, device: str):
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    model, step = load_ckpt(cfg, ckpt, device, tul_rt.model_cfg)
    return cfg, tul_rt, model, step


# ── part: data ───────────────────────────────────────────────────────────────────────────
def part_data(a) -> None:
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    cfg = build_cfg(a.config, [])
    tul_rt = build_tul_runtime(cfg)
    bs = int(cfg.training.batch_size)
    sl = int(cfg.data.seq_len)
    t0 = time.time()
    tr = iter(create_dataloader(cfg.data.tokenizer, cfg.data.dataset, sl, bs, split="train",
                                bag_size=0, tul=tul_rt.data_cfg))
    for _ in range(a.train_skip):
        next(tr)
    print(f"[data] skipped {a.train_skip} train batches in {time.time() - t0:.0f}s", flush=True)
    train = [next(tr) for _ in range(a.train_batches)]
    va = iter(create_dataloader(cfg.data.tokenizer, cfg.data.dataset, sl, bs,
                                split="validation", skip_samples=50_000,
                                tul=tul_rt.val_data_cfg))
    val = [next(va) for _ in range(a.val_batches)]
    os.makedirs(os.path.dirname(os.path.abspath(a.batches_file)), exist_ok=True)
    torch.save({"config": a.config, "train_skip": a.train_skip, "batch_size": bs,
                "train": train, "val": val, "val_skip_docs": 50_000}, a.batches_file)
    print(f"[data] wrote {len(train)} train + {len(val)} val batches of {bs} rows -> "
          f"{a.batches_file} ({time.time() - t0:.0f}s)", flush=True)


# ── part: grad ───────────────────────────────────────────────────────────────────────────
def _gsum(gs, idx):
    """Sum of squares over a group's gradient list (None = no path = 0)."""
    s = 0.0
    for i in idx:
        g = gs[i]
        if g is not None:
            s += float(g.float().pow(2).sum())
    return s


def _gdot(ga, gb, idx):
    s = 0.0
    for i in idx:
        if ga[i] is not None and gb[i] is not None:
            s += float((ga[i].float() * gb[i].float()).sum())
    return s


def find_terms(out: dict, tap: DetachTap, ce: torch.Tensor, model) -> dict[str, torch.Tensor]:
    """The live tensor of every folded term, closure-checked against `out["loss"]`."""
    terms: dict[str, torch.Tensor] = {"token_ce": ce}
    for k in sorted(out):
        if not k.endswith("_weighted") or not torch.is_tensor(out[k]):
            continue
        v = out[k]
        live = v if v.requires_grad else tap.live_of(v)
        if live is None:
            raise RuntimeError(f"`{k}` = {float(v):.6g} is folded into the loss but its live "
                               "tensor was not recorded; the matrix would book it as CE")
        terms[k[: -len("_weighted")]] = live
    total = out["loss"]
    s = sum(float(t.detach().float()) for t in terms.values())
    tv = float(total.detach().float())
    if abs(s - tv) > 1e-4 * max(1.0, abs(tv)):
        raise RuntimeError(f"closure failed: CE + terms = {s:.6f} but out['loss'] = {tv:.6f}; "
                           f"terms {sorted(terms)}")
    # Split the latent-selected loop's folded term into its three summands.
    if "fan_lsel" in terms:
        # The readings `_lsel_finish` stashes on `model._lsel_out["stats"]` are the FIRST
        # detach of each summand (the copies in `out` are a second detach + cast, which the
        # tap cannot trace back), so the split reads the stash.
        tc = model.cfg.tul
        st = model._lsel_out["stats"]
        parts = {}
        for key, w in (("lsel_lat", tc.fan_lsel_lambda), ("lsel_enc", tc.fan_lsel_enc_lambda),
                       ("lsel_router_ce", tc.fan_lsel_router_lambda)):
            live = tap.live_of(st[key])
            if live is None:
                raise RuntimeError(f"fan_lsel split: no live tensor for {key}")
            parts[f"fan_lsel.{key}"] = float(w) * live
        ps = sum(float(p.detach().float()) for p in parts.values())
        fv = float(terms["fan_lsel"].detach().float())
        if abs(ps - fv) > 1e-4 * max(1.0, abs(fv)):
            raise RuntimeError(f"fan_lsel split does not close: {ps:.6f} vs {fv:.6f}")
        del terms["fan_lsel"]
        terms.update(parts)
    return terms


def part_grad(a) -> None:
    device = "cuda"
    cfg, tul_rt, model, step = load_model(a.config, a.ckpt, device)
    data = torch.load(a.batches_file, weights_only=False)
    batches = data["train"][: a.n_batches]
    if len(batches) < a.n_batches:
        raise RuntimeError(f"{a.batches_file} holds {len(batches)} train batches, asked "
                           f"{a.n_batches}")
    shadows = install_shadow_head(model)
    taps = Taps(model)
    model.train()
    model._train_instruments = False           # the trainer's ordinary (non-log) step
    clip_cfg = float(model.cfg.slot_cot_clip)
    # THE HINGE'S PASS. The one-pass gain hinge reads ONE loop pass per forward. This
    # checkpoint (2026-10-08) trained under the pre-e19234c2 rule: the CPU draw was restored
    # every step, so the run read one fixed pass (pass 6 on seed 1, memory
    # gain-hinge-reads-one-fixed-pass). HEAD draws a fresh pass per step from a generator
    # seeded off `torch.initial_seed()`, i.e. a different pass every process. `--gain-pass`
    # pins the pass the run trained with (default 6); -1 keeps HEAD's draw.
    if a.gain_pass >= 0:
        model.cfg.slot_gain_pass_force = int(a.gain_pass)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named] + shadows
    names = [n for n, _ in named] + ["<shadow>.euc", "<shadow>.lor"]
    gidx: dict[str, list[int]] = {g: [] for g in GROUPS}
    np_, nc = int(model.cfg.n_prelude), int(model.cfg.n_core)
    for i, n in enumerate(names):
        g = "lm_head_tied" if n.startswith("<shadow>") else group_of(n, np_, nc)
        gidx[g].append(i)
    group_numel = {g: int(sum(params[i].numel() for i in ix)) for g, ix in gidx.items()}
    print(f"[grad] step {step}; {len(params)} tensors in {len(GROUPS)} groups; "
          f"slot_cot_clip {clip_cfg}", flush=True)

    per_batch = []
    for bi, (x, y, layout) in enumerate(batches):
        t0 = time.time()
        x, y, layout = x.to(device), y.to(device), layout.to(device)
        torch.manual_seed(1000 + bi)
        torch.cuda.manual_seed(1000 + bi)
        taps.reset()
        model.cfg.slot_cot_clip = clip_cfg       # hooks are registered as in training
        tap = DetachTap()
        with torch.autocast("cuda", dtype=torch.bfloat16), tap:
            out = model(x, labels=y, slot_layout=layout)
        if len(taps.ce_calls) != 1:
            raise RuntimeError(f"_tul_group_losses ran {len(taps.ce_calls)} times; expected 1")
        terms = find_terms(out, tap, taps.ce_calls[0], model)
        winner = model._lsel_out["winner"]                       # [B, S]
        B, S = winner.shape
        M = int(model.cfg.tul.fan_k)
        exit_t, written, pre = taps.exit, taps.written, taps.prelude_out
        sd_in = taps.sd_in[0] if taps.sd_in else None
        if exit_t is None or written is None or pre is None or sd_in is None:
            raise RuntimeError("a path tensor was not captured "
                               f"(exit {exit_t is not None}, written {written is not None}, "
                               f"prelude {pre is not None}, spandec {sd_in is not None})")
        path_inputs = [pre, exit_t, written, sd_in]
        # winner mask on the exit's [B, S*M, ...] cell axis
        oh = F.one_hot(winner, M).to(exit_t.dtype).reshape(B, S * M)
        wmask = oh.view(B, S * M, *([1] * (exit_t.dim() - 2)))
        smask = layout.slot_mask.view(B, -1, *([1] * (pre.dim() - 2))).to(pre.dtype)
        # the routed stack `prefix_project` reads is [B, S, M, ...]: the winner cell is the
        # only live one. A LOSER cell is an exact zero that the write multiplies by 0, and
        # the coda's norms see an exactly-zero carrier there, so the gradient AT a loser
        # cell is huge (1e9 measured) and goes nowhere: report the winner cell's alone.
        if written.shape[:3] != (B, S, M):
            raise RuntimeError(f"written cells {tuple(written.shape)} are not [B, S, M, ...]")
        wcell = oh.view(B, S, M, *([1] * (written.dim() - 3)))

        # (1) the trainer's clipped total, (2) the unclipped total (the linear reference)
        g_clip = torch.autograd.grad(out["loss"], params, retain_graph=True, allow_unused=True)
        model.cfg.slot_cot_clip = 0.0
        g_lin = torch.autograd.grad(out["loss"], params, retain_graph=True, allow_unused=True)
        rec = {"batch": bi, "loss": float(out["loss"].detach()),
               "gain_pass": getattr(model, "_gain_pass_last", None),
               "gain_est": float(out["gain_est"]) if "gain_est" in out else None,
               "term_value": {k: float(v.detach().float()) for k, v in terms.items()},
               "groups": {}, "paths": {}, "clip_effect": {}, "vjp_exit": {}}
        g_sum = [None] * len(params)
        tg = {}
        for tname, tval in terms.items():
            gg = torch.autograd.grad(tval, params + path_inputs, retain_graph=True,
                                     allow_unused=True)
            gp, gpath = gg[: len(params)], gg[len(params):]
            for i, g in enumerate(gp):
                if g is not None:
                    g_sum[i] = g.float() if g_sum[i] is None else g_sum[i] + g.float()
            tg[tname] = {grp: {"sq": _gsum(gp, ix), "dot_lin": _gdot(gp, g_lin, ix)}
                         for grp, ix in gidx.items()}
            gpre, gexit, gwr, gsd = gpath
            z = lambda t: 0.0 if t is None else float(t.float().norm())  # noqa: E731
            rec["paths"][tname] = {
                "prelude_tokens": 0.0 if gpre is None else float((gpre.float() * (1 - smask)).norm()),
                "prelude_slots": 0.0 if gpre is None else float((gpre.float() * smask).norm()),
                "exit_all": z(gexit),
                "exit_winner": 0.0 if gexit is None else float((gexit.float() * wmask).norm()),
                "exit_losers": 0.0 if gexit is None else float((gexit.float() * (1 - wmask)).norm()),
                "written_winner_cell": 0.0 if gwr is None else float((gwr.float() * wcell).norm()),
                "spandec_input": z(gsd),
            }
            if tname in ("token_ce", "spandec") and gexit is not None:
                # VJP from the exit cells back to every group, winner / losers apart
                vj = {}
                for part, msk in (("winner", wmask), ("losers", 1 - wmask)):
                    gv = torch.autograd.grad(exit_t, params, grad_outputs=gexit * msk,
                                             retain_graph=True, allow_unused=True)
                    vj[part] = {grp: _gsum(gv, ix) for grp, ix in gidx.items()}
                    vj[part + "_dot_term"] = {grp: _gdot(gv, gp, ix) for grp, ix in gidx.items()}
                    del gv
                rec["vjp_exit"][tname] = vj
            del gg, gp, gpath
        for grp, ix in gidx.items():
            lin_sq = _gsum(g_lin, ix)
            clip_sq = _gsum(g_clip, ix)
            diff_sq = 0.0
            for i in ix:
                if g_sum[i] is not None or g_lin[i] is not None:
                    a_ = g_sum[i] if g_sum[i] is not None else torch.zeros_like(params[i], dtype=torch.float32)
                    b_ = g_lin[i].float() if g_lin[i] is not None else torch.zeros_like(a_)
                    diff_sq += float((a_ - b_).pow(2).sum())
            rec["groups"][grp] = {"lin_sq": lin_sq, "clip_sq": clip_sq,
                                  "clip_dot_lin": _gdot(g_clip, g_lin, ix),
                                  "additivity_rel": math.sqrt(diff_sq) / max(math.sqrt(lin_sq), 1e-30),
                                  "terms": {t: tg[t][grp] for t in tg}}
        per_batch.append(rec)
        del g_clip, g_lin, g_sum, out, terms, tap
        model.cfg.slot_cot_clip = clip_cfg
        torch.cuda.empty_cache()
        print(f"[grad] batch {bi}: loss {rec['loss']:.4f} terms "
              + ", ".join(f"{k}={v:.4g}" for k, v in rec["term_value"].items())
              + f" ({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)",
              flush=True)
    result = {"config": a.config, "ckpt": a.ckpt, "step": step, "n_batches": len(per_batch),
              "batch_rows": int(batches[0][0].shape[0]), "train_skip": data["train_skip"],
              "slot_cot_clip": clip_cfg, "gain_pass_force": a.gain_pass,
              "group_numel": group_numel, "groups": list(GROUPS),
              "per_batch": per_batch, "summary": summarise_grad(per_batch)}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[grad] wrote {a.out}", flush=True)


def _ms(v):
    v = np.asarray(v, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def summarise_grad(per_batch: list[dict]) -> dict:
    """Per (term, group): norm, norm share, projection share; mean and spread over batches."""
    terms = list(per_batch[0]["term_value"])
    out = {"term_value": {t: _ms([b["term_value"][t] for b in per_batch]) for t in terms},
           "matrix": {}, "clip_effect": {}, "additivity_rel_max": {}, "paths": {},
           "vjp_exit": {}}
    for grp in GROUPS:
        col = {}
        for t in terms:
            norms, nshare, pshare = [], [], []
            for b in per_batch:
                g = b["groups"][grp]
                tn = [math.sqrt(g["terms"][u]["sq"]) for u in terms]
                tot = sum(tn)
                norms.append(math.sqrt(g["terms"][t]["sq"]))
                nshare.append(norms[-1] / tot if tot > 0 else 0.0)
                pshare.append(g["terms"][t]["dot_lin"] / g["lin_sq"] if g["lin_sq"] > 0 else 0.0)
            col[t] = {"norm": _ms(norms), "norm_share": _ms(nshare), "proj_share": _ms(pshare)}
        col["_total_norm_lin"] = _ms([math.sqrt(b["groups"][grp]["lin_sq"]) for b in per_batch])
        out["matrix"][grp] = col
        out["additivity_rel_max"][grp] = max(b["groups"][grp]["additivity_rel"] for b in per_batch)
        ce = []
        for b in per_batch:
            g = b["groups"][grp]
            cs, ls = math.sqrt(g["clip_sq"]), math.sqrt(g["lin_sq"])
            ce.append({"ratio": cs / ls if ls > 0 else float("nan"),
                       "cos": g["clip_dot_lin"] / (cs * ls) if cs * ls > 0 else float("nan")})
        out["clip_effect"][grp] = {"clipped_over_unclipped_norm": _ms([c["ratio"] for c in ce]),
                                   "cos_clipped_unclipped": _ms([c["cos"] for c in ce])}
    for t in terms:
        keys = per_batch[0]["paths"][t].keys()
        out["paths"][t] = {k: _ms([b["paths"][t][k] for b in per_batch]) for k in keys}
    for t in per_batch[0]["vjp_exit"]:
        d = {}
        for grp in GROUPS:
            w = [math.sqrt(b["vjp_exit"][t]["winner"][grp]) for b in per_batch]
            l_ = [math.sqrt(b["vjp_exit"][t]["losers"][grp]) for b in per_batch]
            full = [math.sqrt(b["groups"][grp]["terms"][t]["sq"]) for b in per_batch]
            # fraction of the term's gradient in this group that ENTERS via the exit:
            # <vjp_winner + vjp_losers, g_term> / |g_term|^2 (1 = every bit of it)
            via = [((b["vjp_exit"][t]["winner_dot_term"][grp]
                     + b["vjp_exit"][t]["losers_dot_term"][grp])
                    / b["groups"][grp]["terms"][t]["sq"])
                   if b["groups"][grp]["terms"][t]["sq"] > 0 else 0.0 for b in per_batch]
            d[grp] = {"via_winner_norm": _ms(w), "via_losers_norm": _ms(l_),
                      "term_norm": _ms(full), "frac_via_exit": _ms(via)}
        out["vjp_exit"][t] = d
    return out


# ── part: hinge ──────────────────────────────────────────────────────────────────────────
def part_hinge(a) -> None:
    """The gain hinge's reading at EVERY loop pass, train mode, on the first ``--n-batches``
    train batches: the per-pass gain estimate and the hinge penalty it would charge. Says
    which passes fire, so the matrix's hinge row can be read for the pass the run trained
    with and for HEAD's per-step draw. Forward only; no backward."""
    device = "cuda"
    cfg, tul_rt, model, step = load_model(a.config, a.ckpt, device)
    data = torch.load(a.batches_file, weights_only=False)
    model.train()
    model._train_instruments = False
    max_d = int(model.cfg.tul.slot_max_depth or model.cfg.max_depth)
    rows = []
    for bi, (x, y, layout) in enumerate(data["train"][: a.n_batches]):
        x, y, layout = x.to(device), y.to(device), layout.to(device)
        for t in range(max_d):
            model.cfg.slot_gain_pass_force = t
            torch.manual_seed(1000 + bi)
            torch.cuda.manual_seed(1000 + bi)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, labels=y, slot_layout=layout)
            except ValueError as e:
                if "not a grad pass of this step" not in str(e):
                    raise
                rows.append({"batch": bi, "pass": t, "reached": False})
                continue
            rows.append({"batch": bi, "pass": t, "reached": True,
                         "gain_est": float(out["gain_est"]),
                         "gain_est_max": float(out["gain_est_max"]),
                         "penalty": float(out["gain_reg_weighted"])})
            del out
        print(f"[hinge] batch {bi}: " + ", ".join(
            f"t{r['pass']} {r['gain_est']:.3f}/{r['penalty']:.2f}" for r in rows
            if r["batch"] == bi and r["reached"]), flush=True)
    with open(a.out, "w") as f:
        json.dump({"config": a.config, "ckpt": a.ckpt, "step": step,
                   "target": float(model.cfg.slot_gain_target),
                   "lambda": float(model.cfg.slot_gain_lambda), "rows": rows}, f, indent=1)
    print(f"[hinge] wrote {a.out}", flush=True)


# ── part: worth ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _ce_rows(model, x, y, layout, plan_mode="normal"):
    """Per-row (sum of token CE, token count) from the label-free forward's logits (the
    pointer mixture's log-probs on this model), over token positions with a label."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = model.tul_forward_ablated(x, None, layout, plan_mode=plan_mode)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = y.clone()
    keep = (~layout.slot_mask) & (lab >= 0)
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    ce = ce * keep
    return ce.sum(1).double().cpu().numpy(), keep.sum(1).double().cpu().numpy()


def part_worth(a) -> None:
    device = "cuda"
    cfg, tul_rt, model, step = load_model(a.config, a.ckpt, device)
    model.eval()
    data = torch.load(a.batches_file, weights_only=False)
    val = [(x.to(device), y.to(device), lay.to(device)) for x, y, lay in data["val"][: a.n_val]]
    tc = model.cfg.tul
    rows = sum(int(x.shape[0]) for x, _, _ in val)
    print(f"[worth] step {step}; {len(val)} val batches, {rows} rows", flush=True)

    # anchor: the full model's per-token CE on batch 0 vs the labelled forward's ce_tokens
    s0, c0 = _ce_rows(model, *val[0])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lab_out = model.tul_forward_ablated(val[0][0], val[0][1], val[0][2], plan_mode="normal")
    anchor = {"per_token_ce_batch0": float(s0.sum() / c0.sum()),
              "labelled_ce_tokens_batch0": float(lab_out["ce_tokens"])}
    print(f"[worth] anchor {anchor}", flush=True)

    def run(tag, **kw):
        t0 = time.time()
        ss, cc = [], []
        for x, y, lay in val:
            s, c = _ce_rows(model, x, y, lay, **kw)
            ss.append(s)
            cc.append(c)
        s, c = np.concatenate(ss), np.concatenate(cc)
        print(f"[worth] {tag}: CE {s.sum() / c.sum():.4f} ({time.time() - t0:.0f}s)", flush=True)
        return s, c

    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    arms["full"] = run("full")
    arms["write_zeroed"] = run("write_zeroed", plan_mode="zero")
    arms["write_shuffled"] = run("write_shuffled", plan_mode="shuffle")
    orig_mean, orig_max = int(tc.slot_mean_depth), int(tc.slot_max_depth)
    train_depth = orig_mean or int(model.cfg.mean_depth)
    for d in a.depths:
        tc.slot_mean_depth = d
        tc.slot_max_depth = max(d, orig_max or int(model.cfg.max_depth))
        arms[f"depth_{d}"] = run(f"depth_{d}")
    tc.slot_mean_depth, tc.slot_max_depth = orig_mean, orig_max
    orig_cti = tc.coda_token_input
    tc.coda_token_input = "embed"
    arms["coda_tokens_read_embedding"] = run("coda_tokens_read_embedding")
    tc.coda_token_input = orig_cti
    g = model.tul_pointer.gate_norm_head
    saved = (g.weight.detach().clone(), g.bias.detach().clone())
    with torch.no_grad():
        g.weight.zero_()
        g.bias.fill_(-1e4)
        g.bias[0] = 0.0
    arms["pointer_off"] = run("pointer_off")
    with torch.no_grad():
        g.weight.copy_(saved[0])
        g.bias.copy_(saved[1])
    arms["full_again"] = run("full_again")     # the restores left the model unchanged

    s_full, c_full = arms["full"]
    table = {}
    for k, (s, c) in arms.items():
        if not np.array_equal(c, c_full):
            raise RuntimeError(f"{k}: token counts differ from the full arm; rows not paired")
        table[k] = {"ce": float(s.sum() / c.sum()),
                    "delta_vs_full": paired_bootstrap_ci(s, s_full, c)}
    if 1 in a.depths and train_depth in a.depths:
        table["K1-K6"] = paired_bootstrap_ci(arms["depth_1"][0], arms[f"depth_{train_depth}"][0],
                                             c_full)
    result = {"config": a.config, "ckpt": a.ckpt, "step": step, "rows": rows,
              "val_skip_docs": data["val_skip_docs"], "train_eval_depth": train_depth,
              "anchor": anchor, "arms": table}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[worth] wrote {a.out}", flush=True)


def part_report(a) -> None:
    """Markdown tables from grad.json, worth.json and hinge.json in ``--results-dir``,
    printed to stdout (the README's tables are this output)."""
    d = a.results_dir
    g = json.load(open(os.path.join(d, "grad.json")))
    w = json.load(open(os.path.join(d, "worth.json")))
    h = json.load(open(os.path.join(d, "hinge.json")))
    s = g["summary"]
    terms = list(s["term_value"])
    p = print
    p(f"### Gradient credit matrix ({g['n_batches']} train batches x {g['batch_rows']} rows, "
      f"train skip {g['train_skip']} batches, hinge pass pinned to {g['gain_pass_force']})\n")
    p("Term values (mean over batches): " + ", ".join(
        f"`{t}` {s['term_value'][t]['mean']:.4g}" for t in terms) + "\n")
    p("Entry = projection share mean (spread = sd over batches) / norm share mean. "
      "Projection shares sum to 1 per row; norm shares sum to 1 per row. Rows with no "
      "gradient from a term show `.`.\n")
    p("| group | params | total norm | " + " | ".join(f"`{t}`" for t in terms) + " |")
    p("|---|---|---|" + "---|" * len(terms))
    for grp in GROUPS:
        col = s["matrix"][grp]
        cells = []
        for t in terms:
            v = col[t]
            cells.append("." if v["norm"]["mean"] == 0 else
                         f"{v['proj_share']['mean']:+.2f} ({v['proj_share']['std']:.2f}) / "
                         f"{v['norm_share']['mean']:.2f}")
        p(f"| {grp} | {g['group_numel'][grp] / 1e6:.2f}M | "
          f"{col['_total_norm_lin']['mean']:.3g} | " + " | ".join(cells) + " |")
    p("\nAdditivity (max over batches of |sum of term grads - total| / |total|, clip off): "
      + ", ".join(f"{k} {v:.1e}" for k, v in s["additivity_rel_max"].items()))
    ce = s["clip_effect"]
    p("\nCotangent clip (slot_cot_clip 4.0), clipped vs unclipped total, min over batches: "
      + ", ".join(f"{k} ratio {v['clipped_over_unclipped_norm']['min']:.4f} cos "
                  f"{v['cos_clipped_unclipped']['min']:.5f}" for k, v in ce.items()))
    p("\n### Gradient arriving at each tensor (L2 norm, mean (sd) over batches)\n")
    keys = ["prelude_tokens", "prelude_slots", "exit_all", "exit_winner", "exit_losers",
            "written_winner_cell", "spandec_input"]
    p("| term | " + " | ".join(keys) + " |")
    p("|---|" + "---|" * len(keys))
    for t in terms:
        pt = s["paths"][t]
        p(f"| `{t}` | " + " | ".join(
            f"{pt[k]['mean']:.3g} ({pt[k]['std']:.1g})" if pt[k]["mean"] > 0 else "0"
            for k in keys) + " |")
    p("\n### VJP from the loop exit back to the parameters\n")
    p("frac = <VJP(exit -> group), g_term> / |g_term|^2: the part of the term's gradient in "
      "the group that enters through the exit cells (1 = all of it).\n")
    p("| term | group | via winner cell | via loser cells | term norm | frac via exit |")
    p("|---|---|---|---|---|---|")
    for t, dd in s["vjp_exit"].items():
        for grp, v in dd.items():
            if v["term_norm"]["mean"] > 0:
                p(f"| `{t}` | {grp} | {v['via_winner_norm']['mean']:.3g} | "
                  f"{v['via_losers_norm']['mean']:.3g} | {v['term_norm']['mean']:.3g} | "
                  f"{v['frac_via_exit']['mean']:.4f} |")
    p(f"\n### Worth by ablation ({w['rows']} trainer-val rows, eval, paired)\n")
    p(f"Anchor: per-token CE of batch 0 {w['anchor']['per_token_ce_batch0']:.5f}, the labelled "
      f"forward's ce_tokens {w['anchor']['labelled_ce_tokens_batch0']:.5f}.\n")
    p("| arm | CE | delta vs full [95% CI] |")
    p("|---|---|---|")
    for k, v in w["arms"].items():
        if k == "K1-K6":
            continue
        dlt = v["delta_vs_full"]
        p(f"| {k} | {v['ce']:.4f} | {dlt['point']:+.4f} [{dlt['lo']:+.4f}, {dlt['hi']:+.4f}] |")
    if "K1-K6" in w["arms"]:
        k = w["arms"]["K1-K6"]
        p(f"\nK1-K6 = {k['point']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}] "
          f"(eval depth {w['train_eval_depth']}).")
    p(f"\n### Gain hinge reading per loop pass (train mode, target {h['target']}, "
      f"lambda {h['lambda']})\n")
    passes = sorted({r["pass"] for r in h["rows"]})
    p("| batch | " + " | ".join(f"pass {t}" for t in passes) + " |")
    p("|---|" + "---|" * len(passes))
    for b in sorted({r["batch"] for r in h["rows"]}):
        rr = {r["pass"]: r for r in h["rows"] if r["batch"] == b}
        p(f"| {b} | " + " | ".join(
            (f"{rr[t]['gain_est']:.3f} / {rr[t]['penalty']:.2f}" if rr[t]["reached"]
             else "not reached") for t in passes) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["data", "grad", "worth", "hinge", "report"])
    ap.add_argument("--config", default="lxtul_pointer")
    ap.add_argument("--ckpt")
    ap.add_argument("--batches-file", help="data/grad/worth/hinge: the packed batches")
    ap.add_argument("--results-dir", help="report: the dir holding grad/worth/hinge.json")
    ap.add_argument("--train-skip", type=int, default=5000)
    ap.add_argument("--train-batches", type=int, default=8)
    ap.add_argument("--val-batches", type=int, default=80)
    ap.add_argument("--n-batches", type=int, default=8, help="grad: train batches to use")
    ap.add_argument("--n-val", type=int, default=80, help="worth: val batches to use")
    ap.add_argument("--depths", default="1,2,3,6,8")
    ap.add_argument("--gain-pass", type=int, default=6,
                    help="grad/hinge: pin the one-pass gain hinge to this loop pass (-1: HEAD's "
                         "per-step draw); see part_grad")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.depths = [int(d) for d in a.depths.split(",")]
    if a.part == "data":
        part_data(a)
    elif a.part == "grad":
        part_grad(a)
    elif a.part == "hinge":
        part_hinge(a)
    elif a.part == "report":
        part_report(a)
    else:
        part_worth(a)


if __name__ == "__main__":
    main()
