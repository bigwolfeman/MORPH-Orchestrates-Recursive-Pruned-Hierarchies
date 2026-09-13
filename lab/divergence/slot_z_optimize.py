"""How much is a GOOD slot state worth to the coda? (Latent Thought LM, Kong et al. 2025,
used as an INSTRUMENT rather than as a training method.)

The slot loop writes one state `z` per span through `tul.W_prefix` into the span's
`prefix_k` coda cells; the coda reads it. The forward instruments say the core barely
moves that state and the K-curves are flat; the gradient probe
(`lab/divergence/slot_gradient_probe.py`) says the token CE puts about 1 % of the
prelude's gradient into the loop. Before asking HOW the loop should build z, this asks
what a good z is WORTH:

    freeze the trained model, treat z as a free variable, and minimise the trainer's
    token CE over z alone by gradient descent.

If a gradient-optimised z beats the loop's z by a lot, the coda CAN use z and the loop is
a poor inferencer. If the gain is small, the coda cannot use z and no loop will fix it.

WHAT THIS NUMBER IS AND IS NOT (read before quoting it).  The optimiser sees the CE of
the tokens the coda predicts AFTER slot i, and z_i is fitted on exactly those tokens. A
causal inferencer cannot see them: the slot's own span is all the loop is given. So
`ce_zopt` is an ORACLE upper bound on what ANY slot-state producer could deliver to this
frozen coda through this frozen write. It bounds the CODA's capacity to use z. It does
not say the loop could reach it.

## The split point

`_forward_tul` (morph/model/transformer.py) runs, for a slot-loop model:

    x, x0, bigram = _tul_front(...)
    xn, h_slots, depths, g_traj, db_traj, gain, ... = _tul_core(...)  # <- z is h_slots
    [_tul_cond_apply] [mux loss] [gate] [detach_z] [_tul_plan_ablate]
    values, pos = tul.prefix_project(h_slots, layout, L)             # <- the prefix write
    x_coda = scatter_positions(base, pos, values)
    [tul.unpack(h_slots) when tul.bcast]                             # <- the other reader
    xh = _back_region(x_coda, ...)
    groups = _tul_group_losses(xh, labels, layout)                   # <- the token CE

z is `h_slots` as `prefix_project` receives it: `[B, S, n, C]` on an HC model (n = 4
Cayley streams, C = d_model) — the FULL stream tensor, nothing collapsed. The probe
substitutes at the `_tul_core` RETURN and refuses to run when a think-once conditioning
stack, a gate or `tul.detach_z` sits between the two points, so that on every arm it does
run, the tensor it substitutes IS the tensor `prefix_project` receives. That identity is
asserted on every replay (`prefix_project` must receive the substituted object itself).

Only the front and the core are cached; EVERYTHING downstream is the shipped code, called
through `_forward_single`, so the prefix write, the bcast unpack, the token-state dropout
seam, the TG masks, the coda and the weighted CE are the model's own, not a copy.

## The objective mask

The objective is `_tul_group_losses(...)["loss"]`: the trainer's ONE weighted CE, before
the sigreg / gain / mux terms `_forward_tul` adds to its copy of the dict. That is the
§5 double label with `tul.emit_weight` on the slot's emitting cell, `tul.plast_weight` on
the span's last token, 1.0 elsewhere and −100 pads ignored. On these arms
`emit_weight = 0`, so the slot cells carry no loss of their own and the whole objective
lives at TOKEN positions.

## Mode

`model.eval()`: dropout OFF, token-state dropout OFF, the eval depth branch. This is an
instrument on a FIXED function — a dropout mask would make each optimisation step a
different objective. Depth is forced to `--depth` for every slot
(`tul.slot_depth_fixed`), and `model.slot_gain_lambda` is set to 0 because the hinge
applies the core step twice more (it is a probe; it does not change h).

Usage:
  python lab/divergence/slot_z_optimize.py \
      --ckpt LABEL=CONFIG=PATH --rows 12 --batch 3 --depth 6 --steps 200 \
      --lrs 1e-2,3e-3 --out .../LABEL.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from core_anatomy import eff_rank  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")


# ── the split ─────────────────────────────────────────────────────────────────


class ZSplit:
    """Split a slot-loop forward at z: record once, then re-run only what reads z.

    Patches four bound methods for the lifetime of the context:

    * ``_tul_front`` / ``_tul_core`` — in ``replay`` they return the CACHED (detached)
      tensors instead of recomputing, with ``h_slots`` replaced by the substituted z.
      Detached on purpose: the prelude and the core are frozen inputs here, so the only
      leaf the CE can reach is z.
    * ``_tul_group_losses`` — a spy that keeps the returned dict and the coda state
      ``xh`` it was scored from, and can force ``want_groups`` off (the per-group CEs are
      three extra full-vocab passes, and the optimisation does not read them).
    * ``TULSlots.prefix_project`` — asserts that the tensor reaching the prefix write IS
      the substituted object. This is what makes "the split point is right" a checked
      claim rather than a reading of the source.
    """

    def __init__(self, model, amp: bool = True):
        self.model = model
        self.amp = amp
        self.st: dict = {"mode": "off", "z": None, "front": None, "core": None,
                         "groups": None, "xh": None, "h0": None, "plan_mode": "normal",
                         "want_groups": None, "core0": 0}
        self._real: dict = {}
        self._hooks: list = []

    # -- context -------------------------------------------------------------
    def __enter__(self):
        m, st = self.model, self.st
        self._real = {"front": m._tul_front, "core": m._tul_core,
                      "groups": m._tul_group_losses, "prefix": m.tul.prefix_project}

        def front(*a, **k):
            if st["mode"] == "replay":
                return st["front"]
            out = self._real["front"](*a, **k)
            st["front"] = tuple(None if t is None else t.detach() for t in out)
            return out

        def core(*a, **k):
            # ARITY-TOLERANT on purpose: `_tul_core`'s return tuple grows as arms are added
            # (6 entries, then 7 with `tul.mux_every_pass`). The entries past `db_traj` are
            # TRAINING-side — the gain hinge's penalty, the per-pass MUX masks — and this
            # probe replays an EVAL forward that must take neither branch, so they replay
            # as None while the arity stays whatever the live call returned.
            if st["mode"] == "replay":
                xn, _h, depths, g_traj, db_traj, n_extra = st["core"]
                return (xn, st["z"], depths, g_traj, db_traj) + (None,) * n_extra
            out = self._real["core"](*a, **k)
            xn, h, depths, g_traj, db_traj = out[:5]
            st["core"] = (xn.detach(), h.detach(), depths,
                          None if g_traj is None else g_traj.detach(),
                          None if db_traj is None else [t.detach() for t in db_traj],
                          len(out) - 5)
            return out

        def groups(x, labels, layout, want_groups=True):
            if st["want_groups"] is not None:
                want_groups = st["want_groups"]
            r = self._real["groups"](x, labels, layout, want_groups=want_groups)
            st["groups"], st["xh"] = r, x
            return r

        def prefix(h_slots, layout, l_total, cells=None):
            if (st["mode"] == "replay" and st["plan_mode"] == "normal"
                    and h_slots is not st["z"]):
                raise RuntimeError(
                    "prefix_project did not receive the substituted z — something "
                    "between the _tul_core return and the prefix write transformed it, "
                    "so the split point is NOT where this probe assumes it is")
            return self._real["prefix"](h_slots, layout, l_total, cells=cells)

        m._tul_front, m._tul_core, m._tul_group_losses = front, core, groups
        m.tul.prefix_project = prefix

        def h0_hook(_mod, _args, out):
            st["h0"] = out.detach()
        self._hooks.append(m.core_init.register_forward_hook(h0_hook))

        def core0_hook(_mod, _args):
            st["core0"] += 1
        self._hooks.append(m.core[0].register_forward_pre_hook(core0_hook))
        return self

    def __exit__(self, *exc):
        m = self.model
        m._tul_front, m._tul_core = self._real["front"], self._real["core"]
        m._tul_group_losses = self._real["groups"]
        m.tul.prefix_project = self._real["prefix"]
        for h in self._hooks:
            h.remove()
        self._hooks = []
        return False

    # -- the two calls -------------------------------------------------------
    def _run(self, inp, labels, layout, plan_mode, want_groups, grad):
        self.st["plan_mode"] = plan_mode
        self.st["want_groups"] = want_groups
        gctx = contextlib.nullcontext() if grad else torch.no_grad()
        dev = inp.device.type
        with gctx, torch.autocast(dev, dtype=torch.bfloat16,
                                  enabled=self.amp and dev == "cuda"):
            out = self.model._forward_single(inp, labels=labels, slot_layout=layout,
                                             _plan_nats=False, _plan_mode=plan_mode)
        return out

    def record(self, inp, labels, layout, want_groups=True):
        """The trained forward. Returns (out, z, h0)."""
        self.st["mode"] = "record"
        self.st["core0"] = 0
        out = self._run(inp, labels, layout, "normal", want_groups, grad=False)
        return out, self.st["core"][1], self.st["h0"]

    def replay(self, inp, labels, layout, z, plan_mode="normal", want_groups=False,
               grad=False):
        """Re-run ONLY what reads z. `z` is substituted at the `_tul_core` return."""
        if self.st["core"] is None:
            raise RuntimeError("replay() before record()")
        self.st["mode"] = "replay"
        self.st["z"] = z
        before = self.st["core0"]
        out = self._run(inp, labels, layout, plan_mode, want_groups, grad)
        if self.st["core0"] != before:
            raise RuntimeError("a core block ran during replay — the core was not skipped")
        return out

    @property
    def token_ce(self):
        return self.st["groups"]["loss"]

    @property
    def xh(self):
        return self.st["xh"]


def guard_split_point(model) -> None:
    """Refuse the run when something sits between the `_tul_core` return and the write."""
    tc = model.cfg.tul
    if getattr(model, "tul_cond", None) is not None:
        raise SystemExit("tul_cond (think-once conditioning) transforms h_slots AFTER "
                         "_tul_core; the substituted z would be the PRE-conditioning "
                         "state. Not supported by this probe.")
    if getattr(model, "tul_gate", None) is not None:
        raise SystemExit("tul.gate rewrites h_slots between the core and the write.")
    if bool(tc.detach_z):
        raise SystemExit("tul.detach_z cuts the token CE off from z: nothing to optimise.")
    if getattr(model, "fm_planner", None) is not None:
        raise SystemExit("an FM planner takes a different branch of _forward_tul.")


# ── position buckets ──────────────────────────────────────────────────────────


def buckets(layout, head_n: int) -> dict:
    """Boolean [B, L] masks partitioning the TOKEN positions.

    * ``before_slot0`` — tokens of span 0. Causal attention puts them BEFORE every slot
      cell, so no z can reach them: their CE is the causality control and must not move.
    * ``head`` — the first ``head_n`` tokens of a span that HAS a preceding slot.
    * ``tail`` — the rest of such a span's tokens.
    * ``dump`` — tokens past the last slot (``bag_id == max_slots``).
    """
    tok = ~layout.slot_mask
    bag = layout.bag_id
    S = layout.max_slots
    big = torch.iinfo(torch.long).max // 4
    c = tok.long().cumsum(1) - tok.long()                       # tokens strictly before p
    c = torch.where(tok, c, torch.full_like(c, big))
    B = bag.shape[0]
    first = torch.full((B, S + 1), big, dtype=torch.long, device=bag.device)
    first.scatter_reduce_(1, bag.clamp(max=S), c, reduce="amin", include_self=True)
    rank = c - first.gather(1, bag.clamp(max=S))
    has_slot = tok & (bag >= 1) & (bag < S)
    return {"before_slot0": tok & (bag == 0),
            "head": has_slot & (rank < head_n),
            "tail": has_slot & (rank >= head_n),
            "dump": tok & (bag >= S)}


def bucket_ce(real_groups, xh, labels, layout, masks) -> dict:
    """CE and target count per bucket, through the SHIPPED weighted-CE code."""
    out = {}
    with torch.no_grad():
        for name, m in masks.items():
            lab = torch.where(m, labels, torch.full_like(labels, -100))
            r = real_groups(xh, lab, layout, want_groups=False)
            out[name] = (float(r["loss"]), float(r["n_targets"]))
    return out


# ── z statistics ──────────────────────────────────────────────────────────────


def z_stats(z, valid) -> dict:
    """Per-slot norm (mean over valid slots) and participation rank of the stream mean."""
    zf = z.float()
    v = valid
    per = zf.flatten(2).norm(dim=2)[v]
    return {"norm_mean": float(per.mean()), "norm_max": float(per.max()),
            "eff_rank": eff_rank(zf[v].unsqueeze(0)), "n_valid": int(v.sum())}


def z_cos(a, b, valid) -> float:
    x = a.float().flatten(2)[valid]
    y = b.float().flatten(2)[valid]
    return float(torch.nn.functional.cosine_similarity(x, y, dim=1).mean())


# ── the optimisation ──────────────────────────────────────────────────────────


def optimise_z(split, inp, labels, layout, z0, lr: float, steps: int, valid,
               every: int = 10) -> tuple[torch.Tensor, list]:
    """Adam on z alone. Returns (final z in the model's dtype, CE curve).

    An fp32 MASTER copy is optimised and cast to the core's dtype on every call, so the
    model sees exactly the dtype the recorded forward produced (bf16 under autocast)
    while Adam's moments stay fp32. Pad / invalid slots are frozen by zeroing their
    gradient: Adam's update of a coordinate whose gradient is exactly 0 at every step is
    exactly 0, which the test checks rather than assumes.
    """
    dtype = z0.dtype
    m = valid.reshape(*valid.shape, *([1] * (z0.dim() - 2))).to(torch.float32)
    zm = z0.detach().float().clone().requires_grad_(True)
    opt = torch.optim.Adam([zm], lr=lr)
    curve = []
    for s in range(steps):
        split.replay(inp, labels, layout, zm.to(dtype), want_groups=False, grad=True)
        ce = split.token_ce
        if s % every == 0:
            curve.append((s, float(ce.detach())))
        opt.zero_grad(set_to_none=True)
        ce.backward()
        zm.grad.mul_(m)
        opt.step()
        del ce
    return zm.detach().to(dtype), curve


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lrs", default="1e-2,3e-3")
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    label, config, path = a.ckpt.split("=", 2)
    lrs = [float(x) for x in a.lrs.split(",")]

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_z_optimize needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false)")
    model, step = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)
    guard_split_point(model)

    tc = model.cfg.tul
    notes = {
        "slot_depth_fixed": [int(tc.slot_depth_fixed), a.depth],
        "slot_gain_lambda": [float(model.cfg.slot_gain_lambda), 0.0],
        "mode": "eval (dropout OFF, token-state dropout OFF) — an instrument on a fixed "
                "function",
        "objective": "_tul_group_losses(...)['loss'] — the trainer's ONE weighted CE "
                     f"(emit_weight {float(tc.emit_weight)}, plast_weight "
                     f"{float(tc.plast_weight)}, -100 pads), BEFORE the sigreg / gain / "
                     "mux terms _forward_tul adds",
        "split_point": "the h_slots tensor _tul_core returns == the tensor "
                       "prefix_project receives (asserted on every replay)",
        "plan_nats": "off (the §7.2 slots-removed pass is a second coda call this probe "
                     "does not read)",
        "autocast": "bf16, as the trainer; z is optimised as an fp32 master and cast",
        "oracle": "z is fitted on the tokens it is asked to predict — an UPPER BOUND on "
                  "what the coda can use, not on what a causal loop could infer",
    }
    tc.slot_depth_fixed = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth or model.cfg.max_depth))
    model.cfg.slot_gain_lambda = 0.0
    model.eval()
    model.requires_grad_(False)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    real_groups = model._tul_group_losses
    acc: dict = {}
    curves: dict = {}
    n_rows = 0

    def add(key, ce, n):
        s, c = acc.get(key, (0.0, 0.0))
        acc[key] = (s + ce * n, c + n)

    with ZSplit(model) as split:
        for bi, (inp, labels, layout, _) in enumerate(batches):
            inp, labels = inp.cuda(), labels.cuda()
            layout = layout.to("cuda")
            valid = layout.slot_valid
            masks = buckets(layout, a.head)
            torch.manual_seed(a.seed + bi)
            torch.cuda.manual_seed_all(a.seed + bi)

            out, z_loop, h0 = split.record(inp, labels, layout, want_groups=True)
            if split.st["core0"] != a.depth:
                raise RuntimeError(f"core[0] ran {split.st['core0']} times, "
                                   f"expected {a.depth}")
            ce_loop = float(split.token_ce)
            n_tgt = float(split.st["groups"]["n_targets"])
            n_rows += int(inp.shape[0])
            mux_loop = float(out["mux_local"]) if "mux_local" in out else None
            add("ce_loop", ce_loop, n_tgt)
            if mux_loop is not None:
                add("mux_loop", mux_loop, n_tgt)
            b_loop = bucket_ce(real_groups, split.xh, labels, layout, masks)
            for k, (v, n) in b_loop.items():
                add(f"ce_loop/{k}", v, n)
            del out

            # -- the reproduction gate: the recorded z back through the replay -----
            rep = split.replay(inp, labels, layout, z_loop, want_groups=False)
            ce_rep = float(split.token_ce)
            add("ce_repro", ce_rep, n_tgt)
            add("repro_abs_err", abs(ce_rep - ce_loop), n_tgt)
            del rep

            # -- the controls ------------------------------------------------------
            for name, mode, zz in (("ce_zero", "zero", z_loop),
                                   ("ce_shuffle", "shuffle", z_loop),
                                   ("ce_entry", "normal", h0)):
                if zz is None:
                    continue
                torch.manual_seed(a.seed + bi)
                split.replay(inp, labels, layout, zz, plan_mode=mode, want_groups=False)
                add(name, float(split.token_ce), n_tgt)
                if name == "ce_entry":
                    bb = bucket_ce(real_groups, split.xh, labels, layout, masks)
                    for k, (v, n) in bb.items():
                        add(f"ce_entry/{k}", v, n)

            # -- z optimisation, one run per lr, plus the random start -------------
            # A random start with the SAME per-slot norm as the loop's z: the control
            # that says whether the loop's z is a useful starting point or the coda can
            # be driven from anywhere.
            g = torch.Generator(device="cuda").manual_seed(a.seed + bi)
            r = torch.randn(z_loop.shape, generator=g, device="cuda", dtype=torch.float32)
            scale = (z_loop.float().flatten(2).norm(dim=2)
                     / (r.flatten(2).norm(dim=2) + 1e-9))            # [B, S]
            z_rand = (r * scale.reshape(*scale.shape,
                                        *([1] * (z_loop.dim() - 2)))).to(z_loop.dtype)

            starts = [(f"lr{lr:g}", z_loop, lr) for lr in lrs]
            starts += [(f"rand_lr{lrs[0]:g}", z_rand, lrs[0])]
            for tag, z0, lr in starts:
                zf, curve = optimise_z(split, inp, labels, layout, z0, lr, a.steps, valid)
                curves.setdefault(tag, []).append(curve)
                split.replay(inp, labels, layout, zf, want_groups=False)
                ce_f = float(split.token_ce)
                add(f"ce_zopt/{tag}", ce_f, n_tgt)
                bb = bucket_ce(real_groups, split.xh, labels, layout, masks)
                for k, (v, n) in bb.items():
                    add(f"ce_zopt/{tag}/{k}", v, n)
                st_f = z_stats(zf, valid)
                for k, v in st_f.items():
                    add(f"z_{k}/{tag}", v, 1.0)
                add(f"cos_to_loop/{tag}", z_cos(zf, z_loop, valid), 1.0)
                add(f"cos_to_start/{tag}", z_cos(zf, z0, valid), 1.0)
                del zf
            for k, v in z_stats(z_loop, valid).items():
                add(f"z_{k}/loop", v, 1.0)
            if h0 is not None:
                for k, v in z_stats(h0, valid).items():
                    add(f"z_{k}/entry", v, 1.0)
            torch.cuda.empty_cache()
            print(f"  batch {bi + 1}/{len(batches)} done", flush=True)

    res = {k: s / c for k, (s, c) in acc.items()}
    rec = {"label": label, "config": config, "step": step, "rows": n_rows,
           "batch": a.batch, "depth": a.depth, "steps": a.steps, "lrs": lrs,
           "head_n": a.head, "n_batches": len(batches), "notes": notes,
           "results": res, "curves": curves}

    # The causality gate: no z can reach a token of span 0 (causal attention puts every
    # slot cell after them). If optimising z moved that CE, z is leaking backwards.
    cg = {}
    for tag in list(curves):
        b = res.get(f"ce_zopt/{tag}/before_slot0")
        if b is not None:
            cg[tag] = abs(b - res["ce_loop/before_slot0"])
    rec["causality_before_slot0_abs_delta"] = cg
    rec["repro_pass"] = res.get("repro_abs_err", 1.0) == 0.0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)

    print(f"\n{label} step {step} — {n_rows} rows, batch {a.batch}, depth {a.depth}, "
          f"{a.steps} Adam steps")
    print(f"  probe changes: {json.dumps(notes)}")
    print(f"  ce_loop      {res['ce_loop']:.4f}"
          + (f"   mux_local {res['mux_loop']:.4f}" if "mux_loop" in res else ""))
    print(f"  ce_repro     {res['ce_repro']:.6f}  (|Δ| {res['repro_abs_err']:.3e} — "
          f"{'BIT-EXACT' if rec['repro_pass'] else 'NOT EXACT'})")
    for k in ("ce_zero", "ce_shuffle", "ce_entry"):
        if k in res:
            print(f"  {k:12s} {res[k]:.4f}   ({res[k] - res['ce_loop']:+.4f} vs loop)")
    for tag in curves:
        k = f"ce_zopt/{tag}"
        print(f"  ce_zopt {tag:12s} {res[k]:.4f}   ({res[k] - res['ce_loop']:+.4f} vs loop)"
              f"  cos(z*, z_loop) {res['cos_to_loop/' + tag]:+.3f}"
              f"  |z| {res['z_norm_mean/' + tag]:.2f}"
              f"  rank {res['z_eff_rank/' + tag]:.1f}")
    print(f"  z_loop  |z| {res['z_norm_mean/loop']:.2f}  rank {res['z_eff_rank/loop']:.1f}"
          f"  ({res['z_n_valid/loop']:.0f} valid slots/batch)")
    print("  per-bucket CE (head = first "
          f"{a.head} tokens after a slot):")
    hdr = ["before_slot0", "head", "tail", "dump"]
    print("      " + "arm".ljust(20) + "".join(h.rjust(15) for h in hdr))
    rows = [("loop", "ce_loop")] + ([("entry(h0)", "ce_entry")] if "ce_entry" in res else [])
    rows += [(f"zopt {t}", f"ce_zopt/{t}") for t in curves]
    for name, pre in rows:
        print("      " + name.ljust(20)
              + "".join(f"{res.get(pre + '/' + h, float('nan')):15.4f}" for h in hdr))
    print(f"  causality (|Δ| on before_slot0): {json.dumps({k: round(v, 6) for k, v in cg.items()})}")
    print("\nwrote", a.out)
    if not rec["repro_pass"]:
        raise SystemExit(f"{label}: the recorded z does NOT reproduce the trained forward's "
                         "token CE — the split point is wrong; do not report these numbers")


if __name__ == "__main__":
    main()
