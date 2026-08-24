"""Offline decomposition of the AdEMAMix update, and the coherent-drift severity measure.

Two questions, one reader, zero GPU training minutes. Pre-registration:
`docs/experiments/planned/2026-08-24-tul-optimizer-state-decomposition.md`.

Q1  The b1=0 AdEMAMix update is

        u = (g + alpha*m2) / sqrt(nu/bc2 + eps) + wd*p

    Two channels. The FAST one, `g/sqrt(nu/bc2+eps)`, has per-coordinate RMS near 1 by
    construction, because `nu` is the EMA of `g**2`. The SLOW one, `alpha*m2/sqrt(...)`,
    is whatever the accumulator has kept. A checkpoint stores m2 and nu but not g, so the
    slow channel is EXACT here and the fast channel is a near-1 reference, not a
    measurement. Read `slow_rms` as "the slow channel in units of the fast channel".

Q2  `docs/experiments/failures/2026-08-24-tul-takeover-cure.md` names the severity measure
    it should have used and did not: post-optimizer coherent core drift, `||dW_core||`
    times its directional autocorrelation. Consecutive checkpoints give it at the spacing
    of the ladder (25 steps), which is NOT the per-step version. `coh` below is the
    per-step estimate from the accumulator itself, so the two scales cross-check.

WHY THE NAME MAP IS BUILT AND NOT INFERRED
    `optimizer.state_dict()` keys parameters by their POSITION in the flattened
    param_groups, so recovering a name needs the same model and the same
    `_param_groups()` split the trainer used. Guessing it from tensor shapes silently
    mislabels every region on the first collision. This module builds the model through
    `lab/divergence/_build.py` and asserts a shape match on EVERY state entry.

DEQUANTISATION
    State is bnb dynamic-qmap blockwise-8bit for parameters of 4096 elements or more
    (`m2_dcode`/`m2_damax`, `nu_dcode`/`nu_damax`) and plain fp32 (`m2`/`nu`) below that.
    The 8-bit path goes through the OPTIMIZER'S OWN `_deq`, never a hand-written formula.

Usage:
    PYTHONPATH=$PWD python lab/divergence/optstate_probe.py --self-test \
        --ckpt checkpoints/morph/onset-capture/ROLL_step_1850.pt
    PYTHONPATH=$PWD python lab/divergence/optstate_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --drift --out ladder_opt.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lab.divergence._build import build_cfg, build_model          # noqa: E402
from morph.training.ademamix_b1zero import AdEMAMixB1Zero         # noqa: E402
from morph.training.optimizer import _param_groups                # noqa: E402

__all__ = [
    "region_of", "param_names_in_optimizer_order", "dequant_state",
    "param_stats", "aggregate", "drift_between",
]


# ── naming ───────────────────────────────────────────────────────────────────────────
def region_of(name: str) -> str:
    """The region bucket for a parameter name.

    THE SAME RULE as `_preclip_probe` in `morph/training/train.py`: first dotted
    component, with torch.compile's `_orig_mod.` wrapper stripped. Kept identical on
    purpose so `slow_rms(core)` and `preclip/core` count the same tensors.
    """
    return name.replace("_orig_mod.", "").split(".")[0]


def param_names_in_optimizer_order(model, weight_decay: float) -> list[str]:
    """Names in the order `optimizer.state_dict()` indexes them.

    torch flattens param_groups in order and params within each group in order, so this is
    exactly `_param_groups(model, wd)` re-run for names instead of tensors. Reusing the
    trainer's own splitter means a change to `_NO_DECAY_KEYWORDS` cannot silently
    de-synchronise this map.
    """
    by_id = {id(p): n for n, p in model.named_parameters()}
    groups = _param_groups(model, weight_decay)
    out: list[str] = []
    for g in groups:
        for p in g["params"]:
            out.append(by_id[id(p)])
    return out


# ── dequantisation ───────────────────────────────────────────────────────────────────
def _deq_helper(device: str = "cpu") -> AdEMAMixB1Zero:
    """An optimizer instance held ONLY for its `_deq` and its code maps.

    Constructed on a throwaway parameter. Using the real class rather than a local copy of
    the formula is the point: if the storage convention changes, this reader breaks loudly
    instead of reporting wrong numbers quietly.
    """
    dummy = torch.nn.Parameter(torch.zeros(1, device=device))
    return AdEMAMixB1Zero([dummy], lr=1e-4, betas=(0.0, 0.999, 0.999))


def dequant_state(entry: dict, numel: int, helper: AdEMAMixB1Zero,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(m2, nu) as flat fp32 tensors, whatever the storage.

    Raises on an unknown layout rather than returning zeros: a silent zero here reads as
    "the slow channel is empty", which is the exact conclusion this instrument exists to
    test.
    """
    if "m2" in entry and "nu" in entry:
        return entry["m2"].reshape(-1).float(), entry["nu"].reshape(-1).float()
    if "m2_dcode" in entry:
        ref = torch.zeros(numel, device=entry["m2_dcode"].device)
        m2 = helper._deq(entry["m2_dcode"], entry["m2_damax"], True, ref)
        nu = helper._deq(entry["nu_dcode"], entry["nu_damax"], False, ref)
        return m2.reshape(-1), nu.reshape(-1)
    if "m2_code" in entry:
        # Legacy linear-int8 fused layout: nu is stored as its SQUARE ROOT.
        m2 = _linear_deq(entry["m2_code"], entry["m2_amax"])
        nu_sqrt = _linear_deq(entry["nu_sqrt_code"], entry["nu_sqrt_amax"])
        return m2, nu_sqrt.square()
    raise KeyError(f"unknown optimizer state layout: {sorted(entry)}")


def _linear_deq(code: torch.Tensor, amax: torch.Tensor, block: int = 256) -> torch.Tensor:
    """val = code * (absmax/127), per block — the fused kernel's linear-int8 convention."""
    n = code.numel()
    pad = (-n) % block
    c = torch.cat([code.float(), code.new_zeros(pad).float()]) if pad else code.float()
    return (c.view(-1, block) * (amax.float() / 127.0).unsqueeze(1)).reshape(-1)[:n]


# ── the measurement ──────────────────────────────────────────────────────────────────
def param_stats(m2: torch.Tensor, nu: torch.Tensor, alpha: float, bc2: float,
                eps: float, eps_inside: bool, update_clip: float) -> dict:
    """Per-parameter channel decomposition. Sums, not means, so regions aggregate.

    THE DENOMINATOR IS NOT A DETAIL. `eps_inside` selects between

        True   denom = sqrt(nu/bc2 + eps)      floored at sqrt(eps) = 1e-4
        False  denom = sqrt(nu/bc2) + eps      true-Adam normalisation

    and MORPH runs the second (`base.yaml: ademamix_eps_inside: false`). The first version
    of this reader hardcoded the floored form and reported that 99.2 to 100 % of core
    coordinates sat on the floor, which would have meant the optimizer was not normalising
    at all. It was reading a denominator the run never used. The value is taken from the
    resolved config and printed with every row so the substitution cannot repeat silently.

    `update_clip` is the trainer's per-coordinate cap on (g + alpha*m2)/denom, 5.0 here. A
    slow channel above it is not merely large, it is saturating the clip on its own.
    """
    q = nu / bc2
    denom = (q + eps).sqrt() if eps_inside else q.sqrt() + eps
    slow = (alpha * m2) / denom
    # The FAST channel, now measurable rather than assumed: E[g**2] per coordinate IS
    # nu/bc2, so RMS(g/denom) = sqrt(mean(q/denom**2)). Under eps-outside this lands at
    # essentially 1 and the assumption in the pre-registration holds; under eps-inside it
    # would not, which is exactly why it is computed and not asserted.
    fast_sq = q / denom.square()
    live = nu > 0
    if live.any():
        pc = m2[live].abs() / q[live].sqrt()
        coh_pc_sq_sum = float(pc.square().sum())
        n_live = int(live.sum())
    else:
        coh_pc_sq_sum, n_live = 0.0, 0
    return {
        "n": m2.numel(),
        "slow_sq_sum": float(slow.square().sum()),
        "fast_sq_sum": float(fast_sq.sum()),
        "m2_sq_sum": float(m2.square().sum()),
        "nu_sum": float(nu.sum()),
        "coh_pc_sq_sum": coh_pc_sq_sum,
        "n_live": n_live,
        # counts, so they aggregate: how many coordinates the slow channel alone would
        # push past 1 Adam unit, and past the trainer's per-coordinate clip.
        "n_slow_gt1": int((slow.abs() > 1.0).sum()),
        "n_slow_gt_clip": int((slow.abs() > update_clip).sum()) if update_clip else 0,
    }


def aggregate(per_param: dict[str, dict]) -> dict[str, dict]:
    """Region totals -> RMS. `all` is every parameter; `noncore` is everything but `core`."""
    acc: dict[str, dict] = {}
    for name, s in per_param.items():
        for key in (region_of(name), "all", "noncore" if region_of(name) != "core" else None):
            if key is None:
                continue
            a = acc.setdefault(key, {k: 0 if k.startswith("n") else 0.0 for k in
                                     ("n", "slow_sq_sum", "fast_sq_sum", "m2_sq_sum",
                                      "nu_sum", "coh_pc_sq_sum", "n_live",
                                      "n_slow_gt1", "n_slow_gt_clip")})
            for f in a:
                a[f] += s[f]
    out = {}
    for k, a in acc.items():
        n = a["n"] or 1
        slow_rms = (a["slow_sq_sum"] / n) ** 0.5
        fast_rms = (a["fast_sq_sum"] / n) ** 0.5
        out[k] = {
            "n": a["n"],
            "n_live": a["n_live"],
            "slow_rms": slow_rms,
            "fast_rms": fast_rms,
            # THE headline: the slow channel in units of the fast one, both measured.
            "slow_over_fast": slow_rms / fast_rms if fast_rms else 0.0,
            "coh": (a["m2_sq_sum"] / a["nu_sum"]) ** 0.5 if a["nu_sum"] > 0 else 0.0,
            "coh_percoord": ((a["coh_pc_sq_sum"] / a["n_live"]) ** 0.5
                             if a["n_live"] else 0.0),
            "frac_slow_gt1": a["n_slow_gt1"] / n,
            "frac_slow_gt_clip": a["n_slow_gt_clip"] / n,
        }
    return out


def drift_between(prev: dict[str, torch.Tensor], cur: dict[str, torch.Tensor],
                  names: list[str]) -> dict[str, torch.Tensor]:
    """dW per region, as ONE flat vector per region, so cosines are over the whole region."""
    by_region: dict[str, list[torch.Tensor]] = {}
    for n in names:
        if n not in prev or n not in cur:
            continue
        d = (cur[n].float() - prev[n].float()).reshape(-1)
        for key in (region_of(n), "all"):
            by_region.setdefault(key, []).append(d)
    return {k: torch.cat(v) for k, v in by_region.items()}


# ── checkpoint plumbing ──────────────────────────────────────────────────────────────
def _model_tensors(ck: dict, names: list[str]) -> dict[str, torch.Tensor]:
    """The checkpoint's copy of each optimizer-owned parameter, keyed by the model's name.

    Checkpoints written under torch.compile carry an `_orig_mod.` prefix that
    `named_parameters()` on an uncompiled model does not, so both spellings are tried and a
    name that resolves to neither is REPORTED, not skipped in silence.
    """
    sd = ck["model"]
    # torch.compile wraps SUBMODULES too, so `_orig_mod.` appears mid-name
    # (`prelude.0.mlp._orig_mod.0.gate_up...`), not only as a prefix. Normalise by
    # deleting every occurrence on both sides. A collision would silently pair the wrong
    # tensors, so it raises.
    norm: dict[str, str] = {}
    for k in sd:
        nk = k.replace("_orig_mod.", "")
        if nk in norm:
            raise KeyError(f"checkpoint key collision after stripping _orig_mod.: "
                           f"{norm[nk]} and {k}")
        norm[nk] = k
    out, missing = {}, []
    for n in names:
        k = norm.get(n.replace("_orig_mod.", ""))
        if k is None:
            missing.append(n)
        else:
            out[n] = sd[k]
    if missing:
        raise KeyError(f"{len(missing)} optimizer params absent from ck['model'], "
                       f"first: {missing[:3]}")
    return out


def read_checkpoint(path: str, names: list[str], helper: AdEMAMixB1Zero,
                    want_weights: bool, eps_inside: bool, update_clip: float) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    opt = ck["optimizer"]
    groups = opt["param_groups"]
    step = int(groups[0].get("step", ck.get("step", 0)))
    alpha, _b2, _b3 = AdEMAMixB1Zero._sched(step, groups[0])
    beta2 = groups[0]["betas"][1]
    eps = float(groups[0]["eps"])
    bc2 = 1.0 - beta2 ** step

    flat_idx: list[int] = []
    for g in groups:
        flat_idx.extend(g["params"])
    if len(flat_idx) != len(names):
        raise AssertionError(f"{path}: optimizer holds {len(flat_idx)} params, "
                             f"the model offers {len(names)}")

    per_param, n_mapped, n_nostate = {}, 0, 0
    numel_by_name = {n: t.numel() for n, t in _model_tensors(ck, names).items()}
    for pos, idx in enumerate(flat_idx):
        name = names[pos]
        entry = opt["state"].get(idx)
        if entry is None:
            n_nostate += 1
            continue
        numel = numel_by_name[name]
        m2, nu = dequant_state(entry, numel, helper)
        if m2.numel() != numel:
            raise AssertionError(f"{path}: state[{idx}] -> {name}: {m2.numel()} elements, "
                                 f"parameter has {numel}")
        n_mapped += 1
        per_param[name] = param_stats(m2, nu, alpha, bc2, eps, eps_inside,
                                      update_clip)

    out = {
        "path": os.path.basename(path), "step": step, "alpha": alpha, "bc2": bc2,
        "eps": eps, "eps_inside": eps_inside, "update_clip": update_clip,
        "n_mapped": n_mapped, "n_without_state": n_nostate,
        "regions": aggregate(per_param),
        "per_param": {k: v for k, v in per_param.items() if region_of(k) == "core"},
    }
    out["weights"] = _model_tensors(ck, names) if want_weights else None
    return out


def state_by_name(ck: dict, names: list[str], helper: AdEMAMixB1Zero,
                  ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """{parameter name: (m2, nu)} plus the schedule scalars, from one checkpoint.

    The shared entry point for anything that needs the optimizer state keyed by NAME
    rather than by torch's positional index. Parameters with no state are absent from the
    mapping rather than present with zeros.
    """
    opt = ck["optimizer"]
    groups = opt["param_groups"]
    step = int(groups[0].get("step", ck.get("step", 0)))
    alpha, _b2, _b3 = AdEMAMixB1Zero._sched(step, groups[0])
    sched = {"step": step, "alpha": alpha, "eps": float(groups[0]["eps"]),
             "bc2": 1.0 - groups[0]["betas"][1] ** step}
    flat = [i for g in groups for i in g["params"]]
    if len(flat) != len(names):
        raise AssertionError(f"optimizer holds {len(flat)} params, model offers {len(names)}")
    tensors = _model_tensors(ck, names)
    out = {}
    for pos, idx in enumerate(flat):
        e = opt["state"].get(idx)
        if e is None:
            continue
        name = names[pos]
        out[name] = dequant_state(e, tensors[name].numel(), helper)
    return out, sched


# ── self test ────────────────────────────────────────────────────────────────────────
def self_test(ckpt: str, names: list[str], helper: AdEMAMixB1Zero, sabotage: str) -> int:
    """Prove the map and the dequant, and FAIL when either is broken.

    The MAP is proved by a shape assertion over every state entry; `--sabotage map`
    rotates the name list by one, which must trip it.

    The DEQUANT is proved two ways. First against an independent reconstruction of the
    documented convention, `value = code_map[code] * absmax_of_block`, built here with a
    gather rather than by calling the same helper twice — that is what pins the code map,
    the signedness and the blocksize. Second by dequantise -> quantise -> dequantise, whose
    residual is the quantiser's own precision. `--sabotage dequant` perturbs the scale fed
    to `_deq` only, so the independent reconstruction still holds the truth and the check
    must fail.

    Note a NON-bug the strict version of this test tripped on: re-quantising a dequantised
    tensor returns a slightly SMALLER absmax, because the largest code in a block maps to
    at most 1.0 times the old absmax. Comparing absmax for equality is wrong; comparing
    reconstructed VALUES is right.
    """
    import bitsandbytes.functional as bnbF
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    opt = ck["optimizer"]
    flat_idx = [i for g in opt["param_groups"] for i in g["params"]]
    use = list(names)
    if sabotage == "map":
        use = use[1:] + use[:1]

    tensors = _model_tensors(ck, names)
    n_ok, stateless = 0, []
    try:
        for pos, idx in enumerate(flat_idx):
            entry = opt["state"].get(idx)
            if entry is None:
                stateless.append(names[pos])
                continue
            numel = tensors[use[pos]].numel()
            m2, _nu = dequant_state(entry, numel, helper)
            if m2.numel() != numel:
                print(f"MAP_FAIL at position {pos}: state has {m2.numel()}, "
                      f"{use[pos]} has {numel}")
                return 1
            n_ok += 1
    except KeyError as e:
        print(f"MAP_FAIL: {e}")
        return 1
    print(f"MAP_OK {n_ok}/{len(flat_idx)} state entries, "
          f"{len(flat_idx) - n_ok} parameters carry no state")
    if stateless:
        print(f"  no state, first 3 of {len(stateless)}: {stateless[:3]}")
        print(f"  regions without state: "
              f"{sorted({region_of(n) for n in stateless})}")

    big = max((i for i in opt["state"] if "m2_dcode" in opt["state"][i]),
              key=lambda i: opt["state"][i]["m2_dcode"].numel())
    e = opt["state"][big]
    n = e["m2_dcode"].numel()
    ref = torch.zeros(n)
    amax_true = e["m2_damax"]
    amax_used = amax_true * 1.05 if sabotage == "dequant" else amax_true

    code_map = helper._code(torch.device("cpu"), True)
    manual = (code_map[e["m2_dcode"].reshape(-1).long()]
              * amax_true.float().repeat_interleave(256)[:n])
    got = helper._deq(e["m2_dcode"], amax_used, True, ref).reshape(-1)
    rel = float((got - manual).abs().max() / (manual.abs().max() + 1e-30))
    print(f"independent reconstruction of {n} codes: relative max error {rel:.3e}")
    if rel > 1e-6:
        print("DEQUANT_FAIL (does not match code_map[code] * absmax)")
        return 1

    q, qs = bnbF.quantize_blockwise(got.contiguous(), code=code_map, blocksize=256)
    again = helper._deq(q, qs.absmax, True, ref).reshape(-1)
    rt = float((again - got).norm() / (got.norm() + 1e-30))
    print(f"dequant->quant->dequant residual: {rt:.3e} relative")
    if rt > 1e-2:
        print("DEQUANT_FAIL (round trip beyond the quantiser's precision)")
        return 1
    print("DEQUANT_OK")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────────
def _step_of(path: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a1")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--drift", action="store_true",
                    help="also report ||dW|| and its directional autocorrelation between "
                         "consecutive checkpoints (needs 3 or more)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sabotage", default="", choices=["", "map", "dequant"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    paths = list(a.ckpt)
    if a.ckpt_dir:
        paths += sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")), key=_step_of)
    if not paths:
        ap.error("give --ckpt or --ckpt-dir")

    cfg = build_cfg(a.config_name, [o for o in a.overrides.split(",") if o.strip()])
    model, _tul = build_model(cfg, device=a.device)
    wd = float(getattr(cfg.training, "weight_decay", 0.1))
    names = param_names_in_optimizer_order(model, wd)
    # From the RESOLVED CONFIG, never from the checkpoint: `eps_inside` and `update_clip`
    # are instance attributes of the optimizer, not param-group defaults, so
    # `optimizer.state_dict()` does not carry them.
    eps_inside = bool(getattr(cfg.training, "ademamix_eps_inside", False))
    update_clip = float(getattr(cfg.training, "ademamix_update_clip", 0.0))
    print(f"denominator: {'sqrt(nu/bc2 + eps)' if eps_inside else 'sqrt(nu/bc2) + eps'}"
          f"   update_clip={update_clip}")
    helper = _deq_helper(a.device)

    if a.self_test:
        return self_test(paths[0], names, helper, a.sabotage)

    rows, prev_w, prev_d = [], None, None
    for p in paths:
        r = read_checkpoint(p, names, helper, want_weights=a.drift,
                            eps_inside=eps_inside, update_clip=update_clip)
        w = r.pop("weights")
        if a.drift and prev_w is not None:
            d = drift_between(prev_w, w, names)
            r["drift"] = {}
            total = float(d["all"].norm()) if "all" in d else 0.0
            for reg, vec in d.items():
                ent = {"norm": float(vec.norm())}
                # SHARE, not magnitude. `clip_grad_norm_` rescales the whole gradient to a
                # fixed global norm, so ||dW_all|| is nearly a constant of the schedule and
                # ||dW_core|| inherits that constancy. What can still move is how much of
                # that fixed budget the core takes.
                ent["share"] = ent["norm"] / total if total else 0.0
                if prev_d is not None and reg in prev_d:
                    pv = prev_d[reg]
                    ent["ac"] = float(torch.dot(vec, pv)
                                      / (vec.norm() * pv.norm() + 1e-30))
                    ent["coherent"] = ent["norm"] * ent["ac"]
                r["drift"][reg] = ent
            prev_d = d
        prev_w = w
        rows.append(r)
        reg = r["regions"]
        dr = r.get("drift", {}).get("core", {})
        print(f"{r['path']:<24} step={r['step']:<6} alpha={r['alpha']:.2f} "
              f"core s/f={reg['core']['slow_over_fast']:.4f} "
              f"fast={reg['core']['fast_rms']:.3f} coh={reg['core']['coh']:.4f} "
              f"gt1={reg['core']['frac_slow_gt1']:.2e} "
              f"| noncore s/f={reg['noncore']['slow_over_fast']:.4f} "
              f"coh={reg['noncore']['coh']:.4f}"
              + (f" | dW_core={dr['norm']:.4f} share={dr['share']:.4f}"
                 if "norm" in dr else "")
              + (f" ac={dr['ac']:+.4f} coherent={dr['coherent']:.4f}"
                 if "ac" in dr else ""))
        sys.stdout.flush()

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"label": a.label, "config": a.config_name,
                       "overrides": a.overrides, "rows": rows}, f, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
