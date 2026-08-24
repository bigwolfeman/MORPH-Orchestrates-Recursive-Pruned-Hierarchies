"""How much of the core's weight motion lands where the loss actually constrains it?

THE FIRST VERSION OF THIS FILE ASKED A WRONG QUESTION, and its own self-test said so in
one run. The claim was that a core linear's gradient `sum_i delta_i h_i^T` is confined to
the span of ~50 slot states at effective rank 3, so the loss constrains only a few of 1024
directions. That is wrong: the sum runs over batch x slots x loop iterations x blocks, a
few thousand distinct input vectors per linear, so the input span is nearly FULL rank —
measured, 935 of 1024 directions hold 99 % of the energy. Any projector at that level
captures everything and the measurement is vacuous.

What IS concentrated is the input ENERGY: participation ratio 11.2 of 1024. The loss's
curvature in a direction scales with that direction's input energy, so the weight is
strongly constrained in ~11 directions and weakly constrained in the rest. The honest
measurement is therefore not one projector but the whole CURVE over k:

    g@k    share of the raw gradient's energy inside the top-k input directions
    u@k    the same for the APPLIED AdEMAMix update
    m2@k   the same for the slow accumulator

`u` divides `g` elementwise by `sqrt(nu)`, and `nu` is the EMA of `g**2`, so the update is
WHITENED relative to the gradient. `g@k - u@k` is how much energy that rescale moves out of
the well-constrained directions and into the weakly-constrained ones. Tracking that gap
across the onset ladder, and against the token path at the same weights, is the test.

Self-test at both ends: every curve must reach exactly 1.0 at k = in_dim.

Usage:
    PYTHONPATH=$PWD python lab/divergence/subspace_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --out subspace.json
    PYTHONPATH=$PWD python lab/divergence/subspace_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --token-path --out subspace_tok.json
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

from lab.divergence._build import build_cfg, build_model                  # noqa: E402
from lab.divergence.jac_ladder import _raw_weight, core_linears           # noqa: E402
from lab.divergence.optstate_probe import (                               # noqa: E402
    _deq_helper, param_names_in_optimizer_order, state_by_name,
)
from morph.training.data import create_dataloader                         # noqa: E402
from morph.training.train import load_checkpoint                          # noqa: E402

KS = (1, 8, 32, 128)

__all__ = ["input_eigenbasis", "energy_curve", "at_k", "adamix_update"]


def energy_curve(m: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Cumulative share of ||m||_F^2 inside the top-k input directions, for every k.

    `m @ V` rotates the row space into the input eigenbasis, so the squared column norms
    are the per-direction energies and one cumsum gives the whole curve. It ends at exactly
    1.0 at k = in_dim by construction, which is the self-test.
    """
    c = (m @ V).pow(2).sum(0)
    tot = c.sum()
    if float(tot) == 0.0:
        return torch.full_like(c, float("nan"))
    return torch.cumsum(c, 0) / tot


def at_k(curve: torch.Tensor, k: int) -> float:
    return float(curve[min(k, curve.numel()) - 1])


def input_eigenbasis(gram: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Eigenvectors of the input Gram in descending order, plus its concentration.

    `eff_rank` is the participation ratio `(sum lam)^2 / sum(lam^2)`: n for n equal
    eigenvalues, 1 for a rank-1 Gram. `k50` / `k90` / `k99` are the counts needed for that
    share of the trace. They are reported together because a heavy-tailed spectrum makes
    them disagree by two orders of magnitude, and that disagreement is the finding rather
    than a defect.
    """
    lam, V = torch.linalg.eigh(gram.double())
    lam = lam.flip(0).clamp_min(0)
    V = V.flip(1)
    tot = lam.sum()
    eff = float(tot.square() / lam.square().sum()) if float(tot) > 0 else 0.0
    cum = torch.cumsum(lam, 0) / tot
    ks = {f"k{int(q * 100)}": int((cum < q).sum().item()) + 1 for q in (0.5, 0.9, 0.99)}
    return V.float(), {"eff_rank": eff, **ks}


def adamix_update(g: torch.Tensor, m2: torch.Tensor, nu: torch.Tensor,
                  alpha: float, bc2: float, eps: float) -> torch.Tensor:
    """The applied AdEMAMix update, eps OUTSIDE the sqrt.

    MORPH runs `ademamix_eps_inside: false`. The floored form is a ~100x error at MORPH's
    gradient scale — see
    `docs/experiments/failures/2026-08-24-tul-optimizer-state-decomposition.md`.
    """
    return (g + alpha * m2) / ((nu / bc2).sqrt() + eps)


class _InputGram:
    """Forward pre-hooks accumulating `sum_i x_i x_i^T` per module.

    Capture is switched OFF before the backward on purpose: the core is gradient
    checkpointed with `use_reentrant=False`, so every hooked module runs a SECOND time
    during recompute and would be counted twice.
    """

    def __init__(self, mods):
        self.gram: dict[str, torch.Tensor] = {}
        self.rows: dict[str, int] = {}
        self.on = True
        self._h = [mod.register_forward_pre_hook(self._mk(name)) for name, mod in mods]

    def _mk(self, name):
        def hook(_mod, args):
            if not self.on or not args or not isinstance(args[0], torch.Tensor):
                return
            f = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            g = self.gram.get(name)
            self.gram[name] = f.T @ f if g is None else g + f.T @ f
            self.rows[name] = self.rows.get(name, 0) + f.shape[0]
        return hook

    def close(self):
        for h in self._h:
            h.remove()


def measure(model, x, y, layout, mods, opt_state, sched, seed: int,
            token_path: bool) -> dict:
    root = getattr(model, "_orig_mod", model)
    cap = _InputGram(mods)
    model.zero_grad(set_to_none=True)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = (model(x, labels=y) if token_path
                   else model(x, labels=y, slot_layout=layout))
        cap.on = False                      # BEFORE backward: see _InputGram
        out["loss"].backward()
    finally:
        cap.close()

    by_id = {id(p): n for n, p in root.named_parameters()}
    rows = {}
    for name, mod in mods:
        w = _raw_weight(mod)
        if w is None or name not in cap.gram or w.grad is None:
            continue
        V, conc = input_eigenbasis(cap.gram[name])
        g = w.grad.detach().float()
        if g.shape[1] != V.shape[0]:
            raise AssertionError(f"{name}: grad is {tuple(g.shape)} but the captured "
                                 f"input has {V.shape[0]} channels")
        cg = energy_curve(g, V)
        r = {**conc, "in_dim": V.shape[0], "n_rows": cap.rows[name],
             "g@end": float(cg[-1])}
        for k in KS:
            r[f"g@{k}"] = at_k(cg, k)
        st = opt_state.get(by_id.get(id(w)))
        if st is not None:
            m2 = st[0].reshape(w.shape).to(g.device)
            nu = st[1].reshape(w.shape).to(g.device)
            cu = energy_curve(adamix_update(g, m2, nu, sched["alpha"], sched["bc2"],
                                            sched["eps"]), V)
            cm = energy_curve(m2, V)
            r["u@end"] = float(cu[-1])
            for k in KS:
                r[f"u@{k}"] = at_k(cu, k)
                r[f"m2@{k}"] = at_k(cm, k)
        rows[name] = r
    model.zero_grad(set_to_none=True)
    return rows


def _agg(rows: dict, field: str) -> float:
    v = [r[field] for r in rows.values() if field in r and r[field] == r[field]]
    return sum(v) / len(v) if v else float("nan")


def _step_of(p: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a1")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--scope", default="mlp", choices=["mlp", "attn", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-path", action="store_true")
    ap.add_argument("--extra", default="",
                    help="SEMICOLON-separated hydra overrides, for values that contain a "
                         "comma such as tul.boundary_chars=\".;!?,\"")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    paths = list(a.ckpt)
    if a.ckpt_dir:
        paths += sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")), key=_step_of)
    if not paths:
        ap.error("give --ckpt or --ckpt-dir")

    ov = [o for o in a.overrides.split(",") if o.strip()]
    ov += [o for o in a.extra.split(";") if o.strip()]
    cfg = build_cfg(a.config_name, ov)
    model, tul_rt = build_model(cfg, device="cuda")
    loader = iter(create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                                    int(cfg.data.seq_len), int(cfg.training.batch_size),
                                    split="validation", skip_samples=50_000,
                                    tul=tul_rt.val_data_cfg if tul_rt else None))
    batch = next(loader)
    if len(batch) == 3:
        x, y, layout = batch
        layout = layout.to("cuda")
    else:
        (x, y), layout = batch, None
    x, y = x.cuda(), y.cuda()
    model.train()

    names = param_names_in_optimizer_order(
        model, float(getattr(cfg.training, "weight_decay", 0.1)))
    helper = _deq_helper("cpu")
    mods = core_linears(model, a.scope)
    print(f"overrides: {ov}")
    print(f"{len(mods)} core {a.scope} linears; path="
          f"{'TOKEN (slot_layout=None)' if a.token_path else 'SLOT'}")

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    results = []
    for p in paths:
        load_checkpoint(p, model, scaler, torch.device("cuda"))
        ck = torch.load(p, map_location="cpu", weights_only=False)
        opt_state, sched = state_by_name(ck, names, helper)
        del ck
        rows = measure(model, x, y, layout, mods, opt_state, sched, a.seed, a.token_path)
        rec = {"path": os.path.basename(p), "step": sched["step"],
               "alpha": sched["alpha"], "token_path": a.token_path, "per_linear": rows}
        for f in ("eff_rank", "k50", "k90", "k99", "g@end", "u@end"):
            rec[f] = _agg(rows, f)
        for k in KS:
            for w_ in ("g", "u", "m2"):
                rec[f"{w_}@{k}"] = _agg(rows, f"{w_}@{k}")
        results.append(rec)
        print(f"{rec['path']:<24} step={rec['step']:<6} "
              f"effrank={rec['eff_rank']:>7.2f} k50={rec['k50']:>5.0f} "
              f"k90={rec['k90']:>6.0f} | g@8={rec['g@8']:.4f} u@8={rec['u@8']:.4f} "
              f"m2@8={rec['m2@8']:.4f} | g@32={rec['g@32']:.4f} "
              f"u@32={rec['u@32']:.4f} | gap@32={rec['g@32'] - rec['u@32']:+.4f}")
        sys.stdout.flush()

    if a.out:
        json.dump({"config": a.config_name, "scope": a.scope,
                   "token_path": a.token_path, "rows": results}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    bad = [r for r in results
           if abs(r["g@end"] - 1.0) > 1e-4 or abs(r["u@end"] - 1.0) > 1e-4]
    if bad:
        print(f"SELF-TEST FAILED on {len(bad)} checkpoints: an energy curve must reach "
              f"exactly 1.0 at k = in_dim. The eigenbasis is wrong; read nothing else.")
        return 1
    print("SELF-TEST PASS: every energy curve reaches 1.0 at k = in_dim")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
