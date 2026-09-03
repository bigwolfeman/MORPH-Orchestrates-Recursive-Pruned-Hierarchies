"""Does the span bag-mean pool the diversity out of the slot states?

Spec 3.2: a slot's input is `E_slot + mean_j embed(t_j)` over its span. A PLAIN mean. For
roughly independent token embeddings the mean's deviation from the corpus mean shrinks as
`1 / sqrt(L)`, so a slot input is a large CONSTANT plus a signal that vanishes with span
length. If that is what happens, it explains the measured slot-state effective rank of 1.7
to 4.8 in 1024 dimensions from first principles, with no appeal to how many positions the
loop sees — and it says the fix is the pooling operator, not the slot count.

The prediction is sharp and it can fail: `log ||v_i - vbar|| = a - 0.5 log L_i`. A slope
near 0 means pooling costs nothing and the hypothesis is dead. A slope near -0.5 means
every doubling of span length costs a factor 1.41 of slot-state spread.

Reported:

    span_len distribution        so a cap can be read against the DATA, not guessed
    slope of log(dev) vs log(L)  the test, -0.5 under independent tokens
    dev / ||E_slot||             signal-to-constant ratio: how much of a slot input is
                                 the shared constant that every slot carries
    eff_rank of the slot inputs  participation ratio, the quantity everything else moves

Usage:
    PYTHONPATH=$PWD python lab/divergence/pooling_probe.py \
        --ckpt checkpoints/morph/onset-capture/ROLL_step_1700.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lab.divergence._build import build_cfg, build_model            # noqa: E402
from morph.training.data import create_dataloader                   # noqa: E402
from morph.training.train import load_checkpoint                    # noqa: E402

__all__ = ["deviation_by_length", "loglog_slope", "eff_rank"]


def eff_rank(x: torch.Tensor) -> float:
    """Participation ratio of the singular values of the CENTRED rows.

    Centred because a shared constant across rows is exactly what this probe is about: an
    uncentred rank would count that constant as a direction and hide the collapse.
    """
    c = (x - x.mean(0, keepdim=True)).double()
    if c.shape[0] < 2:
        return float("nan")
    p = torch.linalg.svdvals(c).square()
    return float(p.sum().square() / p.square().sum())


def loglog_slope(L: torch.Tensor, d: torch.Tensor) -> tuple[float, float]:
    """Least-squares slope of log d against log L, and its r^2.

    r^2 is returned WITH the slope and is not optional: a slope fitted through a cloud with
    no trend is a number, not a finding.
    """
    x = L.double().log()
    y = d.double().log()
    x = x - x.mean()
    yc = y - y.mean()
    sxx = float((x * x).sum())
    if sxx == 0:
        return float("nan"), float("nan")
    b = float((x * yc).sum() / sxx)
    ss_res = float((yc - b * x).square().sum())
    ss_tot = float(yc.square().sum())
    return b, (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def sub_n(x: torch.Tensor, n: int) -> torch.Tensor:
    """A deterministic even subsample of `n` rows, or all of them if there are fewer.

    Effective rank is bounded above by (rows - 1), so comparing it between groups of
    different size compares the group sizes. Every rank reported at a fixed `n` in this
    file goes through here.
    """
    if x.shape[0] <= n:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, n).long().to(x.device)
    return x[idx]


def deviation_by_length(v: torch.Tensor, L: torch.Tensor, edges=(4, 6, 8, 12, 16, 24, 33),
                        n_fixed: int = 30) -> list[dict]:
    """Mean deviation and effective rank bucketed by span length.

    `eff_rank_n` is measured on a fixed-size subsample so the buckets are comparable;
    `eff_rank` on the whole bucket is kept beside it so the size effect stays visible.
    """
    vbar = v.mean(0, keepdim=True)
    d = (v - vbar).norm(dim=1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L >= lo) & (L < hi)
        if int(m.sum()) < 2:
            continue
        out.append({"lo": lo, "hi": hi - 1, "n": int(m.sum()),
                    "mean_dev": float(d[m].mean()),
                    "eff_rank": eff_rank(v[m]),
                    "eff_rank_n": eff_rank(sub_n(v[m], n_fixed))})
    return out


def collect(model, x, y, layout) -> dict:
    """One forward, capturing every slot's input vector and its span length.

    `slot_input` is a plain method rather than `forward`, so a module hook cannot see it.
    It is wrapped for the duration of this call and restored in `finally` — the model is
    left exactly as it was found.
    """
    tul = getattr(model, "_orig_mod", model).tul
    orig = tul.slot_input
    grabbed = {}

    def wrapped(signal, lay, add_e_slot):
        out = orig(signal, lay, add_e_slot)
        if add_e_slot and "v" not in grabbed:
            idx = lay.slot_index.clamp(min=0)                       # [B, S]
            g = idx.unsqueeze(-1).expand(*idx.shape, out.shape[-1])
            grabbed["v"] = torch.gather(out, 1, g).detach().float()  # [B, S, C]
            grabbed["valid"] = lay.slot_valid.detach()
            # `span_len` is populated only when the TUL gate is configured, so L is
            # DERIVED from bag_id: the number of TOKEN positions carrying each slot's id.
            # That is the same count the bag-mean divides by, which is the quantity this
            # probe is about.
            tok = (~lay.slot_mask)
            B, S = idx.shape
            bid = lay.bag_id.clone()
            bid = torch.where(tok, bid, torch.full_like(bid, S))
            grabbed["len"] = torch.stack([
                torch.bincount(bid[b], minlength=S + 1)[:S] for b in range(B)]).detach()
        return out

    tul.slot_input = wrapped
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=layout)
    finally:
        tul.slot_input = orig
    if "v" not in grabbed:
        raise RuntimeError("slot_input was never called with add_e_slot=True")

    m = grabbed["valid"].reshape(-1)
    v = grabbed["v"].reshape(-1, grabbed["v"].shape[-1])[m]
    L = grabbed["len"].reshape(-1)[m].float()
    e = tul.E_slot.detach().float()
    e_norm = float(e.norm(dim=-1).mean()) if e.dim() == 2 else float(e.norm())
    dev = (v - v.mean(0, keepdim=True)).norm(dim=1)
    slope, r2 = loglog_slope(L, dev)
    q = torch.tensor([0.25, 0.5, 0.75, 0.9])
    return {
        "n_slots": int(m.sum()),
        "span_len": {"mean": float(L.mean()), "median": float(L.median()),
                     "q": [float(t) for t in torch.quantile(L, q.to(L.device))],
                     "max": float(L.max())},
        "mean_dev": float(dev.mean()),
        "E_slot_norm": e_norm,
        "dev_over_const": float(dev.mean()) / (e_norm + 1e-30),
        "slope_log_dev_vs_log_L": slope,
        "slope_r2": r2,
        "eff_rank_all": eff_rank(v),
        # at a FIXED slot count, so configurations that produce different numbers of slots
        # can be compared without comparing their slot counts
        "eff_rank_n200": eff_rank(sub_n(v, 200)),
        "by_length": deviation_by_length(v, L),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a2")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--extra", default="", help="SEMICOLON-separated hydra overrides")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ov = [o for o in a.overrides.split(",") if o.strip()]
    ov += [o for o in a.extra.split(";") if o.strip()]
    cfg = build_cfg(a.config_name, ov)
    model, tul_rt = build_model(cfg, device="cuda")
    loader = iter(create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                                    int(cfg.data.seq_len), int(cfg.training.batch_size),
                                    split="validation", skip_samples=50_000,
                                    tul=tul_rt.val_data_cfg if tul_rt else None))
    x, y, layout = next(loader)
    x, y, layout = x.cuda(), y.cuda(), layout.to("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(a.ckpt, model, scaler, torch.device("cuda"))
    model.train()

    r = collect(model, x, y, layout)
    r["label"], r["ckpt"], r["overrides"] = a.label, a.ckpt, ov
    sl = r["span_len"]
    print(f"\n== {a.label or a.ckpt}   overrides={ov}")
    print(f"slots={r['n_slots']}  span_len mean={sl['mean']:.2f} median={sl['median']:.0f} "
          f"q25/75/90={sl['q'][0]:.0f}/{sl['q'][2]:.0f}/{sl['q'][3]:.0f} max={sl['max']:.0f}")
    print(f"eff_rank(slot inputs, centred) = {r['eff_rank_all']:.2f} over "
          f"{r['n_slots']} slots; at a fixed 200 slots = {r['eff_rank_n200']:.2f}")
    print(f"mean deviation {r['mean_dev']:.4f} vs ||E_slot|| {r['E_slot_norm']:.4f}  "
          f"-> signal/constant = {r['dev_over_const']:.4f}")
    print(f"slope of log(dev) vs log(L) = {r['slope_log_dev_vs_log_L']:+.3f} "
          f"(r2 {r['slope_r2']:.3f})   [-0.5 = plain-mean pooling law]")
    print(f"{'span len':>10}{'n':>6}{'mean dev':>11}{'eff rank':>10}{'at n=30':>10}")
    for b in r["by_length"]:
        print(f"{b['lo']:>4}-{b['hi']:<5}{b['n']:>6}{b['mean_dev']:>11.4f}"
              f"{b['eff_rank']:>10.2f}{b['eff_rank_n']:>10.2f}")
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
