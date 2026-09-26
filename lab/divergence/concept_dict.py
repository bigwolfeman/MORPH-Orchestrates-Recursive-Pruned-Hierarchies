"""LX-Concept step 3: the two concept dictionaries over E's span codes, and CoCoMix
attribution labels through the teacher's frozen decoder.

Prereg: ``lab/experiments/planned/2026-09-26-lx-concept-precheck.md`` (Method, step 3).
Terms: ``_concept.py``'s module docstring.

``fit`` (GPU or CPU; fitted on the TRAIN region's codes only, applied to both regions):

  (a) span TYPES: k-means on the 2,048-d codes (k-means++ init, Lloyd, seed 0) for every M
      in ``--types`` (64, 256). Label = the nearest centroid. Written: ``types_M<M>.npy``
      per region, ``kmeans_M<M>.pt`` (centroids, inertia trace).
  (b) a TopK SAE (``_concept.TopKSAE``: dictionary ``--n-latents`` 4096, ``--k`` 32, AuxK
      for latents dead for 2M samples, unit-norm decoder rows), trained on 95 % of the
      train-region codes, its fraction of variance unexplained (FVU), dead fraction and
      cosine read on the other 5 % and on the val region. Written: ``sae.pt``.

``attr`` (GPU; needs the teacher): for every coded span of a region, the teacher's coda
reads the SAE RECONSTRUCTION ``D(c)`` of E's code at the span's slot (every other slot keeps
E's clean code), and ``_concept.span_grads`` returns the EXACT gradient of THAT span's NLL
w.r.t. the reconstructed cells (one row copy per span; no other span's loss leaks in).
CoCoMix's attribution over every latent is then ``a = c^pre * (W_dec g)``
(``_concept.attribution``); the label is the top-4 latents by HIGHEST ``a`` (the paper's
rule, K_attr = 4). Removing latent i changes the span NLL by about ``-a_i`` to first order,
so the paper's top-4 are the latents whose removal would LOWER the loss the most; the four
LOWEST (``attr_low4``) are the latents the reconstruction leans on hardest. Both are
written; the pre-check grades the paper's set and reports the other beside it. Also written:
``sae_topk_idx`` (the k active latents, by activation), the span NLL at D(c)
(``recon_nll``; minus ``teacher_nll`` = the SAE's cost in nats) and a finite-difference check
of the gradient on a few spans (``--fd-check``).

Usage:
  python lab/divergence/concept_dict.py fit  --dump /home/wolfe/morph-scratch/concept/dump
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/concept_dict.py attr \\
      --dump /home/wolfe/morph-scratch/concept/dump --region val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _concept as cc  # noqa: E402
from _build import parse_ckpt_spec  # noqa: E402
from lxtul_e_stage1_score import load_arm  # noqa: E402


def _codes(dump: str, region: str, device: str) -> torch.Tensor:
    return torch.from_numpy(np.load(os.path.join(dump, region, "codes.npy"))).float().to(device)


def cmd_fit(a) -> None:
    t0 = time.time()
    dev = a.device
    Xtr = _codes(a.dump, "train", dev)
    Xva = _codes(a.dump, "val", dev)
    out = os.path.join(a.dump, "dict")
    os.makedirs(out, exist_ok=True)
    rep: dict = {"n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0])}
    for M in [int(x) for x in a.types.split(",")]:
        cent, lab, inertia = cc.kmeans(Xtr, M, iters=a.km_iters, seed=0)
        lv, dv = cc.assign(Xva, cent)
        torch.save({"centroids": cent.cpu(), "inertia": inertia}, os.path.join(out, f"kmeans_M{M}.pt"))
        np.save(os.path.join(a.dump, "train", f"types_M{M}.npy"), lab.cpu().numpy().astype(np.int64))
        np.save(os.path.join(a.dump, "val", f"types_M{M}.npy"), lv.cpu().numpy().astype(np.int64))
        cnt = torch.bincount(lab, minlength=M).double()
        p = cnt / cnt.sum()
        ent = float(-(p[p > 0] * p[p > 0].log()).sum())
        tot = float((Xtr - Xtr.mean(0)).pow(2).sum())
        rep[f"types_M{M}"] = {"iters": len(inertia) - 1, "inertia_first_last": [inertia[0], inertia[-1]],
                              "train_R2": 1.0 - inertia[-1] / tot, "label_entropy_nats": ent,
                              "log_M": float(np.log(M)), "smallest_cluster": int(cnt.min()),
                              "largest_cluster": int(cnt.max()),
                              "val_mean_sqdist": float(dv.mean()),
                              "train_mean_sqdist": inertia[-1] / Xtr.shape[0]}
        print(f"[types M={M}] {json.dumps(rep[f'types_M{M}'])}", flush=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(Xtr.shape[0], generator=g).to(dev)
    n_ho = max(1, int(0.05 * Xtr.shape[0]))
    ho, fit = perm[:n_ho], perm[n_ho:]
    sae, st = cc.train_sae(Xtr[fit], a.n_latents, a.k, a.sae_steps, batch=a.sae_batch,
                           lr=a.sae_lr, seed=0, log_every=max(1, a.sae_steps // 20))
    rep["sae"] = {"n_latents": a.n_latents, "k": a.k, "train": st,
                  "heldout_train_region": cc.sae_eval(sae, Xtr[ho]),
                  "val_region": cc.sae_eval(sae, Xva)}
    torch.save({"state": sae.state_dict(), "d_in": int(Xtr.shape[1]), "n_latents": a.n_latents,
                "k": a.k}, os.path.join(out, "sae.pt"))
    print(f"[sae] held-out {rep['sae']['heldout_train_region']}  val {rep['sae']['val_region']}",
          flush=True)
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(out, "fit.json"), "w"), indent=1, default=float)
    print(f"wrote {out}/fit.json")


def load_sae(dump: str, device: str) -> cc.TopKSAE:
    d = torch.load(os.path.join(dump, "dict", "sae.pt"), map_location="cpu")
    sae = cc.TopKSAE(d["d_in"], d["n_latents"], d["k"])
    sae.load_state_dict(d["state"])
    return sae.to(device).eval()


def cmd_attr(a) -> None:
    t0 = time.time()
    dev = a.device
    rdir = os.path.join(a.dump, a.region)
    meta = json.load(open(os.path.join(rdir, "meta.json")))
    keys = dict(np.load(os.path.join(rdir, "keys.npz")))
    X = _codes(a.dump, a.region, dev)
    n = X.shape[0] if a.max_spans <= 0 else min(X.shape[0], a.max_spans)
    sae = load_sae(a.dump, dev)
    for p in sae.parameters():
        p.requires_grad_(False)
    label, config, path, ovr = parse_ckpt_spec(a.teacher)
    m, step, cfg, rt = load_arm(config, path, dev, ovr)
    for p in m.parameters():
        p.requires_grad_(False)
    batches = cc.region_batches(cfg, rt, a.region, meta["rows"], meta["batch"],
                                meta["skip_docs"])
    M = int(m.tul_code_enc.m)
    C = X.shape[1] // M
    top_hi = np.zeros((n, a.k_attr), np.int64)
    top_lo = np.zeros((n, a.k_attr), np.int64)
    topk_idx = np.zeros((n, sae.k), np.int64)
    recon_nll = np.zeros(n, np.float64)
    done = 0
    fd = []
    for bi, (inp, lab, lay, idx) in enumerate(batches):
        if done >= n:
            break
        bk = cc.span_keys(lay, idx)
        m_b = bk["row"].size
        j0, j1 = done, min(n, done + m_b)
        # the dump's keys for these spans must be exactly this batch's, in order
        exp = cc.key64(keys["first"][j0:j1], keys["last"][j0:j1])
        got = cc.key64(bk["first"][:j1 - j0], bk["last"][:j1 - j0])
        if not np.array_equal(exp, got):
            raise RuntimeError(f"batch {bi}: re-packed span keys differ from the dump's")
        x = X[j0:j1]
        with torch.no_grad():
            c, pre = sae.encode(x)
            xh = sae.decode(c)
        g, nll = cc.span_grads(m, inp, lab, lay, bk["row"][:j1 - j0], bk["slot"][:j1 - j0],
                               xh.view(-1, M, C), dev, group=a.group)
        gt = torch.from_numpy(g).to(dev)
        att = cc.attribution(sae, pre, gt)
        top_hi[j0:j1] = cc.top_attr(att, a.k_attr, largest=True).cpu().numpy()
        top_lo[j0:j1] = cc.top_attr(att, a.k_attr, largest=False).cpu().numpy()
        topk_idx[j0:j1] = torch.topk(c, sae.k, dim=-1).indices.cpu().numpy()
        recon_nll[j0:j1] = nll
        if a.fd_check > 0 and bi == 0:
            # directional finite difference on the first spans: NLL(xh + h u) - NLL(xh - h u)
            gen = torch.Generator(device="cpu").manual_seed(0)
            for j in range(min(a.fd_check, j1 - j0)):
                u = torch.randn(M * C, generator=gen).to(dev)
                u = u / u.norm()
                hh = 0.5          # the code has norm ~45 (unit RMS x 2048): a 1 % step
                cells = torch.stack([xh[j] + hh * u, xh[j] - hh * u]).view(2, M, C)
                _g2, nl = cc.span_grads(m, inp, lab, lay, bk["row"][[j, j]], bk["slot"][[j, j]],
                                        cells, dev, group=2)
                fd.append({"span": j, "fd": float((nl[0] - nl[1]) / (2 * hh)),
                           "grad_dot_u": float(gt[j] @ u)})
        done = j1
        if bi % 20 == 0:
            print(f"  attr batch {bi}: {done}/{n} spans, {time.time() - t0:.0f} s", flush=True)
    tn = np.load(os.path.join(rdir, "teacher_nll.npy"))[:n].astype(np.float64)
    np.save(os.path.join(rdir, "attr_top4.npy"), top_hi[:done])
    np.save(os.path.join(rdir, "attr_low4.npy"), top_lo[:done])
    np.save(os.path.join(rdir, "sae_topk_idx.npy"), topk_idx[:done])
    np.save(os.path.join(rdir, "recon_nll.npy"), recon_nll[:done])
    rep = {"region": a.region, "n_spans": int(done), "teacher_step": step,
           "recon_nll_mean": float(recon_nll[:done].mean()),
           "teacher_nll_mean": float(tn[:done].mean()),
           "sae_cost_nats_per_span": float((recon_nll[:done] - tn[:done]).mean()),
           "attr_top4_distinct_latents": int(np.unique(top_hi[:done]).size),
           "attr_top4_in_active_topk_frac": float(np.mean([
               len(set(top_hi[i]) & set(topk_idx[i])) / a.k_attr for i in range(done)])),
           "fd_check": fd, "wall_s": round(time.time() - t0, 1)}
    if fd:
        rel = [abs(f["fd"] - f["grad_dot_u"]) / max(1e-6, abs(f["fd"])) for f in fd]
        rep["fd_max_rel_err"] = float(max(rel))
    json.dump(rep, open(os.path.join(rdir, "attr.json"), "w"), indent=1, default=float)
    print(json.dumps(rep, default=float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dump", required=True)
    f.add_argument("--types", default="64,256")
    f.add_argument("--km-iters", type=int, default=60)
    f.add_argument("--n-latents", type=int, default=4096)
    f.add_argument("--k", type=int, default=32)
    f.add_argument("--sae-steps", type=int, default=20000)
    f.add_argument("--sae-batch", type=int, default=1024)
    f.add_argument("--sae-lr", type=float, default=3e-4)
    f.add_argument("--device", default="cuda")
    t = sub.add_parser("attr")
    t.add_argument("--dump", required=True)
    t.add_argument("--region", required=True, choices=["val", "train"])
    t.add_argument("--teacher", default=f"vae={cc.TEACHER_CONFIG}={cc.TEACHER_CKPT}")
    t.add_argument("--k-attr", type=int, default=4)
    t.add_argument("--group", type=int, default=8, help="span copies per forward")
    t.add_argument("--max-spans", type=int, default=0, help="0 = every span of the region")
    t.add_argument("--fd-check", type=int, default=4)
    t.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cmd_fit(a) if a.cmd == "fit" else cmd_attr(a)


if __name__ == "__main__":
    main()
