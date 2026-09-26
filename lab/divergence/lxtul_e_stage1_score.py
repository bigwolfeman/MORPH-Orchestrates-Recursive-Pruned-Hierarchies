"""LXTUL-E Stage 1: the width and depth readings of `lxtul-e4` against `lxtul-e1` and the
ruler, on the same packed validation rows.

Spec: ``.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md``,
acceptance clauses P-1 .. P-5. Arms: ``morph/configs/tul_slot_spandec_strict_e4.yaml``
(K = 4 enumerated codes re-added every slot-loop pass, the parallel head and the coda under
the exact mixture) and ``..._e1.yaml`` (the same head and loop, K = 1, no code); the ruler
``slot-spandec-strict`` step 5000.

TERMS (one meaning each):

  depth d        the forced slot-loop depth of EVERY valid slot, the eval-only
                 ``slot_depths`` table (``transformer._slot_depth_override``).
  head token     a valid target of ``ParallelSpanHead.targets``: token j of span s+1 for a
                 supervised slot s.
  par CE         the parallel head's CE per head token. e4: the K-rollout mixture split over
                 a span's tokens by the chain rule (``mixture_token_nll``), which sums per
                 span to the exact mixture term; e1: the one reader's CE.
  coda token     a coda position with a label and CE weight 1 (the ruler's scored set:
                 ordinary tokens and each span's last token; the emit position has weight 0).
  coda CE        the coda's CE per coda token. e4 "mix": the per-span sequential Bayes read
                 over the K rollouts (``transformer._enum_position_nll``), which sums per span
                 to the exact span mixture the arm trains on; e4 "code k": rollout k's CE
                 alone; e1 / ruler: the one coda's CE.
  width gain     (coda or par) CE of ONE code alone minus the mixture CE, same tokens.
  exit sep       the mean over valid slots of the RMS distance between the K rollouts' exit
                 states (the head's input state), averaged over pairs. ``sep_ratio`` is its
                 value at depth 6 over depth 1; the note's linear theory reads
                 (1 - g^6)/(1 - g) = 4.6 at gain g = 0.89.
  code sep       exit sep restricted to the code subspace span(u_k), per stream, over the
                 same n*C denominator (``sep_code`` <= ``sep_abs``; added 2026-09-26 for
                 ``lx_amp_matched.py``, a diagnostic only).
  block         stream index // 1,024 (``sweep_score.BLOCK``), the bootstrap unit. A head
                 token's unit is its slot's first target token's (a span never splits).

STATISTICS. Token-weighted means; paired block bootstrap (``_stats.paired_bootstrap_ci``,
2,000 resamples, seed 0). Every paired reading is on identical tokens of identical rows.

SELF-CHECK. Each labelled forward also returns the model's own ``par_ce`` and
``ce_tokens``; the offline recomputation must match per batch (``max_abs_dev``; RAISES above
``--tol``), so these numbers are the ones the trainer optimised.

Usage (GPU):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/lxtul_e_stage1_score.py \\
      --e4 checkpoints/morph/lxtul-e4/step_5000.pt --e1 checkpoints/morph/lxtul-e1/step_5000.pt \\
      --rows 480 --out .../stage1.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _stats import paired_bootstrap_ci  # noqa: E402  (ONE home for the paired bootstrap)
from sweep_score import BLOCK  # noqa: E402  (1,024 stream tokens per bootstrap unit)

RULER = "/home/wolfe/morph-to/checkpoints/morph/slot-spandec-strict/step_5000.pt"
N_BOOT = 2000


class _Capture:
    """Record the positional args of ``model.<name>`` calls inside the block."""

    def __init__(self, model, name: str):
        self.model, self.name, self.got = model, name, []

    def __enter__(self):
        real = getattr(self.model, self.name)

        def wrap(*a, **k):
            self.got.append((a, k))
            return real(*a, **k)
        setattr(self.model, self.name, wrap)
        return self

    def __exit__(self, *exc):
        delattr(self.model, self.name)
        return False


def _one(cap: _Capture) -> tuple:
    if len(cap.got) != 1:
        raise RuntimeError(f"seam {cap.name} reached {len(cap.got)} times, expected 1")
    return cap.got[0]


@torch.no_grad()
def forward_readings(m, inp, labels, layout, idx, depth: int, device: str) -> dict:
    """One labelled eval forward at forced depth ``depth``: per-token coda and head CEs
    (per rollout and under the mixture), the exit separation, the self-check deviations."""
    from morph.model.fused_ce import fused_linear_label_logprob
    from morph.model.tul_spandec import span_slots
    from morph.model.tul_spandec_parallel import mixture_token_nll
    tc = m.cfg.tul
    K = max(int(m._code_enum_k), 1)
    table = torch.full(layout.slot_index.shape, depth, dtype=torch.long)
    coda_name = "_enum_mix_losses" if K > 1 else "_tul_group_losses"
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    has_head = m.tul_spandec_par is not None
    hc = _Capture(m, "_tul_spandec_par_loss") if has_head else contextlib.nullcontext()
    with ac, _Capture(m, coda_name) as cc, hc:
        out = m(inp.to(device), labels=labels.to(device), slot_layout=layout,
                slot_depths=table)
    xh = _one(cc)[0][0]                                              # [K*B, L, C]
    B, L = labels.shape
    lab = labels.to(device)
    w_head = m.embed.lm_weight()
    with ac:
        lp = fused_linear_label_logprob(
            xh.reshape(-1, xh.shape[-1]), w_head, lab.repeat(K, 1).reshape(-1),
            chunk_size=m.cfg.ce_chunk_size, mask_token_id=tc.slot_id).view(K, B, L).float()
    row_w, _p, _z = m._tul_half_weights(lab, layout)
    wmask = (row_w.view(B, L) * (lab != -100)).bool()
    if K > 1:
        coda_mix = m._enum_position_nll(lp, inp.to(device), layout)             # [B, L]
    else:
        coda_mix = -lp[0]
    res = {"coda_idx": idx[wmask.cpu()].numpy(),
           "coda_mix": coda_mix[wmask].cpu().numpy(),
           "coda_code": (-lp[:, wmask]).cpu().numpy() if K > 1 else None,
           "dev_coda": abs(float(coda_mix[wmask].double().mean()) - float(out["ce_tokens"]))}
    if has_head:
        h_slots = _one(hc)[0][0]                           # [K*B, S, (n,) C]
        head = m.tul_spandec_par
        B0 = B
        ids, valid = head.targets(inp.to(device), layout)
        with ac:
            z = m._readout(h_slots)
            zr = z.view(K, B0, *z.shape[1:])
            hlp, sup, val_s = head.token_logp(zr, ids, valid, w_head.detach(),
                                              chunk_size=m.cfg.ce_chunk_size,
                                              mask_token_id=tc.slot_id)
        hlp = hlp.float()
        par_mix = mixture_token_nll(hlp, val_s) if K > 1 else -hlp[0]
        sidx, _v = span_slots(idx.to(ids.device), layout, int(ids.shape[-1]),
                              shift=head.target_offset)
        J = int(ids.shape[-1])
        off = torch.arange(J, device=ids.device).view(1, 1, J).expand_as(ids)
        first = sidx[sup][:, 0]
        slot_of = torch.arange(int(sup.sum()), device=ids.device).view(-1, 1).expand_as(val_s)
        res.update({
            "head_idx": sidx[valid].cpu().numpy(),
            "head_first": first[slot_of[val_s]].cpu().numpy(),
            "head_off": off[valid].cpu().numpy(),
            "par_mix": par_mix.cpu().numpy(),
            "par_code": (-hlp).cpu().numpy() if K > 1 else None,
            "dev_par": abs(float(par_mix.double().mean()) - float(out["par_ce"])),
        })
        if K > 1:
            hv = h_slots.detach().float().flatten(2).view(K, B0, h_slots.shape[1], -1)
            v = layout.slot_valid.to(hv.device)
            d = []
            for i in range(K):
                for j in range(i + 1, K):
                    d.append((hv[i] - hv[j]).pow(2).mean(-1).sqrt()[v])
            dist = torch.stack(d).mean(0)                                  # [n_valid]
            rms = hv.pow(2).mean(-1).sqrt().mean(0)[v]
            res["sep_sum"] = float(dist.double().sum())
            res["rms_sum"] = float(rms.double().sum())
            res["n_slots"] = int(v.sum())
            # the same pairwise RMS distance restricted to the code subspace span(u_k),
            # per Hyper-Connection stream (the code is added to every stream), over the
            # same n*C denominator: sep_code <= sep_abs, and sep_code / sep_abs is the
            # square root of the separation's energy share in the code directions
            # (gpt.md C's "code-direction component", a diagnostic).
            from morph.model.tul_code_enum import gram_schmidt_rows
            q = gram_schmidt_rows(m.tul_code_enum.basis.detach().float())    # [K-1, C]
            C = q.shape[-1]
            hc = h_slots.detach().float().reshape(K, B0, h_slots.shape[1], -1, C)
            pr = hc @ q.T                                                  # [K,B0,S,n,K-1]
            n_str = hc.shape[3]
            dc = [((pr[i] - pr[j]).pow(2).sum((-1, -2)) / (n_str * C)).sqrt()[v]
                  for i in range(K) for j in range(i + 1, K)]
            res["sep_code_sum"] = float(torch.stack(dc).mean(0).double().sum())
    return res


def _blocks(idx: np.ndarray) -> np.ndarray:
    _u, inv = np.unique(idx // BLOCK, return_inverse=True)
    return inv


def _ci(a: np.ndarray, b: np.ndarray | None, blocks: np.ndarray, sel: np.ndarray | None,
        n_boot: int, seed: int) -> dict:
    """Token-weighted mean(a) - mean(b) (b None: mean(a)) with the paired block bootstrap."""
    if sel is None:
        sel = np.ones(a.shape[0], dtype=bool)
    nb = int(blocks.max()) + 1
    sa = np.bincount(blocks[sel], weights=a[sel].astype(np.float64), minlength=nb)
    sb = (np.zeros(nb) if b is None
          else np.bincount(blocks[sel], weights=b[sel].astype(np.float64), minlength=nb))
    c = np.bincount(blocks[sel], minlength=nb).astype(np.float64)
    keep = c > 0
    r = paired_bootstrap_ci(sa[keep], sb[keep], c[keep], n_boot=n_boot, seed=seed)
    r["n_tokens"] = int(sel.sum())
    return r


def _fmt(d: dict) -> str:
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['n_tokens']}"


def score_arm(m, batches, depths: list[int], device: str, tol: float) -> dict:
    """All depths of one arm: concatenated per-token arrays and the exit separations."""
    per: dict[int, dict] = {}
    devs = {"coda": 0.0, "par": 0.0}
    for d in depths:
        parts: dict[str, list] = {}
        sep = [0.0, 0.0, 0, 0.0]
        for inp, labels, layout, idx in batches:
            r = forward_readings(m, inp, labels, layout.to(device), idx, d, device)
            devs["coda"] = max(devs["coda"], r.pop("dev_coda"))
            if "dev_par" in r:
                devs["par"] = max(devs["par"], r.pop("dev_par"))
            if "sep_sum" in r:
                sep[0] += r.pop("sep_sum")
                sep[1] += r.pop("rms_sum")
                sep[2] += r.pop("n_slots")
                sep[3] += r.pop("sep_code_sum")
            for k, v in r.items():
                if v is not None:
                    parts.setdefault(k, []).append(v)
        arr = {k: np.concatenate(v, axis=-1) for k, v in parts.items()}
        if sep[2]:
            arr["sep_abs"] = sep[0] / sep[2]
            arr["sep_rel"] = sep[0] / sep[1]
            arr["sep_code"] = sep[3] / sep[2]
        per[d] = arr
        print(f"    depth {d}: coda {arr['coda_mix'].mean():.4f}"
              + (f"  par {arr['par_mix'].mean():.4f}" if "par_mix" in arr else "")
              + (f"  sep_abs {arr['sep_abs']:.4f} sep_rel {arr['sep_rel']:.4f}"
                 if "sep_abs" in arr else ""), flush=True)
    if devs["coda"] > tol or devs["par"] > tol:
        raise RuntimeError(f"offline recomputation disagrees with the forward: {devs} "
                           f"(tol {tol})")
    return {"per": per, "devs": devs}


def k36_clauses(p4: dict, p1: dict, cblk: np.ndarray, hblk: np.ndarray, depths: list[int],
                n_boot: int, seed: int) -> dict:
    """The K3-K6 twins of the P-3 / P-4 K1-K6 clauses (empty when depth 3 or 6 was not
    swept). Keys end the clause dict, so every earlier line of the summary is unchanged."""
    if 3 not in depths or 6 not in depths:
        return {}
    return {
        "P-3 e4 par K3-K6": _ci(p4[3]["par_mix"], p4[6]["par_mix"], hblk, None, n_boot, seed),
        "P-3 e1 par K3-K6": _ci(p1[3]["par_mix"], p1[6]["par_mix"], hblk, None, n_boot, seed),
        "P-4 e4 coda K3-K6 (mixture)": _ci(p4[3]["coda_mix"], p4[6]["coda_mix"], cblk, None,
                                           n_boot, seed),
        "P-4 e1 coda K3-K6": _ci(p1[3]["coda_mix"], p1[6]["coda_mix"], cblk, None, n_boot,
                                 seed),
    }


def abs_path(p: str) -> str:
    """A relative checkpoint path is taken against the repo root."""
    from _build import ROOT
    return p if p.startswith("/") else os.path.join(ROOT, p)


def load_arm(config: str, path: str, device: str, ovr: list[str]):
    """``(model in eval mode, step, cfg, tul runtime)`` for one arm's checkpoint, built from
    its Hydra config with the eager kernels (``model.use_kernels=false``) plus ``ovr``.
    Shared with ``lxtul_e_stage2_score.py``."""
    from _build import ROOT, build_cfg
    if f"{ROOT}/scripts" not in sys.path:
        sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402

    from morph.training.tul_setup import build_tul_runtime
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    rt = build_tul_runtime(cfg)
    m, step = load_ckpt(cfg, abs_path(path), device, rt.model_cfg)
    return m.eval(), step, cfg, rt


def val_batches(cfg, rt, rows: int, batch: int) -> tuple[list, int]:
    """The first ``rows`` packed validation rows (the trainer's packer over the validation
    stream from its start), in batches of ``batch``, with the stream index of every
    position. RAISES when the stream packs fewer rows than asked."""
    from _rows import pack_rows, stream_from_loader

    from morph.training.data import create_dataloader
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, rows * row_tokens)
    batches = pack_rows(stream, rt, cfg, batch, False)[:-(-rows // batch)]
    got = sum(b[0].shape[0] for b in batches)
    if got < rows:
        raise SystemExit(f"packed {got} rows, asked for {rows}")
    return batches, got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e4", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--ruler", default=RULER, help="'none' skips the ruler pairing")
    ap.add_argument("--config-e4", default="tul_slot_spandec_strict_e4")
    ap.add_argument("--config-e1", default="tul_slot_spandec_strict_e1")
    ap.add_argument("--config-ruler", default="tul_slot_spandec_strict")
    ap.add_argument("--depths", default="1,2,3,4,5,6")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ovr", action="append", default=[],
                    help="extra Hydra override for every arm (repeatable). A CPU run needs "
                         "model.tg_scoped_kernels=false and model.hc_use_kernel=false.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    if 1 not in depths or 6 not in depths:
        raise SystemExit("--depths must include 1 and 6 (K1-K6 and the separation ratio)")
    t0 = time.time()
    _abs = abs_path

    def _load(config: str, path: str):
        return load_arm(config, path, a.device, a.ovr)

    m4, s4, cfg4, rt4 = _load(a.config_e4, a.e4)
    if m4._code_enum_k < 2 or m4.tul_spandec_par is None:
        raise SystemExit("--e4 must be a code_enum_k > 1 arm with the parallel head")
    batches, rows = val_batches(cfg4, rt4, a.rows, a.batch)

    print(f"e4 {a.e4} (step {s4})", flush=True)
    r4 = score_arm(m4, batches, depths, a.device, a.tol)
    del m4
    torch.cuda.empty_cache() if a.device == "cuda" else None
    m1, s1, _c1, _r1 = _load(a.config_e1, a.e1)
    if m1._code_enum_k != 0 or m1.tul_spandec_par is None:
        raise SystemExit("--e1 must be the K = 1 parallel-head arm")
    print(f"e1 {a.e1} (step {s1})", flush=True)
    r1 = score_arm(m1, batches, depths, a.device, a.tol)
    del m1
    rr = None
    if a.ruler != "none":
        mr, sr, _cr, _rr = _load(a.config_ruler, a.ruler)
        print(f"ruler {a.ruler} (step {sr})", flush=True)
        rr = score_arm(mr, batches, [6], a.device, a.tol)
        del mr

    nb, sd = a.n_boot, a.seed
    p4, p1 = r4["per"], r1["per"]
    # the token sets must be identical across arms and depths (same rows, same labels)
    for d in depths:
        for p in (p4, p1):
            assert np.array_equal(p[d]["coda_idx"], p4[1]["coda_idx"])
            assert np.array_equal(p[d]["head_idx"], p4[1]["head_idx"])
    cblk = _blocks(p4[1]["coda_idx"])
    hblk = _blocks(p4[1]["head_first"])
    K = p4[6]["par_code"].shape[0]
    res: dict = {"rows": rows, "batch": a.batch, "depths": depths, "K": int(K),
                 "e4": _abs(a.e4), "e1": _abs(a.e1), "step_e4": s4, "step_e1": s1,
                 "ruler": None if rr is None else _abs(a.ruler),
                 "n_coda_tokens": int(p4[1]["coda_idx"].shape[0]),
                 "n_head_tokens": int(p4[1]["head_idx"].shape[0]),
                 "self_check_max_abs_dev": {"e4": r4["devs"], "e1": r1["devs"],
                                            "ruler": None if rr is None else rr["devs"]},
                 "block_tokens": BLOCK, "by_depth": {}}
    for d in depths:
        e4d, e1d = p4[d], p1[d]
        best_c = int(np.argmin(e4d["coda_code"].mean(1)))
        best_p = int(np.argmin(e4d["par_code"].mean(1)))
        res["by_depth"][d] = {
            "e4_par_mix": _ci(e4d["par_mix"], None, hblk, None, nb, sd),
            "e1_par": _ci(e1d["par_mix"], None, hblk, None, nb, sd),
            "e4_coda_mix": _ci(e4d["coda_mix"], None, cblk, None, nb, sd),
            "e1_coda": _ci(e1d["coda_mix"], None, cblk, None, nb, sd),
            "e4_coda_code": [float(x) for x in e4d["coda_code"].mean(1)],
            "e4_par_code": [float(x) for x in e4d["par_code"].mean(1)],
            "e4_coda_width_gain_best": _ci(e4d["coda_code"][best_c], e4d["coda_mix"], cblk,
                                           None, nb, sd) | {"code": best_c},
            "e4_coda_width_gain_per_code": [
                _ci(e4d["coda_code"][k], e4d["coda_mix"], cblk, None, nb, sd)
                for k in range(K)],
            "e4_par_width_gain_best": _ci(e4d["par_code"][best_p], e4d["par_mix"], hblk,
                                          None, nb, sd) | {"code": best_p},
            "e4_exit_sep_abs": e4d["sep_abs"], "e4_exit_sep_rel": e4d["sep_rel"],
        }
    # the clauses
    P = {}
    P["P-1 e1_par - e4_par_mix @6"] = _ci(p1[6]["par_mix"], p4[6]["par_mix"], hblk, None,
                                         nb, sd)
    P["P-2 e4 coda width gain (best code) @6"] = res["by_depth"][6]["e4_coda_width_gain_best"]
    P["P-3 e4 par K1-K6"] = _ci(p4[1]["par_mix"], p4[6]["par_mix"], hblk, None, nb, sd)
    P["P-3 e1 par K1-K6"] = _ci(p1[1]["par_mix"], p1[6]["par_mix"], hblk, None, nb, sd)
    P["P-3 exit sep ratio abs d6/d1"] = p4[6]["sep_abs"] / p4[1]["sep_abs"]
    P["P-3 exit sep ratio rel d6/d1"] = p4[6]["sep_rel"] / p4[1]["sep_rel"]
    P["P-4 e4 coda K1-K6 (mixture)"] = _ci(p4[1]["coda_mix"], p4[6]["coda_mix"], cblk, None,
                                          nb, sd)
    P["P-4 e1 coda K1-K6"] = _ci(p1[1]["coda_mix"], p1[6]["coda_mix"], cblk, None, nb, sd)
    P["e4 coda mix - e1 coda @6"] = _ci(p4[6]["coda_mix"], p1[6]["coda_mix"], cblk, None,
                                      nb, sd)
    if rr is not None:
        pr = rr["per"][6]
        assert np.array_equal(pr["coda_idx"], p4[1]["coda_idx"])
        P["P-5 e4 coda mix - ruler coda @6"] = _ci(p4[6]["coda_mix"], pr["coda_mix"], cblk,
                                                  None, nb, sd)
        P["e1 coda - ruler coda @6"] = _ci(p1[6]["coda_mix"], pr["coda_mix"], cblk, None,
                                         nb, sd)
    # K3-K6 beside K1-K6 (appended, 2026-09-26): forced depth 1 is off the training
    # distribution (T ~ Poisson(6) clamped to [1, 8] draws depth 1 about 1.7 % of the time),
    # so K3-K6 is the within-distribution read of the same curve. Same bootstrap.
    P.update(k36_clauses(p4, p1, cblk, hblk, depths, nb, sd))
    res["clauses"] = P
    res["wall_s"] = round(time.time() - t0, 1)
    npz = a.out.rsplit(".", 1)[0] + ".tokens.npz"
    save = {"coda_idx": p4[1]["coda_idx"].astype(np.int64),
            "head_idx": p4[1]["head_idx"].astype(np.int64),
            "head_first": p4[1]["head_first"].astype(np.int64),
            "head_off": p4[1]["head_off"].astype(np.int16)}
    for d in depths:
        save[f"e4_coda_mix_{d}"] = p4[d]["coda_mix"]
        save[f"e4_coda_code_{d}"] = p4[d]["coda_code"]
        save[f"e4_par_mix_{d}"] = p4[d]["par_mix"]
        save[f"e4_par_code_{d}"] = p4[d]["par_code"]
        save[f"e1_coda_{d}"] = p1[d]["coda_mix"]
        save[f"e1_par_{d}"] = p1[d]["par_mix"]
    if rr is not None:
        save["ruler_coda_6"] = rr["per"][6]["coda_mix"]
    np.savez_compressed(npz, **save)
    res["tokens_npz"] = npz
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"Stage 1: rows {rows}, {res['n_coda_tokens']} coda tokens, "
          f"{res['n_head_tokens']} head tokens")
    for k, v in P.items():
        print(f"  {k:40s} {_fmt(v) if isinstance(v, dict) else f'{v:.4f}'}")
    print(f"  self-check max |dev|: {res['self_check_max_abs_dev']}")
    print(f"wrote {a.out} and {npz}")


if __name__ == "__main__":
    main()
