"""HC Cayley mixing probe — static (bias-level) read of the learned stream routing.

Prereg: lab/experiments/planned/2026-08-31-loop-killer-bisect.md, arm BHC stage 1
(P-HCprobe). Question: did the HyperConnectionResidual learn to route AROUND the
sublayer branch in the core layers (bypass signature)?

Structure facts (morph/model/hyper_connections.py):
- Per token: [Hpre|Hpost|Hres] = proj(norm(vec(x_streams))); Hpre row-stochastic
  (read), Hpost column-stochastic (write), Hres ORTHOGONAL via exact Cayley
  (singular values exactly 1 — it can never attenuate streams).
- Consequences: the total branch-write mass is FIXED (columns of Hpost sum to 1,
  so sum_i Hpost_row[i] = n), and the mixer cannot shrink anything. The ONLY
  bypass route is read-side: Hpre rows putting ~0 weight on the streams that
  Hpost writes the branch output into.
- The mappings are INPUT-DEPENDENT (dynamic HC). A static probe cannot see the
  realized per-token routing; it CAN see (a) the token-independent baseline
  (h_tilde = proj.bias exactly, since the weight term vanishes at zero input
  correlation) and (b) the per-block weight norms (how much input-dependence
  each mapping has). Both are reported; the limitation is stated in the output.

Metric: read-write ALIGNMENT = sum_i Hpre_cm[i] * (Hpost_row[i] / n), computed
from the bias-level mappings. At init/uniform this is exactly 1/n = 0.25 — the
prereg threshold. Below 0.25: the read avoids branch-written streams (bypass
signature). Above: the read seeks them.

Usage:
  python lab/divergence/hc_mixing_probe.py \
    [--ckpt checkpoints/morph/notul-l2nc/step_4500.pt] [--n 4] [--alpha 0.1]
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import torch


def cayley_from_bias(res_raw: torch.Tensor, alpha: float) -> torch.Tensor:
    n = res_raw.shape[-1]
    I = torch.eye(n)
    B = (alpha * 0.5) * (res_raw - res_raw.T)
    B2 = B @ B
    p = 0.5 * (B * B).sum()
    Pf = B[0, 1] * B[2, 3] - B[0, 2] * B[1, 3] + B[0, 3] * B[1, 2]
    q = Pf * Pf
    num = (I + 2.0 * B + B2) @ ((1.0 + p) * I + B2)
    return num / (1.0 + p + q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/morph/notul-l2nc/step_4500.pt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    n = a.n

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st = ck.get("model", ck)
    st = {k.replace("_orig_mod.", ""): v for k, v in st.items()}

    rows = []
    for key in sorted(st):
        m = re.match(r"(prelude|core|coda)\.(\d+)\.(mrr_attn|mrr_mlp)\.proj\.bias$", key)
        if not m:
            continue
        sec, idx, which = m.group(1), int(m.group(2)), m.group(3)
        bias = st[key].float().reshape(3, n, n)
        w = st[key.replace(".bias", ".weight")].float().reshape(3, n, n, -1)
        pre_raw, post_raw, res_raw = bias[0], bias[1], bias[2]
        Hpre = torch.softmax(pre_raw / a.tau, dim=-1)         # row-stochastic [n,n]
        Hpost = torch.softmax(post_raw / a.tau, dim=-2)       # col-stochastic
        Hpre_cm = Hpre.mean(dim=0)                            # read weight per stream [n]
        Hpost_row = Hpost.sum(dim=-1)                         # branch write per stream [n]
        align = float((Hpre_cm * Hpost_row / n).sum())
        Hres = cayley_from_bias(res_raw, a.alpha)
        rot = float((Hres - torch.eye(n)).norm())             # mixer rotation magnitude
        wn = [float(w[i].norm()) for i in range(3)]           # input-dependence per block
        rows.append((sec, idx, which, align, rot, Hpre_cm, Hpost_row, wn))

    if not rows:
        print(f"NO HC proj params found in {a.ckpt}")
        return

    print(f"ckpt: {a.ckpt}  (step {ck.get('step', '?')})")
    print("STATIC bias-level read; realized mappings are input-dependent — this is")
    print("the token-independent baseline. alignment: 0.25 = uniform/init;")
    print("<0.25 = read avoids branch-written streams (bypass signature).")
    print("wnorm[pre,post,res]: input-dependence scale per mapping block (init std")
    print("0.1/sqrt(n*d) => expected init block norm ~ sqrt(16*4096)*0.1/64 ~ 0.4).")
    print(f"{'layer':22s} {'align':>7s} {'rot|Hres-I|':>11s} {'wnorm[pre,post,res]':>22s}  "
          "read Hpre_cm | write Hpost_row")
    agg = defaultdict(list)
    aggw = defaultdict(list)
    for sec, idx, which, align, rot, pre, post, wn in rows:
        name = f"{sec}.{idx}.{which}"
        agg[sec].append(align)
        aggw[sec].append(wn)
        print(f"{name:22s} {align:7.4f} {rot:11.4f} "
              f"[{' '.join(f'{x:6.2f}' for x in wn)}]  "
              f"[{' '.join(f'{x:.3f}' for x in pre)}] | [{' '.join(f'{x:.3f}' for x in post)}]")
    for sec in ("prelude", "core", "coda"):
        if agg[sec]:
            v = agg[sec]
            w = aggw[sec]
            mw = [sum(x[i] for x in w) / len(w) for i in range(3)]
            print(f"AGG {sec:8s} mean alignment {sum(v)/len(v):.4f}  "
                  f"mean wnorm [pre,post,res]=[{mw[0]:.2f} {mw[1]:.2f} {mw[2]:.2f}]  (n={len(v)})")


if __name__ == "__main__":
    main()
