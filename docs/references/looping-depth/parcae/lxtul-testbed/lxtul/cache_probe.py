"""How much can an exact-copy cache recover with NO training? Fixed gate on trained models.

    python -m lxtul.cache_probe --runs /home/wolfe/parcae-runs plain-5k lxtul-5k --out f.json

Each model scores the 480 rows at depth 6 (per token). The cache (`lxtul.cache.copy_probs`, in
token order) is mixed in with a CONSTANT gate: log-weights [0, b_bi, b_uni] over the available
sources, b on a grid; the best grid point is reported. Two numbers per run: the best mixed CE and
the gain over the model alone. Fitting two scalars on 510k tokens is a negligible optimism.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from lxtul.cache import copy_probs, mix_logp
from lxtul.data import morph_cfg, tul_runtime
from lxtul.evaluate import eval_rows, plain_view
from lxtul.gap_probe import load_run, token_nll

GRID = [-8.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("/home/wolfe/parcae-runs"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    res = {}
    rows = None
    for name in a.names:
        model, kind, cfg = load_run(a.runs / name)
        model.cuda()
        if rows is None:
            mcfg = morph_cfg(str(cfg.morph.config)); rt = tul_runtime(mcfg)
            slot_id = int(rt.data_cfg.slot_id)
            rows = eval_rows(mcfg, rt, 480, 6)
        lpm, cs = [], {k: [] for k in ("p_bi", "n_bi", "p_uni", "n_uni")}
        for x, y, lay in rows:
            x, y = x.cuda(), y.cuda()
            for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
                setattr(lay, f, getattr(lay, f).cuda())
            nll = token_nll(model, kind, x, y, lay, slot_id)          # per row, token order
            xp, yp = plain_view(x, y, lay, slot_id)
            ntok = (~lay.slot_mask).sum(1)
            is_tok = torch.arange(xp.shape[1], device=xp.device)[None] < ntok[:, None]
            c = copy_probs(xp, yp, is_tok)
            for b, v in enumerate(nll):
                sel = yp[b] >= 0                    # token_nll's own selection, same order
                if int(sel.sum()) != len(v):
                    raise ValueError("cache and model scored different tokens")
                lpm.append(-v.double())
                for k in cs:
                    cs[k].append(c[k][b][sel].double().cpu())
        lpm = torch.cat(lpm)
        c = {k: torch.cat(v) for k, v in cs.items()}
        base = float(-lpm.mean())
        best = (base, None)
        for b1, b2 in itertools.product(GRID, GRID):
            g = torch.tensor([0.0, b1, b2], dtype=torch.float64).expand(len(lpm), 3)
            ce = float(-mix_logp(lpm, g, c).mean())
            if ce < best[0]:
                best = (ce, (b1, b2))
        res[name] = {"ce_model": base, "ce_mixed": best[0], "gain": base - best[0],
                     "gate_logits": best[1], "n_tokens": len(lpm)}
        print(f"{name:<14} model {base:.4f}  +cache {best[0]:.4f}  gain {base - best[0]:+.4f}  "
              f"gate {best[1]}", flush=True)
        del model
        torch.cuda.empty_cache()
    a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
