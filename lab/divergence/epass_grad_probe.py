"""P-1 of lab/experiments/planned/2026-09-13-arc-trajectory-prefix.md: on the traj
checkpoint at 5000, is E_pass.grad.abs().sum() above 1 % of W_prefix.grad.abs().sum()
under the model's own training loss? One backward per row, grads accumulated over
--rows rows, eval mode (no token-state dropout; stated in the filing)."""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from _diag_common import Arm

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
ap.add_argument("--rows", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
label, config, path = a.ckpt.split("=", 2)
arm = Arm(label, config, path, "cuda", a.rows, row_batch=1)
m = arm.model
assert m.tul.E_pass is not None, "not a trajectory/exit_repeat model"
m.zero_grad(set_to_none=True)
n = 0
losses = []
for inp, labels, layout in arm.rows():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(inp, labels=labels, slot_layout=layout)
    out["loss"].backward()
    losses.append(float(out["loss"]))
    n += 1
ep = float(m.tul.E_pass.grad.abs().sum())
wp = float(m.tul.W_prefix.grad.abs().sum())
per_cell = [float(x) for x in m.tul.E_pass.grad.abs().sum(dim=1)]
res = {"label": label, "config": config, "step": arm.step, "rows": n,
       "mean_loss": sum(losses) / max(n, 1), "E_pass_grad_abs_sum": ep,
       "W_prefix_grad_abs_sum": wp, "ratio": ep / max(wp, 1e-12), "E_pass_grad_per_cell": per_cell,
       "E_pass_norm": float(m.tul.E_pass.detach().float().norm()),
       "E_pass_row_norms": [float(x) for x in m.tul.E_pass.detach().float().norm(dim=1)],
       "mode": "eval"}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("E_pass_grad_per_cell",)}, indent=None))
print("per-cell E_pass grad:", [round(x, 3) for x in per_cell])
