import sys, torch
TREE = "/home/wolfe/morph-scratch/abq/tree-h"
MASTER = "/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies"
sys.path[:0] = [TREE, MASTER + "/lab/divergence"]
import cell_lens as cl
import morph
spec = ("best=nowta_pointer_fast_spdbest=" + TREE +
        "/checkpoints/morph/ab-nowta-spdbest-h/step_5000.pt=model.spandec_ce_row_cap=false")
_, _, m, _ = cl.load_model(spec, "cuda")
m.eval()
val, _ = cl.load_batches("/home/wolfe/morph-scratch/abq/lens/batches.pt", "val", 40, "cuda", m)
acc, n = None, 0
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for x, y, layout in val:
        out = m(x, labels=y, slot_layout=layout)
        w = torch.stack([out[f"spandec_best_win_c{i}"].float() for i in range(4)])
        acc = w if acc is None else acc + w
        n += 1
print("win shares (mean over 40 val batches):", [round(float(v), 4) for v in acc / n])
