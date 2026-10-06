import sys, torch
torch.set_num_threads(1)
sys.path.insert(0, "/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies")
from morph.model.hyper_connections import cayley_orthogonal
path = sys.argv[1]
ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
sd = ck.get("model", ck.get("model_state_dict", ck)) if isinstance(ck, dict) else ck
if not isinstance(sd, dict) or not any("mrr_attn.proj" in k for k in sd):
    for k, v in (ck.items() if isinstance(ck, dict) else []):
        if isinstance(v, dict) and any("mrr_attn.proj" in kk for kk in v):
            sd = v; print("state dict under", k); break
keys = sorted(k for k in sd if k.endswith(".proj.bias") and "mrr_" in k)
print(len(keys), "HC proj biases")
n = 4
one = torch.ones(n, 1) / 2
Pd = torch.eye(n) - one @ one.T
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
for kb in keys:
    b = sd[kb].float().reshape(3, n, n)
    W = sd[kb[:-4] + "weight"].float()
    wn = W.norm(dim=1).reshape(3, n, n)          # per-output-row L2 norm
    Hpre_s = torch.softmax(b[0], -1).mean(-2)
    Hpost_s = torch.softmax(b[1], -2).sum(-1)
    Hres_s = cayley_orthogonal(b[2][None], 3, 0.1)[0]
    leak = (Pd @ Hres_s @ one).norm().item()
    print(f"{kb[:-10]:45s} |b| pre {b[0].abs().max():.2f} post {b[1].abs().max():.2f} res {b[2].abs().max():.2f} | "
          f"rowW typ pre {wn[0].mean():.2f} post {wn[1].mean():.2f} res {wn[2].mean():.2f} (x sqrt(nd)={W.shape[1]**0.5:.0f} max) | "
          f"static Hpre_cm {Hpre_s.numpy().round(2)} Hpost_row {Hpost_s.numpy().round(2)} |Hres-I| {(Hres_s-torch.eye(n)).abs().max():.2f} mean->diff leak {leak:.2f}")
