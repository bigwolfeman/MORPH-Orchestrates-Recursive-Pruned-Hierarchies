"""Toy looped net: does the per-pass map settle at the injection floor when the loop has no
multi-pass job, and rise above it when information must arrive over passes?

Architecture mirrors MORPH's core step in miniature: h_0 = input_norm(prelude(x)); each pass
= DiagonalInjection on a ctx slice (A init 0.447, dt init 1, 20 of 64 dims = 0.3125, the
same fraction as MORPH's 320/1024) then n_core pre-norm residual blocks (attention + MLP),
shared across passes. Per-sample Poisson(6) depth clamped to [1, 8], full BPTT, the
fixed-point term lambda * ||h_T - h_{T-1}||^2 / ||h_T||^2. Reader: one pre-norm block + a
linear head on h_T; optional BYPASS: the head also reads the prelude state directly.

Map instrument (core_map_fd.py's definition): at forced depth 6, eval, per pass t, perturb
every position jointly with d_i Gaussian scaled to eps * ||h_i||, gain_i = ||df_i|| / ||d_i||.
"""
import argparse, json, math, time
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--task", default="hop3")       # hopK | ownhopK | own
p.add_argument("--bypass", type=int, default=0)
p.add_argument("--fp", type=float, default=1.0)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--N", type=int, default=8)
p.add_argument("--d", type=int, default=64)
p.add_argument("--n_core", type=int, default=1)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--noinject", type=int, default=0)   # 1: e enters at pass 0 only (no re-injection)
p.add_argument("--io_attn", type=int, default=1)  # 0: prelude and coda are MLP-only (no lookups)
p.add_argument("--out", default="")
a = p.parse_args()
torch.set_num_threads(2)
torch.manual_seed(a.seed)
D, N, V, PD = a.d, a.N, 4, 16
C0, C1 = 32, 52                                   # ctx slice: 20 of 64 dims

class RMS(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * s.w

class Block(nn.Module):
    def __init__(s, d, H=2, attn=True):
        super().__init__(); s.H = H; s.attn = attn
        s.n1, s.n2 = RMS(d), RMS(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.f1 = nn.Linear(d, 4 * d); s.f2 = nn.Linear(4 * d, d)
    def forward(s, h):
        B, n, d = h.shape
        if not s.attn:
            return h + s.f2(F.gelu(s.f1(s.n2(h))))
        q, k, v = s.qkv(s.n1(h)).view(B, n, 3, s.H, d // s.H).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v)
        h = h + s.o(att.transpose(1, 2).reshape(B, n, d))
        return h + s.f2(F.gelu(s.f1(s.n2(h))))

class Inj(nn.Module):
    def __init__(s):
        super().__init__()
        s.logA = nn.Parameter(torch.full((C1 - C0,), math.log(0.447)))
        s.logdt = nn.Parameter(torch.zeros(C1 - C0))
    def forward(s, h, e):
        A = s.logA.exp().clamp(max=0.9999)
        ctx = A * h[..., C0:C1] + s.logdt.exp() * e[..., C0:C1]
        return torch.cat([h[..., :C0], ctx, h[..., C1:]], -1)
    def floor(s, d):
        A = s.logA.exp().clamp(max=0.9999)
        return math.sqrt((d - (C1 - C0) + (A ** 2).sum().item()) / d)

class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Linear(2 * PD + V, D); s.pre = Block(D, attn=bool(a.io_attn)); s.inorm = RMS(D)
        s.inj = Inj(); s.core = nn.ModuleList([Block(D) for _ in range(a.n_core)])
        s.coda = Block(D, attn=bool(a.io_attn)); s.head = nn.Linear(D, V)
        s.byp = nn.Linear(D, V) if a.bypass else None
    def front(s, x):
        pre = s.pre(s.emb(x)); return pre, s.inorm(pre)
    def step(s, h, e, first):
        if a.noinject and not first:
            h = h  # e enters once: pass 0 only
        else:
            h = s.inj(h, e)
        for b in s.core: h = b(h)
        return h
    def read(s, hT, pre):
        y = s.head(s.coda(hT))
        return y + s.byp(pre) if s.byp is not None else y
    def forward(s, x, T):                    # T: [B] depths in 1..8
        pre, e = s.front(x); h = e; traj = [h]
        for t in range(int(T.max())):
            h = s.step(h, e, t == 0); traj.append(h)
        tr = torch.stack(traj, 0)               # [Tmax+1, B, N, D]
        idx = torch.arange(x.shape[0])
        hT, hTm = tr[T, idx], tr[T - 1, idx]
        fp = ((hT - hTm).pow(2).sum((-1, -2)) / hT.pow(2).sum((-1, -2))).mean()
        return s.read(hT, pre), fp, tr

g = torch.Generator().manual_seed(1234 + a.seed)
PC = torch.randn(N, PD, generator=g); PC = PC / PC.norm(dim=1, keepdim=True)

def batch(B, gen=None):
    ptr = torch.randint(0, N, (B, N), generator=gen)
    val = torch.randn(B, N, V, generator=gen)
    x = torch.cat([PC.expand(B, N, PD), PC[ptr], val], -1)
    t = a.task
    k = int(t[-1]) if t[-1].isdigit() else 0
    cur = torch.arange(N).expand(B, N).clone()
    for _ in range(k): cur = torch.gather(ptr, 1, cur)
    hop = torch.gather(val, 1, cur.unsqueeze(-1).expand(B, N, V))
    if t.startswith("sumhop"):
        acc = torch.zeros(B, N, V); cur = torch.arange(N).expand(B, N).clone()
        for _ in range(k):
            cur = torch.gather(ptr, 1, cur)
            acc = acc + torch.gather(val, 1, cur.unsqueeze(-1).expand(B, N, V))
        y = acc
    elif t.startswith("hop"): y = hop
    elif t.startswith("ownhop"): y = val + hop
    elif t == "own": y = val
    else: raise ValueError(t)
    return x, y

m = Model()
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
t0 = time.time()
for it in range(a.steps):
    x, y = batch(128)
    T = torch.poisson(torch.full((128,), 6.0)).long().clamp(1, 8)
    yh, fp, _ = m(x, T)
    loss = F.mse_loss(yh, y) + a.fp * fp
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 1000 == 0 or it == a.steps - 1:
        print(f"it {it} mse {F.mse_loss(yh, y).item():.4f} fp {fp.item():.4f} {time.time()-t0:.0f}s", flush=True)

# ── eval ──
m = m.double().eval()
PC = PC.double()
ge = torch.Generator().manual_seed(99)
x, y = batch(512, ge); x, y = x.double(), y.double()
res = {"args": vars(a), "floor": m.inj.floor(D),
       "A_mean": m.inj.logA.exp().mean().item(), "dt_mean": m.inj.logdt.exp().mean().item()}
with torch.no_grad():
    for Tf in (1, 2, 3, 6, 8):
        yh, _, _ = m(x, torch.full((512,), Tf))
        res[f"mse_d{Tf}"] = F.mse_loss(yh, y).item()
    res["var_y"] = y.var().item()
    res["depth_value_1_6"] = res["mse_d1"] - res["mse_d6"]
    pre, e = m.front(x); h = e
    passes = []
    gd = torch.Generator().manual_seed(7)
    for t in range(6):
        f0 = m.step(h, e, t == 0)
        gs = []
        for rep in range(4):
            dd = torch.randn(h.shape, generator=gd, dtype=h.dtype)
            dd = dd / dd.norm(dim=-1, keepdim=True) * 1e-5 * h.norm(dim=-1, keepdim=True)
            f1 = m.step(h + dd, e, t == 0)
            gs.append((f1 - f0).norm(dim=-1) / dd.norm(dim=-1))
        G = torch.stack(gs).flatten()
        q = torch.quantile(G.float(), torch.tensor([0.1, 0.5, 0.9, 0.99]))
        passes.append({"pass": t, "mean": G.mean().item(), "rms": G.pow(2).mean().sqrt().item(),
                       "p10": q[0].item(), "p50": q[1].item(), "p90": q[2].item(), "p99": q[3].item(),
                       "frac_gt1": (G > 1).float().mean().item(),
                       "norm_ratio": (f0.norm(dim=-1) / h.norm(dim=-1)).mean().item(),
                       "move": ((f0 - h).norm(dim=-1) / h.norm(dim=-1)).mean().item()})
        h = f0
    res["passes"] = passes
# backward (cotangent) gain per pass along the loss's own direction, forced depth 6
pre, e = m.front(x); pre, e = pre.detach(), e.detach()
hs = [e.clone().requires_grad_()]
for t in range(6):
    hs.append(m.step(hs[-1], e, t == 0)); hs[-1].retain_grad()
lossb = F.mse_loss(m.read(hs[-1], pre), y)
lossb.backward()
cots = [h.grad for h in hs]
for t in range(6):
    res["passes"][t]["cot_gain"] = (cots[t].norm() / cots[t + 1].norm()).item()
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
