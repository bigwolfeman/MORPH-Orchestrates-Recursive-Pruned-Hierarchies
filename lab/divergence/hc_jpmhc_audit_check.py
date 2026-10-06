import sys, torch
torch.set_num_threads(1)
sys.path.insert(0, "/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies")
from morph.model.hyper_connections import cayley_orthogonal, HyperConnectionResidual
torch.manual_seed(0)
n, alpha = 4, 0.1
I = torch.eye(n, dtype=torch.float64)

def paper_iter(A, s, alpha=0.1):
    W = A - A.transpose(-1, -2)
    Y = I + alpha * W
    for _ in range(s):
        Y = I + (alpha / 2) * W @ (I + Y)
    return Y

def true_cayley(A, alpha=0.1):
    B = (alpha / 2) * (A - A.transpose(-1, -2))
    return torch.linalg.solve(I - B, I + B)

print("== 1. closed form vs true Cayley vs paper iteration (fp64) ==")
for scale in [0.1, 1.0, 5.0, 20.0, 100.0]:
    A = scale * torch.randn(2000, n, n, dtype=torch.float64)
    Yc = cayley_orthogonal(A, 3, alpha)
    Yt = true_cayley(A, alpha)
    Y2 = paper_iter(A, 2); Y3 = paper_iter(A, 3)
    orth = lambda Y: (Y.transpose(-1, -2) @ Y - I).abs().amax().item()
    smax = lambda Y: torch.linalg.svdvals(Y).amax().item()
    print(f"|A| entry std {scale:6.1f}: closed-vs-true {(Yc-Yt).abs().max().item():.2e}  "
          f"orth(closed) {orth(Yc):.1e}  orth(paper s=2) {orth(Y2):.1e} smax(s=2) {smax(Y2):.4f}  "
          f"orth(s=3) {orth(Y3):.1e} smax(s=3) {smax(Y3):.4f}  |paper s=2 - true| {(Y2-Yt).abs().max().item():.1e}")

print("== 2. bf16 rounding of Hres (forward casts Hres.to(bf16)) ==")
for th_scale in [0.003, 0.01, 0.03, 0.06, 0.1, 0.3]:
    A = th_scale * torch.randn(20000, n, n, dtype=torch.float64) / (alpha/2) / 2**0.5
    Y = true_cayley(A, alpha)
    Yb = Y.to(torch.bfloat16).to(torch.float64)
    sv = torch.linalg.svdvals(Yb)
    print(f"skew entry std ~{th_scale:5.3f}: bf16 Hres sigma_max mean {sv[:,0].mean().item():.5f} max {sv[:,0].max().item():.5f}"
          f"  sigma_min mean {sv[:,-1].mean().item():.5f}  log-det mean {torch.log(sv).sum(-1).mean().item():+.2e}")
# Repeated application of the SAME bf16-rounded small rotation (weight-tied loop): gain after 96 applications
A = 0.02 * torch.randn(5000, n, n, dtype=torch.float64) / (alpha/2) / 2**0.5
Yb = true_cayley(A, alpha).to(torch.bfloat16).to(torch.float64)
P = torch.linalg.matrix_power(Yb, 96)
print(f"96 applications of one bf16 Hres (skew std 0.02): sigma_max of product mean {torch.linalg.svdvals(P)[:,0].mean().item():.4f}"
      f" max {torch.linalg.svdvals(P)[:,0].max().item():.4f}")

print("== 3. mean-subspace leakage: |(I - 11^T/n) H 1/sqrt(n)| ==")
one = torch.ones(n, 1, dtype=torch.float64) / n**0.5
Pd = I - one @ one.T
for scale in [0.1, 1.0, 10.0, 50.0]:
    A = scale * torch.randn(5000, n, n, dtype=torch.float64)
    Y = true_cayley(A, alpha)
    leak = (Pd @ Y @ one).norm(dim=(-2, -1))
    print(f"res entry std {scale:5.1f}: mean->difference leak mean {leak.mean().item():.4f} max {leak.max().item():.4f}"
          f"  (doubly-stochastic mixer: exactly 0)")

print("== 4. init of the MORPH module (d=1024, init_gain 0.1) ==")
torch.manual_seed(0)
for d in [768, 1024]:
    m = HyperConnectionResidual(d, 4, 1.0, 3, 0.1, 0.1, use_kernel=False)
    x = torch.randn(2, 256, d).unsqueeze(2).expand(2, 256, 4, d).contiguous()
    with torch.no_grad():
        Hpre, Hpost, Hres = m._mappings(x)
        print(f"d={d}: Hpre_cm dev from 1/4 max {(Hpre.mean(-2)-0.25).abs().max().item():.4f}; "
              f"Hpost_row dev from 1 max {(Hpost.sum(-1)-1).abs().max().item():.4f}; "
              f"Hres-I max {(Hres-torch.eye(4)).abs().max().item():.2e}")
        y = torch.randn(2, 256, d)
        out = m(x, lambda z: y)
        ref = x + y.unsqueeze(2)
        print(f"   out vs x + F(mean x): max abs diff {(out-ref).abs().max().item():.4f} (|y| entry ~1); "
              f"stream-difference energy share of out {((out-out.mean(2,keepdim=True))**2).sum().item()/(out**2).sum().item():.2e}")
