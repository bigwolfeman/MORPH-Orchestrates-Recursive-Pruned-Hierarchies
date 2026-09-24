import numpy as np
from math import exp, factorial
lams=np.linspace(0,1.3,1301)
S=lambda l,T: sum(l**s for s in range(T))
# Poisson(6) clamped [1,8] law
p=np.array([exp(-6)*6**k/factorial(k) for k in range(0,40)])
P={T:0.0 for T in range(1,9)}
for k,pk in enumerate(p): P[min(max(k,1),8)]+=pk
# 1) new evidence: FP term (normalized) and profile loss (b optimal) vs lambda
def fp_new(l,T): return ((1-l)**2*S(l*l,T-1)+1)/S(l*l,T)
def prof(l):  # E_T min_b? b shared across T: minimize sum_T P_T sum_s (b l^s -1)^2 over b
    A=sum(P[T]*S(l*l,T) for T in P); B=sum(P[T]*S(l,T) for T in P); C=sum(P[T]*T for T in P)
    return C-B*B/A
fp=np.array([sum(P[T]*fp_new(l,T) for T in P) for l in lams])
pr=np.array([prof(l) for l in lams])
i1=np.searchsorted(lams,1.0)
print("new-evidence: E_T FP antitone on [0,1]:", bool(np.all(np.diff(fp[:i1+1])<=1e-12)), " FP(0)=%.3f FP(1)=%.3f"%(fp[0],fp[i1]))
print("new-evidence: profile loss argmin lambda = %.3f (b optimized, Poisson depth)"%lams[np.argmin(pr)])
for mu in (0.0,0.1,1.0):
    tot=pr+mu*fp; print("   with FP weight %.1f: argmin lambda = %.3f"%(mu,lams[np.argmin(tot)]))
# 2) repeated source, entry x0=1, target m: minimize E_T (r*roll-m)^2 + mu*FP over (b) for each lam, r=1
def roll(l,b,x0,T):
    h=x0
    for _ in range(T): h=l*h+b
    return h
for m in (1.0,1.5):
  for mu in (0.0,1.0):
    best=[]
    for l in np.linspace(0,0.99,100):
        bs=np.linspace(-1,3,801)
        L=[sum(P[T]*((roll(l,b,1.0,T)-m)**2 + mu*((roll(l,b,1.0,T)-roll(l,b,1.0,T-1))/roll(l,b,1.0,T))**2) for T in P) for b in bs]
        best.append((min(L),l))
    lo=min(best)[0]; flat=[l for L,l in best if L<lo+1e-6]
    print(f"repeated source m={m} fp={mu}: min loss {lo:.2e}; lambdas within 1e-6 of min: {flat[0]:.2f}..{flat[-1]:.2f} ({len(flat)} of 100)")
