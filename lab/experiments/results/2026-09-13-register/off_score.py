"""Score an offset arm against the strict ruler: P-6 (token-paired depth-6 CE, 480 rows),
P-7 (worth profile, zero mode, offset bin 16+), P-8 (off3 vs off2) from the runner's JSONs."""
import json, sys, numpy as np
R = '/home/wolfe/morph-scratch/arc/results'
def sweep(name, d):
    p = f'{R}/{d}/sweep_{name}_5000.json'; return json.load(open(p))[name]
def worth(name, d):
    p = f'{R}/{d}/worth_{name}_5000.json'; return json.load(open(p))[name]
def paired(a, b, depth='6', nboot=2000, seed=0):
    na, nb = np.array(a['row_n_tokens']), np.array(b['row_n_tokens'])
    assert (na == nb).all(), 'row token counts differ: not token-paired'
    sa, sb = np.array(a['row_ce_sum'][depth]), np.array(b['row_ce_sum'][depth])
    diff = (sa - sb).sum() / na.sum()
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(na), (nboot, len(na)))
    boots = ((sa - sb)[idx].sum(1) / na[idx].sum(1)); lo, hi = np.percentile(boots, [2.5, 97.5])
    return diff, lo, hi
arm, d_arm = sys.argv[1], sys.argv[2]
ruler = sweep('slot-spandec-strict', '2026-09-12-strict'); s = sweep(arm, d_arm)
for k in ('1', '2', '3', '6', '9', '12', '16'):
    print(f'{arm} depth={k:>2} ce={s["depths"][k]["ce_tokens"]:.4f}  ruler {ruler["depths"][k]["ce_tokens"]:.4f}')
for c in ('K1-K6', 'K3-K6'):
    ci = s['ci_ce_tokens'][c]; print(f'{arm} {c} {ci["point"]:+.4f} [{ci["lo"]:+.4f}, {ci["hi"]:+.4f}]')
diff, lo, hi = paired(s, ruler)
print(f'P-6 depth-6 token-paired: {arm} - ruler = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  (bar: worse by 0.00..0.06)')
try:
    w = worth(arm, d_arm); wr = worth('slot-spandec-strict', '2026-09-12-strict')
    for mode in ('zero', 'shuffle'):
        m, mr = w['modes'][mode], wr['modes'][mode]
        print(f'worth {mode} bins {w["bins"]}')
        print(f'  {arm:28s} ' + ' '.join(f'{x:+.4f}' for x in m['mean']) + f'  total {m["total"]:+.4f}')
        print(f'  {"ruler":28s} ' + ' '.join(f'{x:+.4f}' for x in mr['mean']) + f'  total {mr["total"]:+.4f}')
        print(f'  bin16+ {arm} {m["mean"][-1]:+.4f} [{m["ci_lo"][-1]:+.4f},{m["ci_hi"][-1]:+.4f}] vs ruler {mr["mean"][-1]:+.4f} [{mr["ci_lo"][-1]:+.4f},{mr["ci_hi"][-1]:+.4f}]  delta {m["mean"][-1]-mr["mean"][-1]:+.4f}')
    print('P-7 bar: zero-mode bin 16+ at least +0.02 above ruler (0.0905) -> needs >= 0.1105')
except FileNotFoundError as e:
    print('worth profile not yet on disk:', e)
if len(sys.argv) > 3:
    s3 = sweep(sys.argv[3], d_arm); diff, lo, hi = paired(s3, s)
    print(f'P-8 depth-6 token-paired: {sys.argv[3]} - {arm} = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  (fails if off3 better by > 0.02)')
