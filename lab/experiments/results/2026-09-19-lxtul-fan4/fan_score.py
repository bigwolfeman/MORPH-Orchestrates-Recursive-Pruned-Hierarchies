"""Score the LXTUL fan arms (lab/experiments/planned/2026-09-19-lxtul-fan4.md) from wandb
summaries at 5000, the runner's sweep JSON (K-curve) and its RATE line. Usage:
  fan_score.py ARM [ARM ...]   (arms: slot-spandec-strict-fan4, -norepel, -mean, pk4)"""
import json, re, sys, wandb
CK = '/home/wolfe/morph-to/checkpoints/morph'; R = '/home/wolfe/morph-scratch/arc/results/2026-09-19-lxtul-fan4'
Q = '/home/wolfe/morph-scratch/arc/queue.log'
api = wandb.Api(timeout=60)
def summary(arm):
    rid = open(f'{CK}/{arm}/wandb_id.txt').readline().strip()
    r = api.run(f'adew-me/morph-tul/{rid}'); s = r.summary
    keys = ['fan/single_ce', 'fan/oracle_ce', 'fan/mixed_ce', 'fan/oracle_pick0', 'fan/stream_rank_t1',
            'fan/stream_rank_t6', 'val/fan_stream_cos_t1', 'val/fan_stream_cos_t6', 'val/fan_mix_entropy',
            'fan/stream_ce_k0', 'fan/stream_ce_k1', 'fan/stream_ce_k2', 'fan/stream_ce_k3']
    # the trainer's FINAL val (step 5000) logs every key with a `_final` suffix; the plain
    # key is the last periodic val (step 4750). Prefer the final one and say which was used.
    out = {'_step': s.get('_step'), 'final_keys': all(s.get(k + '_final') is not None for k in keys)}
    for k in keys:
        out[k] = s.get(k + '_final') if s.get(k + '_final') is not None else s.get(k)
    return out, r.state
def sweep(arm):
    try: return json.load(open(f'{R}/sweep_{arm}_5000.json'))[arm]
    except FileNotFoundError: return None
def rate(arm):
    m = [re.search(r'tok/s=(\d+)', l) for l in open(Q) if f'RATE OK {arm}:' in l]
    return int(m[-1].group(1)) if m and m[-1] else None
out = {}
for arm in sys.argv[1:]:
    s, st = summary(arm); out[arm] = s
    print(f'== {arm} (wandb {st}, step {s["_step"]}, final-val keys used: {s["final_keys"]})')
    for k, v in s.items():
        if k not in ('_step', 'final_keys') and v is not None: print(f'   {k:26s} {v:.4f}')
    if s['fan/mixed_ce'] is not None and s['fan/oracle_ce'] is not None:
        print(f'   oracle - mixed            {s["fan/oracle_ce"] - s["fan/mixed_ce"]:+.4f}   (selection over the streams vs the state the coda trained on)')
        ks = [s[f'fan/stream_ce_k{i}'] for i in range(4) if s.get(f'fan/stream_ce_k{i}') is not None]
        if ks: print(f'   stream CE spread          {max(ks) - min(ks):.4f}; every stream minus mixed: ' + ' '.join(f'{v - s["fan/mixed_ce"]:+.4f}' for v in ks))
    if s['fan/single_ce'] is not None and s['fan/oracle_ce'] is not None:
        gap = s['fan/single_ce'] - s['fan/oracle_ce']; print(f'   single-oracle gap         {gap:+.4f}   (P-1 bar > 0.022 on fan4)')
    if s['val/fan_mix_entropy'] is not None: print(f'   P-4 mix_entropy < 1.236: {s["val/fan_mix_entropy"] < 1.236}  (fan4 only; mean control must read 1.3863)')
    sw = sweep(arm)
    if sw:
        for c in ('K1-K6', 'K3-K6'):
            ci = sw['ci_ce_tokens'][c]; print(f'   {c} {ci["point"]:+.4f} [{ci["lo"]:+.4f}, {ci["hi"]:+.4f}]')
        print(f'   P-3 (fan4): K1-K6 > 0.005: {sw["ci_ce_tokens"]["K1-K6"]["point"] > 0.005}; K3-K6 > 0.001: {sw["ci_ce_tokens"]["K3-K6"]["point"] > 0.001}')
        print(f'   depth-6 ce_tokens {sw["depths"]["6"]["ce_tokens"]:.4f}')
    else: print('   sweep JSON not on disk yet')
    r = rate(arm); print(f'   rate {r} tok/s  P-7 band 8500-11000: {8500 <= (r or 0) <= 11000}')
a, b = out.get('slot-spandec-strict-fan4'), out.get('slot-spandec-strict-fan4-norepel')
if a and b and b['val/fan_stream_cos_t6'] is not None:
    print('== cross-arm (P-2, P-5, P-6)')
    print(f'   P-2 norepel cos_t6 >= 0.90: {b["val/fan_stream_cos_t6"]:.4f} -> {b["val/fan_stream_cos_t6"] >= 0.90}')
    print(f'   P-2 fan4 cos_t6 < 0.80: {a["val/fan_stream_cos_t6"]:.4f} -> {a["val/fan_stream_cos_t6"] < 0.80}')
    print(f'   P-2 fan4 cos_t1 at least 0.10 below norepel: {a["val/fan_stream_cos_t1"]:.4f} vs {b["val/fan_stream_cos_t1"]:.4f} -> {a["val/fan_stream_cos_t1"] <= b["val/fan_stream_cos_t1"] - 0.10}')
    print(f'   P-2 monotone (cos_t1 < cos_t6) fan4 {a["val/fan_stream_cos_t1"] < a["val/fan_stream_cos_t6"]}, norepel {b["val/fan_stream_cos_t1"] < b["val/fan_stream_cos_t6"]}')
    print(f'   P-5 fan4 rank_t1 > 2.0: {a["fan/stream_rank_t1"]:.4f}; norepel <= 1.35: {b["fan/stream_rank_t1"]:.4f}')
    print(f'   P-6 fan4 pick0 < 0.40: {a["fan/oracle_pick0"]:.4f}; norepel > 0.60: {b["fan/oracle_pick0"]:.4f}')
