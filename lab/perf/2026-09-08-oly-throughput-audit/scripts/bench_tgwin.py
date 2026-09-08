"""TG-masked window branch: shipped eager SDPA path vs alternatives, at the run's shapes."""
import sys, time, json
import torch, torch.nn.functional as F
sys.path.insert(0, "/home/wolfe/morph-to")
from morph.model.attention import _window_fallback, _tg_slot_attention

dev = "cuda"
B, H, S, D = 12, 8, 640, 64
W = 256
scale = D ** -0.5
torch.manual_seed(0)

# A realistic TG allow mask: spans of ~12 tokens with a slot after each (max_slots 64).
bag = torch.zeros(B, S, dtype=torch.long, device=dev)
slot = torch.zeros(B, S, dtype=torch.bool, device=dev)
for b in range(B):
    i = 0; sid = 0
    while i < S:
        span = int(torch.randint(8, 18, (1,)).item())
        end = min(i + span, S)
        bag[b, i:end] = sid
        if end < S:
            bag[b, end:end + 2] = sid
            slot[b, end:end + 2] = True
        i = end + 2; sid += 1
row = torch.arange(S, device=dev).unsqueeze(1); col = torch.arange(S, device=dev).unsqueeze(0)
causal = (col <= row)
tg_allow = (((bag.unsqueeze(2) == bag.unsqueeze(1)) | slot.unsqueeze(1)) & causal).unsqueeze(1)
print("tg_allow density", float(tg_allow.float().mean()))
base = ((row - col >= 0) & (row - col < W) & (row - col != 0)).unsqueeze(0).unsqueeze(0)
comb = base & tg_allow
print("window&tg density", float(comb.float().mean()))

def mk():
    g = torch.Generator(device=dev).manual_seed(1)
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16, generator=g)
    for t in (q, k, v): t.requires_grad_(True)
    return q, k, v

def timeit(fn, n=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

q, k, v = mk()

def shipped():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return _window_fallback(q, k, v, W, dev, scale, 0, extra_mask=tg_allow)

def sdpa_bool():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=comb, scale=scale)

out_ref = shipped()
out_bool = sdpa_bool()
print("sdpa bool-mask maxdiff", float((out_ref - out_bool).abs().max()))

res = {}
res["shipped_fwd_ms"] = timeit(shipped)
res["sdpa_boolmask_fwd_ms"] = timeit(sdpa_bool)

def fwdbwd(fn):
    def g():
        o = fn(); o.sum().backward()
        q.grad = k.grad = v.grad = None
    return g
res["shipped_fwdbwd_ms"] = timeit(fwdbwd(shipped), n=20, warm=5)
res["sdpa_boolmask_fwdbwd_ms"] = timeit(fwdbwd(sdpa_bool), n=20, warm=5)

# flex_attention with a block mask
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    bagd, slotd = bag, slot
    def mask_mod(b, h, qi, kj):
        return (kj <= qi) & (qi - kj < W) & (qi != kj) & \
               ((bagd[b, qi] == bagd[b, kj]) | slotd[b, kj])
    t0 = time.perf_counter()
    bm = create_block_mask(mask_mod, B, None, S, S, device=dev)
    torch.cuda.synchronize()
    res["blockmask_build_ms"] = (time.perf_counter() - t0) * 1e3
    print(bm)
    flex_c = torch.compile(flex_attention, dynamic=False)
    def flex():
      with torch.autocast('cuda', dtype=torch.bfloat16):
        return flex_c(q, k, v, block_mask=bm, scale=scale)
    o = flex()
    res["flex_maxdiff_vs_shipped"] = float((out_ref - o).abs().max())
    res["flex_fwd_ms"] = timeit(flex)
    res["flex_fwdbwd_ms"] = timeit(fwdbwd(flex), n=20, warm=5)
except Exception as e:
    res["flex_error"] = repr(e)

# compressed branch (TG slot attention), M = 64 slot columns
sink = torch.zeros(H, device=dev)
def comp():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return _tg_slot_attention(q, k, v, slot, sink, scale)
res["tg_slot_fwd_ms"] = timeit(comp)
res["tg_slot_fwdbwd_ms"] = timeit(fwdbwd(comp), n=20, warm=5)
print(json.dumps(res, indent=1))
