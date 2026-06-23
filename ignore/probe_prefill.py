#!/usr/bin/env python
"""Isolate: does the EAGER batched prefill match the EAGER solo prefill per-stream?
If gen-0 tokens already differ, the divergence is upstream of the static engine."""
import os, sys, torch
from bench_decode import build_model, load_ckpt, _AC

device = torch.device("cuda")
ckpt = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                       else "checkpoints/morph/tst_stp_off_50k/step_50000.pt")
model, cfg = build_model(device)
load_ckpt(ckpt, model, device, cfg); model.eval()
from morph.model import packed_ternary_infer as pti
with _AC():
    pti.to_deploy_inference(model, device="cuda")
vocab = int(model.cfg.vocab_size)
from morph.model.kv_cache import MORPHKVCache, decode_step

def prefill_logit(seeds):
    cache = MORPHKVCache(); cache.csa_pool_len = int(model.cfg.context_len)
    lg = None
    with _AC():
        for p in range(seeds.shape[1]):
            lg = decode_step(model, seeds[:, p], cache)
    return lg  # [B, vocab]

B, L = 4, 16
torch.manual_seed(1004)
seeds = torch.randint(0, vocab, (B, L), device=device)
for b in range(B): seeds[b, 0] = (seeds[b, 0] + b) % vocab

batched = prefill_logit(seeds)                      # [B, vocab]
print("per-stream EAGER prefill: solo vs batched (argmax + max|Δlogit|)")
allok = True
for b in range(B):
    solo = prefill_logit(seeds[b:b+1])              # [1, vocab]
    da = int(solo.argmax(-1)) ; db = int(batched[b].argmax(-1))
    dl = float((solo[0].float() - batched[b].float()).abs().max())
    same = da == db
    allok = allok and same
    print(f"  b={b}: solo_argmax={da} batched_argmax={db} max|Δ|={dl:.4f} {'OK' if same else 'DIFFER'}")
print("EAGER_PREFILL_BATCH_MATCH" if allok else "EAGER_PREFILL_BATCH_DIFFER")
