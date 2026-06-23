#!/usr/bin/env python
"""Classify the lone B=8 gen-24 divergence: real indexing bug vs benign bf16
argmax tie-break. Runs engine-batched AND eager-batched in lockstep on the SAME
B=8 batched prefill; at each step records, for the diverging stream, the logit
GAP between the two top candidate tokens. A razor-thin gap ⇒ FP reduction-order
tie-break (benign); a large gap ⇒ the engine computed a genuinely different value."""
import os, sys, torch
from bench_decode import build_model, load_ckpt, _AC

device = torch.device("cuda")
ckpt = os.path.abspath(sys.argv[1])
model, cfg = build_model(device); load_ckpt(ckpt, model, device, cfg); model.eval()
from morph.model import packed_ternary_infer as pti
with _AC():
    pti.to_deploy_inference(model, device="cuda")
vocab = int(model.cfg.vocab_size)
from morph.model.kv_cache import MORPHKVCache, decode_step
from morph.model.kv_cache_static import StaticDecodeEngine

B, L, NGEN = 8, 16, 128
torch.manual_seed(1008)
seeds = torch.randint(0, vocab, (B, L), device=device)
for b in range(B): seeds[b, 0] = (seeds[b, 0] + b) % vocab

def prefill():
    c = MORPHKVCache(); c.csa_pool_len = int(model.cfg.context_len); lg = None
    with _AC():
        for p in range(L): lg = decode_step(model, seeds[:, p], c)
    return c, lg

# eager-batched golden
ec, elg = prefill()
# engine-batched
gc, glg = prefill()
eng = StaticDecodeEngine(model, batch_size=B)
with _AC():
    eng.load_from_eager(gc); eng.capture()

with _AC():
    e_next = elg.argmax(-1); g_next = eng.logits.copy_(glg).argmax(-1) if False else glg.argmax(-1)
    e_next = elg.argmax(-1)
    g_next = glg.argmax(-1)
    first = None
    for t in range(NGEN):
        # advance both with their OWN greedy choice (independent streams)
        elog = decode_step(model, e_next, ec)            # [B, vocab]
        glog = eng.decode_step(g_next).clone()
        e_arg = elog.argmax(-1); g_arg = glog.argmax(-1)
        for b in range(B):
            if int(e_arg[b]) != int(g_arg[b]) and first is None:
                # margins on EACH engine's own logits between the two candidates
                ea, ga = int(e_arg[b]), int(g_arg[b])
                e_gap = float(elog[b, ea] - elog[b, ga])   # eager prefers ea by this
                g_gap = float(glog[b, ga] - glog[b, ea])   # engine prefers ga by this
                cos = torch.nn.functional.cosine_similarity(
                    elog[b].float(), glog[b].float(), dim=0).item()
                mxd = float((elog[b].float() - glog[b].float()).abs().max())
                print(f"FIRST divergence: stream b={b} gen={t} eager_tok={ea} engine_tok={ga}")
                print(f"  eager  margin (tok{ea} over tok{ga}) = {e_gap:.5f}")
                print(f"  engine margin (tok{ga} over tok{ea}) = {g_gap:.5f}")
                print(f"  cos(eager_logits, engine_logits) = {cos:.6f}  max|Δlogit| = {mxd:.4f}")
                first = (b, t)
        e_next, g_next = e_arg, g_arg
    if first is None:
        print("NO divergence in 128 gen for this seed (seed-dependent tie-break)")
print("PROBE_DONE")
