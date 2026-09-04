"""Bit-exactness gate for the paid-loop forward across trees.

    PYTHONPATH=<tree> CUBLAS_WORKSPACE_CONFIG=:4096:8 python lab/divergence/paid_loop_gate.py \
        --config tul_a2 --ckpt <A2 checkpoint>.pt --out <json>

One checkpoint, one fixed validation batch (the first rows of the validation split, packed
with the config's boundary rule), eval mode, eager kernels, deterministic algorithms,
bf16 autocast. Writes the loss and every scalar the forward returns at full precision, a
sha1 of the logits when they are present, the parameter count and a sha1 of the sorted
parameter names. Run it in two trees and compare the JSON files key by key.

Record, 2026-09-04 (the tul/think-once merge, .agents/notes/implemented/architecture/
2026-09-03-ship-the-paid-loop-cut-the-arms.md amendment): on
morph-scratch/checkpoints-keep/tul-a2-20k-wu/step_5000.pt with tul_a2, master 3a94963 and
the merged tree both give loss 3.6009058952331543, ce_tokens 3.600905656814575,
266125943 parameters, the same parameter-name sha and the same logits sha; the master run
repeated bit-for-bit. Every key equal.
"""
import argparse, json, os, sys, hashlib
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="tul_a2"); ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--overrides", nargs="*", default=[])
a = ap.parse_args()
tree = os.environ["PYTHONPATH"].split(":")[0]
sys.path.insert(0, os.path.join(tree, "lab", "divergence")); sys.path.insert(0, os.path.join(tree, "scripts"))
from _build import build_cfg
from tul_samples import load_ckpt
from morph.model.tul_layout import pack_tul_batch
from morph.training.data import create_dataloader
from morph.training.tul_setup import build_tul_runtime
torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.benchmark = False
cfg = build_cfg(a.config, ["model.use_kernels=false", "training.compile=false", "model.hc_use_kernel=false"] + a.overrides)
rt = build_tul_runtime(cfg)
model, step = load_ckpt(cfg, a.ckpt, "cuda", rt.model_cfg)
model.eval()
spec = rt.data_cfg.spec_for(cfg.data.seq_len)
loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8, split="validation", skip_samples=0, bag_size=0, tul=None)
buf = []
need = a.batch * (spec.l_total + 1)
while len(buf) < need: buf.extend(next(loader)[0].reshape(-1).tolist())
x, y, layout = pack_tul_batch(buf, rt.data_cfg.rule, spec, a.batch)
x, y, layout = x.cuda(), y.cuda(), layout.to("cuda")
torch.manual_seed(0)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(x, labels=y, bag_size=0, slot_layout=layout)
res = {"tree": tree, "config": a.config, "ckpt": a.ckpt, "step": step, "n_params": sum(p.numel() for p in model.parameters()),
       "param_names_sha": hashlib.sha1("\n".join(sorted(n for n, _ in model.named_parameters())).encode()).hexdigest(),
       "x_sha": hashlib.sha1(x.cpu().numpy().tobytes()).hexdigest()}
for k, v in out.items():
    if torch.is_tensor(v) and v.numel() == 1: res[k] = float(v.double())
    elif torch.is_tensor(v): res[f"{k}_sha"] = hashlib.sha1(v.detach().float().cpu().numpy().tobytes()).hexdigest(); res[f"{k}_sum"] = float(v.detach().double().sum())
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k in ("tree","step","n_params","loss","ce_tokens","ce_main","logits_sha","logits_sum","param_names_sha")}, indent=1))
