"""Memory probe for the curriculum: max micro-batch that fits per seq_len on this GPU.

Each (seq_len, micro_batch) cell runs in a FRESH SUBPROCESS — the only trustworthy
methodology. A shared-process probe leaks persistent optimizer state + allocator
fragmentation from earlier (large) cells into later ones, which produces false OOMs and
false "super-linear" curves (learned the hard way: a shared-process probe reported 16K
OOM@mb1 when an isolated run fits at 16.6 GB).

Builds the REAL MORPH model (looped core, HC n=4, GLA, windowed attn, activation ckpt)
dense-bf16, runs ONE full fwd+bwd+AdamW step with random tokens, prints peak alloc.

Run: PYTHONPATH=$PWD /home/wolfe/.venv/bin/python scripts/mem_probe.py
"""
from __future__ import annotations
import argparse, os, subprocess, sys
import torch
from omegaconf import OmegaConf

SEQ_LENS = [4096, 8192, 16384]
MICRO_BATCHES = [1, 2, 3, 4, 6, 8]
MAX_SEQ = max(SEQ_LENS)


def run_cell(L: int, mb: int) -> None:
    """One isolated full step; prints 'OK <peak>' or 'OOM'. Called in a subprocess."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from morph.training.train import build_morph_config
    from morph.model.transformer import MORPHTransformer
    cfg = OmegaConf.load("morph/configs/base.yaml")
    cfg.model.max_seq_len = MAX_SEQ
    cfg.model.context_len = MAX_SEQ
    model = MORPHTransformer(build_morph_config(cfg)).to("cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    V = int(model.cfg.vocab_size)
    try:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, V, (mb, L), device="cuda")
        y = torch.randint(0, V, (mb, L), device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        out["loss"].backward()
        opt.step()
        print(f"OK {torch.cuda.max_memory_allocated() / 2**30:.2f}")
    except torch.cuda.OutOfMemoryError:
        print("OOM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None, help="L,mb — internal single-cell mode")
    args = ap.parse_args()
    if args.cell:
        L, mb = (int(x) for x in args.cell.split(","))
        run_cell(L, mb)
        return

    print(f"[memprobe] subprocess-isolated cells; GPU={torch.cuda.get_device_name(0)} "
          f"total={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GB")
    results = {}
    for L in SEQ_LENS:
        best, best_pk = 0, 0.0
        for mb in MICRO_BATCHES:
            out = subprocess.run([sys.executable, __file__, "--cell", f"{L},{mb}"],
                                 capture_output=True, text=True,
                                 env={**os.environ, "PYTHONPATH": os.getcwd()})
            line = [l for l in out.stdout.splitlines() if l.startswith(("OK", "OOM"))]
            res = line[-1] if line else "OOM"
            if res.startswith("OK"):
                best, best_pk = mb, float(res.split()[1])
                print(f"  L={L:6d} mb={mb}: OK  peak={best_pk:.2f}GB")
            else:
                print(f"  L={L:6d} mb={mb}: OOM")
                break
        results[L] = (best, best_pk)

    print("\n[memprobe] SUMMARY (max micro-batch / peak GB):")
    for L in SEQ_LENS:
        mb, pk = results[L]
        print(f"  seq_len={L:6d} → max micro-batch {mb} (peak {pk:.2f}GB)")


if __name__ == "__main__":
    main()
