"""Throughput + memory GRID over (seq_len, micro_batch, grad_accum) for the curriculum.

One tool, two axes: for each cell prints tok/s, ms/opt-step, AND peak VRAM (supersedes a
separate mem probe). Each cell runs in a FRESH SUBPROCESS — the only trustworthy methodology
(a shared process leaks optimizer state + allocator fragmentation into later cells → false
OOMs / false super-linear curves).

Sections:
  1. (L × micro_batch) grid at ga=1 — the tok/s & memory ceiling per micro-batch.
  2. ga-invariance check — same micro-batch, ga ∈ {1,2,4}: confirms accumulation is ~free
     at FIXED micro-batch (the penalty is micro-batch size, not accumulation).
  3. matched-eff-batch: no-accum baseline (ga=1, one big batch) vs accumulated, same eff batch.

Run: PYTHONPATH=$PWD /home/wolfe/.venv/bin/python scripts/throughput_probe.py
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
import torch
from omegaconf import OmegaConf

MAX_SEQ = 16384
# per-L micro-batch caps (a bit above the measured sustained ceiling so we SEE the OOM edge)
GRID = {4096: [1, 2, 3, 4, 6], 8192: [1, 2, 3, 4], 16384: [1, 2]}
N_STEPS = {4096: 20, 8192: 12, 16384: 6}
GA_INVARIANCE = [(4096, 2, [1, 2, 4]), (8192, 1, [1, 2, 4]), (16384, 1, [1, 4])]
MATCHED_EFF = [  # (L, eff, [mb options]) — mb=eff is the no-accum baseline
    (4096, 4, [4, 2, 1]),
    (8192, 2, [2, 1]),
]


def run_cell(L, mb, ga, n):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from morph.training.train import build_morph_config
    from morph.model.transformer import MORPHTransformer
    cfg = OmegaConf.load("morph/configs/base.yaml")
    cfg.model.max_seq_len = MAX_SEQ; cfg.model.context_len = MAX_SEQ
    model = MORPHTransformer(build_morph_config(cfg)).to("cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    V = int(model.cfg.vocab_size)

    def opt_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(ga):
            x = torch.randint(0, V, (mb, L), device="cuda")
            y = torch.randint(0, V, (mb, L), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, labels=y)["loss"] / ga
            loss.backward()
        opt.step()

    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            opt_step()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            opt_step()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        pk = torch.cuda.max_memory_allocated() / 2**30
        print(f"OK {mb*ga*L*n/dt:.0f} {dt/n*1000:.0f} {pk:.2f}")
    except torch.cuda.OutOfMemoryError:
        print("OOM")


def _cell(L, mb, ga, n):
    out = subprocess.run([sys.executable, __file__, "--cell", f"{L},{mb},{ga},{n}"],
                         capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    line = [l for l in out.stdout.splitlines() if l.startswith(("OK", "OOM"))]
    return line[-1] if line else "OOM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None)
    args = ap.parse_args()
    if args.cell:
        L, mb, ga, n = (int(x) for x in args.cell.split(","))
        run_cell(L, mb, ga, n)
        return

    print(f"[grid] {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GB) | eager | 276M dense bf16\n")

    print("=== 1. (seq_len × micro_batch) @ ga=1 — tok/s & peak VRAM ===")
    print(f"  {'L':>6} {'mb':>3} {'tok/s':>9} {'ms/step':>8} {'peak GB':>8}")
    fits = {}
    for L in GRID:
        for mb in GRID[L]:
            r = _cell(L, mb, 1, N_STEPS[L])
            if r.startswith("OK"):
                _, tps, ms, pk = r.split()
                fits[L] = mb
                print(f"  {L:>6} {mb:>3} {float(tps):>9,.0f} {float(ms):>8.0f} {float(pk):>8.2f}")
            else:
                print(f"  {L:>6} {mb:>3} {'OOM':>9}")
                break

    print("\n=== 2. ga-invariance (fixed micro-batch; is accumulation free?) ===")
    print(f"  {'L':>6} {'mb':>3} {'ga':>3} {'eff':>4} {'tok/s':>9} {'ms/step':>8}")
    for (L, mb, gas) in GA_INVARIANCE:
        for ga in gas:
            r = _cell(L, mb, ga, max(4, N_STEPS[L] // ga))
            if r.startswith("OK"):
                _, tps, ms, _pk = r.split()
                print(f"  {L:>6} {mb:>3} {ga:>3} {mb*ga:>4} {float(tps):>9,.0f} {float(ms):>8.0f}")
            else:
                print(f"  {L:>6} {mb:>3} {ga:>3} {mb*ga:>4} {'OOM':>9}")

    print("\n=== 3. matched eff-batch: no-accum baseline (mb=eff) vs accumulated ===")
    print(f"  {'L':>6} {'eff':>4} {'mb':>3} {'ga':>3} {'tok/s':>9} {'note':>12}")
    for (L, eff, mbs) in MATCHED_EFF:
        for mb in mbs:
            ga = eff // mb
            if mb * ga != eff:
                continue
            r = _cell(L, mb, ga, N_STEPS[L])
            note = "baseline" if ga == 1 else f"accum×{ga}"
            if r.startswith("OK"):
                _, tps, ms, _pk = r.split()
                print(f"  {L:>6} {eff:>4} {mb:>3} {ga:>3} {float(tps):>9,.0f} {note:>12}")
            else:
                print(f"  {L:>6} {eff:>4} {mb:>3} {ga:>3} {'OOM':>9} {note:>12}")


if __name__ == "__main__":
    main()
