import contextlib, sys, torch
sys.path.insert(0, "/home/wolfe/morph-perf")
from lab.divergence._build import build_cfg, build_model
from morph.training.data import create_dataloader
from morph.training.train import load_checkpoint
from lab.divergence.slot_path_worth import plan_off, eval_groups

cfg = build_cfg("tul_a1", ["training.batch_size=6", "model.use_kernels=false"])
model, tul_rt = build_model(cfg, device="cuda")
root = getattr(model, "_orig_mod", model)
load_checkpoint("checkpoints/morph/onset-capture/ROLL_step_1750.pt", model,
                torch.amp.GradScaler("cuda", enabled=False), torch.device("cuda"))
loader = iter(create_dataloader(cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
                                int(cfg.training.batch_size), split="validation",
                                skip_samples=60_000,
                                tul=tul_rt.val_data_cfg if tul_rt else None))
batches = []
for _ in range(8):
    bx, by, bl = next(loader)
    batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))

tul = root.tul
orig_drop = tul.apply_token_dropout

@contextlib.contextmanager
def force_drop(p):
    """Apply token-state dropout at EVAL with probability p, deterministically seeded so
    every condition masks the SAME positions and the plan comparison is paired."""
    def patched(x, layout, training):
        if p <= 0.0:
            return x, None
        g = torch.Generator(device=x.device).manual_seed(1234)
        drop = (torch.rand(layout.slot_mask.shape, device=x.device, generator=g) < p) \
               & (~layout.slot_mask)
        keep = (~drop).to(x.dtype).unsqueeze(-1)
        mv = tul.E_mask.to(x.dtype)
        x = torch.where(drop[:, :, None, None] if x.dim() == 4 else drop[:, :, None], mv, x)
        return x, keep
    tul.apply_token_dropout = patched
    try:
        yield
    finally:
        tul.apply_token_dropout = orig_drop

print(f"{'token drop p':>13} {'ce_main w/ plan':>16} {'ce_main no plan':>16} {'plan worth':>11}")
for p in (0.0, 0.15, 0.5, 0.9, 1.0):
    with force_drop(p):
        full = eval_groups(model, batches)
        with plan_off(root):
            noplan = eval_groups(model, batches)
    print(f"{p:>13.2f} {full['ce_main']:>16.4f} {noplan['ce_main']:>16.4f} "
          f"{noplan['ce_main'] - full['ce_main']:>11.4f}")
