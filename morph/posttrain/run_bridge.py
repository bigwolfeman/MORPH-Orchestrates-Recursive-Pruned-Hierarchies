"""Post-hoc bridge-metric runner: checkpoint -> generations -> comparable numbers.

Plan A7. Contract: docs/diffusionblocks-experiment-sheet.md §1.3.

Run AFTER training, in its own process. The teacher wants its own VRAM and the training loop
must not pay for it.

Usage:
    python -m morph.posttrain.run_bridge --ckpt <path> --config-name db_b3 [--no-teacher]

Both families go through the SAME prompts and the SAME decoding settings, because that is
the only thing that makes their numbers comparable. A baseline checkpoint decodes with
top-p; a DB checkpoint decodes with the Euler walk. Everything else is held fixed and
recorded in the output row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="db_b3")
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=50)
    ap.add_argument("--db-steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--eager", action="store_true",
                    help="force model.use_kernels=false so the whole path can be smoke-tested "
                         "on CPU (Triton needs a CUDA driver). Kernel-vs-eager is the same "
                         "math and adds no parameters, so the checkpoint still loads — but "
                         "REAL bridge numbers should be produced on GPU with kernels on.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from hydra import compose, initialize_config_dir
    cfgdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name=a.config_name)

    from transformers import AutoTokenizer

    from morph.model.transformer import MORPHTransformer
    from morph.posttrain.bridge_metrics import BridgeConfig, bridge_report
    from morph.training.db_setup import build_db_runtime
    from morph.training.train import build_morph_config

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    if a.eager:
        cfg.model.use_kernels = False
        cfg.model.hc_use_kernel = False
        print("  [eager] kernels OFF (CPU smoke path)", flush=True)

    db = build_db_runtime(cfg)
    mc = build_morph_config(cfg, tul=None)
    model = MORPHTransformer(mc)
    if db is not None:
        model.build_db_modules(db.model_cfg)

    # Apply the SAME quantisation transforms, in the same order, that the training run
    # applied. They RENAME tensors in the state_dict (ternary/int6 add
    # `parametrizations.weight.original`), so a model rebuilt without them has no home for
    # 45 saved tensors and the project's load_checkpoint correctly refuses. quant_setup's
    # own docstring says it: "anything rebuilding a trained model must apply the SAME ones."
    from morph.training.quant_setup import apply_quantization
    _qm = apply_quantization(model, cfg)
    print(f"  quant applied: ternary={_qm['ternary'] is not None} "
          f"embed={_qm['embed_quant'] is not None}", flush=True)

    model = model.to(dev).eval()

    # Use the project's own loader, NOT a bare load_state_dict. train.py::load_checkpoint
    # already reconstructs module structure in the order the live run mutated it (carve/BCSR,
    # ReMoE routers, prune mask, saliency EMA) and handles the `mlp._orig_mod.` nesting that
    # torch.compile introduces. A naive load reported "missing 129, unexpected 129" — 112 of
    # those are compiled-MLP tensors whose prefix did not line up, i.e. it would have
    # generated from a partly RANDOM model and produced a bridge row that looked real.
    from morph.training.train import load_checkpoint

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    step_loaded, _meta, _ok = load_checkpoint(a.ckpt, model, scaler, torch.device(dev))
    print(f"loaded {a.ckpt} @ step {step_loaded}", flush=True)

    # Post-load audit. The db_*-only check that shipped first was too narrow: it passed while
    # 129 other keys silently missed. Assert the WHOLE state matched.
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("model_state_dict")) or {}
    # Normalise the SAME equivalence class load_checkpoint uses internally: torch.compile
    # nests compiled submodules under `._orig_mod.`, so `coda.0.mlp._orig_mod.0.down` and
    # `coda.0.mlp.0.down` are the same tensor. Comparing raw key sets flagged 112 false
    # mismatches AFTER a load that had in fact succeeded.
    def _norm(k: str) -> str:
        return k.replace("._orig_mod.", ".")

    live = {_norm(k) for k in model.state_dict()}
    saved = {_norm(k) for k in sd}
    miss, unexp = sorted(saved - live), sorted(live - saved)
    if miss or unexp:
        raise SystemExit(
            f"state_dict mismatch after load: {len(miss)} saved-but-absent, "
            f"{len(unexp)} live-but-unsaved.\n  saved-only: {miss[:4]}\n"
            f"  live-only: {unexp[:4]}\n"
            f"Generating from a partly-random model would produce a meaningless bridge row.")
    if db is not None:
        n_db = sum(1 for k in saved if k.startswith(("db_gates", "db_sigma_cond")))
        if n_db == 0:
            raise SystemExit("checkpoint carries no DB conditioning params")
        print(f"  DB conditioning params restored: {n_db}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(cfg.data.tokenizer))
    seq = int(cfg.data.seq_len)

    # Prompts from the SAME val stream both families see, so neither arm gets easier text.
    from morph.training.data import create_dataloader
    # Match train.py::_make_val_loader EXACTLY: split="validation" with
    # skip_samples=50_000. The arrow source has only a `train` split, so "validation" is a
    # held-out slice of the same stream — and the skip is what holds it out. Getting this
    # wrong would draw bridge prompts from TRAINING text, quietly flattering every arm.
    val_loader = iter(create_dataloader(
        str(cfg.data.tokenizer), str(cfg.data.dataset), seq, a.batch,
        split="validation", skip_samples=50_000,
    ))
    prompts = []
    while sum(p.shape[0] for p in prompts) < a.n_prompts:
        b = next(val_loader)
        x = b[0] if isinstance(b, (tuple, list)) else b
        prompts.append(x[:, : a.prompt_tokens])
    prompts = torch.cat(prompts, dim=0)[: a.n_prompts].to(dev)
    print(f"prompts {tuple(prompts.shape)}", flush=True)

    gens = []
    with torch.no_grad():
        for i in range(0, prompts.shape[0], a.batch):
            chunk = prompts[i : i + a.batch]
            if db is not None:
                from morph.inference.db_generate import db_sample
                pad = torch.zeros(chunk.shape[0], a.gen_tokens, dtype=torch.long,
                                  device=dev)
                ctx = torch.cat([chunk, pad], dim=1)[:, :seq]
                # The generator MUST be on the same device as the tensors it seeds; a CPU
                # generator with a CUDA randn raises. The CPU shakeout could not catch this
                # because there the two happened to agree.
                lg, _ = db_sample(model, ctx, db, n_steps=a.db_steps,
                                  generator=torch.Generator(device=dev).manual_seed(a.seed))
                gens.append(lg[:, chunk.shape[1]:chunk.shape[1] + a.gen_tokens]
                            .argmax(-1).cpu())
            else:
                raise SystemExit("baseline decoding path not wired in this runner yet; "
                                 "use the existing generation test for A0")
    gen_ids = torch.cat(gens, dim=0)
    print(f"generated {tuple(gen_ids.shape)}", flush=True)

    teacher = tok_t = None
    if not a.no_teacher:
        from transformers import AutoModelForCausalLM
        bc0 = BridgeConfig()
        tok_t = AutoTokenizer.from_pretrained(bc0.teacher)
        if tok_t.pad_token is None:
            tok_t.pad_token = tok_t.eos_token
        teacher = AutoModelForCausalLM.from_pretrained(
            bc0.teacher, dtype=torch.bfloat16).to(dev).eval()
        print(f"teacher {bc0.teacher} loaded", flush=True)

    bc = BridgeConfig(n_prompts=a.n_prompts, prompt_tokens=a.prompt_tokens,
                      gen_tokens=a.gen_tokens, db_steps=a.db_steps, seed=a.seed,
                      batch=a.batch)
    rep = bridge_report(gen_ids, bc, tok, teacher=teacher, tok_teacher=tok_t, device=dev)
    rep["bridge/ckpt"] = a.ckpt
    rep["bridge/config"] = a.config_name
    rep["bridge/step"] = int(ck.get("step", -1))

    out = a.out or (os.path.splitext(a.ckpt)[0] + ".bridge.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
