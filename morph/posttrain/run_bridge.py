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

    db = build_db_runtime(cfg)
    mc = build_morph_config(cfg, tul=None)
    model = MORPHTransformer(mc)
    if db is not None:
        model.build_db_modules(db.model_cfg)
    model = model.to(dev).eval()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("model_state_dict"))
    if sd is None:
        raise SystemExit(f"no model state in {a.ckpt}: keys={list(ck)[:8]}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    db_missing = [k for k in missing if k.startswith(("db_gates", "db_sigma_cond"))]
    if db is not None and db_missing:
        # Loud: silently generating from randomly-initialised conditioning would produce a
        # bridge row that looks real and means nothing.
        raise SystemExit(f"checkpoint is missing DB conditioning params: {db_missing[:5]}")
    print(f"loaded {a.ckpt}  (missing {len(missing)}, unexpected {len(unexpected)})",
          flush=True)

    tok = AutoTokenizer.from_pretrained(str(cfg.data.tokenizer))
    seq = int(cfg.data.seq_len)

    # Prompts from the SAME val stream both families see, so neither arm gets easier text.
    from morph.training.data import get_data_loaders
    _, val_loader = get_data_loaders(cfg)
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
                lg, _ = db_sample(model, ctx, db, n_steps=a.db_steps,
                                  generator=torch.Generator(device="cpu").manual_seed(a.seed))
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
