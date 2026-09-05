"""Pinned, paired Huginn depth evaluation. See the frozen September 4 prereg."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab/divergence"))
from _earning import EarningProfile, offsets_from_ids
from _stats import paired_bootstrap_ci


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_huginn(cfg):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    source = str(Path(cfg.model.snapshot).expanduser())
    if Path(source).name != cfg.model.revision:
        raise ValueError("Snapshot directory does not match the frozen revision")
    model_cfg = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    base = get_class_from_dynamic_module("raven_modeling_minimal.RavenPreTrainedModel", source,
                                         local_files_only=True)
    if int(transformers.__version__.split(".")[0]) >= 5:
        base._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    model_cfg.tie_word_embeddings = True
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        source, config=model_cfg, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True).to(cfg.device).eval()
    if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        raise RuntimeError("Checkpoint embedding and prediction weights are not tied")
    return model, tok, model_cfg


def build_rule(tok, cfg):
    from morph.model.tul_layout import BoundaryRule, boundary_lut_from_strings
    vocab = len(tok)
    if tok.eos_token_id is None:
        raise ValueError("Pinned tokenizer must define EOS")
    strings = tok.batch_decode([[i] for i in range(vocab)])
    lut = boundary_lut_from_strings(strings, tok.eos_token_id,
                                    str(cfg.boundary.suffix_chars), tuple(cfg.boundary.substrings))
    return BoundaryRule(lut, min_span=cfg.boundary.min_span,
                        span_cap=cfg.boundary.span_cap, eos_id=tok.eos_token_id)


def target_offsets(x: torch.Tensor, y: torch.Tensor, rule) -> np.ndarray:
    """Classify the predicted token, including the last target beyond the input."""
    if not torch.equal(x[1:], y[:-1]):
        raise ValueError("Labels are not next-token shifted inputs")
    complete = torch.cat((x, y[-1:])).cpu().numpy()
    return offsets_from_ids(complete, rule)[1:]


@torch.inference_mode()
def ce_map(model, x, y, depth: int, device: str, seed: int) -> torch.Tensor:
    # Huginn samples its initial latent even in eval. Pair it across depths.
    devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            output = model(input_ids=x.to(device), num_steps=int(depth), use_cache=False,
                           output_details={"return_logits": True, "return_latents": False,
                                           "return_head": False, "return_stats": False})
        logits = output.logits.float()
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten(), reduction="none")
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite token CE at depth {depth}")
        return loss.reshape_as(y).cpu()


def comparisons(depths, row_sums, counts, bootstrap):
    pairs = set(zip(depths[:-1], depths[1:]))
    pairs.update([(1, 3), (3, 6), (6, 16), (16, 32), (32, 64), (1, 6), (3, 32)])
    return {f"K{lo}-K{hi}": paired_bootstrap_ci(
        np.asarray(row_sums[str(lo)]), np.asarray(row_sums[str(hi)]), counts,
        n_boot=bootstrap.n_boot, seed=bootstrap.seed, level=bootstrap.level)
        for lo, hi in sorted(pairs) if str(lo) in row_sums and str(hi) in row_sums}


def restore_profile(profile: EarningProfile, saved: dict) -> None:
    profile.row_n[:] = saved["row_n_tokens"]
    profile.bin_n[:] = saved["bin_n_tokens"]
    for key, values in saved["row_ce_sum"].items():
        profile.row_ce[int(key)][:] = values
        profile.bin_ce[int(key)][:] = saved["bin_ce_sum"][key]
    profile._counted.update(range(len(profile.row_n)))


def wandb_model_config(model_cfg):
    """Match the JSON key types restored by W&B, without allowing value changes."""
    return json.loads(json.dumps(model_cfg.to_dict(), allow_nan=False))


def execute(cfg):
    import transformers
    import wandb
    from morph.training.data import create_dataloader
    if cfg.rows < 1 or cfg.batch < 1 or cfg.seq < 2:
        raise ValueError("rows, batch, and sequence length must be positive")
    steps = list(cfg.steps)
    if steps != sorted(set(steps)) or min(steps) < 1:
        raise ValueError("Depths must be unique, positive, increasing integers")
    if cfg.wandb.mode != "online":
        raise ValueError("This experiment requires online W&B")
    torch.set_num_threads(cfg.cpu_threads)
    torch.set_num_interop_threads(cfg.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    output = Path(cfg.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    # Resume may differ only in operational output and resume flags.
    contract = {k: v for k, v in config.items() if k not in {"output", "resume"}}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    checkpoint = output / "sweep_huginn.json"
    saved = json.loads(checkpoint.read_text()) if cfg.resume and checkpoint.exists() else None
    if checkpoint.exists() and not cfg.resume:
        raise FileExistsError(f"Refusing to replace existing result {checkpoint}")
    if saved and saved["huginn"]["config_fingerprint"] != fingerprint:
        raise ValueError("Resume config differs from the recorded experiment")
    source = Path(cfg.model.snapshot).expanduser()
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "wandb": wandb.__version__, "python": sys.version}
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in [
        Path(__file__), ROOT / "lab/divergence/_earning.py", ROOT / "lab/divergence/_stats.py",
        ROOT / "morph/training/data.py", ROOT / "morph/model/tul_layout.py"]}
    if saved and saved["huginn"]["source_hashes"] != source_hashes:
        raise ValueError("Resume source hashes differ from the recorded experiment")
    model_config = json.loads((source / "config.json").read_text())
    complete_config = {**config, "source_model_config": model_config, "versions": versions,
                       "source_hashes": source_hashes,
                       "remote_code_hash": digest(source / "raven_modeling_minimal.py"),
                       "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    wandb_id = saved["huginn"]["wandb_id"] if saved else uuid.uuid4().hex[:8]
    run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, id=wandb_id,
                     resume="must" if saved else "never", config=complete_config,
                     dir=str(output), mode="online")
    try:
        atomic_json(output / "config.json", complete_config)
        print(f"WANDB {run.url}", flush=True)
        model, tok, model_cfg = load_huginn(cfg)
        rule = build_rule(tok, cfg)
        loader = create_dataloader(str(source), cfg.dataset, cfg.seq, cfg.batch,
                                   split="train", skip_samples=cfg.skip_samples)
        batches, offsets, row_hashes = [], [], []
        remaining = cfg.rows
        while remaining:
            x, y = next(loader)[:2]
            x, y = x[:remaining], y[:remaining]
            batches.append((x, y))
            offsets.append([target_offsets(a, b, rule) for a, b in zip(x, y)])
            row_hashes.extend(hashlib.sha256(torch.cat((a, b[-1:])).numpy().tobytes()).hexdigest()
                              for a, b in zip(x, y))
            remaining -= len(x)
        data_hash = hashlib.sha256("".join(row_hashes).encode()).hexdigest()
        run.config.update({"row_hashes": row_hashes, "data_hash": data_hash,
                           "effective_model_config": wandb_model_config(model_cfg),
                           "resume_source_migrations": saved["huginn"].get("resume_migrations", []) if saved else []})
        profile = EarningProfile(steps, cfg.rows)
        arm = saved["huginn"] if saved else {
            "model": cfg.model.name, "revision": cfg.model.revision, "rows": cfg.rows,
            "seq": cfg.seq, "batch": cfg.batch, "depths": {}, "row_ce_sum": {},
            "wandb_id": wandb_id, "wandb_url": run.url, "config_fingerprint": fingerprint,
            "source_hashes": source_hashes, "data_hash": data_hash, "row_hashes": row_hashes,
            "offset_semantics": "predicted target token; row-local span state starts at zero",
            "sampling": "train shard prefix, not held out; rows matched within Huginn only",
            "mean_recurrence": int(model_cfg.mean_recurrence), "status": "running"}
        if saved:
            if arm["data_hash"] != data_hash:
                raise ValueError("Resume data differs from the recorded rows")
            restore_profile(profile, arm["profile"])
        if cfg.smoke_checks:
            x, y = batches[0]
            a = ce_map(model, x, y, steps[0], cfg.device, cfg.seed)
            b = ce_map(model, x, y, steps[0], cfg.device, cfg.seed)
            if not torch.equal(a, b):
                raise AssertionError("Same-seed Huginn CE is not repeatable")
            # Actual Huginn labels are already shifted; validate independent CE against its loss.
            with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(cfg.seed)
                out = model(input_ids=x.to(cfg.device), labels=y.to(cfg.device), num_steps=steps[0])
            torch.testing.assert_close(out.loss.cpu(), a.mean(), rtol=2e-5, atol=2e-5)
            del out
            print("SMOKE repeatability and shifted-label CE passed", flush=True)
        for depth in steps:
            if str(depth) in arm["depths"]:
                print(f"RESUME skip completed depth {depth}", flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            start, row = time.monotonic(), 0
            for i, (x, y) in enumerate(batches):
                ce = ce_map(model, x, y, depth, cfg.device, cfg.seed + i)
                valid = y >= 0
                for b in range(len(x)):
                    profile.add(depth, row, ce[b], valid[b], offsets[i][b])
                    row += 1
                if (i + 1) % cfg.log_every_batches == 0 or row == cfg.rows:
                    print(f"PROGRESS depth={depth} rows={row}/{cfg.rows}", flush=True)
                    run.log({"depth": depth, "rows_completed": row,
                             "batch_ce": float(ce.mean())})
            metric = {"ce_tokens": float(profile.row_ce[depth].sum() / profile.row_n.sum()),
                      "n_tokens": int(profile.row_n.sum()), "seconds": time.monotonic() - start,
                      "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
            arm["depths"][str(depth)] = metric
            arm["row_ce_sum"][str(depth)] = profile.row_ce[depth].tolist()
            arm["row_n_tokens"] = profile.row_n.tolist()
            arm["profile"] = profile.to_json()
            # Do not represent uncomputed profiles as measured zero loss.
            for key in ("row_ce_sum", "bin_ce_sum"):
                arm["profile"][key] = {k: v for k, v in arm["profile"][key].items() if k in arm["depths"]}
            arm["ci_ce_tokens"] = comparisons(steps, arm["row_ce_sum"], profile.row_n, cfg.bootstrap)
            arm["status"] = "complete" if len(arm["depths"]) == len(steps) else "running"
            atomic_json(checkpoint, {"huginn": arm})
            run.log({"depth": depth, **metric})
            print(f"DEPTH {depth} {json.dumps(metric)}", flush=True)
        print(f"COMPLETE {checkpoint}", flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)


@hydra.main(version_base="1.3", config_path="configs", config_name="depth_sweep")
def main(cfg: DictConfig) -> None:
    execute(cfg)


if __name__ == "__main__":
    main()
