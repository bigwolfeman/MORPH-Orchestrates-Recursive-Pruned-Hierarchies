"""The FROZEN plain model behind the stage-1 latent targets (``tul.latent_pre_target``
``plain_prelude`` / ``plain_final``; morph/model/tul_latent_pre.py).

Built from the plain run's OWN Hydra config (``tul.latent_pre_ref_config``) and its
checkpoint (``tul.latent_pre_ref_ckpt``): the architecture, the quantisation transforms
(``apply_quantization``, which renames tensors, so they must exist before the load) and
the weights are the plain run's, never the live model's. The load is STRICT after the
``._orig_mod.`` canonicalisation both sides (``load_weights_only``'s rule): one missing or
unexpected tensor raises, because a target computed by a half-loaded model looks fine and
means nothing.

Refusals: a missing checkpoint; a reference config with TUL on (the target is the plain
model's own forward); a checkpoint inside the live run's own checkpoint directory (a live
copy of the model being trained, the note's first rule); a ``d_model`` other than the
live model's. The model is frozen (``requires_grad`` False), in eval mode, and never
compiled; ``MORPHTransformer.tul_latent_pre_attach_ref`` stores it outside the module tree.

THE CALIBRATION (``tul.latent_pre_target_norm: standard``). The fixed per-coordinate mean
and std of the frozen target are computed ONCE at build (:func:`calibration_batches` then
:func:`calibrate_latent_pre_ref`) and stored on the frozen model as NON-persistent buffers
``latent_pre_mu`` / ``latent_pre_sigma`` (never saved; a resume recomputes them, bit-
identically: same documents, same order, order-fixed fp64 sums). The batches are the
TRAINING stream from document ``tul.latent_pre_cal_doc_offset`` (default 40 000) on: past
the run's own documents (refused when the run's estimated document count reaches the
offset) and before the val documents (``VAL_DOC_OFFSET`` = 50 000, train.py's val loader
skip; refused when the calibration would read into them). They are drawn in a SEPARATE
PROCESS (``python -m morph.training.latent_pre_ref --draw``): its own loader instance, so
the run's training stream is untouched (pinned: the first training batches are bit-
identical with and without a calibration), and no tokenizer / datasets thread is left
alive in the trainer before its compile warmup (train.py's fork-safety ordering; an
in-process draw leaves three OS threads, measured).
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile

import torch
from omegaconf import DictConfig

from morph.model.transformer import MORPHTransformer
from morph.model.tul_latent_pre import PLAIN_TARGET_FNS, TargetMoments

_CFG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VAL_DOC_OFFSET = 50_000          # train.py `_make_val_loader` skip_samples (pinned by a test)
SIGMA_FLOOR_FRAC = 1e-3          # sigma floor = this x median(sigma)


def compose_ref_config(name: str) -> DictConfig:
    """The reference run's composed Hydra config. Inside a running Hydra app (the trainer)
    the app's own config search path is used; anywhere else (tests, probes) the tree's
    ``morph/configs``."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        return compose(config_name=name)
    with initialize_config_dir(version_base=None, config_dir=_CFG_DIR):
        return compose(config_name=name)


def _assert_plain(ref_cfg: DictConfig, name: str) -> None:
    tc = ref_cfg.get("tul", None)
    raw = "never" if tc is None else tc.get("activate_at", "never")
    if not (raw is None or str(raw).lower() == "never"):
        raise ValueError(
            f"tul.latent_pre_ref_config={name!r} has tul.activate_at={raw!r}: the stage-1 "
            f"target model must be a PLAIN model (TUL off), so the target is its own forward.")


def load_latent_pre_ref(ref_cfg: DictConfig, ckpt_path: str, device,
                        live_d_model: int, name: str = "<cfg>") -> MORPHTransformer:
    """Build the plain model from ``ref_cfg``, quantise it as its run did, load
    ``ckpt_path`` strictly, move it to ``device``, freeze it. Returns the model."""
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"tul.latent_pre_ref_ckpt not found: {ckpt_path}")
    _assert_plain(ref_cfg, name)
    if int(ref_cfg.model.d_model) != int(live_d_model):
        raise ValueError(f"the stage-1 target model ({name}) has d_model "
                         f"{int(ref_cfg.model.d_model)}, the live model {int(live_d_model)}.")
    with torch.random.fork_rng(devices=[]):
        # The build draws its random init from the CPU generator; the weights are
        # overwritten by the load below, and the fork keeps the live run's stream unmoved.
        ref = MORPHTransformer(build_morph_config(ref_cfg, tul=None))
        apply_quantization(ref, ref_cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    raw = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    def _canon(k: str) -> str:
        return k.replace("._orig_mod.", ".")
    canon_to_model = {_canon(k): k for k in ref.state_dict().keys()}
    state = {canon_to_model.get(_canon(k), k): v for k, v in raw.items()}
    missing, unexpected = ref.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"stage-1 target model {name} <- {ckpt_path}: {len(missing)} missing "
            f"(first {sorted(missing)[:5]}), {len(unexpected)} unexpected (first "
            f"{sorted(unexpected)[:5]}). The plain config and the checkpoint do not describe "
            f"the same model; refusing a half-loaded target.")
    ref = ref.to(device)
    for prm in ref.parameters():
        prm.requires_grad_(False)
    ref.eval()
    n = sum(p.numel() for p in ref.parameters())
    print(f"  [latent-pre] frozen target model {name}: {n / 1e6:.1f}M parameters, "
          f"checkpoint step {ck.get('step', '?') if isinstance(ck, dict) else '?'} "
          f"({ckpt_path}), {len(state)} tensors loaded strictly, eval mode, no grad",
          flush=True)
    return ref


def build_latent_pre_ref(tul_cfg, device, live_d_model: int,
                         run_ckpt_dir: str | None = None) -> MORPHTransformer:
    """The trainer's entry: compose ``tul_cfg.latent_pre_ref_config`` and load
    ``tul_cfg.latent_pre_ref_ckpt``. ``run_ckpt_dir`` (the live run's checkpoint
    directory, when known) refuses a reference that sits inside it."""
    path = os.path.abspath(str(tul_cfg.latent_pre_ref_ckpt))
    if run_ckpt_dir is not None:
        assert_not_live(path, run_ckpt_dir)
    name = str(tul_cfg.latent_pre_ref_config)
    return load_latent_pre_ref(compose_ref_config(name), path, device, live_d_model, name)


def assert_not_live(ref_ckpt: str, run_ckpt_dir: str) -> None:
    """Refuse a reference checkpoint inside the live run's own checkpoint directory."""
    a, d = os.path.realpath(ref_ckpt), os.path.realpath(run_ckpt_dir)
    if os.path.commonpath([a, d]) == d:
        raise ValueError(
            f"tul.latent_pre_ref_ckpt {ref_ckpt} is inside this run's own checkpoint "
            f"directory {run_ckpt_dir}: that is a live copy of the model being trained, "
            f"and the stage-1 target must be a FROZEN plain model.")


# ── the calibration (tul.latent_pre_target_norm: standard) ──────────────────────────


def _draw_batches(args: dict) -> dict:
    """The child process's body: a fresh loader, ``n_batches`` batches, the doc count."""
    from morph.training.data import create_dataloader
    stats: dict = {}
    it = create_dataloader(args["tokenizer"], args["dataset"], args["seq_len"],
                           args["batch_size"], split="train",
                           skip_samples=args["doc_offset"], tul=args["tul_data_cfg"],
                           doc_stats=stats)
    batches = [next(it) for _ in range(args["n_batches"])]
    it.close()
    return {"batches": batches, "docs": int(stats["docs"]), "tokens": int(stats["tokens"])}


def calibration_batches(tokenizer: str, dataset: str, seq_len: int, batch_size: int,
                        tul_data_cfg, doc_offset: int, n_batches: int,
                        run_tokens: int) -> tuple[list, dict]:
    """``n_batches`` TUL batches of the training stream from document ``doc_offset``, drawn
    in a separate process (module docstring). ``run_tokens`` (steps x batch x seq_len) is
    the run's own token budget: with the documents' measured mean length it estimates the
    run's document count, and a run that would reach ``doc_offset`` is REFUSED, as is a
    calibration that would read into the val documents. Returns ``(batches, info)``."""
    if not 0 <= int(doc_offset) < VAL_DOC_OFFSET:
        raise ValueError(f"tul.latent_pre_cal_doc_offset={doc_offset} must lie in "
                         f"[0, {VAL_DOC_OFFSET}) (the val loader starts at document "
                         f"{VAL_DOC_OFFSET}).")
    out = draw_in_subprocess(tokenizer, dataset, seq_len, batch_size, tul_data_cfg,
                             doc_offset, n_batches)
    return out["batches"], check_calibration_docs(int(doc_offset), out["docs"],
                                                  out["tokens"], int(run_tokens))


def draw_in_subprocess(tokenizer: str, dataset: str, seq_len: int, batch_size: int,
                       tul_data_cfg, doc_offset: int, n_batches: int) -> dict:
    """``{"batches", "docs", "tokens"}``: ``n_batches`` training batches from document
    ``doc_offset``, drawn by a fresh loader in a child process (``--draw`` below)."""
    args = {"tokenizer": str(tokenizer), "dataset": str(dataset), "seq_len": int(seq_len),
            "batch_size": int(batch_size), "tul_data_cfg": tul_data_cfg,
            "doc_offset": int(doc_offset), "n_batches": int(n_batches)}
    with tempfile.TemporaryDirectory(prefix="latent_pre_cal_") as tmp:
        a_path, o_path = os.path.join(tmp, "args.pkl"), os.path.join(tmp, "out.pt")
        with open(a_path, "wb") as f:
            pickle.dump(args, f)
        env = dict(os.environ)
        env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
        r = subprocess.run([sys.executable, "-m", "morph.training.latent_pre_ref", "--draw",
                            a_path, o_path], env=env, cwd=_REPO_ROOT, capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise RuntimeError(f"the calibration draw failed (exit {r.returncode}):\n"
                               f"{r.stderr[-4000:]}")
        out = torch.load(o_path, weights_only=False)
    return out


def check_calibration_docs(doc_offset: int, docs: int, tokens: int, run_tokens: int) -> dict:
    """Refuse a calibration that reads into the val documents, or a run whose estimated
    document count (``run_tokens`` at the calibration's measured tokens per document)
    reaches the calibration's first document. Returns the bookkeeping dict."""
    end = doc_offset + docs
    if end > VAL_DOC_OFFSET:
        raise ValueError(f"the calibration read documents {doc_offset}..{end}, into the val "
                         f"documents (from {VAL_DOC_OFFSET}). Lower the offset or the count.")
    tok_per_doc = tokens / max(docs, 1)
    est_run_docs = int(run_tokens / tok_per_doc)
    if est_run_docs >= doc_offset:
        raise ValueError(
            f"the run reads ~{est_run_docs} documents ({run_tokens} tokens at "
            f"{tok_per_doc:.0f} tokens/document), reaching the calibration's first document "
            f"{doc_offset}: the targets' statistics would come from documents the run "
            f"trains on. Raise tul.latent_pre_cal_doc_offset (below {VAL_DOC_OFFSET}).")
    return {"doc_first": doc_offset, "doc_end": end, "docs": docs, "tokens": tokens,
            "est_run_docs": est_run_docs}


@torch.no_grad()
def calibrate_latent_pre_ref(ref: MORPHTransformer, mode: str, batches: list,
                             device) -> dict:
    """Run the frozen model's ``mode`` target over ``batches`` (real span slots only: the
    targets' ``ok`` mask), set the fixed ``latent_pre_mu`` / ``latent_pre_sigma`` buffers
    (non-persistent) and return the statistics without the tensors. On CUDA the targets
    are computed under the training forward's bf16 autocast, so the statistics describe
    the same numbers the loss sees."""
    if ref.training or any(p.requires_grad for p in ref.parameters()):
        raise RuntimeError("calibrate_latent_pre_ref: the frozen model is not frozen.")
    fn = PLAIN_TARGET_FNS[mode]
    dev = torch.device(device)
    mom = TargetMoments(int(ref.cfg.d_model), dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for x, y, lay in batches:
            z, ok = fn(ref, x.to(dev), y.to(dev), lay.to(dev))
            mom.update(z, ok)
    st = mom.finalize(SIGMA_FLOOR_FRAC)
    ref.register_buffer("latent_pre_mu", st.pop("mu").to(dev), persistent=False)
    ref.register_buffer("latent_pre_sigma", st.pop("sigma").to(dev), persistent=False)
    return st


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--draw":
        raise SystemExit("usage: python -m morph.training.latent_pre_ref --draw ARGS OUT")
    with open(sys.argv[2], "rb") as f:
        _args = pickle.load(f)
    torch.save(_draw_batches(_args), sys.argv[3])
