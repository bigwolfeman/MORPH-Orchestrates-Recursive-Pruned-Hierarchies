"""Offline probe: does HRM-Text-1B's two-level (H/L) recurrent state rotate at a stable
scale, like Huginn's loop, or grow along a shared direction? Frozen question and
predictions, plus the dated Method amendment with the architecture facts this script
relies on: lab/experiments/planned/2026-10-03-hrm-text-loop-geometry.md (read-only, do not
edit the Predictions; the amendment was appended after it, before any GPU run).

Mirrors lab/huginn/huginn_loop_geometry.py (read-only, not edited by this file). Reused
from it: `load_rows` (the Huginn-tokenizer row loader over OpenWebText), `chunks_of`,
`pr_from_buf` (participation ratio of centred, per-token-normalised states). Reused from
lab/divergence/_stats.py: `paired_bootstrap_ci`.

Why two states instead of one: HRM-Text has a slow H state and a fast L state, iterated
`H_cycles x (L_cycles + 1)` times with additive injection (z_L = L_module(z_L + z_H), then
z_H = H_module(z_H + z_L) once per H cycle). Both `L_module` and `H_module` are genuine
`nn.Module`s with a real `forward()`, called through ordinary `__call__` -- unlike
Huginn's plain-method `core_block_forward`, so this probe captures them with the official,
additive `register_forward_hook` rather than the `__dict__`-shadow trick.

Faithfulness is a one-line check here, not a chain: `HrmTextModel.forward` returns the
H_module's last output directly as `last_hidden_state` (no further norm — `final_norm`
inside `HrmTextStack.forward` already parameterless-RMSNorms it), and
`HrmTextForCausalLM.forward` applies `lm_head` straight to that. So replaying CE at a
captured z_H is just `model.lm_head(z_h_state)`.

R-1 needs CE at 1 H cycle vs the trained count (2) vs up to 2x (4). `self.config.H_cycles`
drives the Python for-loop in `HrmTextModel.forward` and -- under inference, with
`use_cache=False` on every call so there is no pre-sized KV cache to overrun -- can be
monkey-patched upward for one extended run (H_cycles set to `2 * H_trained`) that captures
every depth from 1 to 2x in a single forward (no lookahead dependency: the trajectory up
to h = H_trained is identical whether the loop then stops or keeps going). A SEPARATE,
small, natural-depth (unmodified config) forward is run first for the primary
faithfulness proof; the extended run's own h = H_trained state is checked against it as a
bonus consistency proof.

Usage (GPU; run on the 3070, NOT the local 5090 -- see the 2026-10-03 amendment. No
gpu.lock: this probe does not touch the 5090):
  ~/morph-venv/bin/python lab/hrm/hrm_text_loop_geometry.py \\
    --device cuda --rows 64 --seq 1024 --batch 8 \\
    --dataset '/mnt/bigdata/hf/datasets/openwebtext/**/openwebtext-train-*.arrow' \\
    --out lab/experiments/results/2026-10-03-hrm-text-loop-geometry/probe.json

CPU smoke (tiny slice, build/dev only):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 19 \\
    /home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python lab/hrm/hrm_text_loop_geometry.py \\
    --device cpu --rows 2 --seq 64 --batch 2 --pr_subsample 32 \\
    --out /tmp/hrm_text_loop_geometry_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab/huginn"))
sys.path.insert(0, str(ROOT / "lab/divergence"))
from huginn_loop_geometry import (  # noqa: E402
    DEFAULT_DATASET, DEFAULT_REVISION as HUGINN_REVISION,
    DEFAULT_SNAPSHOT as HUGINN_SNAPSHOT, DEFAULT_SEED, chunks_of, load_rows, pr_from_buf,
)
from _stats import paired_bootstrap_ci  # noqa: E402

DEFAULT_HRM_REVISION = "22097cbcecdd1301afe30a19a3ee61b96a9863e5"
DEFAULT_HRM_SNAPSHOT = (
    "/home/wolfe/.cache/huggingface/hub/models--sapientinc--HRM-Text-1B/snapshots/"
    f"{DEFAULT_HRM_REVISION}"
)
# The `noisy` condition tag: "noisy / web-crawl style" -- the one documented condition
# that matches raw OpenWebText (see the 2026-10-03 Method amendment).
PREFIX_TOKENS = ["<|im_start|>", "<|quad_start|>"]


def load_hrm(snapshot: str, revision: str, device: str):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    source = str(Path(snapshot).expanduser())
    if Path(source).name != revision:
        raise ValueError("Snapshot directory does not match the frozen revision")
    cfg = AutoConfig.from_pretrained(source, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        source, config=cfg, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    if model.config is not model.model.config:
        raise RuntimeError(
            "model.config is not model.model.config -- the H_cycles monkey-patch this "
            "probe relies on would not reach the recurrence loop")
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("lm_head unexpectedly tied to embed_tokens (config says untied)")
    print(f"[load] transformers={transformers.__version__} H_cycles={cfg.H_cycles} "
          f"L_cycles={cfg.L_cycles} hidden={cfg.hidden_size} "
          f"num_layers_per_stack={cfg.num_layers_per_stack} prefix_lm={cfg.prefix_lm}",
          flush=True)
    return model, tok, cfg


def load_huginn_tokenizer(snapshot: str, revision: str):
    from transformers import AutoTokenizer
    source = str(Path(snapshot).expanduser())
    if Path(source).name != revision:
        raise ValueError("Huginn snapshot directory does not match the frozen revision")
    return AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)


def decode_rows_text(x_all: torch.Tensor, y_all: torch.Tensor, huginn_tok) -> list[str]:
    """Each row's full text: `cat(x_row, y_row[-1:])` is the row's raw token-id sequence
    (seq+1 ids) exactly as `huginn_depth_sweep.py::target_offsets` reconstructs it."""
    texts = []
    for i in range(x_all.shape[0]):
        ids = torch.cat((x_all[i], y_all[i, -1:])).tolist()
        texts.append(huginn_tok.decode(ids, skip_special_tokens=False))
    return texts


def row_text_sha256(texts: list[str]) -> str:
    return hashlib.sha256("\x1e".join(texts).encode("utf-8", errors="surrogatepass")).hexdigest()


def hrm_tokenize_rows(texts: list[str], hrm_tok, prefix_ids: list[int], body_len: int):
    """Builds fixed-length rows: `prefix_ids` (bidirectional, never scored) + `body_len`
    real-text tokens (causal). Needs `body_len + 1` HRM tokens per row's text (one extra
    for the last position's label) -- raises loudly if a row's text is too short, rather
    than silently padding or truncating the requirement away."""
    need = body_len + 1
    P = len(prefix_ids)
    L = P + body_len
    input_ids = torch.empty((len(texts), L), dtype=torch.long)
    labels = torch.full((len(texts), L), -100, dtype=torch.long)
    token_type_ids = torch.zeros((len(texts), L), dtype=torch.long)
    token_type_ids[:, :P] = 1
    raw_counts = np.zeros(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = hrm_tok(text, add_special_tokens=False)["input_ids"]
        raw_counts[i] = len(ids)
        if len(ids) < need:
            raise ValueError(
                f"row {i}: HRM tokenizer gave {len(ids)} tokens for this row's text, "
                f"need >= {need} (body_len={body_len} + 1 for the last label)")
        body = ids[:need]
        input_ids[i, :P] = torch.tensor(prefix_ids, dtype=torch.long)
        input_ids[i, P:] = torch.tensor(body[:body_len], dtype=torch.long)
        labels[i, P - 1:] = torch.tensor(body, dtype=torch.long)
    return input_ids, labels, token_type_ids, raw_counts


class DualCapture:
    """Registers `register_forward_hook` on `h_module` and `l_module` (official,
    additive -- the hook observes the already-computed output and returns None, so the
    recurrence's own computation is never touched). `on_h(h_idx, state)` / `on_l(l_idx,
    state)` fire with a 1-indexed, per-`__enter__` counter: h_idx counts H_module calls
    (= H cycles), l_idx counts L_module calls globally across H cycles (= total L steps)."""

    def __init__(self, h_module, l_module, on_h, on_l):
        self.h_module, self.l_module, self.on_h, self.on_l = h_module, l_module, on_h, on_l
        self.h = self.l = 0

    def __enter__(self):
        self.h = self.l = 0

        def hook_h(module, inputs, output):
            self.h += 1
            self.on_h(self.h, output)
            return None

        def hook_l(module, inputs, output):
            self.l += 1
            self.on_l(self.l, output)
            return None

        self._handle_h = self.h_module.register_forward_hook(hook_h)
        self._handle_l = self.l_module.register_forward_hook(hook_l)
        return self

    def __exit__(self, *exc):
        self._handle_h.remove()
        self._handle_l.remove()
        return False


def amp(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))


def rebuild_mask_and_rope(model, input_ids: torch.Tensor, token_type_ids: torch.Tensor):
    """Reproduces, verbatim, the mask/RoPE preprocessing `HrmTextModel.forward` runs once
    before the H/L loop -- needed only so the z_L intervention can re-run ONE `H_module`
    call by hand (the main readings never need this; the real `model(...)` call already
    does it internally)."""
    from transformers.masking_utils import create_causal_mask
    inputs_embeds = model.model.embed_tokens(input_ids) * model.model.embedding_scale
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    mask_kwargs = {"config": model.config, "inputs_embeds": inputs_embeds,
                   "attention_mask": torch.ones_like(input_ids), "past_key_values": None,
                   "position_ids": position_ids}
    if model.config.prefix_lm:
        mask_kwargs["block_sequence_ids"] = torch.where(token_type_ids == 1, 0, -1)
    attn_mask = create_causal_mask(**mask_kwargs)
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    return attn_mask, position_embeddings, position_ids


@torch.inference_mode()
def faithfulness_check(model, x_chunk, tt_chunk, device: str) -> tuple[float, float]:
    """Natural-depth (unmodified config.H_cycles) faithfulness: ref = model(...).logits;
    rebuilt = lm_head(captured z_H at the final H cycle). Also returns the captured
    z_H-at-final-cycle's own norm, logged for sanity (expected ~sqrt(hidden_size))."""
    H_trained = model.config.H_cycles
    captured: dict = {}

    def on_h(h, state):
        if h == H_trained:
            captured["z_h_final"] = state.detach().clone()

    def on_l(l, state):
        pass

    xb = x_chunk.to(device)
    ttb = tt_chunk.to(device)
    with DualCapture(model.model.H_module, model.model.L_module, on_h, on_l):
        with amp(device):
            ref = model(input_ids=xb, attention_mask=torch.ones_like(xb),
                       token_type_ids=ttb, use_cache=False)
    if "z_h_final" not in captured:
        raise RuntimeError("H_module hook never fired at the trained H_cycles depth")
    with amp(device):
        rebuilt_logits = model.lm_head(captured["z_h_final"])
    diff = float((ref.logits.float() - rebuilt_logits.float()).abs().max())
    return diff, float(captured["z_h_final"].float().norm(dim=-1).mean())


@torch.inference_mode()
def pass_moments(model, x_all, tt_all, chunks, H_eval: int, device: str):
    """Pass 1 (extended depth H_eval = 2x trained): accumulate per-h / per-l mean state
    and mean per-token norm. `H_eval` is written into `model.config.H_cycles` for the
    duration of this pass only (see the 2026-10-03 amendment: safe under inference with
    use_cache=False) and restored after."""
    C = model.config.hidden_size
    L_cycles = model.config.L_cycles
    L_eval = H_eval * L_cycles
    h_sum = {h: torch.zeros(C, dtype=torch.float64, device=device) for h in range(1, H_eval + 1)}
    h_norm = {h: torch.zeros((), dtype=torch.float64, device=device) for h in range(1, H_eval + 1)}
    h_count = {h: 0 for h in range(1, H_eval + 1)}
    l_sum = {l: torch.zeros(C, dtype=torch.float64, device=device) for l in range(1, L_eval + 1)}
    l_norm = {l: torch.zeros((), dtype=torch.float64, device=device) for l in range(1, L_eval + 1)}
    l_count = {l: 0 for l in range(1, L_eval + 1)}

    original_H = model.config.H_cycles
    model.config.H_cycles = H_eval
    try:
        for r0, r1 in chunks:
            xb, ttb = x_all[r0:r1].to(device), tt_all[r0:r1].to(device)

            def on_h(h, state):
                flat = state.float().reshape(-1, C).double()
                h_sum[h] += flat.sum(0)
                h_norm[h] += flat.norm(dim=-1).sum()
                h_count[h] += flat.shape[0]

            def on_l(l, state):
                flat = state.float().reshape(-1, C).double()
                l_sum[l] += flat.sum(0)
                l_norm[l] += flat.norm(dim=-1).sum()
                l_count[l] += flat.shape[0]

            with DualCapture(model.model.H_module, model.model.L_module, on_h, on_l):
                with amp(device):
                    model(input_ids=xb, attention_mask=torch.ones_like(xb),
                         token_type_ids=ttb, use_cache=False)
    finally:
        model.config.H_cycles = original_H

    mu_h = {h: h_sum[h] / h_count[h] for h in h_sum}
    mu_l = {l: l_sum[l] / l_count[l] for l in l_sum}
    mean_norm_h = {h: float(h_norm[h] / h_count[h]) for h in h_norm}
    mean_norm_l = {l: float(l_norm[l] / l_count[l]) for l in l_norm}
    v_h = F.normalize(sum(h_sum.values()) / sum(h_count.values()), dim=0)
    v_l = F.normalize(sum(l_sum.values()) / sum(l_count.values()), dim=0)
    return mu_h, mu_l, mean_norm_h, mean_norm_l, v_h, v_l


@torch.inference_mode()
def pass_main(model, x_all, y_mask, chunks, H_eval: int, mu_h, mu_l, v_h, v_l, device: str,
             pr_rows, pr_toks, rand_dir, H_trained: int):
    """Pass 2 (extended depth H_eval): cos/along/perp/centred RMS + the k-1->k step for
    both states, CE per H cycle (z_H only, via `lm_head`), the fixed-subsample
    participation ratio for both states, and the three interventions on the TRAINED
    depth's final z_H and final z_L (recombined through one honest `H_module` call, see
    `rebuild_mask_and_rope`), paired per row."""
    C = model.config.hidden_size
    L_cycles = model.config.L_cycles
    L_eval = H_eval * L_cycles
    total_rows, L = x_all.shape

    def zeros_f64():
        return torch.zeros((), dtype=torch.float64, device=device)

    acc_h = {h: {"cos": zeros_f64(), "along": zeros_f64(), "perp": zeros_f64(),
                "cen": zeros_f64(), "n": 0} for h in range(1, H_eval + 1)}
    acc_l = {l: {"cos": zeros_f64(), "along": zeros_f64(), "perp": zeros_f64(),
                "cen": zeros_f64(), "n": 0} for l in range(1, L_eval + 1)}
    step_h = {h: {"step": zeros_f64(), "base": zeros_f64(), "along": zeros_f64()}
             for h in range(2, H_eval + 1)}
    step_l = {l: {"step": zeros_f64(), "base": zeros_f64(), "along": zeros_f64()}
             for l in range(2, L_eval + 1)}
    pr_buf_h = {h: [] for h in range(1, H_eval + 1)}
    pr_buf_l = {l: [] for l in range(1, L_eval + 1)}
    ce_row_sum = {h: np.zeros(total_rows, dtype=np.float64) for h in range(1, H_eval + 1)}
    ce_row_n = np.zeros(total_rows, dtype=np.float64)
    interv_h = {"rescale": np.zeros(total_rows, dtype=np.float64),
               "remove_v": np.zeros(total_rows, dtype=np.float64),
               "remove_rand": np.zeros(total_rows, dtype=np.float64)}
    interv_l = {"rescale": np.zeros(total_rows, dtype=np.float64),
               "remove_v": np.zeros(total_rows, dtype=np.float64),
               "remove_rand": np.zeros(total_rows, dtype=np.float64)}
    base_ce_final = np.zeros(total_rows, dtype=np.float64)
    norm_h1 = {"v": None}
    norm_l1 = {"v": None}

    def ce_from_zh(state, yb) -> torch.Tensor:
        # 65536-vocab logits over a full 1024-token row are the dominant memory cost on an
        # 8 GB card (fp32 would be [b, seq, 65536] * 4 bytes). Keep logits in bf16 and let
        # F.cross_entropy's fused kernel upcast internally for the log-sum-exp -- this is
        # the same numerical path HF's own loss functions use for a bf16 model; it halves
        # the resident bytes versus an explicit `.float()` copy of the same tensor.
        with amp(device):
            logits = model.lm_head(state.to(model.lm_head.weight.dtype))
        loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten().clamp_min(0),
                               reduction="none", ignore_index=-100).reshape_as(yb).float()
        valid = yb != -100
        if not torch.isfinite(loss[valid]).all():
            raise FloatingPointError("non-finite CE in the HRM loop-geometry probe")
        return loss

    original_H = model.config.H_cycles
    model.config.H_cycles = H_eval
    try:
        for r0, r1 in chunks:
            b = r1 - r0
            xb = x_all[r0:r1].to(device)
            ttb = y_mask["token_type_ids"][r0:r1].to(device)
            yb = y_mask["labels"][r0:r1].to(device)
            in_chunk = (pr_rows >= r0) & (pr_rows < r1)
            pr_local = (torch.as_tensor(pr_rows[in_chunk] - r0, device=device, dtype=torch.long),
                       torch.as_tensor(pr_toks[in_chunk], device=device, dtype=torch.long))
            prev_h, prev_l = {"x": None}, {"x": None}
            retained_h, retained_l = {}, {}

            def on_h(h, state):
                xf = state.float()
                flat = xf.reshape(-1, C).double()
                acc_h[h]["cos"] += F.normalize(flat, dim=-1).matmul(v_h.double()).sum()
                cen = flat - mu_h[h].double()
                a = cen @ v_h.double()
                acc_h[h]["along"] += a.square().sum()
                acc_h[h]["perp"] += (cen.square().sum(-1) - a.square()).sum()
                acc_h[h]["cen"] += cen.square().sum(-1).sum()
                acc_h[h]["n"] += flat.shape[0]
                if prev_h["x"] is not None and h >= 2:
                    step = xf.double().reshape(-1, C) - prev_h["x"]
                    sa = step @ v_h.double()
                    step_h[h]["step"] += step.square().sum()
                    step_h[h]["base"] += prev_h["x"].square().sum()
                    step_h[h]["along"] += sa.square().sum()
                prev_h["x"] = xf.double().reshape(-1, C)
                ce = ce_from_zh(state, yb)
                ce_row_sum[h][r0:r1] += ce.double().sum(dim=1).cpu().numpy()
                if h == 1:
                    ce_row_n[r0:r1] += (yb != -100).sum(dim=1).double().cpu().numpy()
                if pr_local[0].numel():
                    sub = xf[pr_local[0], pr_local[1]]
                    pr_buf_h[h].append(F.normalize(sub.double(), dim=-1).cpu())
                if h == 1:
                    norm_h1["v"] = xf.norm(dim=-1, keepdim=True).detach().clone()
                if h == H_trained - 1 or h == H_trained:
                    retained_h[h] = state.detach().clone()

            def on_l(l, state):
                xf = state.float()
                flat = xf.reshape(-1, C).double()
                acc_l[l]["cos"] += F.normalize(flat, dim=-1).matmul(v_l.double()).sum()
                cen = flat - mu_l[l].double()
                a = cen @ v_l.double()
                acc_l[l]["along"] += a.square().sum()
                acc_l[l]["perp"] += (cen.square().sum(-1) - a.square()).sum()
                acc_l[l]["cen"] += cen.square().sum(-1).sum()
                acc_l[l]["n"] += flat.shape[0]
                if prev_l["x"] is not None and l >= 2:
                    step = xf.double().reshape(-1, C) - prev_l["x"]
                    sa = step @ v_l.double()
                    step_l[l]["step"] += step.square().sum()
                    step_l[l]["base"] += prev_l["x"].square().sum()
                    step_l[l]["along"] += sa.square().sum()
                prev_l["x"] = xf.double().reshape(-1, C)
                if pr_local[0].numel():
                    sub = xf[pr_local[0], pr_local[1]]
                    pr_buf_l[l].append(F.normalize(sub.double(), dim=-1).cpu())
                if l == 1:
                    norm_l1["v"] = xf.norm(dim=-1, keepdim=True).detach().clone()
                if l == model.config.L_cycles * H_trained:
                    retained_l["final"] = state.detach().clone()

            with DualCapture(model.model.H_module, model.model.L_module, on_h, on_l):
                with amp(device):
                    model(input_ids=xb, attention_mask=torch.ones_like(xb),
                         token_type_ids=ttb, use_cache=False)

            if norm_h1["v"] is None or norm_l1["v"] is None:
                raise RuntimeError("h=1 / l=1 state never captured")
            if (H_trained - 1) not in retained_h and H_trained != 1:
                raise RuntimeError(f"z_H_prev (h={H_trained - 1}) never captured")
            z_h_final = retained_h[H_trained].float()
            z_l_final = retained_l["final"].float()
            z_h_prev = retained_h[H_trained - 1].float() if H_trained > 1 else None

            base_ce = ce_from_zh(z_h_final, yb)
            base_ce_final[r0:r1] += base_ce.double().sum(dim=1).cpu().numpy()

            cur_norm = z_h_final.norm(dim=-1, keepdim=True)
            rescale = (z_h_final.double() * (norm_h1["v"].double() / cur_norm.double().clamp_min(1e-12))).to(z_h_final.dtype)
            av = (z_h_final.double() @ v_h.double()).unsqueeze(-1)
            remove_v = (z_h_final.double() - av * v_h.double()).to(z_h_final.dtype)
            ar = (z_h_final.double() @ rand_dir.double()).unsqueeze(-1)
            remove_rand = (z_h_final.double() - ar * rand_dir.double()).to(z_h_final.dtype)
            for name, st in (("rescale", rescale), ("remove_v", remove_v), ("remove_rand", remove_rand)):
                ce = ce_from_zh(st, yb)
                interv_h[name][r0:r1] += ce.double().sum(dim=1).cpu().numpy()

            if H_trained > 1:
                attn_mask4d, pos_emb, pos_ids = rebuild_mask_and_rope(model, xb, ttb)

                def ce_from_zl(zl_state):
                    model_dtype = model.model.embed_tokens.weight.dtype
                    combined = (z_h_prev.to(zl_state.dtype) + zl_state).to(model_dtype)
                    with amp(device):
                        z_h_recombined = model.model.H_module(
                            combined, attention_mask=attn_mask4d,
                            past_key_values=None, position_embeddings=pos_emb,
                            position_ids=pos_ids, cycle_offset=0, use_cache=False)
                    return ce_from_zh(z_h_recombined, yb)

                cur_norm_l = z_l_final.norm(dim=-1, keepdim=True)
                rescale_l = (z_l_final.double() * (norm_l1["v"].double() / cur_norm_l.double().clamp_min(1e-12))).to(z_l_final.dtype)
                avl = (z_l_final.double() @ v_l.double()).unsqueeze(-1)
                remove_v_l = (z_l_final.double() - avl * v_l.double()).to(z_l_final.dtype)
                arl = (z_l_final.double() @ rand_dir.double()).unsqueeze(-1)
                remove_rand_l = (z_l_final.double() - arl * rand_dir.double()).to(z_l_final.dtype)
                for name, st in (("rescale", rescale_l), ("remove_v", remove_v_l), ("remove_rand", remove_rand_l)):
                    ce = ce_from_zl(st.to(device))
                    interv_l[name][r0:r1] += ce.double().sum(dim=1).cpu().numpy()
    finally:
        model.config.H_cycles = original_H

    return {"acc_h": acc_h, "acc_l": acc_l, "step_h": step_h, "step_l": step_l,
            "pr_buf_h": pr_buf_h, "pr_buf_l": pr_buf_l, "ce_row_sum": ce_row_sum,
            "ce_row_n": ce_row_n, "base_ce_final": base_ce_final, "interv_h": interv_h,
            "interv_l": interv_l}


def grade_predictions(readings_h: dict, readings_l: dict, interventions_h: dict,
                      interventions_l: dict, H_trained: int, L_cycles: int) -> dict:
    """Mechanical check of R-1..R-6, graded against the TRAINED depth (h<=H_trained,
    l<=H_trained*L_cycles), not the extended 2x run -- the extended depths exist only to
    answer R-1's sweep. n=1 run, one seed: report, do not over-read."""
    g: dict = {}
    L_trained = H_trained * L_cycles
    if 1 in readings_h and H_trained in readings_h:
        ce1, ce_trained = readings_h[1]["ce"], readings_h[H_trained]["ce"]
        g["R1_ce_drop_1_to_trained"] = {"value": ce1 - ce_trained,
                                        "holds": (ce1 - ce_trained) > 0.1}
        ratio_h = readings_h[H_trained]["mean_state_norm"] / readings_h[1]["mean_state_norm"]
        g["R2_norm_ratio_H_z_H"] = {"value": ratio_h, "holds": 0.0 < ratio_h <= 1.5}
    if 1 in readings_l and L_trained in readings_l:
        ratio_l = readings_l[L_trained]["mean_state_norm"] / readings_l[1]["mean_state_norm"]
        g["R2_norm_ratio_L_z_L"] = {"value": ratio_l, "holds": 0.0 < ratio_l <= 1.5}
    shares = [readings_h[h]["step_share_along_v"] for h in range(2, H_trained + 1)
             if readings_h.get(h, {}).get("step_share_along_v") is not None]
    if shares:
        g["R3_max_step_share_along_v_H_from_cycle2"] = {"value": max(shares),
                                                         "holds": max(shares) < 0.30}
    rel_h = [readings_h[h]["step_rel_rms"] for h in range(2, H_trained + 1)
            if readings_h.get(h, {}).get("step_rel_rms") is not None]
    rel_l = [readings_l[l]["step_rel_rms"] for l in range(2, L_trained + 1)
            if readings_l.get(l, {}).get("step_rel_rms") is not None]
    if rel_h and rel_l:
        mean_h, mean_l = float(np.mean(rel_h)), float(np.mean(rel_l))
        g["R4_mean_step_rel_H_vs_L"] = {"value": mean_h - mean_l, "mean_H": mean_h,
                                        "mean_L": mean_l, "holds": mean_h < mean_l}
    if 1 in readings_h and H_trained in readings_h:
        pr1, pr_t = readings_h[1].get("pr"), readings_h[H_trained].get("pr")
        if pr1 is not None and pr_t is not None:
            growth = (pr_t - pr1) / pr1
            g["R5_pr_growth_H_1_to_trained"] = {"value": growth, "holds": growth > 0.30}
    if "remove_rand" in interventions_h and "remove_v" in interventions_h:
        dr, dv = interventions_h["remove_rand"]["point"], interventions_h["remove_v"]["point"]
        g["R6_remove_rand_delta_ce_H"] = {"value": dr, "holds": dr < 0.01}
        g["R6_remove_v_delta_ce_H"] = {"value": dv, "holds": dv > 0.05}
    return g


def build_table(readings: dict, with_ce: bool) -> str:
    header = f"{'k':>4} {'norm':>9} {'cos_v':>7} {'rms_v':>9} {'rms_perp':>9} {'step_rel':>9} {'step_v':>7} {'pr':>8}"
    if with_ce:
        header += f" {'ce':>8}"
    lines = [header]
    for k in sorted(readings):
        r = readings[k]
        line = (f"{k:>4} {r['mean_state_norm']:>9.3f} {r['cos_to_v']:>7.3f} "
               f"{r['rms_along_v']:>9.3f} {r['rms_perp_v']:>9.3f} "
               f"{r.get('step_rel_rms', float('nan')):>9.4f} "
               f"{r.get('step_share_along_v', float('nan')):>7.3f} "
               f"{r.get('pr', float('nan')):>8.2f}")
        if with_ce:
            line += f" {r.get('ce', float('nan')):>8.4f}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hrm_revision", default=DEFAULT_HRM_REVISION)
    ap.add_argument("--hrm_snapshot", default=DEFAULT_HRM_SNAPSHOT)
    ap.add_argument("--huginn_revision", default=HUGINN_REVISION)
    ap.add_argument("--huginn_snapshot", default=HUGINN_SNAPSHOT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024, help="total row length, matching Huginn")
    ap.add_argument("--batch", type=int, default=8, help="sub-batch of rows per forward call")
    ap.add_argument("--pr_subsample", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip_samples", type=int, default=0)
    ap.add_argument("--bootstrap_n", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    ap.add_argument("--bootstrap_level", type=float, default=0.95)
    ap.add_argument("--cpu_threads", type=int, default=0)
    ap.add_argument("--faithfulness_only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    t0 = time.time()
    model, hrm_tok, hrm_cfg = load_hrm(args.hrm_snapshot, args.hrm_revision, args.device)
    H_trained, L_cycles = hrm_cfg.H_cycles, hrm_cfg.L_cycles
    huginn_tok = load_huginn_tokenizer(args.huginn_snapshot, args.huginn_revision)
    print(f"[load] HRM H_trained={H_trained} L_cycles={L_cycles} ({time.time() - t0:.1f}s)",
          flush=True)

    x_huginn, y_huginn = load_rows(Path(args.huginn_snapshot), args.dataset, args.seq,
                                   args.rows, args.skip_samples)
    print(f"[data] {args.rows} rows x {args.seq} Huginn tokens loaded ({time.time() - t0:.1f}s)",
          flush=True)
    texts = decode_rows_text(x_huginn, y_huginn, huginn_tok)
    row_hash = row_text_sha256(texts)
    print(f"ROWTEXT_SHA256={row_hash}", flush=True)

    prefix_ids = hrm_tok.convert_tokens_to_ids(PREFIX_TOKENS)
    if any(i is None or i == hrm_tok.unk_token_id for i in prefix_ids):
        raise ValueError(f"prefix tokens did not resolve cleanly: {PREFIX_TOKENS} -> {prefix_ids}")
    body_len = args.seq - len(prefix_ids)
    if body_len < 4:
        raise ValueError(f"--seq {args.seq} too small for a {len(prefix_ids)}-token prefix")
    input_ids, labels, token_type_ids, raw_counts = hrm_tokenize_rows(
        texts, hrm_tok, prefix_ids, body_len)
    print(f"[data] HRM row tokens: min={int(raw_counts.min())} max={int(raw_counts.max())} "
         f"mean={raw_counts.mean():.1f} (need >= {body_len + 1}); prefix={PREFIX_TOKENS}="
         f"{prefix_ids} ({time.time() - t0:.1f}s)", flush=True)

    chunks = chunks_of(args.rows, args.batch)
    r0, r1 = 0, min(2, args.rows)  # full-vocab fp32 logits are large; keep this check small
    diff, h_final_norm = faithfulness_check(model, input_ids[r0:r1], token_type_ids[r0:r1],
                                            args.device)
    print(f"FAITHFULNESS max_abs_logit_diff={diff:.6g} (H_cycles={H_trained}, "
         f"{r1 - r0} rows, z_H_final norm={h_final_norm:.4f}) ({time.time() - t0:.1f}s)",
         flush=True)
    if diff > 0.05:
        raise RuntimeError(
            f"faithfulness check failed: max abs logit diff {diff} exceeds 0.05 -- "
            "lm_head(captured z_H) does not match model(...).logits")
    if args.faithfulness_only:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"faithfulness_max_abs_logit_diff": diff, "row_text_sha256": row_hash},
                  open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
        return

    H_eval = 2 * H_trained
    mu_h, mu_l, mean_norm_h, mean_norm_l, v_h, v_l = pass_moments(
        model, input_ids, token_type_ids, chunks, H_eval, args.device)
    print(f"[pass1] moments + v done ({time.time() - t0:.1f}s)", flush=True)

    rng = np.random.default_rng(args.seed + 777)
    total_tok = args.rows * args.seq
    n_sub = min(args.pr_subsample, total_tok)
    flat_idx = rng.choice(total_tok, size=n_sub, replace=False)
    pr_rows, pr_toks = flat_idx // args.seq, flat_idx % args.seq

    g_cpu = torch.Generator(device="cpu").manual_seed(args.seed + 999)
    rand_dir = F.normalize(torch.randn(hrm_cfg.hidden_size, generator=g_cpu), dim=0).to(args.device)

    acc = pass_main(model, input_ids, {"token_type_ids": token_type_ids, "labels": labels},
                    chunks, H_eval, mu_h, mu_l, v_h, v_l, args.device, pr_rows, pr_toks,
                    rand_dir, H_trained)
    print(f"[pass2] main readings done ({time.time() - t0:.1f}s)", flush=True)

    readings_h: dict = {}
    for h in range(1, H_eval + 1):
        n = acc["acc_h"][h]["n"]
        row = {"mean_state_norm": mean_norm_h[h],
              "cos_to_v": float(acc["acc_h"][h]["cos"] / n),
              "rms_along_v": float((acc["acc_h"][h]["along"] / n) ** 0.5),
              "rms_perp_v": float((acc["acc_h"][h]["perp"] / n) ** 0.5),
              "rms_centred": float((acc["acc_h"][h]["cen"] / n) ** 0.5),
              "ce": float(acc["ce_row_sum"][h].sum() / acc["ce_row_n"].sum())}
        if h >= 2:
            sn = float(acc["step_h"][h]["step"])
            row["step_rel_rms"] = (float(acc["step_h"][h]["step"] / n) ** 0.5
                                   / float(acc["step_h"][h]["base"] / n) ** 0.5)
            row["step_share_along_v"] = float(acc["step_h"][h]["along"]) / sn if sn > 0 else None
        if acc["pr_buf_h"][h]:
            row["pr"] = pr_from_buf(acc["pr_buf_h"][h])
        readings_h[h] = row

    L_eval = H_eval * L_cycles
    readings_l: dict = {}
    for l in range(1, L_eval + 1):
        n = acc["acc_l"][l]["n"]
        row = {"mean_state_norm": mean_norm_l[l],
              "cos_to_v": float(acc["acc_l"][l]["cos"] / n),
              "rms_along_v": float((acc["acc_l"][l]["along"] / n) ** 0.5),
              "rms_perp_v": float((acc["acc_l"][l]["perp"] / n) ** 0.5),
              "rms_centred": float((acc["acc_l"][l]["cen"] / n) ** 0.5),
              "h_cycle": (l - 1) // L_cycles + 1, "l_within_h": (l - 1) % L_cycles + 1}
        if l >= 2:
            sn = float(acc["step_l"][l]["step"])
            row["step_rel_rms"] = (float(acc["step_l"][l]["step"] / n) ** 0.5
                                   / float(acc["step_l"][l]["base"] / n) ** 0.5)
            row["step_share_along_v"] = float(acc["step_l"][l]["along"]) / sn if sn > 0 else None
        if acc["pr_buf_l"][l]:
            row["pr"] = pr_from_buf(acc["pr_buf_l"][l])
        readings_l[l] = row

    interventions_h = {name: paired_bootstrap_ci(acc["interv_h"][name], acc["base_ce_final"],
                                                  acc["ce_row_n"], n_boot=args.bootstrap_n,
                                                  seed=args.bootstrap_seed, level=args.bootstrap_level)
                       for name in ("rescale", "remove_v", "remove_rand")}
    interventions_l = ({name: paired_bootstrap_ci(acc["interv_l"][name], acc["base_ce_final"],
                                                   acc["ce_row_n"], n_boot=args.bootstrap_n,
                                                   seed=args.bootstrap_seed, level=args.bootstrap_level)
                        for name in ("rescale", "remove_v", "remove_rand")}
                       if H_trained > 1 else {})

    grading = grade_predictions(readings_h, readings_l, interventions_h, interventions_l,
                                H_trained, L_cycles)

    print("\n== z_H (per H cycle; CE via lm_head(z_H)) ==")
    print(build_table(readings_h, with_ce=True))
    print("\n== z_L (per L step, global; h_cycle/l_within_h in the JSON) ==")
    print(build_table(readings_l, with_ce=False))
    print("\nInterventions on the TRAINED-depth final z_H (delta CE vs unmodified, 95% paired CI):")
    for name, ci in interventions_h.items():
        print(f"  z_H {name:>12}: {ci['point']:+.5f} [{ci['lo']:+.5f}, {ci['hi']:+.5f}] (n_units={ci['n_units']})")
    if interventions_l:
        print("Interventions on the TRAINED-depth final z_L (recombined through one honest H_module call):")
        for name, ci in interventions_l.items():
            print(f"  z_L {name:>12}: {ci['point']:+.5f} [{ci['lo']:+.5f}, {ci['hi']:+.5f}] (n_units={ci['n_units']})")
    print("\nPrediction grading:")
    for name, g in grading.items():
        print(f"  {name}: {g}")

    out = {
        "config": {k: v for k, v in vars(args).items()},
        "wall_s": round(time.time() - t0, 1),
        "row_text_sha256": row_hash,
        "raw_hrm_token_counts": {"min": int(raw_counts.min()), "max": int(raw_counts.max()),
                                 "mean": float(raw_counts.mean())},
        "prefix_tokens": PREFIX_TOKENS, "prefix_ids": prefix_ids, "body_len": body_len,
        "H_trained": H_trained, "L_cycles": L_cycles, "H_eval": H_eval,
        "faithfulness_max_abs_logit_diff": diff,
        "readings_by_h": readings_h, "readings_by_l": readings_l,
        "interventions_h": interventions_h, "interventions_l": interventions_l,
        "grading": grading,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out} ({out['wall_s']}s)")


if __name__ == "__main__":
    main()
