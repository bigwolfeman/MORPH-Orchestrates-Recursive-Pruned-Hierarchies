"""MORPH training entry point.

Usage:
    python morph/training/train.py                        # base config
    python morph/training/train.py training.steps=50000   # override
    python morph/training/train.py +training.ternary=true # phase 2

Config is managed by Hydra. See morph/configs/base.yaml for defaults.
All hyperparameters are logged to wandb at run start (full config dict).
"""

from __future__ import annotations

import gc
import math
import random as _random
import json
import os
import sys
import time
from typing import Optional

from morph.training.ckpt_retention import RetentionRing, existing_step_checkpoints
from morph.training.optimizer import align_optimizer_state

# Diagnostic-only: env-guarded single-shot faulthandler to capture the stack of an
# intermittent step-0 backward hang (gradient-checkpoint recompute). Single dump
# (NOT repeat=True — repeated dumps SIGSEGV under live CUDA). Set MORPH_FAULT_TIMEOUT
# to a value past a healthy step-0 (~40s) so it only fires when actually wedged.
if os.environ.get("MORPH_FAULT_TIMEOUT"):
    import faulthandler as _fh
    _fh.dump_traceback_later(int(os.environ["MORPH_FAULT_TIMEOUT"]), repeat=False)

# ── torch.compile safety ───────────────────────────────────────────────────────
# The looped core uses gradient checkpointing (use_reentrant=False); disabling
# donated buffers below avoids buffer-aliasing conflicts under compile.
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable donated-buffer reuse: the looped core's compiled + checkpointed
# (use_reentrant=False) backward can otherwise alias input buffers. The knob
# lives in torch._functorch.config but that submodule is not auto-imported by
# `import torch`, so import it explicitly and set it where it exists (the path
# has shifted across torch versions — guard rather than guess).
import torch._functorch.config as _functorch_config  # noqa: E402

if hasattr(_functorch_config, "donated_buffer"):
    _functorch_config.donated_buffer = False

# Inductor compile workers via SPAWN, not the default fork-based "subprocess" pool.
# WHY: any torch.compile RECOMPILE during the training loop forks compile workers /
# gcc while background threads (wandb asyncio + status, HF-streaming httpx, the
# inductor read-thread) hold a glibc malloc-arena lock → the forked child deadlocks
# in __triton_launcher.c (intermittent; cost a full night — see Ai-notes 06-01-2026/
# MORPH-eval-recompile-hang). Recompiles are unavoidable here (the active-set's
# grad_mode/dtype/size guards leak past warmup). Spawn workers are FRESH processes
# that never inherit the main thread-lock state, so compilation can never fork-deadlock
# regardless of when a recompile fires. Verified: 60 real training steps with live
# recompiles, no wedge. Pair with the single-threaded warmup below (handles the initial
# bulk compile). This applies to ALL runs incl. the future pruning run.
# Spawn workers (MORPH_COMPILE_CARVED) are only needed when compiling the carved path.
# At d=768 carved-eager is faster than carved-compiled (grad_mode guard thrashing) and
# spawn caused a BrokenProcessPool on startup compile. Default: eager inductor workers +
# carved path runs eager via eager_on_recompile. Gate kept for cloud-scale revisit.
import torch._inductor.config as _inductor_config  # noqa: E402
if os.environ.get("MORPH_COMPILE_CARVED"):
    if hasattr(_inductor_config, "worker_start_method"):
        _inductor_config.worker_start_method = "spawn"
    os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path so morph.* imports work when run from repo root.
_MORPH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Package should be pip-installed; no sys.path hack needed
# sys.path.insert(0, _MORPH_ROOT)

import wandb

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.routing import collect_routing_aux_losses, collect_routing_stats
from morph.training.divergence_guard import BlockGainGuard, CoreShareGuard
from morph.training.code_eval import code_marginal_ce
from morph.training.data import create_dataloader
from morph.training.optimizer import create_optimizer, create_lr_schedule
from morph.training.pruning import PruningSchedule


# ── Evaluation ────────────────────────────────────────────────────────────────

# The eval readings of `tul.slot_cell_pass_norm` (`MORPHTransformer._fold_cell_norm_stats`).
# Explicit prefixes, not "slot_cell_": the register's `slot_cell_eff_rank` /
# `slot_cell_pairwise_cos` have their own home and must not be logged twice.
_CELL_NORM_KEYS = ("slot_cell_rms", "slot_cell_mean_rms", "slot_cell_norm_g")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    device: torch.device,
    loader,
    n_batches: int = 20,
    tul: bool = False,
    extra: dict | None = None,
    halt: bool = False,
) -> tuple[float, float]:
    """Return (avg_loss, ppl) over n_batches validation steps.

    ``tul=True``: the val loader yields the 3-tuple (input_ids, labels, slot_layout) and
    the model is called with the layout ON and bag_size 0 (spec invariant 6). The §7.2
    metrics — val/ppl_tokens, val/first_tok_ce, val/plan_nats,
    val/first_tok_counterfactual, layer-passes/token — are accumulated into ``extra``.
    val/ppl_tokens is over TOKEN positions only, so it stays comparable to the
    baseline's token PPL."""
    model.eval()
    losses: list[float] = []
    acc: dict[str, list[float]] = {}
    # getattr-chained on purpose: this function is also driven by the CE stub models in
    # tests/test_train_phase.py, which have no `cfg` at all. Assigned HERE, before the
    # tul/plain branch, because the code-rank abort at the end reads it on BOTH paths:
    # from d0a2308 (2026-09-15) to 2026-09-21 it was assigned inside the TUL branch only,
    # and every plain-model eval died with UnboundLocalError (the reach-split smokes,
    # tests/test_train_evaluate_plain.py).
    _tul_cfg = getattr(getattr(getattr(model, "_orig_mod", model), "cfg", None), "tul", None)
    for _ in range(n_batches):
        try:
            batch = next(loader)
        except StopIteration:
            break
        if tul:
            x, y, layout = batch
            x, y, layout = x.to(device), y.to(device), layout.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _m = getattr(model, "_orig_mod", model)
                out = _m.tul_forward_with_plan_nats(x, y, layout)
            # tul.slot_cell_pass_norm (eval only): the per-pass carried-cell RMS
            # (`slot_cell_rms_t{t}`, `slot_cell_mean_rms_t{t}`, `slot_cell_rms_entry`) and
            # the shared gain's `slot_cell_norm_g_{mean,std,rms}`. A scan: the number of
            # passes is the batch's realised max depth. Before the stage-1 `continue`, so a
            # normed latent_pre model logs them too.
            for _sk in [k for k in out if str(k).startswith(_CELL_NORM_KEYS)]:
                if torch.is_tensor(out[_sk]):
                    acc.setdefault(f"val/{_sk}", []).append(float(out[_sk]))
            if out.get("latent_pre") is not None:
                # Stage-1 latent pretraining (tul.latent_pre_target): NO coda ran, so there
                # is no token CE and none of the coda readings below exist. The val loss of
                # this model IS its latent objective (`latent_pre`, unweighted); every
                # `latent_pre_*` reading (per-pass retrieval, chance, R^2, ranks) is
                # accumulated under `val/`. (`latent_pre_weighted` is in the subtraction
                # tuple below for every model that also has a CE; here nothing is left.)
                losses.append(float(out["latent_pre"]))
                for _lk in [k for k in out if str(k).startswith("latent_pre")]:
                    if torch.is_tensor(out[_lk]):
                        acc.setdefault(f"val/{_lk}", []).append(float(out[_lk]))
                for _gk in ("gain_est", "fixed_point"):
                    if out.get(_gk) is not None:
                        acc.setdefault(f"val/{_gk}", []).append(float(out[_gk]))
                continue
            _l = out["loss"].item()
            if out.get("mux_weighted") is not None:
                _l -= float(out["mux_weighted"])   # val loss = model CE (see train/loss note)
            if out.get("sigreg_weighted") is not None:
                _l -= float(out["sigreg_weighted"])
            if out.get("gain_reg_weighted") is not None:
                _l -= float(out["gain_reg_weighted"])
            if out.get("spandec_weighted") is not None:
                _l -= float(out["spandec_weighted"])
            for _aux2 in ("egrad_weighted", "pass_res_weighted", "oracle_z_weighted",
                          "spandec_pass_weighted", "coda_span_weighted",
                          "core_token_aux_weighted", "critic_weighted",
                          "vq_weighted", "row_contrast_weighted", "horizon_weighted",
                          "code_fm_weighted", "code_sigreg_weighted", "code_target_weighted",
                          "code_grade_weighted",
                          # tul.code_enc_var_lambda (LCTUL-J Stage 1, 2026-09-22): the
                          # online variance floor, weighted separately from
                          # code_target_weighted (see _tul_code_target_write's
                          # docstring). Subtracted so val loss stays the MODEL's CE.
                          "code_enc_var_weighted",
                          # tul.loop_denoise (LXTUL-P change 1, 2026-09-21): the
                          # per-pass denoising sum. Subtracted so val loss stays the
                          # MODEL's CE and the ppl guard fires on the language model.
                          "loop_denoise_weighted",
                          # tul.gram (LXTUL-G, 2026-09-23): beta * KL / n_tokens. At
                          # eval the default draw is the PRIOR and no KL is built, so it
                          # is absent there; listed so a posterior eval stays the CE.
                          "gram_kl_weighted",
                          # tul.coda_logit_l2 (spectral decoupling, 2026-09-23): folded
                          # straight into the fused CE kernel, so subtracted the same
                          # way so val loss stays the MODEL's CE.
                          "coda_logit_l2_weighted",
                          # tul.spandec_parallel (LXTUL-E, 2026-09-23): the parallel
                          # head's mixture term, so val loss stays the MODEL's CE.
                          "par_weighted",
                          # span-level NextLat (tul.nextlat_weight, 2026-09-25)
                          "nextlat_weighted",
                          # LX hard credit (tul.code_enum_credit="hard", 2026-09-26): the
                          # hard objective's excess over the mixture NLL. Train-only (the
                          # eval forward is the mixture), listed so a train-mode val
                          # forward still reports the MODEL's CE.
                          "enum_hard_weighted",
                          # LX efficient-exploration knobs (morph/model/tul_explore.py,
                          # 2026-09-29): the set loss (knob 4) and the score-head KL /
                          # learned merge (knobs 3 / 6), so val loss stays the MODEL's CE.
                          "explore_set_weighted", "explore_extra_weighted",
                          # arm B, the code policy (tul.code_policy_k, 2026-09-29): the
                          # REINFORCE + entropy + baseline objective. Folded at train
                          # only; listed so a train-mode val forward still reports the
                          # MODEL's CE.
                          "code_policy_weighted",
                          # tul.fan_all_wta_grader="head" (2026-09-29): the head-graded
                          # relaxed-WTA term is built at eval too (the coda grader's is
                          # train-only), so val loss must subtract it to stay the CE.
                          "fan_head_wta_weighted",
                          # tul.fan_opf (arm F, 2026-09-30): the OPF objective and
                          # tul.fan_route="latent" (arm T): the teacher's router CE + g's
                          # MSE. Both are built at eval too (labels present).
                          "opf_weighted", "rlat_weighted",
                          # tul.fan_loop_select (the latent-selected loop, 2026-09-30):
                          # the exit latent loss + floor + router CE, built at eval too.
                          "fan_lsel_weighted",
                          # tul.latent_pre_target (stage 1, 2026-10-02): the latent term.
                          # A stage-1 forward never reaches this line (no CE; see the
                          # branch above); listed for the both-tuples rule.
                          "latent_pre_weighted",
                          # tul.recon_weight (CE queue item 10, 2026-10-05): own-span
                          # reconstruction. TRAIN ONLY — a val forward (model.eval())
                          # never builds this term, so it is never actually present here;
                          # listed for the both-tuples rule, the same as latent_pre above.
                          "recon_weighted"):
                if out.get(_aux2) is not None:
                    _l -= float(out[_aux2])   # 2026-09-12 energy / bounded-residual arms
            # FM1: val loss is the MODEL's CE, so the ppl divergence guard fires on the
            # language model and not on an auxiliary (the spectral-penalty precedent).
            for _aux in ("fm_weighted", "fm_sigreg_weighted"):
                if out.get(_aux) is not None:
                    _l -= float(out[_aux])
            losses.append(_l)
            ce_tok = float(out["ce_tokens"])
            acc.setdefault("val/ce_tokens", []).append(ce_tok)
            if "ce_first_tok" in out:
                acc.setdefault("val/first_tok_ce", []).append(float(out["ce_first_tok"]))
                acc.setdefault("val/first_tok_counterfactual", []).append(
                    float(out["first_tok_counterfactual"]))
            if "mux_local" in out:
                acc.setdefault("val/mux_local", []).append(float(out["mux_local"]))
            if "sigreg" in out:
                acc.setdefault("val/sigreg", []).append(float(out["sigreg"]))
            for _mk in ("mux_local", "mux_kl", "mux_entropy", "mux_null", "mux_rel",
                        "mux_n_supervised", "spandec", "spandec_ce", "spandec_n_tokens",
                        # tul.recon_weight (CE queue item 10): TRAIN ONLY, so this never
                        # actually fires in a val forward (model.eval()) — listed for the
                        # both-tuples rule beside spandec above.
                        "recon", "recon_ce", "recon_n_tokens",
                        # tul.row_contrast_lambda: `row_contrast_acc` is the distinctness
                        # reading (1/n_valid is chance) and `row_contrast` its term
                        # (log n_valid at chance). Both belong beside val/slot_pairwise_cos.
                        "row_contrast", "row_contrast_acc", "row_contrast_n_rows",
                        "row_contrast_n_anchors",
                        # WHICH span `spandec_ce` grades (tul.spandec_target_offset). 1 on
                        # every arm before that key; at k > 1 the column is a CE over span
                        # s+k and is NOT comparable with an offset-1 arm's.
                        "spandec_target_offset",
                        # tul.code_target: the regression and the cosine of the loop's
                        # projected exit state to the frozen code, on the eval forward.
                        "code_target", "code_target_mse", "code_target_cos",
                        # the generic floor and the InfoNCE top-1 rate (code-only arms)
                        "code_target_cos_shuf", "code_target_acc",
                        # tul.code_enc_var_lambda (LCTUL-J Stage 1): the online variance
                        # floor's raw value, its mean sigma and active fraction; the
                        # TARGET-side collapse reading (code_tgt_std) runs whenever
                        # code_target is on, lambda 0 or not.
                        "code_enc_var", "code_enc_std", "code_enc_active", "code_tgt_std",
                        # tul.coda_logit_l2 (spectral decoupling, 2026-09-23): the raw
                        # mean ||z||^2 over labelled token positions. The weighted term
                        # is train-side only (it is subtracted back out of val/loss).
                        "coda_logit_sq",
                        # tul.loop_denoise: no per-pass TERM is built at eval (the
                        # rollout has no teacher), so only the exit readings above
                        # appear there; `loop_denoise*` is train-side only.
                        # LCM's contrastive accuracy on the SAMPLED code (tul.code): does
                        # the sample retrieve its own span's true code out of the batch,
                        # the two temporal neighbours excluded? Read it against
                        # `val/code_ca_chance`, which is 1/n_candidates averaged over
                        # slots and NOT 1/n — an interior slot has two fewer candidates
                        # than an end slot. A code at chance is a conditional mean,
                        # whatever `code_fm_rel` says (LCM Table 3: Base-LCM won l2 on
                        # all four corpora and lost CA on all four).
                        "code_ca", "code_ca_chance"):
                # (TUL-Code's flow statistics are TRAIN-side only: the eval forward runs
                # the sampler, not the flow term. Read train/code_fm_rel and tul/code_fm_*.)
                if _mk in out:
                    acc.setdefault(f"val/{_mk}", []).append(float(out[_mk]))
            # tul.spandec_parallel (LXTUL-E): `par` (the term), `par_ce` (the mixture CE
            # per span token), `par_n_tokens`, `par_k` and at K > 1 the code-usage stats.
            # A variable number of keys (one `par_code_win_{k}` per code), so a scan.
            for _pk in [k for k in out if str(k) == "par" or str(k).startswith("par_")]:
                if torch.is_tensor(out[_pk]):
                    acc.setdefault(f"val/{_pk}", []).append(float(out[_pk]))
            # tul.code_enum_k (LXTUL-E Stage 1): the coda's per-span mixture over the K
            # code rollouts `enum_ce_mix`, each code alone `enum_ce_code{k}`, the width
            # gains, the credit entropy and wins, and the exit separation. A scan.
            for _ek in [k for k in out if str(k).startswith("enum_")]:
                if torch.is_tensor(out[_ek]):
                    acc.setdefault(f"val/{_ek}", []).append(float(out[_ek]))
            # span-level NextLat (tul.nextlat_weight): the term, the copy baseline it must
            # beat and, with the parallel head, the draft reading on HELD-OUT rows
            # (`nextlat_draft_gap`: the head's CE of span s+2 from the drafted state minus
            # from the loop's own). A scan, like `par_*`.
            for _nk in [k for k in out if str(k).startswith("nextlat")]:
                if torch.is_tensor(out[_nk]):
                    acc.setdefault(f"val/{_nk}", []).append(float(out[_nk]))
            # LX efficient-exploration knobs (tul_explore.py): `explore_*` (the set
            # loss and its spread) and `hyp_*` (the score head's KL, its read gaps against
            # the exact mixture, the merge probe). A scan, like `par_*`.
            for _xk in [k for k in out if str(k).startswith(("explore_", "hyp_"))]:
                if torch.is_tensor(out[_xk]):
                    acc.setdefault(f"val/{_xk}", []).append(float(out[_xk]))
            # arm B, the code policy (tul.code_policy_k): the policy's readings on held-out
            # rows (`code_policy_entropy`, `_share_k{i}`, `_reward_mean`, `_value_mse`,
            # `_adv_mean`, `_agree`). A scan, like `nextlat*`.
            for _ck in [k for k in out if str(k).startswith("code_policy")]:
                if torch.is_tensor(out[_ck]):
                    acc.setdefault(f"val/{_ck}", []).append(float(out[_ck]))
            if str(getattr(_tul_cfg, "fan_loop_select", "off")) != "off":
                # The latent-selected loop's VAL-ONLY EXTRA PASS: the same rows with the
                # loop following the latent TEACHER's pick (as training does) instead of
                # the router's. Eval depth and dropout are deterministic, so the two
                # passes differ in the picks alone. `fan/lsel_teacher_pick_gap` = token CE
                # (router path, the shipped `val/ce_tokens`) - token CE (teacher path):
                # what a perfect router would buy. Same autocast as the main val row.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ce_teach = float(_m.tul_forward_ablated(
                        x, y, layout, lsel_follow="teacher")["ce_tokens"])
                acc.setdefault("fan/lsel_teacher_pick_ce", []).append(_ce_teach)
                acc.setdefault("fan/lsel_teacher_pick_gap", []).append(ce_tok - _ce_teach)
            if getattr(_m, "tul_code_policy", None) is not None:
                # THE DECISIVE READING, a VAL-ONLY EXTRA PASS (never at train): the same
                # rows with a uniformly RANDOM code per slot (the policy's private
                # generator; eval depth and eval dropout are deterministic, so the two
                # passes differ in the codes alone). `val/code_policy_vs_random` = token CE
                # (random codes) - token CE (the policy's ARGMAX codes), nats per token.
                # Positive: the choice matters. About 0: the policy picks noise. Under the
                # SAME autocast as the main val row, so precision is not a difference.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ce_rand = float(_m.tul_forward_ablated(
                        x, y, layout, code_policy_pick="random")["ce_tokens"])
                if str(getattr(_tul_cfg, "code_policy_eval", "argmax")) == "argmax":
                    _ce_arg = ce_tok
                else:
                    # `code_policy_eval: sample` — the main val row above read SAMPLED
                    # codes, so the argmax needs its own pass, and the gap between the two
                    # is the train/deploy reading that mode exists for.
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ce_arg = float(_m.tul_forward_ablated(
                            x, y, layout, code_policy_pick="argmax")["ce_tokens"])
                    acc.setdefault("val/code_policy_sample_minus_argmax", []).append(
                        ce_tok - _ce_arg)
                acc.setdefault("val/code_policy_vs_random", []).append(_ce_rand - _ce_arg)
            # ── LXTUL's fan (tul.fan_k), accumulated over the val batches ─────────
            # The ORACLE family is eval-only and has no train-side twin, so it keeps the
            # `fan/` namespace the arm is read in. The two readings the TRAIN step also
            # emits (`stream_cos_t{t}`, `mix_entropy`) go to `val/fan_*` instead, so the
            # two writers never contend for one wandb series at the same step.
            # `head_wta*` (tul.fan_all_wta_grader="head") is built at train AND eval, so it
            # goes to `val/fan_head_wta_*`; the head-vs-coda readings (`head_coda_agree`,
            # `head_pick_regret`, `rand_pick_regret`) are eval-only and stay in `fan/`.
            # The fan's OPF / router arms (2026-09-30): `opf_*`, `route_*`, `rlat_*` and
            # `teacher_router_agree` are built at train AND eval, so they go to `val/fan_*`;
            # the router-vs-coda readings (`router_coda_agree`, `router_pick_regret`,
            # `rand_pick_regret`, `teacher_coda_agree`) are eval-only and stay in `fan/`.
            # The latent-selected loop's `lsel_*` readings (and its `lsel_weighted`) are
            # built at train AND eval, so they go to `val/fan_lsel_*` the same way.
            _fan_train_side = ("mix_entropy", "mix_w_max", "repel_terms",
                               "teacher_router_agree")
            for _fk in [k for k in out if str(k).startswith("fan_")]:
                if out[_fk] is None:
                    continue
                _name = str(_fk)[4:]
                _dest = (f"val/fan_{_name}"
                         if _name.startswith(("stream_cos_t", "epi_t", "vol_t", "head_wta",
                                              "opf_", "route_", "rlat_", "lsel_"))
                         or _name in _fan_train_side
                         else f"fan/{_name}")
                acc.setdefault(_dest, []).append(float(out[_fk]))
            if "ce_tokens_no_slots" in out:
                # §7.2: CE without the plan MINUS CE with it. Positive ⇒ the coda is
                # actually using the slot state (the C2 number, the h_z ablation).
                acc.setdefault("val/plan_nats", []).append(
                    float(out["ce_tokens_no_slots"]) - ce_tok)
            acc.setdefault("val/layer_passes_per_token", []).append(
                float(out["layer_passes"]) / max(float(out["n_tokens"]), 1.0))
            for _k, _v in out.items():
                if _k.startswith("gate/") and _v is not None:
                    acc.setdefault(f"val/{_k}", []).append(float(_v.detach().mean()))
            if halt:
                # Arm TUL-halt (docs/tul-gate-spec.md §7/§11), on the SAME batch and the
                # SAME weights as the row above: §4 teacher-forces the depth in training,
                # so the two arms differ only in the depth policy at scoring time and the
                # comparison is exactly paired — no second run, no seed noise.
                out_h = _m.tul_forward_halt(x, y, layout)
                _ceh = float(out_h["ce_tokens"])
                acc.setdefault("val/halt_loss", []).append(out_h["loss"].item())
                acc.setdefault("val/halt_ce_tokens", []).append(_ceh)
                acc.setdefault("val/halt_depth_mean", []).append(
                    float(out_h["gate/depth_mean"]))
                acc.setdefault("val/halt_layer_passes_per_token", []).append(
                    float(out_h["layer_passes"]) / max(float(out_h["n_tokens"]), 1.0))
            _has_fm = getattr(_m, "fm_planner", None) is not None
            _ablate = bool(getattr(_tul_cfg, "eval_ablations", False))
            if _has_fm or _ablate:
                # Plan WORTH is the ce_tokens COST of removing the plan (zero) or of
                # destroying only its correspondence to the slot (shuffle). Report the
                # COST, never a specificity fraction: the fraction's denominator
                # collapses through zero (docs/tul-fm-probing.md §4 rule 1, the tg3b
                # -55.4 % reading).
                _oz = _m.tul_forward_ablated(x, y, layout, plan_mode="zero")
                _os = _m.tul_forward_ablated(x, y, layout, plan_mode="shuffle")
                acc.setdefault("val/plan_worth_zero", []).append(
                    float(_oz["ce_tokens"]) - ce_tok)
                acc.setdefault("val/plan_worth_shuffle", []).append(
                    float(_os["ce_tokens"]) - ce_tok)
            _is_code = getattr(_m, "tul_code_enc", None) is not None
            if _ablate and _is_code:
                # TUL-Code (docs/tul-code-spec.md §8): `val/loss` above is the CE with
                # SAMPLED codes at code_infer_steps (the honest number). `val/ce_tf` reads
                # the encoder's own codes in the cells — the CEILING, and NOT an LM number
                # (the cells hold the span being scored) — and `val/code_gap` is how much
                # of a span's code the sampler could not guess. `wrong_seed` is refused
                # on a code model (the seed feeds the thinker, not the cells).
                _ot = _m.tul_forward_ablated(x, y, layout, code_mode="encoder")
                acc.setdefault("val/ce_tf", []).append(float(_ot["ce_tokens"]))
                acc.setdefault("val/code_gap", []).append(ce_tok - float(_ot["ce_tokens"]))
                # The K-sample marginal (the likelihood a latent-variable LM is owed):
                # `val/ce_marginal` is a LOWER bound on the span log-likelihood averaged
                # over K sampled codes, per token; `val/ce_single_mean` the K-average of
                # one-sample CE (Jensen: marginal <= single_mean, = at K=1).
                # A code-TARGET model (tul.code_target) has E and no sampler: `val/ce_tf`
                # above is its ceiling, and there is no K-sample marginal to take.
                _K = (int(getattr(_tul_cfg, "code_marginal_k", 0))
                      if bool(getattr(_tul_cfg, "code", False)) else 0)
                if _K > 0:
                    for _k, _v in code_marginal_ce(_m, x, y, layout, _K,
                                                   int(_tul_cfg.code_infer_steps)).items():
                        acc.setdefault(f"val/{_k}", []).append(_v)
            elif _ablate:
                # THE WRONG-PLAN PROBE (arm GL1). A valid-but-wrong slot value instead
                # of no value. TG4b: 0.48-0.56 nats here against 0.10 for zeroing —
                # "removing LESS hurts MORE", which is how we know the coda reads the
                # slot's VALUE even when shuffling costs nothing. It is an OOD-shock
                # number, not a worth number (see tul_forward_ablated's docstring).
                _ow = _m.tul_forward_ablated(x, y, layout, plan_mode="wrong_seed")
                acc.setdefault("val/plan_worth_wrong_seed", []).append(
                    float(_ow["ce_tokens"]) - ce_tok)
            if _ablate:
                for _k, _v in _m.tul_slot_state_probe(x, layout).items():
                    acc.setdefault(f"val/{_k}", []).append(float(_v))
                # MUX §8.3 reasoning attention lift, WINDOW branch only (see
                # morph/model/attn_lift.py for exactly which branch and why). Eager only.
                for _k, _v in _m.tul_attn_lift_probe(x, layout).items():
                    if _v == _v:                       # skip NaN (no eligible query)
                        acc.setdefault(f"val/{_k}", []).append(float(_v))
                if float(getattr(_tul_cfg, "mux_beta", 0.0)) > 0.0:
                    # The FM2 scar, observable: how much of the tied embedding table's
                    # gradient the auxiliary owns.
                    for _k, _v in _m.tul_mux_grad_share(x, y, layout).items():
                        acc.setdefault(f"val/{_k}", []).append(float(_v))
            if _has_fm:
                if "fm" in out:
                    acc.setdefault("val/fm", []).append(float(out["fm"]))
                    acc.setdefault("val/fm_rel", []).append(float(out["fm_rel"]))
                if "fm_sigreg" in out:
                    acc.setdefault("val/fm_sigreg", []).append(float(out["fm_sigreg"]))
                for _k, _v in _m.fm_eval_probe(x, layout).items():
                    acc.setdefault(f"val/{_k}", []).append(float(_v))
            if layout.stats:
                for k, v in layout.stats.items():
                    acc.setdefault(f"val/span_{k}", []).append(float(v))
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, labels=y)
            _l = out["loss"].item()
            if out.get("mtp_weighted") is not None:
                # val loss = the next-token CE (the number every arm is compared on);
                # the lookahead heads are reported on their own (arc E8).
                _l -= float(out["mtp_weighted"])
                for _k, _v in out.items():
                    if _k.startswith("ce_mtp_"):
                        acc.setdefault(f"val/{_k}", []).append(float(_v))
            losses.append(_l)
    model.train()
    if extra is not None:
        extra.update({k: sum(v) / len(v) for k, v in acc.items() if v})
        # TUL-Code collapse guard (`tul.code_rank_abort`, the jepa arm): a code whose
        # effective rank fell under the bar is a constant, and the run is over.
        _abort = float(getattr(_tul_cfg, "code_rank_abort", 0.0)) if _tul_cfg is not None else 0.0
        if _abort > 0.0 and extra.get("val/code_eff_rank", float("inf")) < _abort:
            raise RuntimeError(
                f"[code] ABORT: val/code_eff_rank {extra['val/code_eff_rank']:.2f} < "
                f"tul.code_rank_abort {_abort}: the code collapsed.")
        # PPL is exp of the MEAN CE, never the mean of the per-batch exp(CE). Jensen
        # makes the latter strictly larger: on tul-a1-acap1 it read 25.89 against the
        # true 25.14, and that 0.75 gap is 59 % of the 1.27 PPL A1-vs-A0 effect it
        # exists to measure. The baseline path returns exp(mean) (see the return
        # below), so the per-batch form was NOT comparable to it, despite this
        # function's docstring claiming exactly that comparability.
        for _ce_k, _ppl_k in (("val/ce_tokens", "val/ppl_tokens"),
                              ("val/halt_ce_tokens", "val/halt_ppl_tokens")):
            if _ce_k in extra:
                extra[_ppl_k] = math.exp(min(extra[_ce_k], 20.0))
    avg = sum(losses) / max(len(losses), 1)
    return avg, math.exp(min(avg, 20.0))


# ── Generation test ────────────────────────────────────────────────────────────

@torch.compiler.set_stance("force_eager")
def run_generation_test(
    model: nn.Module,
    device: torch.device,
    tokenizer_name: str,
    seq_len: int,
    step: int,
    n_tokens: int = 100,
    tul_rt=None,
) -> tuple[str, dict]:
    """Run a short SAMPLED generation and return ``(text, metrics)``.

    Not greedy, despite what this line said until 2026-08-23: both paths below decode at
    ``temperature 0.8, top_k 50``. That is one decode mode, and a single mode cannot see
    degeneration on its own — a repetition loop is a GREEDY failure and truncated
    sampling hides it. For the multi-mode table (greedy / top-k / pure ancestral) run
    ``scripts/tul_samples.py`` over the checkpoints; it is deliberately a separate script
    so a campaign in flight is never edited underneath its own arms.

    The metrics are empty on the plain path and, on the TUL path, are the
    docs/tul-gate-spec.md §10 generation numbers averaged over the prompts: rep4 /
    distinct-3 (a repetition loop scores an EXCELLENT perplexity — 1.46 against real
    text's 32.44 — so fluency is meaningless without diversity beside it), the realised
    span geometry, and how often a span actually ended on a boundary rather than
    spending the gate's whole budget. §5's teacher-forcing leak is invisible in val CE
    by construction; this is where it shows.

    ``tul_rt`` set ⇒ sample through ``morph.inference.tul_generate`` so the slot layout
    is built by the SAME boundary rule the loader used (spec §6, invariant 1). That path
    is an eager recompute-per-step sampler with no KV cache (v1, by spec), so it is
    slower than the plain sampler below — the generation test is 3 × n_tokens tokens.

    Decorated with ``force_eager``: generation runs the model token-by-token at
    batch=1 with a sequence length that grows by one every step, so the MLPs
    (``torch.compile``d for the *training* shapes B×S) see a brand-new shape on
    every token. Under the training stance (``eager_on_recompile``) dynamo still
    pays per-token guard-eval to route each novel shape to its eager fallback —
    measured >10× slower and it tripped the 90s watchdog mid-gen. ``force_eager``
    makes the compiled MLPs run their original eager code directly (~42 ms/tok,
    stable as seqlen grows; verified ignore/gen_isolated.py), and the decorator
    restores the prior stance on return. Eval (full-batch fixed shape) is
    unaffected and stays on the compiled path.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "[transformers not installed]"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompts = [
        "The theory of relativity states that",
        "Once upon a time in a distant land, there lived a",
        "In machine learning, the key insight is that",
    ]
    output_lines: list[str] = []
    model.eval()

    if tul_rt is not None:
        from morph.inference.gen_metrics import generation_metrics
        from morph.inference.tul_generate import generate_tul
        _m = getattr(model, "_orig_mod", model)
        spec = tul_rt.data_cfg.spec_for(seq_len)
        rule = tul_rt.data_cfg.rule
        per: list[dict] = []
        for prompt in prompts:
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            new, builder = generate_tul(_m, ids, rule, spec,
                                        max_new_tokens=n_tokens, temperature=0.8,
                                        top_k=50, seed=step, device=device,
                                        emit_source=("token"
                                                     if tul_rt.model_cfg.emit_weight == 0.0
                                                     else "slot"))
            text = tokenizer.decode(ids + new, skip_special_tokens=True)
            mt = generation_metrics(new, builder, rule)
            per.append(mt)
            output_lines.append(
                f"PROMPT: {prompt}\nOUTPUT: {text}\n"
                f"[slots={builder.n_slots} mean_span={mt['mean_span']:.1f} "
                f"on_boundary={mt['boundary_frac']:.2f} rep4={mt['rep4']:.3f} "
                f"distinct3={mt['distinct3']:.3f}]")
        model.train()
        agg = {f"gen/{k}": float(sum(d[k] for d in per) / len(per)) for k in per[0]}
        return "\n---\n".join(output_lines), agg

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)
        gen = ids.clone()
        with torch.no_grad():
            for _ in range(n_tokens):
                if gen.shape[1] >= seq_len:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(gen)
                logits = out["logits"][:, -1, :] / 0.8
                topk_v, _ = logits.topk(50, dim=-1)
                logits[logits < topk_v[:, -1:]] = float("-inf")
                next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
                gen = torch.cat([gen, next_tok], dim=1)
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        output_lines.append(f"PROMPT: {prompt}\nOUTPUT: {text}")

    model.train()
    return "\n---\n".join(output_lines), {}


# ── Config → MORPHConfig ───────────────────────────────────────────────────────

def _span_rule(cfg: DictConfig):
    """The BoundaryRule for ``model.span_mask: "span"`` — the TUL rule, not a new one."""
    from morph.training.tul_setup import build_boundary_rule

    rule, _lut, _eos, _subs = build_boundary_rule(cfg)
    print(f"  SPAN MASK: cut by the TUL BoundaryRule (min_span={rule.min_span}, "
          f"span_cap={rule.span_cap}, eos_id={rule.eos_id}, "
          f"|B|={int(rule.is_boundary.sum())} of {rule.is_boundary.shape[0]} ids)",
          flush=True)
    return rule


def build_morph_config(cfg: DictConfig, tul=None, fm=None) -> MORPHConfig:
    """``tul`` (a TULConfig or None) gates CONSTRUCTION of the TUL parameters;
    None ⇒ byte-identical to the baseline model (runtime-invariants §6b).
    ``fm`` (an FMArmConfig or None) does the same for the FM1 planner."""
    m = cfg.model
    tr = cfg.training

    d_ff_raw = int(getattr(m, "d_ff", 0))

    ch = m.get("channel_dims", [384, 256, 128])
    channel_dims = tuple(int(c) for c in ch)

    return MORPHConfig(
        tul=tul,
        fm=fm,
        d_model=int(m.d_model),
        n_heads=int(m.n_heads),
        d_ff=d_ff_raw,
        vocab_size=int(m.vocab_size),
        max_seq_len=int(m.max_seq_len),
        n_prelude=int(m.n_prelude),
        n_core=int(m.n_core),
        n_coda=int(m.n_coda),
        mean_depth=int(m.mean_depth),
        max_depth=int(m.max_depth),
        bptt_depth=int(m.bptt_depth),
        depth_fixed=bool(getattr(m, "depth_fixed", False)),
        ckpt_grad_iters=int(getattr(m, "ckpt_grad_iters", -1)),
        core_init_scale=float(getattr(m, "core_init_scale", 0.0)),
        # SCSE, the full method (docs/scse-spec.md). Every field goes through the
        # config so a run is reproducible from its wandb config alone.
        scse_enabled=bool(getattr(m, "scse_enabled", False)),
        scse_step_scale=float(getattr(m, "scse_step_scale", 0.5)),
        scse_anchor_scale=float(getattr(m, "scse_anchor_scale", 0.1)),
        scse_init_scale=float(getattr(m, "scse_init_scale", 0.1)),
        scse_eps=float(getattr(m, "scse_eps", 1.0e-8)),
        scse_kappa=float(getattr(m, "scse_kappa", 0.0)),
        scse_input_mode=str(getattr(m, "scse_input_mode", "deviation")),
        scse_delta_clip=float(getattr(m, "scse_delta_clip", 0.0)),
        channel_dims=channel_dims,
        compression=int(m.compression),
        n_kv_heads=int(m.n_kv_heads),
        csa_compress_ratio=int(m.csa_compress_ratio),
        hca_compress_ratio=int(m.hca_compress_ratio),
        core_hca_compress_ratio=(None if m.get("core_hca_compress_ratio", None) is None
                                 else int(m.core_hca_compress_ratio)),
        core_impl=str(m.get("core_impl", "morph")),
        parcae_core_d_head=(None if m.get("parcae_core_d_head", None) is None
                            else int(m.parcae_core_d_head)),
        # The cross-span budget pair (.agents/notes/proposed/architecture/
        # 2026-09-11-cross-span-budget.md). "span" resolves the ONE TUL boundary rule
        # from the tokenizer HERE, through the same `build_boundary_rule` every TUL arm
        # uses — a second rule would make the budget incomparable to every slot number
        # already measured. "row" (the control) gets no rule by contract.
        span_mask=str(m.get("span_mask", "off")),
        span_rule=(_span_rule(cfg) if str(m.get("span_mask", "off")) == "span"
                   else None),
        # The reach dial on the "span" relation (`MORPHConfig.span_reach`'s own
        # comment). 0 is the "span" arm's own same-span relation, bit-identical to
        # leaving the key out of the YAML.
        span_reach=int(m.get("span_reach", 0)),
        # Which block gets the reach relation (`MORPHConfig.span_reach_layer`'s own
        # comment). -1 is every block (today's default), bit-identical to leaving the
        # key out of the YAML.
        span_reach_layer=int(m.get("span_reach_layer", -1)),
        top_k=int(m.top_k),
        window_size=int(m.window_size),
        context_len=int(m.context_len),
        lorentz_fraction=float(m.lorentz_fraction),
        bigram_hash_vocab=int(m.bigram_hash_vocab),
        n_ve=int(m.n_ve) if getattr(m, "n_ve", None) is not None else None,
        ce_chunk_size=int(getattr(m, "ce_chunk_size", 1024)),
        ce_softmax_kernel=bool(getattr(m, "ce_softmax_kernel", False)),
        ce_compact_rows=bool(getattr(m, "ce_compact_rows", False)),
        use_kernels=bool(getattr(m, "use_kernels", True)),
        tg_scoped_kernels=bool(getattr(m, "tg_scoped_kernels", False)),
        hc_streams=int(getattr(m, "hc_streams", 4)),
        hc_tau=float(getattr(m, "hc_tau", 1.0)),
        hc_cayley_iters=int(getattr(m, "hc_cayley_iters", 3)),
        hc_cayley_alpha=float(getattr(m, "hc_cayley_alpha", 0.1)),
        hc_init_gain=float(getattr(m, "hc_init_gain", 0.1)),
        hc_use_kernel=bool(getattr(m, "hc_use_kernel", True)),
        l2_persist=bool(getattr(m, "l2_persist", False)),
        retention=bool(getattr(m, "retention", True)),
        retention_layers=tuple(int(x) for x in getattr(m, "retention_layers", (1,))),
        retention_sections=tuple(
            str(x) for x in getattr(m, "retention_sections", ("prelude", "core", "coda"))),
        retention_write_shift=bool(getattr(m, "retention_write_shift", False)),
        retention_heads=int(getattr(m, "retention_heads", 0)),
        retention_chunk=int(getattr(m, "retention_chunk", 128)),
        retention_gate_init=float(getattr(m, "retention_gate_init", -6.0)),
        # str|bool pass-through; MORPHConfig.retention_carry_mode normalizes+validates.
        # Default "none" = causal (post-2026-08-31 fix).
        retention_carry=getattr(m, "retention_carry", "none"),
        retention_gate_bias=float(getattr(m, "retention_gate_bias", 2.0)),
        core_gain_clip=float(getattr(m, "core_gain_clip", 0.0)),
        core_gain_clip_iter_lo=int(getattr(m, "core_gain_clip_iter_lo", 0)),
        core_gain_clip_iter_hi=int(getattr(m, "core_gain_clip_iter_hi", -1)),
        slot_cot_clip=float(getattr(m, "slot_cot_clip", 0.0)),
        slot_state_renorm=bool(getattr(m, "slot_state_renorm", False)),
        slot_gain_lambda=float(getattr(m, "slot_gain_lambda", 0.0)),
        slot_gain_target=float(getattr(m, "slot_gain_target", 0.9)),
        slot_gain_eps=float(getattr(m, "slot_gain_eps", 0.02)),
        slot_gain_all_iters=bool(getattr(m, "slot_gain_all_iters", False)),
        slot_gain_tail_lambda=float(getattr(m, "slot_gain_tail_lambda", 0.0)),
        slot_gain_tail_target=float(getattr(m, "slot_gain_tail_target", 1.1)),
        slot_gain_floor_lambda=float(getattr(m, "slot_gain_floor_lambda", 0.0)),
        slot_gain_floor_target=float(getattr(m, "slot_gain_floor_target", 0.95)),
        slot_gain_renorm=bool(getattr(m, "slot_gain_renorm", False)),
        mtp_heads=int(getattr(m, "mtp_heads", 1)),
        injection_channels=str(getattr(m, "injection_channels", "ctx")),
        core_fixed_point_lambda=float(getattr(m, "core_fixed_point_lambda", 0.0)),
        core_depth_state=bool(getattr(m, "core_depth_state", False)),
        core_gain_lambda=float(getattr(m, "core_gain_lambda", 0.0)),
        core_gain_target=float(getattr(m, "core_gain_target", 20.0)),
        core_gain_eps=float(getattr(m, "core_gain_eps", 0.02)),
        core_gain_direction=str(getattr(m, "core_gain_direction", "power")),
        core_gain_power_iters=int(getattr(m, "core_gain_power_iters", 0)),
        injection_all_decay=float(getattr(m, "injection_all_decay", 0.9)),
        injection_all_dt=(None if getattr(m, "injection_all_dt", None) is None
                          else float(m.injection_all_dt)),
        injection_B=bool(getattr(m, "injection_B", False)),
        core_state_init=str(getattr(m, "core_state_init", "prelude")),
        core_state_init_std=float(getattr(m, "core_state_init_std", 0.02)),
        mtp_weight=float(getattr(m, "mtp_weight", 1.0)),
        # LoopMTP (arXiv 2608.03624). Every field goes through the config so a run is
        # reproducible from its wandb config alone.
        core_readout=str(getattr(m, "core_readout", "last")),
        loopmtp_weight=float(getattr(m, "loopmtp_weight", 0.0)),
        loopmtp_free_first=bool(getattr(m, "loopmtp_free_first", True)),
        loopmtp_proj=str(getattr(m, "loopmtp_proj", "linear")),
        loopmtp_ponder_weight=float(getattr(m, "loopmtp_ponder_weight", 0.0)),
        loopmtp_gate_eps=float(getattr(m, "loopmtp_gate_eps", 1.0e-6)),
        dropout=float(tr.dropout),
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optimizer,
    scaler: torch.amp.GradScaler,
    pruning: Optional[PruningSchedule],
    *,
    next_step: Optional[int] = None,
) -> None:
    """Save a full training checkpoint.

    `step` is the label for the checkpoint filename/logs. `next_step` is the loop
    index to execute on resume. They differ for ordinary post-step checkpoints:
    after completing loop step N, resume must start at N+1 and fast-forward N+1
    batches. Pre-step transition checkpoints pass next_step=N.
    """
    resume_step = int(step if next_step is None else next_step)
    ckpt = {
        "step": int(step),
        "next_step": resume_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        # RNG state so a resume continues the SAME stochastic stream (per-sequence Poisson
        # depth draws, dropout, etc.) — "like nothing happened". CPU + all CUDA devices.
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    _mm = getattr(model, "_orig_mod", model)
    _ref = _mm.tul_code_ref_state() if hasattr(_mm, "tul_code_ref_state") else None
    if _ref is not None:
        # tul.code_target_ref (spec §17.1): the frozen VAE-stage twin every target and
        # every grade is measured against. It rides its OWN key, not `model`, because it
        # is deliberately not a registered submodule (see MORPHTransformer.__init__), and
        # a resume must keep THIS reference: re-snapshotting the live weights mid-run
        # would move the target, which is the failure the mechanism exists to prevent.
        ckpt["code_ref"] = _ref
    _fan_tgt = _mm.tul_fan_target_state() if hasattr(_mm, "tul_fan_target_state") else None
    if _fan_tgt is not None:
        # tul.fan_opf / tul.fan_route="latent" (2026-09-30): the EMA target twin of the
        # prelude. Its OWN key for the `code_ref` reason (not a registered submodule); a
        # resume loads it back (`tul_fan_target_sync`) so the EMA state continues instead
        # of jumping to the live weights. The router's balance bias is a model buffer and
        # rides in `model`.
        ckpt["fan_target"] = _fan_tgt
    if getattr(_mm, "tul", None) is not None:
        # Audit trail, not a reconstruction flag: the TUL parameters are built from the
        # config at model-build time (never mid-run), so load_state_dict already has a
        # home for them. This records WHICH layout a checkpoint was trained under so a
        # later loader cannot silently pair TUL weights with a plain-MORPH config.
        ckpt["tul"] = {"prefix_k": _mm.cfg.tul.prefix_k, "slot_id": _mm.cfg.tul.slot_id,
                       "coda_sees_slots": _mm.cfg.tul.coda_sees_slots,
                       "tokens_through_core": _mm.cfg.tul.tokens_through_core}
    if pruning is not None:
        # Both topology-phase flags are needed to RECONSTRUCT module structure (carve +
        # routers) before load_state_dict on resume. _is_compact alone is insufficient:
        # a routed checkpoint also needs its routers re-attached or their params are
        # silently dropped (strict=False) and routing comes back OFF.
        ckpt["pruning_compact"] = pruning.is_compact
        ckpt["pruning_routed"] = pruning.is_routed
    torch.save(ckpt, path)


# TUL tensors a checkpoint may carry that the model being loaded does not build. `tul.W_prefix`
# is the slot loop's prefix projection (and the FM planner's write path); the paid loop
# (tul.tokens_through_core, the shipped forward since 2026-09-03) builds none — its slot is a
# looped position and writes nothing through a projection. Every A2 checkpoint saved before
# 2026-09-03 carries the key. Kept as a tuple so the next retirement is one line and the drop
# stays visible in the log rather than silent.
RETIRED_TUL_KEYS = ("tul.W_prefix",)


def drop_retired_tul_keys(state: dict, model: nn.Module, path: str) -> list[str]:
    """Remove RETIRED_TUL_KEYS from a checkpoint's model state, LOUDLY, for a model that
    does not build them. A model whose TULSlots has a live W_prefix (the slot loop, the FM
    planner) keeps every key, and the strict unexpected-key check downstream still fires
    for anything else. Returns the dropped keys. Mutates ``state`` in place."""
    tul = getattr(model, "tul", None)
    if tul is None or getattr(tul, "W_prefix", None) is not None:
        return []
    dropped = [k for k in state if k.replace("_orig_mod.", "") in RETIRED_TUL_KEYS]
    for k in dropped:
        del state[k]
    if dropped:
        print(f"  load {path}: dropped {len(dropped)} retired TUL tensor(s) {dropped} — the "
              f"forward runs the core over the tokens themselves (tul.tokens_through_core "
              f"or tul.loop_reads_tokens), and neither builds a prefix projection; "
              f"every other tensor is checked as before", flush=True)
    return dropped


# ── code target (tul.code_target): the VAE stage's checkpoint carries the flow thinker ──
#
# `tul-code-vae` (tul_code_vae.yaml) is a `tul.code` model, so its checkpoint holds the
# thinker's tensors (velocity head, time embedding, cell markers, and on a CFG or discrete
# model the null seed / symbol table / quantiser) beside E and the coda. A code-TARGET model
# builds E and a projection and no thinker, so those tensors have no home. They are dropped
# HERE, loudly and by name, for that ONE model shape; every other homeless key still raises.
CODE_THINKER_PREFIXES = ("tul_code_head.", "tul_code_time.", "tul_code_cell", "tul_code_clean",
                         "tul_code_null", "tul_code_sym.", "tul_code_sym_head.", "tul_code_vq.")


def drop_code_thinker_keys(state: dict, model: nn.Module, path: str) -> list[str]:
    """Remove the flow / masked-denoiser thinker's tensors from a checkpoint's model state,
    LOUDLY, for a code-TARGET model (E + projection, no thinker). Any other model keeps every
    key and the strict unexpected-key check downstream fires as before. Returns the dropped
    keys. Mutates ``state`` in place."""
    if getattr(model, "tul_code_proj", None) is None or getattr(model, "tul_code_head", None) is not None:
        return []
    dropped = [k for k in state
               if k.replace("_orig_mod.", "").startswith(CODE_THINKER_PREFIXES)]
    for k in dropped:
        del state[k]
    if dropped:
        print(f"  load {path}: dropped {len(dropped)} code-thinker tensor(s) "
              f"{sorted(dropped)[:6]}{'...' if len(dropped) > 6 else ''} — a code-target "
              f"model builds E and a projection and no thinker (tul.code_target); every "
              f"other tensor is checked as before", flush=True)
    return dropped


# ── plain → TUL bootstrap: the ONE resume that legitimately crosses a structure change ──
#
# `slot-strict-bootstrap` (morph/configs/tul_slot_strict_bootstrap.yaml) seeds a TUL model
# from a checkpoint of a PLAIN run, so two classes of tensor have no partner:
#
#   MISSING    every TUL-owned parameter (`tul.*`, `tul_spandec.*`, `tul_register.*`, …).
#              The plain seed never had them; they keep their fresh init.
#   UNEXPECTED every pooled compressor / top-k indexer tensor of the plain seed. Under
#              `tul.tg_restrict` MORPHAttention builds NEITHER (morph/model/attention.py:
#              "Build nothing rather than build-and-ignore") because the compressed branch
#              attends slot positions directly instead of pooling. The seed's copies are
#              DROPPED — real learned state, thrown away, and unavoidable: no plain
#              checkpoint can carry a tg_restrict attention, since tg_restrict is only
#              reachable with a TUL config.
#
# Both lists are ENUMERATED FROM THE LIVE MODEL, never from a hardcoded name guess, so a
# module added to the tree tomorrow is classified by what it is rather than by what this
# file remembers. Everything outside the two lists still raises, and with the flag off
# (`training.resume_plain_to_tul`, default false) this file behaves exactly as before.


def tul_owned_prefixes(model: nn.Module) -> tuple[str, ...]:
    """state_dict key prefixes owned by a TUL-only module of ``model``.

    The TUL modules are the root children named ``tul`` or ``tul_*`` — `TULSlots` and
    every optional head hung beside it (`tul_spandec`, `tul_register`, `tul_cond`,
    `tul_gate`, `tul_reread`, `tul_grad_pass`, `tul_egrad`, `tul_chain`, …). A plain model
    assigns ``self.tul = None`` before any Module lands there, so ``named_children`` yields
    nothing and this returns ``()`` — which is what makes the classifier refuse everything
    on a plain→plain load.
    """
    root = getattr(model, "_orig_mod", model)
    return tuple(sorted(f"{n}." for n, _ in root.named_children()
                        if n == "tul" or n.startswith("tul_")))


def tg_dropped_attention_prefixes(model: nn.Module) -> tuple[str, ...]:
    """state_dict key prefixes a ``tul.tg_restrict`` model does NOT build.

    Read off the live modules: an attention implementation that declares ``tg_restrict``
    True and holds ``None`` in one of the pooled-branch slots did not build that submodule,
    so a plain seed's tensors under it are homeless. A model WITHOUT tg_restrict returns
    ``()`` and nothing may be dropped.
    """
    root = getattr(model, "_orig_mod", model)
    out: list[str] = []
    for name, mod in root.named_modules():
        if getattr(mod, "tg_restrict", False) is not True:
            continue
        for sub in ("compressor", "comp_norm", "indexer"):
            if getattr(mod, sub, "absent") is None:
                out.append(f"{name}.{sub}." if name else f"{sub}.")
    return tuple(sorted(out))


def classify_plain_to_tul_keys(
    missing: list[str], unexpected: list[str], model: nn.Module,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split load_state_dict's homeless keys into ALLOWED and REFUSED, for the bootstrap.

    Returns ``(ok_missing, bad_missing, ok_unexpected, bad_unexpected)``. A missing key is
    allowed only under a TUL-owned module; an unexpected key only under a pooled branch
    this model's tg_restrict attention did not build. `_prune_mask` buffers keep the
    back-compat exemption they already had. Caller decides what to do with the refusals —
    this function has no opinion and no side effect.
    """
    tul_pre = tul_owned_prefixes(model)
    tg_pre = tg_dropped_attention_prefixes(model)

    def _canon(k: str) -> str:
        return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")

    ok_m, bad_m, ok_u, bad_u = [], [], [], []
    for k in missing:
        c = _canon(k)
        (ok_m if (c.startswith(tul_pre) or c.endswith("_prune_mask")) else bad_m).append(k)
    for k in unexpected:
        (ok_u if _canon(k).startswith(tg_pre) else bad_u).append(k)
    return ok_m, bad_m, ok_u, bad_u


def _report_plain_to_tul(tag: str, path: str, ok_m: list[str], bad_m: list[str],
                         ok_u: list[str], bad_u: list[str]) -> None:
    """Print the two allowed lists in full and RAISE on anything else."""
    if bad_m or bad_u:
        raise RuntimeError(
            f"{tag} {path}: training.resume_plain_to_tul allows ONLY TUL-owned tensors to "
            f"be missing and ONLY the pooled compressor/indexer tensors tg_restrict does "
            f"not build to be unexpected. Refused: {len(bad_m)} missing "
            f"{bad_m[:8]}{'...' if len(bad_m) > 8 else ''}; {len(bad_u)} unexpected "
            f"{bad_u[:8]}{'...' if len(bad_u) > 8 else ''}"
        )
    print(f"  {tag} {path}: resume_plain_to_tul=True — {len(ok_m)} TUL-owned tensors kept "
          f"their FRESH init and {len(ok_u)} pooled-branch tensors of the seed were "
          f"DROPPED (tg_restrict builds no compressor/indexer).", flush=True)
    for label, keys in (("fresh-init (TUL-owned)", ok_m), ("dropped (seed had it, this "
                                                           "model does not)", ok_u)):
        if keys:
            print(f"    {label}: {len(keys)}", flush=True)
            for k in keys:
                print(f"      {k}", flush=True)


def load_checkpoint(
    path: str,
    model: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pruning: Optional[PruningSchedule] = None,
    allow_plain_to_tul: bool = False,
) -> tuple[int, dict, bool]:
    """FULL resume — restore the run exactly ("like nothing happened").

    Restores: model weights + topology (carve/BCSR + ReMoE routers), the pre-carve
    dead-tile prune mask (now a buffer), the saliency EMA, the GradScaler, CPU+CUDA RNG,
    the training step, and the pruning-schedule phase flags. Reconstructs module structure
    in the SAME order the live run mutated it, so every saved tensor finds a home:

        routers (if routed)  →  load_state_dict (auto-rebuilds carve)  →  rng/scaler

    The OPTIMIZER is handled by the caller: a carved/routed checkpoint's optimizer state is
    keyed on a DIFFERENT param set (mortar_data / router params) than the freshly-built
    dense optimizer, so the caller must REBUILD the optimizer on the reconstructed topology
    BEFORE loading state. Returns (next_step, optimizer_state_dict, needs_optimizer_rebuild).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    is_compact = bool(ckpt.get("pruning_compact", False))
    is_routed = bool(ckpt.get("pruning_routed", False))

    # 1. Re-attach ReMoE routers BEFORE load_state_dict. CMSBlockLinear._load_from_state_dict
    #    auto-reconstructs the BCSR carve, but routers are separate submodules it does NOT
    #    rebuild — without this their params have no home and strict=False drops them.
    if is_routed:
        if pruning is None:
            raise RuntimeError(
                "load_checkpoint: checkpoint is ROUTED but no PruningSchedule was passed to "
                "reconstruct the routers — cannot resume faithfully."
            )
        pruning._activate_routing(model)
    if pruning is not None:
        pruning._is_compact = is_compact
        pruning._is_routed = is_routed

    # 2. Load weights. CMSBlockLinear._load_from_state_dict rebuilds carve (BCSR) storage
    #    from the mortar_* keys; the _prune_mask buffer restores the pre-carve dead tiles.
    #    KEY ALIGNMENT: torch.compile wraps the MLPs in-place (layer.mlp = compile(...)), so
    #    BOTH the checkpoint and the live model nest keys under `mlp._orig_mod.…`. The old
    #    code stripped `_orig_mod.` from ONLY the checkpoint → every compiled-MLP tensor
    #    mismatched the (still-`_orig_mod`) model and strict=False silently dropped them (a
    #    near-empty "resume" — latent theater). Fix: align the checkpoint's key CONVENTION to
    #    the model's, but pass ALL keys through INTACT so the carve/router load-hooks fire
    #    (pre-filtering to the dense model's keys would drop mortar_data before it exists).
    ckpt_model = ckpt["model"]
    model_keys = list(model.state_dict().keys())
    model_has_orig = any("_orig_mod" in k for k in model_keys)
    ckpt_has_orig = any("_orig_mod" in k for k in ckpt_model)
    if ckpt_has_orig and not model_has_orig:
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt_model.items()}
    else:
        # Same convention (both compiled, or neither) → as-is. (model-compiled/ckpt-not is
        # not produced by this codebase — compile is applied unconditionally before save.)
        state = dict(ckpt_model)
    # Let load_state_dict report truthfully AFTER the hooks reconstruct mortar_data/routers.
    drop_retired_tul_keys(state, model, path)
    drop_code_thinker_keys(state, model, path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # No-theater: an UNEXPECTED key means a saved tensor found no home (structure drift) →
    # state was silently lost. Fail loud. MISSING keys are tolerated only for back-compat
    # buffers a pre-this-change checkpoint legitimately lacks (e.g. _prune_mask), and warned.
    if allow_plain_to_tul:
        # The plain → TUL bootstrap. BOTH classes are checked here and the rest still
        # raises — including a missing key outside a TUL module, which the branch below
        # only WARNS about. This path is therefore strictly tighter than the default.
        _report_plain_to_tul("load_checkpoint", path,
                             *classify_plain_to_tul_keys(missing, unexpected, model))
    else:
        if unexpected:
            raise RuntimeError(
                f"load_checkpoint: {len(unexpected)} checkpoint tensors had no home in the "
                f"reconstructed model (structure mismatch — state would be silently lost): "
                f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
            )
        _benign_missing = tuple(m for m in missing if not m.endswith("_prune_mask"))
        if _benign_missing:
            print(f"  Warning: {len(_benign_missing)} model tensors absent from checkpoint "
                  f"(kept their init): {_benign_missing[:8]}"
                  f"{'...' if len(_benign_missing) > 8 else ''}")

    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    # 3. RNG — continue the SAME stochastic stream (Poisson depth draws, dropout).
    if ckpt.get("rng_cpu") is not None:
        torch.set_rng_state(ckpt["rng_cpu"].cpu().to(torch.uint8))
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in ckpt["rng_cuda"]])
        except Exception as e:  # device-count mismatch etc. — surface, don't pretend
            print(f"  Warning: could not restore CUDA RNG state ({e}); RNG continues fresh")

    ckpt_step = int(ckpt.get("step", 0))
    if "next_step" in ckpt:
        step = int(ckpt["next_step"])
    else:
        # Legacy periodic checkpoints were written after completing their loop step but
        # only persisted `step`. Treat them as post-step saves so resume executes the next
        # unseen batch. Legacy pre-step transition checkpoints are ambiguous; the warning is
        # intentional because exact replay cannot be inferred from old metadata alone.
        step = ckpt_step + 1
        print(f"  Warning: checkpoint lacks next_step metadata; assuming legacy post-step "
              f"save and resuming at step {step} (saved step label {ckpt_step})")
    needs_rebuild = bool(is_compact or is_routed)
    print(f"  Resumed model+scaler+RNG from checkpoint step {ckpt_step}; "
          f"next_step={step} "
          f"(compact={is_compact} routed={is_routed} → optimizer_rebuild={needs_rebuild})")
    # The checkpoint's MODEL parameter names travel with the optimizer state so the caller
    # can re-index it when the live model has parameters the checkpoint does not (an
    # intervention arm that adds a module). See optimizer.align_optimizer_state.
    return step, ckpt["optimizer"], needs_rebuild, set(ckpt["model"].keys())


def assert_code_target_front_frozen(model) -> list[str]:
    """A ``tul.code_target`` model may not train its FRONT without the frozen reference.

    E is frozen; its INPUT is not. It pools the LIVE prelude's states of the next span, so
    letting the prelude, the embeddings, the input norm or a prelude-range ``x0_inject``
    train moves the target with the model. Measured on the code-only draw (wandb
    38naddpq, killed at step 3500): train own cosine 0.61 / shuffled 0.56 at steps
    500-1000 and 0.99 / 0.98 by 2500-3000 (val 0.989 / 0.984), regression loss 0.016, the
    frozen coda's oracle CE 13.1 nats against arm A's 1.4. E's codes collapse onto one
    direction and the target measures nothing.

    Returns the trainable front tensors (empty when the model is legal). Called once, after
    ``training.train_only`` has been applied and before the optimizer is built.
    """
    tc = getattr(getattr(model, "cfg", None), "tul", None)
    if tc is None or not tc.code_target or tc.code_target_ref:
        return []
    n_pre = int(model.cfg.n_prelude)

    def _front(n: str) -> bool:
        if n.startswith(("embed.", "prelude.", "input_norm.", "value_embed")):
            return True
        if n.startswith("x0_injects."):
            part = n.split(".")[1]
            return part.isdigit() and int(part) < n_pre
        return False

    bad = [n for n, prm in model.named_parameters() if prm.requires_grad and _front(n)]
    if bad:
        raise ValueError(
            f"tul.code_target with a trainable front and no tul.code_target_ref: "
            f"{len(bad)} front tensors train (first: {bad[:3]}). E pools the LIVE "
            f"prelude's states of the next span, so the target drifts with the model and "
            f"its codes collapse - measured 2026-09-17 on wandb 38naddpq, own cosine "
            f"0.61 -> 0.99 and shuffled 0.56 -> 0.98 by step 3000, oracle CE 13.1 nats "
            f"against arm A's 1.4. Set tul.code_target_ref: true (a frozen VAE-stage twin "
            f"computes the target) or freeze the front with training.train_only.")
    return bad


def read_code_ref_state(path: str):
    """The ``code_ref`` state of a checkpoint, or ``None`` if it carries none.

    A checkpoint written before ``tul.code_target_ref`` existed (the VAE stage's, every
    `init_from` seed) has no such key, and that is the SNAPSHOT case, not an error. Read
    with ``mmap`` so a 2 GB file is not paged in twice on a resume.
    """
    return read_checkpoint_side_state(path, "code_ref")


def read_checkpoint_side_state(path: str, key: str):
    """A checkpoint's top-level ``key`` (a state carried beside ``model``: ``code_ref``,
    ``fan_target``), or ``None`` when the checkpoint carries none — the snapshot case
    for a checkpoint written before that key existed. Read with ``mmap``."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:                              # older serialisation: no mmap support
        ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get(key) if isinstance(ck, dict) else None


def load_weights_only(path: str, model: nn.Module, device: torch.device,
                      allow_plain_to_tul: bool = False) -> tuple[list, list]:
    """Initialise model WEIGHTS from a checkpoint, but reset the run to step 0.

    Returns ``(missing, unexpected)`` from ``load_state_dict``. The trainer ignores the
    return (its own guard is the 50 %-matched raise below); a caller that rebuilds a
    trained model outside the trainer SHOULD check it, because the count-based guard is
    weak where it matters: the QAT parametrizations rename only ~27 of 348 tensors, and
    those 27 are the embedding table and every MLP. See
    ``morph/training/quant_setup.py`` — apply the same transforms BEFORE this call.

    Unlike load_checkpoint (full resume: weights + optimizer + scaler + step), this
    loads ONLY the model tensors and leaves the optimizer/scaler FRESH and the step
    counter at 0. Used by `training.init_from` to seed a brand-new schedule (e.g. the
    25k whole-body gradual-prune run) from dense pretrained weights while keeping the
    schedule's step axis absolute from 0 — and sidestepping the optimizer-resume ppl
    spike (a fresh optimizer + the schedule's dense warmup absorb any startup bump).
    The weight-load path is byte-identical to load_checkpoint's (strict=False, same
    ._orig_mod. strip), which the live B5 resume already proved loads this seed cleanly.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt["model"]
    # _orig_mod-robust key alignment, SYMMETRIC on both sides. torch.compile inserts
    # `._orig_mod.` into wrapped-submodule keys; a checkpoint and this model may EACH carry it
    # independently — compiled↔compiled (match natively), or an UNcompiled init_from seed loaded
    # into a COMPILED model (the seed lacks `._orig_mod.` that the model's compiled MLP keys have).
    # The old raw-vs-strip pick only stripped the checkpoint side, so the uncompiled-seed→compiled-
    # model case silently dropped every compiled-submodule tensor to random init. Fix: canonicalize
    # BOTH sides (strip `._orig_mod.`) and map each checkpoint tensor onto the model's ACTUAL key.
    model_keys = set(model.state_dict().keys())
    def _canon(k):
        return k.replace("._orig_mod.", ".")
    canon_to_model = {_canon(k): k for k in model_keys}
    state = {}
    for k, v in raw.items():
        state[canon_to_model.get(_canon(k), k)] = v   # onto the model's real key, else leave → unexpected
    n_raw = sum(1 for k in raw if k in model_keys)
    n_strip = sum(1 for k in raw if _canon(k) in canon_to_model)   # true canonical match count
    drop_retired_tul_keys(state, model, path)
    drop_code_thinker_keys(state, model, path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if allow_plain_to_tul:
        # The plain → TUL bootstrap (training.resume_plain_to_tul). Without it this
        # function reports missing/unexpected and leaves the decision to a caller that
        # historically ignored both — so under the flag every homeless tensor is
        # classified and anything outside the two allowed classes RAISES here.
        _report_plain_to_tul("init_from", path,
                             *classify_plain_to_tul_keys(missing, unexpected, model))
    # Hard guard against a silent partial load: the MLP backbone (gate_up/down shadows)
    # MUST land. If almost nothing matched, the seed is incompatible — fail LOUD.
    n_loaded = len(model_keys) - len(missing)
    if n_loaded < 0.5 * len(model_keys):
        raise RuntimeError(
            f"init_from {path}: only {n_loaded}/{len(model_keys)} model tensors matched "
            f"(raw-match={n_raw}, strip-match={n_strip}). Seed/model key structure mismatch "
            f"— refusing to train from a mostly-random model."
        )
    print(f"  init_from {path}: loaded WEIGHTS only (step reset → 0, fresh optimizer); "
          f"seed step was {ckpt.get('step', '?')}; matched {n_loaded}/{len(model_keys)} "
          f"tensors via {'raw' if state is raw else 'stripped'} keys; "
          f"{len(missing)} missing / {len(unexpected)} unexpected", flush=True)
    return list(missing), list(unexpected)


def e_slot_seeded(missing: list[str], model: nn.Module) -> bool:
    """True when a weights-only load put a TRAINED ``tul.E_slot`` into ``model``.

    ``TULSlots.init_at_activation`` sets ``E_slot`` to the embedding-table mean at the step
    TUL switches on. That is right for a fresh TUL model and for a plain-to-TUL seed (the
    checkpoint has no ``E_slot``), and WRONG for ``init_from`` of a TUL checkpoint: the
    step resets to 0, the activation branch fires, and the checkpoint's trained slot
    embedding is overwritten. Found 2026-09-23 on LXTUL-E Stage 0, where the "frozen" ruler
    came back with one changed tensor (``tul.E_slot``, max |d| 0.10). Every earlier
    ``init_from`` of a TUL checkpoint (the LCTUL arms seeded from ``tul-code-vae``, gl1c,
    gl1bc, l3wake) started from the mean instead of the trained vector.

    ``missing`` is the load's missing-key list (``load_weights_only``); ``False`` when the
    model has no ``tul`` module or the checkpoint lacked the key."""
    mm = getattr(model, "_orig_mod", model)
    if getattr(mm, "tul", None) is None or not hasattr(mm.tul, "E_slot"):
        return False
    return not any(k.replace("_orig_mod.", "") == "tul.E_slot" for k in missing)


def check_init_from_keys(missing: list[str], unexpected: list[str], model: nn.Module,
                         new_modules: list[str] | tuple[str, ...], path: str) -> list[str]:
    """``training.init_from_new_modules``: the STRICT init_from key audit.

    ``load_weights_only`` on its own only refuses a load that matches under half the
    model; a single homeless tensor is printed as a count and dropped. When a run seeds a
    model that ADDS modules to a checkpoint (LXTUL-E Stage 0: the ruler plus a parallel
    head), exactly those modules may be missing and nothing else may move. So:

    * every UNEXPECTED key raises (a checkpoint tensor with no home is lost state);
    * every MISSING key must lie under a listed prefix (``_prune_mask`` buffers keep the
      back-compat exemption ``load_checkpoint`` gives them);
    * every listed prefix must be WHOLLY missing — each model tensor under it absent from
      the checkpoint — so a prefix cannot excuse a partial load of a module the
      checkpoint did carry.

    Prints the missing list in full and returns it."""
    prefixes = tuple(str(p) for p in new_modules)

    def _canon(k: str) -> str:
        return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")

    if unexpected:
        raise RuntimeError(
            f"init_from {path}: {len(unexpected)} checkpoint tensor(s) have no home in this "
            f"model: {sorted(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
    bad = [k for k in missing
           if not _canon(k).startswith(prefixes) and not k.endswith("_prune_mask")]
    if bad:
        raise RuntimeError(
            f"init_from {path}: {len(bad)} model tensor(s) are missing from the checkpoint "
            f"outside training.init_from_new_modules={list(prefixes)}: "
            f"{sorted(bad)[:8]}{'...' if len(bad) > 8 else ''}")
    miss = {_canon(k) for k in missing}
    for p in prefixes:
        under = [_canon(k) for k in model.state_dict() if _canon(k).startswith(p)]
        if not under:
            raise RuntimeError(f"init_from {path}: training.init_from_new_modules prefix "
                               f"{p!r} names no tensor of this model")
        loaded = [k for k in under if k not in miss]
        if loaded:
            raise RuntimeError(
                f"init_from {path}: prefix {p!r} is listed as NEW but the checkpoint "
                f"carries {len(loaded)} of its {len(under)} tensor(s) ({loaded[:4]}). A "
                f"new module is wholly missing; drop it from the list or fix the seed.")
    print(f"  init_from {path}: {len(missing)} tensor(s) NEW in this model (kept their "
          f"fresh init), all under {list(prefixes)}; 0 unexpected:", flush=True)
    for k in sorted(missing):
        print(f"      {k}", flush=True)
    return list(missing)


@torch.no_grad()
def _host_floats(xs: list) -> list[float]:
    """``[float(x) for x in xs]`` with ONE device-to-host copy for all the tensors.

    Each ``float()`` of a CUDA scalar is its own copy and stream sync; the per-step probe
    read about 45 of them after the backward while the GPU sat idle (2026-10-04 speed
    pass). The values are the same: every tensor is widened to fp32 on the device (exact
    for fp32 / bf16 / fp16 sources) and fp32 converts to a Python float exactly, as
    ``float()`` does. Python numbers pass through. Tensors are flattened in order, so a
    multi-element tensor contributes one float per element (``t.tolist()`` order)."""
    tens = [x for x in xs if torch.is_tensor(x)]
    flat = (torch.cat([t.detach().reshape(-1).float() for t in tens]).tolist()
            if tens else [])
    out, i = [], 0
    for x in xs:
        if torch.is_tensor(x):
            n = x.numel()
            out.extend(flat[i:i + n])
            i += n
        else:
            out.append(float(x))
    return out


def _block_gain(acc: dict[str, float], region: str) -> dict[str, float]:
    """Per-block backward gain of a stacked region, from its per-block squared grad norms.

    Returns ``{}`` when the region has fewer than 3 blocks — a two-point "fit" is a ratio
    wearing a regression's clothes, and its r2 is always exactly 1.
    """
    import math

    vals = []
    i = 0
    while f"{region}.{i}" in acc:
        vals.append(acc[f"{region}.{i}"])
        i += 1
    pts = [(j, 0.5 * math.log(v)) for j, v in enumerate(vals) if v > 0]  # 0.5: acc holds squares
    if len(pts) < 3:
        return {}
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return {}
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in pts)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    return {
        f"preclip/{region}_block_gain": math.exp(-b),
        f"preclip/{region}_block_gain_r2": (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def _jacobian_probe(model, probe, x, y, layout, bag_size, iters) -> dict[str, float]:
    """sigma_max(J_core) at this step's operating point, as a flat wandb-ready dict.

    The gradient probe measures MAGNITUDES; this measures the OPERATOR. The block
    backward gain says the core amplifies, but not whether the map became expansive or
    whether the realized direction merely rotated into an amplifying direction the map
    always had. Only sigma_max(J) can tell those apart (see the module docstring).

    RNG NEUTRALITY IS LOAD-BEARING. The probe needs the real operating point, so it runs
    one extra forward in TRAINING mode, which draws the Poisson slot depths and the token
    dropout mask — that would shift every later step and destroy the bit-reproducibility
    the whole divergence programme rests on. The CPU and CUDA generator states are saved
    and restored around the forward, so a run with the probe on is bit-identical to the
    same run with it off. `tests/test_core_jacobian.py::test_probe_is_rng_neutral` fails
    if that stops being true.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    out: dict[str, float] = {}
    try:
        with probe.capture() as points:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, labels=y, bag_size=bag_size, slot_layout=layout)
        # The MEASUREMENT stays inside the restore too. It rebuilds the core step in
        # training mode, so the blocks' dropout (0.1 in base.yaml) draws from the global
        # CUDA generator on every power iteration; a restore around the capture alone
        # left the next training step's dropout masks and Poisson depths shifted by
        # exactly that draw (onset-capture smoke, 2026-09-03: the replay diverged at the
        # step after every probed step, weights and gradients bit-identical).
        by_iter = {int(pt["iter_idx"]): pt for pt in points}
        results = {int(t): probe.measure(by_iter[int(t)]) for t in iters if int(t) in by_iter}
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
    for t, res in results.items():
        out[f"jac/sigma_t{t}"] = res.sigma_step
        out[f"jac/sigma_conv_t{t}"] = res.rel_change
        # Typical gain (||J||_F / sqrt(n)): what a generic direction sees. sigma is the
        # worst direction; the two together say whether the map is expansive on average
        # or only along a direction the data may never occupy (module docstring).
        out[f"jac/rms_t{t}"] = res.rms_step
        if res.sigma_blocks:
            out[f"jac/sigma_blockprod_t{t}"] = res.block_product
            out[f"jac/rms_blockgeo_t{t}"] = res.rms_block_gain
            for i, sb in enumerate(res.sigma_blocks):
                out[f"jac/sigma_b{i}_t{t}"] = sb
            for i, rb in enumerate(res.rms_blocks):
                out[f"jac/rms_b{i}_t{t}"] = rb
    if points:
        out["jac/n_iters_captured"] = float(len(points))
    return out


def _preclip_probe(model) -> dict[str, float]:
    """PRE-CLIP per-region / per-block gradient norms plus the looped-core state probe.

    The onset of the TUL core takeover lasts about 140 steps and every existing gradient
    log is POST-clip and every 100 steps, so nobody has ever seen inside it (plan task
    1.1). Post-clip ratios between regions are exact — the clip is one uniform rescale —
    but the absolute values are not, and it is the absolute pre-clip magnitude that says
    whether the core is producing a 1e8 gradient or merely the largest share of a healthy
    one.

    Cost: one fused ``_foreach_norm`` over every gradient plus ONE host sync, so this is
    affordable every step. Call it AFTER ``scaler.unscale_`` and BEFORE
    ``clip_grad_norm_`` — that window is the only place the gradients are both unscaled
    and unclipped.

    CAUTION, measured 2026-08-24: this reads `p.grad` after the backward of the FULL
    objective, so a regulariser that is not uniform over the parameter tree lands inside the
    region it constrains and inflates that region's share. The spectral penalty is
    core-local, and on a penalised arm `preclip/total` reached 1.6e5 while its control sat at
    1.35 — the core share went to 0.998 because of the PENALTY's gradient, not the model's.
    Separating them needs a second backward, which this probe deliberately does not do. Read
    `preclip/core_share` as contaminated on any run with a region-local loss term.

    Returns a flat wandb-ready dict; the caller logs it at the step it belongs to.
    """
    names, grads = [], []
    for name, p in model.named_parameters():
        if p.grad is not None:
            names.append(name)
            grads.append(p.grad.detach())
    out: dict[str, float] = {}
    if grads:
        # torch.compile wraps the module, so names arrive as "_orig_mod.core.0.…";
        # strip the wrapper or every parameter lands in one bucket named "_orig_mod".
        sq = torch.stack(torch._foreach_norm(grads)).float().square().tolist()  # 1 sync
        acc: dict[str, float] = {}
        for name, v in zip(names, sq):
            parts = name.replace("_orig_mod.", "").split(".")
            acc[parts[0]] = acc.get(parts[0], 0.0) + v
            # Per-BLOCK for the stacked regions: a region total says the core exploded,
            # the per-block profile says whether it amplifies geometrically layer by layer
            # (an unstable backward operator) or runs away in one block alone.
            if parts[0] in ("prelude", "core", "coda") and len(parts) > 1 and parts[1].isdigit():
                bk = f"{parts[0]}.{parts[1]}"
                acc[bk] = acc.get(bk, 0.0) + v
        out = {f"preclip/{k}": v ** 0.5 for k, v in acc.items()}
        # Regions only (keys with no dot) — the per-block entries are a subset of them.
        out["preclip/total"] = sum(v for k, v in acc.items() if "." not in k) ** 0.5
        # The per-block BACKWARD gain of the core, as one number. See
        # lab/experiments/results/2026-08-23-tul-block-backward-gain.md: the backward runs
        # core.N-1 -> core.0, so a uniform per-block amplification g puts block 0 a factor
        # g^(N-1) above the last block. Fit log‖grad_i‖ = a + b·i and report exp(-b).
        # r2 is reported WITH it and is not optional: a healthy profile is flat and noisy
        # (r2 ~ 0.1) so the gain estimate means nothing there, while a sick one is cleanly
        # geometric (median r2 0.971). Read them as a pair.
        out.update(_block_gain(acc, "core"))

    # The GLA carried state and the realized per-iteration core gain, from the forward
    # that produced these gradients. The carry is a SECOND recurrent loop inside the core
    # loop, with a forget gate biased to alpha near 1, and nothing has ever watched it.
    lp = getattr(getattr(model, "_orig_mod", model), "_loop_probe", None)
    if lp:
        _lp_items = [(key, t) for key, t in lp.items() if t is not None]
        _lp_flat = _host_floats([t.reshape(-1) for _, t in _lp_items])   # one copy
        _lp_at = 0
        for key, t in _lp_items:
            seq = _lp_flat[_lp_at:_lp_at + t.numel()]
            _lp_at += t.numel()
            out[f"loop/{key}_max"] = max(seq)
            out[f"loop/{key}_last"] = seq[-1]
            # The per-iteration profile itself: a gain that COMPOUNDS with the iteration
            # index is a different disease from one that spikes at t=0.
            for i, v in enumerate(seq):
                out[f"loop/{key}_t{i}"] = v
    # Onset capture: the cotangent norm that reached each grad iteration's output during
    # the backward that produced these gradients (`training.loop_cot_probe`). A norm that
    # GROWS with decreasing iteration index is the backward product through the loop; a
    # flat one with a huge core gradient says the blow-up sits in the weights' own path.
    root = getattr(model, "_orig_mod", model)
    # Progressive loss (`tul.progressive_p`): the fraction of valid slots that drew a
    # private no-grad prefix this step, the mean prefix length, and the mean number of
    # passes that actually carried gradient. None on every model without the knob.
    prog = getattr(root, "_loop_prog", None)
    if prog:
        for key, v in prog.items():
            out[f"loop/{key}"] = float(v)
    # Per-pass MUX (`tul.mux_every_pass`): how many terms the local loss averaged this
    # step and the value of each one. A per-pass signal that is doing anything should make
    # the deep passes' terms fall faster than the shallow ones. None on every other model.
    mux = getattr(root, "_loop_mux", None)
    if mux:
        for key, v in mux.items():
            out[f"loop/{key}"] = float(v)
    # Gradient-conditioned passes (`tul.grad_pass`): the local own-span target's value at
    # each pass — the trajectory the loop is handed the gradient OF, so a loop that is
    # descending it reads a falling ladder — and the injected term's size relative to the
    # state. None on every other model.
    gp = getattr(root, "_loop_gradpass", None)
    if gp:
        for key, v in gp.items():
            out[f"loop/{key}"] = float(v)
    cot = getattr(root, "_loop_cot", None)
    if cot:
        ts = sorted(cot)
        # Clip-through-time (model.slot_cot_clip): what the backward actually carried after
        # the clip, and the fraction of rows the cap shrank, per iteration.
        post = getattr(root, "_loop_cot_post", None) or {}
        bind = getattr(root, "_loop_cot_bind", None) or {}
        pts = sorted(post)
        # Every reading below in ONE device-to-host copy (`_host_floats`), same values.
        _v = _host_floats([cot[t] for t in ts] + [post[t] for t in pts]
                          + [bind[t] for t in pts] + list(bind.values()))
        _n, _m = len(ts), len(pts)
        cot_h = dict(zip(ts, _v[:_n]))
        post_h = dict(zip(pts, _v[_n:_n + _m]))
        bind_h = dict(zip(pts, _v[_n + _m:_n + 2 * _m]))
        bind_all = _v[_n + 2 * _m:]
        for t in ts:
            out[f"loop/cot_norm_t{t}"] = cot_h[t]
        # The product through the loop in one number: first grad iteration over the last.
        out["loop/cot_ratio"] = cot_h[ts[0]] / (cot_h[ts[-1]] + 1e-12)
        for t in pts:
            out[f"loop/cot_post_t{t}"] = post_h[t]
            out[f"loop/cot_bind_t{t}"] = bind_h[t]
        if post:
            out["loop/cot_post_ratio"] = post_h[pts[0]] / (cot_h[ts[-1]] + 1e-12)
            out["loop/cot_bind_max"] = max(bind_all)
    return out


def diag_prune_optstate(model, optimizer, step: int, path: str) -> None:
    """Root-cause the AdEMAMix prune divergence (env MORPH_DIAG_OPT=<path>).

    For every CMSBlockLinear, dequant the optimizer's slow-EMA m₂ and second-moment ν,
    pair with the live grad, and reconstruct the per-element update (g+α·m₂)/(√(ν/bc2)+ε).
    Split positions DEAD (pruned, _prune_mask==0) vs LIVE and report the GLOBAL max|update|
    with its components — so we see EXACTLY what blows up: numerator (α·m₂ on a charged
    slow EMA) vs denominator (ν collapse), and whether it is a dead or a LIVE param. Handles
    all three state formats (fused linear-int8 m2_code, de-fused dynamic-map m2_q, fp32 m2).
    """
    opt = optimizer
    if not hasattr(opt, "state"):
        return
    from morph.model.layers.block_sparse import CMSBlockLinear

    def _grp(p):
        for g in opt.param_groups:
            for q in g["params"]:
                if q is p:
                    return g
        return None

    def _deq_any(st, key, signed, p):
        if key == "nu" and "nu_sqrt_code" in st:         # fused sqrt-ν int8 (BLOCK=256)
            code = st["nu_sqrt_code"].float()
            amax = st["nu_sqrt_amax"].float()
            scale = (amax / 127.0).repeat_interleave(256)[: code.numel()]
            nu_sqrt = code * scale
            return (nu_sqrt * nu_sqrt).view_as(p)
        if f"{key}_code" in st:                          # fused linear-int8 (BLOCK=256)
            code = st[f"{key}_code"].float()
            amax = st[f"{key}_amax"].float()
            scale = (amax / 127.0).repeat_interleave(256)[: code.numel()]
            return (code * scale).view_as(p)
        if f"{key}_q" in st:                             # de-fused dynamic-map
            return opt._deq(st[f"{key}_q"], st[f"{key}_amax"], signed, p)
        if key in st:                                    # fp32
            return st[key].view_as(p)
        return None

    root = getattr(model, "_orig_mod", model)
    worst = (-1.0, None)
    amom_dead = amom_live = 0.0
    minnu_dead = minnu_live = float("inf")
    zero_nu_dead = zero_nu_live = 0
    floor_den_dead = floor_den_live = 0
    total_dead = total_live = 0
    n_layers = n_with_state = 0
    per_layer = {}   # NEW: name -> (zeroNu_live_count, maxU_in_layer) to localize ν-collapse
    clip10_live = clip25_live = 0   # NEW: live coords whose ACTUAL |update| would be clipped at 10/25
    max_rms = (-1.0, None)          # NEW: max per-tensor update-RMS (collective-move magnitude) + layer
    max_rel = (-1.0, None)          # NEW: max per-tensor rel-step lr·‖u‖/‖w‖ (trust-ratio quantity) + layer
    # β1=0 root-cause measurement: snr = |m₂|/denom ≈ |mean|/rms. Low snr on failing
    # (saturating) coords → magnitude problem → SNR gate is appropriate. Sign-flip between
    # g and m₂ → oscillation → directional gate. Measures both globally + on the saturating
    # subset so the mechanism can be confirmed from data.
    snr_lt01_live = snr_lt03_live = 0          # live coords with snr<0.1 / <0.3
    sat_count = 0                              # live coords with |update|>10 (the "failing" coords)
    sat_sign_agree = 0                         # of those, how many have sign(g)==sign(m₂)
    sat_snr_sum = 0.0                          # Σ snr over saturating coords (÷ sat_count = mean)
    for name, layer in root.named_modules():
        if not isinstance(layer, CMSBlockLinear):
            continue
        n_layers += 1
        p = layer._prune_target_weight()                 # the param the optimizer holds
        st = opt.state.get(p)
        if not st or st.get("init"):
            continue
        m2 = _deq_any(st, "m2", True, p)
        nu = _deq_any(st, "nu", False, p)
        if m2 is None or nu is None or p.grad is None:
            continue
        n_with_state += 1
        grp = _grp(p)
        a_t, b2, b3_t = opt._sched(grp["step"], grp)
        bc2 = 1.0 - b2 ** grp["step"]
        eps = grp["eps"]
        g = p.grad.float()
        # Match the optimizer's ACTUAL eps placement so maxU is the REAL update magnitude
        # (eps-OUTSIDE runs were being under-reported by the old always-eps-inside reconstruction).
        eps_inside = bool(getattr(opt, "eps_inside", True))
        denom = (nu / bc2 + eps).sqrt() if eps_inside else ((nu / bc2).sqrt() + eps)
        amom = a_t * m2
        upd = ((g + amom) / denom).reshape(-1).abs()
        # dead mask: expand [R,C] _prune_mask → [out,in] elementwise (True=alive)
        B = layer.tile_size
        keep = layer._prune_mask.view(layer.R, 1, layer.C, 1).expand(
            layer.R, B, layer.C, B).reshape(layer.out_features, layer.in_features).reshape(-1)
        dead = ~keep.bool().to(upd.device)
        amf = amom.reshape(-1).abs()
        nuf = nu.reshape(-1)
        zero_nu = (nuf == 0)
        floor_den = (denom.reshape(-1) <= (eps ** 0.5) * 1.0001)
        if dead.any():
            amom_dead = max(amom_dead, float(amf[dead].max()))
            minnu_dead = min(minnu_dead, float(nuf[dead].min()))
            zero_nu_dead += int(zero_nu[dead].sum())
            floor_den_dead += int(floor_den[dead].sum())
            total_dead += int(dead.sum())
        live = ~dead
        layer_zero_live = 0
        if live.any():
            amom_live = max(amom_live, float(amf[live].max()))
            minnu_live = min(minnu_live, float(nuf[live].min()))
            layer_zero_live = int(zero_nu[live].sum())
            zero_nu_live += layer_zero_live
            floor_den_live += int(floor_den[live].sum())
            total_live += int(live.sum())
            updl = upd[live]
            clip10_live += int((updl > 10.0).sum())
            clip25_live += int((updl > 25.0).sum())
            # SNR + sign-agreement measurement: snr uses raw |m₂| (matches the optimizer
            # gate exactly, not amf=|α·m₂|).
            m2f = m2.reshape(-1).abs()
            snr = m2f / denom.reshape(-1)                    # |m₂|/denom ≈ |mean|/rms, per coord
            snr_live = snr[live]
            snr_lt01_live += int((snr_live < 0.1).sum())
            snr_lt03_live += int((snr_live < 0.3).sum())
            sign_agree = g.reshape(-1).sign() == m2.reshape(-1).sign()
            sat = live & (upd > 10.0)                        # the FAILING (saturating) coords
            if sat.any():
                sat_count += int(sat.sum())
                sat_sign_agree += int(sign_agree[sat].sum())
                sat_snr_sum += float(snr[sat].sum())
        lrms = float(upd.pow(2).mean().sqrt())                   # NEW: per-tensor update-RMS
        if lrms > max_rms[0]:
            max_rms = (lrms, name)
        lr_p = float(grp.get("lr", 0.0))                         # NEW: rel-step = lr·‖u‖/‖w‖
        rel = lr_p * float(upd.norm()) / (float(p.float().norm()) + 1e-12)
        if rel > max_rel[0]:
            max_rel = (rel, name)
        per_layer[name] = (layer_zero_live, float(upd.max()), rel)  # localize collapse + rel-step
        j = int(upd.argmax())
        if float(upd[j]) > worst[0]:
            gf, df = g.reshape(-1), denom.reshape(-1)
            worst = (float(upd[j]), dict(
                layer=name, dead=bool(dead[j]), g=float(gf[j]), amom=float(amf[j]),
                denom=float(df[j]), nu=float(nuf[j]), m2=float(m2.reshape(-1)[j]), a_t=a_t, b3=b3_t))
    b = worst[1] or {}
    with open(path, "a") as f:
        f.write(
            f"step={step} layers={n_with_state}/{n_layers} maxU={worst[0]:.3e} "
            f"dead={b.get('dead')} layer={b.get('layer')} g={b.get('g',0):.3e} "
            f"amom={b.get('amom',0):.3e} denom={b.get('denom',0):.3e} nu={b.get('nu',0):.3e} "
            f"m2={b.get('m2',0):.3e} a_t={b.get('a_t',0):.2f} b3={b.get('b3',0):.5f} "
            f"| amomMax d/l={amom_dead:.3e}/{amom_live:.3e} "
            f"minNu d/l={minnu_dead:.2e}/{minnu_live:.2e} "
            f"zeroNu d/l={zero_nu_dead}/{total_dead}:{zero_nu_live}/{total_live} "
            f"floorDen d/l={floor_den_dead}/{total_dead}:{floor_den_live}/{total_live} "
            f"clipLive >10={clip10_live} >25={clip25_live} "
            f"maxUpdRMS={max_rms[0]:.3f}@{max_rms[1]} "
            f"relStep={max_rel[0]:.2e}@{max_rel[1]}\n"
        )
        # NEW per-layer line: top layers by ν-collapse (only when any) + top-3 by maxU.
        # Localizes whether collapse concentrates at prelude.0.down (first-layer/arch) or spreads.
        z_top = sorted(((z, k) for k, (z, _, _) in per_layer.items() if z > 0), reverse=True)[:8]
        u_top = sorted(((u, k) for k, (_, u, _) in per_layer.items()), reverse=True)[:3]
        r_top = sorted(((r, k) for k, (_, _, r) in per_layer.items()), reverse=True)[:6]
        zstr = " ".join(f"{k}={z}" for z, k in z_top) if z_top else "none"
        ustr = " ".join(f"{k}={u:.2f}" for u, k in u_top)
        rstr = " ".join(f"{k}={r:.2e}" for r, k in r_top)
        f.write(f"  PERLAYER zeroNuLive[{zstr}] topMaxU[{ustr}] topRelStep[{rstr}]\n")
        # SNR distribution (live) + sign-agreement on saturating (|update|>10) coords.
        # Low sat-SNR + high sat-signAgree → magnitude problem → SNR gate appropriate.
        # Low sat-signAgree → oscillation → directional/sign gate.
        tl = max(total_live, 1)
        sat_sa = (sat_sign_agree / sat_count) if sat_count else float("nan")
        sat_snr = (sat_snr_sum / sat_count) if sat_count else float("nan")
        f.write(
            f"  SNRGATE snr<0.1={snr_lt01_live}/{total_live}({snr_lt01_live/tl:.3f}) "
            f"snr<0.3={snr_lt03_live}/{total_live}({snr_lt03_live/tl:.3f}) "
            f"| satCoords(|u|>10)={sat_count} satSignAgree={sat_sa:.3f} satMeanSNR={sat_snr:.3e}\n"
        )


def diag_optstate_allparams(model, optimizer, step: int, path: str) -> None:
    """All-param, gate-aware blow-up localizer (env MORPH_DIAG_OPT=<path>).

    Sweeps every optimizer-state param, reconstructs the real update (gate·g + α·m₂)/denom
    with the SNR gate applied, and decomposes it into the gated-g term |gate·g|/denom vs the
    slow-EMA term |α·m₂|/denom. Reports the global-worst param by full name + which term
    drives it, plus the worst per name-category (prelude/core/coda/embed/attn/hc/etc.).
    """
    opt = getattr(optimizer, "_opt", optimizer)
    if not hasattr(opt, "state"):
        return
    root = getattr(model, "_orig_mod", model)
    kappa = float(getattr(opt, "g_snr_gate_kappa", 0.0))
    floor = float(getattr(opt, "g_snr_gate_floor", 0.1))
    eps_inside = bool(getattr(opt, "eps_inside", True))
    grp_of = {}
    for g in opt.param_groups:
        for q in g["params"]:
            grp_of[id(q)] = g

    def _deq(st, key, signed, p):
        if f"{key}_q" in st:                       # de-fused dynamic-map (8-bit)
            return opt._deq(st[f"{key}_q"], st[f"{key}_amax"], signed, p).reshape(-1)
        if key in st:                              # fp32 (no-decay group: embed/HC/norm)
            return st[key].reshape(-1)
        return None

    def _cat(nm):
        for k in ("prelude", "core", "coda", "embed", "attn", "mhc", "hc",
                  "inject", "norm", "lm", "log_"):
            if k in nm:
                return k
        return "other"

    worst = (-1.0, None)
    max_g = (-1.0, None)
    max_am = (-1.0, None)
    cat_worst = {}
    for nm, p in root.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        st = opt.state.get(p)
        if not st or st.get("init"):
            continue
        grp = grp_of.get(id(p))
        if grp is None:
            continue
        a_t, b2, b3_t = opt._sched(grp["step"], grp)
        bc2 = 1.0 - b2 ** grp["step"]
        eps = grp["eps"]
        m2 = _deq(st, "m2", True, p)
        nu = _deq(st, "nu", False, p)
        if m2 is None or nu is None:
            continue
        g = p.grad.float().reshape(-1)
        denom = (nu / bc2 + eps).sqrt() if eps_inside else ((nu / bc2).sqrt() + eps)
        if kappa > 0.0:
            gate = (m2.abs() / denom / kappa).clamp(0.0, 1.0).mul_(1.0 - floor).add_(floor)
        else:
            gate = torch.ones_like(g)
        g_term = (gate * g).abs() / denom              # gated raw-g contribution
        am_term = (a_t * m2).abs() / denom             # UNGATED slow-EMA contribution
        upd = ((gate * g + a_t * m2) / denom).abs()
        mu = float(upd.max())
        cat = _cat(nm)
        if mu > cat_worst.get(cat, (-1.0, None))[0]:
            cat_worst[cat] = (mu, nm)
        mg, ma = float(g_term.max()), float(am_term.max())
        if mg > max_g[0]:
            max_g = (mg, nm)
        if ma > max_am[0]:
            max_am = (ma, nm)
        if mu > worst[0]:
            j = int(upd.argmax())
            worst = (mu, dict(name=nm, cat=cat, gterm=float(g_term[j]), amterm=float(am_term[j]),
                              gate=float(gate[j]), denom=float(denom[j]), m2=float(m2[j]),
                              nu=float(nu[j]), g=float(g[j]), a_t=a_t))
    b = worst[1] or {}
    driver = "amom" if b.get("amterm", 0) > b.get("gterm", 0) else "g"
    with open(path, "a") as f:
        f.write(
            f"  ALLPARAM step={step} worstU={worst[0]:.3e}@{b.get('name')}[{b.get('cat')}] "
            f"driver={driver} gTerm={b.get('gterm', 0):.3e} amTerm={b.get('amterm', 0):.3e} "
            f"gate={b.get('gate', 0):.3f} denom={b.get('denom', 0):.2e} m2={b.get('m2', 0):.2e} "
            f"nu={b.get('nu', 0):.2e} a_t={b.get('a_t', 0):.2f} "
            f"| maxGterm={max_g[0]:.3e}@{max_g[1]} maxAMterm={max_am[0]:.3e}@{max_am[1]}\n"
        )
        cats = " ".join(
            f"{c}={v:.2e}" for c, (v, _n) in sorted(cat_worst.items(), key=lambda x: -x[1][0])
        )
        f.write(f"  ALLPARAM_CAT {cats}\n")


_M2G_SNAP: dict = {}   # name -> (snap_step, m2_cpu_fp32_vec) for slow-EMA self-coherence


def diag_m2g_geometry(model, optimizer, step: int, path: str, snap_every: int = 50) -> None:
    """Slow-EMA geometry localizer on the CORE params (env MORPH_DIAG_M2G=<path>).

    Measures two diagnostic quantities for the slow-EMA / gradient geometry:
      (i)  cos(m₂, g): drops if the slow EMA has rotated away from the current gradient
           → indicates stale off-manifold direction → cure: directional-trust gate.
      (ii) cos(m₂_t, m₂_{t-k}): sustained ≈1 if drift is a coherent ramp → cure: damp
           persistence rather than mis-alignment.
    Logs both per core tensor each step + medians. Off by default (core params only; cheap).
    """
    opt = getattr(optimizer, "_opt", optimizer)
    if not hasattr(opt, "state"):
        return
    root = getattr(model, "_orig_mod", model)

    def _deq(st, key, signed, p):
        if f"{key}_q" in st:
            return opt._deq(st[f"{key}_q"], st[f"{key}_amax"], signed, p).reshape(-1)
        if key in st:
            return st[key].reshape(-1)
        return None

    rows = []
    cos_mg_all, cos_self_all = [], []
    for nm, p in root.named_parameters():
        if "core." not in nm or not p.requires_grad or p.grad is None:
            continue
        st = opt.state.get(p)
        if not st or st.get("init"):
            continue
        m2 = _deq(st, "m2", True, p)
        if m2 is None:
            continue
        g = p.grad.float().reshape(-1)
        m2n, gn = float(m2.norm()), float(g.norm())
        cos_mg = float((m2 @ g) / (m2n * gn + 1e-12))
        cos_self, age = float("nan"), -1
        prev = _M2G_SNAP.get(nm)
        if prev is not None:
            ps, pv = prev
            pv = pv.to(m2.device)
            cos_self = float((m2 @ pv) / (m2n * float(pv.norm()) + 1e-12))
            age = step - ps
        if prev is None or (step - prev[0]) >= snap_every:
            _M2G_SNAP[nm] = (step, m2.detach().to("cpu"))
        rows.append((nm, m2n, gn, cos_mg, cos_self, age))
        cos_mg_all.append(cos_mg)
        if cos_self == cos_self:
            cos_self_all.append(cos_self)
    if not rows:
        return
    import statistics as _stat
    med_mg = _stat.median(cos_mg_all)
    med_self = _stat.median(cos_self_all) if cos_self_all else float("nan")
    rows.sort(key=lambda r: r[3])     # ascending cos(m₂,g) → most-rotated-from-g first
    with open(path, "a") as f:
        f.write(f"M2G step={step} n={len(rows)} medCos_m2g={med_mg:.3f} medCos_self={med_self:.3f} "
                "| lowestCos_m2g: "
                + " ".join(f"{r[0].split('core.', 1)[1]}={r[3]:.2f}" for r in rows[:4]) + "\n")
        for nm, m2n, gn, cmg, cself, age in rows:
            f.write(f"  M2G_T step={step} {nm} m2n={m2n:.3e} gn={gn:.3e} "
                    f"cos_m2g={cmg:.4f} cos_self={cself:.4f} age={age}\n")


def diag_m2g_numerator(optimizer, model, step: int, path: str, dense: bool,
                       name_filter=None):
    """Drain the optimizer's g-vs-numerator geometry capture.

    Metrics (computed inside the optimizer step from the exact working copies):
      cos(g, m₂)            → if this drops after a prune event, the slow EMA is stale relative
                               to the new landscape.
      cos(g, α·m₂+gated_g) → whether the full numerator still aligns with the current gradient.
      ‖α·m₂‖ / ‖gated_g‖   → how much the stale term outweighs the fresh (gated) gradient.
    Re-arms capture lazily after any optimizer rebuild (compact/route swap the param objects).
    Writes medians every call; per-tensor rows only when `dense` (near prune events) to keep
    log size bounded. Returns a dict of medians for wandb.
    """
    base = getattr(optimizer, "_opt", optimizer)
    if not hasattr(base, "set_diag_capture"):
        return None                                   # not an AdEMAMixB1Zero (e.g. AdamW)
    if not base._diag_capture or not base._diag_names:
        base.set_diag_capture(model, name_filter=name_filter)  # (re)arm after build/rebuild
    rows = base._diag_rows
    base._diag_rows = []                              # drain — bounded memory
    if not rows:
        return None
    import statistics as _stat
    cos_m2g = [r[1] for r in rows]
    cos_num = [r[2] for r in rows]
    ratio = [r[3] for r in rows]
    med = {
        "diag/cos_g_m2_med": _stat.median(cos_m2g),
        "diag/cos_g_num_med": _stat.median(cos_num),
        "diag/alpha_m2_over_gatedg_med": _stat.median(ratio),
        "diag/cos_g_m2_min": min(cos_m2g),
    }
    rows.sort(key=lambda r: r[1])                     # ascending cos(g,m₂) → most-stale first
    with open(path, "a") as f:
        f.write(f"M2N step={step} n={len(rows)} medCos_g_m2={med['diag/cos_g_m2_med']:.3f} "
                f"medCos_g_num={med['diag/cos_g_num_med']:.3f} "
                f"medRatio_am_gg={med['diag/alpha_m2_over_gatedg_med']:.3f} "
                f"minCos_g_m2={med['diag/cos_g_m2_min']:.3f}"
                + (" PRUNE_WINDOW" if dense else "") + "\n")
        if dense:
            for nm, cgm, cgn, rt, m2n, gn, amn, ggn in rows:
                short = nm.split("core.", 1)[-1]
                f.write(f"  M2N_T step={step} {short} cos_g_m2={cgm:.4f} cos_g_num={cgn:.4f} "
                        f"am/gg={rt:.3f} m2n={m2n:.3e} gn={gn:.3e} amn={amn:.3e} ggn={ggn:.3e}\n")
    return med


def diag_forward_norms(model, step: int, path: str) -> None:
    """Forward-side blow-up localizer (env MORPH_DIAG_FWD=1, writes to MORPH_DIAG_OPT path).

    When optimizer-state diagnostics show no anomaly but loss explodes, the cause is likely
    forward-side: a residual-stream activation blowup that the looped core amplifies. This
    probe captures, per training step:

      FWDNORM : per-block output (residual-stream) L2 norm — localizes which block's
                activations grow first. Core blocks run T× per step; the max over iterations
                is reported.
      TERNFLIP: count of backbone ternary weights that changed {-1,0,+1} state since the
                previous step — tests the hypothesis of mass ternary sign-flip leading to a
                discontinuous effective-weight change. Pre-carve _ternary_mode path only.

    Hooks are registered once (idempotent). Run with MORPH_DIAG_OPT_EVERY=1 for per-step norms.
    """
    import torch
    import torch.nn.utils.parametrize as _P
    from morph.model.ternary_qat import TernarySTE

    root = getattr(model, "_orig_mod", model)

    # ── Lazy one-time hook registration ────────────────────────────────────
    if not getattr(model, "_diag_fwd_hooked", False):
        model._diag_fwd = {}

        def _pre_hook(_m, _inp):
            model._diag_fwd.clear()

        def _mk(name):
            def _hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    n = t.detach().float().norm().item()
                    d = model._diag_fwd
                    if n > d.get(name, 0.0):
                        d[name] = n
            return _hook

        root.register_forward_pre_hook(_pre_hook)
        for sec in ("prelude", "core", "coda"):
            mods = getattr(root, sec, None)
            if mods is None:
                continue
            for i, blk in enumerate(mods):
                blk.register_forward_hook(_mk(f"{sec}.{i}"))
        for nm in ("embed", "final_norm"):
            sub = getattr(root, nm, None)
            if sub is not None:
                sub.register_forward_hook(_mk(nm))
        model._diag_fwd_hooked = True
        model._diag_prev_tern = {}
        # First call: hooks fire on the NEXT forward → nothing to report yet.
        return

    fwd = dict(model._diag_fwd)
    if not fwd:
        return

    # ── Ternary {-1,0,+1} state-flip count since previous step ─────────────────
    # Ternary backbone = a TernarySTE PARAMETRIZATION on module.weight (NOT the CMSBlockLinear
    # _ternary_mode flag). The parametrization runs on .weight access → m.weight IS the realized
    # ternary (code·scale, scale>0) → sign(m.weight) == the {-1,0,+1} code exactly. Count how
    # many codes changed vs the previous step, totalled + per-section, to test whether a CORE-layer
    # flip burst coincides with the single-step core residual-stream excursion.
    prev = model._diag_prev_tern
    flip_total = 0
    flip_worst = (-1, None)
    flip_sec = {}   # section -> flip count this step
    with torch.no_grad():
        for name, m in root.named_modules():
            if not _P.is_parametrized(m, "weight"):
                continue
            if not any(isinstance(p, TernarySTE) for p in m.parametrizations.weight):
                continue
            tern = torch.sign(m.weight.detach()).to(torch.int8)   # {-1,0,+1} code
            old = prev.get(name)
            if old is not None and old.shape == tern.shape:
                f = int((tern != old).sum().item())
                flip_total += f
                sec = name.split(".")[0]
                flip_sec[sec] = flip_sec.get(sec, 0) + f
                if f > flip_worst[0]:
                    flip_worst = (f, name)
            prev[name] = tern
    flip_sec_str = " ".join(f"{s}={flip_sec[s]}" for s in sorted(flip_sec, key=lambda k: -flip_sec[k]))

    # ── Worst block + per-section max residual-stream norm ──────────────────
    worst = max(fwd.items(), key=lambda kv: kv[1])
    sec_max = {}
    for k, v in fwd.items():
        sec = k.split(".")[0]
        if v > sec_max.get(sec, 0.0):
            sec_max[sec] = v
    sec_str = " ".join(f"{s}={sec_max[s]:.3e}" for s in
                       ("embed", "prelude", "core", "coda", "final_norm") if s in sec_max)
    with open(path, "a") as f:
        f.write(
            f"  FWDNORM step={step} worstBlock={worst[0]}={worst[1]:.3e} | {sec_str}\n"
            f"  TERNFLIP step={step} total={flip_total} worst={flip_worst[0]}@{flip_worst[1]} | {flip_sec_str}\n"
        )


def warmup_compile_all_shapes(
    model, batch_size: int, seq_len: int, device, passes_per_size: int,
    tag: str = "startup",
    tul_rt=None,
) -> None:
    """Forced-depth fwd+bwd passes so EVERY compile variant builds NOW, not mid-loop.

    Forces the active-set to hit every sub-batch size (incl. n_active==1, the rare
    Poisson draw) in BOTH the no_grad prefix and the checkpointed BPTT window, so
    fwd AND bwd variants of every torch.compile guard-set and every hand-written
    Triton kernel size-specialization compile here. Shared by the thread-free
    startup window and the MORTAR/route phase-boundary recompile — see the two
    call sites for the (different) fork-safety reasoning at each.

    ``tul_rt``: required when the model is built with ``tul.tg_restrict`` — a TG model
    has NO plain-forward path (transformer.forward raises without a slot_layout, by
    design: docs/tul-tg-spec.md), so the warmup synthesizes a TUL batch with the SAME
    packer the loader uses and warms the REAL path instead. The forced-size loop is
    skipped there on purpose: it exists for the hand-written Triton kernels' per-size
    JIT specializations, and tg_restrict is use_kernels=false-only (no Triton on the
    path); the core MLPs are compiled dynamic=True, so slot-count variance needs no
    per-size passes either.
    """
    if getattr(model, "_tg_restrict", False):
        if tul_rt is None:
            raise RuntimeError("warmup for a tg_restrict model needs tul_rt (its packer)")
        from morph.model.tul_layout import pack_tul_batch
        spec = tul_rt.data_cfg.spec_for(seq_len)
        rule = tul_rt.data_cfg.rule
        print(f"  Warmup compile [{tag}] (TG path, {passes_per_size} packed TUL passes)...",
              flush=True)
        t0 = time.perf_counter()
        g = torch.Generator().manual_seed(0)
        # tul.loop_attn_center: these are train-mode forwards with grad on RANDOM token rows,
        # so the EMA of the slot loop's attention input must not learn from them (the
        # first update would set every mu_l to a random-token mean). Frozen = recorded and
        # discarded; a no-op on every model without the key.
        with model.loop_attn_center_frozen():
            for _ in range(passes_per_size):
                need = batch_size * (spec.l_total + 1)
                buf = torch.randint(0, model.cfg.vocab_size, (need + 8,),
                                    generator=g).tolist()
                ids, labs, layout = pack_tul_batch(buf, rule, spec, batch_size)
                ids, labs, layout = ids.to(device), labs.to(device), layout.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids, labels=labs, slot_layout=layout)
                out["loss"].backward()
                model.zero_grad(set_to_none=True)
                del ids, labs, layout, out
        torch.cuda.synchronize()
        print(f"  Warmup compile [{tag}] done in {time.perf_counter()-t0:.1f}s "
              f"({passes_per_size} TG passes)", flush=True)
        return

    mx = int(model.cfg.max_depth)
    # A depth_fixed model (LoopMTP and its *fixed controls) never has a partial active
    # set: every row runs max_depth iterations, so the ONE shape it will ever compile is
    # the full batch at max_depth. The forced-size ladder would build shapes the model
    # never runs and trips LoopMTP's constant-T guard (the 2026-09-14 smoke failure).
    fixed = bool(getattr(model.cfg, "depth_fixed", False))

    def _forced(K):
        if fixed:
            return torch.full((batch_size,), mx, device=device, dtype=torch.long)
        d = [1] * batch_size
        for j in range(min(K, batch_size)):
            d[j] = mx
        return torch.tensor(d, device=device, dtype=torch.long)

    orig_sample = model._sample_depths
    sizes = ([batch_size] if fixed else
             list(range(batch_size, 0, -1)))   # [B, B-1, ..., 1] — size>1 AND size==1
    print(f"  Warmup compile [{tag}] (active-set sizes {sizes} × {passes_per_size})...",
          flush=True)
    t0 = time.perf_counter()
    try:
        for K in sizes:
            model._sample_depths = (lambda _b, _dev, _K=K: _forced(_K))
            for _ in range(passes_per_size):
                ids = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len),
                                    device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids, labels=ids)
                out["loss"].backward()
                model.zero_grad(set_to_none=True)
                del ids, out
    finally:
        model._sample_depths = orig_sample          # restore real Poisson sampling
    torch.cuda.synchronize()
    print(f"  Warmup compile [{tag}] done in {time.perf_counter()-t0:.1f}s "
          f"({len(sizes) * passes_per_size} passes, all active-set sizes)", flush=True)


def build_step_mix_cycle(step_mix: dict) -> list[str]:
    """Deterministic ``tul_step_mode`` schedule from an integer-ratio dict
    (``training.step_mix``, e.g. ``{bptt: 1, db1: 1}``) — the faithful DiffusionBlocks
    interleave arm (``tul_ilv50``, CLAUDE.md).

    Returns a cycle of length ``sum(step_mix.values())``; the mode at global step
    ``s`` is ``cycle[s % len(cycle)]`` — a pure function of the step INDEX, so it is
    resume-safe (no RNG, no run-local counter) and independent of the seed.

    Uses the standard weighted-round-robin ("most uniform spread") construction
    rather than laying out all of one mode followed by all of the other: at 1:1 that
    means alternating ``bptt, db1, bptt, db1, …`` instead of a run of 50 followed by a
    run of 50, which would correlate a long block of optimizer steps with the same
    objective — bad for anything that assumes steps are roughly IID (e.g. AdEMAMix's
    slow EMA). Ties broken by key order in ``step_mix`` (the YAML's own order, which
    Hydra/OmegaConf preserve), so the construction is fully deterministic.
    """
    if not step_mix:
        raise ValueError("build_step_mix_cycle needs a non-empty step_mix dict")
    keys = list(step_mix.keys())
    counts = {k: int(v) for k, v in step_mix.items()}
    if any(c <= 0 for c in counts.values()):
        raise ValueError(f"training.step_mix ratios must be positive ints, got {step_mix}")
    total = sum(counts.values())
    cycle: list[str] = []
    produced = {k: 0 for k in keys}
    for i in range(total):
        best_k, best_score = None, None
        for k in keys:
            score = (i + 1) * counts[k] / total - produced[k]
            if best_score is None or score > best_score + 1e-12:
                best_k, best_score = k, score
        cycle.append(best_k)
        produced[best_k] += 1
    assert produced == counts, (produced, counts)   # the construction's own invariant
    return cycle


# ── Main training loop ────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Resolve paths ────────────────────────────────────────────────────
    tr = cfg.training
    data_cfg = cfg.data
    wb_cfg = cfg.wandb

    # ── Reproducibility: APPLY training.seed ─────────────────────────────
    # cfg.training.seed was logged to wandb but never applied — model init, Poisson
    # depth draws, dropout and TST all drew from an UNSEEDED default generator, so two
    # identical-config runs produced different trajectories (found by the perf-pass
    # bit-exactness gate, 2026-07-03; violates the reproducible-from-config rule).
    # Seeded before ANY tensor creation (model build, warmup randint, data).
    # ── Bit-reproducibility (opt-in) ──────────────────────────────────────
    # Measured 2026-08-23 (ignore/perf/phase1/attn_determinism.py): the fused CSA/HCA
    # attention backward is nondeterministic and is NOT reachable by PyTorch's determinism
    # machinery — with kernels on, 6/6 repeated backwards differ across 119 of 150
    # parameters at 2.4e-4 relative, deterministic mode or not. With `model.use_kernels=false`
    # AND this flag, the backward is bit-identical 0/6 across 0 of 150 parameters.
    #
    # Costs, measured on tul_a1: 2.28x fewer tokens/s (24780 -> 10845) and roughly half the
    # batch, because eager attention materialises what the kernels fuse (batch 12 OOMs;
    # batch 6 peaks at 18.6 GB). Use it for experiments that must bisect, not for production
    # runs. CUBLAS_WORKSPACE_CONFIG must be exported BEFORE the process starts — setting it
    # in-process is too late, and with warn_only the result is silently wrong rather than an
    # error, which is how a first attempt at this measurement produced garbage.
    if bool(getattr(tr, "deterministic", False)):
        if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            raise RuntimeError(
                "training.deterministic=true requires CUBLAS_WORKSPACE_CONFIG=:4096:8 to be "
                "exported BEFORE the process starts; setting it here would be too late and "
                "would give wrong results instead of an error.")
        # warn_only=False, not True (2026-10-04): under warn_only=True SDPA's memory-
        # efficient and flash backward KEEP their non-deterministic algorithms and only
        # warn. On the strict slot-loop arm (the span decoder's causal SDPA and the
        # register core) two identical 20-step runs then split at step 2. With
        # warn_only=False they select the deterministic backward and the two runs are
        # bit-identical (loss, grad norm, final state hash). An op with no deterministic
        # implementation now RAISES, which is the honest outcome for a mode whose whole
        # contract is reproducibility.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        _uk = bool(getattr(cfg.model, "use_kernels", True))
        print(f"  [determinism] use_deterministic_algorithms(True), cudnn.benchmark=False"
              + ("" if not _uk else
                 "\n  [determinism] WARNING: model.use_kernels=true — the fused attention "
                 "backward stays nondeterministic and this run will NOT be reproducible."))

    _seed = int(getattr(tr, "seed", 0))
    import random as _random
    _random.seed(_seed)
    torch.manual_seed(_seed)          # seeds CPU + all CUDA generators
    try:
        import numpy as _np_seed
        _np_seed.random.seed(_seed % 2**32)
    except ImportError:
        pass

    total_steps = int(tr.steps)
    batch_size = int(tr.batch_size)
    seq_len = int(data_cfg.seq_len)
    grad_clip = float(getattr(tr, "grad_clip", 1.0))
    eval_every = int(getattr(tr, "eval_every", 500))
    ckpt_every = int(getattr(tr, "ckpt_every", 2500))
    # Retention for the periodic `step_*.pt` checkpoints. 0 = keep every one, which is
    # UNBOUNDED: a 100k-step run at ckpt_every=2500 writes 40 files of ~2.3 GB = 90 GB,
    # and a four-seed 3500-step sweep at ckpt_every=500 wrote 63 GB. That is how
    # checkpoints/morph/ reached 292 GB on 2026-08-25. The rolling ring buffer below has
    # always rotated; this path never did. Only `step_*.pt` is rotated — whatever the
    # abort guards write (DIVERGED_*.pt, TAKEOVER_*.pt) is a normal file and is kept.
    _ck_ring = RetentionRing(int(getattr(tr, "ckpt_keep_last", 8)))
    # Rolling pre-onset capture (see the ring buffer in the training loop). 0 = off.
    _roll_every = int(getattr(tr, "ckpt_rolling_every", 0))
    _roll_ring = RetentionRing(max(1, int(getattr(tr, "ckpt_rolling_keep", 8))), tag="roll")
    gen_every = int(getattr(tr, "gen_every", 0))  # 0 = disabled
    n_eval_batches = int(getattr(tr, "n_eval_batches", 20))
    resume_path: Optional[str] = getattr(tr, "resume", None)
    init_from_path: Optional[str] = getattr(tr, "init_from", None)

    # ── Token-Superposition Training (TST, #231 — arXiv 2605.06546) ──────────
    # Two phases in ONE run: superposition (bag_size=s, multi-hot CE) for the first
    # tst_ratio·total_steps, then recovery (bag_size=0, standard NTP — code path
    # inactive). bag_size is a per-forward kwarg (eval/gen always use 0). The switch
    # is in-process: only the MLP submodules are compiled and they see [*, L, d] in
    # BOTH phases (bagging happens before the loop), so the switch triggers no
    # recompile. tst_bag_size=0 → bit-identical to the pre-TST baseline.
    tst_bag_size = int(getattr(tr, "tst_bag_size", 0))
    tst_ratio = float(getattr(tr, "tst_ratio", 0.0))
    # The phase BOUNDARIES are derived by PhaseSchedule, built below once total_steps is
    # FINAL. Deriving them here used training.steps, but the curriculum scheduler overrides
    # total_steps afterwards, so both the TST and the TUL boundary landed at the wrong step
    # on every curriculum run (morph/training/phase.py, defect 3).

    use_compile = bool(getattr(tr, "compile", True))
    compile_mode = str(getattr(tr, "compile_mode", "default"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model ───────────────────────────────────────────────────────
    # ORDERING IS LOAD-BEARING: model build + torch.compile + warmup run BEFORE
    # wandb.init() and the streaming dataloader. All Triton/Inductor compilation —
    # including the gcc subprocess fork that builds each kernel launcher — therefore
    # happens in a single-threaded process. A fork deadlocks only when a thread holds a
    # non-reentrant lock (glibc malloc arena) at fork time; with no wandb/httpx threads
    # alive yet, every compile fork is safe. The fused CCA kernels JIT-specialize size==1
    # separately; without this ordering, the first n_active==1 Poisson draw would compile
    # against live threads. wandb.init() is deferred to just after the warmup.
    # ── TUL (docs/tul-spec.md §8) ──────────────────────────────────────────
    # Resolved BEFORE the model build: `tul.activate_at: never` → tul_rt is None →
    # no TUL parameters are constructed and every path below is the baseline's,
    # bit-identical (runtime-invariants §6b). The parameters exist from step 0 when
    # TUL is configured but stay inert (grad None ⇒ the optimizer skips them) until a
    # forward is called with a layout, so a mid-run activation needs no optimizer
    # rebuild — only E_slot's re-init from the live embedding table (spec §5).
    from morph.training.flops import build_flop_model, perf_metrics
    from morph.training.tul_setup import build_tul_runtime
    from morph.training.fm_setup import build_fm_runtime
    from morph.training.phase import PhaseSchedule
    tul_rt = build_tul_runtime(cfg)
    fm_rt = build_fm_runtime(cfg, tul_rt)
    morph_cfg = build_morph_config(cfg, tul=tul_rt.model_cfg if tul_rt else None,
                                   fm=fm_rt)
    model = MORPHTransformer(morph_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params on {device}")

    # Analytic FLOP model (gate A3). Built from the LIVE module tree, so the weight-GEMM
    # half is exact for whatever config this run actually has.
    _flops = build_flop_model(model, cfg, seq_len=int(cfg.data.seq_len))
    print(f"  [flops] model v{_flops.version} nominal A0-shape proxy="
          f"{_flops.flop_proxy():.2f} passes/token, realized depth="
          f"{_flops.manifest()['flops/realized_depth']:.3f}")

    # Core-map spectral-norm penalty: soft hinge L=λ·Σ relu(σ_max(W)−cap)² over core MLP
    # linears. Binds module references (stable across compile/resume). penalty() calls
    # forward() at train time so the ternary STE is applied live. OFF when cap or lambda ≤ 0
    # (exact baseline, never constructed).
    # Core-map Jacobian probe. 0 (the default) never constructs it and never runs the
    # extra forward, so the training path is untouched. It costs one no-grad forward plus
    # ~2*power_iters backward passes over ONE core step, so it belongs at a coarse cadence.
    _jac_every = int(getattr(cfg.training, "jac_probe_every", 0))
    _jac_iters = list(getattr(cfg.training, "jac_probe_iters", [0]) or [0])
    # Onset capture knobs (lab/experiments/planned/2026-09-03-tul-onset-capture.md). All
    # off by default; each needs the loop probe (grad_probe_every > 0) to reach the row.
    _rank_every = int(getattr(cfg.training, "loop_rank_every", 0))
    _cot_probe = bool(getattr(cfg.training, "loop_cot_probe", False))
    _bdump_every = int(getattr(cfg.training, "batch_dump_every", 0))
    _bdump_dir = getattr(cfg.training, "batch_dump_dir", None) or None
    if (_rank_every or _cot_probe or _bdump_every) and int(getattr(cfg.training, "grad_probe_every", 0)) <= 0:
        raise ValueError("training.loop_rank_every / loop_cot_probe / batch_dump_every need "
                         "training.grad_probe_every > 0 (they are read by the pre-clip probe)")
    if _bdump_every > 0 and not _bdump_dir:
        raise ValueError("training.batch_dump_every > 0 needs training.batch_dump_dir")
    if _bdump_dir:
        os.makedirs(_bdump_dir, exist_ok=True)
    _jac_power = int(getattr(cfg.training, "jac_probe_power_iters", 200))
    # Core-map Jacobian probe — built only when a cadence is configured (0 = never), so
    # the default path allocates nothing.
    _jac_probe = None
    if _jac_every > 0:
        from morph.training.core_jacobian import CoreJacobianProbe
        _jac_probe = CoreJacobianProbe(model, n_iter=_jac_power, seed=0)
        print(f"  [jac] core-map Jacobian probe every {_jac_every} steps at core "
              f"iterations {_jac_iters}, {_jac_power} power iterations")

    _spec_pen = None
    _sp_cap = float(getattr(cfg.training, "spectral_penalty_cap", 0.0))
    _sp_lam = float(getattr(cfg.training, "spectral_penalty_lambda", 0.0))
    # σ_max is the quantity the looped core detonates on (CLAUDE.md, the iterative-map note),
    # and it was measurable ONLY from a checkpoint autopsy. Log it on EVERY run, penalised or
    # not: with lam=0 `penalty()` early-returns an exact zero, so a logging-only construction
    # leaves the arm bit-exact but no longer blind. 0 disables.
    # Phase-1 onset probe cadence (steps). 0 = off, and off is bit-exact: the model-side
    # flag is not set, so _tul_core traces the same graph it always has.
    # Core-takeover abort criterion (plan task 3.2). The shipped ppl guard struck at step
    # 2620 on the measured control, 587 steps after the takeover began; this fires at 2033.
    # 0.0 = off. The rule lives in morph/training/divergence_guard.py and is replayed
    # against a real diverging trajectory in tests/test_divergence_guard.py.
    _core_guard = CoreShareGuard(
        threshold=float(getattr(cfg.training, "abort_core_share", 0.0)),
        window=int(getattr(cfg.training, "abort_core_share_window", 50)),
        fraction=float(getattr(cfg.training, "abort_core_share_fraction", 0.3)),
        warmup=int(getattr(cfg.training, "abort_core_share_warmup", 200)),
    )
    # The mechanism criterion. Leads the share by ~500-600 steps on every run that took
    # over, and stays quiet on the one that recovered. See divergence_guard.py.
    _gain_guard = BlockGainGuard(
        threshold=float(getattr(cfg.training, "abort_block_gain", 0.0)),
        min_r2=float(getattr(cfg.training, "abort_block_gain_min_r2", 0.5)),
        window=int(getattr(cfg.training, "abort_block_gain_window", 200)),
        fraction=float(getattr(cfg.training, "abort_block_gain_fraction", 0.3)),
        warmup=int(getattr(cfg.training, "abort_core_share_warmup", 200)),
    )
    _gprobe_every = int(getattr(cfg.training, "grad_probe_every", 0))
    if (_core_guard.enabled or _gain_guard.enabled) and _gprobe_every != 1:
        # The guard reads the share the probe computes. Rather than let a config produce a
        # guard that silently never fires, turn the probe on and say so — it costs 0.5 %.
        print(f"  [guard] abort criteria need the per-step probe; forcing "
              f"grad_probe_every 1 (was {_gprobe_every})")
        _gprobe_every = 1
    _gprobe_path = getattr(cfg.training, "grad_probe_path", None) or None
    _gprobe_fh = None
    # ── HARD spectral projection (the cure; the soft hinge above is the one that can lose) ──
    # W <- W * min(1, cap/sigma) applied after each optimizer step. 0 disables and nothing is
    # constructed. See CoreSpectralProjection's docstring for why this and not the penalty.
    _spec_proj = None
    # gamma slow-EMA pairs (cusp-vault fix): collected ONCE; empty when off.
    from morph.model.ternary_qat import collect_scale_ema_stes, update_scale_emas
    _tern_ema_pairs = collect_scale_ema_stes(model)
    _spj_cap = float(getattr(cfg.training, "spectral_project_cap", 0.0))
    _sp_log = int(getattr(cfg.training, "spectral_penalty_log_every", 100))
    _sp_on = _sp_cap > 0.0 and _sp_lam > 0.0
    if _spj_cap > 0.0:
        from morph.training.spectral_penalty import CoreSpectralProjection
        _spec_proj = CoreSpectralProjection(
            model, cap=_spj_cap,
            n_iter=int(getattr(cfg.training, "spectral_project_n_iter", 2)),
            include_attn=bool(getattr(cfg.training, "spectral_penalty_include_attn", False)),
            verify=bool(getattr(cfg.training, "spectral_project_verify", False)))
        print(f"  Core spectral PROJECTION ON: cap={_spj_cap} "
              f"n_iter={_spec_proj.n_iter} verify={_spec_proj.verify} "
              f"on {len(_spec_proj._linears)} core linears "
              f"({_spec_proj._n_mlp} MLP + "
              f"{len(_spec_proj._linears) - _spec_proj._n_mlp} attention)")

    # A model with no core (arm TUL-A3, `model.n_core: 0`) has no core linears to
    # penalise or log, and `collect_core_linears` rightly REFUSES an empty enumeration
    # rather than run a silent no-op. That guard is correct for every model that HAS a
    # core; here zero is the configuration, not a broken enumeration. Skip the whole
    # block and say so, rather than weakening the guard for everyone else.
    # (A3 crashed on this at 10:10 on 2026-08-28: "found 0 core MLP linears".)
    if (_sp_on or _sp_log > 0) and int(getattr(cfg.model, "n_core", 0)) == 0:
        print("  Core spectral-norm penalty SKIPPED: model.n_core=0, so there are no core "
              "linears to penalise or log (arm TUL-A3's compute floor).")
    elif _sp_on or _sp_log > 0:
        from morph.training.spectral_penalty import CoreSpectralPenalty
        _sp_attn = bool(getattr(cfg.training, "spectral_penalty_include_attn", False))
        _spec_pen = CoreSpectralPenalty(model, cap=_sp_cap if _sp_on else 0.0,
                                        lam=_sp_lam if _sp_on else 0.0,
                                        n_iter=int(getattr(cfg.training, "spectral_penalty_n_iter", 1)),
                                        include_attn=_sp_attn)
        print(f"  Core spectral-norm penalty {'ON' if _sp_on else 'OFF (logging only)'}: "
              f"cap={_sp_cap} lambda={_sp_lam} log_every={_sp_log} include_attn={_sp_attn} "
              f"on {len(_spec_pen._linears)} core linears "
              f"({_spec_pen._n_mlp} MLP + {len(_spec_pen._linears) - _spec_pen._n_mlp} attention)")

    # ── step_mix interleave schedule (faithful DiffusionBlocks, CLAUDE.md) ─────
    # training.step_mix: {bptt: 1, db1: 1} → alternate tul_step_mode per step, a pure
    # function of the GLOBAL step index (build_step_mix_cycle above) so it survives a
    # resume unchanged. Absent (the default) → `_step_mix_cycle` is None and every
    # forward call passes `tul_step_mode=None`, bit-identical to before this existed.
    _step_mix_raw = getattr(cfg.training, "step_mix", None)
    _step_mix_cycle = (build_step_mix_cycle(dict(_step_mix_raw))
                       if _step_mix_raw else None)
    _step_mix_stats: dict[str, dict[str, float]] = {}   # mode -> {"n": int, "loss_sum": float}
    if _step_mix_cycle is not None:
        print(f"  step_mix ON: cycle={_step_mix_cycle} (from {dict(_step_mix_raw)})",
              flush=True)
        for _m in set(_step_mix_cycle):
            _step_mix_stats[_m] = {"n": 0, "loss_sum": 0.0}

    # ── Quantization / QAT ─────────────────────────────────────
    # Ternary → embedding → CMS scoring → attention-projection → FP8, in that order
    # (each step checks disjointness against the previous). MUST run BEFORE
    # torch.compile and BEFORE create_optimizer — see quant_setup.apply_quantization.
    # Shared with the checkpoint-loading path: these transforms rename tensors in
    # state_dict, so anything rebuilding a trained model must apply the SAME ones.
    from morph.training.quant_setup import apply_quantization
    _qm = apply_quantization(model, cfg)
    ternary_manifest = _qm["ternary"]
    embed_quant_manifest = _qm["embed_quant"]
    attn_proj_quant_manifest = _qm["attn_proj_quant"]
    fp8_manifest = _qm["fp8"]

    # ── tul.fan_opf / tul.fan_route="latent" (2026-09-30): the EMA target twin ─────────
    # Built HERE: AFTER quantisation, so the twin's prelude carries the same ternary
    # parametrisation as the live one (a target computed by a different function would be
    # a different target), and BEFORE torch.compile, so the twin's blocks stay eager and
    # the compiled live MLPs are not copied. The compile warmup below then runs with a
    # twin in place. After the weights load, `tul_fan_target_sync` re-syncs it (resume: the
    # checkpoint's EMA state; otherwise the loaded live weights). No-op on other models.
    if model.tul_fan_target_build():
        _nt = sum(p.numel() for p in model.__dict__["_fan_target"].parameters())
        print(f"  [fan-target] EMA prelude twin built: {_nt/1e6:.1f}M parameters (the "
              f"prelude blocks + their x0/value-embedding projections; the lookup tables "
              f"are shared), frozen, eval mode, momentum "
              f"m={model.cfg.tul.fan_target_ema}", flush=True)

    # ── tul.latent_pre_target (stage-1 latent pretraining, 2026-10-02) ────────────────
    # The "plain_*" targets read a FROZEN plain model built from its own config and
    # checkpoint (morph/training/latent_pre_ref.py). Built before the compile warmup,
    # whose labelled forwards already compute the target. Stage 1 has no trained coda, so
    # a generation test would decode from random coda weights: refused, not skipped.
    _lp_mode = str(getattr(model, "_latent_pre_mode", "off"))
    _lp_cal: dict | None = None        # the stage-1 target calibration's statistics
    if _lp_mode != "off":
        if gen_every > 0 or bool(getattr(tr, "gen_test", False)):
            raise ValueError(
                f"tul.latent_pre_target={_lp_mode!r} (stage 1) with training.gen_every="
                f"{gen_every} / gen_test={bool(getattr(tr, 'gen_test', False))}: a stage-1 "
                f"model has no trained coda and builds no logits. Set gen_every: 0.")
        if _lp_mode in ("plain_prelude", "plain_final"):
            from morph.training.latent_pre_ref import build_latent_pre_ref
            _lp_ref = build_latent_pre_ref(model.cfg.tul, device, int(model.cfg.d_model))
            model.tul_latent_pre_attach_ref(_lp_ref)
            if model.cfg.tul.latent_pre_target_norm == "standard":
                # The FIXED target statistics, once, before the warmup's labelled forwards
                # (which already standardise). The batches come from a separate PROCESS
                # (fork-safe here, and the run's own stream is untouched).
                from morph.training.latent_pre_ref import (calibrate_latent_pre_ref,
                                                           calibration_batches)
                _t_cal = time.perf_counter()
                _cal_b, _cal_info = calibration_batches(
                    data_cfg.tokenizer, data_cfg.dataset, seq_len, batch_size,
                    tul_rt.data_cfg, int(model.cfg.tul.latent_pre_cal_doc_offset),
                    int(model.cfg.tul.latent_pre_cal_batches),
                    run_tokens=total_steps * batch_size * seq_len)
                _lp_cal = calibrate_latent_pre_ref(_lp_ref, _lp_mode, _cal_b, device)
                del _cal_b
                _lp_cal.update(_cal_info)
                print(f"  [latent-pre] target standardised with FIXED stats from "
                      f"{_lp_cal['n']} target slots of training documents "
                      f"{_lp_cal['doc_first']}..{_lp_cal['doc_end']} (the run reads ~"
                      f"{_lp_cal['est_run_docs']}; val starts at 50000): "
                      f"|mu|={_lp_cal['mu_norm']:.3f} sigma median="
                      f"{_lp_cal['sigma_median']:.5f} min={_lp_cal['sigma_min']:.6f} "
                      f"({_lp_cal['n_floored']} floored); participation ratio "
                      f"LN {_lp_cal['pr_ln']:.2f} -> standardised "
                      f"{_lp_cal['pr_standard']:.2f}; {time.perf_counter() - _t_cal:.1f}s",
                      flush=True)

    # Phase-1 onset probe: arm the model-side half (the looped-core state collector in
    # TULTransformer._tul_core). Set before the first forward. Left unset — the default —
    # _tul_core takes the identical code path it always has.
    if _gprobe_every > 0:
        model._probe_loop = True
        # The cotangent hook is a per-forward decision made at trace time in _tul_core;
        # set it on the ROOT so a compiled wrapper cannot shadow it.
        getattr(model, "_orig_mod", model)._probe_cot = _cot_probe
        if _cot_probe or _rank_every or _bdump_every:
            print(f"  [capture] loop_cot_probe={_cot_probe} loop_rank_every={_rank_every} "
                  f"batch_dump_every={_bdump_every} dir={_bdump_dir}")
        if _gprobe_path:
            os.makedirs(os.path.dirname(os.path.abspath(_gprobe_path)), exist_ok=True)
            _gprobe_fh = open(_gprobe_path, "a", buffering=1)
        print(f"  [probe] pre-clip gradient + loop probe ON, every {_gprobe_every} step(s)"
              + (f", mirrored to {_gprobe_path}" if _gprobe_path else ""))

    # ── torch.compile ─────────────────────────────────────────────────────
    # Compile only the MLP sub-modules (attention uses Triton/SDPA kernels,
    # which are incompatible with fullgraph compile).
    compile_attention = bool(getattr(tr, "compile_attention", False))
    compile_blocks = bool(getattr(tr, "compile_blocks", False))
    # training.compile_core_dynamic (default true = the tree before the key): compile the
    # core's modules with a dynamic batch. The slot loop runs every pass on the FULL cell
    # axis (inactive slots are masked by `torch.where`, not dropped), so its shapes are
    # static and dynamic guards only cost host time per call: 22 ms per step on the lead
    # slot-loop arm with compile_blocks (2026-10-04). A model whose core shrinks its
    # active set (the plain Parcae path) wants the default.
    core_dynamic = bool(getattr(tr, "compile_core_dynamic", True))
    if use_compile:
        for group in [model.prelude, model.core, model.coda]:
            # Core MLPs see a VARIABLE batch each loop iteration (active-set
            # shrinking processes the still-active prefix), so compile them with
            # dynamic batch to avoid a recompile per distinct sub-batch size.
            # Prelude/coda see a fixed batch → let Dynamo auto-decide (None).
            dyn = (True if core_dynamic else False) if group is model.core else None
            for i in range(len(group)):
                layer = group[i]
                if compile_blocks:
                    # Opt-in (training.compile_blocks): compile the WHOLE MORPHBlock —
                    # norms, HC residual mixing, injections, and casts included. The
                    # eager step is launch-bound (self-CPU 820 ms vs self-CUDA 371 ms
                    # per step; lab/perf/compile_coverage_bench.py), and block compile
                    # measured 1.84× fwd+bwd over eager, 1.64× over the mlp+attention
                    # compile, at the SAME numeric drift class as mlp+attention. The
                    # GLA retention branch stays a graph break (the same eager code
                    # runs): Inductor hits an upstream SplitScan codegen bug on its
                    # chunked cumsum. Eager-only path (use_kernels=false) — the fused
                    # Triton kernels are not compilable and do not need this.
                    if getattr(layer, "retention", None) is not None:
                        layer.retention.forward = torch.compiler.disable(
                            layer.retention.forward)
                    group[i] = torch.compile(layer, mode=compile_mode, dynamic=dyn)
                    continue
                if hasattr(layer, "mlp"):
                    layer.mlp = torch.compile(layer.mlp, mode=compile_mode, dynamic=dyn)
                # Opt-in (training.compile_attention): compile the eager attention too.
                # Only meaningful when use_kernels=false (the TUL/TG path) — the fused
                # Triton attention is not compilable and does not need this. Shapes on
                # the TUL path are static per region (tokens [B,L]; slots [B,S]), so
                # let Dynamo auto-decide guards.
                if compile_attention and hasattr(layer, "attention"):
                    layer.attention = torch.compile(layer.attention, mode=compile_mode,
                                                    dynamic=dyn)
        if compile_blocks:
            print(f"  BLOCKS compiled (mode={compile_mode}, core dynamic-batch, "
                  "GLA graph-broken)")
        else:
            print(f"  MLPs compiled (mode={compile_mode}, core dynamic-batch)"
                  + (", attention compiled" if compile_attention else ""))

        # ── Warmup compile — runs in the THREAD-FREE window (pre-wandb, pre-dataloader) ──
        # Two compilation systems fork subprocesses here and must finish before any thread
        # spawns: (a) torch.compile/Inductor for the MLPs, which lazily compiles a variant
        # per (sub-batch size × grad_mode × autocast dtype) guard on the first few forwards;
        # (b) the hand-written fused CCA Triton kernels, which JIT-specialize size==1 apart
        # from size>1. Either, if compiled DURING the training loop, forks gcc/Inductor
        # workers while the HF-streaming httpx + wandb threads hold a glibc malloc-arena lock
        # → the forked child deadlocks in __triton_launcher.c (intermittent; cost us a night
        # — Ai-notes 06-01-2026/MORPH-eval-recompile-hang). Mitigation is layered: (1) build +
        # warm up BEFORE wandb.init/dataloader so the fork window is single-threaded → safe by
        # construction; (2) the forced-size loop below compiles EVERY Triton variant (incl.
        # the rare size==1) here, so none JIT-compiles at runtime; (3) eager_on_recompile
        # (set after warmup) catches any leftover MLP guard → runs it eager (no compile, no
        # fork) rather than recompiling mid-loop. Raise the Dynamo cache limit so all variants
        # coexist without eviction.
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = max(getattr(_dynamo.config, "cache_size_limit", 8), 64)
        _dynamo.config.accumulated_cache_size_limit = max(
            getattr(_dynamo.config, "accumulated_cache_size_limit", 256), 512)

        # Force the active-set to hit EVERY sub-batch size (incl. n_active==1) so all
        # Triton kernel variants compile HERE, in this thread-free window. The fused CCA
        # attention kernels are hand-written Triton (NOT torch.compile), so the stance
        # below does NOT govern them — Triton JIT-specializes size==1 separately from
        # size>1. If the size-1 variant is left to compile on the first runtime n_active==1
        # (a rare Poisson draw with one sequence far deeper than the rest), it forks gcc for
        # its launcher stub while wandb/httpx threads hold the glibc malloc-arena lock → the
        # forked child deadlocks (the intermittent step-0 wedge; py-spy caught the autograd
        # engine blocked mid-recompute on the next malloc). Patterns: K sequences at
        # max_depth, rest at depth 1 → n_active==K in BOTH the no_grad prefix and the
        # checkpointed BPTT window, so fwd AND bwd Triton variants for every size compile now.
        warmup_compile_all_shapes(
            model, int(cfg.training.batch_size), seq_len, device,
            int(getattr(tr, "warmup_passes_per_size", 4)), tag="startup thread-free",
            tul_rt=tul_rt,
        )

        # Final safety net: forbid NEW compilation during the training loop. The warmup
        # above + the @torch.compiler.disable on the CMS stats hook cover the COMMON shape
        # space (verified: 100 steps, 0 recompiles), so this rarely fires. But a rare
        # Poisson-depth draw can still produce an (size × grad_mode × dtype) combo the
        # warmup missed — and any such recompile would fork gcc for the Triton launcher in
        # the MAIN process (NOT covered by the spawn worker pool) against wandb/httpx
        # threads → intermittent deadlock (this bit the real campaign at step 1 while
        # 160 diag steps ran clean — pure timing luck). "eager_on_recompile" makes that
        # rare uncovered shape run EAGER (no compile, no fork, no deadlock) instead — one
        # slightly-slow step, never a hang. Common shapes keep their compiled kernels.
        torch.compiler.set_stance("eager_on_recompile")
        print("  torch.compiler stance = eager_on_recompile (rare uncovered shapes run eager, never recompile/fork)", flush=True)

    # ── W&B init — log FULL config dict ──────────────────────────────────
    # DEFERRED until AFTER the warmup: the compile/gcc-fork window above must be
    # thread-free (no wandb asyncio/httpx threads) for the fork to be deadlock-safe.
    # OmegaConf → plain Python dict so wandb can serialise it; fold n_params in directly.
    full_config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    full_config_dict["n_params"] = n_params
    # Everything DERIVED from the tul block (resolved ids, |B|, L_total) goes in the run
    # config too, so an arm is reproducible from wandb alone without re-deriving ids from
    # a tokenizer version. None when TUL is off.
    full_config_dict["tul_manifest"] = tul_rt.manifest if tul_rt else None
    # Everything DERIVED about FM1 — the planner shapes, the analytic loss scale the fm
    # term is divided by, the param count — so a run is reproducible from its wandb
    # config alone and "what source_std did we try?" is greppable.
    if fm_rt is not None:
        _mdl = getattr(model, "_orig_mod", model)
        full_config_dict["fm_manifest"] = {
            **fm_rt.manifest(),
            "fm/planner_params": int(sum(p.numel() for p in _mdl.fm_planner.parameters())),
            "fm/loss_scale_value": float(_mdl._fm_loss_scale),
            "fm/d_target": int(_mdl.fm_planner.cfg.d_target),
        }
    else:
        full_config_dict["fm_manifest"] = None
    # The FLOP model's own version + per-region costs: an MFU from model v1 is not
    # comparable to one from v2, and without this nobody can tell them apart later.
    full_config_dict["flops_manifest"] = _flops.manifest()
    if ternary_manifest is not None:
        # Derived ternary facts (scope/threshold already live in cfg). Drop the
        # verbose module_names list from the logged config; keep the greppable counts.
        full_config_dict["ternary_manifest"] = {
            k: v for k, v in ternary_manifest.items() if k != "module_names"
        }
    if fp8_manifest is not None:
        full_config_dict["fp8_manifest"] = {
            "scope": fp8_manifest["scope"], "recipe": fp8_manifest["recipe"],
            "min_dim": fp8_manifest["min_dim"], "n_converted": fp8_manifest["n_converted"],
        }
    if embed_quant_manifest is not None:
        full_config_dict["embed_quant_manifest"] = {
            k: v for k, v in embed_quant_manifest.items()
            if k not in ("module_names", "lm_head_note")
        }
    if attn_proj_quant_manifest is not None:
        # Keep the greppable counts/bits; drop the verbose per-module name list.
        full_config_dict["attn_proj_quant_manifest"] = {
            k: v for k, v in attn_proj_quant_manifest.items() if k != "module_names"
        }
    # ── Dataset manifests + content hashes for every curriculum blend source ──
    # The logged config must fully identify the DATA, not just the hyperparams
    # (PLAN.md: "W&B logs ... dataset manifests/hashes"). Missing shard files are
    # a hard error HERE, before wandb.init — never config around a missing shard.
    _curr_mf = getattr(cfg, "curriculum", None)
    if _curr_mf is not None and getattr(_curr_mf, "enabled", False):
        import hashlib as _hashlib
        import json as _json
        _manifests = {}
        for _src_name, _src_w in dict(_curr_mf.blend).items():
            if float(_src_w) <= 0:
                continue
            _sdir = os.path.join(str(_curr_mf.pretok_dir), str(_src_name))
            _meta_p = os.path.join(_sdir, "meta.json")
            _bin_p = os.path.join(_sdir, "tokens.u16.bin")
            if not (os.path.isfile(_meta_p) and os.path.isfile(_bin_p)):
                raise FileNotFoundError(
                    f"curriculum source {_src_name!r}: shard files missing under {_sdir}"
                )
            with open(_meta_p, "rb") as _f:
                _meta_bytes = _f.read()
            _meta = _json.loads(_meta_bytes)
            _st = os.stat(_bin_p)
            # Full sha256 of the token bin, cached in a sidecar keyed by
            # (size, mtime_ns) so the ~seconds cost is paid once per shard build.
            _sidecar = os.path.join(_sdir, "tokens.sha256")
            _bin_sha = None
            if os.path.isfile(_sidecar):
                try:
                    _key, _sha = open(_sidecar).read().split()
                    if _key == f"{_st.st_size}:{_st.st_mtime_ns}":
                        _bin_sha = _sha
                except (ValueError, OSError):
                    pass
            if _bin_sha is None:
                _h = _hashlib.sha256()
                with open(_bin_p, "rb") as _f:
                    for _chunk in iter(lambda: _f.read(1 << 24), b""):
                        _h.update(_chunk)
                _bin_sha = _h.hexdigest()
                try:
                    with open(_sidecar, "w") as _f:
                        _f.write(f"{_st.st_size}:{_st.st_mtime_ns} {_bin_sha}")
                except OSError:
                    pass  # read-only shard dir: hash still logged, just not cached
            _manifests[str(_src_name)] = {
                "weight": float(_src_w),
                "n_docs": _meta.get("n_docs"),
                "n_tokens": _meta.get("n_tokens"),
                "role": _meta.get("role"),
                "meta_sha256": _hashlib.sha256(_meta_bytes).hexdigest(),
                "tokens_sha256": _bin_sha,
                "path": os.path.realpath(_sdir),
            }
        full_config_dict["dataset_manifests"] = _manifests
        print(f"  [data] dataset manifests hashed for {len(_manifests)} sources "
              f"→ wandb config", flush=True)
    # A resume path that does not exist is a HARD failure, never a silent fresh start.
    # Every `if resume_path and os.path.isfile(resume_path):` below skips its whole block
    # when the file is missing, so without this check the run trains from step 0 with the
    # caller believing it continued. Measured 2026-09-18 on `tul-code-target-ce-uf`: the
    # queue line named a step_20000.pt that was never written (that arm was stopped at
    # 15000), the trainer started at step 0, and `training.train_only` then froze 151.6M
    # parameters at their RANDOM initialisation — the prelude and the tied embedding table
    # among them — so the model could not learn at all. It read loss 10.10 at step 1000
    # against ln(49152) = 10.80 and would have held the GPU for 4.5 more hours. The run
    # log's only tell was the ABSENCE of the "Resumed model+scaler+RNG" line.
    if resume_path and not os.path.isfile(resume_path):
        raise FileNotFoundError(
            f"training.resume={resume_path!r} does not exist. Asking to resume and then "
            f"starting from step 0 is never what the caller meant, and with "
            f"training.train_only set it silently freezes a RANDOM front. Point at a "
            f"checkpoint that exists, or drop training.resume to start fresh on purpose.")
    # Resume the same wandb run (continuous metric history) when resuming a checkpoint:
    # the prior run wrote its id to a wandb_id.txt sidecar next to its checkpoints.
    _wandb_resume_id = None
    if resume_path and os.path.isfile(resume_path):
        _sidecar = os.path.join(os.path.dirname(resume_path), "wandb_id.txt")
        if os.path.isfile(_sidecar):
            _lines = [ln.strip() for ln in open(_sidecar).read().splitlines()]
            _wandb_resume_id = (_lines[0] if _lines else "") or None
            # A FORK (a different wandb.name than the run that wrote the checkpoint) is a
            # new experiment and gets its own wandb run; only a resume under the same name
            # continues the history (2026-09-15: a fork silently extended its parent's run).
            _parent_name = _lines[1] if len(_lines) > 1 else None
            _this_name = getattr(wb_cfg, "name", None)
            if _wandb_resume_id and _parent_name and _this_name and _this_name != _parent_name:
                print(f"  [wandb] fork: checkpoint written by run {_wandb_resume_id} "
                      f"({_parent_name!r}); this run is {_this_name!r} -> NEW wandb run",
                      flush=True)
                _wandb_resume_id = None
            elif _wandb_resume_id:
                print(f"  [wandb] resuming run id {_wandb_resume_id}", flush=True)
    wandb.init(
        project=wb_cfg.project,
        entity=getattr(wb_cfg, "entity", None),
        name=getattr(wb_cfg, "name", None),
        config=full_config_dict,
        id=_wandb_resume_id,
        resume=("allow" if _wandb_resume_id else None),
        settings=wandb.Settings(_service_wait=60),
    )
    if _lp_cal is not None:
        # Logged ONCE: the fixed statistics of the standardised stage-1 target (summary,
        # and the config, so a run's calibration is greppable beside its keys).
        _lp_cal_log = {f"latent_pre/target_{k}": v for k, v in _lp_cal.items()}
        wandb.run.summary.update(_lp_cal_log)
        wandb.config.update({"latent_pre_calibration": dict(_lp_cal)}, allow_val_change=True)

    # ── training.train_only: freeze everything outside the listed prefixes (before the
    # optimizer is built, which skips parameters that need no gradient) ────────────────
    _train_only = list(getattr(cfg.training, "train_only", None) or [])
    if _train_only:
        from morph.training.freeze import apply_train_only
        _nt, _nf, _groups = apply_train_only(model, _train_only)
        print(f"  [train_only] {_nt/1e6:.1f}M trainable / {_nf/1e6:.1f}M frozen; trainable groups: "
              + ", ".join(f"{k}({v/1e6:.2f}M)" for k, v in _groups.items()), flush=True)
        full_config_dict["train_only_groups"] = dict(_groups)
    assert_code_target_front_frozen(model)

    # ── training.frozen_eval: the FROZEN model runs in eval mode (LXTUL-E Stage 0) ──────
    # `train_only` freezes PARAMETERS; this freezes BEHAVIOUR. With it on, every place the
    # loop below would call `model.train()` calls `_train_mode()` instead: the whole tree
    # goes to eval (no dropout, no token-state dropout, the eval slot depth, no
    # training-only loop terms) and only the modules the train_only prefixes name go back
    # to train. So the head is fitted on the deterministic exit state the val pass and the
    # offline scorer read. A per-step check below refuses a forward with the root in train
    # mode, so no path can leave it there silently.
    _frozen_eval = bool(getattr(cfg.training, "frozen_eval", False))
    if _frozen_eval and not _train_only:
        raise ValueError("training.frozen_eval needs training.train_only: with nothing "
                         "trained there is no module to leave in train mode")
    _frozen_root = getattr(model, "_orig_mod", model)

    def _train_mode() -> None:
        if _frozen_eval:
            from morph.training.freeze import apply_frozen_eval
            apply_frozen_eval(model, _train_only)
        else:
            model.train()

    if _frozen_eval:
        from morph.training.freeze import apply_frozen_eval
        _fe_mods = apply_frozen_eval(model, _train_only)
        print(f"  [frozen_eval] model in EVAL mode; {len(_fe_mods)} module(s) in train mode "
              f"under {list(_train_only)}", flush=True)

    # ── Optimizer + LR schedule ───────────────────────────────────────────
    optimizer = create_optimizer(model, cfg)
    lr_fn = create_lr_schedule(cfg)
    scaler = torch.amp.GradScaler("cuda")

    # ── Pruning schedule ──────────────────────────────────────────────────
    pruning = PruningSchedule.from_cfg(cfg)

    # ── Data loaders (train + val share the same generator; val uses a
    #    separate iterator so they don't pollute each other) ────────────────
    tokenizer_name = data_cfg.tokenizer
    dataset_name = data_cfg.dataset

    # Data-runtime knobs (shared with the curriculum loader; env MORPH_DATA_* overrides).
    from morph.training.data_placement import DataRuntimeConfig, Prefetcher
    _data_rt = DataRuntimeConfig.resolve(getattr(cfg, "data_runtime", None))

    def _make_train_loader(bag: int, skip_batches: int = 0, tul_on: bool = False):
        it = iter(create_dataloader(tokenizer_name, dataset_name, seq_len,
                                    batch_size, split="train", bag_size=bag,
                                    tul=tul_rt.data_cfg if (tul_rt and tul_on) else None))
        # Resume: the stream is DETERMINISTIC and UNSHUFFLED (fixed shard order, no per-epoch
        # seed), so replaying `skip_batches` next() calls advances to the EXACT batch the
        # interrupted run would serve next — "like nothing happened" for data too. Cost is
        # re-tokenizing the skipped prefix (CPU, ~1-2 min for a few-k-step resume); logged.
        if skip_batches > 0:
            print(f"  [data] fast-forwarding train stream by {skip_batches} batches "
                  f"(deterministic replay to exact resume position)…", flush=True)
            t_ff = time.perf_counter()
            for _ in range(skip_batches):
                next(it)
            print(f"  [data] fast-forward done in {time.perf_counter() - t_ff:.1f}s", flush=True)
        # Background prefetch: the in-process HF tokenization is a measured 35ms/step
        # CPU stall at bag=0 and 130-163ms at bag=6 (perf pass 2026-07-03), fully serial
        # with GPU work. Producing the SAME stream ahead on a thread is bit-identical —
        # and unlike the curriculum loader there is no rebuild-discard caveat here: this
        # generator is infinite (no StopIteration mid-phase) and a TST-switch rebuild
        # abandons the old stream wholesale, so no consumed batch ever differs.
        # prefetch_batches=0 (data_runtime / MORPH_DATA_PREFETCH) → synchronous as before.
        if _data_rt.prefetch_batches > 0:
            it = Prefetcher(it, depth=_data_rt.prefetch_batches, name=f"owt-train(bag={bag})",
                            pin=torch.cuda.is_available())
        return it

    # val/gen ALWAYS use standard NTP (bag_size=0) so val ppl is comparable to the
    # baseline regardless of which TST phase training is in.
    # INVARIANT 6: whenever the TUL layout is ACTIVE, val runs with it on and bag_size 0
    # (val ppl is over token positions only, so it stays comparable to the baseline).
    # Before a mid-run activation the model is still plain MORPH, so val is plain too;
    # _make_val_loader is called again at the switch.
    # Under a curriculum with `curriculum.val_source`, validation is the corpus's own
    # held-out shard instead of `data.dataset`'s split: `_curr_val_batches` is set in the
    # curriculum block below and the eval sites rebuild the loader before every eval, so
    # every eval (and every arm) scores the SAME held-out docs.
    _curr_val_batches = None

    def _make_val_loader(tul_on: bool):
        if _curr_val_batches is not None:
            return _curr_val_batches(tul_on)
        return iter(
            create_dataloader(tokenizer_name, dataset_name, seq_len, batch_size,
                              split="validation", skip_samples=50_000,
                              tul=tul_rt.val_data_cfg if (tul_rt and tul_on) else None)
        )

    # val_loader is built below, from the SAME PhaseSchedule the training loop reads.
    # It used to ask `tul_rt.activate_at == 0.0` while the live flag asked
    # `start_step >= tul_step` — two predicates for one question (phase.py, defect 1).

    # ── Checkpoint dir ────────────────────────────────────────────────────
    # wandb only auto-generates a run NAME online; offline runs with `wandb.name: null`
    # leave it None, which used to make this join() raise. Fall back to the run id so
    # every offline run still gets its own checkpoint directory.
    _run_tag = "run"
    if wandb.run is not None:
        _run_tag = wandb.run.name or str(wandb.run.id) or "run"
    ckpt_dir = os.path.join(_MORPH_ROOT, "checkpoints", "morph", _run_tag)
    if _lp_mode in ("plain_prelude", "plain_final"):
        # The stage-1 target must be FROZEN: never a checkpoint this run writes.
        from morph.training.latent_pre_ref import assert_not_live
        assert_not_live(str(model.cfg.tul.latent_pre_ref_ckpt), ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Seed the retention ring from what is already on disk, so a RESUMED run enforces
    # ckpt_keep_last over the whole run and not just over the checkpoints this process
    # happens to write. Sorted by step number, not lexically: step_900 precedes step_1000.
    if _ck_ring.enabled:
        _ck_ring.seed(existing_step_checkpoints(ckpt_dir))
        print(f"  [ckpt] retention: keeping the last {_ck_ring.keep} step_*.pt "
              f"({len(_ck_ring.paths)} already on disk). "
              f"Set training.ckpt_keep_last=0 to keep all.", flush=True)
    else:
        print("  [ckpt] retention: OFF (ckpt_keep_last=0) — step_*.pt will accumulate "
              "without bound.", flush=True)
    # Persist the wandb run id so a future resume continues the same run (read back as
    # the wandb_id.txt sidecar above, before wandb.init).
    if wandb.run is not None:
        try:
            with open(os.path.join(ckpt_dir, "wandb_id.txt"), "w") as _f:
                # line 1 the id, line 2 the run name: a resume under ANOTHER name is a
                # fork and starts its own wandb run (read back before wandb.init).
                _f.write(f"{wandb.run.id}\n{wandb.run.name or ''}\n")
        except OSError as e:
            print(f"  [wandb] could not write run-id sidecar ({e}); resume will start a new run")

    # Generation samples go to a sidecar file, NOT stdout. Generated text is
    # uncontrolled token output that can contain substrings ("RuntimeError:",
    # "Killed", "Traceback...") which would false-trigger a log-scraping watcher
    # (ignore/ab_watch.sh) into reporting a crash. Stdout gets only a safe summary.
    gen_samples_path = os.path.join(ckpt_dir, "generation_samples.txt")

    def _emit_gen(label: str, gen_text: str) -> None:
        with open(gen_samples_path, "a") as _f:
            _f.write(f"\n===== {label} =====\n{gen_text}\n")
        print(f"  [GEN {label}] {len(gen_text)} chars → {gen_samples_path}", flush=True)

    # ── Optional resume (FULL: model+topology+optimizer+scaler+RNG+step) ────
    # `training.resume_plain_to_tul` — the plain → TUL bootstrap. Off (the default) both
    # loaders behave exactly as they did; on, the two structural classes a plain seed
    # cannot match are allowed and listed, and every other homeless tensor raises. See
    # `classify_plain_to_tul_keys` and morph/configs/tul_slot_strict_bootstrap.yaml.
    _plain_to_tul = bool(getattr(tr, "resume_plain_to_tul", False))
    start_step = 0
    _e_slot_from_ckpt = False   # an init_from seed that carried a trained tul.E_slot
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        start_step, _opt_state, _needs_rebuild, _ckpt_pnames = load_checkpoint(
            resume_path, model, scaler, device, pruning,
            allow_plain_to_tul=_plain_to_tul)
        if _needs_rebuild:
            # Carve/route changed the param set → the dense optimizer built above is stale.
            # Free it (bnb keeps optimizer↔state↔param ref-cycles → explicit clear+gc, same
            # pattern as the in-loop phase-boundary rebuild) and rebuild on the NOW-
            # reconstructed (carved/routed) topology so its state keys line up before load.
            optimizer.zero_grad(set_to_none=True)
            if hasattr(optimizer, "state"):
                optimizer.state.clear()
            optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
            optimizer = create_optimizer(model, cfg)
            _lr0 = lr_fn(start_step)
            for pg in optimizer.param_groups:
                pg["lr"] = _lr0 * pg.get("lr_mult", 1.0)
            print("  [opt] rebuilt optimizer on reconstructed (carved/routed) topology",
                  flush=True)
        # Restore momentum/variance. HARD-FAIL on mismatch — swallowing it here would
        # silently continue training with ZERO momentum (the theater this whole task kills).
        # EXCEPTION — resume_fresh_optimizer=True (FORK-continue / new experiment off a ckpt):
        # deliberately start with a FRESH optimizer (no inherited momentum). This is the correct
        # choice for an A/B fork (both arms start identical; no stale 50k slow-EMA confound) and
        # it also sidesteps the bnb-8bit per-param "step"-key restore path. Topology + weights +
        # RNG are still restored above; ONLY the optimizer STATE is intentionally dropped.
        if bool(getattr(cfg.training, "resume_fresh_optimizer", False)):
            print("  [opt] resume_fresh_optimizer=True → FRESH optimizer (momentum starts at "
                  "zero; topology+weights+RNG restored). Fork-continue mode.", flush=True)
        else:
            # torch's load_state_dict replaces param_groups WHOLESALE with the saved
            # ones, hyperparameters included. Every optimizer setting a resume passes on
            # the command line is therefore silently reverted to whatever the checkpoint
            # was written with, and only `lr` survives because the scheduler rewrites it
            # each step. Found 2026-08-24: an arm resuming with ademamix_alpha_cap=1.0
            # against a checkpoint written at 3.5 came out BIT-IDENTICAL to the control.
            #
            # So snapshot the hyperparameters of the freshly-built (config-derived)
            # optimizer, let load_state_dict bring in the moment/EMA tensors, then put the
            # configured hyperparameters back and say which ones moved.
            _cfg_hp = [{k: v for k, v in g.items() if k != "params"}
                       for g in optimizer.param_groups]
            # Re-index by NAME when the live model has parameters the checkpoint lacks.
            # torch matches saved state to live parameters by POSITION, and an added module
            # inserts its parameters wherever it sits in `named_parameters()` — for SCSE
            # that is the middle of the decay group — which shifts every later index and
            # pairs parameters with other parameters' moments. torch only raises when the
            # group SIZES differ; that raise is the lucky case, and it is what stopped the
            # first SCSE screen on 2026-08-25.
            _opt_state, _added = align_optimizer_state(
                _opt_state, model, _ckpt_pnames,
                injection_lr_mult=float(getattr(cfg.training, "injection_lr_mult", 1.0)))
            if _added:
                print(f"  [opt] {len(_added)} parameters are new since this checkpoint and "
                      f"start with fresh optimizer state: "
                      + ", ".join(_added[:4]) + (" …" if len(_added) > 4 else ""),
                      flush=True)
            optimizer.load_state_dict(_opt_state)
            _changed = {}
            for _g, _want in zip(optimizer.param_groups, _cfg_hp):
                for _k, _v in _want.items():
                    if _g.get(_k) != _v:
                        _changed.setdefault(_k, (_g.get(_k), _v))
                    _g[_k] = _v
            _n_restored = sum(len(g["params"]) for g in optimizer.param_groups)
            print(f"  [opt] optimizer state restored ({_n_restored} param tensors)", flush=True)
            if _changed:
                print("  [opt] re-applied config hyperparameters over the checkpoint's: "
                      + ", ".join(f"{k} {a}→{b}" for k, (a, b) in sorted(_changed.items())),
                      flush=True)
        # MEMORY: optimizer.load_state_dict deep-copies into the live optimizer's tensors;
        # the checkpoint copy is a dead duplicate (~1.7GB on GPU for this model) that
        # otherwise lingers for the whole run (a local held by the train() frame). Dropping
        # it + empty_cache() also returns freed-but-reserved blocks from torch.load.
        del _opt_state
        gc.collect()
        torch.cuda.empty_cache()
    elif init_from_path:
        # Weights-only seed (step stays 0, fresh optimizer). resume takes precedence if both set.
        if not os.path.isfile(init_from_path):
            raise FileNotFoundError(f"training.init_from not found: {init_from_path}")
        print(f"Init-from (weights only) {init_from_path}")
        _if_missing, _if_unexp = load_weights_only(init_from_path, model, device,
                                                   allow_plain_to_tul=_plain_to_tul)
        _new_mods = list(getattr(tr, "init_from_new_modules", None) or [])
        if _new_mods:
            if _plain_to_tul:
                raise ValueError("training.init_from_new_modules and "
                                 "training.resume_plain_to_tul are two key audits; set one")
            check_init_from_keys(_if_missing, _if_unexp, model, _new_mods, init_from_path)
        _e_slot_from_ckpt = e_slot_seeded(_if_missing, model)

    # ── tul.code_target_ref (spec §17.1): the frozen VAE-stage twin ──────────────────
    # AFTER quantisation and AFTER the weights load, so the twin is an exact copy of what
    # the run was seeded with, parametrisations included. On a resume the checkpoint's own
    # reference wins: re-snapshotting the live weights at step N would silently move the
    # target mid-run, which is exactly the collapse this mechanism exists to prevent.
    _mdl0 = getattr(model, "_orig_mod", model)
    if getattr(getattr(_mdl0, "cfg", None), "tul", None) is not None \
            and bool(getattr(_mdl0.cfg.tul, "code_target_ref", False)):
        _ref_state = None
        if resume_path and os.path.isfile(resume_path):
            _ref_state = read_code_ref_state(resume_path)
            if _ref_state is None:
                print("  [code_ref] the resume checkpoint carries NO reference: "
                      "snapshotting the loaded weights instead", flush=True)
        _how = _mdl0.tul_code_ref_snapshot(_ref_state)
        _nref = sum(p.numel() for p in _mdl0.code_ref.parameters())
        print(f"  [code_ref] frozen VAE-stage twin {_how}: {_nref/1e6:.1f}M parameters, "
              f"0 trainable, eval mode; every code target and every grade is measured "
              f"against it", flush=True)

    # ── tul.fan_opf / tul.fan_route="latent": re-sync the EMA target twin ──────────────
    # The twin was built right after quantisation, from the weights at that moment. The
    # weights may have loaded since (init_from / resume): a resume loads the checkpoint's
    # own EMA state STRICTLY (re-snapshotting would jump the target to the live weights);
    # anything else copies the loaded live prelude into it.
    if _mdl0.__dict__.get("_fan_target") is not None:
        _ft_state = None
        if resume_path and os.path.isfile(resume_path):
            _ft_state = read_checkpoint_side_state(resume_path, "fan_target")
            if _ft_state is None:
                print("  [fan-target] the resume checkpoint carries NO EMA twin: "
                      "snapshotting the loaded weights instead", flush=True)
        print(f"  [fan-target] EMA twin {_mdl0.tul_fan_target_sync(_ft_state)}", flush=True)

    # ── tul.code_target_ema (LCTUL-J Stage 1) ─────────────────────────────────────────
    # Resolved ONCE, outside the training loop: whether and at what momentum the frozen
    # twin above moves toward the live model after every optimizer step.
    _code_ema_m = float(getattr(getattr(_mdl0, "cfg", None).tul, "code_target_ema", 0.0)
                        if getattr(_mdl0, "cfg", None) is not None
                        and getattr(_mdl0.cfg, "tul", None) is not None else 0.0)
    if _code_ema_m > 0.0:
        print(f"  [code_ref] EMA target: momentum m={_code_ema_m} (the twin keeps "
              f"fraction m of itself after every optimizer step, weight (1 - m) toward "
              f"the live model)", flush=True)

    # The phase (bag_size, tul_on) and both loaders are built after the curriculum block,
    # where total_steps is final. See phase.py.
    # Data fast-forward to the exact resume position (deterministic unshuffled stream). Only
    # for the base (non-curriculum) loader — the curriculum multi-source loader is rebuilt
    # below with its own stage logic, so skipping here would be wasted re-tokenization.
    _curr_on = bool(getattr(cfg, "curriculum", None) is not None
                    and getattr(cfg.curriculum, "enabled", False))
    # Fork-continue (fresh optimizer) is explicitly NOT a faithful resume → no point replaying
    # the deterministic stream to the exact position; start from the stream head (saves the
    # ~10min/arm CPU re-tokenization). Faithful resume still replays for exact continuation.
    _fork_continue = bool(getattr(cfg.training, "resume_fresh_optimizer", False))
    _resume_skip = start_step if (start_step > 0 and not _curr_on and not _fork_continue) else 0
    # `training.data_skip_batches` — an EXPLICIT fast-forward for a run whose step axis
    # restarts at 0 but whose weights did not. `training.init_from` resets the step counter,
    # so the stream would otherwise replay from its head and the run would take a second
    # epoch over batches the seed already trained on — a confound no matched arm carries.
    # Refuses rather than guesses when the resume path already asked for a skip, or under
    # the curriculum loader, which owns its own stage position.
    _explicit_skip = int(getattr(tr, "data_skip_batches", 0) or 0)
    if _explicit_skip:
        if _resume_skip:
            raise ValueError(
                f"training.data_skip_batches={_explicit_skip} and a faithful resume that "
                f"already fast-forwards {_resume_skip} batches both ask for a stream "
                f"position. Set only one.")
        if _curr_on:
            raise ValueError(
                "training.data_skip_batches is not defined under the curriculum loader: "
                "the multi-source loader owns its own stage position.")
        _resume_skip = _explicit_skip
        print(f"  [data] training.data_skip_batches={_explicit_skip} → the train stream "
              f"starts where the seed run stopped", flush=True)

    # ── Curriculum pretraining (Phase P) — length-bucketed multi-source ramp ──
    # GATED: absent/disabled → base.yaml path is byte-identical (curriculum_enabled False,
    # cur_grad_accum 1, no transitions, total_steps unchanged). When ON: overrides total_steps
    # and train_loader, and ramps seq_len / RoPE-context / micro-batch per stage with a
    # checkpoint-before-step-up (the stage transition fires at the top of the loop below).
    _curr_cfg = getattr(cfg, "curriculum", None)
    curriculum_enabled = bool(_curr_cfg is not None and getattr(_curr_cfg, "enabled", False))
    cur_grad_accum = 1
    cur_stage = -1
    if curriculum_enabled:
        from morph.training.curriculum import CurriculumScheduler
        from morph.training.curriculum_data import MultiSourceCurriculumLoader
        from morph.model.attention import CoPEEmbedding
        _stages = list(_curr_cfg.stages)
        _boundaries = [int(s.seq_len) for s in _stages]
        _contexts = [int(s.context_len) for s in _stages]
        _microbatch = [int(s.micro_batch) for s in _stages]
        _stage_steps = [int(s.steps) for s in _stages]
        _eff_batch = int(getattr(_curr_cfg, "eff_batch", 8))
        _weights = {str(k): float(v) for k, v in dict(_curr_cfg.blend).items()}
        # per-stage blend overrides (a data curriculum at one length, arc E15): a stage
        # may carry `blend: {source: weight}`; None keeps the global blend for that stage
        _stage_weights = [({str(k): float(v) for k, v in dict(s.blend).items()}
                           if s.get("blend") is not None else None) for s in _stages]
        if all(x is None for x in _stage_weights):
            _stage_weights = None
        _allowed_roles = [str(x) for x in getattr(
            _curr_cfg, "allowed_source_roles", ("pretrain_bulk", "reasoning_midtrain")
        )]
        _sched = CurriculumScheduler(_stage_steps)
        total_steps = _sched.total_steps                      # override training.steps
        from morph.training.data_placement import DataRuntimeConfig
        _data_rt = DataRuntimeConfig.resolve(getattr(cfg, "data_runtime", None))
        _curr_loader = MultiSourceCurriculumLoader(
            str(_curr_cfg.pretok_dir), _weights, _boundaries,
            seed=int(getattr(tr, "seed", 0)), allowed_roles=_allowed_roles,
            data_runtime=_data_rt, stage_weights=_stage_weights)
        # Held-out validation (arc E16): `curriculum.val_source` names an eval-holdout
        # shard under pretok_dir (scripts/olympiad_holdout_shard.py; the loader refuses
        # it in a training blend and refuses anything else here). Read synchronously
        # (no producer thread to close) at the live stage's seq_len and micro-batch, and
        # REWOUND before every eval: the same docs every time, on every arm.
        _val_source = getattr(_curr_cfg, "val_source", None)
        _curr_val = None
        if _val_source is not None:
            import dataclasses as _dcs
            _curr_val = MultiSourceCurriculumLoader(
                str(_curr_cfg.pretok_dir), {str(_val_source): 1.0}, _boundaries,
                seed=int(getattr(_curr_cfg, "val_seed", 1)), allowed_roles=_allowed_roles,
                data_runtime=_dcs.replace(_data_rt, prefetch_batches=0),
                holdout=True, verbose=False)

            def _curr_val_batches(tul_on: bool):
                _curr_val.rewind()
                return _curr_val.batches(
                    _microbatch[max(cur_stage, 0)], bag_size=0,
                    tul=tul_rt.val_data_cfg if (tul_rt and tul_on) else None)
            _vs = _curr_val.sources[0]
            print(f"[curriculum] val_source={_val_source!r}: {len(_vs.lens):,} held-out docs, "
                  f"{int(_vs.lens.sum()):,} tokens, max_len {int(_vs.lens.max())}, "
                  f"val_seed={_curr_val.seed} (rewound before every eval)", flush=True)
        # RoPE modules to re-anchor on each step-up (attention is EAGER → safe to mutate
        # cos/sin cache mid-run; compile only wraps the MLPs). Reach through _orig_mod.
        _rope_mods = [m for m in getattr(model, "_orig_mod", model).modules()
                      if isinstance(m, CoPEEmbedding)]
        def _ceil_div(a, b):
            return max(1, -(-a // b))
        print(f"[curriculum] ENABLED: {len(_stages)} stages seq={_boundaries} "
              f"context={_contexts} micro_batch={_microbatch} eff_batch={_eff_batch} "
              f"stage_steps={_stage_steps} total_steps={total_steps} "
              f"allowed_roles={_allowed_roles} stage_blends={_stage_weights} "
              f"val_source={getattr(_curr_cfg, 'val_source', None)!r} | "
              f"{len(_rope_mods)} RoPE modules", flush=True)

    # ── Training phase schedule (morph/training/phase.py) ────────────────────
    # ONE object owns every mid-run phase change. Built HERE because total_steps is only
    # final after the curriculum override above. `phase` is the live value; the loop
    # detects a change with `!=` instead of one bespoke predicate per schedule, and every
    # loader rebuild derives bag_size and the TUL layout FROM it, so a rebuild cannot
    # silently drop either. phase.py records the three defects this replaces.
    schedule = PhaseSchedule(
        total_steps=total_steps,
        tst_bag_size=tst_bag_size, tst_ratio=tst_ratio,
        tul_activate_at=tul_rt.activate_at if tul_rt else None,
    )
    phase = schedule.at(start_step)
    print(f"  {schedule}; start_step={start_step} → {phase}", flush=True)
    if phase.tul_on and start_step == 0:
        # Spec §5 / Block Transformer §3.7: E_slot starts as the MEAN of the embedding
        # table. On a resume the trained value comes back from the checkpoint instead, and
        # so it does on an init_from seed that carried one (`e_slot_seeded`).
        _mm0 = getattr(model, "_orig_mod", model)
        if _e_slot_from_ckpt:
            print("[TUL] layout ACTIVE from step 0; E_slot KEPT from the init_from "
                  "checkpoint", flush=True)
        else:
            _mm0.tul.init_at_activation(_mm0.embed.lm_weight())
            print("[TUL] layout ACTIVE from step 0; E_slot ← mean(embedding table)",
                  flush=True)
    # docs/tul-gate-spec.md §10. Pending until the first real batch: seating reads the
    # corpus base rate off actual span lengths rather than a hardcoded constant, and the
    # audit then refuses the run if the seated gate still cannot reach its targets.
    _gate_pending = getattr(getattr(model, "_orig_mod", model), "tul_gate", None) is not None
    _gate_alive_step = int(getattr(cfg.training, "gate_alive_check_step", 2000))

    def _rebuild_train_loader(skip_batches: int = 0):
        """The ONE way a train loader is built or rebuilt.

        Reads the live `phase` and `cur_stage` rather than taking them as arguments, so no
        call site can forget bag_size or the TUL layout — the omission that made a
        curriculum run train DENSE while `tul_on` stayed True (phase.py, defect 2)."""
        _tul = tul_rt.data_cfg if (tul_rt and phase.tul_on) else None
        if curriculum_enabled:
            # The multi-source curriculum loader, NOT _make_train_loader: the latter streams
            # the single base `data.dataset` and would silently drop the curriculum blend.
            return _curr_loader.batches(_microbatch[cur_stage],
                                        bag_size=phase.bag_size, tul=_tul)
        if isinstance(train_loader, Prefetcher):
            train_loader.close()          # stop the old producer thread (stream abandoned)
        return _make_train_loader(phase.bag_size, skip_batches=skip_batches,
                                  tul_on=phase.tul_on)

    # Curriculum builds its loader at the stage-0 transition on the first iteration
    # (cur_stage starts at -1), so there is nothing to build here for that path.
    train_loader = (None if curriculum_enabled else
                    _make_train_loader(phase.bag_size, skip_batches=_resume_skip,
                                       tul_on=phase.tul_on))
    val_loader = _make_val_loader(phase.tul_on)

    # ── NTP dropout (.agents/notes/proposed/feature/2026-08-27-ntp-dropout.md) ──
    # A fraction of steps run with NO slots at all: the plain MORPH path, core
    # looping over token positions. `slot_layout=None` is already the
    # bit-identical baseline forward and the loader already yields a 2-tuple for
    # it, so this needs no model change and no runtime flag — only a second
    # loader and a per-step coin.
    #
    # The batch MUST come from a slot-free loader. Reusing the TUL batch with
    # slot_layout=None would train the model to predict the inserted slot_id
    # positions as if they were text.
    _ntp_p = float(getattr(tr, "ntp_dropout_p", 0.0) or 0.0)
    if not 0.0 <= _ntp_p < 1.0:
        raise ValueError(f"training.ntp_dropout_p must be in [0,1), got {_ntp_p}")
    _ntp_loader = None
    if _ntp_p > 0.0:
        if not phase.tul_on:
            raise ValueError(
                "training.ntp_dropout_p > 0 with TUL off: every step is already an "
                "NTP step, so the knob would silently do nothing.")
        if curriculum_enabled:
            raise NotImplementedError(
                "ntp_dropout_p is not defined for the curriculum loader (it blends "
                "sources per stage); raises rather than quietly using the base stream.")
        _ntp_loader = _make_train_loader(phase.bag_size, tul_on=False)
        print(f"  [ntp-drop] p={_ntp_p}: that fraction of steps run slot-free "
              f"(plain MORPH path, separate loader)", flush=True)
    # Dedicated stream so the coin does not consume torch's RNG and shift every
    # other stochastic decision relative to a p=0 run at the same seed.
    _ntp_rng = _random.Random(int(getattr(tr, "seed", 0)) * 7919 + 13)
    _ntp_steps = 0
    # tul.fan_route (2026-09-30): the compile warmup ran train-mode forwards with grad on,
    # and the router counted their winners. Drop those counts so the first balance step
    # reads only the first real step's load. A no-op without a router.
    _mdl0.tul_fan_reset_route_load()

    # ── Optimizer step closure (resolved once, no per-step isinstance) ───
    def _step_optimizer():
        scaler.step(optimizer)
        scaler.update()

    # ── Training loop ─────────────────────────────────────────────────────
    _train_mode()
    step_times: list[float] = []
    t_start = time.perf_counter()
    # In-process divergence guard: counts consecutive eval-cadence points with train ppl
    # over a hard ceiling (after step 2000); aborts after K consecutive strikes so finite
    # prune-bounces (ppl≲70) don't trip it but a real divergence (ppl→1e5) does. Env-tunable.
    # `training.div_ppl_ceiling` / `training.div_strikes` (2026-09-17): the SAME guard as a
    # config key, for an arm whose train CE is legitimately high by design — the code-target
    # arm's frozen strict coda reads the loop's projected cell and its CE runs from 7.4 nats
    # (the cell carries nothing) toward 4.4 (ruler) and 1.3 (oracle), so a 1000-ppl ceiling
    # (6.9 nats) sits INSIDE the instrument's range. The env var stays the default so every
    # existing run keeps its guard; a config value wins when set.
    _tr_ceiling = getattr(cfg.training, "div_ppl_ceiling", None)
    _tr_strikes = getattr(cfg.training, "div_strikes", None)
    _div_ceiling = (float(_tr_ceiling) if _tr_ceiling is not None
                    else float(os.environ.get("MORPH_DIV_PPL", "1000")))
    _div_strikes_max = (int(_tr_strikes) if _tr_strikes is not None
                        else int(os.environ.get("MORPH_DIV_STRIKES", "2")))
    print(f"  [div-guard] ppl ceiling {_div_ceiling:.0f}, {_div_strikes_max} strikes", flush=True)
    _div_strikes = 0
    _aborted = False  # set by the non-finite / divergence guards → skip the post-loop final
                      # save+eval so a DIVERGED_step_N.pt is NOT shadowed by a misleading
                      # "completed" step_{total_steps}.pt (no-theater: don't fake a finished run).

    # ── Activation-memory probe (MORPH_MEM_PROBE) ──────────────────────────────
    # Root-causes the post-compact activation regression (+8 GB at b4). When set, we
    # reset the peak counter at the START of each step and print THAT step's fwd+bwd
    # high-water mark — correlate with the [compact]/[route] log lines to read the
    # masked-dense → sparse → routed deltas WITHIN ONE faithful training process.
    # MORPH_MEM_SNAPSHOT_STEP=N additionally dumps a full allocation snapshot (every
    # block + its Python alloc stack) at the first step >= N, for line-level attribution.
    _mem_probe = bool(os.environ.get("MORPH_MEM_PROBE"))
    _diag_optstate_path = os.environ.get("MORPH_DIAG_OPT")  # prune-divergence root-cause probe
    _diag_optstate_every = int(os.environ.get("MORPH_DIAG_OPT_EVERY", "1"))  # stride (1 = every step)
    _diag_fwd = bool(os.environ.get("MORPH_DIAG_FWD"))      # forward-side blow-up localizer
    _diag_m2g_path = os.environ.get("MORPH_DIAG_M2G")       # slow-EMA m₂/g geometry localizer
    # Prune cadence (for the M2N stale-m₂ diagnostic: log per-tensor rows DENSELY in a ±2-step
    # window around each prune event so cos(g,m₂) before vs after the topology change is visible).
    _prune_start = int(getattr(cfg.training, "prune_start", 10**9))
    _prune_interval = max(1, int(getattr(cfg.training, "prune_interval", 1)))
    # Core/CMS MLP params are where pruning acts → the stale-m₂ candidates we track.
    _m2g_filter = (lambda nm: "core." in nm)
    _mem_snap_step = int(os.environ.get("MORPH_MEM_SNAPSHOT_STEP", "-1"))
    _mem_snapped = False
    # History recording installs CUDA-allocator hooks that can make a Triton
    # autograd.Function (the fused HC kernels) return NULL — so we ONLY enable it when a
    # snapshot is explicitly requested (MORPH_MEM_SNAPSHOT_STEP>=0). The default probe is
    # peak-only (reset_peak_memory_stats + max_memory_allocated) which touches no hooks.
    if _mem_probe and _mem_snap_step >= 0:
        # stacks="python" (not the default "all"): the C++ unwinder in the allocator
        # hook makes the fused-HC autograd Function's apply() return NULL (confirmed
        # 2026-07-03 — SystemError at _FusedHCPost.apply under stacks="all").
        torch.cuda.memory._record_memory_history(max_entries=300_000, stacks="python")
        print(f"[memprobe] recording allocation history (snapshot @ step>={_mem_snap_step})",
              flush=True)
    elif _mem_probe:
        print("[memprobe] peak-only mode (no allocator hooks; set MORPH_MEM_SNAPSHOT_STEP for a snapshot)",
              flush=True)

    # ── Static-region CUDA graphs (MORPH_STATIC_GRAPHS): one-time lazy build ────
    # Built a few steps after (re)start so the allocator + Triton compiles are settled.
    # The build MUST run with no prior-step autograd graph alive (see
    # MORPHTransformer.build_static_graphs) — this loop owns those refs, so the build
    # hook below dels them first. Requires grad_accum==1 and bag_size==0 (a second
    # micro-backward would overwrite the bwd graphs' static grad buffers instead of
    # accumulating).
    # The captured regions are the PLAIN front/back at the plain shape. The TUL forward
    # takes an earlier branch and its L_total shape would never match, so a capture under
    # TUL would permanently reserve its ~9 GB private pool for graphs that never replay.
    _sg_want = os.environ.get("MORPH_STATIC_GRAPHS", "0").lower() not in ("0", "", "false")
    _sg_pending = _sg_want and not phase.tul_on

    # Cumulative alignment axes (see flops.py::perf_metrics). Accumulated EVERY step, not
    # only on logging ticks, so the two curves can be aligned exactly.
    _cum_tokens = 0.0
    _cum_passes = 0.0
    _sg_build_step = start_step + 3
    _sg_shape = None

    # ── Perf-pass region timing (MORPH_PERF_REGIONS=1) ─────────────────────────
    # CUDA-event + wall region timing around data/fwd/aux/bwd/prune/clip/opt, means
    # printed every 20 steps. Default OFF → shared nullcontext, zero cost, bit-identical.
    from morph.training.perf_regions import RegionTimer
    _rt = RegionTimer(bool(os.environ.get("MORPH_PERF_REGIONS")))
    # nsys capture window (MORPH_NSYS_WINDOW="a:b"): cudaProfilerStart at step a,
    # Stop at step b — pair with `nsys profile -c cudaProfilerApi` for a clean
    # steady-state trace that excludes warmup compile. Default unset → never fires.
    _nsys_win = os.environ.get("MORPH_NSYS_WINDOW")
    _nsys_a, _nsys_b = (int(v) for v in _nsys_win.split(":")) if _nsys_win else (-1, -1)
    # In-process kineto window (MORPH_PROF_WINDOW="a:b:/out/prefix"): torch.profiler
    # from step a to step b, then chrome-trace export + top-kernel table to
    # <prefix>.json / <prefix>.kernels.txt. Used instead of nsys — nsys's injected
    # threads hold the launcher pipe's write end across Triton's gcc fork → the
    # warmup-compile fork-deadlock (observed 2026-07-03: main thread anon_pipe_read
    # on a defunct child, twice, 40min). Default unset → never constructed.
    _prof_win = os.environ.get("MORPH_PROF_WINDOW")
    _prof_a = _prof_b = -1
    _prof_prefix = ""
    _prof = None
    if _prof_win:
        _pa, _pb, _prof_prefix = _prof_win.split(":", 2)
        _prof_a, _prof_b = int(_pa), int(_pb)
    # Bit-exactness gate (MORPH_EXACT_TRACE=<path>): append every step's loss as an
    # exact float hex + a SHA256 over the final state_dict. Two runs (baseline vs
    # change) are bit-identical iff their trace files are byte-identical. Adds one
    # loss.item() sync per step — observation-only, no math/RNG touched; use ONLY
    # for gate runs. Default unset → zero cost.
    _exact_trace_path = os.environ.get("MORPH_EXACT_TRACE")
    _exact_trace = open(_exact_trace_path, "w") if _exact_trace_path else None

    # Diagnostic-only (MORPH_DEBUG_STEP): per-step wall time + the exact Poisson depths
    # that step sampled, to catch the intermittent slow step and its trigger. Wrap
    # _sample_depths to stash the last-returned depths; printed in the timing block.
    _dbg_step = bool(os.environ.get("MORPH_DEBUG_STEP"))
    _dbg = {"depths": None, "step_start": time.perf_counter(), "cur_step": -1, "dumped": False}
    if _dbg_step:
        _real_sample = model._sample_depths
        def _logged_sample(b, dev_, _f=_real_sample):
            d = _f(b, dev_)
            _dbg["depths"] = d.tolist()
            # print at SAMPLE time (step start) so a step that wedges still has its
            # trigger logged (the end-of-step timing print would never fire).
            print(f"  [dbg] >>> forward start depths={_dbg['depths']} max={int(d.max())}", flush=True)
            return d
        model._sample_depths = _logged_sample

        # Watchdog: if any single step exceeds 60s (vs ~0.6s normal), dump the FULL
        # all-thread stack ONCE — captures the wedge in situ (the failure only manifests
        # in the real campaign launcher, not in controlled repros). faulthandler shows the
        # autograd-engine thread's Python frame + every other thread → what is actually stuck.
        import faulthandler as _fh, threading as _thr
        def _watchdog():
            while True:
                time.sleep(5)
                wedged = time.perf_counter() - _dbg["step_start"]
                if wedged > 60 and not _dbg["dumped"] and _dbg["cur_step"] >= 0:
                    _dbg["dumped"] = True
                    print(f"\n[dbg] !!! WEDGE: step {_dbg['cur_step']} running {wedged:.0f}s "
                          f"depths={_dbg['depths']} — dumping all threads:\n", flush=True)
                    _fh.dump_traceback()
        _thr.Thread(target=_watchdog, daemon=True).start()

    # Auxiliary-objective warmup gates (tul.mux_activate_at / sigreg_activate_at).
    # Resolved once: fractions of the run, same shape as tul.activate_at. Written
    # into non-persistent BUFFERS each step, so flipping one costs no recompile
    # and the arm stays a schedule rather than a branch in the forward.
    _mdl = getattr(model, "_orig_mod", model)
    _tulc = getattr(_mdl.cfg, "tul", None)
    _mux_on_at = int(float(getattr(_tulc, "mux_activate_at", 0.0)) * total_steps) if _tulc else 0
    _sig_on_at = int(float(getattr(_tulc, "sigreg_activate_at", 0.0)) * total_steps) if _tulc else 0
    if _tulc is not None and (_mux_on_at or _sig_on_at):
        print(f"  [aux] mux head on at step {_mux_on_at}, sigreg on at step {_sig_on_at} "
              f"(of {total_steps})", flush=True)
    # TUL-Code phases (docs/tul-code-spec.md §6): a Python int on the model, read at trace
    # time — NOT a gate buffer, because phase 1 must run no thinker pass and phases 1-2 no
    # sampler. Two recompiles per run, at the two switches, by design.
    _is_code = bool(getattr(_tulc, "code", False)) if _tulc is not None else False
    _code_p2 = int(float(getattr(_tulc, "code_phase2_at", 0.0)) * total_steps) if _is_code else 0
    _code_p3 = int(float(getattr(_tulc, "code_phase3_at", 0.0)) * total_steps) if _is_code else 0
    if _is_code:
        print(f"  [code] phase 2 (flow loss) at step {_code_p2}, phase 3 (rollout) at step "
              f"{_code_p3} (of {total_steps})", flush=True)

    _fan_select_anneal = bool(
        _tulc is not None and str(getattr(_tulc, "fan_mix", "mean")) == "select"
        and str(getattr(_tulc, "fan_select_write", "oracle")) == "anneal"
        and hasattr(_mdl, "fan_select_step"))
    if _fan_select_anneal:
        print(f"  [fan] select write anneals oracle -> gate over the first "
              f"{int(getattr(_tulc, 'fan_select_write_anneal', 1500))} steps", flush=True)
    for step in range(start_step, total_steps):
        # The model's train-only readings (the fan's per-pass instrument passes) are read
        # only by the 20-step log block below, so they are computed on those steps alone
        # (MORPHTransformer._train_instruments, 2026-10-04). Same cadence test as the block.
        _mdl._train_instruments = (step % 20 == 0)
        if _tulc is not None and hasattr(_mdl, "mux_gate"):
            _mdl.mux_gate.fill_(1.0 if step >= _mux_on_at else 0.0)
            _mdl.sigreg_gate.fill_(1.0 if step >= _sig_on_at else 0.0)
        if hasattr(_mdl, "code_grade_step"):
            # tul.code_grade (spec §17.2): the graded term runs every `code_grade_every`-th
            # step and seeds its own generators from this. A buffer, so flipping it costs no
            # recompile, and read only inside the compile-disabled grader.
            _mdl.code_grade_step.fill_(int(step))
        if _fan_select_anneal:
            # tul.fan_select_write=anneal: the select arm's scheduled sampling reads this
            # buffer inside the compile-disabled `_tul_fan_select`. Filled only on that arm.
            _mdl.fan_select_step.fill_(int(step))
        if _is_code:
            _ph = 1 if step < _code_p2 else (2 if step < _code_p3 else 3)
            if _ph != int(_mdl.code_phase):
                print(f"  [code] step {step}: phase {int(_mdl.code_phase)} -> {_ph}", flush=True)
                _mdl.code_phase = _ph
        if step == _nsys_a:
            torch.cuda.profiler.start()
            print(f"[nsys] cudaProfilerStart @ step {step}", flush=True)
        elif step == _nsys_b:
            torch.cuda.profiler.stop()
            print(f"[nsys] cudaProfilerStop @ step {step}", flush=True)
        if step == _prof_a:
            from torch.profiler import profile as _kineto_profile, ProfilerActivity
            torch.cuda.synchronize()
            _prof = _kineto_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            _prof.__enter__()
            print(f"[prof] kineto start @ step {step}", flush=True)
        elif step == _prof_b and _prof is not None:
            torch.cuda.synchronize()
            _prof.__exit__(None, None, None)
            _prof.export_chrome_trace(_prof_prefix + ".json")
            for _sort, _suffix in (("self_cuda_time_total", ".kernels.txt"),
                                   ("self_cpu_time_total", ".cpu.txt")):
                with open(_prof_prefix + _suffix, "w") as _pf:
                    # max_name_column_width: the default (55) truncates every templated
                    # CUDA kernel name to "void at::native::vectorized_elementwise_kernel<4, at...",
                    # which makes the table useless for attribution — the elementwise kernels are
                    # distinguished only by the functor in the truncated tail.
                    _pf.write(_prof.key_averages().table(sort_by=_sort, row_limit=120,
                                                        max_name_column_width=160))
            print(f"[prof] kineto stop @ step {step} → {_prof_prefix}.json/.kernels.txt/.cpu.txt",
                  flush=True)
            _prof = None
        if _dbg_step:
            _dbg["step_start"] = time.perf_counter()
            _dbg["cur_step"] = step
        if _mem_probe:
            torch.cuda.reset_peak_memory_stats()

        # ── One-shot peak reset once compilation has settled ────────────────
        # Without this, perf/peak_mem_* reports the WARMUP COMPILE, not the training step.
        # The warmup runs 56 passes over every active-set size (14…1), which allocates far
        # more than a steady step, and max_memory_allocated is monotonic. The tell, found on
        # a single-pass arm: peak=19.85GB logged identically at step 0, 20, 40 … 280 — a real
        # steady-state peak moves. Measured in isolation the same step was 10.35 GB, so the
        # logged figure overstated it ~1.9x and hid a genuine 2.1x memory win against A0's
        # 22.15 GB. Reset a few steps in, once compile and the allocator have settled.
        if step == start_step + 5:
            torch.cuda.reset_peak_memory_stats()
            print(f"  [mem] peak stats reset at step {step} — perf/peak_mem_* now reflects "
                  f"the TRAINING STEP, not the warmup compile", flush=True)

        # A stage change and a phase change can land on the SAME step. Both only mark the
        # loader dirty; ONE rebuild happens after both, so the second never discards a
        # freshly-prefetched stream the first just started.
        _rebuild_loader = False

        # ── Curriculum stage transition: checkpoint → RoPE re-anchor → loader.set_stage →
        #    micro-batch/grad-accum swap. Two independent risks at a step-up (activation OOM
        #    and the PE-shift loss spike) → the pre-step-up checkpoint is the recovery point. ──
        if curriculum_enabled and _sched.stage_at(step) != cur_stage:
            _k = _sched.stage_at(step)
            if step > start_step:                                  # nothing to save at step 0
                _cp = os.path.join(ckpt_dir, f"curriculum_pre_stage{_k}_step{step}.pt")
                save_checkpoint(_cp, step, model, optimizer, scaler, pruning, next_step=step)
                print(f"[curriculum] stage {cur_stage}→{_k} @ step {step}: pre-step-up ckpt {_cp}",
                      flush=True)
            for _m in _rope_mods:                                  # re-anchor taper + rebuild cache
                _m.set_context(_contexts[_k])
            # set_context rebuilt RoPE cache buffers as NEW tensors and the stage-up
            # changes seq shape → captured static-region graphs are stale; drop them.
            _mm = getattr(model, "_orig_mod", model)
            if hasattr(_mm, "static_graphs_invalidate"):
                _mm.static_graphs_invalidate(f"curriculum stage {_k} context re-anchor")
            _curr_loader.set_stage(_k)
            if _curr_val is not None:
                _curr_val.set_stage(_k)               # val follows the stage's seq_len/micro
            cur_stage = _k
            cur_grad_accum = _ceil_div(_eff_batch, _microbatch[_k])
            seq_len = _boundaries[_k]
            batch_size = _microbatch[_k] * cur_grad_accum          # effective, for tok/s logging
            _rebuild_loader = True
            print(f"[curriculum] → stage {_k}: seq_len={seq_len} context={_contexts[_k]} "
                  f"micro_batch={_microbatch[_k]} grad_accum={cur_grad_accum} eff_batch={batch_size} "
                  f"(RoPE re-anchored on {len(_rope_mods)} modules)", flush=True)

        # ── Phase transition: TST superposition → recovery, and/or TUL layout ON ──
        # ONE site, replacing the two hand-rolled switch blocks. `schedule.at(step)` is the
        # only predicate, and the rebuild reads the phase, so neither bag_size nor the TUL
        # layout can be dropped. Placed AFTER the stage change so cur_stage is already
        # valid; if both fire on the same step the loader is rebuilt twice, which is
        # correct and happens at most once per run.
        _next = schedule.at(step)
        if _next != phase:
            _sw = os.path.join(ckpt_dir, f"phase_switch_step_{step}.pt")
            save_checkpoint(_sw, step, model, optimizer, scaler, pruning, next_step=step)
            if _next.tul_on and not phase.tul_on:
                _mm = getattr(model, "_orig_mod", model)
                # Spec §5 / Block Transformer §3.7: E_slot ← mean of the embedding table,
                # unless an init_from seed carried a trained one (`e_slot_seeded`).
                if _e_slot_from_ckpt:
                    print(f"[TUL] activation at step {step}; E_slot KEPT from the "
                          f"init_from checkpoint", flush=True)
                else:
                    _mm.tul.init_at_activation(_mm.embed.lm_weight())
                # The captured front/back graphs are the PLAIN regions at the plain shape;
                # the TUL forward neither replays them nor matches L_total. Drop them so
                # their ~9 GB private pool is returned.
                if hasattr(_mm, "static_graphs_invalidate"):
                    _mm.static_graphs_invalidate("TUL activation")
            print(f"[phase] {phase} → {_next} @ step {step}. Switch ckpt: {_sw}", flush=True)
            phase = _next
            _rebuild_loader = True
            val_loader = _make_val_loader(phase.tul_on)

        # The ONE rebuild. Reads the live phase and cur_stage, so whichever block(s) fired,
        # the new loader carries both.
        if _rebuild_loader:
            train_loader = _rebuild_train_loader()

        # ── Static-region CUDA graphs: one-time build (MORPH_STATIC_GRAPHS) ──
        if _sg_pending and step >= _sg_build_step:
            _sg_pending = False
            _ga_now = cur_grad_accum if curriculum_enabled else 1
            if _ga_now != 1 or phase.bag_size != 0 or _sg_shape is None:
                print(f"[static-graph] NOT built (needs grad_accum==1, bag==0, a seen "
                      f"batch shape; got ga={_ga_now} bag={phase.bag_size} shape={_sg_shape}) "
                      f"— regions stay eager", flush=True)
            else:
                # HARD PRECONDITION: the previous step's autograd graph must be dead
                # before capture (stale default-stream AccumulateGrad nodes invalidate
                # the capture stream — cudaErrorStreamCaptureImplicit, measured — and a
                # failed capture poisons the CUDA generator). This loop holds the only
                # refs: loss/out AND the separately-bound routing_aux / spectral-penalty
                # locals (routing_aux's graph reaches the prelude/coda ROUTER params —
                # exactly the captured regions' accumulators).
                loss = out = routing_aux = _sp = None
                gc.collect()
                _m = getattr(model, "_orig_mod", model)
                # Dummy ids built WITHOUT any RNG draw (arange, not randint): a CUDA
                # randint here would advance the training generator and shift every
                # later poisson/dropout draw off the baseline stream (measured 2.6e-2
                # loss divergence from exactly this in the probe). Values are dummy —
                # the build's warmup math is discarded (fork_rng + buffer restore).
                _sg_ids = (torch.arange(int(torch.tensor(_sg_shape).prod()),
                                        device=device, dtype=torch.long)
                           % int(_m.cfg.vocab_size)).reshape(_sg_shape)
                _m.build_static_graphs(_sg_ids)   # raises on failure — do NOT catch
                del _sg_ids

        lr = lr_fn(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_mult", 1.0)

        optimizer.zero_grad(set_to_none=True)

        # Grad accumulation: _ga micro-steps before one optimizer step. _ga==1 (no curriculum)
        # → byte-identical to the single fwd/bwd path (loss/1 == loss). The curriculum uses it to
        # hold a constant effective batch as the per-stage micro-batch drops with context length.
        _ga = cur_grad_accum if curriculum_enabled else 1
        # The spectral penalty's value at this step, so train/loss can report the MODEL's
        # loss and train/loss_total the full objective. 0.0 when no penalty is built.
        _sp_value = 0.0
        # One coin per STEP, not per micro-batch: a step whose micro-batches
        # disagreed would mix two objectives into one optimizer update.
        _is_ntp = _ntp_loader is not None and _ntp_rng.random() < _ntp_p
        _ntp_steps += int(_is_ntp)
        # step_mix mode for THIS step — a function of `step` alone (see
        # build_step_mix_cycle), computed once per step (not per micro-batch) so a
        # grad-accumulated step never mixes two objectives into one optimizer update.
        _cur_tul_mode = (_step_mix_cycle[step % len(_step_mix_cycle)]
                         if _step_mix_cycle is not None else None)
        for _micro in range(_ga):
            with _rt.region("data"):
                try:
                    batch = next(_ntp_loader if _is_ntp else train_loader)
                except StopIteration:
                    # Curriculum .batches() is infinite so this is a non-curriculum-only refill; still,
                    # branch defensively so a curriculum run can never silently fall back to the base OWT stream.
                    if _is_ntp:
                        _ntp_loader = _make_train_loader(phase.bag_size, tul_on=False)
                        batch = next(_ntp_loader)
                    else:
                        train_loader = _rebuild_train_loader()
                        batch = next(train_loader)
                # TUL loaders yield a 3-tuple (input_ids, labels, slot_layout); every
                # other path keeps the 2-tuple contract untouched.
                if len(batch) == 3:
                    x, y, _layout = batch
                    _layout = _layout.to(device)
                else:
                    (x, y), _layout = batch, None
                # non_blocking: the prefetcher pins every batch (data_placement.pin_batch),
                # so the copy is queued behind the previous optimizer step instead of a
                # pageable copy that waited for it (a host sync per step, 2026-10-04).
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                _sg_shape = x.shape          # static-graph build uses the live shape

            if _gate_pending and _layout is not None and _layout.span_len is not None:
                from morph.training.gate_audit import audit_gate_travel, seat_gate_bias
                _gm = getattr(model, "_orig_mod", model)
                _gstats = seat_gate_bias(_gm.tul_gate, _layout)
                # ‖z‖ at init is exact, not estimated: RMSNorm's scale starts at ones, so
                # the normalised readout input has RMS 1 and L2 norm √d_model.
                _gstats.update(audit_gate_travel(
                    _gm.tul_gate, optimizer, _gstats, total_steps,
                    z_norm=float(cfg.model.d_model) ** 0.5))
                wandb.config.update({"gate_audit": _gstats}, allow_val_change=True)
                _gate_pending = False

            # Accumulate the alignment axes BEFORE the forward so a mid-step crash still
            # leaves a consistent count. tokens = real input tokens this step; passes =
            # tokens x nominal proxy, i.e. cumulative layer applications.
            _step_tokens = float(batch_size) * float(seq_len)
            _cum_tokens += _step_tokens
            _cum_passes += _step_tokens * _flops.flop_proxy()

            # step_mix requires a slot layout (it selects a TUL core loop mode); a step
            # where TUL is off (no layout — e.g. before tul.activate_at) or a non-TUL
            # loader batch runs the ordinary forward, same as step_mix being unset.
            _tsm = _cur_tul_mode if _layout is not None else None

            # Onset capture: arm the per-iteration rank reading for this forward (a plain
            # attribute read at trace time in _tul_core) and dump the batch the model is
            # about to see, so a spike step can be replayed offline span by span.
            if _rank_every > 0:
                getattr(model, "_orig_mod", model)._probe_rank = (step % _rank_every == 0)
            if _bdump_every > 0 and step % _bdump_every == 0:
                torch.save({"step": step, "x": x.cpu(), "y": y.cpu(),
                            "layout": None if _layout is None else
                            {k: (v.cpu() if torch.is_tensor(v) else v)
                             for k, v in vars(_layout).items()}},
                           os.path.join(_bdump_dir, f"batch_{step:06d}.pt"))
            if _frozen_eval and _frozen_root.training:
                raise RuntimeError(
                    f"training.frozen_eval: the frozen model is in TRAIN mode at step "
                    f"{step}. Some path called model.train() instead of _train_mode(); the "
                    f"head would be fitted on training-mode cells.")
            with _rt.region("fwd"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, labels=y, bag_size=phase.bag_size,
                                slot_layout=_layout, tul_step_mode=_tsm)
                    loss = out["loss"]
                    if _tsm is not None:
                        _sms = _step_mix_stats.setdefault(_tsm, {"n": 0, "loss_sum": 0.0})
                        _sms["n"] += 1
                        _sms["loss_sum"] += float(loss.detach())

                # Routing aux loss (load balance) — only active after route_start
                if pruning.is_routed:
                    routing_aux = collect_routing_aux_losses(model)
                    loss = loss + routing_aux

                # Core-map spectral-norm penalty. Zero while every core MLP linear is below
                # `cap` (bit-exact); only fires on σ_max runaway. Loss-side → optimizer-agnostic.
                if _spec_pen is not None:
                    _sp = _spec_pen.penalty()
                    # At lam == 0 `penalty()` returns an exact zero, so the value is known on
                    # the host; `float()` there was a host sync at the end of every forward
                    # that kept the CPU from queueing the backward (2026-10-04).
                    _sp_value = float(_sp.detach()) if _spec_pen.lam != 0.0 else 0.0
                    loss = loss + _sp.to(loss.dtype)

            with _rt.region("bwd"):
                scaler.scale(loss / _ga).backward()

        if _mem_probe:
            _pk = torch.cuda.max_memory_allocated() / 2**30
            _rsv = torch.cuda.max_memory_reserved() / 2**30
            print(f"[memprobe] step={step} routed={pruning.is_routed} "
                  f"fwdbwd_peak_alloc={_pk:.2f}GB reserved={_rsv:.2f}GB", flush=True)
            if _mem_snap_step >= 0 and step >= _mem_snap_step and not _mem_snapped:
                _snap_path = os.environ.get("MORPH_MEM_SNAPSHOT_PATH",
                                            "experiments/mem_snapshot.pickle")
                torch.cuda.memory._dump_snapshot(_snap_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                _mem_snapped = True
                print(f"[memprobe] dumped allocation snapshot → {_snap_path} "
                      f"(recording stopped)", flush=True)

        with _rt.region("prune"):
            prune_stats = pruning.step(model, step)

        # A prune event rewrites _dead_mask contents → a captured optimizer CUDA graph
        # (MORPH_OPT_CUDA_GRAPH) holds stale dead masks; drop it so the next steps re-warm
        # and recapture with the new topology. No-op when the flag is off / no graph yet.
        # (The optimizer-REBUILD boundary below needs nothing: it creates a new optimizer
        # object, so any captured graph dies with the old one.)
        if (prune_stats and prune_stats.get("pruning/prune_step")
                and hasattr(optimizer, "graph_invalidate")):
            optimizer.graph_invalidate("prune event")

        # Phase boundary (compact / routing) changed the param set → rebuild a FRESH
        # optimizer (fresh optimizer after compact). This step's backward grads
        # live on the OLD params (weight, pre-router); the new params (values, router)
        # have no grads yet, so we skip this step's update and train normally next step.
        # _step_optimizer closes over `optimizer` by name → reassigning here is picked up.
        if prune_stats and prune_stats.pop("_rebuild_optimizer", False):
            wandb.log({k: v for k, v in prune_stats.items()
                       if isinstance(v, (int, float))}, step=step)
            # FREE the old optimizer BEFORE building the new one. The old AdamW8bit holds
            # 8-bit moment tensors for the PRE-rebuild param set (e.g. the now-deleted dense
            # `weight` Parameters at compact). bitsandbytes optimizers keep internal reference
            # CYCLES (optimizer ↔ state ↔ param), so plain reassignment does NOT free them via
            # refcounting — they linger as LIVE GPU memory until a cyclic GC pass. Without this
            # the dense-weight optimizer state survives compact+route and stacks on top of the
            # new sparse state → b4 OOM even though the compacted model is smaller. So: clear
            # state, drop the name, gc.collect() to break the cycle, empty_cache() to return the
            # freed blocks to the driver — THEN allocate the new optimizer into the cleared pool.
            _mem_before = torch.cuda.memory_allocated() / 1e9
            # Phase boundary replaced modules → any captured static-region graphs read
            # stale storages; drop them (they stay eager for the rest of the run).
            _mm = getattr(model, "_orig_mod", model)
            if hasattr(_mm, "static_graphs_invalidate"):
                _mm.static_graphs_invalidate("phase-boundary rebuild")
            optimizer.zero_grad(set_to_none=True)
            if hasattr(optimizer, "state"):
                optimizer.state.clear()
            del loss
            optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
            _mem_freed = torch.cuda.memory_allocated() / 1e9
            optimizer = create_optimizer(model, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * pg.get("lr_mult", 1.0)
            _mem_after = torch.cuda.memory_allocated() / 1e9
            _n_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
            print(f"[opt] rebuilt optimizer @ step {step}: {_n_opt:,} params; "
                  f"cuda_alloc {_mem_before:.2f}→{_mem_freed:.2f} (freed)→{_mem_after:.2f} GB",
                  flush=True)

            # ── Phase-boundary controlled recompile (MORTAR carve / route) ────
            # GATED OFF BY DEFAULT (MORPH_COMPILE_CARVED). MEASURED NET-NEGATIVE at
            # d=768: carved-COMPILED 742ms vs carved-EAGER 698ms (-6.2%) — the carved
            # path's compute is the opaque BCSR custom-op GEMM (not fusable), the
            # surrounding elementwise is cheap, and compiling it thrashes on grad_mode
            # guards (recompile_limit-64 hit). The eager_on_recompile fallback below IS
            # the fast path. The ~+5% seen in early carve A/Bs is overhead dilution of the +22%
            # model-compute carving win, NOT lost fusion. Kept for cloud-scale revisit
            # where a larger d_model changes the GEMM-vs-elementwise economics.
            # When ON: open ONE controlled recompile window (default stance, warm every
            # active-set size, re-arm the stance), fork-safe via spawn workers.
            #   Fork-safety vs the step-0 wedge: Inductor codegen still runs in
            # the worker pool PRE-SPAWNED at startup (worker_start_method=
            # subprocess → no new forks from this now-threaded process). The
            # residual risk is the main-process cc launch for new Triton
            # launcher stubs — the SAME class of risk the pre-fix code already
            # took when the carved stk kernels JIT'd on their first post-carve
            # eager forward. Taking it here, in a bounded window we control,
            # beats letting it fire on a random later training step.
            #   RNG: fork_rng so the warmup's randint doesn't shift the
            # training stream's draw sequence.
            if use_compile and os.environ.get("MORPH_COMPILE_CARVED"):
                torch.compiler.set_stance("default")
                try:
                    with torch.random.fork_rng():
                        warmup_compile_all_shapes(
                            model, int(cfg.training.batch_size), seq_len, device,
                            int(getattr(tr, "warmup_passes_per_size", 4)),
                            tag=f"phase-boundary step {step}", tul_rt=tul_rt,
                        )
                finally:
                    torch.compiler.set_stance("eager_on_recompile")
                    print("  torch.compiler stance restored = eager_on_recompile",
                          flush=True)

            t_start = time.perf_counter()
            continue

        # ── Phase-1 onset probe (plan task 1.1) ───────────────────────────────
        # PRE-CLIP per-region and per-block gradient norms, at their own cadence,
        # independent of the 20-step log block and of the 100-step POST-clip block below.
        # Post-clip ratios are exact but the absolute values are not, and the onset lasts
        # about 140 steps — 100-step post-clip sampling is why nobody has ever seen inside
        # it. One fused _foreach_norm over every grad and ONE host sync, so it is cheap
        # enough to run every step. `grad_probe_every: 0` (the default) skips all of it.
        _probe_now = _gprobe_every > 0 and step % _gprobe_every == 0

        # ── Core-map Jacobian probe ───────────────────────────────────────────
        # Run BEFORE the optimizer step so sigma_max(J) is measured at the SAME weights
        # that produced this step's gradients — the two numbers are then comparable.
        _jac_log: dict[str, float] = {}
        if _jac_probe is not None and step % _jac_every == 0:
            _jac_log = _jacobian_probe(model, _jac_probe, x, y, _layout,
                                       phase.bag_size, _jac_iters)

        with _rt.region("clip"):
            # unscale_ ONCE: torch's GradScaler raises if it is called twice for the same
            # optimizer between update()s, which is why the probe lives inside this block
            # rather than before it. Between here and clip_grad_norm_ is the only place the
            # grads are both unscaled and unclipped.
            scaler.unscale_(optimizer)
            if _probe_now or _jac_log:
                _probe_log = _preclip_probe(model) if _probe_now else {}
                _probe_log.update(_jac_log)
                # The losses of the SAME step, so a gradient spike can be read against
                # whether the forward on that batch was itself abnormal (a forward
                # explosion moves the loss; a backward-only blow-up does not).
                _probe_log["loss/total"] = float(loss.detach())
                _probe_lk: list[str] = []
                for _lk in ("ce_main", "mux_local", "spandec_ce", "spandec_weighted", "gain_est", "gain_est_max", "gain_est_raw", "gain_reg_weighted", "gain_n_iters",
                            "gain_slot_p50", "gain_slot_p90", "gain_slot_max", "gain_slot_frac_gt1",
                            "gain_tail_pen", "gain_floor_pen", "mtp_weighted", "fixed_point", "fp_weighted", "core_gain_est", "core_gain_max", "core_gain_weighted",
                            "egrad", "egrad_weighted", "egrad_train", "egrad_auc", "egrad_pos_frac",
                            "pass_residual", "pass_res_weighted",
                            "oracle_z", "oracle_z_weighted",
                            "code_target", "code_target_weighted",
                            "loop_denoise", "loop_denoise_weighted",
                            "gram_kl", "gram_kl_weighted",
                            "code_grade", "code_grade_weighted",
                            "spandec_pass", "spandec_pass_weighted",
                            "coda_span", "coda_span_weighted",
                            "core_token_aux", "core_token_aux_weighted",
                            "loopmtp_align", "loopmtp_weighted",
                            "loopmtp_ponder", "loopmtp_ponder_weighted",
                            "vq", "vq_weighted", "vq_commit", "vq_codebook_loss",
                            "vq_perplexity", "vq_used",
                            "critic", "critic_weighted", "critic_agree",
                            "critic_gap_traj",
                            "horizon", "horizon_weighted", "horizon_terms",
                            "recon", "recon_ce", "recon_weighted"):
                    if _lk in out and out[_lk] is not None:
                        _probe_lk.append(_lk)
                # One device-to-host copy for all of them (`_host_floats`), same values.
                for _lk, _lv in zip(_probe_lk, _host_floats([out[k] for k in _probe_lk])):
                    _probe_log[f"loss/{_lk}"] = _lv
                wandb.log(_probe_log, step=step)
                if _gprobe_path is not None:
                    # Local mirror. wandb is the record of truth, but a per-step probe is
                    # 100+ series and the API is slow to page through; the onset analysis
                    # wants a file it can load in one read. Appended, one JSON per line,
                    # flushed per step so a killed run keeps everything up to the kill.
                    _gprobe_fh.write(json.dumps({"step": step, **_probe_log}) + "\n")
                    _gprobe_fh.flush()
                _tot = _probe_log.get("preclip/total", 0.0)
                _hit = None
                if _core_guard.enabled and _tot > 0.0:
                    _share = _probe_log.get("preclip/core", 0.0) / _tot
                    wandb.log({"preclip/core_share": _share}, step=step)
                    if _core_guard.update(step, _share):
                        _hit = _core_guard.reason()
                if _hit is None and _gain_guard.enabled:
                    if _gain_guard.update(step,
                                          _probe_log.get("preclip/core_block_gain", float("nan")),
                                          _probe_log.get("preclip/core_block_gain_r2", float("nan"))):
                        _hit = _gain_guard.reason()
                if _hit is not None:
                    _ep = os.path.join(ckpt_dir, f"TAKEOVER_step_{step}.pt")
                    print(f"[ABORT] core takeover — {_hit}. Saving {_ep} and stopping.",
                          flush=True)
                    try:
                        save_checkpoint(_ep, step, model, optimizer, scaler, pruning,
                                        next_step=step)
                    except Exception as _e:
                        print(f"[ABORT] emergency ckpt failed: {_e}", flush=True)
                    _aborted = True
                    break
            # KEEP THE RETURN VALUE. clip_grad_norm_ already computes the pre-clip global
            # norm; throwing it away costs nothing to compute and everything to diagnose.
            # The 2026-08-17 TUL divergence was invisible in wandb for exactly this reason —
            # the failure was a 1e8 gradient through the looped core, and the only surviving
            # evidence was a CMS saliency buffer inside a checkpoint. It is one scalar.
            _gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))

        with _rt.region("opt"):
            _step_optimizer()
            # Projected gradient: the constraint is enforced AFTER the update, so it cannot
            # be argued with by the data gradient. Optimizer moments are left alone — they
            # describe the unprojected step, which is what momentum should be built from.
            _proj_log = _spec_proj.step() if _spec_proj is not None else {}
            if _tern_ema_pairs:
                update_scale_emas(_tern_ema_pairs)
            # tul.code_target_ema (LCTUL-J Stage 1): AFTER the optimizer step and AFTER
            # the projected-gradient constraint and the ternary scale EMA above, so the
            # twin always lags the weights those actually produced. On the un-compiled
            # module (`_mdl0`), never `model` — the twin's own parameters live there.
            if _code_ema_m > 0.0:
                _mdl0.tul_code_ref_ema_update(_code_ema_m)
            # The fan's OPF / router arms (2026-09-30): the EMA target twin follows the
            # weights this step produced, P is QR-retracted onto the orthonormal matrices,
            # the router's balance bias moves from this step's load. Same placement rule as
            # the code_ref EMA (after every other post-step constraint, on `_mdl0`). A
            # no-op method on every other model.
            _mdl0.tul_fan_after_step()

        # ── Prune-divergence diagnostic (env MORPH_DIAG_OPT=<path>) ─────────
        # Post-step, grads still live (zero_grad is top-of-next-iter). Dequants m₂/ν and
        # attributes the worst update dead-vs-live, numerator-vs-denominator. Off by default.
        if _diag_optstate_path and (step % _diag_optstate_every == 0 or step <= 5):
            diag_prune_optstate(model, optimizer, step, _diag_optstate_path)
            diag_optstate_allparams(model, optimizer, step, _diag_optstate_path)
        # Slow-EMA geometry localizer (MORPH_DIAG_M2G=<path>) — decides the de-coherence operator.
        if _diag_m2g_path:
            diag_m2g_geometry(model, optimizer, step, _diag_m2g_path)
            # Authoritative g-vs-numerator capture (drained from the optimizer): dense per-tensor
            # logging in a ±2-step window around prune events to expose cos(g,m₂) before/after.
            _is_prune = bool(prune_stats and prune_stats.get("pruning/prune_step"))
            _near_prune = (step >= _prune_start
                           and ((step - _prune_start) % _prune_interval) in (0, 1, 2,
                                _prune_interval - 1, _prune_interval - 2))
            _m2n = diag_m2g_numerator(optimizer, model, step, _diag_m2g_path,
                                      dense=(_is_prune or _near_prune or step % 200 == 0),
                                      name_filter=_m2g_filter)
            if _m2n:
                wandb.log(_m2n, step=step)
        # Forward-side blow-up localizer (MORPH_DIAG_FWD=1): per-block residual-stream norm +
        # backbone ternary {-1,0,+1} flip count — tracks STE-cusp flip spikes
        # (calm optimizer, exploding loss). Registered lazily; run every step here for per-step norms.
        if _diag_fwd and _diag_optstate_path:
            diag_forward_norms(model, step, _diag_optstate_path)

        # ── Timing ────────────────────────────────────────────────────────
        if _exact_trace is not None:
            # loss, then the pre-clip global grad norm: both must match for a step to be
            # bit-identical (the grad norm sees every gradient, the loss only the forward).
            _exact_trace.write(f"{step} {float(loss.item()).hex()} {float(_gnorm).hex()}\n")
            _exact_trace.flush()

        t_now = time.perf_counter()
        _dt = t_now - t_start
        step_times.append(_dt)
        _rt.step_end(step, _dt)
        # t_start is reset at the END of the loop body (after eval/gen/ckpt) so
        # those non-training blocks are excluded from the NEXT step's _dt — keeps
        # steps_per_sec a pure training-throughput metric regardless of eval cadence.
        if len(step_times) > 100:
            step_times = step_times[-100:]
        if _dbg_step:
            _flag = "  <<< SLOW" if _dt > 3.0 else ""
            print(f"  [dbg] step {step}: {_dt:.2f}s depths={_dbg['depths']}{_flag}", flush=True)

        # ── Logging (every 20 steps) ──────────────────────────────────────
        if step % 20 == 0:
            sps = 1.0 / (sum(step_times) / max(len(step_times), 1))
            # Memory: allocated = peak live tensors; reserved = what the caching
            # allocator grabbed from the driver (alloc overhead/fragmentation).
            # The eager-vs-kernel gap in BOTH is the real "alloc overhead" delta.
            peak_alloc = torch.cuda.max_memory_allocated() / 2**20
            peak_resv = torch.cuda.max_memory_reserved() / 2**20
            _lv = loss.item()
            # train/loss must be the MODEL's loss, not the objective's. A regulariser added
            # to `loss` used to land in train/loss and train/ppl, which made a penalised arm
            # incomparable to its control and let the perplexity divergence guard fire on the
            # penalty instead of on the model. Measured: `a35-spec` logged ppl 4.9e8 while its
            # validation CE was 8.19. `train/loss_total` keeps the full objective.
            _lv_total = _lv
            _lv = _lv - _sp_value
            # Same rule for the MUX local head (arm v1a): keep train/loss and the
            # divergence guard on the model's CE, not the composite objective.
            if isinstance(out, dict) and out.get("mux_weighted") is not None:
                _lv = _lv - float(out["mux_weighted"])
            if isinstance(out, dict) and out.get("sigreg_weighted") is not None:
                _lv = _lv - float(out["sigreg_weighted"])
            if isinstance(out, dict) and out.get("gain_reg_weighted") is not None:
                _lv = _lv - float(out["gain_reg_weighted"])
            if isinstance(out, dict) and out.get("spandec_weighted") is not None:
                _lv = _lv - float(out["spandec_weighted"])
            if isinstance(out, dict) and out.get("mtp_weighted") is not None:
                _lv = _lv - float(out["mtp_weighted"])   # arc E8: train/loss = next-token CE
            for _ak in ("fp_weighted", "core_gain_weighted", "egrad_weighted",
                        "pass_res_weighted", "oracle_z_weighted", "code_target_weighted",
                        "code_grade_weighted",   # tul.code_grade, 2026-09-17 (spec §17.2)
                        "loop_denoise_weighted",  # tul.loop_denoise, LXTUL-P change 1
                        "spandec_pass_weighted", "coda_span_weighted",
                        "core_token_aux_weighted",
                        "loopmtp_weighted",          # LoopMTP Eq 13 (arXiv 2608.03624)
                        "loopmtp_ponder_weighted",   # LoopMTP's ponder regulariser
                        "vq_weighted",       # arc: the discrete thought, 2026-09-13
                        "row_contrast_weighted",   # tul.row_contrast_lambda, 2026-09-13
                        "fan_repel_weighted",      # LXTUL tul.fan_repel_lambda, 2026-09-19
                        "fan_select_gate_weighted",  # LXTUL tul.fan_mix=select, 2026-09-20
                        "fan_wta_weighted",        # LXTUL tul.fan_mix=all, 2026-09-20
                        "fan_head_wta_weighted",   # tul.fan_all_wta_grader=head, 2026-09-29
                        "opf_weighted",            # tul.fan_opf (arm F), 2026-09-30
                        "rlat_weighted",           # tul.fan_route=latent (arm T), 2026-09-30
                        "fan_lsel_weighted",       # tul.fan_loop_select, 2026-09-30
                        "latent_pre_weighted",     # tul.latent_pre_target (stage 1),
                                                   # 2026-10-02: no CE on that forward, so
                                                   # train/loss reads the loop constraint's
                                                   # remainder 0; the objective is
                                                   # train/loss_total and tul/latent_pre
                        "critic_weighted",   # arc E10 / 2026-09-12
                        "horizon_weighted",  # LoopMTP horizon alignment, 2026-09-14
                        "code_fm_weighted",   # TUL-Code flow term (the val side already
                                              # subtracts it; the tul-code draw at 2b6b321
                                              # reported CE + flow as train/loss)
                        "code_sigreg_weighted",   # LeJEPA SIGReg on the code cells
                        "gram_kl_weighted",   # tul.gram (LXTUL-G), 2026-09-23
                        "code_enc_var_weighted",  # LCTUL-J Stage 1, 2026-09-22: the
                                                   # online variance floor, its own weight
                                                   # (code_enc_var_lambda), separate from
                                                   # code_target_weight
                        "par_weighted",   # tul.spandec_parallel (LXTUL-E)
                        "nextlat_weighted",   # span-level NextLat, 2026-09-25
                        "enum_hard_weighted",  # LX hard credit: objective - mixture NLL
                        "explore_set_weighted",    # tul.latent_set_loss (knob 4)
                        "explore_extra_weighted",  # tul.hyp_score_head / hyp_merge (3, 6)
                        "code_policy_weighted",    # arm B, tul.code_policy_k (2026-09-29)
                        "coda_logit_l2_weighted",  # tul.coda_logit_l2 (spectral
                                                     # decoupling, 2026-09-23): folded
                                                     # into the fused CE kernel
                        "recon_weighted"):   # tul.recon_weight: own-span reconstruction
                                              # (CE queue item 10, 2026-10-05)
                if isinstance(out, dict) and out.get(_ak) is not None:
                    _lv = _lv - float(out[_ak])
            # ── Non-finite self-abort (no-theater: the αcap35 run spewed 600 steps of NaN
            #    after its external watchdog died in a power loss). A NaN/Inf loss NEVER
            #    recovers — save an emergency ckpt for forensics and stop, instead of burning
            #    the GPU. (Finite-but-huge prune bounces are NOT caught here — that's the
            #    external watchdog's job; this only fires on genuine non-finite.) ──
            if not math.isfinite(_lv):
                _ep = os.path.join(ckpt_dir, f"NONFINITE_step_{step}.pt")
                print(f"[ABORT] non-finite loss={_lv} at step {step} — saving {_ep} and stopping",
                      flush=True)
                try:
                    save_checkpoint(_ep, step, model, optimizer, scaler, pruning, next_step=step)
                except Exception as _e:
                    print(f"[ABORT] emergency ckpt failed: {_e}", flush=True)
                _aborted = True
                break
            # Finite-divergence guard: sustained ppl over the ceiling past the warmup descent.
            _ppl_now = math.exp(min(_lv, 20.0))
            if step > 2000 and _ppl_now > _div_ceiling:
                _div_strikes += 1
                print(f"[DIV-GUARD] strike {_div_strikes}/{_div_strikes_max}: ppl={_ppl_now:.1f} "
                      f"> {_div_ceiling:.0f} at step {step}", flush=True)
                if _div_strikes >= _div_strikes_max:
                    _ep = os.path.join(ckpt_dir, f"DIVERGED_step_{step}.pt")
                    print(f"[ABORT] sustained divergence — saving {_ep} and stopping", flush=True)
                    try:
                        save_checkpoint(_ep, step, model, optimizer, scaler, pruning, next_step=step)
                    except Exception as _e:
                        print(f"[ABORT] emergency ckpt failed: {_e}", flush=True)
                    _aborted = True
                    break
            else:
                _div_strikes = 0
            log: dict = {
                "train/loss": _lv,
                "train/loss_total": _lv_total,
                **({f"train/{_k}": float(_v.detach()) for _k, _v in out.items()
                    if _k.startswith("ce_mtp_") or _k in ("mtp_weighted", "fixed_point",
                                                          "fp_weighted", "core_gain_est",
                                                          "core_gain_max", "core_gain_weighted",
                                                          "loopmtp_align", "loopmtp_weighted",
                                                          "loopmtp_ponder",
                                                          "loopmtp_ponder_weighted")
                    # TUL-Code: `code_fm`, `code_fm_weighted`, `code_fm_rel`, the four
                    # t-band ratios and `code_rollout_frac` (phase 3). Scalars by
                    # construction (transformer.py wraps the stats as 0-d tensors).
                    or _k.startswith("code_fm") or _k.startswith("code_sigreg")
                    or _k.startswith("code_target") or _k.startswith("code_grade")
                    # tul.code_enc_var_lambda (LCTUL-J Stage 1): the online variance
                    # floor, its weighted twin (both "code_enc*") and the target-side
                    # collapse reading.
                    or _k.startswith("code_enc") or _k == "code_tgt_std"
                    # tul.loop_denoise: the sum, its weighted twin and the per-pass
                    # `loop_denoise_l2_t{t}` / `_cos_t{t}` / `_level_t{t}` (a VARIABLE
                    # number of keys — the batch's realised max depth decides how many)
                    or _k.startswith("loop_denoise")
                    or _k in ("code_rollout_frac", "code_cfg_drop_frac", "code_tape_rollout_frac",
                              "code_xm_score_mean", "code_xm_score_best",
                              # the flow thinker's noise schedule AS DRAWN
                              # (tul.code_t_logit_mean): uniform reads ~0.5, the
                              # mu=-1 / sigma=1 logit-normal reads ~0.303
                              "code_t_mean",
                              "code_mdm_nats", "code_mask_frac", "code_sub_frac")}
                   if isinstance(out, dict) else {}),
                "train/ppl": math.exp(min(_lv, 20.0)),
                "train/lr": lr,
                "perf/steps_per_sec": sps,
                # TST superposition ingests s× raw tokens per step (same FLOPs); count them.
                "perf/tokens_per_sec": sps * batch_size * seq_len * (phase.bag_size or 1),
                "perf/peak_mem_alloc_mib": peak_alloc,
                "perf/peak_mem_reserved_mib": peak_resv,
                "perf/step": step,
                "train/tst_bag": phase.bag_size,
                # Pre-clip global gradient norm and the factor grad_clip applied to it.
                # clip_factor << 1 sustained means the reported loss curve is being driven
                # by a gradient the clip is mostly discarding — read this BEFORE believing
                # any loss comparison between arms.
                "train/grad_norm": _gnorm,
                "train/clip_factor": min(1.0, grad_clip / max(_gnorm, 1e-12)),
            }
            # step_mix: per-mode cumulative step count + running mean loss (mission
            # spec keys, e.g. train/steps_db1, train/loss_db1). Cumulative across the
            # whole run (like _ntp_steps above), not just this log interval, so a
            # dashboard reads the SAME curve whether it samples every step or every
            # log_every steps.
            for _m, _s in _step_mix_stats.items():
                log[f"train/steps_{_m}"] = _s["n"]
                log[f"train/loss_{_m}"] = _s["loss_sum"] / max(_s["n"], 1)
            # ── FLOP efficiency (gate A3) ─────────────────────────────────
            # tok/s and peak memory alone are misleading on a launch-bound model: A0's step
            # is ~16 % fixed overhead and a DB arm's is ~50-60 %. Always read flop_proxy
            # next to them. `perf/flop_proxy` is NOMINAL (config-derived, comparable across
            # runs); `perf/layer_passes_per_token` is REALIZED (from the depths actually
            # sampled). A0 is 44.0 nominal and ~42.1 realized -- they are NOT the same
            # number and must not be compared to each other.
            # The denominator for BOTH ratios is REAL TOKENS PER ROW, not seq_len. Without
            # TUL they are equal; under TUL slot positions eat row budget, so a row's
            # L_total positions carry fewer real tokens and using seq_len understates the
            # ratio (measured ~2.6x off at a shrunken TUL shape). MORPH's own
            # tul/layer_passes_per_token divides by out["n_tokens"], so match it.
            # `out` is the LAST MICRO-BATCH, so its n_tokens divides by the MICRO size, not
            # the effective batch: at micro 12 x accum 2 the effective divisor read the
            # proxy 16.36 against 12.15 at micro 16 x accum 1 for the same model (the
            # 2026-09-08 Olympiad throughput audit). tok/s and peak were never affected.
            _tok_per_row = float(seq_len)
            if out is not None and out.get("n_tokens") is not None:
                _tok_per_row = max(float(out["n_tokens"]) / max(batch_size // _ga, 1), 1.0)
            _ppt = 1.0
            _cpf = None
            if phase.tul_on and _layout is not None:
                # TUL: the core runs on SLOT positions only -- that is where its win is.
                _n_slots = float(getattr(_layout, "max_slots", 0) or 0)
                _l_total = float(getattr(_layout, "total_positions", 0) or 0)
                if _l_total > 0:
                    _ppt = _ppt * (_l_total / _tok_per_row)
                if _n_slots > 0:
                    _cpf = _n_slots / _tok_per_row
            _ceiling = float(getattr(cfg.training, "gemm_ceiling_tflops", 0.0)) or None
            log.update(perf_metrics(
                _flops, batch=batch_size, seq_len=seq_len,
                step_time_s=1.0 / max(sps, 1e-9),
                positions_per_token=_ppt, core_position_frac=_cpf,
                ceiling_tflops=_ceiling,
                cum_tokens=_cum_tokens, cum_layer_passes=_cum_passes,
            ))

            if phase.tul_on:
                # Boundary statistics (spec §4) + the layer-pass accounting (spec §2).
                log["tul/active"] = 1
                if _ntp_loader is not None:
                    log["tul/ntp_step_frac"] = _ntp_steps / max(step - start_step + 1, 1)
                if _layout is not None and _layout.stats:
                    for _k, _v in _layout.stats.items():
                        log[f"tul/{_k}"] = _v
                if "layer_passes" in out and out["layer_passes"] is not None:
                    _npos = float(out["n_tokens"])
                    log["tul/layer_passes_per_token"] = float(out["layer_passes"]) / max(_npos, 1.0)
                    log["tul/tokens_per_batch"] = _npos
                for _k in ("ce_tokens", "ce_first_tok", "first_tok_counterfactual", "mux_local",
                           "gain_est", "gain_est_max", "gain_est_raw", "gain_reg_weighted",
                           "gain_n_iters",
                           "gain_slot_p50", "gain_slot_p90", "gain_slot_max",
                           "gain_slot_frac_gt1", "gain_tail_pen", "gain_floor_pen",
                           "sigreg", "spandec_ce", "spandec_n_tokens", "spandec_weighted",
                           "egrad", "egrad_weighted", "egrad_train", "egrad_auc",
                           "egrad_pos_frac", "pass_residual", "pass_res_weighted",
                           "oracle_z", "oracle_z_weighted", "oracle_z_mse",
                           "oracle_z_steps_used",
                           "spandec_pass", "spandec_pass_weighted", "spandec_pass_ce",
                           "spandec_pass_terms", "spandec_pass_n_tokens",
                           "coda_span", "coda_span_weighted", "coda_span_ce",
                           "core_token_aux", "core_token_aux_weighted",
                           "core_token_aux_ce", "core_token_aux_n", "core_token_aux_fp",
                           "vq", "vq_weighted", "vq_commit", "vq_codebook_loss",
                           "vq_perplexity", "vq_used", "vq_n_codes",
                           "vq_codebook_size",
                           "critic", "critic_weighted", "critic_train", "critic_agree",
                           "critic_agree_traj", "critic_agree_pert", "critic_n_traj",
                           "critic_n_pert", "critic_gap_traj", "critic_gap_pert",
                           "critic_replays",
                           "row_contrast", "row_contrast_weighted", "row_contrast_acc",
                           "row_contrast_n_rows", "row_contrast_n_anchors",
                           "coda_span_n_tokens",
                           "horizon", "horizon_weighted", "horizon_ce", "horizon_terms",
                           "code_fm", "code_fm_weighted", "code_fm_rel", "code_fm_raw",
                           "code_fm_null", "code_fm_band0_rel", "code_fm_band1_rel",
                           "code_fm_band2_rel", "code_fm_band3_rel", "code_rollout_frac",
                           "code_cfg_drop_frac", "code_tape_rollout_frac", "code_phase",
                           "code_xm_score_mean", "code_xm_score_best",
                           "code_t_mean",
                           "code_sigreg", "code_sigreg_weighted",
                           "code_mdm_nats", "code_mask_frac", "code_sub_frac",
                           "horizon_n_tokens",
                           # tul.coda_logit_l2 (spectral decoupling, 2026-09-23):
                           # `tul/coda_logit_sq` (the raw stat) and its weighted twin.
                           "coda_logit_sq", "coda_logit_l2_weighted"):
                    if _k in out and out[_k] is not None:
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.latent_pre_target (stage 1): the latent term and its exit readings
                # (retrieval vs chance, R^2, ranks; the per-pass ones are val-only).
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("latent_pre") and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.code_target: the term, its weighted twin, the exit cosine and the
                # per-pass cosines `code_target_cos_l{t}` (one key per realised pass).
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("code_target") and out[_k] is not None:
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.code_enc_var_lambda (LCTUL-J Stage 1, 2026-09-22): the online
                # variance floor ("code_enc_var", its weighted twin, "code_enc_std",
                # "code_enc_active") and the target-side collapse reading
                # ("code_tgt_std", live whenever code_target is on, lambda 0 or not).
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if (_k.startswith("code_enc") or _k == "code_tgt_std") \
                            and out[_k] is not None:
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.gram (LXTUL-G, 2026-09-23): the KL per slot summed over passes
                # (`gram_kl`, collapse reads below 0.5 nats), its per-pass split
                # `gram_kl_t{t}` (pass t = 1..T, a VARIABLE number of keys), its p90, the
                # weighted term and the KL per token, sigma/r and |mu|/r for the prior and
                # the posterior, and the RMS of u_t and h_t (before / after the step).
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("gram_") and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.gram_objective="iw" (LXTUL-GK, 2026-09-23): the bound per token
                # `gk_ce_iw` (it IS train/loss's CE part; nothing is subtracted for it),
                # rollout 0's weighted CE `gk_ce_single`, `gk_width_gain` = single - iw,
                # and `gk_w_entropy`, the per-span credit entropy over log K.
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("gk_") and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.spandec_parallel (LXTUL-E Stage 0 / Phase B): `par` and its weighted
                # twin, `par_ce` (the mixture CE per span token), `par_n_tokens`, `par_k`,
                # and at K > 1 `par_resp_entropy`, `par_code_win_{k}`, `par_ce_code_best`
                # and `par_width_gain`. Detached tensors from the forward (no sync there).
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if (_k == "par" or _k.startswith("par_")) and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # span-level NextLat (tul.nextlat_weight): `nextlat` / `nextlat_weighted`,
                # `nextlat_l1`, `nextlat_copy_l1`, `nextlat_cos`, `nextlat_pairs` and, with
                # the parallel head, `nextlat_draft_ce` / `_true_ce` / `_draft_gap`.
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("nextlat") and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.recon_weight (CE queue item 10, 2026-10-05): `recon` / `_weighted`,
                # `recon_ce` (the term itself — unweighted), `recon_n_tokens`. Train only,
                # so this scan only ever fires on a training step's `out`.
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("recon") and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # arm B, the code policy (tul.code_policy_k): `code_policy_weighted`, the raw
                # REINFORCE term `code_policy_pg`, `_entropy` (normalised by log C),
                # `_share_k{i}`, `_adv_mean` / `_adv_std`, `_value_mse`, `_reward_mean`,
                # `_agree` (sampled == argmax) and `_n_slots`. Detached tensors (no sync
                # in the forward).
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("code_policy") and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.code_enum_k (LXTUL-E Stage 1): `enum_ce_mix` (it IS train/loss's CE
                # part), `enum_ce_code{k}`, `enum_width_gain[_best]`, `enum_w_entropy`,
                # `enum_code_win{k}`, `enum_exit_sep` / `enum_exit_rms`.
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("enum_") and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.code_grade (spec §17.2): the grades, the cosines to E(best) and
                # E(true), the degenerate fraction and the per-pass `code_grade_cos_l{t}`.
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("code_grade") and out[_k] is not None:
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # LXTUL's fan (tul.fan_k): the repulsion term, the gate's entropy and the
                # per-pass stream cosines. A VARIABLE number of keys (the batch's realised
                # max depth decides how many `fan_stream_cos_t{t}` there are), so it is a
                # scan and not a tuple. They get their OWN `fan/` namespace rather than
                # `tul/` because the arm is scored on them: `fan/oracle_ce` against
                # `fan/single_ce` is the falsifier, and burying it among 90 `tul/` keys
                # is how a headline number stops being read.
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("fan_") and out[_k] is not None:
                        log[f"fan/{_k[4:]}"] = float(out[_k].detach())
                # tul.loop_carry: the carry's per-pass RMS, the gate's mean and the
                # injection ratio. A variable number of keys (the batch's realised max
                # depth decides how many), so it is a scan and not a tuple. Own `carry/`
                # namespace: `carry/rms_t{t}` against `carry/rms_t1` is how the arm is
                # read, and burying it among 90 `tul/` keys is how a number stops being
                # read (the `fan/` precedent, 2026-09-19).
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith("carry_") and out[_k] is not None:
                        log[f"carry/{_k[6:]}"] = float(out[_k].detach())
                # LX efficient-exploration knobs (tul_explore.py): `explore_*` and `hyp_*`.
                for _k in (list(out.keys()) if isinstance(out, dict) else []):
                    if _k.startswith(("explore_", "hyp_")) and torch.is_tensor(out[_k]):
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.oracle_z's honesty instrument: the ORACLE's own decoder loss at each
                # of its T steps. A variable number of keys, so it is a scan and not a
                # tuple — if these do not fall, the trajectory is not a descent and the
                # per-pass target is teaching noise.
                for _k in [k for k in out if str(k).startswith("oracle_z_l")]:
                    if out[_k] is not None:
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # tul.spandec_per_pass: the per-PASS CE and the horizon it was graded at.
                # A variable number of keys (the batch's realised max depth decides how
                # many passes there are), so it is a scan. These are the arm's instrument:
                # if `spandec_pass_t{t}` does not FALL with t, a deeper pass is not making
                # a better plan and the target is not being met by depth.
                _pp_pref = ("spandec_pass_t", "spandec_pass_h")
                for _k in [k for k in out
                           if str(k).startswith(_pp_pref) and str(k)[14:].isdigit()]:
                    if out[_k] is not None:
                        log[f"tul/{_k}"] = float(out[_k].detach())
                # docs/tul-gate-spec.md §10 — every step, because a gate that stops moving
                # is only visible as a FLAT curve, and a curve sampled at eval_every is
                # too coarse to tell "flat" from "converged".
                for _k, _v in out.items():
                    if _k.startswith("gate/") and _v is not None:
                        log[_k] = float(_v.detach().mean())

            # Retention gate diagnostic (#230): sigmoid(ret_gate) per retention block — THE key
            # signal for whether the model actually USES the retention branch (gate opens from ~0)
            # vs treats it as dead weight (stays ~0). A few scalars; log every step.
            _rm = getattr(model, "_orig_mod", model)
            if getattr(_rm.cfg, "retention", False):
                for _nm, _sec in (("prelude", _rm.prelude), ("core", _rm.core), ("coda", _rm.coda)):
                    for _i, _blk in enumerate(_sec):
                        if getattr(_blk, "ret_gate", None) is not None:
                            log[f"retention/gate_{_nm}{_i}"] = torch.sigmoid(_blk.ret_gate).item()

            # Per-region gradient norm (every 100 steps). The global norm says the
            # backward exploded; this says WHERE, which is the difference between a
            # one-step diagnosis and a checkpoint autopsy. Grads are POST-clip here
            # (zero_grad runs at the top of the next iteration), and the clip is a single
            # uniform rescale, so the ratios BETWEEN regions are exact and the absolute
            # values are recovered by multiplying with train/grad_norm / grad_clip.
            if step % 100 == 0:
                _reg: dict[str, float] = {}
                for _pn, _pp in model.named_parameters():
                    if _pp.grad is None:
                        continue
                    # torch.compile wraps the module, so names arrive as
                    # "_orig_mod.core.0.…"; strip every wrapper segment before taking
                    # the region, or every parameter lands in one bucket named "_orig_mod".
                    _parts = _pn.replace("_orig_mod.", "").split(".")
                    _key = _parts[0]
                    _reg[_key] = _reg.get(_key, 0.0) + float(_pp.grad.detach().float().pow(2).sum())
                    # Per-BLOCK too, for the stacked regions. A region total says the core
                    # exploded; the per-layer profile says whether it amplifies geometrically
                    # layer by layer (an unstable backward operator) or spikes in one block.
                    # The 2026-08-17 checkpoint autopsy had to reconstruct this from a
                    # pruning saliency buffer.
                    if _key in ("prelude", "core", "coda") and len(_parts) > 1 and _parts[1].isdigit():
                        _bk = f"{_key}.{_parts[1]}"
                        _reg[_bk] = _reg.get(_bk, 0.0) + float(_pp.grad.detach().float().pow(2).sum())
                for _key, _sq in _reg.items():
                    log[f"gradnorm/{_key}"] = _sq ** 0.5

            # Core-map σ_max (docs: the iterative-map note). The per-region gradnorm above says
            # the core owns the whole gradient; this says WHY — the core map's gain. Cheap:
            # 10 power-iteration matvecs on 12 linears. Never affects the loss when lam=0.
            if _spec_proj is not None and _proj_log:
                log.update(_proj_log)
            if _spec_pen is not None and _sp_log > 0 and step % _sp_log == 0:
                _sg = _spec_pen.sigmas()
                _vals = list(_sg.values())
                log["spec/sigma_max"] = max(_vals)
                log["spec/sigma_mean"] = sum(_vals) / len(_vals)
                _gu = [v for k, v in _sg.items() if "gate_up" in k]
                if _gu:
                    log["spec/sigma_gate_up_max"] = max(_gu)
                if _sp_cap > 0.0:
                    log["spec/n_over_cap"] = float(sum(v > _sp_cap for v in _vals))
                for _k, _v in _sg.items():
                    log[f"spec/sigma/{_k}"] = _v
                # Console too, not only wandb: the 2026-08-21 bake-off ran an hour into a
                # sigma runaway with nothing in the log to see it by, and the value was
                # recoverable only from a checkpoint autopsy.
                _worst = max(_sg, key=_sg.get)
                print(f"  [spec] step={step} sigma_max={max(_vals):.2f} ({_worst}) "
                      f"mean={sum(_vals) / len(_vals):.2f}", flush=True)

            # Pruning stats
            if prune_stats:
                log.update(prune_stats)

            # Routing diagnostics (every 100 steps, only when routed)
            if step % 100 == 0 and pruning.is_routed:
                rt_stats = collect_routing_stats(model)
                log.update(rt_stats)

            # Realized per-source token fractions (curriculum blend) — the drawn
            # fractions, not the configured targets, so share realization is a
            # logged fact rather than an assumption.
            if curriculum_enabled:
                for _sn, _fv in _curr_loader.realized_token_fractions().items():
                    log[f"data/frac_{_sn}"] = _fv

            # β1=0 SNR-gate activity (only when the gate is active) — mean gate applied + how many
            # coords were heavily noise-gated (<0.5), reset each log interval. Direct evidence the
            # gate fires (vs inferring from the loss curve).
            _o = getattr(optimizer, "_opt", optimizer)
            if getattr(_o, "g_snr_gate_kappa", 0.0) > 0.0 and getattr(_o, "_gate_n", 0) > 0:
                log["optim/snr_gate_mean"] = _o._gate_sum / _o._gate_n
                log["optim/snr_gate_lt0.5_coords"] = _o._gate_low
                _o._gate_sum, _o._gate_n, _o._gate_low = 0.0, 0, 0

            wandb.log(log, step=step)

            if step % (20 if total_steps <= 400 else 200) == 0:
                # flush: with stdout redirected to a log file this line otherwise sits in
                # the 8KB block buffer for MINUTES — a healthy run reads as hung/dead.
                # Every speed claim has to cite tok/s AND flop_proxy AND peak alloc
                # together (MORPH is launch-bound, so any one of them alone is misleading).
                # Put all three on the console line so a log file is self-sufficient and
                # nobody has to reconstruct them from wandb.
                _second = f"ppl={math.exp(min(loss.item(), 20.0)):.1f}  "
                # TUL-Code: the thinker's flow share on the console too (a resumed run's
                # wandb history is not readable locally; the log file must be sufficient)
                if isinstance(out, dict) and out.get("code_fm_rel") is not None:
                    _second += f"fm={float(out['code_fm_rel']):.3f}  "
                # tul.latent_pre_target (stage 1): the latent term, exit R^2 and same-row
                # retrieval against its chance (there is no CE on this forward).
                if isinstance(out, dict) and out.get("latent_pre") is not None:
                    _second += (f"lat={float(out['latent_pre']):.4f}  "
                                f"r2={float(out['latent_pre_r2_exit']):+.4f}  "
                                f"retr_same={float(out['latent_pre_retr_same_exit']):.3f}"
                                f"/{float(out['latent_pre_chance_same']):.3f}  ")
                # tul.spandec_parallel (LXTUL-E): the head's mixture CE per span token is
                # the arm's reading (Stage 0 trains nothing else), so the log carries it,
                # with the code-usage entropy and the in-head width gain at K > 1.
                if isinstance(out, dict) and out.get("par_ce") is not None:
                    _second += f"par_ce={float(out['par_ce']):.4f}  "
                    if out.get("par_resp_entropy") is not None:
                        _second += (f"par_H={float(out['par_resp_entropy']):.3f}  "
                                    f"par_gain={float(out['par_width_gain']):+.4f}  ")
                # tul.code_grade (spec §17.2): the graded term fires on one step in
                # `code_grade_every`, and its readings are the arm. Same reason the flow
                # share is here: a resumed run's wandb history is not readable locally, so
                # the log file has to carry the numbers that decide the arm.
                if isinstance(out, dict) and out.get("code_grade_n") is not None:
                    _g = lambda k: float(out[k]) if out.get(k) is not None else float("nan")
                    _second += (
                        f"grade[n={_g('code_grade_n'):.0f} frac={_g('code_grade_slot_frac'):.2f} "
                        f"best={_g('code_grade_best'):.2f} mean={_g('code_grade_mean'):.2f} "
                        f"worst={_g('code_grade_worst'):.2f} true={_g('code_grade_true'):.2f} "
                        f"rank={_g('code_grade_true_rank'):.2f} "
                        f"cosB={_g('code_grade_cos_best_true'):.3f} "
                        f"cosW={_g('code_grade_cos_worst_true'):.3f} "
                        f"degen={_g('code_grade_degen'):.2f}]  ")
                print(
                    f"[{step:7d}/{total_steps}] loss={loss.item():.4f}  "
                    f"{_second}"
                    f"lr={lr:.2e}  sps={sps:.2f}  "
                    f"tok/s={log.get('perf/tokens_per_sec', 0):.0f}  "
                    f"proxy={log.get('perf/flop_proxy', 0):.2f}  "
                    f"tflops={log.get('perf/model_tflops', 0):.1f}  "
                    f"peak={log.get('perf/peak_mem_alloc_mib', 0) / 1024:.2f}GB",
                    flush=True,
                )

        # ── Validation (every eval_every steps) ──────────────────────────
        if step % eval_every == 0 and step > 0:
            _val_extra: dict = {}
            _gm_e = getattr(model, "_orig_mod", model)
            _halt_eval = (phase.tul_on and _gm_e.tul_gate is not None
                          and _gm_e.cfg.tul.gate.drives_depth)
            if _curr_val_batches is not None:
                val_loader = _make_val_loader(phase.tul_on)   # rewound: the same docs
            val_loss, val_ppl = evaluate(model, device, val_loader, n_eval_batches,
                                         tul=phase.tul_on, extra=_val_extra,
                                         halt=_halt_eval)
            val_log: dict = {"val/loss": val_loss, "val/ppl": val_ppl}
            if _lp_mode != "off":
                # Stage 1: val/loss is the LATENT objective (no token CE exists), so a
                # perplexity of it would be a number with no meaning. Not logged.
                del val_log["val/ppl"]
            val_log.update(_val_extra)

            wandb.log(val_log, step=step)
            _tul_msg = ""
            if phase.tul_on:
                # plan_nats is ABSENT (not NaN) on a tg_restrict model — eval skips the
                # ablation pass there (see tul_forward_with_plan_nats). Printing a NaN
                # placeholder would trip every divergence grep on a healthy run.
                _plan = _val_extra.get("val/plan_nats")
                _tul_msg = (f"  ppl_tok={_val_extra.get('val/ppl_tokens', float('nan')):.2f}"
                            f"  first_tok={_val_extra.get('val/first_tok_ce', float('nan')):.4f}"
                            + (f"  plan_nats={_plan:+.4f}" if _plan is not None else "")
                            + f"  cf={_val_extra.get('val/first_tok_counterfactual', float('nan')):+.4f}"
                            f"  lp/tok={_val_extra.get('val/layer_passes_per_token', float('nan')):.2f}")
                if "val/gate/loss_gate" in _val_extra:
                    # docs/tul-gate-spec.md §10: the numbers that separate a WORKING gate
                    # from one sitting at a low loss emitting a constant. In the console,
                    # not only in wandb — a dead gate must be visible while the run runs.
                    _tul_msg += (
                        f"\n              gate: loss={_val_extra['val/gate/loss_gate']:.4f}"
                        f" corr={_val_extra.get('val/gate/gate_k_corr', float('nan')):+.3f}"
                        f" skill={_val_extra.get('val/gate/gate_k_skill', float('nan')):+.2f}tok"
                        f" k={_val_extra.get('val/gate/gate_k_mean', float('nan')):.2f}"
                        f"/gold {_val_extra.get('val/gate/gate_gold_mean', float('nan')):.2f}"
                        f" |err|={_val_extra.get('val/gate/gate_k_abs_err', float('nan')):.2f}"
                        f"/const {_val_extra.get('val/gate/gate_k_mae_const', float('nan')):.2f}"
                        f" k0={_val_extra.get('val/gate/gate_k_zero_frac', float('nan')):.3f}"
                        f" b={_val_extra.get('val/gate/gate_bias', float('nan')):+.3f}"
                        f" |w|={_val_extra.get('val/gate/gate_w_norm', float('nan')):.3f}")
                if "val/halt_ce_tokens" in _val_extra:
                    # The bake-off's headline: the SAME weights under the two depth
                    # policies (§11). Positive `Δ` = fixed depth wins, the pre-registered
                    # prediction.
                    _d = _val_extra["val/halt_ce_tokens"] - _val_extra.get(
                        "val/ce_tokens", float("nan"))
                    _tul_msg += (
                        f"\n              halt: ce_tok={_val_extra['val/halt_ce_tokens']:.4f}"
                        f" (Δ vs fixed {_d:+.4f})"
                        f" depth={_val_extra.get('val/halt_depth_mean', float('nan')):.2f}"
                        f" lp/tok={_val_extra.get('val/halt_layer_passes_per_token', float('nan')):.2f}")
            if "val/slot_cell_rms_t0" in _val_extra:
                # tul.slot_cell_pass_norm: the per-pass carried-cell RMS must sit at rms(g)
                # (the scale is fixed); the stream mean's RMS is free to move.
                def _cn(k: str) -> float:
                    return _val_extra.get(f"val/slot_cell_{k}", float("nan"))
                _ts = [t for t in range(0, 17) if f"val/slot_cell_rms_t{t}" in _val_extra]
                _tul_msg += (
                    f"\n              cell-norm: entry={_cn('rms_entry'):.3f} rms "
                    + " ".join(f"t{t}:{_cn(f'rms_t{t}'):.3f}" for t in _ts)
                    + " | mean-rms " + " ".join(f"t{t}:{_cn(f'mean_rms_t{t}'):.3f}" for t in _ts)
                    + f" | g mean={_cn('norm_g_mean'):.4f} std={_cn('norm_g_std'):.4f}"
                      f" rms={_cn('norm_g_rms'):.4f}")
            if "val/ce_tf" in _val_extra:
                _tul_msg += (f"  ce_tf={_val_extra['val/ce_tf']:.4f}"
                             + (f" ce_marginal={_val_extra['val/ce_marginal']:.4f}"
                                if "val/ce_marginal" in _val_extra else ""))
            if _lp_mode != "off":
                _lpk = lambda k: _val_extra.get(f"val/latent_pre_{k}", float("nan"))
                _tul_msg = (f"  latent={val_loss:.4f}  r2_exit={_lpk('r2_exit'):+.4f}  "
                            f"retr_same_exit={_lpk('retr_same_exit'):.4f} "
                            f"(chance {_lpk('chance_same'):.4f})  "
                            f"retr_all_exit={_lpk('retr_all_exit'):.4f} "
                            f"(chance {_lpk('chance_all'):.4f})  "
                            + "  ".join(f"t{t}:{_val_extra[f'val/latent_pre_retr_same_t{t}']:.3f}"
                                        for t in range(0, 17)
                                        if f"val/latent_pre_retr_same_t{t}" in _val_extra))
                print(f"  [VAL {step:7d}] stage-1{_tul_msg}", flush=True)
            else:
                print(
                    f"  [VAL {step:7d}] loss={val_loss:.4f}  ppl={val_ppl:.2f}{_tul_msg}",
                    flush=True,
                )
            _train_mode()

        # docs/tul-gate-spec.md §10: `w` starts at exactly zero and takes a gradient
        # every step, so a norm still at the floor here means the parameter is frozen.
        # Fail at step ~2k rather than score a 3-hour arm whose mechanism never engaged.
        if step == _gate_alive_step:
            _gm_a = getattr(model, "_orig_mod", model)
            if _gm_a.tul_gate is not None and _gm_a.cfg.tul.gate.lam > 0.0:
                from morph.training.gate_audit import assert_gate_is_alive
                assert_gate_is_alive(_gm_a.tul_gate, step)

        # ── Generation test ───────────────────────────────────────────────
        if gen_every > 0 and step % gen_every == 0 and step > 0:
            gen_text, gen_metrics = run_generation_test(
                model, device, tokenizer_name, seq_len, step,
                tul_rt=tul_rt if phase.tul_on else None,
            )
            wandb.log({"gen/sample": wandb.Html(f"<pre>{gen_text}</pre>"), **gen_metrics},
                      step=step)
            if gen_metrics:
                print("  [GEN] " + "  ".join(
                    f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(gen_metrics.items())),
                    flush=True)
            _emit_gen(f"step {step}", gen_text)
            _train_mode()

        # ── Checkpoint ────────────────────────────────────────────────────
        # ckpt_every <= 0 means "never checkpoint" (short probe / smoke runs). Guarded
        # because `step % 0` is a ZeroDivisionError that kills the run after step 0 —
        # hit for real on 2026-08-25 with `training.ckpt_every=0`.
        if ckpt_every > 0 and step % ckpt_every == 0 and step > 0:
            ck_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            save_checkpoint(ck_path, step, model, optimizer, scaler, pruning, next_step=step + 1)
            print(f"  Checkpoint: {ck_path}")
            _ck_ring.add(ck_path)

        # ── Rolling pre-onset capture ─────────────────────────────────────
        # A ring buffer of recent checkpoints. Its purpose is a state saved just BEFORE a
        # failure, so the failure can be replayed from it on demand instead of waited for:
        # in the deterministic configuration a resume continues the same trajectory, so a
        # pre-onset checkpoint turns a thousand-step wait into a short, repeatable audit.
        # Anything the abort guards write is a normal file and is never rotated away.
        if _roll_every > 0 and step % _roll_every == 0 and step > 0:
            _rp = os.path.join(ckpt_dir, f"ROLL_step_{step}.pt")
            save_checkpoint(_rp, step, model, optimizer, scaler, pruning, next_step=step + 1)
            _roll_ring.add(_rp)
            print(f"  [roll] {_rp}  (keeping {len(_roll_ring.paths)} of the last "
                  f"{_roll_ring.keep} × {_roll_every} steps)", flush=True)

        # ── Reset step timer ───────────────────────────────────────────────
        # Anchor the next step's _dt here, AFTER logging/eval/gen/ckpt, so those
        # non-training blocks don't inflate steps_per_sec (see Timing block above).
        t_start = time.perf_counter()

    # ── Final checkpoint ──────────────────────────────────────────────────
    if _aborted:
        # The run hit the non-finite/divergence guard — a DIVERGED_/NONFINITE_ ckpt already
        # holds the real (failed) state. Do NOT write step_{total_steps}.pt: that would label
        # diverged weights as a completed run (theater that misleads the next resume).
        print(f"[ABORT] run aborted at step {step}; skipping final save+eval "
              f"(forensic ckpt already written). No completed-run checkpoint.", flush=True)
        wandb.finish(exit_code=1)
        # EXIT NON-ZERO. A diverged run is a FAILED run, and the process must say so:
        # the arm queue keys "start the next arm" off the exit code, and returning 0 here
        # made it treat two diverged TUL arms (2026-08-17, tul-a1 @4540 / tul-a1r @3240)
        # as successes and march on. An abort that reports success is the worst kind of
        # silent failure — the operator reads "exit=0" and believes the arm ran.
        raise SystemExit(4)
    final_path = os.path.join(ckpt_dir, f"step_{total_steps}.pt")
    save_checkpoint(final_path, total_steps, model, optimizer, scaler, pruning, next_step=total_steps)
    print(f"Final checkpoint: {final_path}")

    if _exact_trace is not None:
        # Raw-byte SHA256 of every state tensor (dtype-agnostic reinterpret) — the
        # second half of the bit-exactness gate: identical trace + identical hash
        # ⇒ the change is trajectory-identical.
        import hashlib
        _h = hashlib.sha256()
        _sd = model.state_dict()
        for _k in sorted(_sd):
            _t = _sd[_k]
            if isinstance(_t, torch.Tensor) and _t.numel() > 0:
                _h.update(_k.encode())
                _h.update(_t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        _exact_trace.write(f"FINAL_SHA256 {_h.hexdigest()}\n")
        _exact_trace.close()
        print(f"[exact] state hash {_h.hexdigest()}", flush=True)

    # ── Final eval + generation ───────────────────────────────────────────
    # The eval()/train() + grad-mode toggle is safe under eager_on_recompile (set
    # after warmup): a guard miss runs that region eager instead of recompiling, so
    # there is no recompilation storm. We still skip the final eval when periodic
    # eval is disabled (eval_every > total_steps) — a pure throughput/mem run has no
    # val_loader worth touching and the skip lets it exit promptly.
    if eval_every <= total_steps:
        _val_extra = {}
        if _curr_val_batches is not None:
            val_loader = _make_val_loader(phase.tul_on)
        val_loss, val_ppl = evaluate(model, device, val_loader, n_eval_batches,
                                     tul=phase.tul_on, extra=_val_extra)
        _final = {"val/loss_final": val_loss, "val/ppl_final": val_ppl}
        if _lp_mode != "off":
            del _final["val/ppl_final"]   # stage 1: the latent objective has no perplexity
        _final.update({f"{k}_final": v for k, v in _val_extra.items()})
        wandb.log(_final, step=total_steps)
        print(f"Final val_loss={val_loss:.4f}"
              + ("  (stage 1: the latent objective)" if _lp_mode != "off"
                 else f"  ppl={val_ppl:.2f}")
              + "".join(f"  {k}={v:.4f}" for k, v in sorted(_val_extra.items())))

    if gen_every > 0 or bool(getattr(tr, "gen_test", False)):
        gen_text, gen_metrics = run_generation_test(
            model, device, tokenizer_name, seq_len, total_steps, n_tokens=200,
            tul_rt=tul_rt if phase.tul_on else None,
        )
        wandb.log({"gen/final": wandb.Html(f"<pre>{gen_text}</pre>"),
                   **{f"{k}_final": v for k, v in gen_metrics.items()}}, step=total_steps)
        _emit_gen(f"FINAL step {total_steps}", gen_text)
        if gen_metrics:
            print("  [GEN final] " + "  ".join(
                f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(gen_metrics.items())),
                flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
