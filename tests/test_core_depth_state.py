"""``model.core_depth_state`` — per-depth persistent state in the looped core
(.agents/notes/proposed/architecture/2026-09-29-per-depth-state-and-resonant-depth.md).

Core layer l on pass t+1 reads its OWN output from pass t through a gated, zero-init
projection add at its input: x_l^{t+1} <- x_l^{t+1} + g_l * P_l(out_l^t). Pass 1 reads
nothing. Plain path only (`_core_region`); `_tul_core` (the slot loop) never reads this
knob.

Contracts, one test each:
  1. off is byte-identical to the code BEFORE this feature existed (git commit b4753d2,
     run as a subprocess against an archived copy of the tree) — no new params, same
     loss, same logits, same post-construction RNG stream.
  2. on-at-init (P_l zero-init) is a no-op: same OTHER parameters as an off model built
     from the same seed, and the same forward logits under a multi-pass Poisson draw.
  3. a nonzero P_l changes the forward, and ONLY for samples whose depth >= 2 (a depth-1
     sample never reads a previous pass, so it is untouched by construction).
  4. the active-set gather is per-sample exact: perturbing one sample's tokens does not
     change another sample's logits at a different depth.
  5. causal: perturbing a later token does not change an earlier position's logits.
  6. gradient reaches P_l and g_l after one backward with nonzero P_l.

CPU only, tiny config, no tokenizer — mirrors tests/test_core_loop_aux.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer

V = 64
REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_COMMIT = "b4753d2"          # the tip this feature was built on top of
VENV_PY = sys.executable        # the same interpreter pytest is running under


def _cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=3, n_coda=1, mean_depth=3, max_depth=5, bptt_depth=5,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed: int, train: bool = True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_cfg(**kw))
    m.train(train)
    return m


def _batch(seed: int = 0, B: int = 4, T: int = 32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (B, T), generator=g)
    y = torch.randint(0, V, (B, T), generator=g)
    return x, y


def _perturb_proj(m: MORPHTransformer, seed: int = 999, scale: float = 0.3) -> None:
    """Move core_depth_proj / core_depth_gate off their zero/one init, deterministically."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.core_depth_proj:
            p.weight.copy_(scale * torch.randn(p.weight.shape, generator=g))
        m.core_depth_gate.copy_(1.0 + scale * torch.randn(m.core_depth_gate.shape, generator=g))


# ── Test 1: off matches the pre-change code exactly (subprocess, archived tree) ──────

_OLD_REF_CFG = dict(
    d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
    n_prelude=1, n_core=3, n_coda=1, mean_depth=3, max_depth=5, bptt_depth=5,
    channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
    hca_compress_ratio=8, top_k=8, window_size=16,
    retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
    dropout=0.0,
)

_OLD_REF_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
from morph.model.transformer import MORPHConfig, MORPHTransformer

cfg = MORPHConfig(**%r)
torch.manual_seed(11)
m = MORPHTransformer(cfg)
m.train()
g = torch.Generator().manual_seed(1)
x = torch.randint(0, %d, (4, 32), generator=g)
y = torch.randint(0, %d, (4, 32), generator=g)
out = m(x, labels=y)
result = {
    "loss": float(out["loss"].item()),
    "logits": out["logits"].detach().flatten().tolist(),
    "param_keys": sorted(n for n, _ in m.named_parameters()),
}
with open(sys.argv[2], "w") as f:
    json.dump(result, f)
""" % (_OLD_REF_CFG, V, V)


@pytest.fixture(scope="module")
def old_ref():
    """Archive commit b4753d2's `morph/` tree (read-only: `git archive` touches no
    working-tree state, index, or ref) and run the SAME tiny forward under it, in a
    separate process so its `morph.model.transformer` module never collides with the
    one already imported (new code) in this test process."""
    tmp = Path(tempfile.mkdtemp(prefix="morph_perdepth_old_ref_"))
    tar_path = tmp / "old.tar"
    subprocess.run(["git", "archive", "-o", str(tar_path), OLD_COMMIT, "morph"],
                   cwd=REPO_ROOT, check=True, capture_output=True)
    subprocess.run(["tar", "-xf", str(tar_path)], cwd=tmp, check=True, capture_output=True)
    script_path = tmp / "_old_ref_script.py"
    script_path.write_text(_OLD_REF_SCRIPT)
    out_path = tmp / "out.json"
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(
        ["nice", "-n", "19", "taskset", "-c", "0", VENV_PY, str(script_path),
         str(tmp), str(out_path)],
        cwd=tmp, check=True, env=env, capture_output=True, text=True,
    )
    return json.loads(out_path.read_text())


def test_off_matches_pre_change_code_exactly(old_ref):
    """model.core_depth_state=False (the default) on the CURRENT code must reproduce the
    exact loss/logits/param-set of the code as it stood BEFORE this feature (b4753d2) —
    same seed, same batch, no seed reset between construction and forward, so this also
    proves construction draws zero extra RNG when off (the two processes' post-
    construction RNG streams must agree for the Poisson depth draw to agree)."""
    torch.manual_seed(11)
    m = MORPHTransformer(_cfg())
    m.train()
    assert m.core_depth_proj is None and m.core_depth_gate is None
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, V, (4, 32), generator=g)
    y = torch.randint(0, V, (4, 32), generator=g)
    out = m(x, labels=y)
    assert float(out["loss"].item()) == old_ref["loss"]
    assert out["logits"].detach().flatten().tolist() == old_ref["logits"]
    assert sorted(n for n, _ in m.named_parameters()) == old_ref["param_keys"]
    assert "core_depth_proj" not in dict(m.named_modules())


# ── Test 2: on-at-init is a no-op ─────────────────────────────────────────────────────

def test_on_at_init_matches_off():
    off = _model(11, core_depth_state=False)
    on = _model(11, core_depth_state=True)
    # Every OTHER parameter is byte-identical (core_depth_proj is built LAST): proves
    # the new construction draws no RNG that reaches an earlier-built parameter.
    off_sd = dict(off.named_parameters())
    on_sd = dict(on.named_parameters())
    shared = set(off_sd) & set(on_sd)
    assert shared == set(off_sd)  # off has no keys on doesn't
    for k in shared:
        assert torch.equal(off_sd[k], on_sd[k]), k
    assert set(on_sd) - set(off_sd) == {
        f"core_depth_proj.{i}.weight" for i in range(3)
    } | {"core_depth_gate"}
    assert torch.equal(on.core_depth_gate, torch.ones_like(on.core_depth_gate))
    for p in on.core_depth_proj:
        assert torch.equal(p.weight, torch.zeros_like(p.weight))

    # Forward: reset the seed before EACH call so both draw the SAME Poisson depths —
    # isolating "does P_l=0 make the read a no-op" from "did construction perturb the
    # RNG stream" (already covered above).
    x, y = _batch(seed=2, B=4, T=32)
    torch.manual_seed(21)
    out_off = off(x, labels=y)
    torch.manual_seed(21)
    out_on = on(x, labels=y)
    assert torch.equal(out_off["logits"], out_on["logits"])
    assert torch.equal(out_off["loss"], out_on["loss"])


# ── Test 3: nonzero P_l changes the forward, only for depth >= 2 samples ─────────────

def test_nonzero_proj_changes_only_depth_ge_2_samples():
    m = _model(3, core_depth_state=True)
    m._sample_depths = lambda B, device: torch.tensor([1, 3], device=device, dtype=torch.long)
    x, y = _batch(seed=8, B=2, T=20)
    torch.manual_seed(41)
    logits_zero = m(x)["logits"]
    _perturb_proj(m)
    torch.manual_seed(41)
    logits_pert = m(x)["logits"]
    # depth-1 sample (row 0): pass 1 never reads a previous pass, so it is UNTOUCHED —
    # bit-identical regardless of P_l.
    assert torch.equal(logits_zero[0], logits_pert[0])
    # depth-3 sample (row 1): passes 2 and 3 DO read, so the output must move, and by a
    # real amount (not floating-point noise).
    assert not torch.equal(logits_zero[1], logits_pert[1])
    assert (logits_zero[1] - logits_pert[1]).float().norm() > 1e-3


# ── Test 4: active-set gather is per-sample exact (no cross-sample leak) ─────────────

def test_active_set_gather_has_no_cross_sample_leak():
    m = _model(5, core_depth_state=True)
    _perturb_proj(m)
    m._sample_depths = lambda B, device: torch.tensor([1, 3, 5], device=device, dtype=torch.long)
    x, y = _batch(seed=13, B=3, T=16)
    torch.manual_seed(51)
    out1 = m(x)["logits"]
    x2 = x.clone()
    x2[1, 4] = (x2[1, 4] + 1) % V   # perturb ONLY the depth-3 sample's tokens (row 1)
    torch.manual_seed(51)
    out2 = m(x2)["logits"]
    # rows 0 (depth 1) and 2 (depth 5) must be UNTOUCHED by row 1's perturbation — a
    # permutation/gather bug in ds_state_s (sort/slice/concat across the active-set
    # shrink) is exactly the kind of error that would leak row 1's state into row 2's
    # read at a later iteration.
    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[2], out2[2])
    # sanity: the perturbation actually did something to the row it targeted.
    assert not torch.equal(out1[1], out2[1])


# ── Test 5: causal — a later token cannot move an earlier position's logits ─────────

def test_causal_future_token_does_not_move_earlier_logits():
    m = _model(6, core_depth_state=True)
    _perturb_proj(m)
    x, y = _batch(seed=14, B=2, T=24)
    torch.manual_seed(61)
    out1 = m(x)["logits"]
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % V
    torch.manual_seed(61)
    out2 = m(x2)["logits"]
    assert torch.equal(out1[:, :-1], out2[:, :-1])
    assert not torch.equal(out1[:, -1], out2[:, -1])


# ── Test 6: gradient reaches P_l and g_l ─────────────────────────────────────────────

def test_gradient_reaches_proj_and_gate():
    m = _model(7, core_depth_state=True)
    _perturb_proj(m)
    # Force every sample to the full (bptt-covered) depth so the depth-state read fires
    # on every pass 2..max_depth for every row, under FULL BPTT (bptt_depth == max_depth).
    m._sample_depths = lambda B, device: torch.full((B,), 5, device=device, dtype=torch.long)
    x, y = _batch(seed=15, B=2, T=16)
    out = m(x, labels=y)
    out["loss"].backward()
    for i, p in enumerate(m.core_depth_proj):
        assert p.weight.grad is not None, i
        assert p.weight.grad.abs().sum().item() > 0.0, i
    assert m.core_depth_gate.grad is not None
    assert m.core_depth_gate.grad.abs().sum().item() > 0.0


def test_refuses_the_gain_hinge():
    """The gain hinge re-runs `_core_step` without the per-depth state, so it would probe
    a different map than the trained one: construction must refuse the pair."""
    with pytest.raises(NotImplementedError, match="core_gain_lambda"):
        _model(0, core_depth_state=True, core_gain_lambda=1.0)


@pytest.mark.parametrize("name,want", [
    ("notul_panel_norm_match_ds", True), ("notul_panel_norm_match_ds_s2", True),
    ("notul_panel_norm_match_s1r", False), ("notul_panel_norm_match_s2", False)])
def test_arm_configs_reach_the_model_config(name, want):
    """The YAML key must reach MORPHConfig through the trainer's builder. A key missing
    from `build_morph_config` is dropped silently (it was, in the first build: the
    'depth-state' memory trace ran the control)."""
    import os
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    cdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "morph", "configs")
    with initialize_config_dir(version_base=None, config_dir=cdir):
        cfg = compose(config_name=name)
    assert build_morph_config(cfg).core_depth_state is want
