"""DiffusionBlocks forward contract, on a real (tiny) MORPH model. CPU only.

Protects the things a silent break would hide inside a healthy-looking loss curve:
``db_step=None`` is bit-identical to the baseline, a B=3 step really does train ONLY its
own block, the target is the NEXT token (no leak), and the tied head lives in the same
space as the target.

Pattern follows ``tests/test_tul_forward.py``: tiny config, ``use_kernels=False``, no
tokenizer, no CUDA.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.diffusion_blocks import DBConfig, DBSchedule, EDMPrecond
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.training.db_setup import build_db_step, db_loss

V = 64


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


class _RT:
    """Minimal stand-in for DbRuntime (avoids needing a Hydra cfg in a unit test)."""

    def __init__(self, cfg: DBConfig, mean_depth: int = 2):
        self.model_cfg = cfg
        self.schedule = DBSchedule(cfg, mean_depth=mean_depth)
        self.precond = EDMPrecond(cfg.sigma_data)


def _model(db_cfg: DBConfig | None = None, seed: int = 0):
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny()).eval()
    if db_cfg is not None:
        m.build_db_modules(db_cfg)
        m.eval()
    return m


# ── the parity gate ──────────────────────────────────────────────────────────

def test_db_off_forward_is_bit_identical_to_the_baseline():
    """Building DB modules must not perturb the DB-OFF path by a single bit.

    This is the CPU half of hard-constraint 4. The AdaLN gates are zero-init and the DB
    path is a separate method, so a baseline forward must be unchanged. If this fails, the
    conversion has leaked into the baseline and every A0 comparison is void.
    """
    ids = torch.randint(0, V, (2, 32))
    labels = torch.randint(0, V, (2, 32))

    plain = _model(None, seed=1)
    with torch.no_grad():
        a = plain(ids, labels=labels)["loss"]

    withdb = _model(DBConfig(mode="b3", conditioning="x0_inject"), seed=1)
    with torch.no_grad():
        b = withdb(ids, labels=labels)["loss"]

    assert torch.equal(a, b), f"DB-off drifted: {a.item()!r} vs {b.item()!r}"


def test_building_db_modules_adds_parameters_but_none_are_used_when_off():
    m = _model(DBConfig(mode="b3", conditioning="x0_inject"))
    assert any("db_sigma_cond" in n for n, _ in m.named_parameters())
    assert any("db_gates" in n for n, _ in m.named_parameters())


# ── the DB forward actually runs ─────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["b1", "b3"])
def test_db_forward_produces_finite_logits_of_the_right_shape(mode):
    cfg = DBConfig(mode=mode, conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(0, V, (2, 32))
    labels = torch.randint(0, V, (2, 32))
    step = build_db_step(rt, m, labels)
    out = m(ids, db_step=step, db_precond=rt.precond)
    assert out["logits"].shape == (2, 32, V)
    assert torch.isfinite(out["logits"]).all()
    assert out["denoised"].shape == (2, 32, 64)
    assert torch.isfinite(out["denoised"]).all()


@pytest.mark.parametrize("mode", ["b1", "b3"])
def test_db_loss_is_finite_and_backpropagates(mode):
    cfg = DBConfig(mode=mode, conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(0, V, (2, 32))
    labels = torch.randint(0, V, (2, 32))
    step = build_db_step(rt, m, labels)
    out = m(ids, db_step=step, db_precond=rt.precond)
    loss, metrics = db_loss(out["logits"], step, rt.precond)
    assert torch.isfinite(loss), loss
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert metrics["db/block_idx"] == step.block_idx


# ── block independence: the claim the whole method rests on ──────────────────

def test_b3_step_trains_only_its_own_block():
    """A B=3 step must leave the OTHER two sections' parameters gradient-free.

    This is the memory argument made falsifiable. If a section that did not run picks up a
    gradient, the blocks are not independent and the B-fold saving is fiction.
    """
    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    rt = _RT(cfg)
    sections = ("prelude", "core", "coda")

    for target in range(3):
        m = _model(cfg)
        ids = torch.randint(0, V, (2, 24))
        labels = torch.randint(0, V, (2, 24))
        step = build_db_step(rt, m, labels)
        step.block_idx = target                      # force the block under test
        out = m(ids, db_step=step, db_precond=rt.precond)
        loss, _ = db_loss(out["logits"], step, rt.precond)
        m.zero_grad(set_to_none=True)
        loss.backward()

        for si, name in enumerate(sections):
            stack = getattr(m, name)
            got = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in stack.parameters())
            if si == target:
                assert got, f"block {name} ran but received NO gradient"
            else:
                assert not got, (
                    f"block {sections[target]} ran but {name} ALSO got a gradient — "
                    f"the blocks are not independent")


def test_b1_step_trains_every_section():
    """mode='b1' is the whole net as one denoiser, so all three must get gradient."""
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(0, V, (2, 24))
    labels = torch.randint(0, V, (2, 24))
    step = build_db_step(rt, m, labels)
    out = m(ids, db_step=step, db_precond=rt.precond)
    loss, _ = db_loss(out["logits"], step, rt.precond)
    loss.backward()
    for name in ("prelude", "core", "coda"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in getattr(m, name).parameters()), f"{name} got no gradient"


# ── no leak ──────────────────────────────────────────────────────────────────

def test_target_is_the_next_token_not_the_current_one():
    """y must be embed(labels), and labels[t] = input_ids[t+1] from the loader.

    Pins the correction made during implementation: because the target is the NEXT token,
    MORPH's unshifted x0 (the CURRENT token) is legitimate conditioning, and shifting it
    would have destroyed a position of real context.
    """
    cfg = DBConfig(mode="b3", conditioning="x0_inject", slice_scale=False)
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.arange(0, 16).view(1, 16)
    labels = torch.arange(1, 17).view(1, 16) % V     # labels[t] = ids[t] + 1
    step = build_db_step(rt, m, labels)
    with torch.no_grad():
        expect = m.embed(labels)
    assert torch.allclose(step.y_clean, expect, atol=1e-6)
    assert not torch.allclose(step.y_clean, m.embed(ids), atol=1e-3), \
        "target equals the CURRENT token's embedding — that is the leak"


def test_slice_scaling_puts_target_and_tied_head_in_the_same_space():
    """Audit §4: the sampler uses softmax(logits) @ E, so E must carry the same transform.

    Compares the per-component RMS of the target against the tied head's rows. If they
    diverge, training converges and generation walks in the wrong units.
    """
    cfg = DBConfig(mode="b3", conditioning="x0_inject", slice_scale=True)
    m = _model(cfg)
    rt = _RT(cfg)
    labels = torch.randint(0, V, (2, 16))
    step = build_db_step(rt, m, labels)
    with torch.no_grad():
        w = m.db_lm_weight()
    euc = m.embed.hybrid.euclidean_dim
    for lo, hi in ((0, euc), (euc, m.cfg.d_model)):
        t_rms = step.y_clean[..., lo:hi].float().pow(2).mean(-1).sqrt().mean()
        w_rms = w[..., lo:hi].float().pow(2).mean(-1).sqrt().mean()
        assert torch.allclose(t_rms, w_rms, atol=1e-3), (
            f"slice [{lo}:{hi}] target RMS {t_rms:.4f} != head RMS {w_rms:.4f}")
        assert torch.allclose(t_rms, torch.tensor(cfg.sigma_data), atol=1e-3)


# ── the guards fire ──────────────────────────────────────────────────────────

def test_concat_forward_runs_and_has_no_future_leak():
    """The faithful two-source concat path (design doc): it runs, and a future clean
    token cannot influence an earlier position's output.

    We hold ``db_step`` (hence ``z_noisy`` and ``labels``) fixed and perturb ONLY the
    last clean token id. That id feeds clean_{L-1}, x0[L-1] and bigram[L-1] — all of
    which are visible only to position L-1. Every earlier position's logits must be
    bit-identical. A next-token leak anywhere (attention mask, bigram, x0) breaks this.

    CPU-EAGER by design (tensors are on CPU): this is a bit-exact check of the LOGIC.
    On GPU the fused/flash kernels leave ~1e-4 mask-bleed on masked positions (masked
    logits are very-negative, not true -inf), which the BASELINE causal model also has
    (~1e-4) — so a GPU version of this test needs a tolerance, not torch.equal. See
    docs/diffusionblocks-concat-design.md build status.
    """
    for mode in ("b1", "b3"):
        cfg = DBConfig(mode=mode, conditioning="concat")
        m = _model(cfg)
        rt = _RT(cfg)
        ids = torch.randint(0, V, (2, 16))
        labels = torch.randint(0, V, (2, 16))
        step = build_db_step(rt, m, labels)

        with torch.no_grad():
            logits0 = m(ids, db_step=step, db_precond=rt.precond)["logits"]
        assert logits0.shape == (2, 16, V)
        assert torch.isfinite(logits0).all(), f"{mode}: non-finite concat logits"

        ids2 = ids.clone()
        ids2[:, -1] = (ids[:, -1] + 1) % V
        with torch.no_grad():
            logits2 = m(ids2, db_step=step, db_precond=rt.precond)["logits"]
        assert torch.equal(logits0[:, :-1], logits2[:, :-1]), (
            f"{mode}: perturbing the last clean token changed an earlier position — future leak")
        # And the perturbation DID reach the last position (control: the path is live).
        assert not torch.equal(logits0[:, -1], logits2[:, -1]), (
            f"{mode}: last position ignored its own clean token — path is dead, not safe")


@pytest.mark.parametrize("attr,kind", [("prelude", "CSA"), ("core", "HCA")])
def test_two_source_is_differentiable_wrt_clean_context(attr, kind):
    """Design decision #2: the clean context must be grad-bearing, not detached.

    This isolates the claim that a b1 shared-weight test cannot: make the captured
    clean bundle a leaf and confirm gradient flows back INTO it from the noisy output.
    An accidental ``.detach()`` in ``forward_two_source`` would leave these grads None,
    which is exactly the dead-context failure `no_grad` would cause.
    """
    torch.manual_seed(0)
    m = _model(DBConfig(mode="b1", conditioning="concat"))
    impl = getattr(m, attr)[0].attention._impl
    B, S, d = 2, 16, m.cfg.d_model
    x_clean = torch.randn(B, S, d)
    x_noisy = torch.randn(B, S, d)

    ctx = impl.capture_context(x_clean)
    ctx = {k: v.detach().requires_grad_(True) for k, v in ctx.items()}
    out = impl.forward_two_source(x_noisy, ctx)
    out.sum().backward()

    # The DIFFERENTIABLE context is the value blocks (C_comp) and the window k/v — these
    # carry the attention output, so grad MUST flow into them or the clean encoder is dead.
    # K_I feeds only the top-k block SELECTION (argmax/topk is non-differentiable), exactly
    # like the baseline CSA indexer, which the model also trains with grad=None — so K_I is
    # legitimately exempt.
    diff_keys = [k for k in ctx if k != "K_I"]
    for name in diff_keys:
        t = ctx[name]
        assert t.grad is not None, f"{kind}: no grad into clean ctx[{name}] — context detached"
        assert float(t.grad.abs().sum()) > 0, f"{kind}: zero grad into clean ctx[{name}]"


def test_db_plus_tst_is_refused():
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(0, V, (2, 16))
    labels = torch.randint(0, V, (2, 16))
    step = build_db_step(rt, m, labels)
    with pytest.raises(ValueError, match="mutually exclusive"):
        m(ids, db_step=step, db_precond=rt.precond, bag_size=4)


def test_db_step_without_precond_is_refused():
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    labels = torch.randint(0, V, (2, 16))
    step = build_db_step(rt, m, labels)
    with pytest.raises(ValueError, match="db_precond"):
        m(torch.randint(0, V, (2, 16)), db_step=step)


# ── it can actually learn ────────────────────────────────────────────────────

def test_db_can_overfit_a_single_batch():
    """The honest end-to-end check: does the objective reduce loss at all?

    A forward that produces finite numbers but cannot learn is the failure mode a shape
    test cannot see. Small budget — this asserts the gradient signal is real, not that the
    method works at scale.
    """
    torch.manual_seed(0)
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    m.train()
    rt = _RT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ids = torch.randint(0, V, (2, 16))
    labels = torch.randint(0, V, (2, 16))

    gen = torch.Generator().manual_seed(0)
    first = last = None
    for i in range(40):
        step = build_db_step(rt, m, labels, generator=gen)
        out = m(ids, db_step=step, db_precond=rt.precond)
        loss, _ = db_loss(out["logits"], step, rt.precond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # unweighted CE is the comparable number across steps (w(σ) varies with the draw)
        with torch.no_grad():
            ce = float(torch.nn.functional.cross_entropy(
                out["logits"].detach().reshape(-1, V).float(), labels.reshape(-1)))
        if i < 5:
            first = ce if first is None else min(first, ce)
        if i >= 35:
            last = ce if last is None else min(last, ce)
    assert last < first, f"CE did not improve: first {first:.3f} -> last {last:.3f}"
