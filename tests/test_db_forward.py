"""DiffusionBlocks forward contract, on a real (tiny) MORPH model. CPU only.

Protects the things a silent break would hide inside a healthy-looking loss curve:
``db_step=None`` is bit-identical to the baseline, a B=3 step really does train ONLY its
own block, the target is the NEXT token (no leak), and the tied head lives in the same
space as the target.

Pattern follows ``tests/test_tul_forward.py``: tiny config, ``use_kernels=False``, no
tokenizer, no CUDA.
"""

from __future__ import annotations

import math

import pytest
import torch

from morph.model.diffusion_blocks import (
    DBConfig,
    DBSchedule,
    DBStep,
    EDMPrecond,
    SliceScaler,
)
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
    out = m(ids, db_step=step, db_precond=rt.precond, db_want_logits=True)
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
    loss, metrics = db_loss(out, step, rt.precond, m)
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
        loss, _ = db_loss(out, step, rt.precond, m)
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
    loss, _ = db_loss(out, step, rt.precond, m)
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
    import math
    for lo, hi in ((0, euc), (euc, m.cfg.d_model)):
        t_rms = step.y_clean[..., lo:hi].float().pow(2).mean(-1).sqrt().mean()
        w_rms = w[..., lo:hi].float().pow(2).mean(-1).sqrt().mean()
        # The contract is that the two SIDES agree — not that either equals sigma_data.
        # (sigma_data stays a fixed EDM preconditioner constant; the target is unit-norm
        # per slice, deliberately well below it. See SliceScaler's docstring.)
        assert torch.allclose(t_rms, w_rms, atol=1e-3), (
            f"slice [{lo}:{hi}] target RMS {t_rms:.4f} != head RMS {w_rms:.4f}")
        expect = 1.0 / math.sqrt(hi - lo)
        assert abs(float(t_rms.detach()) - expect) < 1e-3, (
            f"slice [{lo}:{hi}] RMS {t_rms:.4f} != unit-norm per-component {expect:.4f}")


# ── the guards fire ──────────────────────────────────────────────────────────

def test_concat_conditioning_raises_until_the_mask_lands():
    cfg = DBConfig(mode="b3", conditioning="concat")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(0, V, (2, 16))
    labels = torch.randint(0, V, (2, 16))
    step = build_db_step(rt, m, labels)
    with pytest.raises(NotImplementedError, match="clean|noisy|mask"):
        m(ids, db_step=step, db_precond=rt.precond)


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
        out = m(ids, db_step=step, db_precond=rt.precond, db_want_logits=True)
        loss, _ = db_loss(out, step, rt.precond, m)
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


# ── the degeneracy guard (2026-08-19) ────────────────────────────────────────

def test_untrained_model_gets_no_free_lunch_at_median_sigma():
    """An UNTRAINED model must NOT already solve the task at the σ values p_noise samples.

    This is the test whose absence let a degenerate objective reach the GPU. `db_b1` trained
    for 100 steps and printed loss=0.0000 at step 60; the cause was that the target scale
    made `c_skip·z_t` hand the answer straight to the tied head, so ~66 % of draws had no
    gradient at all and the loss curve was a σ-lottery.

    The mechanism is specific to a DISCRETE target with a weight-tied readout: `z` only has
    to land in the right Voronoi cell of the embedding table, not to reconstruct `y`. So the
    task stays trivial at SNRs where an image would still be hard.

    Measured at d=64/V=512, untrained, ln(V) = 6.24:
        per-component std 0.5   -> CE 0.000  (degenerate)
        per-component std 1/√d  -> CE 6.14   (nothing free)   <- unit norm per slice

    Guards `SliceScaler`'s unit-norm target. If someone rescales to per-component `σ_data`
    to make EDM's constant "literally true", this fails.
    """
    import math

    # d=512 rather than the module's d=64 fixture. This criterion is scale-sensitive by
    # nature: unit norm gives per-component std 1/sqrt(slice_dim), so d=64 (slices 48/16 ->
    # 0.144/0.25) is partly readable no matter what, while d=512 (384/128 -> 0.051/0.088)
    # and the real d=1024 (768/256 -> 0.036/0.063) are not. Testing at d=64 would assert a
    # property the fixture cannot have.
    # BOTH d and V must be representative, because the criterion is sensitive to both:
    #   * d sets per-component std (unit norm -> 1/sqrt(slice_dim)). The module's d=64
    #     fixture gives 0.144, the real d=1024 gives 0.036.
    #   * V sets how separable the embedding table is. 64 near-orthogonal rows in 384 dims
    #     are trivially readable; the real vocab is 49152.
    # d=512 / V=2048 is the smallest pair that reproduces the real regime on CPU.
    BIG_V = 2048
    cfg = DBConfig(mode="b1", conditioning="x0_inject", slice_scale=True)
    torch.manual_seed(0)
    big = _tiny(d_model=512, channel_dims=(256, 160, 96), n_heads=4, n_kv_heads=4,
                vocab_size=BIG_V, bigram_hash_vocab=BIG_V)
    m = MORPHTransformer(big).eval()
    m.build_db_modules(cfg)
    m.eval()
    rt = _RT(cfg)
    ids = torch.randint(5, BIG_V, (4, 32))
    labels = torch.randint(5, BIG_V, (4, 32))
    ln_v = math.log(BIG_V)

    # Only the sigmas that carry real p_noise mass. sigma=0.3 is the MEDIAN of the
    # log-normal (P_mean=-1.2, P_std=1.2); 0.5 is its 66th percentile. sigma=0.05 is
    # excluded on purpose: it holds 6.7 % of the mass, and this tiny d=64 test model has
    # per-component std 1/sqrt(48)=0.144 vs the real model's 1/sqrt(768)=0.036, so at
    # d=64 the low-sigma tail is partly readable no matter what. That is a property of the
    # 64-dim fixture, not of the configuration under test.
    # sigma >= 0.3 is >= 50 % of p_noise's mass. The low-sigma tail is deliberately NOT
    # asserted: sigma <= 0.1 carries under 18 % of the mass and is intrinsically easy in
    # ANY diffusion model -- at small sigma the input already IS nearly the answer, which is
    # exactly why EDM applies w(sigma) and why p_noise is log-normal rather than uniform.
    # Measured here at d=512: sigma 0.05 -> CE 1.90, sigma 0.3 -> above the bar. Demanding
    # non-triviality at 0.05 would be asserting against the method's design, not against a
    # bug.
    for sigma_val in (0.3, 0.5, 1.0):
        step = build_db_step(rt, m, labels)
        step.sigma = torch.full((4,), sigma_val)
        y = step.y_clean
        step.z_noisy = (y.float() + sigma_val * torch.randn(
            y.shape, generator=torch.Generator().manual_seed(3))).to(y.dtype)
        with torch.no_grad():
            out = m(ids, db_step=step, db_precond=rt.precond, db_want_logits=True)
            ce = float(torch.nn.functional.cross_entropy(
                out["logits"].reshape(-1, BIG_V).float(), labels.reshape(-1)))
        # Bar at 0.4·ln(V). Partial information at the median sigma is EXPECTED and is not
        # a leak: z_t genuinely is a noisy view of the answer, so a nearest-embedding
        # readout earns partial credit. What must never happen is CE ~ 0, i.e. no gradient
        # at all. Measured at d=512 / V=2048 (ln V = 7.62):
        #     unit-norm per slice   -> CE 5.06 at sigma=0.3   (66 % of ln V)  PASS
        #     per-component 0.5     -> CE 0.0003              (0.004 %)       the old bug
        # The two regimes are three orders of magnitude apart, so any bar in between works;
        # 0.4 is chosen to be comfortably clear of both.
        assert ce > 0.4 * ln_v, (
            f"DEGENERATE at sigma={sigma_val}: untrained CE {ce:.4f} is far below ln(V)="
            f"{ln_v:.2f}. The target scale is letting c_skip*z_t reveal the label, so these "
            f"draws carry no gradient. See SliceScaler's docstring."
        )


def test_slice_scaler_targets_unit_norm_not_sigma_data():
    """Pins the scale decision itself, so a future 'cleanup' cannot silently undo it."""
    import math
    sc = SliceScaler((768, 256), sigma_data=0.5)
    assert sc.target_norms == (1.0, 1.0)
    x = torch.randn(2, 3, 1024)
    out = sc(x)
    assert out[..., :768].norm(dim=-1).allclose(torch.ones(2, 3), atol=1e-3)
    assert out[..., 768:].norm(dim=-1).allclose(torch.ones(2, 3), atol=1e-3)
    # per-component std lands in the band measured to be non-degenerate
    for lo, hi, d in ((0, 768, 768), (768, 1024, 256)):
        rms = out[..., lo:hi].pow(2).mean(-1).sqrt().mean()
        assert abs(float(rms) - 1.0 / math.sqrt(d)) < 1e-3
        assert float(rms) < 0.15, "per-component std too large -> degenerate objective"


def test_the_degeneracy_guard_actually_detects_the_degenerate_scaling():
    """A guard nobody has seen fail is not a guard. Reproduce the bug and catch it.

    Rescales the target to per-component std 0.5 -- the setting that shipped to the GPU and
    printed loss=0.0000 -- and asserts the untrained CE collapses. If this ever passes with
    a healthy CE, the mechanism has changed and
    test_untrained_model_gets_no_free_lunch_at_median_sigma is no longer meaningful.
    """
    import math

    BIG_V = 2048
    cfg = DBConfig(mode="b1", conditioning="x0_inject", slice_scale=True)
    torch.manual_seed(0)
    big = _tiny(d_model=512, channel_dims=(256, 160, 96), n_heads=4, n_kv_heads=4,
                vocab_size=BIG_V, bigram_hash_vocab=BIG_V)
    m = MORPHTransformer(big).eval()
    m.build_db_modules(cfg)
    m.eval()
    pre = EDMPrecond(cfg.sigma_data)
    ids = torch.randint(5, BIG_V, (4, 32))
    labels = torch.randint(5, BIG_V, (4, 32))

    euc = m.embed.hybrid.euclidean_dim
    lor = m.embed.hybrid.lorentz_dim
    # The OLD behaviour: scale each slice to per-component std == sigma_data.
    class _Degenerate(SliceScaler):
        def __init__(self):
            super().__init__((euc, lor), sigma_data=0.5)
            self.target_norms = (0.5 * math.sqrt(euc), 0.5 * math.sqrt(lor))

    bad = _Degenerate()
    y = bad(m.embed(labels))
    z = y + 0.3 * torch.randn(y.shape, generator=torch.Generator().manual_seed(3))
    step = DBStep(block_idx=0, sigma=torch.full((4,), 0.3), z_noisy=z.to(y.dtype),
                  y_clean=y, labels=labels)
    m.db_scaler = bad          # so db_lm_weight() carries the same (bad) transform
    with torch.no_grad():
        out = m(ids, db_step=step, db_precond=pre, db_want_logits=True)
        ce = float(torch.nn.functional.cross_entropy(
            out["logits"].reshape(-1, BIG_V).float(), labels.reshape(-1)))
    assert ce < 0.1, (
        f"expected the degenerate scaling to collapse CE toward 0, got {ce:.4f}. The "
        f"leak mechanism has changed -- re-derive the guard.")


def test_db_conditioning_is_never_ternarized():
    """The σ path must stay bf16 regardless of ternary_scope.

    `ternary_scope: backbone` means "every weight matrix that is not attention and not
    embeddings", which silently swept in db_sigma_cond.mlp and db_gates.*.to_mod. Snapping
    the σ embedding and the AdaLN shift/scale to {-1,0,+1} destroys the resolution of the
    progress coordinate the whole method rests on.

    Found by reading parametrization keys out of a step_300 checkpoint, NOT by a test —
    which is why this one exists. Same precedent as the HC W_fused proj and the SSM control
    matrices, both already excluded as precision-sensitive control paths.
    """
    import torch.nn.utils.parametrize as parametrize

    from morph.model.ternary_qat import apply_ternary_qat

    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    m = _model(cfg)
    apply_ternary_qat(m, scope="backbone", threshold=0.5)

    db_mods = [(n, mod) for n, mod in m.named_modules()
               if n.startswith("db_sigma_cond") or n.startswith("db_gates")]
    assert db_mods, "no DB conditioning modules found — did they get renamed?"
    for n, mod in db_mods:
        if hasattr(mod, "weight"):
            assert not parametrize.is_parametrized(mod, "weight"), (
                f"{n} was ternarized; the sigma-conditioning path must stay bf16")

    # and the guard must not have disabled ternary everywhere by accident
    n_tern = sum(1 for _, mod in m.named_modules()
                 if hasattr(mod, "weight") and parametrize.is_parametrized(mod, "weight"))
    assert n_tern > 0, "ternary QAT applied to nothing — the guard is too broad"


def test_fixed_sigma_val_is_deterministic_and_block_follows_sigma():
    """Validation must be a CURVE, not a lottery.

    A sampled-σ val CE cannot be compared between two evals 200 steps apart, because the
    draw dominates: the training loss moved 4.84–6.65 on σ alone. build_db_step(fixed_sigma)
    pins σ and derives the block from it, so repeated evals at the same weights agree
    exactly and a val series measures the model rather than the RNG.
    """
    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    labels = torch.randint(5, V, (3, 24))

    a = build_db_step(rt, m, labels, fixed_sigma=0.3)
    b = build_db_step(rt, m, labels, fixed_sigma=0.3)
    assert torch.equal(a.sigma, b.sigma)
    assert a.block_idx == b.block_idx
    assert float(a.sigma[0]) == pytest.approx(0.3)

    # the block must be the one that OWNS this sigma, not a random draw
    assert a.block_idx == int(rt.schedule.block_of_sigma(a.sigma[:1])[0])
    # and different sigmas must map to different blocks somewhere across the grid
    blocks = {build_db_step(rt, m, labels, fixed_sigma=s).block_idx
              for s in (0.01, 0.3, 10.0)}
    assert len(blocks) > 1, f"every val sigma mapped to the same block: {blocks}"


def test_db_checkpoint_round_trips_the_conditioning_params():
    """Resume must restore the σ path, or a 20k run that dies at 15k restarts blind.

    The step_300 checkpoint carries 10 db_* keys, but "the keys are present" is not
    "resume works". This round-trips a state_dict through a freshly built model and checks
    the forward is bit-identical.
    """
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    src = _model(cfg, seed=7)
    # move the zero-init gates off zero so a failure to restore them is visible
    with torch.no_grad():
        for p_ in src.db_gates.parameters():
            p_.add_(torch.randn_like(p_) * 0.05)
        for p_ in src.db_sigma_cond.parameters():
            p_.add_(torch.randn_like(p_) * 0.05)

    ids = torch.randint(5, V, (2, 24))
    labels = torch.randint(5, V, (2, 24))
    rt = _RT(cfg)
    step = build_db_step(rt, src, labels, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        want = src(ids, db_step=step, db_precond=rt.precond, db_want_logits=True)["logits"]

    sd = src.state_dict()
    assert any(k.startswith(("db_gates", "db_sigma_cond")) for k in sd), "no db_* in state_dict"

    dst = _model(cfg, seed=99)          # different init on purpose
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith(("db_gates", "db_sigma_cond"))], missing
    assert not [k for k in unexpected if k.startswith(("db_gates", "db_sigma_cond"))], unexpected
    with torch.no_grad():
        got = dst(ids, db_step=step, db_precond=rt.precond, db_want_logits=True)["logits"]
    assert torch.equal(want, got), "restored model does not reproduce the source forward"


# ── the fixes for what the audit found (2026-08-19) ──────────────────────────

def test_logit_scale_starts_at_identity_and_is_learnable():
    """Init MUST be identity: a high starting temperature re-creates the degenerate regime.

    Measured: at logit_scale_init=4.0 an untrained model scored CE 1.13 at sigma=0.3 against
    ln V = 7.62, because amplifying c_skip*z hands over the answer before anything is learned.
    Starting at 1.0 means the change can only help relative to the audited baseline.
    """
    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    assert cfg.logit_scale_init == 1.0
    m = _model(cfg)
    assert m.db_logit_scale is not None
    assert float(m.db_logit_scale.detach().exp()) == pytest.approx(1.0, abs=1e-6)
    assert m.db_logit_scale.requires_grad

    x = torch.randn(2, 4, 7)
    assert torch.allclose(m.db_scale_logits(x), x), "init must be exactly identity"


def test_logit_scale_actually_scales_and_receives_gradient():
    """A temperature that cannot move is theatre — it must sharpen AND train."""
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    with torch.no_grad():
        m.db_logit_scale.fill_(math.log(5.0))
    x = torch.randn(2, 4, 7)
    assert torch.allclose(m.db_scale_logits(x), x * 5.0, atol=1e-5)

    rt = _RT(cfg)
    labels = torch.randint(5, V, (2, 16))
    step = build_db_step(rt, m, labels)
    out = m(torch.randint(5, V, (2, 16)), db_step=step, db_precond=rt.precond)
    loss, mt = db_loss(out, step, rt.precond, m)
    loss.backward()
    assert m.db_logit_scale.grad is not None
    assert float(m.db_logit_scale.grad.abs()) > 0, "temperature got no gradient"
    assert mt["db/logit_scale"] == pytest.approx(5.0, rel=1e-3)


def test_logit_scale_is_consistent_between_train_and_eval_paths():
    """db_loss scales the INPUT; db_scale_logits scales the OUTPUT. They must agree.

    (s*x) @ W.T == s*(x @ W.T). If these drift apart, training and generation sharpen
    differently and generations will not match the loss curve.
    """
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    with torch.no_grad():
        m.db_logit_scale.fill_(math.log(3.0))
    d = torch.randn(2, 5, m.cfg.d_model)
    w = m.db_lm_weight()
    via_out = m.db_scale_logits(d @ w.T)
    via_in = (d * m.db_logit_scale.exp()) @ w.T
    assert torch.allclose(via_out, via_in, atol=1e-4)


def test_collapse_detector_is_alive_and_detects_collapse():
    """The old target_norm_mean metric was DEAD: SliceScaler pins unit norm, so it read
    sqrt(2) forever with a flat sparkline and kill criterion 4 could never fire.

    Pairwise cosine is the real detector — collapse means every embedding converges to one
    vector, so cosine -> 1.
    """
    from morph.training.db_setup import _collapse_metrics

    torch.manual_seed(0)
    diverse = torch.nn.functional.normalize(torch.randn(4, 16, 128), dim=-1)
    md = _collapse_metrics(diverse)
    assert abs(md["db/target_cos_mean"]) < 0.2, md

    collapsed = torch.nn.functional.normalize(
        torch.randn(1, 1, 128).expand(4, 16, 128) + 1e-4, dim=-1)
    mc = _collapse_metrics(collapsed)
    assert mc["db/target_cos_mean"] > 0.99, mc
    assert mc["db/target_cos_max"] > 0.99

    # and it must MOVE between the two regimes, unlike the metric it replaces
    assert mc["db/target_cos_mean"] - md["db/target_cos_mean"] > 0.7


def test_val_sigma_grid_probes_every_block():
    """A third of training visits went to the coda, which had ZERO val coverage — and the
    coda turned out to be the broken block (model-only CE 9.64, worse than doing nothing)."""
    from morph.training.db_setup import DbRuntime

    cfg = DBConfig(mode="b3", conditioning="x0_inject")
    sch = DBSchedule(cfg, mean_depth=6)
    grid = DbRuntime(model_cfg=cfg, schedule=sch, precond=EDMPrecond(cfg.sigma_data),
                     scaler=None, activate_at=0.0).val_sigmas
    blocks = {int(sch.block_of_sigma(torch.tensor([s]))[0]) for s in grid}
    assert blocks == {0, 1, 2}, (
        f"val grid covers blocks {sorted(blocks)}, not all three. grid={grid}")
