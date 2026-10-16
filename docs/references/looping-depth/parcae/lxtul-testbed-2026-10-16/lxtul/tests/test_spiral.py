"""CPU tests of the spiral arms (lxtul/spiral.py): published spiral, span-level loop, span-to-pair spiral and
span-to-pair MeSH.

fp32, tiny shapes, Parcae's own blocks. The causality tests perturb every token in turn (flipping its boundary
status too, so the span layout of the later positions changes) and demand that no earlier logit moves, bit for
bit. Each guard is then broken once (`test_broken_*`) with the same detector to show the detector can fail.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_spiral.py
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

import lxtul.attention as A
import lxtul.spiral as sp
import parcae_lm
from lxtul.spiral import LevelSpec, SpanLayout, SpiralConfig, SpiralParcae, chunk_geometry, pool_groups, span_layout

@pytest.fixture(autouse=True)
def cpu_flex(monkeypatch):
    """The block-memory coda runs FlexAttention: SDPA stand-ins on CPU (the same ones test_compact uses)."""
    from torch.nn.attention.flex_attention import create_block_mask
    from lxtul.tests.test_compact import _flex_ref
    monkeypatch.setattr(A, "_flex", _flex_ref)
    monkeypatch.setattr(A, "_mask", create_block_mask)


V, L, EOS = 64, 64, 1
CHUNK = lambda n: LevelSpec("chunk", size=n)
ARMS = {
    "published": dict(levels=(CHUNK(16), CHUNK(8), CHUNK(4), CHUNK(2))),
    "span": dict(levels=(LevelSpec("span"),)),
    "pair": dict(levels=(LevelSpec("span"), CHUNK(4), CHUNK(2))),
    "mesh": dict(levels=(LevelSpec("span"), CHUNK(4), CHUNK(2)), topology="mesh"),
    "chunk4_loop": dict(levels=(LevelSpec("chunk", size=4, depth="poisson"),)),
    "chunk2_loop": dict(levels=(LevelSpec("chunk", size=2, depth="poisson"),)),
    "token_skip_loop": dict(levels=(LevelSpec("chunk", size=1, depth="poisson"),)),
    "block4_loop": dict(levels=(LevelSpec("chunk", size=4, offset=0, shift=4, depth="poisson"),), topology="block"),
    "block2_loop": dict(levels=(LevelSpec("chunk", size=2, offset=0, shift=2, depth="poisson"),), topology="block"),
    "block4_depth1": dict(levels=(LevelSpec("chunk", size=4, offset=0, shift=4),), topology="block"),
    "blockmem4_loop": dict(levels=(LevelSpec("chunk", size=4, offset=0, shift=4, depth="poisson"),), topology="block",
                           block_memory=True),
    "blockmem2_loop": dict(levels=(LevelSpec("chunk", size=2, offset=0, shift=2, depth="poisson"),), topology="block",
                           block_memory=True),
    "pub_poisson2": dict(levels=(CHUNK(16), CHUNK(8), CHUNK(4), LevelSpec("chunk", size=2, depth="poisson"))),
    "pub_poisson4": dict(levels=(CHUNK(16), CHUNK(8), LevelSpec("chunk", size=4, depth="poisson"), CHUNK(2))),
}
BLOCK_ARMS = {"block4_loop": 4, "block2_loop": 2, "block4_depth1": 4}
BLOCKMEM_ARMS = {"blockmem4_loop": 4, "blockmem2_loop": 2}
SPAN_ARMS = ["span", "pair", "mesh"]
LOOP_ARMS = {"chunk4_loop": 4, "chunk2_loop": 2, "token_skip_loop": 1}      # arm -> chunk size


def toy_rule():
    from morph.model.tul_layout import BoundaryRule
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    lut[EOS] = True
    return BoundaryRule(lut, min_span=2, span_cap=8, eos_id=EOS)


def make(arm: str, max_spans: int = 24, seed: int = 0) -> SpiralParcae:
    torch.manual_seed(seed)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=32, num_attention_heads=2, num_key_value_heads=2, intermediate_size=64,
        recurrent_embedding_dimension=32, recurrent_intermediation_embedding_dimension=64,
        n_layers_in_prelude=2, n_layers_in_recurrent_block=2, n_layers_in_coda=2,
        mean_recurrence=3, mean_backprop_depth=3, vocab_size=V, block_size=L + 32)
    m = SpiralParcae(cfg, SpiralConfig(max_spans=max_spans, max_depth=5, **ARMS[arm]), toy_rule())
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():                    # zero-init weights and zero routers would hide wiring bugs
        for n, p in m.named_parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            if "pool_w" in n or "mesh" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.4)
    return m.eval()


def tokens(B: int = 2, seed: int = 0, n: int = L) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(20, V, (B, n), generator=g)
    punct = torch.rand((B, n), generator=g) < 0.22
    x[punct] = torch.randint(10, 20, (int(punct.sum()),), generator=g)
    x[torch.rand((B, n), generator=g) < 0.02] = EOS
    return x


def fwd(m, x, **kw):
    torch.manual_seed(123)                   # the loop's state init is random: same draw for both calls
    m.step = 7
    with torch.no_grad():
        return m(x, return_logits=True, **kw)["logits"]


def flip(x, t):
    x2 = x.clone()
    x2[:, t] = torch.where((x[:, t] >= 10) & (x[:, t] < 20), torch.full_like(x[:, t], 30), torch.full_like(x[:, t], 12))
    return x2


def causality_violation(m, x, ts=None) -> float:
    """Largest logit change at a position BEFORE the perturbed token, over every perturbed token."""
    base, worst = fwd(m, x), 0.0
    for t in (range(1, x.shape[1]) if ts is None else ts):
        lg = fwd(m, flip(x, t))
        worst = max(worst, float((lg[:, :t] - base[:, :t]).abs().max()))
    return worst


def garbage_violation(m, x, monkeypatch) -> float:
    """Logit change when every EMPTY (padded) group's pooled state is replaced by large noise."""
    base = fwd(m, x)
    real = sp.pool_groups

    def bad(feat, score, flat, n_flat):
        z, cnt = real(feat, score, flat, n_flat)
        noise = 1e3 * torch.randn(z.shape, generator=torch.Generator().manual_seed(5))
        return torch.where((cnt == 0)[:, None], noise, z), cnt
    monkeypatch.setattr(sp, "pool_groups", bad)
    try:
        return float((fwd(m, x) - base).abs().max())
    finally:
        monkeypatch.setattr(sp, "pool_groups", real)


# ---------------------------------------------------------------- causality
@pytest.mark.parametrize("arm", list(ARMS))
def test_strictly_causal_in_every_position_eval(arm):
    m, x = make(arm), tokens()
    assert causality_violation(m, x) == 0.0
    base = fwd(m, x)                                           # the detector is not blind: later logits move
    assert float((fwd(m, flip(x, 20))[:, 20:] - base[:, 20:]).abs().max()) > 1e-4


@pytest.mark.parametrize("arm", list(ARMS))
def test_strictly_causal_in_train_mode_with_sampled_depth(arm):
    m, x = make(arm).train(), tokens(seed=3)
    assert causality_violation(m, x, ts=range(1, L, 3)) == 0.0


@pytest.mark.parametrize("arm", SPAN_ARMS)
def test_strictly_causal_when_rows_overflow_max_spans(arm):
    m, x = make(arm, max_spans=4), tokens(seed=2)
    lay = m.span_layout(x)
    assert bool((lay.n_bound > lay.n_src).all())               # every row overflows the 4-span axis
    assert causality_violation(m, x) == 0.0


def test_causal_on_boundary_edges():
    """The last token of a span, the first token after it and every chunk edge, explicitly."""
    m, x = make("pair"), tokens(seed=4)
    bpos, _ = m.rule.cut(x[0].numpy(), 0)
    ts = sorted({int(p) + d for p in bpos for d in (0, 1) if 0 < p + d < L} | {15, 16, 17, 31, 32, 33, 62, 63})
    assert len(ts) > 20 and causality_violation(m, x, ts) == 0.0


# ---------------------------------------------------------------- who receives a span's state
def _probe(m, x, j, which=0):
    """Replace the `which`-th level's group states by a one-hot on group j; return that level's u [B, L, d]."""
    calls, real = [], m._broadcast

    def bc(out, gain, recv_flat, B, L_):
        if len(calls) == which:
            probe = torch.zeros_like(out)
            probe[:, j] = 1.0
            out = probe
        u = real(out, gain, recv_flat, B, L_)
        calls.append(u)
        return u
    m._broadcast = bc
    try:
        fwd(m, x)
    finally:
        del m._broadcast
    return calls[which]


@pytest.mark.parametrize("arm", SPAN_ARMS)
def test_span_update_reaches_only_the_next_span(arm):
    from morph.model.tul_layout import span_ids_from_ids
    m, x = make(arm), tokens(seed=6)
    lay = m.span_layout(x)
    sid = torch.from_numpy(span_ids_from_ids(x.numpy(), m.rule))
    for j in range(int(lay.n_src.min())):
        u = _probe(m, x, j)
        got = u.abs().sum(-1) > 0
        want = torch.minimum(sid, lay.n_src[:, None]) == j + 1         # tokens of span j + 1 (tail: the last)
        assert torch.equal(got, want), j
        for b in range(x.shape[0]):                                     # nothing at or before span j's last token
            last = int(m.rule.cut(x[b].numpy(), 0)[0][j])
            assert not bool(got[b, : last + 1].any())
            assert bool(got[b, last + 1:].any())


@pytest.mark.parametrize("size", [16, 8, 4, 2])
def test_chunk_update_reaches_only_tokens_after_its_last_token(size):
    spec = CHUNK(size)
    group, recv, G = chunk_geometry(L, spec, "cpu")
    assert G == L // size and spec.offset == size // 2 and spec.shift == size - 1
    for g in range(G):
        last = int((group == g).nonzero().max())
        rec = (recv == g).nonzero().flatten()
        assert rec.numel() > 0 or g == G - 1
        if rec.numel():
            assert int(rec.min()) >= last, (size, g)
            assert int(rec.min()) == last - 0 or size > 1            # single-token overlap at most
    # the dropped incomplete chunk (eq. 14) has no tokens that read anything it produced
    assert int((group < 0).sum()) <= size - 1 + 1


def test_chunk_probe_matches_geometry():
    m, x = make("published"), tokens(seed=7)
    group, recv, G = chunk_geometry(L, CHUNK(16), "cpu")
    for g in (0, 1, G - 1):
        u = _probe(m, x, g)
        assert torch.equal((u.abs().sum(-1) > 0)[0], recv == g)


def test_levelspec_refuses_a_leaky_shift():
    with pytest.raises(ValueError, match="causality"):
        LevelSpec("chunk", size=8, shift=3)
    with pytest.raises(ValueError):
        LevelSpec("span", size=4)


# ---------------------------------------------------------------- padded spans
@pytest.mark.parametrize("arm", SPAN_ARMS)
def test_padded_spans_never_influence_a_real_token(arm, monkeypatch):
    m, x = make(arm), tokens(seed=8)
    lay = m.span_layout(x)
    assert int(lay.n_src.max()) < 24                           # the axis is padded
    assert bool((lay.group < lay.n_src[:, None]).all()) and bool((lay.recv < lay.n_src[:, None]).all())
    assert garbage_violation(m, x, monkeypatch) == 0.0


def test_span_layout_matches_morphs_span_ids():
    from morph.model.tul_layout import span_ids_from_ids
    x = tokens(B=4, seed=9)
    for cap in (24, 5):
        lay = span_layout(x, toy_rule(), cap)
        sid = torch.from_numpy(span_ids_from_ids(x.numpy(), toy_rule()))
        n = lay.n_src[:, None]
        assert torch.equal(lay.group, torch.where(sid < n, sid, torch.full_like(sid, -1)))
        assert torch.equal(lay.recv, torch.minimum(sid, n) - 1)
        assert bool((lay.n_src == torch.clamp(lay.n_bound, max=cap)).all())


# ---------------------------------------------------------------- the update forced to zero
def manual_no_loop(m, x):
    """Plain's own prelude + ln_prelude + coda + head, no loop: what a zero update must give."""
    tr, cfg = m.transformer, m.config
    fr = m.freqs_cis[:, : x.shape[1]]
    h = tr.wte(x) * m.emb_scale
    for i, b in enumerate(tr.prelude):
        h = b(h, fr, None, ve=m.value_embeds[str(i)](x) if str(i) in m.value_embeds else None)
    h = tr.ln_prelude(h)
    off = cfg.n_layers_in_prelude + cfg.n_layers_in_recurrent_block
    for j, b in enumerate(tr.coda):
        k = str(off + j)
        h = b(h, fr, None, ve=m.value_embeds[k](x) if k in m.value_embeds else None)
    return m.lm_head(tr.ln_f(h)).float() * cfg.init.logit_scale


@pytest.mark.parametrize("arm", ["published", "span", "pair"])
def test_zero_update_equals_the_plain_prelude_coda_path(arm, monkeypatch):
    m, x = make(arm), tokens(seed=10)
    monkeypatch.setattr(SpiralParcae, "_broadcast", lambda self, out, gain, rf, B, L_: out.new_zeros(B, L_, out.shape[-1]))
    with torch.no_grad():
        got, want = fwd(m, x), manual_no_loop(m, x)
    assert float((got - want).abs().max()) < 1e-6
    assert float(want.abs().max()) > 1e-3


def test_zero_update_equals_a_gpt_built_from_the_same_prelude_and_coda_weights(monkeypatch):
    """The testbed's gpt arm has no ln_prelude, so the comparison removes it (Identity) and copies the
    prelude blocks to gpt layers 0..1 and the coda blocks to 2..3, the value embeddings of those layers,
    the embedding and the final norm."""
    m, x = make("pair"), tokens(seed=11)
    cfg = m.config
    gc = parcae_lm.create_config("gpt-small-140m", n_embd=32, num_attention_heads=2, num_key_value_heads=2,
                                 intermediate_size=64, n_layer=4, vocab_size=V, block_size=L + 32,
                                 use_fused_head=False)
    gpt = gc.construct_model(gradient_checkpointing=False).eval()
    if gpt.emb_scale != m.emb_scale or gc.init.logit_scale != cfg.init.logit_scale:
        pytest.skip("gpt and parcae configs scale the embedding or the logits differently")
    for i in range(2):
        gpt.transformer.h[i].load_state_dict(m.transformer.prelude[i].state_dict())
        gpt.transformer.h[2 + i].load_state_dict(m.transformer.coda[i].state_dict())
    gpt.transformer.wte.load_state_dict(m.transformer.wte.state_dict())
    gpt.transformer.ln_f.load_state_dict(m.transformer.ln_f.state_dict())
    gpt.value_embeds["1"].load_state_dict(m.value_embeds["1"].state_dict())
    gpt.value_embeds["3"].load_state_dict(m.value_embeds["5"].state_dict())     # coda layer 1 = effective layer 5
    m.transformer.ln_prelude = torch.nn.Identity()
    monkeypatch.setattr(SpiralParcae, "_broadcast", lambda self, out, gain, rf, B, L_: out.new_zeros(B, L_, out.shape[-1]))
    with torch.no_grad():
        got = fwd(m, x)
        want = gpt(x, return_logits=True)["logits"]
    assert float((got - want).abs().max()) < 1e-6 and float(want.abs().max()) > 1e-3


# ---------------------------------------------------------------- MeSH buffer
def test_mesh_buffer_starts_with_the_embeddings_and_the_transitional_read_is_as_defined():
    m, x = make("mesh"), tokens(seed=12)
    mesh, S = m.mesh, m.spiral.n_slots
    assert S == len(m.spiral.levels) + 3 == 6
    seen = {}
    real = mesh.start

    def start(emb, v):
        out = real(emb, v)
        seen.update(emb=emb, v=v, m=out[0], h=out[1])
        return out
    mesh.start = start
    fwd(m, x)
    del mesh.start
    emb, v = seen["emb"], seen["v"]
    with torch.no_grad():
        assert torch.equal(emb, m.transformer.wte(x) * m.emb_scale)       # the embedding the prelude reads
        ww = torch.softmax(v @ torch.stack(list(mesh.write[0].w)).T + mesh.write[0].b, -1)
        wr = torch.softmax(v @ torch.stack(list(mesh.read[0].w)).T + mesh.read[0].b, -1)
        buf = torch.zeros(S, *emb.shape)
        buf[0] = emb
        for b in range(S):
            buf[b] = buf[b] + v * ww[..., b: b + 1]
        want = sum(buf[b] * wr[..., b: b + 1] for b in range(S))
    assert float((seen["h"] - want).abs().max()) < 1e-6 and float((seen["m"] - buf).abs().max()) < 1e-6
    # zero routers: weight 1/S everywhere, so slot 0 holds emb + v/S, the rest v/S, and h = (emb + v) / S
    for r in list(mesh.write) + list(mesh.read):
        for p in r.parameters():
            torch.nn.init.zeros_(p)
    m0, h0 = mesh.start(emb, v)
    assert float((m0[0] - (emb + v / S)).abs().max()) < 1e-6 and float((m0[1:] - v / S).abs().max()) < 1e-6  # noqa
    assert float((h0 - (emb + v) / S).abs().max()) < 1e-6


def test_mesh_buffer_slot_count_follows_the_levels():
    for n in (1, 2, 5):
        levels = (LevelSpec("span"),) + tuple(CHUNK(2 ** (i + 1)) for i in range(n - 1))
        assert SpiralConfig(levels=levels, topology="mesh").n_slots == n + 3
    with pytest.raises(ValueError):
        SpiralConfig(levels=(LevelSpec("span"),), topology="highway")


# ---------------------------------------------------------------- depth
class _Count:
    def __init__(self, m):
        self.n, self.real, self.m = 0, m._core_pass, m

    def __enter__(self):
        def cp(*a, **k):
            self.n += 1
            return self.real(*a, **k)
        self.m._core_pass = cp
        return self

    def __exit__(self, *a):
        del self.m._core_pass


@pytest.mark.parametrize("arm,extra", [("span", 0), ("pair", 2), ("mesh", 2)])
@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_depth_k_runs_exactly_k_span_passes(arm, extra, k):
    m, x = make(arm), tokens(seed=13)
    with _Count(m) as c:
        fwd(m, x, depth=k)
    assert c.n == k + extra                                    # chunk levels stay at one pass


def test_published_spiral_has_fixed_depth_and_raises_on_a_depth_override():
    m, x = make("published"), tokens(seed=14)
    with _Count(m) as c:
        fwd(m, x)
    assert c.n == 4
    for d in (1, 6):
        with pytest.raises(ValueError, match="no sampled-depth level"):
            fwd(m, x, depth=d)


def test_sampled_depths_follow_parcaes_law_truncated_at_max_depth():
    m = make("span").train()
    tot = []
    for step in range(300):
        m.step = step
        n, k = m._depths(6, None)
        assert int((n + k).max()) <= 5 and int((n + k).min()) >= 1 and int(k.max()) <= 3 and int(n.min()) >= 0
        tot.append((n + k).float())
    assert 2.5 < float(torch.cat(tot).mean()) < 4.5            # Poisson(3) clamped to [1, 5]
    assert not m.eval()._depths(2, None)[1].tolist() == [0, 0]
    assert m._depths(2, None)[1].tolist() == [3, 3]            # eval default = mean recurrence


def test_per_row_gating_equals_running_each_row_alone():
    m = make("span").eval()
    g = torch.Generator().manual_seed(3)
    z = torch.randn(3, 24, 32, generator=g)
    x0 = torch.randn(3, 24, 32, generator=g)
    ves = [None, None]
    fr = m.freqs_cis[:, :24]
    n, k = torch.tensor([0, 2, 1]), torch.tensor([3, 1, 2])
    m.initialize_state = lambda zz: x0[: zz.shape[0]].clone()
    with torch.no_grad():
        both = m._loop(z, ves, fr, n, k)
        for b in range(3):
            m.initialize_state = lambda zz, b=b: x0[b: b + 1].clone()
            alone = m._loop(z[b: b + 1], ves, fr, torch.zeros(1, dtype=torch.long), (n[b] + k[b]).view(1))
            assert float((both[b] - alone[0]).abs().max()) < 1e-6, b


# ---------------------------------------------------------------- pool
def test_pool_is_a_softmax_weighted_mean_per_group_and_empty_groups_are_zero():
    g = torch.Generator().manual_seed(1)
    B, Lx, C, G = 2, 11, 5, 4
    feat, score = torch.randn(B, Lx, C, generator=g), torch.randn(B, Lx, generator=g)
    group = torch.randint(-1, G - 1, (B, Lx), generator=g)             # group 3 is always empty
    flat = sp.flat_index(group, B, G)
    z, cnt = pool_groups(feat, score, flat, B * G)
    for b in range(B):
        for gi in range(G):
            sel = group[b] == gi
            if sel.any():
                w = torch.softmax(score[b][sel], 0)
                assert torch.allclose(z[b * G + gi], (w[:, None] * feat[b][sel]).sum(0), atol=1e-6)
                assert int(cnt[b * G + gi]) == int(sel.sum())
            else:
                assert float(z[b * G + gi].abs().max()) == 0.0 and int(cnt[b * G + gi]) == 0


# ---------------------------------------------------------------- training and wiring
@pytest.mark.parametrize("arm", list(ARMS))
def test_one_train_step_reaches_every_new_parameter(arm):
    from lxtul.plain_pointer import PlainPointer
    m = PlainPointer(make(arm), 4).train()
    x = tokens(seed=15)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    lay = m.inner.span_layout(x) if m.inner.has_span_level else None
    out = m(x, labels=y, **({"layout": lay} if lay is not None else {}))
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    names = [n for n, p in m.named_parameters() if "pool_w" in n or "mesh" in n or "transformer.C" in n]
    assert names
    for n, p in m.named_parameters():
        if n in names or "core_block" in n:
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
    for i in range(len(m.inner.pool_w)):
        if arm == "token_skip_loop":                          # a softmax over one score is 1: the scorer gets exactly no gradient
            assert float(m.inner.pool_w[i].grad.abs().sum()) == 0.0, i
        else:
            assert float(m.inner.pool_w[i].grad.abs().sum()) > 0, i
    if arm == "mesh":
        assert all(float(r.w[0].grad.abs().sum()) > 0 for r in list(m.inner.mesh.write) + list(m.inner.mesh.read))
    assert m(x, labels=y, **({"layout": lay} if lay is not None else {}))["loss"].shape == ()


def test_pool_and_router_tensors_go_to_adamw_not_muon():
    import recpre.optim
    m = make("mesh")
    groups = recpre.optim.get_muon_param_groups_from_config(
        m.named_parameters(), {"lr": 1e-3, "muon_lr": 1e-3}, no_weight_decay_for_bias_and_norm_params=True)
    ada = {id(p) for g in groups if g["kind"] == "adamw" for p in g["params"]}
    for n, p in m.named_parameters():
        if "pool_w" in n or "mesh" in n:
            assert id(p) in ada, n


def test_labels_and_unsupported_arguments_raise():
    m, x = make("span"), tokens()
    for kw in ({"labels": x}, {"num_steps_pair": torch.tensor([1, 0])}, {"position_ids": torch.arange(L)}):
        with pytest.raises(ValueError):
            m(x, **kw)


# ---------------------------------------------------------------- build.py wiring (hydra, tiny shape)
CONF = str(Path(sp.__file__).parent / "configs")
SEQ = 48
TINY = ["shape.n_embd=64", "shape.num_attention_heads=2", "shape.num_key_value_heads=2",
        "shape.intermediate_size=128", f"shape.block_size={SEQ + 32}", "run.name=t", "run.wandb_mode=disabled",
        "shape.n_layers_in_prelude=2", "shape.n_layers_in_recurrent_block=2", "shape.n_layers_in_coda=2"]
HYDRA = ["spiral_published", "spiral_span", "spiral_span_pair", "spiral_span_pair_mesh",
         "spiral_chunk4_loop", "spiral_chunk2_loop", "spiral_token_skip_loop"]


def _built(arm, monkeypatch, extra=()):
    from lxtul.build import build
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    with initialize_config_dir(CONF, version_base=None):
        cfg = compose("train", overrides=[f"arm={arm}", *TINY, *extra])
    torch.manual_seed(0)
    model, mcfg, rt = build(cfg, str(cfg.arm.kind), compile=False)
    return model, cfg, mcfg


@pytest.mark.parametrize("arm", HYDRA)
def test_build_wraps_the_pointer_head_and_shares_the_plain_arms_init(arm, monkeypatch):
    m, cfg, mcfg = _built(arm, monkeypatch, ["++arm.spiral.max_spans=24"])
    plain, _, _ = _built("plain_pointer", monkeypatch)
    assert m.pointer.h == 4 and m.inner.spiral.levels
    sd, pd = m.inner.state_dict(), plain.inner.state_dict()
    assert set(pd) < set(sd)                                              # the plain parameters, plus the new ones
    assert all(torch.equal(sd[k], pd[k]) for k in pd if "freqs" not in k), "shared init differs from plain"
    extra = set(sd) - set(pd)
    assert extra and all(("pool_w" in k or "mesh" in k or "up_norm" in k) for k in extra)
    assert ("mesh" in "".join(extra)) == (arm == "spiral_span_pair_mesh")
    assert m.inner.has_sampled_depth == (arm != "spiral_published")
    assert m.inner.has_span_level == ("span" in arm)


def test_depth_sweep_scores_a_spiral_with_span_level_per_depth_and_one_without_once(monkeypatch):
    from lxtul.evaluate import depth_sweep
    for arm, differs in (("spiral_span", True), ("spiral_published", False)):
        m, cfg, mcfg = _built(arm, monkeypatch, ["++arm.spiral.max_spans=24"])
        for p in m.inner.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.data.normal_(0, 0.05)
        m.inner.initialize_state = lambda z: torch.full_like(z, 0.01)     # the random state init would differ per call
        x = tokens(B=2, seed=16, n=SEQ)
        y = torch.roll(x, -1, 1)
        y[:, -1] = -100
        lay = SimpleNamespace(slot_mask=torch.zeros_like(x, dtype=torch.bool))
        sw = depth_sweep(m, "spiral", [(x, y, lay)], [1, 6], slot_id=3, device="cpu")
        assert (sw["sums"]["1"] != sw["sums"]["6"]) == differs
        assert sw["count"] == [SEQ - 1, SEQ - 1]
        assert len(sw["spans"]["n_src"]) == 2 and all(a <= b for a, b in zip(sw["spans"]["n_src"], sw["spans"]["n_bound"]))
        with torch.no_grad():                                              # equals the model's own mixture NLL
            ref = m.token_nll(x, y, layout=m.inner.span_layout(x), depth=6 if differs else None).sum(1).double().tolist()
        assert sw["sums"]["6"] == pytest.approx(ref, rel=1e-4)


# ---------------------------------------------------------------- flop tally
def test_attention_tally_sees_the_short_group_sequences_and_layer_equivalents():
    import torch as t
    from parcae_lm.modules import mixer
    import lxtul.attention as A
    from lxtul.flop_tally import install_attn_tally, spiral_layer_equivalents
    m, x = make("published"), tokens(B=2, seed=17)
    saved = (mixer.flash_attn.flash_attn_func, A._flex, A._mask)
    try:
        tally = install_attn_tally(m)
        fwd(m, x)
    finally:
        mixer.flash_attn.flash_attn_func, A._flex, A._mask = saved
    heads, dh, B = 2, 16, 2
    want = sum(2 * 4.0 * B * heads * g * g * dh * (g + 1) / (2 * g) for g in (L // 16, L // 8, L // 4, L // 2))
    assert tally.calls[("core", "flash")] == 8
    assert tally.fwd["core"] == pytest.approx(want)
    assert tally.calls[("prelude", "flash")] == 2 and tally.calls[("coda", "flash")] == 2
    lv = m.spiral.levels
    assert spiral_layer_equivalents(lv, L, 2, 2, 2, 0, 0) == pytest.approx(4 + 2 * (1 / 16 + 1 / 8 + 1 / 4 + 1 / 2))
    pair = ARMS["pair"]["levels"]
    assert spiral_layer_equivalents(pair, 1024, 4, 6, 4, 51, 6) == pytest.approx(8 + 6 * 51 * 6 / 1024 + 6 / 4 + 6 / 2)


# ---------------------------------------------------------------- the guards are tested by breaking them
def test_broken_chunk_shift_is_caught(monkeypatch):
    real = sp.chunk_geometry

    def no_shift(L_, spec, device):
        g, _, G = real(L_, spec, device)
        return g, g.clone(), G                                 # a token reads its OWN chunk, shift 0
    monkeypatch.setattr(sp, "chunk_geometry", no_shift)
    assert causality_violation(make("published"), tokens(), ts=range(1, L, 3)) > 1e-4


def test_broken_span_next_span_rule_is_caught(monkeypatch):
    real = sp.span_layout

    def own_span(ids, rule, max_spans):
        lay = real(ids, rule, max_spans)
        return SpanLayout(lay.group, lay.group.clone(), lay.n_src, lay.n_bound)   # a span writes into itself
    monkeypatch.setattr(sp, "span_layout", own_span)
    assert causality_violation(make("span"), tokens(), ts=range(1, L, 3)) > 1e-4


def test_broken_padding_isolation_is_caught(monkeypatch):
    def mix(self, x, z, ves, freqs):                           # every group sees the mean of ALL groups
        return x + z + x.mean(dim=1, keepdim=True)
    monkeypatch.setattr(SpiralParcae, "_core_pass", mix)
    m, x = make("span"), tokens(seed=8)
    assert garbage_violation(m, x, monkeypatch) > 1e-3
    assert causality_violation(m, x, ts=range(1, L, 5)) > 1e-4


def test_broken_mesh_router_is_caught(monkeypatch):
    def future_router(self, h):                                # routes from the sequence mean: sees the future
        h = h.mean(dim=1, keepdim=True).expand_as(h)
        logits = h.float() @ torch.stack(list(self.w)).float().T + self.b.float()
        return torch.softmax(logits, dim=-1).permute(2, 0, 1).unsqueeze(-1)
    monkeypatch.setattr(sp.Router, "forward", future_router)
    assert causality_violation(make("mesh"), tokens(), ts=range(1, L, 3)) > 1e-4


def test_broken_zero_update_reference_is_sensitive(monkeypatch):
    m, x = make("span"), tokens(seed=10)
    with torch.no_grad():                                      # with the real update the logits differ from the no-loop path
        assert float((fwd(m, x) - manual_no_loop(m, x)).abs().max()) > 1e-4


# ---------------------------------------------------------------- update scale (2026-10-15 fix)
def _level_updates(m, x):
    """Every level's broadcast update u [B, L, d], in level order."""
    calls, real = [], m._broadcast

    def bc(out, gain, recv_flat, B, L_):
        u = real(out, gain, recv_flat, B, L_)
        calls.append(u)
        return u
    m._broadcast = bc
    try:
        fwd(m, x)
    finally:
        del m._broadcast
    return calls


def _check_update_scale(m, x):
    """With the up-norm weight at its init (1), a receiving token's update has RMS exactly 1/sqrt(g): g the
    chunk size, or the length of the span it reads. The unnormed sqrt(g) gain read 536 at the first level of
    a trained published spiral (RMS(h0) 1.04) and compounded to 6.9e9 by the fourth."""
    lay = m.span_layout(x) if m.has_span_level else None
    for lvl, (spec, u) in enumerate(zip(m.spiral.levels, _level_updates(m, x))):
        rms = u.float().pow(2).mean(-1).sqrt()                          # [B, L], 0 where nothing is received
        got = rms[rms > 0]
        if spec.kind == "chunk":
            want = torch.full_like(got, 1.0 / math.sqrt(spec.size))
        else:
            cnt = torch.zeros(lay.recv.shape[0], m.spiral.max_spans)
            for b in range(cnt.shape[0]):
                g = lay.group[b]
                cnt[b] = torch.bincount(g[g >= 0], minlength=m.spiral.max_spans)[: m.spiral.max_spans].float()
            src = torch.gather(cnt, 1, lay.recv.clamp_min(0))[lay.recv >= 0]
            want = src.rsqrt()
        assert got.numel() > 0, lvl
        torch.testing.assert_close(got, want, rtol=2e-3, atol=1e-5)


@pytest.mark.parametrize("arm", list(ARMS))
def test_update_rms_is_one_over_sqrt_group_size(arm):
    _check_update_scale(make(arm), tokens(seed=11))


def test_broken_update_norm_is_caught():
    m = make("published")
    m.up_norm = torch.nn.ModuleList(torch.nn.Identity() for _ in m.spiral.levels)
    with pytest.raises(AssertionError):
        _check_update_scale(m, tokens(seed=11))


# ---------------------------------------------------------------- Poisson-depth chunk loops (2026-10-15)


@pytest.mark.parametrize("arm,size", list(LOOP_ARMS.items()))
@pytest.mark.parametrize("train", [False, True])
def test_loop_arms_are_strictly_causal(arm, size, train):
    m, x = make(arm), tokens(seed=20)
    m.train(train)
    assert causality_violation(m, x, ts=range(1, L, 1 if not train else 3)) == 0.0
    base = fwd(m, x)                                           # the detector is not blind: later logits move
    assert float((fwd(m, flip(x, 20))[:, 20:] - base[:, 20:]).abs().max()) > 1e-4


@pytest.mark.parametrize("size", [4, 2])
def test_loop_arm_with_a_leaky_shift_is_caught(size):
    import copy
    import dataclasses
    m = make({4: "chunk4_loop", 2: "chunk2_loop"}[size])
    bad = copy.copy(m.spiral.levels[0])                             # ARMS shares its LevelSpecs: never mutate in place
    bad.__dict__["shift"] = size - 2                                # bypass the frozen dataclass check, as the guard test does
    m.spiral = dataclasses.replace(m.spiral, levels=(bad,))
    assert m.spiral.levels[0].shift == size - 2 and ARMS[{4: "chunk4_loop", 2: "chunk2_loop"}[size]]["levels"][0].shift == size - 1
    assert causality_violation(m, tokens(seed=20), ts=range(1, L, 3)) > 1e-4
    with pytest.raises(ValueError, match="causality"):
        LevelSpec("chunk", size=size, shift=size - 2, depth="poisson")


def test_size_one_guard_and_geometry():
    spec = LevelSpec("chunk", size=1, depth="poisson")
    assert (spec.offset, spec.shift, spec.passes) == (0, 0, 1)
    group, recv, G = chunk_geometry(L, spec, "cpu")
    assert G == L and torch.equal(group, torch.arange(L)) and torch.equal(recv, torch.arange(L))
    with pytest.raises(ValueError, match="size >= 1"):
        LevelSpec("chunk", size=0)


def test_size_one_pooling_is_the_identity():
    m, x = make("token_skip_loop"), tokens(seed=21)
    seen = {}
    real = sp.pool_groups

    def spy(feat, score, flat, n_flat):
        z, cnt = real(feat, score, flat, n_flat)
        seen.update(feat=feat, z=z, cnt=cnt)
        return z, cnt
    sp.pool_groups = spy
    try:
        fwd(m, x)
    finally:
        sp.pool_groups = real
    B, d = x.shape[0], m.config.n_embd
    assert torch.equal(seen["z"].view(B, L, -1), seen["feat"].float())     # every pooled row IS its token's feature row
    assert torch.equal(seen["z"].view(B, L, -1)[..., :d], seen["feat"][..., :d].float())
    assert bool((seen["cnt"] == 1).all())


@pytest.mark.parametrize("arm", list(LOOP_ARMS))
def test_loop_arm_depth_is_honored(arm):
    m, x = make(arm), tokens(seed=22)
    a, b = fwd(m, x, depth=1), fwd(m, x, depth=3)
    assert float((a - b).abs().max()) > 1e-4                     # depth 1 and 3 differ
    assert torch.equal(a, fwd(m, x, depth=1)) and torch.equal(b, fwd(m, x, depth=3))
    with _Count(m) as c:
        fwd(m, x, depth=3)
    assert c.n == 3
    with _Count(m) as c:                                         # eval, no depth: mean_recurrence (3 in the toy config)
        e = fwd(m, x)
    assert c.n == 3 and torch.equal(e, b)


@pytest.mark.parametrize("arm", list(LOOP_ARMS))
def test_loop_arm_train_mode_samples_parcaes_law(arm):
    m, x = make(arm).train(), tokens(seed=23)
    tot = []
    for step in range(100):
        m.step = step
        n, k = m._depths(6, None)
        tot.append(int((n + k).max()))
        assert int((n + k).max()) <= 5 and int(k.max()) <= 3
    assert len(set(tot)) > 1                                     # the draw varies
    with _Count(m) as c:
        m.step = 0
        torch.manual_seed(1)
        m(x, return_logits=True)
    assert 1 <= c.n <= 5 * 6


def test_published_and_fixed_chunk_arms_still_refuse_a_depth():
    for arm in ("published",):
        with pytest.raises(ValueError, match="sampled-depth"):
            fwd(make(arm), tokens(), depth=2)


def test_depth_config_guards():
    with pytest.raises(ValueError, match="at most one sampled-depth"):
        SpiralConfig(levels=(LevelSpec("span"), LevelSpec("chunk", size=4, depth="poisson")))
    with pytest.raises(ValueError, match="at most one sampled-depth"):
        SpiralConfig(levels=(LevelSpec("chunk", size=4, depth="poisson"), LevelSpec("chunk", size=2, depth="poisson")))
    with pytest.raises(ValueError, match="span level always samples"):
        LevelSpec("span", depth="poisson")
    with pytest.raises(ValueError, match="passes must stay 1"):
        LevelSpec("chunk", size=4, depth="poisson", passes=2)
    with pytest.raises(ValueError, match="unknown depth"):
        LevelSpec("chunk", size=4, depth="geometric")
    # a poisson chunk level beside FIXED chunk levels is allowed
    SpiralConfig(levels=(LevelSpec("chunk", size=8), LevelSpec("chunk", size=4, depth="poisson")))


def test_loop_arms_in_a_mixed_stack_run_fixed_levels_once():
    torch.manual_seed(0)
    m = make("chunk4_loop")
    m2 = SpiralParcae(m.config, SpiralConfig(levels=(CHUNK(8), LevelSpec("chunk", size=4, depth="poisson")), max_depth=5),
                      toy_rule()).eval()
    x = tokens(seed=24)
    with _Count(m2) as c:
        fwd(m2, x, depth=3)
    assert c.n == 3 + 1
    assert causality_violation(m2, x, ts=range(1, L, 3)) == 0.0


# ---------------------------------------------------------------- Block-Parcae: the series hierarchy (2026-10-15)


def _cross_chunk_leak(m, x, g: int, monkeypatch, zero_update: bool, zero_memory: bool = False) -> float:
    """Largest logit change OUTSIDE the perturbed token's chunk, over perturbed tokens. With the update zeroed, a
    series model has no route between chunks, so this must be 0. `zero_memory` also zeroes the loop memory the
    block-memory coda attends over."""
    if zero_memory:
        real = SpiralParcae._block_memory_coda
        monkeypatch.setattr(SpiralParcae, "_block_memory_coda",
                            lambda self, x, mem, ids, off: real(self, x, torch.zeros_like(mem), ids, off))
    if zero_update:
        monkeypatch.setattr(SpiralParcae, "_broadcast",
                            lambda self, out, gain, recv_flat, B, L: out.new_zeros(B, L, out.shape[-1]))
    base, worst = fwd(m, x), 0.0
    for t in range(1, x.shape[1], 3):
        lg = fwd(m, flip(x, t))
        c0, c1 = (t // g) * g, (t // g) * g + g
        diff = (lg - base).abs()
        diff[:, c0:c1] = 0.0
        worst = max(worst, float(diff.max()))
    return worst


@pytest.mark.parametrize("arm,g", list(BLOCK_ARMS.items()))
def test_block_coda_has_no_route_between_chunks_but_the_loop(arm, g, monkeypatch):
    m, x = make(arm), tokens(seed=30)
    assert _cross_chunk_leak(m, x, g, monkeypatch, zero_update=True) == 0.0


@pytest.mark.parametrize("arm,g", list(BLOCK_ARMS.items()))
def test_block_loop_carries_earlier_chunks_to_later_ones(arm, g):
    m, x = make(arm), tokens(seed=31)
    base = fwd(m, x)
    lg = fwd(m, flip(x, 5))                                    # token 5 lives in chunk 5 // g
    nxt = (5 // g + 1) * g
    assert float((lg[:, nxt:] - base[:, nxt:]).abs().max()) > 1e-4


def test_block_detector_catches_the_anchor_skip(monkeypatch):
    """The same detector on the parallel twin (chunk-4 loop, coda reads h0 + u): h0 leaks across chunks."""
    m, x = make("chunk4_loop"), tokens(seed=30)
    assert _cross_chunk_leak(m, x, 4, monkeypatch, zero_update=True) > 1e-4


def test_block_coda_attends_only_inside_its_chunk(monkeypatch):
    """A broken local decoder (full-row coda on e + u) is caught by the same detector."""
    m, x = make("block4_loop"), tokens(seed=30)

    def full_row(self, x, input_ids, coda_off):
        freqs = self.freqs_cis[:, :x.shape[1]]
        for i, block in enumerate(self.transformer.coda):
            ve = self.value_embeds[str(coda_off + i)](input_ids) if str(coda_off + i) in self.value_embeds else None
            x = block(x, freqs, None, ve=ve)
        return x
    monkeypatch.setattr(SpiralParcae, "_block_coda", full_row)
    assert _cross_chunk_leak(m, x, 4, monkeypatch, zero_update=True) > 1e-4


def test_block_config_guards():
    with pytest.raises(ValueError, match="shift == size"):
        SpiralConfig(levels=(LevelSpec("chunk", size=4, depth="poisson"),), topology="block")      # offset 2, shift 3
    with pytest.raises(ValueError, match="exactly one chunk level"):
        SpiralConfig(levels=(LevelSpec("chunk", size=4, offset=0, shift=4), LevelSpec("chunk", size=2, offset=0, shift=2)),
                     topology="block")
    with pytest.raises(ValueError, match="exactly one chunk level"):
        SpiralConfig(levels=(LevelSpec("span"),), topology="block")


@pytest.mark.parametrize("arm", ["pub_poisson2", "pub_poisson4"])
def test_published_spiral_with_one_poisson_level_has_a_k_curve(arm):
    m, x = make(arm), tokens(seed=40)
    assert m.has_sampled_depth
    a, b = fwd(m, x, depth=1), fwd(m, x, depth=3)
    assert float((a - b).abs().max()) > 1e-4
    with _Count(m) as c:                                         # 3 fixed levels once + the poisson level 3 times
        fwd(m, x, depth=3)
    assert c.n == 3 + 3


def test_published_spiral_arm_configs_compose():
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1] / "configs" / "arm"
    for name, want in (("spiral_pub_poisson2", 2), ("spiral_pub_poisson4", 4)):
        sc = SpiralConfig.from_cfg(OmegaConf.load(root / f"{name}.yaml").spiral)
        assert [l.size for l in sc.levels] == [16, 8, 4, 2]
        assert [l.size for l in sc.levels if l.depth == "poisson"] == [want]


# ---------------------------------------------------------------- Block-Parcae with a loop memory (spiral.block_memory)
def _strip_memory(arm: str) -> str:
    return arm.replace("blockmem", "block")


@pytest.mark.parametrize("arm", list(BLOCKMEM_ARMS))
def test_block_memory_off_is_bit_identical_to_the_block_arm(arm):
    """The key off IS today's block path: same parameters, same logits, bit for bit."""
    off = make(_strip_memory(arm))
    assert not off.spiral.block_memory
    on = make(arm)
    assert on.spiral.block_memory
    assert [n for n, _ in off.named_parameters()] == [n for n, _ in on.named_parameters()]
    x = tokens(seed=50)
    # the off model must not have been touched by the key (no flex swap, local coda)
    assert type(off.transformer.coda[0].attn).__name__ == "CausalSelfAttention"
    assert torch.equal(fwd(off, x), fwd(make(_strip_memory(arm)), x))
    assert float((fwd(on, x) - fwd(off, x)).abs().max()) > 1e-4          # and on is a different function


@pytest.mark.parametrize("arm,g", list(BLOCKMEM_ARMS.items()))
@pytest.mark.parametrize("train", [False, True])
def test_block_memory_is_strictly_causal(arm, g, train):
    m, x = make(arm), tokens(seed=51)
    m.train(train)
    assert causality_violation(m, x, ts=range(1, L, 1 if not train else 3)) == 0.0


@pytest.mark.parametrize("arm,g", list(BLOCKMEM_ARMS.items()))
def test_block_memory_with_memory_and_update_zeroed_has_no_route_between_chunks(arm, g, monkeypatch):
    """The series property: the loop (update + memory) is the only route; cut both and nothing leaves a chunk."""
    m, x = make(arm), tokens(seed=52)
    assert _cross_chunk_leak(m, x, g, monkeypatch, zero_update=True, zero_memory=True) == 0.0


@pytest.mark.parametrize("arm,g", list(BLOCKMEM_ARMS.items()))
def test_block_memory_alone_reaches_chunks_beyond_the_next(arm, g, monkeypatch):
    """Update zeroed, memory live: a token in chunk c moves logits of chunks >= c + 1 (the memory m_c), including
    chunks beyond c + 1 that the previous-chunk update could never reach; the rest of chunk c moves (same chunk)."""
    m, x = make(arm), tokens(seed=53)
    monkeypatch.setattr(SpiralParcae, "_broadcast",
                        lambda self, out, gain, recv_flat, B, L: out.new_zeros(B, L, out.shape[-1]))
    base = fwd(m, x)
    t = 5
    c = t // g
    diff = (fwd(m, flip(x, t)) - base).abs()
    assert float(diff[:, :t].max()) == 0.0                                # causal: nothing before the token moves
    assert float(diff[:, (c + 1) * g:(c + 2) * g].max()) > 1e-4           # the next chunk reads m_c
    assert float(diff[:, (c + 2) * g:].max()) > 1e-4                      # and so do chunks further on


@pytest.mark.parametrize("arm", list(BLOCKMEM_ARMS))
def test_block_memory_with_the_memory_live_moves_later_chunks(arm):
    m, x = make(arm), tokens(seed=54)
    g = BLOCKMEM_ARMS[arm]
    base = fwd(m, x)
    diff = (fwd(m, flip(x, 5)) - base).abs()
    assert float(diff[:, (5 // g + 1) * g:].max()) > 1e-4


def test_block_memory_mask_matches_its_definition():
    """The mask, cell by cell, against the written rule."""
    from torch.nn.attention.flex_attention import create_mask
    G, g = 6, 4
    Lr = G * g
    bm = A.block_memory_mask(G, Lr, g, "cpu")
    mk = create_mask(bm.mask_mod, 1, 1, G + Lr, G + Lr, device="cpu")[0, 0]
    for q in range(G + Lr):
        for kv in range(G + Lr):
            if q >= G:
                i, c = q - G, (q - G) // g
                want = (kv >= G and kv - G <= i and (kv - G) // g == c) or (kv < G and kv <= c - 1)
            else:
                want = kv < G and kv <= q
            assert bool(mk[q, kv]) == want, (q, kv)


def test_block_memory_mask_that_reads_the_own_chunks_memory_is_caught(monkeypatch):
    """Break test: let token i read the memory of its OWN chunk (j <= c). That memory pools the chunk's later tokens,
    so the causality detector must fire."""
    from torch.nn.attention.flex_attention import create_block_mask

    def bad(G, Lr, g, device):
        def mod(b, h, q, kv):
            q_tok, kv_tok = q >= G, kv >= G
            tt = (kv - G <= q - G) & ((kv - G) // g == (q - G) // g)
            tm = kv <= (q - G) // g                                   # the leak: <= instead of <
            return torch.where(q_tok, torch.where(kv_tok, tt, tm), (~kv_tok) & (kv <= q))
        return create_block_mask(mod, None, None, G + Lr, G + Lr, device=device)
    monkeypatch.setattr(sp, "block_memory_mask", bad)
    assert causality_violation(make("blockmem4_loop"), tokens(seed=55), ts=range(1, L, 3)) > 1e-4


def test_block_memory_without_the_block_topology_raises():
    with pytest.raises(ValueError, match="block_memory needs topology 'block'"):
        SpiralConfig(levels=(LevelSpec("chunk", size=4, depth="poisson"),), block_memory=True)
    with pytest.raises(ValueError, match="block_memory needs topology 'block'"):
        SpiralConfig(levels=(LevelSpec("chunk", size=4, depth="poisson"),), topology="mesh", block_memory=True)


@pytest.mark.parametrize("arm", list(BLOCKMEM_ARMS))
def test_block_memory_logits_are_token_shaped_and_train_step_reaches_every_parameter(arm):
    m = make(arm).train()
    x = tokens(seed=56)
    out = m(x, return_logits=True)["logits"]
    assert out.shape == (x.shape[0], L, V)
    out.square().mean().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not p.grad.abs().any()]
    assert dead == [], dead


def test_block_memory_arm_configs_compose():
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1] / "configs" / "arm"
    for name, size in (("spiral_blockmem4_loop", 4), ("spiral_blockmem2_loop", 2)):
        sc = SpiralConfig.from_cfg(OmegaConf.load(root / f"{name}.yaml").spiral)
        assert sc.block_memory and sc.topology == "block" and sc.levels[0].size == size
        off = SpiralConfig.from_cfg(OmegaConf.load(root / (name.replace("blockmem", "block") + ".yaml")).spiral)
        assert not off.block_memory and off.levels == sc.levels
