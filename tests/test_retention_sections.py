"""retention_sections: per-section GLA attachment (loop-detach arm support)."""
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer


def _tiny(**kw):
    cfg = MORPHConfig(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=256, max_seq_len=256,
        context_len=256, n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3,
        bptt_depth=1, channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=8, bigram_hash_vocab=256,
        use_kernels=False, hc_use_kernel=False, dropout=0.0,
        retention=True, retention_layers=(1,), retention_chunk=8, **kw)
    torch.manual_seed(7)
    return MORPHTransformer(cfg)


def _has_ret(section):
    return [blk.retention is not None for blk in section]


def test_default_sections_attach_everywhere():
    m = _tiny()
    assert _has_ret(m.prelude) == [False, True]
    assert _has_ret(m.core) == [False, True]
    assert _has_ret(m.coda) == [False, True]
    assert m._core_has_retention


def test_prelude_coda_only_detaches_core():
    m = _tiny(retention_sections=("prelude", "coda"))
    assert _has_ret(m.prelude) == [False, True]
    assert _has_ret(m.core) == [False, False]
    assert _has_ret(m.coda) == [False, True]
    assert not m._core_has_retention
    # no core retention params in the state dict
    assert not any(k.startswith("core.") and ("retention" in k or "ret_gate" in k)
                   for k in m.state_dict())


def test_base_weights_identical_across_section_choice():
    # Retention attaches AFTER the base modules (RNG tail), so the base net must
    # be byte-identical whichever sections attach.
    full = _tiny().state_dict()
    detached = _tiny(retention_sections=("prelude", "coda")).state_dict()
    shared = [k for k in detached
              if "retention" not in k and "ret_gate" not in k]
    assert shared, "no base keys found"
    for k in shared:
        assert torch.equal(full[k], detached[k]), f"base weight differs: {k}"


def test_detached_forward_runs():
    m = _tiny(retention_sections=("prelude", "coda")).eval()
    x = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        out = m(x, labels=x)
    assert torch.isfinite(out["loss"])
