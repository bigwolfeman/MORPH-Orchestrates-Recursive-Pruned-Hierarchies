"""LXTUL-E Stage 2: two one-factor knobs on the parallel span head, on top of e4.

Files: morph/model/tul.py (`spandec_parallel_span_cap`, `spandec_parallel_detach`,
`_check_spandec_parallel`), morph/model/transformer.py (the head's build at J = cap and the
`_tul_spandec_par_loss` seam), morph/training/tul_setup.py (keys, manifest),
morph/configs/tul_slot_spandec_strict_e4{j4,probe}.yaml.
Filing that motivates both: lab/experiments/failures/2026-09-24-lxtul-e-stage1.md.

What each test pins:
  * both knobs at their defaults are the PRE-CHANGE model: loss, logits and every
    parameter gradient are bit-identical to a model whose seam is the pre-change method
    (kept verbatim below as `_pre_change_par_loss`), at K = 1 and on the K = 4 rollouts;
  * cap = 4 builds the head at J = 4 (four queries, a four-row position table, every other
    tensor the uncapped head's), its targets are the first four tokens of each next span
    (checked against a Python walk over the ids and the layout), its term is the brute-force
    per-span mixture over only those tokens, and the capped head is the prefix of the
    uncapped one (causal queries);
  * detach = true: the head's term sends gradient to the head's own tensors and to nothing
    else (not the loop, the codes, the front, the readout or the embedding), it still
    enters the total loss, and every logged `par_*` reading is unchanged; detach = false
    reaches the loop from the same setup (the positive control);
  * both configs compose, build, run, and differ from e4 in one key and the name;
  * the refusals hold.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from morph.model.transformer import MORPHTransformer
from morph.model.tul_spandec_parallel import code_usage_stats, mixture_span_nll
from test_tul_lxtul_e import _D_FF, K, _model, _Spy, _tc
from test_tul_strict_geometry import _pack, _runtime, _tiny

_HEAD = "tul_spandec_par."


def _pre_change_par_loss(self, h_slots, input_ids, layout, stats=None, n_rollouts=1):
    """`MORPHTransformer._tul_spandec_par_loss` at master 945b51a, VERBATIM (the reference
    the default path must reproduce bit for bit)."""
    tc = self.cfg.tul
    head = self.tul_spandec_par
    assert head is not None
    z = self._readout(h_slots)                                    # [R*B, S, C]
    w = self.embed.lm_weight().detach()
    if n_rollouts > 1:
        if head.n_codes > 1:
            raise NotImplementedError(
                "a head code table on top of loop rollouts (refused in TULConfig)")
        B0 = z.shape[0] // n_rollouts
        ids, valid = head.targets(input_ids[:B0], layout.head_rows(B0))
        zr = z.view(n_rollouts, B0, *z.shape[1:])
        _kw = dict(chunk_size=self.cfg.ce_chunk_size, mask_token_id=tc.slot_id)
        if self.training:
            logp_slot, n_tok, _sup = checkpoint(head.rollout_logp, zr, ids, valid, w,
                                                use_reentrant=False, **_kw)
        else:
            logp_slot, n_tok, _sup = head.rollout_logp(zr, ids, valid, w, **_kw)
        n_read = n_rollouts
    else:
        ids, valid = head.targets(input_ids, layout)
        logp_slot, n_tok, _sup = head.slot_logp(z, ids, valid, w,
                                                chunk_size=self.cfg.ce_chunk_size,
                                                mask_token_id=tc.slot_id)
        n_read = head.n_codes
    n = n_tok.clamp_min(1.0)
    loss = mixture_span_nll(logp_slot, n)
    if stats is not None:
        stats["par_ce"] = loss.detach()
        stats["par_n_tokens"] = n_tok.detach()
        stats["par_k"] = loss.new_tensor(float(n_read))
        if n_read > 1:
            stats.update(code_usage_stats(logp_slot.detach(), n, loss))
    return loss


def _grads(m) -> dict[str, torch.Tensor | None]:
    return {n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()}


# ── the defaults are the pre-change model ────────────────────────────────────────────


@pytest.mark.parametrize("k", [1, K], ids=["e1", "e4"])
def test_defaults_are_the_pre_change_model_bit_for_bit(k):
    """Train mode at dropout 0.1 (the K = 4 head runs checkpointed there), then eval: the
    new seam at cap 0 / detach false against the verbatim pre-change seam, same weights,
    same seed. torch.equal on the loss, every `par_*` reading, the logits and EVERY
    parameter gradient."""
    _ids, inp, lab, layout = _pack()
    new = _model(k=k, dropout=0.1)
    old = _model(k=k, dropout=0.1)
    old._tul_spandec_par_loss = _pre_change_par_loss.__get__(old)
    tc = new.cfg.tul
    assert tc.spandec_parallel_span_cap == 0 and not tc.spandec_parallel_detach
    assert new.tul_spandec_par.max_tokens == tc.bound_span_cap == 32
    for a, b in zip(new.state_dict().values(), old.state_dict().values()):
        assert torch.equal(a, b)
    outs = []
    for m in (new, old):
        m.train()
        m.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        o = m(inp, labels=lab, slot_layout=layout)
        o["loss"].backward()
        outs.append((o, _grads(m)))
    (on, gn), (oo, go) = outs
    assert torch.equal(on["loss"], oo["loss"])
    par_keys = [key for key in on if key == "par" or key.startswith("par_")]
    assert "par_ce" in par_keys and "par_n_tokens" in par_keys
    for key in par_keys:
        assert torch.equal(on[key], oo[key]), key
    assert gn.keys() == go.keys()
    n_live = 0
    for name in gn:
        if gn[name] is None:
            assert go[name] is None, name
        else:
            assert torch.equal(gn[name], go[name]), name
            n_live += 1
    assert n_live > 0 and gn[_HEAD + "z_in.weight"] is not None
    for m in (new, old):
        m.eval()
    with torch.no_grad():
        assert torch.equal(new(inp, slot_layout=layout)["logits"],
                           old(inp, slot_layout=layout)["logits"])
        assert torch.equal(new(inp, labels=lab, slot_layout=layout)["par_ce"],
                           old(inp, labels=lab, slot_layout=layout)["par_ce"])


# ── the span cap ─────────────────────────────────────────────────────────────────────


def _first_c_tokens(inp, layout, c: int) -> tuple[dict, int]:
    """{(b, s): [ids]}: the first ``c`` tokens of span s+1 for every slot s the target
    grades, found by walking the ids and the layout in Python. Slot s is graded when it
    and slot s+1 both exist (slot s+1 proves the span complete)."""
    B, S = layout.slot_valid.shape
    out, n = {}, 0
    for b in range(B):
        for s in range(S - 1):
            if not (bool(layout.slot_valid[b, s]) and bool(layout.slot_valid[b, s + 1])):
                continue
            pos = [p for p in range(inp.shape[1])
                   if int(layout.bag_id[b, p]) == s + 1 and not bool(layout.slot_mask[b, p])]
            toks = [int(inp[b, p]) for p in pos][:c]
            if toks:
                out[(b, s)] = toks
                n += len(toks)
    return out, n


def test_cap_builds_a_j4_head_whose_other_tensors_are_the_uncapped_heads():
    full, cap = _model(), _model(spandec_parallel_span_cap=4)
    hf, hc = full.tul_spandec_par, cap.tul_spandec_par
    assert hc.max_tokens == 4 and tuple(hc.pos.shape) == (4, full.cfg.d_model)
    assert hf.max_tokens == 32
    sf, sc = full.state_dict(), cap.state_dict()
    assert sf.keys() == sc.keys()
    for key in sf:
        if key == _HEAD + "pos":
            assert torch.equal(sc[key], sf[key][:4])
        else:
            assert torch.equal(sc[key], sf[key]), key


def test_cap_targets_are_the_first_four_tokens_of_each_next_span():
    _ids, inp, lab, layout = _pack(B=3, seed=2)
    m = _model(spandec_parallel_span_cap=4)
    ids, valid = m.tul_spandec_par.targets(inp, layout)
    assert ids.shape[-1] == 4 and valid.shape[-1] == 4
    want, n_want = _first_c_tokens(inp, layout, 4)
    got = {}
    for b, s in valid.any(-1).nonzero().tolist():
        got[(b, s)] = ids[b, s][valid[b, s]].tolist()
        # the valid positions are a prefix: token j is valid iff the span has > j tokens
        nv = int(valid[b, s].sum())
        assert valid[b, s, :nv].all() and not valid[b, s, nv:].any()
    assert got == want
    assert int(valid.sum()) == n_want
    # some span is longer than the cap, so the cap actually cut something here
    full_ids, full_valid = _model().tul_spandec_par.targets(inp, layout)
    assert int(full_valid.sum()) > n_want
    assert torch.equal(full_ids[..., :4][full_valid[..., :4]], ids[valid])


def test_cap_term_is_the_brute_force_mixture_over_the_first_four_tokens():
    """Eval, K = 4 rollouts: `par_ce` against a mixture computed with FULL vocabulary
    logits (slot_id masked) and a Python loop over slots; `par_n_tokens` against the
    walk's count; the head's blocks see 4 positions, never 32."""
    _ids, inp, lab, layout = _pack(B=3, seed=2)
    m = _model(spandec_parallel_span_cap=4).eval()
    head = m.tul_spandec_par
    seen = []
    head.blocks[0].register_forward_hook(lambda mod, a, o: seen.append(tuple(a[0].shape)))
    spy = _Spy(m, "_tul_spandec_par_loss")
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
        want, n_want = _first_c_tokens(inp, layout, 4)
        assert float(out["par_n_tokens"]) == n_want
        assert seen and all(s[1] == 4 for s in seen), seen
        h_slots = spy.calls[0][0][0]
        B0, C = inp.shape[0], m.cfg.d_model
        z = m._readout(h_slots).view(K, B0, h_slots.shape[1], C)
        w = m.embed.lm_weight().float()
        num = 0.0
        for (b, s), toks in sorted(want.items()):
            st = head.states(z[:, b, s].unsqueeze(1), len(toks))[:, 0]      # [K, n, C]
            logits = st.float() @ w.T
            logits[..., m.cfg.tul.slot_id] = float("-inf")
            lp = F.log_softmax(logits, -1)[:, torch.arange(len(toks)), torch.tensor(toks)]
            num = num + (torch.logsumexp(lp.sum(-1), 0) - math.log(K))
        brute = -num / n_want
    torch.testing.assert_close(out["par_ce"], brute, rtol=1e-5, atol=1e-5)
    assert float(out["par_k"]) == K


def test_the_capped_head_is_the_prefix_of_the_uncapped_head():
    """Causal input-free queries: position j reads positions <= j only, so the J = 4 head
    at shared weights computes the first 4 positions of the J = 32 head."""
    full, cap = _model(), _model(spandec_parallel_span_cap=4)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        full.tul_spandec_par.pos.copy_(torch.randn(full.tul_spandec_par.pos.shape,
                                                   generator=g))
        cap.tul_spandec_par.pos.copy_(full.tul_spandec_par.pos[:4])
        zr = torch.randn(K, 5, full.cfg.d_model, generator=g)
        a = full.tul_spandec_par.states(zr)[:, :, :4]
        b = cap.tul_spandec_par.states(zr)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


# ── the probe ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("detach", [True, False], ids=["probe", "control"])
def test_probe_trains_the_head_alone_and_the_control_reaches_the_loop(detach):
    """TRAIN mode (the K = 4 head runs checkpointed). The head's own term, the exact
    tensor the total loss adds at weight 1.0, backpropagated alone. Probe: every head
    tensor gets a nonzero gradient and every other parameter gets None or exact zeros.
    Control (same weights, same seed): the term reaches the core, the codes, the prelude,
    the readout and the embedding. Then, probe only: the TOTAL loss gives the head the
    same gradient (the term is in it and nothing else touches the head)."""
    _ids, inp, lab, layout = _pack()
    m = _model(spandec_parallel_detach=detach).train()
    spy = _Spy(m, "_tul_spandec_par_loss")
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    par = spy.outs[0]
    assert par.requires_grad
    torch.testing.assert_close(out["par_weighted"], par.detach(), rtol=0, atol=0)
    names = [n for n, _ in m.named_parameters()]
    params = [p for _, p in m.named_parameters()]
    g = dict(zip(names, torch.autograd.grad(par, params, retain_graph=True,
                                            allow_unused=True)))
    head = [n for n in names if n.startswith(_HEAD)]
    assert head and all(g[n] is not None and g[n].abs().sum() > 0 for n in head), head
    rest = {n: g[n] for n in names if not n.startswith(_HEAD)}

    def live(prefix: str) -> bool:
        return any(v is not None and v.abs().sum() > 0
                   for n, v in rest.items() if n.startswith(prefix))
    if detach:
        leaked = [n for n, v in rest.items() if v is not None and v.abs().sum() > 0]
        assert leaked == [], leaked
        hp = [p for n, p in m.named_parameters() if n.startswith(_HEAD)]
        g_tot = torch.autograd.grad(out["loss"], hp)
        for n, a in zip(head, g_tot):
            torch.testing.assert_close(a, g[n], rtol=1e-6, atol=0, msg=n)
    else:
        for prefix in ("core.", "tul_code_enum.", "prelude.", "final_norm.", "lm_mixer.",
                       "embed."):
            assert live(prefix), f"the control's head term did not reach {prefix}"


def test_probe_logs_the_same_readings_as_the_trained_head():
    """Eval, same weights: detaching changes no value, only where the gradient goes."""
    _ids, inp, lab, layout = _pack()
    a, b = _model().eval(), _model(spandec_parallel_detach=True).eval()
    with torch.no_grad():
        oa = a(inp, labels=lab, slot_layout=layout)
        ob = b(inp, labels=lab, slot_layout=layout)
    keys = sorted(k for k in oa if k == "par" or k.startswith("par_"))
    assert keys == sorted(k for k in ob if k == "par" or k.startswith("par_"))
    assert {"par_ce", "par_n_tokens", "par_k", "par_resp_entropy", "par_width_gain"} <= set(keys)
    for key in keys:
        assert torch.equal(oa[key], ob[key]), key
    assert torch.equal(oa["loss"], ob["loss"])


# ── configs and refusals ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,key,val,wb", [
    ("tul_slot_spandec_strict_e4j4", "spandec_parallel_span_cap", 4, "lxtul-e4j4"),
    ("tul_slot_spandec_strict_e4probe", "spandec_parallel_detach", True, "lxtul-e4probe"),
])
def test_the_stage2_configs_compose_build_run_and_differ_from_e4_by_one_key(
        name, key, val, wb, monkeypatch):
    from omegaconf import OmegaConf
    cfg, rt = _runtime(name, monkeypatch)
    tc = rt.model_cfg
    assert cfg.wandb.name == wb and getattr(tc, key) == val
    assert rt.manifest[key] == val
    assert tc.code_enum_k == K and tc.spandec_parallel and not tc.spandec
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, d_ff=_D_FF)).eval().float()
    assert m.tul_spandec_par.max_tokens == (4 if key == "spandec_parallel_span_cap" else 32)
    _ids0, inp, lab, layout = _pack()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["par_ce"]) and torch.isfinite(out["loss"])
    c = OmegaConf.to_container(cfg, resolve=True)
    e4, _rt4 = _runtime("tul_slot_spandec_strict_e4", monkeypatch)
    e4 = OmegaConf.to_container(e4, resolve=True)
    assert c["tul"].pop(key) == val and key not in e4["tul"]
    assert c["wandb"].pop("name") == wb and e4["wandb"].pop("name") == "lxtul-e4"
    assert c == e4, f"{name} differs from e4 in more than {key} and the name"


@pytest.mark.parametrize("kw,exc,match", [
    (dict(spandec_parallel_span_cap=-1), ValueError, "span_cap must be >= 0"),
    (dict(spandec_parallel_span_cap=32), ValueError, "change nothing"),
    (dict(spandec_parallel_span_cap=4, parallel=False, spandec=True), ValueError,
     "spandec_parallel=false"),
    (dict(spandec_parallel_detach=True, parallel=False, spandec=True), ValueError,
     "spandec_parallel=false"),
    (dict(spandec_parallel_span_cap=4, spandec=True), NotImplementedError,
     "SAME J tokens"),
])
def test_refusals(kw, exc, match):
    k = kw.pop("k", K)
    with pytest.raises(exc, match=match):
        _tc(k, **kw)


def test_the_wandb_manifest_carries_both_keys_at_their_defaults(monkeypatch):
    _cfg, rt = _runtime("tul_slot_spandec_strict_e4", monkeypatch)
    assert rt.manifest["spandec_parallel_span_cap"] == 0
    assert rt.manifest["spandec_parallel_detach"] is False
