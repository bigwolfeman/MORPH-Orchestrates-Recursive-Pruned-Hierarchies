"""Build plain Parcae and Parcae-LXTUL from one testbed config (shared by bench and train)."""
from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

import parcae_lm
import recpre.optim

from lxtul.data import layout_for_cells, morph_cfg_of, packed_length, tul_runtime
from lxtul.model import LXTULConfig, LXTULParcae


def parcae_config(cfg: DictConfig, vocab_size: int, plain: bool, block_size: int | None = None):
    """`block_size` overrides `shape.block_size` (the LXTUL arm passes its packed row length)."""
    s = cfg.shape
    kw = dict(n_embd=s.n_embd, num_attention_heads=s.num_attention_heads,
              num_key_value_heads=s.num_key_value_heads, intermediate_size=s.intermediate_size,
              recurrent_embedding_dimension=s.n_embd,
              recurrent_intermediation_embedding_dimension=s.intermediate_size,
              n_layers_in_prelude=s.n_layers_in_prelude,
              n_layers_in_recurrent_block=s.n_layers_in_recurrent_block,
              n_layers_in_coda=s.n_layers_in_coda, vocab_size=vocab_size,
              block_size=int(block_size or s.block_size))
    if plain:
        kw.update(mean_recurrence=cfg.plain.mean_recurrence,
                  mean_backprop_depth=cfg.plain.mean_backprop_depth,
                  row_compact=bool(cfg.plain.get("row_compact", False)))
    return parcae_lm.create_config(s.base, **kw)


def build(cfg: DictConfig, arm: str, overrides: dict | None = None, compile: bool | None = None):
    """-> (model on cuda, morph cfg, tul runtime or None). arm in {plain, lxtul}.

    `overrides` updates LXTULConfig after `cfg.lxtul`; `compile` defaults to cfg.bench.compile."""
    mcfg = morph_cfg_of(cfg)
    vocab = int(mcfg.model.vocab_size)
    if compile is None:
        compile = bool(cfg.bench.compile)
    if arm == "gpt":
        # Parcae's own non-looped baseline: the same blocks, init and value embeddings,
        # n_layer = prelude + coda, i.e. the per-token depth an LXTUL token gets
        # (`arm.layers` overrides it: flat14 = prelude 4 + core 6 + coda 4, the plain loop unrolled once)
        s = cfg.shape
        gc = parcae_lm.create_config(
            "gpt-small-140m", n_embd=s.n_embd, num_attention_heads=s.num_attention_heads,
            num_key_value_heads=s.num_key_value_heads, intermediate_size=s.intermediate_size,
            n_layer=int(cfg.arm.get("layers") or (int(s.n_layers_in_prelude) + int(s.n_layers_in_coda))),
            vocab_size=vocab,
            block_size=s.block_size,
            # GPT's fused head needs vocab % 4096 == 0; MORPH's padded vocab is 49216
            use_fused_head=False)
        model = gc.construct_model(gradient_checkpointing=False)
        rt = None
        if compile:
            c = lambda b: torch.compile(b, backend="inductor", mode="default", dynamic=False)
            model.transformer.h = torch.nn.ModuleList(c(b) for b in model.transformer.h)
        if int(cfg.arm.get("pointer_heads", 0)) > 0:      # plain+pointer head on the flat model (plain_pointer.py)
            from lxtul.plain_pointer import PlainPointer
            model = PlainPointer(model, int(cfg.arm.pointer_heads))
    elif arm == "plain":
        if bool(cfg.plain.get("sync_free", False)) and bool(cfg.plain.get("row_compact", False)):
            raise ValueError("plain.sync_free and plain.row_compact are exclusive: row_compact runs its own "
                             "loop (_iterate_rows_compacted), so sync_free would silently do nothing")
        model = parcae_config(cfg, vocab, plain=True).construct_model(
            gradient_checkpointing=bool(cfg.plain.get("gradient_checkpointing", False)))   # off: today's path
        model.sync_free_loop = bool(cfg.plain.get("sync_free", False))   # exact, see Parcae.iterate_forward
        rt = None
        if compile:
            t = model.transformer
            c = lambda b: torch.compile(b, backend="inductor", mode="default", dynamic=False)
            t.prelude = torch.nn.ModuleList(c(b) for b in t.prelude)
            t.core_block = torch.nn.ModuleList(c(b) for b in t.core_block)
            t.coda = torch.nn.ModuleList(c(b) for b in t.coda)
        if int(cfg.arm.get("pointer_heads", 0)) > 0:
            from lxtul.plain_pointer import PlainPointer
            model = PlainPointer(model, int(cfg.arm.pointer_heads))
    elif arm == "lxtul":
        rt = tul_runtime(mcfg)
        tc = mcfg.tul
        base = dict(slot_id=int(rt.data_cfg.slot_id), max_slots=int(tc.max_slots),
                    cells=int(tc.fan_k), span_cap=int(tc.span_cap),
                    mean_depth=int(mcfg.model.mean_depth), max_depth=int(mcfg.model.max_depth),
                    token_state_dropout=float(tc.token_state_dropout),
                    spandec_layers=int(tc.spandec_layers))
        base.update(OmegaConf.to_container(cfg.lxtul, resolve=True))
        base.update(overrides or {})
        hop = cfg.get("task") is not None and str(cfg.task.get("kind", "none")) == "hops"
        if hop:                              # the hop task's slot capacity (the OWT stream keeps MORPH's)
            base["max_slots"] = int(cfg.task.max_slots)
        # the packed row follows `lxtul.cells` (a static layout: prefix_k = cells), and so does the model's
        # block_size; with cells = MORPH's fan_k (4) both are the reference 1280
        rt = layout_for_cells(rt, int(base["cells"]))
        block = int(mcfg.data.seq_len) + int(base["cells"]) * int(base["max_slots"]) if hop else packed_length(mcfg, rt)
        model = LXTULParcae(parcae_config(cfg, vocab, plain=False, block_size=block), LXTULConfig(**base))
        model.init_slot_seed()
        if compile:
            model.compile_blocks()
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return model.cuda().train(), mcfg, rt


def make_optimizer(model, cfg: DictConfig):
    oc = OmegaConf.to_container(cfg.optimizer, resolve=True)
    groups = recpre.optim.get_muon_param_groups_from_config(
        model.named_parameters(), oc, no_weight_decay_for_bias_and_norm_params=True)
    return recpre.optim.MuonAdamW(groups)
