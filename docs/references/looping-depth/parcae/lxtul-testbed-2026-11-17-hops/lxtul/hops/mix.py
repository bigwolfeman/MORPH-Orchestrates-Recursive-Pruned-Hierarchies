"""The hop task: sources, row shares, the training mix and the fixed evaluation rows.

`task.kind: none` (the default) builds nothing and the loader is untouched. With `kind: hops`, `HopTask`
draws, for training batch `step` of B rows, which rows are hop rows (`assign`), and builds hop row `r` of
source `s` from `(task.seed, s, r)` ALONE (`train_row`), r = step * B + row. The documents therefore do not
depend on the arm kind, the packer or any earlier draw: a flat arm and a slot arm see the same hop documents
at the same step. Which documents go into a row depends only on (seq_len, `task.max_slots`, boundary rule).
The MORPH config stays stock: OWT rows are packed as without the task, and the slot arms get them WIDENED to the
hop layout (`base.widen_batch`: same tokens, extra tail pads).

`mix_hop_rows` swaps those rows into the batches of the ordinary stream (the OWT rows still advance the
stream, so every non-hop row is the row the arm without the task would see).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from lxtul.hops.base import HopEnv, HopRow, HopSource, build_row, pack_row, plain_row, token_texts, widen_batch
from lxtul.hops.chain import ChainSource
from lxtul.hops.graph import GraphSource

SOURCES = {"chain": ChainSource, "graph": GraphSource}
TASK_KEYS = {"kind", "seed", "eval_seed", "max_slots", "sources", "eval"}
EVAL_KEYS = {"rows_per_bucket", "val_rows_per_bucket", "depths", "chain", "graph"}
# rng key tags, so the three streams of draws never share a key
_ASSIGN, _TRAIN, _EVAL = 0, 1, 2


@dataclass
class HopTask:
    sources: list[HopSource]
    shares: list[float]
    seed: int
    eval_seed: int
    seq_len: int
    max_slots: int
    rule: object
    slot_id: int
    pad_id: int
    rows_per_bucket: int
    val_rows_per_bucket: int
    depths: list[int]
    manifest: dict

    def __post_init__(self):
        if not self.sources or len(self.sources) != len(self.shares):
            raise ValueError("a hop task needs at least one source with a share")
        if min(self.shares) <= 0 or sum(self.shares) > 1.0 + 1e-9:
            raise ValueError(f"row shares must be non-negative and sum to at most 1, got {self.shares}")
        if self.seed == self.eval_seed:
            raise ValueError("task.seed and task.eval_seed must differ: the eval rows may not be training rows")
        if not 0 < self.val_rows_per_bucket <= self.rows_per_bucket:
            raise ValueError("task.eval.val_rows_per_bucket must lie in [1, rows_per_bucket]")

    def spec(self, prefix_k: int):
        """The layout hop rows are packed with, and OWT rows are widened to (slot arms): seq_len, `task.max_slots`
        slots, `prefix_k` cells per slot. The stock OWT layout is untouched for every other purpose."""
        from morph.model.tul_layout import TulLayoutSpec
        return TulLayoutSpec(seq_len=self.seq_len, prefix_k=int(prefix_k), max_slots=self.max_slots, slot_id=self.slot_id)

    # ------------------------------------------------------------------ training
    def assign(self, step: int, batch_size: int) -> list[int]:
        """Per row of batch `step`: the source index of a hop row, or -1 for an ordinary row. A share that
        makes B * share a whole number for every source is exact (a seeded shuffle of the positions); any
        other share is a seeded draw per row."""
        counts = [s * batch_size for s in self.shares]
        if all(abs(c - round(c)) < 1e-9 for c in counts):
            slots = [i for i, c in enumerate(counts) for _ in range(round(c))]
            pos = np.random.default_rng([self.seed, _ASSIGN, step]).permutation(batch_size)[: len(slots)]
            out = [-1] * batch_size
            for p, i in zip(pos, slots):
                out[int(p)] = i
            return out
        edges = np.cumsum(self.shares)
        u = np.random.default_rng([self.seed, _ASSIGN, step]).random(batch_size)
        return [int(np.searchsorted(edges, x, side="right")) if x < edges[-1] else -1 for x in u]

    def train_row(self, src: int, r: int) -> HopRow:
        rng = np.random.default_rng([self.seed, _TRAIN, src, r])
        return build_row(self.sources[src].train_document, rng, self.seq_len, self.max_slots, self.rule)

    # ---------------------------------------------------------------- evaluation
    def eval_rows(self) -> list[tuple[str, HopRow]]:
        """The fixed evaluation rows, bucket-major: `rows_per_bucket` rows per bucket of every source (a source may
        set its own `eval_rows`), row j of bucket b of source s from `(eval_seed, s, b, j)`. Every arm scores these
        same documents. These buckets group ROWS; accuracy is bucketed per answer (family, hops) by the eval."""
        out = []
        for s, src in enumerate(self.sources):
            for b, (name, make_doc) in enumerate(src.eval_buckets()):
                for j in range(getattr(src, "eval_rows", self.rows_per_bucket)):
                    rng = np.random.default_rng([self.eval_seed, _EVAL, s, b, j])
                    out.append((name, build_row(make_doc, rng, self.seq_len, self.max_slots, self.rule)))
        return out

    def val_rows(self, rows: list) -> list:
        """The training-time subset of `eval_rows()`: the first `val_rows_per_bucket` rows of each bucket (a source
        may set `eval_val_rows`)."""
        seen, out = {}, []
        for name, r in rows:
            j = seen[name] = seen.get(name, -1) + 1
            src = next(s for s in self.sources if any(name == b for b, _ in s.eval_buckets()))
            if j < getattr(src, "eval_val_rows", self.val_rows_per_bucket):
                out.append(r)
        return out

    # ------------------------------------------------------------------ tensors
    def tul_batch(self, rows: list[HopRow], spec):
        """-> (x, y, SlotLayout, hops [B, L], family [B, L]) of packed rows (CPU tensors)."""
        from morph.model.tul_layout import SlotLayout
        packed = [pack_row(r, self.rule, spec) for r in rows]
        stack = lambda k: torch.from_numpy(np.stack([p[k] for p in packed]))   # noqa: E731
        lay = SlotLayout(slot_mask=stack("slot_mask"), bag_id=stack("bag_id"), slot_index=stack("slot_index"),
                         slot_valid=stack("slot_valid"), prefix_k=spec.prefix_k)
        return stack("input_ids"), stack("labels"), lay, stack("hops"), stack("family")

    def plain_batch(self, rows: list[HopRow]):
        """-> (x, y, hops [B, seq_len], family [B, seq_len]) of flat rows (CPU tensors)."""
        flat = [plain_row(r, self.seq_len, self.pad_id) for r in rows]
        stack = lambda k: torch.from_numpy(np.stack([p[k] for p in flat]))      # noqa: E731
        return stack("x"), stack("y"), stack("hops"), stack("family")


def build_task(cfg, mcfg, rt) -> HopTask | None:
    """The `task` block of the config -> a HopTask, or None for `kind: none`. `mcfg` is the MORPH config (the
    tokenizer name, seq_len), `rt` the TUL runtime (the boundary rule, slot id, max_slots)."""
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    c = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    kind = str(c.get("kind", "none"))
    if kind == "none":
        return None
    if kind != "hops":
        raise ValueError(f"task.kind must be none or hops, got {kind!r}")
    unknown = set(c) - TASK_KEYS
    if unknown or set(c["eval"]) - EVAL_KEYS:
        raise ValueError(f"unknown task keys: {sorted(unknown | (set(c['eval']) - EVAL_KEYS))}")
    tok = AutoTokenizer.from_pretrained(str(mcfg.data.tokenizer))
    texts = token_texts(tok)
    env = HopEnv(texts, lambda s: tok(s, add_special_tokens=False)["input_ids"], rt.data_cfg.rule,
                 int(tok.eos_token_id) if tok.eos_token_id is not None and tok.eos_token_id < len(texts) else 0,
                 int(rt.data_cfg.slot_id))
    srcs, shares = [], []
    for name, scfg in c["sources"].items():
        if name not in SOURCES:
            raise ValueError(f"unknown hop source {name!r}; known: {sorted(SOURCES)}")
        if float(scfg["row_share"]) > 0:
            srcs.append(SOURCES[name].from_cfg(scfg, c["eval"][name], env))
            shares.append(float(scfg["row_share"]))
    ev = c["eval"]
    return HopTask(
        sources=srcs, shares=shares, seed=int(c["seed"]), eval_seed=int(c["eval_seed"]),
        seq_len=int(mcfg.data.seq_len), max_slots=int(c["max_slots"]), rule=rt.data_cfg.rule,
        slot_id=int(rt.data_cfg.slot_id), pad_id=env.eos, rows_per_bucket=int(ev["rows_per_bucket"]),
        val_rows_per_bucket=int(ev["val_rows_per_bucket"]), depths=[int(d) for d in ev["depths"]],
        manifest={s.name: {"row_share": sh, "n_names": s.n_names} for s, sh in zip(srcs, shares)})


def mix_hop_rows(it, task: HopTask, spec=None) -> Iterator:
    """Swap the hop rows of `task` into the batches of `it`. `spec` is `task.spec(cells)` for a TUL stream
    ((x, y, SlotLayout) batches: every batch is first widened from the stock layout to it, `widen_batch`), None
    for a flat stream ((x, y) batches). Batch `i` of the stream is training step `i`. A flat batch with no hop
    row passes through untouched."""
    for step, batch in enumerate(it):
        if spec is not None:
            batch = widen_batch(batch, spec)
        x, y = batch[0], batch[1]
        B = x.shape[0]
        rows = task.assign(step, B)
        if all(s < 0 for s in rows):
            yield batch
            continue
        x, y = x.clone(), y.clone()
        if spec is None:
            for j, s in enumerate(rows):
                if s >= 0:
                    p = plain_row(task.train_row(s, step * B + j), task.seq_len, task.pad_id)
                    x[j], y[j] = torch.from_numpy(p["x"]), torch.from_numpy(p["y"])
            yield x, y
            continue
        lay = batch[2]
        if int(lay.prefix_k) != spec.prefix_k or x.shape[1] != spec.l_total:
            raise ValueError(f"batch rows ({x.shape[1]} positions, prefix_k {lay.prefix_k}) do not match the "
                             f"task's layout spec ({spec.l_total}, {spec.prefix_k})")
        mask, bag, sidx, valid = (t.clone() for t in (lay.slot_mask, lay.bag_id, lay.slot_index, lay.slot_valid))
        for j, s in enumerate(rows):
            if s >= 0:
                p = pack_row(task.train_row(s, step * B + j), task.rule, spec)
                x[j], y[j] = torch.from_numpy(p["input_ids"]), torch.from_numpy(p["labels"])
                mask[j], bag[j] = torch.from_numpy(p["slot_mask"]), torch.from_numpy(p["bag_id"])
                sidx[j], valid[j] = torch.from_numpy(p["slot_index"]), torch.from_numpy(p["slot_valid"])
        yield x, y, type(lay)(slot_mask=mask, bag_id=bag, slot_index=sidx, slot_valid=valid, prefix_k=lay.prefix_k,
                              stats=lay.stats, span_len=lay.span_len, len_supervised=lay.len_supervised,
                              ditto_prev=lay.ditto_prev, cell_roles=lay.cell_roles)
