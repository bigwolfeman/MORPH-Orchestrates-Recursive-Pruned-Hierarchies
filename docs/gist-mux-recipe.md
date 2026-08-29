# The Gist-Slot Recipe — the code that made slot content load-bearing

Status: LIVE. Written 2026-08-29, the night GL1b inverted the mask's price.
This is a literate record: the ACTUAL shipped code of every load-bearing piece,
with the measured number each piece produced, so the win cannot be lost to a
refactor or a compaction. Source of truth for the code is the tree at commit
`1a161d9` (+ the CPU-eig fix `4ea5a56`); this file quotes it verbatim and says
where it lives.

## The result this documents

| arm | mask | write grad | write supervision | worth_shuffle | ce_tokens @4250 |
|---|---|---|---|---|---|
| every prefix arm (FM1/FM2/CW…) | no | no (z detached) | none or myopic | ≤ 0.0006 | 4.29–4.35 |
| gl1-ctrl-s1 (`hvxl7vky`) | no | yes | none | 0.001–0.005 | **4.6656** (the ruler) |
| GL1 `gl1-s1` (`qbr69h5r`) | yes | yes | none (SIGReg only) | 0.05–0.07 | 5.0033 (+0.34 price) |
| **GL1b `gl1b-s1` (`juatgwkg`)** | yes | yes | **MUX span targets** | **0.05–0.096** | **4.4047 (−0.26 vs ruler)** |

Three pieces, all required. The mask makes the slots *necessary*. The gradient-
carrying write makes them *trainable*. The MUX target makes them *worth reading*.
Remove any one and the record shows exactly what you get (last section).

The loss is:

```
total = token_CE + β · mux_local        (β = tul.mux_beta = 1.0; SIGReg off in GL1b)
```

## Piece 1 — the mask: slots are the only route between spans

`morph/model/tul_layout.py::tg_allow_mask` (the TG restriction, spec
`docs/tul-tg-spec.md` §1). A token attends its own span and any earlier slot —
nothing else. Content from an earlier span has NO path to a later token except
through a slot state.

```python
def tg_allow_mask(layout: "SlotLayout", soft_prev_span: bool = False) -> Tensor:
    """``[B, 1, L, L]`` bool: TG1's within-span-or-slot allow relation (spec §1).

        allow(i, j) = (j <= i)                             # causal
                      AND ( bag_id[i] == bag_id[j]          # same span (tokens+own slot)
                            OR slot_mask[j] )                # or j is any slot position
    """
    bag_id = layout.bag_id                                    # [B, L] int64
    slot_mask = layout.slot_mask                               # [B, L] bool
    device = bag_id.device
    L = bag_id.shape[1]
    row = torch.arange(L, device=device).unsqueeze(1)
    col = torch.arange(L, device=device).unsqueeze(0)
    causal = (col <= row)                                       # [L, L], j <= i

    bag_i = bag_id.unsqueeze(2)                                 # [B, L, 1]
    bag_j = bag_id.unsqueeze(1)                                 # [B, 1, L]
    allow = (bag_i == bag_j) | slot_mask.unsqueeze(1)            # [B, L, L]
    if soft_prev_span:
        i_not_dump = bag_i < layout.max_slots                    # [B, L, 1]
        allow = allow | ((bag_i == bag_j + 1) & i_not_dump)
    allow = allow & causal.unsqueeze(0)
    return allow.unsqueeze(1)                                   # [B, 1, L, L]
```

(Elided here: the docstring's dump-bin gating note — read it in source before
touching `soft_prev_span`.) The falsifier that proves the mask is really closed
is TG spec §7 T3, run with retention ON: with the slot channel hook-zeroed,
`∂ later-span logits / ∂ earlier-span embeddings` is EXACTLY zero
(`tests/test_tul_gl1.py`, imported from `test_tg_restrict.py`). GLA retention
would be a mask-invisible second route between spans; `tg_reset_mask` segments
the scan at every span boundary so it is not.

Measured alone (TG campaign, 2026-08-27): the mask WITHOUT the pieces below
collapses content specificity to 0.1–0.6% — position-reliance, not content.

## Piece 2 — the gradient-carrying write, with no loop anywhere

`morph/model/transformer.py::_tul_core`, the `n_core == 0` branch. The slot
state is the prelude's own output at the slot position — nothing is detached,
and there is no iterated map for the gradient to unroll. This cell of the design
space was STRUCTURALLY UNREACHABLE before 2026-08-29 (the stack below raised on
an empty TensorList); no prior arm ever ran it.

```python
        # ── n_core == 0: NO LOOP AT ALL (arm GL1, the gist baseline) ─────────
        # The slot state IS the prelude's own output at the slot position, after the
        # same boundary norm the coreless TOKEN path applies — so a coreless TUL model
        # is exactly a coreless baseline that happens to have slot positions.
        #
        # Nothing is detached. Under `tg_restrict` the slot is the only route from an
        # earlier span to a later one, so a later span's CE MUST backpropagate through
        # this state into the boundary tap and the prelude. That gradient-carrying write
        # is the arm's entire mechanism (gisting; TG paper Table 1: detaching the write
        # costs 10x PPL), and there is no iterated map left for it to unroll — which is
        # what makes it safe here and unsafe in every arm that kept the loop.
        if n_core == 0:
            ...
            depths = torch.zeros_like(gidx)
            return xn, e, depths, None
```

Why both halves matter (the campaign's pincer): TG Table 1 — detaching the
memory write costs 10x PPL; and BPTT through an ITERATED write is the takeover
mechanism (`.agents/notes/implemented/architecture/2026-06-19-iterative-map-dynamics.md`).
The viable region is their intersection, and this branch is it.

Measured alone (GL1): mask + this write + SIGReg = worth_shuffle 0.05–0.07 (the
campaign's first load-bearing content) at a 0.34-nat price vs the unmasked twin.

## Piece 3 — the MUX write supervision (arXiv 2607.18264, verified read)

Two functions. First the target — `morph/model/tul.py::mux_span_targets`, the
`target="own"` branch GL1b runs: slot `i` is supervised toward the geometric
position-weighted superposition of the span it terminates (their Eq. 2). The
dense `|V|` simplex vector is NEVER built — the KL reduces to a weighted CE over
the span's own token ids.

```python
    if target == "own":
        # Span k supervises slot k. Valid when slot k is real and the position is a
        # token of that span. Span 0 starts at position 0; span k>0 starts right after
        # slot k-1's prefix block.
        own_ok = torch.gather(layout.slot_valid, 1, kc)
        pos_valid = (~layout.slot_mask) & (k < S) & own_ok
        prev_end = torch.gather(layout.slot_index, 1, (kc - 1).clamp(min=0))
        start = torch.where(kc >= 1, prev_end + layout.prefix_k,
                            torch.zeros_like(prev_end))
        j = (torch.arange(L, device=dev).unsqueeze(0) - start).clamp(min=0)
        w = torch.exp(j.to(torch.float32) * math.log(rho))          # w_j = rho^j (Eq. 2)
        w = torch.where(pos_valid, w, torch.zeros_like(w))
        # Normalise per (row, span); invalid positions scatter into a dump column.
        idx = torch.where(pos_valid, kc, torch.full_like(kc, S))
        denom = torch.zeros(B, S + 1, device=dev, dtype=w.dtype)
        denom.scatter_add_(1, idx, w)
        alpha = w / torch.gather(denom, 1, idx).clamp(min=1e-20)
        return pos_valid, alpha, kc * pos_valid.long(), denom[:, :S] > 0
```

Then the head — `morph/model/transformer.py::_tul_mux_loss`. Zero new
parameters: the slot state goes through the model's OWN `_readout` and the
weight-tied unembedding. The optimized quantity is the weighted CE, which equals
the paper's Eq. 4 KL up to the target's entropy (a constant in the parameters —
the gradient is the paper's exactly).

```python
        tc = self.cfg.tul
        z = self._readout(h_slots)                            # [B, S, C]
        w_head = self.embed.lm_weight()                       # [V, C]  (weight-tied)
        if tc.mux_detach_head:
            w_head = w_head.detach()
        logits = (z @ w_head.t()).float() / tc.mux_tau        # fp32: stable log_softmax
        logits = logits.index_fill(
            -1, torch.tensor([tc.slot_id], device=logits.device), float("-inf"))
        logp = torch.log_softmax(logits, dim=-1)              # [B, S, V]
        pos_valid, alpha, tgt_slot, sup = mux_span_targets(
            input_ids, layout, tc.mux_rho, target=tc.mux_target)
        ...
        safe_ids = torch.where(pos_valid, input_ids, torch.zeros_like(input_ids))
        lp = logp.reshape(B, -1).gather(1, tgt_slot * V + safe_ids)    # [B, L]
        ce = -torch.where(pos_valid, alpha * lp, torch.zeros_like(lp)).sum()
        n_sup = sup.sum().to(ce.dtype).clamp(min=1.0)
        loss = ce / n_sup
```

Why the target works where every prior supervision failed: it is provably
span-distinctive (their Prop. 3 — geometric weights are injective for practical
ρ), so it cannot be satisfied by anything that washes out span identity, and it
cannot be won by the previous-token shortcut the way the one-token emit CE was
(the FM2 postmortem). Their Prop. 9 makes low local loss FORCE the realized
states apart — measured here as slot effective rank 52–62 with SIGReg OFF
(GL1's SIGReg-on run managed 41.5). Their Prop. 16 says low local loss also
protects attention routing TO the latents — measured here as attn_lift rising
0.35 → 0.49.

Honesty instruments, in the same function: `mux_rel` = CE against a null that
knows only the corpus marginal (== 1.0 exactly at that null, tested), plus the
KL and the entropy floor separately, so "the CE fell" can never masquerade as
learning it is not. GL1b: mux_rel 1.07 → 0.51.

Hyperparameters (all from the paper's Table 9 or its text, none tuned by us):
`mux_rho: 0.9`, `mux_tau: 1.0`, `mux_beta: 1.0`, `mux_target: own`,
`mux_detach_head: false` — and the measured warning that detaching the head does
NOT isolate the embedding table anyway (the gradient also arrives through the
slot seed; `mux_embed_grad_share` ≈ 0.40 either way, all run, harmless in the
event).

## What each piece buys — the ablation map the campaign paid for

| configuration | arms that ran it | worth_shuffle | verdict |
|---|---|---|---|
| no mask, no write grad, no/myopic supervision | FM1, FM2, FM1-CW, oracle probe | ≤ 0.0006 | content unread; even a PERFECT plan at the prefix = 0.0000 |
| mask only (write grad through a LOOP, aux off) | TG2 family | shuffle ~0, wrong-value 0.5 | pipe reads VALUES, writes homogeneous |
| mask + loop-free write grad + SIGReg | GL1 | 0.05–0.07 | content load-bearing, price 0.34 |
| **mask + loop-free write grad + MUX target** | **GL1b** | **0.05–0.096** | **price INVERTED: −0.26 vs the unmasked ruler** |

## Run it

```bash
cd /home/wolfe/morph-perf
PYTHONPATH=$PWD python -m morph.training.train --config-name tul_gl1b \
  training.steps=4500 training.batch_size=6 training.seed=1 \
  training.ademamix_alpha_cap=3.5 training.ademamix_t_beta3=3500 \
  model.use_kernels=false tul.eval_ablations=true \
  training.eval_every=250 training.gen_every=0 training.ckpt_every=500
```

`tul_gl1b.yaml` is the whole arm; `tul.tg_restrict=false` on the CLI is the
unmasked twin. Note `tg_restrict` REPLACES the HCA branch (+2.04M params in the
twin — config-matched, not parameter-matched) and forces eager kernels, at NO
measured speed cost (3.44 sps vs the FM arms' 3.30 — the removed branch pays for
the mask).

## The way out of the test chambers

The mechanism question — CAN a slot carry content a decoder will read — is
CLOSED. Answered, measured, done. No further reliance-forcing ablations are
authorized by this document; anyone proposing one must first explain what this
table does not already answer. What remains is promotion, in order:

1. The mux'd unmasked twin (one CLI override) — makes the inversion claim
   airtight against the one ruler asymmetry the prereg named.
2. GL2/GL3 from the gist-loop note: the detached refiner loop and the
   plan-as-router, now building on slots that are provably read.
3. Integration into the main recipe (`base.yaml`) once a second seed and a
   longer budget hold.

Provenance: preregs+filings `lab/experiments/*/2026-08-29-tul-gl1*.md`; decision
note `.agents/notes/proposed/architecture/2026-08-29-gist-loop.md`; wandb runs
named in the table; commits `769b50c` (GL1), `1a161d9` (GL1b), `6a9a524` (GL1c),
`4ea5a56` (probe fix). The cake is real.
