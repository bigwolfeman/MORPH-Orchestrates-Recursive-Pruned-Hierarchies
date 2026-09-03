"""Is it the LOOP that is worth nothing, or the whole SLOT PATH?

`region_shapley.py` ablates the core by making each loop iteration the identity. That
removes the LOOP. It does NOT remove the slot: the slot position still carries
`E_slot + mean(embed(span))` through the prelude, and the coda still reads a span summary
from it. So "core Shapley on ce_main = 0.0007" answers "is looping worth anything BEYOND
the span bag-mean", which is a narrower question than it looked.

This separates the two. Four conditions on one checkpoint, one fixed eval set:

| condition        | loop | slot values reaching the coda |
|------------------|------|-------------------------------|
| full             | on   | looped state                  |
| no-loop          | OFF  | un-looped slot input (bag-mean) |
| no-plan          | on   | ZERO                          |
| neither          | OFF  | ZERO                          |

`no-plan` is the honest test of "is the plan path used at all": it zeroes what
`prefix_project` writes into the shared sequence, so the coda's slot positions carry
nothing and every token predicting from them has only its own token path.

    python lab/divergence/slot_path_worth.py --ckpt checkpoints/morph/onset-capture/ROLL_step_1750.pt
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402
from morph.training.train import load_checkpoint            # noqa: E402


@contextlib.contextmanager
def loop_off(root):
    orig = root._apply_core_step
    root._apply_core_step = lambda h_in, *a, **kw: (h_in, None)
    try:
        yield
    finally:
        root._apply_core_step = orig


@contextlib.contextmanager
def plan_off(root):
    """Zero the values `prefix_project` writes into the shared sequence.

    Positions and validity are untouched, so the sequence LAYOUT is identical and only the
    content of the plan is removed. Anything the coda gains from those positions has to
    come from the plan, so this is the plan's whole worth.
    """
    tul = root.tul
    orig = tul.prefix_project

    def zeroed(h_slots, layout, l_total):
        values, pos = orig(h_slots, layout, l_total)
        return torch.zeros_like(values), pos

    tul.prefix_project = zeroed
    try:
        yield
    finally:
        tul.prefix_project = orig


@contextlib.contextmanager
def token_tax(root, p: float):
    """Force the coda's token-state dropout ON at EVAL, at rate ``p`` (spec §3.4).

    ADDED 2026-08-28 to answer the question the 2026-08-25 note at the bottom of this file
    posed and never implemented: **is the plan EMPTY, or is the coda BYPASSING a usable
    one?**

    `apply_token_dropout` returns unchanged input when `training` is False, so at eval the
    coda always has every token state and can predict a span straight from its own tokens.
    `token_state_dropout` (0.15 in training) is the ONLY pressure that makes it consult the
    plan instead. Measuring the plan with the token path fully intact therefore measures
    what the coda BOTHERS to use, not what it COULD use.

    At p = 1.0 every token state is replaced by ``E_mask`` — Bowman's inputless decoder,
    the extreme end of the §3.4 arm sweep — so the coda has nothing but the plans and the
    positions. Combined with `plan_shuffled`, that is the decisive pair:

        shuffle costs ~0 at p=1.0  -> z holds no span-specific content. Nothing to read.
        shuffle costs a lot at p=1.0 -> z HOLDS content the coda ignores whenever the
                                        tokens are available. The READER is the bottleneck,
                                        not the target, and an objective that writes MORE
                                        into z (flow matching) is well motivated.

    Implemented by flipping the config value and forcing the training branch of the SHIPPED
    function, not by reimplementing the drop: the mask also has to zero the coda's x0 and
    bigram injections at dropped positions (see `apply_token_dropout`'s docstring), and a
    copy of that logic here would drift from it silently.

    The caller must seed the global RNG before each condition — the drop is sampled inside
    the shipped function — so that every condition drops the SAME positions. Without that
    the conditions differ by which tokens were masked as well as by the plan, and the
    comparison is worthless.
    """
    tul = root.tul
    orig_fn = tul.apply_token_dropout
    orig_p = tul.tul.token_state_dropout

    def taxed(x, layout, training):
        return orig_fn(x, layout, True)          # force the train-time path at eval

    tul.tul.token_state_dropout = float(p)
    tul.apply_token_dropout = taxed
    try:
        yield
    finally:
        tul.apply_token_dropout = orig_fn
        tul.tul.token_state_dropout = orig_p


@contextlib.contextmanager
def plan_shuffled(root, seed: int = 0):
    """Permute the plan ACROSS SLOTS, keeping every value a real plan.

    ADDED 2026-08-28, after the blind-decoder probe
    (lab/divergence/plan_content_probe.py) failed its own power check twice: at 41k fit
    examples it beat a unigram model by 0.04 nats, against a pre-registered 0.30 line, so
    it could not have detected a 0.20-nat signal. That probe had to LEARN LANGUAGE from
    scratch to read z. This condition does not — it uses the model's own coda, which was
    trained to read z, as the decoder.

    `plan_off` answers "is the plan path used at all" by zeroing it. It cannot separate
    two very different worlds:

        z carries SPAN-SPECIFIC content  -> shuffling it should cost about as much as
                                            zeroing it
        z is a useful CONSTANT           -> shuffling costs ~0 while zeroing costs a lot,
                                            because every slot's plan is interchangeable

    The second world is what "the plan is EMPTY" means in the only sense that matters: the
    coda gains from the slot POSITIONS, not from what any particular span put there.

    The permutation is over the SLOT axis within each row, so every value the coda reads is
    still a genuine plan produced by this model on this batch — only the correspondence
    between a plan and its own span is destroyed. That is what makes this a control and not
    a corruption: the distribution the coda sees is unchanged, unlike `seed_bagmean`, whose
    out-of-distribution shock swamped the signal it was meant to isolate (prediction B2,
    falsified 3.6-7.3x).

    Rows are permuted independently and derangement is not enforced: with S=64 slots the
    expected fixed-point count is 1, so about 1.6% of slots keep their own plan. That
    biases the measured cost DOWNWARD by ~1.6%, far below the effect sizes in question,
    and enforcing a derangement would cost more in complexity than it buys.
    """
    tul = root.tul
    orig = tul.prefix_project

    def shuffled(h_slots, layout, l_total):
        values, pos = orig(h_slots, layout, l_total)
        # values is [B, S*K, *tail]: slot s owns rows [s*K, (s+1)*K). `tail` is (C,) for a
        # plain carrier and (n, C) for the HC carrier — `prefix_project` documents
        # `mid = h_slots.shape[2:-1]  # () plain, (n,) HC`, and MORPH runs HC n=4, so the
        # shipped rank is 4, not 3. Written rank-agnostically after a 3-D assumption here
        # crashed the first real cap64 worth pass.
        # Permute WHOLE slots so a plan's prefix_k values stay together and in their own
        # order — scrambling within a slot would test something else entirely.
        B, SK = values.shape[0], values.shape[1]
        tail = values.shape[2:]
        K = tul.tul.prefix_k
        S = SK // K
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.stack([torch.randperm(S, generator=g) for _ in range(B)]).to(values.device)
        idx = (perm[:, :, None] * K
               + torch.arange(K, device=values.device)[None, None, :]).reshape(B, SK)
        idx = idx.reshape(B, SK, *([1] * len(tail))).expand(B, SK, *tail)
        return values.gather(1, idx), pos

    tul.prefix_project = shuffled
    try:
        yield
    finally:
        tul.prefix_project = orig


@contextlib.contextmanager
def seed_bagmean(root):
    """Force ``slot_input`` back to the bag-mean seed, whatever the arm was TRAINED with.

    ADDED 2026-08-28, and the reason is a confound that made the round-2 panel
    unreadable. ``loop_off`` leaves the slot carrying its own INPUT, so what "no-loop"
    falls back to is arm-dependent:

        slot_seed=bag_mean  -> falls back to E_slot + mean(embed(span))   INFORMATIVE
        slot_seed=e_slot    -> falls back to E_slot alone                 A CONSTANT
        slot_seed=boundary  -> falls back to E_slot + W_sent.embed(t_last)

    So `full - no-loop` on a TG4a-style arm measures "loop vs NOTHING" while the same
    column on a bag_mean arm measures "loop vs a span summary". Reading the two side by
    side credits the loop for the seed's deletion. Measured on tg4a-s1/step_3000: the
    naive column says the loop is worth 0.0921 nats, ~25x the control band, entirely
    because its fallback was stripped of span content by design.

    Flipping ``slot_seed`` is enough because ``TULSlots.slot_input`` dispatches on it at
    CALL time, not at construction; only the ``add_e_slot=True`` (token-embedding)
    signal changes, and the bigram / value-embed signals were already the bag-mean in
    every mode.

    CAVEAT, and it is not removable without retraining: for an arm not TRAINED on the
    bag-mean this fallback is OUT OF DISTRIBUTION, so `full - no-loop-bagmean` is an
    UPPER bound on the loop's cross-arm worth. The truth for such an arm is bracketed
    BELOW that number and BELOW the naive `full - no-loop`. Report both; claim neither
    as the loop's worth.

    MEASURED 2026-08-28, AND IT FALSIFIED THE PREDICTION THIS CONDITION WAS ADDED ON.
    Pre-registered prediction B2 said the forced bag-mean fallback would be MORE
    informative than a constant, so `full - no-loop-bagmean` would come out SMALLER than
    the naive `full - no-loop`. On tg4a-s2 it came out 6-7x LARGER:

        step 3000:  own seed 0.0764   bag-mean 0.5589   (predicted <, measured 7.3x >)
        step 3500:  own seed 0.1333   bag-mean 0.4794   (predicted <, measured 3.6x >)

    Forcing the bag-mean is worse than zeroing the plan outright (no-plan 0.0953/0.0988).
    The distribution shift DOMINATES the information the bag-mean carries, by a wide
    margin. So this condition does NOT deliver a cross-arm loop worth, and no eval-time
    substitution can: swapping a seed the weights never saw measures the SHOCK, not the
    loop. Read it as an OOD-sensitivity number — how far the arm's downstream weights
    have specialised to their own seed — and nothing else.

    WHAT THIS MEANS FOR THE CAMPAIGN'S LOOP METRIC: loop worth is comparable only WITHIN
    a fixed `slot_seed`. Comparing it across seed modes needs matched TRAINING, not a
    smarter ablation. The confound the condition was built to expose is real; the repair
    is not available at eval time.

    A SECOND reason the naive column is uninterpretable on an e_slot arm, also measured
    here: at step 3500 `no-loop` (0.1333) costs MORE than `no-plan` (0.0988) — removing
    less hurts more. E_slot is a constant the coda has learned to READ, so an
    uninformative-but-valid plan actively misleads it, while zeroing is clean. The
    fallback is not merely uninformative; it is harmful.
    """
    tul_cfg = root.tul.tul
    orig = tul_cfg.slot_seed
    tul_cfg.slot_seed = "bag_mean"
    try:
        yield
    finally:
        tul_cfg.slot_seed = orig


def eval_groups(model, batches) -> dict:
    keys = ("loss", "ce_main", "ce_plast", "ce_emit")
    tot = {k: 0.0 for k in keys}
    n = 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y, layout in batches:
            out = model(x, labels=y, slot_layout=layout)
            for k in keys:
                if out.get(k) is not None:
                    tot[k] += float(out[k])
            n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = build_cfg(a.config, ["training.batch_size=6", "model.use_kernels=false"])
    model, tul_rt = build_model(cfg, device="cuda")
    root = getattr(model, "_orig_mod", model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(a.ckpt, model, scaler, torch.device("cuda"))

    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg if tul_rt else None))
    batches = []
    for _ in range(a.batches):
        bx, by, bl = next(loader)
        batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
    print(f"  fixed eval set: {len(batches)} batches\n")

    @contextlib.contextmanager
    def both(root):
        with loop_off(root), plan_off(root):
            yield

    @contextlib.contextmanager
    def loop_off_bagmean(root):
        with loop_off(root), seed_bagmean(root):
            yield

    # From the BUILT model, not the YAML: this is the value slot_input actually reads.
    seed = str(root.tul.tul.slot_seed)
    # (row label, SHORT label for the worth table, context manager). The short label is
    # carried, not derived by splitting the long one: the first cut of this patch derived
    # it and printed "removing no-loop" for BOTH no-loop rows, which is precisely the
    # ambiguity the patch exists to remove.
    conds = [("full", "full", contextlib.nullcontext()),
             ("no-loop (slot keeps its seed)", "removing no-loop [own seed]", loop_off(root)),
             ("no-plan (slot values zeroed)", "removing no-plan", plan_off(root)),
             ("plan SHUFFLED across slots", "SHUFFLING the plan", plan_shuffled(root)),
             ("neither", "removing neither", both(root))]
    # On a bag_mean arm this condition IS `no-loop`, so running it would only spend eval
    # time to reprint the same row. Emit it exactly when the arm's seed differs.
    if seed != "bag_mean":
        conds.insert(2, (f"no-loop, bag-mean seed (was {seed})",
                         "removing no-loop [bag-mean]", loop_off_bagmean(root)))

    res = {}
    W = max(32, max(len(n) for n, _, _ in conds) + 1)
    print(f"{'condition':<{W}} {'loss':>8} {'ce_main':>8} {'ce_plast':>9} {'ce_emit':>8}")
    for name, _short, ctx in conds:
        with ctx:
            g = eval_groups(model, batches)
        res[name] = g
        print(f"{name:<{W}} {g['loss']:>8.4f} {g['ce_main']:>8.4f} "
              f"{g['ce_plast']:>9.4f} {g['ce_emit']:>8.4f}")

    f = res["full"]
    print()
    print(f"{'what each costs (nats)':<{W}} {'loss':>8} {'ce_main':>8} {'ce_plast':>9} {'ce_emit':>8}")
    for name, short, _ctx in conds[1:]:
        d = res[name]
        print(f"{short:<{W}} "
              + " ".join(f"{d[k] - f[k]:>8.4f}" if k != 'ce_plast' else f"{d[k] - f[k]:>9.4f}"
                         for k in ("loss", "ce_main", "ce_plast", "ce_emit")))
    print()
    print("If 'no-plan' costs about the same as 'no-loop', the LOOP is the worthless part")
    print("and the span bag-mean carries the plan. If 'no-plan' also costs ~0, the whole")
    print("slot path is inert for token prediction and TUL is not doing what it claims.")
    if seed != "bag_mean":
        print()
        print(f"slot_seed={seed}: NEITHER no-loop row is a cross-arm loop worth.")
        print("  [own seed] falls back to a seed this arm stripped of span content, so it")
        print("  credits the loop for the seed's deletion — and on an e_slot arm that seed is")
        print("  a constant the coda still READS, so the fallback is actively harmful, not")
        print("  merely uninformative (measured: no-loop can cost MORE than no-plan).")
        print("  [bag-mean] forces a seed these weights never trained on. MEASURED on")
        print("  tg4a-s2: 6-7x LARGER than [own seed], and worse than zeroing the plan. The")
        print("  distribution shift dominates; this is an OOD-SHOCK number, not a loop worth.")
        print("  Loop worth compares only WITHIN one slot_seed. Across seed modes it needs")
        print("  matched TRAINING — see seed_bagmean's docstring.")
    print()
    zero = res["no-plan (slot values zeroed)"]["ce_main"] - f["ce_main"]
    shuf = res["plan SHUFFLED across slots"]["ce_main"] - f["ce_main"]
    frac = shuf / zero if abs(zero) > 1e-9 else float("nan")
    print()
    print(f"IS THE PLAN SPAN-SPECIFIC?  shuffled costs {shuf:+.4f} of the {zero:+.4f} that")
    print(f"  zeroing costs = {100*frac:.1f}%. Near 100%: the coda uses WHICH span wrote the")
    print("  plan. Near 0%: the plan is an interchangeable constant and the coda gains from")
    print("  the slot POSITIONS, not from any span's content — 'empty' in the sense that")
    print("  matters. This uses the model's OWN coda as the reader, so unlike a blind")
    print("  decoder it needs to learn nothing.")
    print()
    print("CROSS-ARM CAVEAT on 'no-plan' (added 2026-08-28): under tg_restrict a token's ONLY")
    print("route to any earlier span is the slot path, so zeroing the plan removes ALL")
    print("cross-span information. An UNRESTRICTED arm keeps full causal attention and can")
    print("recover the same information directly, so its plan worth is low BY CONSTRUCTION.")
    print("Restricted-vs-unrestricted plan worth is therefore not evidence that the plan")
    print("CONTAINS more; only a direct read of z (lab/divergence/plan_content_probe.py) is.")

    # KEY RENAME 2026-08-28: the "no-loop" row used to be keyed
    # "no-loop (slot keeps bag-mean)". That label is FALSE on any arm whose slot_seed is
    # not bag_mean, and believing it is what nearly turned tg4a-s1's 0.0921 into a
    # reported 25x-control loop win. JSONs written before this date carry the old key.
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()


# ── appended 2026-08-25: is the plan EMPTY, or CAPABLE but never asked? ────────────
#
# The plan is worth 0.0191 nats on ce_main because the coda can read span i's raw TOKENS
# and predict span i+1 from them directly. `token_state_dropout` (0.15) is the only tax
# that forces the coda to use the plan instead. This sweeps that tax at EVAL time.
#
# If the plan's worth on ce_main rises steeply as the token path is masked out, the plan
# HOLDS the information and the coda simply prefers tokens — a design leak, fixable by
# raising the tax. If it stays near zero even with every token state masked, the plan is
# empty and raising the tax fixes nothing.
