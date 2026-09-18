# The Illusion of Superposition: reading and local cache

Read 2026-09-18.

## Citation (verified)

Michael Rizvi-Martel, Guillaume Rabusseau, Marius Mosbach. *The Illusion of
Superposition? A Principled Analysis of Latent Thinking in Language Models*.
[arXiv 2604.06374v2](https://arxiv.org/abs/2604.06374), submitted 2026-04-07,
v2 dated 2026-08-03. Affiliations: Mila, Universite de Montreal, McGill.

**Venue correction.** The brief listed this as an unplaced 2026 paper. Every page of
the v2 PDF carries the header "Published as a conference paper at COLM 2026". Cite it
as COLM 2026.

Read in full: main body and the discussion. Appendices B through E were consulted only
where the main text points at a specific table.

## Local cache

- [PDF](../../../../../ignore/papers/2604.06374v2-illusion-of-superposition.pdf)
- [Extracted text](../../../../../ignore/papers/2604.06374v2-illusion-of-superposition.txt)
- 1,787,054 bytes, 32 pages, 2,355 text lines.
- SHA256: `6a18a6faf83c926dd3c7a807e6d05aa309646c9a863ec0382538f432e100187b`.

## What it actually does

The paper asks one question in three regimes: does a language model reasoning in a
continuous latent actually hold several candidates at once?

It defines two kinds of superposition. FORCED superposition is built at the input as a
convex combination of token embeddings. LEARNED superposition is whatever emerges when a
model is trained to use latent thoughts on a task that rewards parallel exploration.

    regime          model                 verdict
    training-free   Soft Thinking on      collapses in the first few layers
                    QwQ-32B, Qwen2-1.5B
    fine-tuned      Coconut on GPT-2,     never learned; a shortcut is found instead
                    SmolLM2 135M/360M/1.7B
    from scratch    small GPT-2 style     superposition appears and the latent is load
                    on symbolic ProsQA    bearing

The instruments are Logit Lens (project an intermediate hidden state through the
unembedding and read the entropy of the resulting distribution) and entity-level
probing (classify graph entities into correct-next, wrong-neighbour, target and other,
then watch the belief evolve across reasoning steps).

## The key numbers

Training-free, QwQ-32B on MATH500:

- Entropy across layers is visually identical for Soft Thinking and discrete CoT.
- KL(soft || discrete) falls to about 1e-4 in the middle layers.
- Cosine similarity between the soft-token forward pass and the argmax forward pass is
  0.996 +/- 0.025 on QwQ-32B, 0.998 +/- 0.013 on Qwen2-1.5B.
- The mixing weights are already peaky: entropy 0.18 +/- 0.34 nats on QwQ-32B. There was
  little superposition at the input to collapse.

Fine-tuned Coconut on ProsQA, counterfactual no-latent evaluation:

    model             CoT     Coconut   no latent   drop
    GPT-2 124M        85.3    99.0      99.0        0.0
    SmolLM2-135M      72.7    93.3      92.3       -1.0
    SmolLM2-360M      85.0    98.7      98.0       -0.7
    SmolLM2-1.7B      98.3    100.0     100.0       0.0

Deleting the latent thoughts costs at most 1.0 point. Entity probing shows the TARGET
entity dominating from step 0 with no progression: the model solves the task in one
forward pass and copies the answer through the latent positions.

From scratch, 8 heads, 768-dim, with and without latents:
2 layers 94.5 / 13.8, 4 layers 96.2 / 16.0, 8 layers 95.2 / 53.0, 12 layers 91.2 / 34.1.
Here the latents are necessary.

The decisive table is the parameter-matched one at about 15M parameters (their Table 3,
mean +/- std over 3 seeds):

    config       params    with latents   without latents
    2L-768d      14.97M    96.0 +/- 0.9   30.9 +/- 8.4
    4L-544d      14.78M    91.8 +/- 4.9   39.7 +/- 20.8
    8L-384d      14.61M    81.6 +/- 4.3   73.2 +/- 1.4
    12L-320d     15.14M    72.4 +/- 7.1   70.0 +/- 6.3

WIDTH decides whether the latent is used. At 320 dimensions the latent is worth 2.4
points and the model has degenerated to the no-latent shortcut. Below a critical width
(their Table 4, d <= 64) depth buys nothing: the best of eight configurations reaches
68.3 % with latents and 65.4 % without.

The explanation offered for the pretrained collapse is the next-token objective itself.
Final-layer entropy over uniform k-token mixtures collapses in a pretrained Pythia-1B
and stays near maximum in the same architecture with random weights, and the collapse
emerges GRADUALLY across Pythia pretraining checkpoints.

## Where it sits in MORPH's 2x2

It is the negative reading of the deterministic axis, on both the one-stream and the
K-stream sides. It says a pretrained next-token model projects a mixed state onto its
nearest discrete interpretation, and it says a narrow model never learns to use the
latent at all.

## What in MORPH already tested this

The Thought Register is the local replication of the fine-tuned half. Four cells,
seeded apart by four learned queries, collapsed to effective rank 1.24 of 4 at mean
pairwise cosine 0.94 (`tul.slot_cells: 4`, filed 2026-09-13). The distinct-init and
same-init arms landed 0.008 rank units apart, which is their "superposition is not a
naturally preferred strategy" in one number. Our own conclusion, that the 0.022 CE win
was PREFIX WIDTH, is their width finding arrived at from the other side.

The zero-worth instrument is the local version of their no-latent evaluation. On arm A
the frozen coda preferred NO cell to the loop's by 4.64 nats (`zero - own` = -4.6413,
twin-read probe, 96 rows). Their maximum drop is 1.0 point; ours is a sign flip. Both
say the reader does not need what the latent holds, but ours is confounded in a way
theirs is not: `the-reader-was-the-limit` showed that unfreezing the coda alone turns
the same cell from -4.64 to +0.177. They never test an adapted reader, so their number
is also a statement about a fixed pair.

The width finding is the uncomfortable one. MORPH is d_model 768 with one slot cell of
768 dimensions and `prefix_k: 2` coda positions. That is the 2L-768d row, not the
12L-320d row, so their width explanation does not predict our flatness. What our tree
does share with their deep-narrow row is DEPTH: a mean-6 loop over a shared core.

## What a MORPH arm implementing it would need

Nothing new to build. Their instruments map onto instruments we already run:

- Entropy across layers becomes the per-pass entropy of `lm_weight() @ cell`. We do not
  currently log this; the closest is `val/slot_cell_eff_rank`.
- The no-latent counterfactual is our zero-worth probe, already in the twin-read probe.
- Entity belief evolution has no clean analogue on web text. The nearest is the
  set-membership readout proposed in the Reasoning by Superposition note.

The arm their Table 3 actually calls for is a WIDTH sweep of the cell channel at fixed
parameters: `tul.prefix_k` in 1, 2, 4, 8 against a matched-parameter control. The
pk8 control (`tul_slot_spandec_strict_pk8.yaml`) is already written and queued as the
clean width control for the register.

## The instrument it implies

Per-pass entropy of the cell read through the tied LM head, against the entropy of the
plain model's hidden state at the same depth. Their claim is that a next-token
pretrained model drives this to near zero in the last layers. MORPH's cell is not
pretrained on tokens, so if the entropy is ALSO near zero the cause is ours, not
inherited.

The second instrument is the width-matched pair, because their result is that width and
not depth decides whether the latent gets used at all.

## What it does NOT establish for us

It does not establish that a narrow latent is our problem. Our cell is 768 wide, in
their winning row.

It does not establish that from-scratch training fixes it. Our slot loop IS trained from
scratch and still reads flat, so the from-scratch / fine-tuned split is not the axis
that separates us from their positive result.

It does not measure a K-VECTOR register. Their superposition is inside one vector over a
token vocabulary. It says nothing directly about M mutable cells, which is the shape
the Thought Register ran.

It does not settle the cause. Their own discussion says the honest open question is
whether token-level superposition is even desirable, and suggests that meaningful
parallel exploration probably needs superposition over whole strategies rather than
individual tokens.
