# Response to your coarse-to-fine design: what we measured, and a request for critique

Thank you for the design. Many of its parts are things we have built and measured in the last two months, so I can
give you evidence instead of opinion. I will say where our results agree with your proposal, where they disagree, and
what we still do not know. Then I ask for your critique.

Repository (our current testbed code and the filings behind every number below):
`https://github.com/bigwolfeman/MORPH-Orchestrates-Recursive-Pruned-Hierarchies/tree/master/docs/references/looping-depth/parcae/lxtul-testbed-2026-10-16`
Start at its `README.md`. The model code is `lxtul/spiral.py` (coarse-to-fine loop and the cycling variant),
`lxtul/plain_pointer.py` (the plain looped baseline), `lxtul/plain_deep.py` (HRM deep supervision on the plain loop).
Each claim below has a filing in `filings/` with the predictions we wrote before the run.

## 1. The goal

We want a looped transformer that thinks in whole semantic thoughts (spans), not only in next tokens, and that is
cheaper in TRAINING than a plain looped model. We want to keep autoregressive generation, because AR models behave
in ways we need (for text modelling and for RL). Our long-term design (MORPH, the same repo) is a Parcae-style looped
model; the idea we call TUL puts a loop over span-level states and a small decoder (the coda) that turns them into
tokens.

## 2. The testbed and the metric

- Parcae (repo sandyresearch/parcae): prelude 4 layers, a 6-layer core looped at a per-row
  Poisson depth (mean 6) with a contractive diagonal injection, coda 4 layers. d 1024, seq 1024, OpenWebText,
  StarCoder2 tokenizer (vocab 49152, uniform CE 10.80), 5000 steps, batch 6, one RTX 5090.
- Every arm has a 4-head output-only pointer (copy) head.
- CE is measured on 480 fixed rows (510,003 tokens), paired across runs. Seed spread of CE@6 is about 0.010 nats.
- Loop contribution: K1-K6 = CE with 1 loop pass at eval minus CE with 6. It measures dependence on depth, not
  quality: a skip connection around the loop made K1-K6 fall from 0.0377 to 0.0008 while CE got 0.014 better.
- The bar: loops are judged against loops (plain Parcae). At 5k steps a flat model wins; looped models need more
  tokens to reach their knee, so the flat models are references only.
- Our rules: only the main next-token CE may buy depth (no auxiliary depth losses); no per-pass jobs; no curricula
  that force depth; compare at equal FLOPs and at equal wall clock.

## 3. Results (5k steps, 2-seed means unless named)

| arm | CE@6 | K1-K6 | train ms/step | train TFLOP |
| --- | --- | --- | --- | --- |
| plain Parcae loop + pointer (1 seed) | 3.7256 | +0.0377 | 283.0 | 28.5 |
| SpiralFormer as published (arXiv 2602.11698): one core pass per level, chunks 16, 8, 4, 2 | 3.6912 | 0 (fixed depth) | 138.0 | 8.7 |
| spiral, finest level (chunks of 2) looped at Poisson depth | 3.6907 | +0.0071 | 234.2 | about 17 |
| spiral, chunks of 4 at Poisson depth | 3.6930 | +0.0008 | | |
| a span level (punctuation-bounded spans) inside the spiral | (flat) | +0.0001 | | |
| Block-Parcae (series cut between blocks), chunks of 4 | 3.9529 | +0.0076 | | |
| NCM-style span code (residual VQ levels) on a flat model | bound worse | +0.0003 to +0.0185 for 1 to 4 levels | | |
| variational LCM (one-stage ELBO) | bound 4.41 | flat | | |
| best-of-8 span plans (Explorative Modeling style) | bound worse by 0.08 | +0.0012 (control +0.0002) | 190.2 | 11.0 |
| flat 8 layers + pointer (reference) | 3.6996 | | | |
| flat 14 layers + pointer (reference) | 3.6731 | | | |

The coarse-to-fine loop (SpiralFormer) is our best result: better CE than plain looping at 0.31x the FLOPs. But its
gain comes from looping at reduced resolution, not from any span or concept mechanism. Every span-level mechanism we
added used almost no depth.

## 4. Your design against our evidence

**Strict inter-span masking (your section 4, first bullet).** We built exactly this ("strict geometry": the prelude
sees its own span only, the coda's only view of earlier spans is the loop's output). It does make the latent
necessary: with no slot channel at all, web text loses about 0.4 nats of cross-span information. But it has two
costs we measured. First, CE: strict LXTUL trailed plain Parcae by 0.294 at 5k, and 86 % of that gap was
cross-span copying (far bigram tokens). An output-only pointer head closed most of it (strict + pointer vs plain +
pointer: +0.045). Second, and more important: making the latent necessary did NOT make the loop use depth. The span
loop carries the information in one pass; extra passes add 0.002 to 0.02 nats. "Necessary" and "deep" are different
properties. Also, from our history: never decode a span from one vector with no token path at all; that collapsed
every time (Huginn-style span decode, MegaByte-style, Bowman-style).

**JEPA targets with an EMA encoder (your section 4, second bullet).** Tried in three forms. A latent-prediction
target on the loop's exit state collapsed (the loop's K-curve fell). An EMA target moved during training but the
moved target reached nothing the reader used. A frozen encoder that pooled the live prelude was not a fixed target:
its codes collapsed (pairwise cosine 0.61 to 0.99 by step 3000). Stop-gradient prevents the trivial collapse but not
the reader ignoring the target.

**Flow matching for span expansion (section 3).** Our flow "thinker" over span codes was context-blind: the past
explained about 1 % of its loss, and more Euler steps gave no gain (we have a Lean proof that Euler depth is flat on
that objective). The blurry-mean problem is real; flow matching fixed the blur but not the use of context.

**Best-of-K winner-take-all (section 4, third bullet).** Measured today. K = 8 plan heads after a span loop, the
winner picked by a parameter-free score through the tied output head, Sinkhorn-balanced. The coda uses the plan (a
random plan costs 0.13 to 0.17 nats per token), but the loop does not need depth to make it (K-curve +0.0012), and
the ln K per span is not repaid at 5k (sequential importance sampling bound 3.725 vs 3.704).

**The NCM shape.** A paper after your training window (Next Concept Modeling, arXiv 2610.14162) does much of your
section 3: multiscale residual-quantized span codes, generated coarse to fine. In our port, the code cost more to
predict than it saved (0.226 nats per token cost against 0.118 saved; a PMI-shaped code: 0.260). But the K-curve grew
with the number of levels (+0.0003 at 1 level to +0.0185 at 4). That is the only lever we found where a span-level
loop uses depth, and it points the right way for scaling. Their speed claims are partly FLOP accounting, and their
open-web generative perplexity is 83 against 35 for AR.

**The chunk lever.** Finer chunks use more depth (chunks of 2: +0.0063; chunks of 4: +0.0029; at Poisson depth,
+0.0071 vs +0.0008). Coarse levels are cheap but shallow; that is the wrong direction for a hierarchy that should
think at the coarse level.

**Cayley hyper-connections for stability (section 3).** Our production model uses Cayley n=4 hyper-connections.
They did not prevent loop instability. The instability was an alignment of amplifying directions and a collapse of
state rank (1.7 to 4.8 effective dimensions in 1024), and four interventions that bound the weight spectrum failed.
What held: a 1000-step LR ramp, and a fixed-point term. And that term (it penalizes change between passes) froze the
loop's depth use; with it off, the loop earned more. A recent dynamics paper (arXiv 2610.18174) found the same:
damping oscillation before the answer stabilizes hurts.

**Hyperbolic geometry and learned chunking.** We have a hybrid Euclidean + Lorentz embedding (25 % Lorentz) in
production, but we never isolated its effect, so I cannot confirm or refute your geometry argument. We have not tried
H-Net-style learned boundaries; we use fixed chunks and a punctuation boundary rule.

**Your takeaway box.** "Less than 2 % of the KV cache and FLOPs of token-level CoT" has no measurement behind it that
I can see. Our spiral gets 0.31x the training FLOPs of plain looping, and that came from resolution, not from planning.

## 5. In flight

The "cycling spiral": the whole coarse-to-fine descent (chunks 16, 8, 4, 2) repeats per loop, with a Poisson number
of cycles, a sampled start level and random level skips (so the start level tells the model what comes next, and no
level is always present). Two credit rules: one CE at the end (Parcae, backprop through the last 3 cycles), or a CE
after every cycle on a detached state (HRM deep supervision). And plain Parcae with deep supervision, to separate the
HRM credit rule from the hierarchy. The hypothesis: a single descent converges in a pass because its conditioning is
fixed; repeating the hierarchy moves the fixed point (each cycle's coarse levels pool what the fine levels just
inferred), like a multigrid V-cycle or HRM's two timescales.

## 6. What I ask you

1. Is there a flaw in our metric? We treat K1-K6 (CE at depth 1 minus depth 6) as the loop's contribution. Would you
   measure "the loop thinks in spans" differently?
2. Every span mechanism carries information in ONE pass. Your design puts the looped planner at the coarsest scale.
   What would make a loop at the coarse scale need more than one pass on next-token CE alone, with no auxiliary depth
   loss? Is that even possible for web text at this scale, or does it only appear on tasks with search?
3. Our chunk lever goes the wrong way: coarse levels are shallow. Why, in your view? Is it the information content
   per position (a coarse state is easy to compute), or the credit path (the fine levels absorb the gradient)?
4. Do you predict the cycling spiral uses depth? Which credit rule, end or deep, and why?
5. You propose byte-level input with learned chunking. Our results say resolution is the main source of savings.
   Would learned boundaries change the depth question, or only the FLOPs?
6. Given that best-of-K plans are used but shallow, and that NCM levels are deep but expensive, is there a design
   that keeps AR token generation, adds one sampled span-level state per span, and pays for itself in likelihood?
7. Read the code: is there anything in `lxtul/spiral.py` (pooling, broadcast with gain sqrt(group size), causal
   shift, Anchor topology `h = h0 + u`) that would stop a coarse level from using depth?

Please be direct about where you think we are wrong.
