# Brief for literature readers (MORPH / TUL slot loop, 2026-09-13)

You are reading papers FOR a specific measured failure. Read the paper against it, not in general.

## The system
MORPH is a Parcae-style looped transformer (3 prelude blocks, a 6-block weight-shared core
looped T times with per-sample Poisson depth mean 6, 3 coda blocks; d_model 1024; ternary
weights; ~330M params; web text, seq 1024, 5k-step runs at 1e-4). TUL ("Thought Unpack Loop")
adds ONE slot position per sentence-like span (about 50 slots per 1024-token row). Only the
slot positions go through the loop ("the slot loop"); tokens go prelude -> coda once. The
slot's exit state z is written into 2 prefix cells the coda reads, and z is graded by a span
decoder (teacher-forced decode of the NEXT span from z) or by a next-span bag target. Intent:
think once per span, decode cheaply per token (amortized computation).

## What is measured (all with paired bootstrap CIs over 480 rows)
- Loop contribution is read from a forced-depth sweep: K1-K6 = CE(depth 1) - CE(depth 6) on
  tokens. A plain (no-slot) MORPH earns 0.185 nats from its loop under a noise entry, 0.033
  under a prelude entry. The slot loop earns 0.0003 to 0.002 on tokens in ELEVEN+ arms, and
  K3-K6 is ~0 or negative. Passes 2-6 do nothing; pass 1 does everything.
- Per-pass anatomy: one pass sets the scale, the rest rotate; the loop is a power iteration
  (rank of the update collapses 181 -> 9); consecutive pass updates cancel (cosine -0.2 to
  -0.6); the state reaches a fixed point by pass 6; gain ~0.2-0.9.
- Every per-pass target we gave (progressive loss, per-pass LoRA, oracle gradient
  trajectory, denoising-style staged targets, gradient-conditioned passes, token map in the
  core) is MET IN ONE PASS: pass-1 already achieves it, passes 2-6 add <= 0.002.
- Rank of z: the prelude makes ~13 effective dims per row of ~50 slots; the loop keeps it
  (13.2); the prefix write cuts it to 7. Levers that separate z's (a 4-cell register, centering,
  a 4-layer non-shared post-loop stack) lower it or leave it. A reader with MORE capacity (4
  non-shared blocks after the loop) makes the passes matter LESS and costs 0.006 nats.
- Causal fitted-z probe: a z fitted by gradient descent using only past context is 0.25 nats
  WORSE than the loop's own z; a 10 % rms perturbation of z moves the coda by 0.0005 nats.
  So the coda barely reads the channel; a one-pass map already gives it everything the
  target asks for.
- Our information-theoretic note (Lean-checked): a deterministic map after or inside the
  loop adds no information about the future beyond what the entry carried; value must come
  from RELAY (moving information the reader could not otherwise reach) or EXTRACTABILITY
  (putting it in a form the reader can use). Strict geometry arms (slots the ONLY cross-span
  channel) confirm: the cross-span budget is 0.40 nats and the slot channel carries 0.18 of
  it, all from pass 1.
- The paid loop (tokens AND slots through the loop) earns 0.010-0.17 nats but is not the
  design (no amortization). A token loop restricted to its own span earns 0.010.
- Matched compute: a depth-1 plain model with the same block-passes per token beats every
  slot arm by 0.25-0.33 nats at 5k steps.

## The open questions Wolfe asked tonight
1. Is the geometry the loop needs and the geometry next-token prediction needs "too
   disjoint"? (i.e. does a web-text NTP target ever produce an iteration-requiring map?)
2. Is scale the issue (multi-token prediction only works at ~3B params)? Does depth earning
   in looped LMs appear only above a parameter threshold?
3. What "dramatically different" design still AMORTIZES computation (think once per span or
   per chunk, decode cheaply per token) but gives later passes a job a one-pass map cannot do?
4. What are the mathematical principles: when does an iterated map have to iterate? (saddles,
   attractors, set-valued maps, stochastic trajectories, memory-budget bounds, power method).

## What I need from you, per paper
- What they measured and how (model size, data, the loop-contribution instrument, and the
  matched-compute control if any). Numbers.
- The MECHANISM they claim makes later passes matter, in one paragraph.
- Does their evidence bear on Q1-Q4? Answer each explicitly, "no bearing" is fine.
- The one concrete arm it suggests for OUR setting (a target, a wiring, or a training rule),
  with what we would measure (K-curve on tokens, per-pass instrument) and your prior.
- Anything that CONTRADICTS a claim in this brief.
Cite the section/figure/table for every number. Do not pad. If the paper is thin, say so.
