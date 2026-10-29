# Looped H-Net: seed ideas from MORPH and Parcae

Status: written 2026-10-29 for Wolfe's H-Net project; seeds, not designs.

## How to read this

Wolfe's frame: the objective of almost all our prior work (next-token CE with span targets that were too weak, slot targets that one pass could meet) was wrong. So every CE verdict in our records ("arm A beat arm B by N nats") is inconclusive. This doc does not use them as evidence. It uses what survives a wrong objective:

- theorems and loop dynamics,
- mechanisms and interpretability readings,
- loop-contribution behavior as a phenomenon,
- measurement pitfalls.

Tags follow the lessons doc. **[theorem]** means proved in Lean. **[robust]** means measured on several arms or seeds. **[one-shot]** means one run or one seed. **[speculation]** means the source says so.

Short names for sources:

- **LL n.m** is `docs/lessons-learned-loops-and-spans.md`, section n.m. It is the primary source. Its own citations are the way to the raw filings.
- **SO** is `docs/span-only-model.md` (laws 1 to 11).
- **Draft prereg** is `/home/wolfe/parcae-wt-spandraft/docs/experiments/planned/2026-10-28-parcae-span-draft-hierarchy-5k.md`. It was never run.
- **Harvest** is `/home/wolfe/morph-scratch/postmortem-harvest/` (A to G files).
- **PW** is `/home/wolfe/parcae-wt-spandraft/docs/experiments/` (Parcae filings: `failures/`, `successes/`, `planned/`).

Terms. *Commitment test*: a pass earns depth only if it reads something the previous pass committed to and could not compute itself. *Depth-1 twin*: the same config trained at one pass. *K1-K6*: CE at 1 pass minus CE at 6 passes. A K-curve measures depth dependence, not depth value (LL 2.3). So every kill experiment below needs a depth-1 twin, or a flat twin at equal parameters or equal compute.

H-Net facts I rely on, checked in `docs/references/tul-latent-emission/h-net/h-net.md`: a router scores adjacent cosine similarity and cuts boundaries. A smoothing module (an EMA over chunk vectors, weighted by router confidence) feeds an upsampler that repeats each chunk vector to the next boundary, multiplied by an STE confidence. An encoder-to-decoder residual passes through a Linear. A ratio loss sets the compression. The outer encoder and decoder are SSMs. The main network is a standard transformer or SSM over chunks.

Two things I did not do. I did not re-run or re-measure anything. I did not verify the paste's citations (see section B).

---

## Index of seeds

| # | Seed | What it loops | Lesson behind it | Kill experiment |
|---|---|---|---|---|
| 1 | Which stage loops | inner stage, word-rate stage, or both | LL 3.11, SO law 10, LL 2.6 | equal wall-clock pair: looped inner vs flat twin vs byte-rate loop |
| 2 | Windowed inner attention | inner main net | LL 2.7 | window vs global inner attention, depth-1 twin each |
| 3 | Series cut on the decoder | nothing new; removes the SSM crossing path | LL 2.1, SO law 1 | reset decoder state at chunk edges, read loop share |
| 4 | Narrow the residual | nothing new; audits the encoder-to-decoder skip | LL 3.3b, LL 2.3 | zero the loop output with the residual on and off |
| 5 | Byte pointer with abstain | none; takes copy off the chunk state | LL 3.8, LL 4 (copy head row) | pointer off, split the gap by copy distance |
| 6 | Rule boundaries at the apex | apex stage over our sentence-like spans | LL 2.5, SO stage 5 | learned vs rule apex cut at equal ratio |
| 7 | Wide read of one chunk state | inner state, read as several rows | LL 3.4, SO stage 3b | repeat-upsampler vs multi-row read, same loop |
| 8 | Cached coarse-to-fine draft | apex block diffusion over several chunks | Draft prereg, LL 1.4 | one-chunk draft at equal forwards, donor read |
| 9 | Thought-level verification | shallow draft, deep verify | commitment test, LL 3.6 | acceptance of depth-1 drafts by a deep vs a depth-1-trained verifier |
| 10 | Own-draft training | the verifier, trained on its own samples | LL 3.6 (teacher forcing, posterior) | verify loss on true drafts vs own drafts |
| 11 | Independent cell fan | K cells per chunk, set reader | LL 3.7, LL 1.3 [theorem] | one cell vs K cells, zeroing worth per cell |
| 12 | Injection loop and tripwires | the loop map itself | LL 1.1, 1.2, 1.10, 1.11 | injection on vs off; log `A`, gain, preclip |
| 13 | Free-running thought grade | the evaluator, not the model | LL 3.10, SO stage 6 | grade a depth-1 twin and a depth-6 model on next 1 to 4 chunks |
| 14 | Test-time breadth knob | cells or drafts at inference | CLAUDE.md reminder, LL 3.7 | train at M, infer at more than M |
| 15 | Predictability-priced code | an optional discrete thought code | LL 3.6, SO laws 8 and 9 | code CE into the encoder at beta below 1, count predictable nats |

Conflicts between seeds:

- Seeds 3 and 4 remove bypasses. Seed 5 adds a bypass for copy. They fit together only if the pointer carries exact tokens and nothing else.
- Seed 1 favors a loop at word rate (many positions). Seeds 6, 8 and 9 favor an apex at sentence rate (few positions, cheap decoding). Our data says the first earns and the second saves cost (LL 3.11). Pick one for the first run.
- Seed 11 (independent cells) and seed 15 (a code) both claim the same job, "hold several futures". Run one at a time.
- Seeds 8 and 9 both commit tokens. Seed 15 commits a code. A design uses one of the two commit forms.

---

## A. Seed ideas

### Where the loop goes

**Seed 1. Which stage loops.**

*Idea.* The paste loops only the inner stage. That is the cheapest place, because the inner stage has the fewest positions. Our evidence points the other way for value. The loop earned where it ran on every position whose loss is measured through it (LL 2.6). A span model that looped once per span trained 3.8x the tokens per second and trailed plain token Parcae by 0.314 nats CE@6 at equal GPU time, because it skipped the per-token loop that earns (LL 3.11, SO law 10). Treat "which stage loops" as the first experiment, not a decision. Arms: loop the inner stage only; loop a word-rate stage (stage 1 of a 3-stage H-Net); loop the whole U over bytes (expensive, the plain-loop analog).

*Lessons.* LL 3.11 [one-shot, equal wall clock]; LL 2.6 [robust]; SO law 10. Source for the cost case: `PW failures/2026-10-27-parcae-plain-wallclock-3h.md`.

*Commitment test.* A plain inner loop does not pass by itself. Pass r reads pass r-1 states, and pass r could compute them. Looping alone has no job. The job has to come from seeds 2, 8, 9 or 11. State this for every arm.

*Cheapest kill.* Three arms at equal GPU time, not equal steps, each with its flat twin at equal parameters: inner-loop, word-rate loop, no loop. Judge by CE in bits per byte and by the free-running grade (seed 13). The control is the flat twin, and a K-curve is reported only beside the depth-1-trained twin. If the inner loop ties its flat twin at equal wall clock, the loop has no job there yet. That outcome is "can't tell" about the loop, because the objective is the open question.

**Seed 2. Windowed inner attention so depth buys reach.**

*Idea.* If the inner net sees every earlier chunk in each pass, a pass has nothing to wait for. Bound the inner attention to a window of recent chunks, plus a coarse global level. Then information from h chunks back arrives at pass h-1, and depth buys reach.

*Lessons.* LL 2.7 [robust]: content h spans back arrives at pass h-1 and decays about a third per pass. Bounding each core layer to an 8-token window raised plain-style K1-K6 at the same CE. Giving every pass a query into far content made the loop flatter. The risk is also in LL 2.7: unbounded carry grew state RMS to 5 to 75, and a register's reach needs a segment reset on every position-local op.

*Commitment test.* Weak pass. The far content arrives late, but it was not "committed" by the previous pass. It was merely not yet visible. It is a legitimate reason for depth, but a K-curve cannot tell it from a worse depth 1.

*Cheapest kill.* Window W vs global attention, each with a depth-1 twin, at equal parameters. Read the hop staircase from LL 2.7 (K1-K6 by hop distance) on a synthetic reach-1 chain. If the staircase does not show arrival at pass h-1, the mechanism is not what moved the number.

**Seed 3. Series cut on the decoder.**

*Idea.* H-Net's outer decoder is an SSM that scans over all bytes. A scan carries state across chunk edges. That is the same kind of crossing path as our 64-token window or a multi-span read, and the bypass law says it takes the loop's share. Reset (or mask) the decoder state at each chunk boundary, so the only way for the past to reach a byte in chunk c is through the inner stage's chunk states (plus the pointer from seed 5).

*Lessons.* LL 2.1 [robust]: a span stack as a side channel was worth 0.002 to 0.011 nats; as the only cross-span path it was worth 0.12 to 0.14. SO law 1. SO law 4: exact copy does not fit a pooled state, so a series cut without a copy path pays 0.3 to 0.4 nats.

*Commitment test.* This does not give the loop a job. It raises the loop's share (how much loss must go through it). Share times serial benefit is the loop's contribution (SO), so this is necessary and not sufficient.

*Cheapest kill.* Cut vs uncut decoder, same loop. Zero the inner stage's output at eval and read the loss increase (its share). Do this at depth 1 and depth 6. If the share rises at depth 1 as much as at depth 6, the cut moved load and not depth (LL 2.2: load is not depth).

*Open point.* The encoder SSM also crosses chunk edges, and the router reads it. I do not know how much cross-chunk context the router needs. That is a question for the router, not a claim.

**Seed 4. Narrow or gate the encoder-to-decoder residual.**

*Idea.* The residual in H-Net carries fine detail past the inner stage. It is also an Anchor skip. Measure what it bypasses before building anything on the inner loop. Variants: keep it; pass only the in-chunk bytes (equivalent to seed 3 for the residual); gate it by a learned scalar per chunk and watch the gate.

*Lessons.* LL 2.3 [robust]: an Anchor skip (the coda reads h0 + u) took plain Parcae's K1-K6 from +0.0377 to +0.0008 at a slightly better CE@6. Topology alone moves the metric. LL 3.3b [one-shot, CPU probe]: under a skip, each hierarchy level learned to ignore its incoming update. Removing a level's update changed the output by only 2.9 to 5.8 %. Coarse levels were worth almost nothing once the coda could read tokens.

*Commitment test.* If the residual lets the decoder solve the byte without the inner stage, the inner stage has no job on that byte. The residual cannot carry what the byte's past (outside the chunk) holds, so there is still room. The test is empirical.

*Cheapest kill.* With the model trained as is, replace the inner stage output by its depth-1 version at eval, and separately zero it, residual on. Compare the two losses. Then repeat with the residual narrowed. Control: a model with no residual at all. The worry from LL 3.3b is the inner stage output being ignored. The audit reads that directly.

### What a "thought" is

**Seed 5. Byte pointer with abstain.**

*Idea.* Copying at byte level is a large share of any cross-chunk gap, and a chunk vector does not hold token identity. Put a pointer over earlier bytes (keys from the encoder states) with explicit null mass, so it can say "no copy". Then the inner stage is free to carry meaning.

*Lessons.* LL 3.8 [robust]: far-bigram tokens were 17.9 % of tokens and 85.7 % of the strict-slot model's gap to plain. Bigram repeats 33 to 256 tokens back were 65 % of the gap; the gap fell off at 256, the plain model's attention window. A normalized pointer closed about two thirds. A pointer without null mass gave pessimistic CE, and re-scoring old checkpoints did not fix it; retraining did (LL 4, copy-head row). Byte level changes the units, not the mechanism. A byte repeat is shorter, so the distance classes must be redone in bytes.

*Commitment test.* None. It is a bypass by design. It exists so the loop is not asked to do a job it cannot do.

*Cheapest kill.* Pointer on, off. Split the loss difference by copy distance in bytes. If the pointer's gain sits at short distances that the decoder SSM already covers, drop it. Watch the generation: greedy loops appeared in every pointer arm (SO stage 3b), so a repetition check is a gate here (LL 4, generation readouts).

**Seed 6. Rule boundaries at the apex stage.**

*Idea.* Our boundary rule (cut at `.;!?`, newline and dashes; no comma; min span 4, cap 32 in tokens) is causal and used by loader and generator alike. Use it for the apex stage's cut, with H-Net's learned router for the lower stage. Or train the router and read its agreement with the rule. This makes a "thought" a sentence by construction, which is what Wolfe's amortized decoding needs: whole thoughts as units.

*Lessons.* LL 2.5 [one-shot, eval-only]: per-token depth is private to its token. Boundary tokens given depth 6 kept 0.898 of their own gain with neighbors frozen at pass 1; later tokens got 0.004. So a loop's gain does not travel along the sequence by itself. SO stage 5: a sentence-rate loop lost its cost case. Conflict: that is the case against a sentence-rate loop (seed 1). A learned router at the apex might pick units that the loop likes. I do not know.

*Commitment test.* Not given by the cut. It sets the unit. The job comes from seeds 8 and 9.

*Cheapest kill.* Learned apex cut vs rule apex cut at equal compression ratio. Track the router's own distribution of chunk lengths against ours. Control: a stage with the same ratio and a fixed stride. Do not pass or fail on a cosine between cuts (Wolfe rule). Compare on bits per byte and on the free-running grade.

**Seed 7. Wide read of one chunk state.**

*Idea.* H-Net's upsampler repeats one vector to the next boundary. Replace the repeat by a read of the chunk state as several K/V rows (cross-attention from the decoder). The width is free in cost.

*Lessons.* SO stage 3b [robust within the line]: a split read of one state as 5 rows was free, and the extra rows carried over half of the arm's K1-K6, while extra loop cells took share and filled in pass 1 (LL 2.8: width is not depth). SO law 6. Caveat: this was a single seed at 5k.

*Commitment test.* Does not apply. It helps the work of later passes reach the decoder. A one-vector repeat may hide whatever later passes add.

*Cheapest kill.* Repeat-upsampler vs 5-row read, same loop, depth-1 twin each. Read the K1-K6 delta against the CE@1 delta, because K1-K6 ranks arms in CE@1 order (LL 2.3). If the CE@6 is the same and only K1-K6 moves, nothing was gained.

### Hierarchy over several thoughts

**Seed 8. Cached coarse-to-fine draft in bytes.**

*Idea.* This is the unfinished design from the draft prereg, moved to H-Net. The apex stage works as block masked diffusion over several future chunks. It first commits a coarse fraction of the slots of every chunk in the block (level 2, cached), then more (level 1), then the leaf chunk to completion. The level below reads the committed bytes. Tree variant: blocks tile the chunks. Zoom variant: a rolling window of the leaf plus the three after it. In H-Net the inner stage already works on chunks, so the "canvas" is a chunk's bytes (or the decoder's byte canvas) and the draft lives in byte space.

*Lessons.* Draft prereg, Hypothesis: every latent hierarchy here lost to its bypass (span NCM paid 0.226 nats per token for level 1 and saved 0.118, LL 3.6). A level of committed tokens avoids a latent. In training a level holds true tokens; at inference it holds the model's own samples, which is a real commitment. LL 1.4 [theorem, linear]: a depth-independent target read at a random depth is only met by a one-step map. So a draft hierarchy must be a different job from "reach the answer".

*Commitment test.* Passed by design. A sampled byte cannot be recomputed by the next pass. Whether the leaf reads it is the test. The prereg's donor read measures that (swap the cached drafts for a donor context's, grade the leaf).

*Cheapest kill.* Use the prereg's own controls. The control is the one-chunk draft decoded at the tree's forward count before its first span (16 per span), also at 8. Donor read: a donor's drafts must lower the leaf's grade, or the leaf ignores its draft (LL 3.3b predicts this failure). The prereg warns that the tree's training signal is thinner by design (62.5 % of the one-span arm's spans per row carry loss). A fail with the donor read supported is "can't tell", and the next test is a longer run. No result exists to compare with.

**Seed 9. Thought-level verification (speculation over whole thoughts).**

*Idea.* This is Wolfe's goal in one move. A shallow pass of the same weights (depth 1, or the outer stages only) drafts the next k chunks. One deep pass over all k drafts verifies them in parallel, the way speculative decoding verifies tokens. Accepted chunks are kept. The first rejected chunk is resampled. The deep pass reads the draft, which is committed and not recomputable. That gives the loop a job by the commitment test, and the speedup is the accepted length per deep pass.

*Lessons.* LL 3.6 [robust]: a deterministic head on a high-entropy future converges to the marginal, and a sampled code is an unconditional draw. So the draft must be a real sample, and acceptance must use probabilities, not cosines. LL 2.3: do not read depth value from a K-curve. LL 4 (fixed vs sampled depth): state the depth law. LL 2.2: a verifier that finishes in one pass earns no depth, so check whether the deep pass beats a depth-1 verifier trained the same way.

*Commitment test.* Passed in form. The open point is whether verifying needs depth at all. A depth-1 verifier trained for the same job is the control.

*Cheapest kill.* Measure the acceptance rate of depth-1 drafts under (a) a deep verifier and (b) a verifier trained at depth 1 with the same weights budget. If (b) accepts as much as (a), depth has no job in verification. Report accepted length per deep pass and the wall clock together (LL 3.11: throughput is not CE per second). Report the draft quality with the free-running grade (seed 13).

**Seed 10. Own-draft training.**

*Idea.* Training the verifier on true drafts teaches it nothing about acceptance. Train on the model's own drafts, with a loss that is measured through the verification (the main CE on the chunk bytes after the verify pass), so the loss reaches the loop only through the committed draft.

*Lessons.* LL 3.6 [robust]: under teacher forcing a committed prior draw carries no target information, because proper scoring makes the optimal leaf ignore it; CE through samples rewards hedged draws; commitment needs a posterior at train. Teacher forcing is a bypass (memory: `teacher-forcing-is-a-bypass.md`; the denoise filing `lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md` is where passes 2 to 6 denoised the entry and pass 1 never learned). LL 2.1: under teacher forcing, pass 1 learned nothing (cosine 0.02 for 20k steps).

*Commitment test.* This is the condition that makes seed 9 real. Without it the verifier will ignore the draft slot.

*Cheapest kill.* Verify loss with true drafts vs own drafts (sampled by the shallow pass, detached). Control: verifier with the draft slot zeroed. Measure how much the loss falls when the draft is zeroed (draft worth), in both training modes. A zero worth means the draft is ignored. Mind the paste's "terminal-reward RL": a sampled draft with a reward needs a variance plan, and I have no measurement of it.

**Seed 11. Independent cell fan with a set reader.**

*Idea.* To "explore several ideas", give each chunk K independent cells (no router, no reset). The decoder cross-attends to all K as a set. Cells that must work alone stay distinct.

*Lessons.* LL 1.3 [theorem]: two fixed points of a contraction cannot coexist, so keeping K hypotheses needs expansion or a persistent per-particle input. LL 3.7 [robust]: a shared write or a mean read blurred cells (cosine 0.94 to 0.997). Winner-take-all with a write-all read kept cells distinct. A set reader kept cosine at -0.28. Repulsion terms were gamed (rank-1 sign split; a scalar epiplexity term paid by gating one stream); volume plus epiplexity was the one term not gamed. Particles without persistent identity collapsed to one (LL 1.3). Stream identity is not kept by supplying it. The fan recipe note: `.agents/notes/implemented/architecture/2026-10-11-ungraded-fan-set-reader.md` (one seed, caveats there). LL 2.8: width moves reader nats and instruments, not the loop.

*Commitment test.* Not passed. Cells fill in pass 1 (SO stage 3b). The fan is a width lever. It may make the exploration idea real only if later passes read what a selector committed. I have no evidence for that.

*Cheapest kill.* K cells vs 1 cell at equal parameters. Read zeroing worth per cell, and the share of chunks where each cell wins. Never pass or fail on a cosine. Use a depth-1 twin. Include the near-copy floor: four cells at cosine 0.99 still read an oracle gap (LL 4, oracle over samples).

### Stability and the loop map

**Seed 12. Injection loop with a logged map, and the tripwire kit.**

*Idea.* If the inner net is looped, re-supply the chunk embedding each pass (Parcae `DiagonalInjection`) and log the map. Ship the instruments on day one: the learned decay `A`, the first-iteration scale (`loop/core_gain_t0`), the composed Jacobian gain along the loss direction in fp32, and the `preclip/total` tripwire. Warm up the LR over about 1000 steps. Use a fixed-point term at a small dose. Keep rounding out of the loop (bf16 carrier yes, FP8 forward no).

*Lessons.*

- LL 1.2 [theorem]: re-injection keeps inputs apart; a once-only input is forgotten at rate L^T.
- LL 1.1 [robust]: a quiet core sits on its injection floor (typical gain `sqrt((704 + sum A^2)/1024)`, 0.865 at init `A` for our width). The numbers depend on width and init. In a new model recompute the floor and read "floor plus blocks' part". `A` never moved in any model we read.
- LL 1.10 [robust]: detonation is a backward-product spike near gain 1; the tripwire `preclip/total > 1e4` at step 200 or later caught 17 of 17 with 0 false positives in 44 healthy runs. This was measured on MORPH's recipe. It is a guide for a first threshold and not a transferable constant.
- LL 1.11: the LR ramp and the fixed-point term (dose matters: 1.0 vs 0.1 decided whether the loop moved at all) stopped detonation, and the doc reading is that earning and instability were one quantity.
- LL 1.12: FP8 forward inside the loop cost 55 % of K1-K6. LL 1.5: per-linear spectrum caps miss the composed map. LL 1.9: read the gain along the loss direction, not a random one. LL 1.13: ternary scale rule sets per-pass strength (only if H-Net is quantized).
- DIVERGENCE: `lab/divergence/DIVERGENCE-README.md`.

*Commitment test.* No. It is stability, not job (LL 1.11: separate axes, except when they are the same event).

*Cheapest kill.* Injection on vs off, same loop, depth-1 twin each. If the K-curve or CE does not separate them, the injection is not doing what the theorem says here. The logged `A` at the end of training is the read of whether the floor argument transfers.

### What makes thinking pay

**Seed 13. Free-running grade of whole thoughts.**

*Idea.* Wolfe's belief is that the span objective was too weak. Before choosing a new objective, fix the judge. Next-token CE is the chain rule and is blind to whether a whole thought is right. Grade free-running: the model writes the next 1 to 4 chunks from many contexts cut on a boundary, and a frozen encoder scores meaning against the truth. Use the tool and design in `PW successes/2026-10-28-parcae-span-grade.md` (tool: Parcae `lxtul/span_grade.py` on branch `lxtul/spangrade`, also in `/home/wolfe/parcae-wt-spandraft/`).

*Lessons.* LL 3.10, SO stage 6 [one-shot, 480 contexts]: the grade restated CE's ordering; the span model's gap was content. Both models scored under the echo baseline at horizon 4, so the encoder grades topic and content, not fluency (SO law 11). LL 4: gen-PPL rewards repetition ("the the the" 1.46 vs real text 32); use a diversity guard. LL 2.9: a loop can be depth-flat on a task it solves, so flatness is not a verdict.

*Commitment test.* Not a model change. It decides which seeds are even readable.

*Cheapest kill.* Run the grade on a depth-1 twin and a depth-6 model, plus the echo and donor anchors. If the grade does not separate a trained model from the echo at horizon 1, the encoder is too coarse for the job and the grade is not usable.

**Seed 14. Test-time breadth knob.**

*Idea.* Wolfe's standing reminder: train at M cells (or drafts), infer at more. Under a latent-selected loop the cell count is search breadth per pass, a test-time compute knob. In an H-Net the same knob is the number of drafted chunks or the number of samples per chunk.

*Lessons.* CLAUDE.md (variable cell count note). v1 measured 2026-10-13: a downward knob works (M = 1 cost +0.005); no arm gained above the trained M; cyclic or fresh cells cost +0.05 to +0.14. The design note for v2 is `.agents/notes/proposed/architecture/2026-10-12-fan-variable-cell-count-design.md` (exchangeable cells, pooled read). The theorem LL 1.3 says a contractive loop cannot hold several hypotheses, so breadth above training needs per-cell identity that does not collapse.

*Commitment test.* Only if the extra breadth is read by a selector or a verifier (seed 9).

*Cheapest kill.* Train at M, infer at 2M and 4M. Control: M trained as a fixed number. If nothing gains above M, the knob is downward only.

**Seed 15. Predictability-priced code.**

*Idea.* Only if a latent between thoughts is wanted. A code per chunk, priced by how predictable it is, with its own code CE flowing into the encoder at a weight below 1. The code is worth only its predictable part.

*Lessons.* SO laws 8 and 9 [one-shot x2]: a KL-priced code sits on whatever free-bits floor you give it; a fixed-rate FSQ code is unpredictable (the prior learned 0.5 to 0.7 of 11 nats per span, so generation feeds near-random codes); feeding the loop the features opens the history channel. LL 3.6: sampled codes are unconditional draws; a predicted continuous concept decodes worse than none (CE 14.91 vs 14.63); a variational concept collapsed. Stage M (predictability pricing) is "not planned" in SO.

*Commitment test.* A sampled code passes the form, but our evidence says the decoder's token path makes it unneeded. Honest: no evidence it helps.

*Cheapest kill.* Count nats of the code the prior predicts vs total. If the predictable part is near zero, drop it. Control: the same model with the code off and the loop fed features.

---

## B. Collisions with the research paste

Rule for this section: the paste is LLM output, and I treat each of its claims as unverified. Cartesia's "From Tokens to Bytes II", "Latent Thought Flow arXiv:2606.16222", "RMSNorm achieves 95% of the gradient stability of Stiefel projection", "cut D_0 by 2x", "<0.2% FLOP overhead" and every compute multiplier are claims I could not check and did not use. The Hwang et al. mechanics were checked (see top). For each proposal, what our measurements say.

| Paste proposal | What our lessons say | Survives? |
|---|---|---|
| Halt when `1 - cos(z^r, z^(r-1)) < eps` | A settled loop is a constant write to the optimizer (`settled_block_is_write`, `MapCause.lean:463`, LL 1.4 [theorem, linear]). Web-text loops finish by pass 3 (LL 1.7), so a convergence rule exits almost at once. A forced-depth K-curve measures convergence time, not computation (Harvest G, L10). Halting papers report step counts, not loop contribution (Harvest E, L5). Zero-ablating the shared bias costs 0.4 to 1.3 nats in Huginn and HRM-Text, and the steps are 99.6 to 99.9 % perpendicular to it (LL 1.8), so a cosine between iterates mixes a big constant with a small step. | Only with a change. Halt on a verifier's acceptance (seed 9), or at a depth drawn from a law you state. Use convergence as a log, not a controller. |
| RMSNorm or orthogonal/Cayley loops "to keep gradient norm" | The carrier constant: after pass 1, 99.6 to 99.8 % of a carrier's energy is one shared vector, hidden in a normalized stream mean (LL 3.1). Per-linear spectrum caps miss the composed map (LL 1.5). Successful operators are non-normal, with sigma_max/rho of 4 to 5.4, and forcing gain above 1 killed training (Think Shallow, Solve Deep; `docs/references/paper-notes/think-shallow-solve-deep-2608-18222.md`). An orthogonal pass cannot expand, but two fixed points need expansion (LL 1.3 [theorem]). MORPH already uses Cayley n=4 hyper-connections, and the carrier constant formed anyway. | Weak. Orthogonal or RMS passes are fine as a stability baseline. They do not give the loop a job and they limit particles. Test them against a plain injection loop with logged gain (seed 12). |
| Silent thought chunks, dechunk gate off, no loss through them | A loop earns only at positions whose loss is measured through it (LL 2.6 [robust]; Harvest G, L20). The trajectory of loop states carries no more information than its entry (Harvest G, L11). An additive write delivers no decodable content even when perfect (LL 3.4). | Survives if later chunks' CE reaches the silent chunks through attention and the read is wide (seed 7). Without a measured loss on them it will be a free ride. Control: pad chunks (LL 4, width and padding). |
| JEPA with an EMA target encoder plus terminal-reward RL | EMA code target ran (LCTUL-J): the target drifted generic in 5k steps (rank 77 to 40, cosine to the corpus mean 0.06 to 0.53) and nothing reached the reader (memory `ema-target-moved-nothing-reached-the-reader.md`; `lab/experiments/failures/2026-09-22-lctul-ema-target.md`). The collapse guard sat on the wrong tensor. A frozen encoder on a live front is not a fixed target (memory `frozen-encoder-on-live-front-is-not-a-fixed-target.md`). Latent prediction on the slot exit collapsed (NextLat; `.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md`). Only main CE buys depth (memory `only-main-ce-buys-depth.md`). | Weak as the depth driver. Allowed as a side term only if the collapse guard sits on the tensor whose EMA is the target, and it must not be the thing the loop is credited for. RL on terminal reward is untested here. |
| Flow matching over inner chunks with a duration coordinate | A flow thinker over span latents put 1 % of its loss on the past (0.3125 with the past vs 0.3161 with a null) (LL 3.6). The Euler k-curve is flat by theorem for isotropic-Gaussian fields: one step returns the conditional mean, then one scalar (`.agents/notes/proposed/architecture/2026-09-16-lctul-euler-depth-theorem.md`, Lean in `lab/theory/lctul_euler_depth/`). A predicted continuous concept decoded worse than none. A variational concept collapsed (KL 0.0185 per token). The paste's claim that the EMA dechunker tolerates l2 errors in denoised latents is untested here. | Weak. Survives only if the flow acts on discrete or committed things (seed 8's masked denoising is the discrete version) and the duration is predicted by the main CE, not a separate target. |
| Per-chunk targets met in one pass (JEPA on silent chunks, per-pass losses) | Every per-pass target we tried was met by pass 1: staged corruption, oracle trajectory, gradient-conditioned passes, critic, LoopMTP horizons, graded targets (LL 2.2; Harvest G, L12). Deep CE supervision on plain Parcae removed its depth use (+0.0376 to +0.0015). | Collides directly. Avoid per-pass targets. |
| Lorentz router and centroid dechunker | I find no measured Lorentz result about loops or chunking in the lessons doc. MORPH has a hybrid Lorentz embedding (`lorentz_fraction=0.25`, CLAUDE.md) but the lessons doc contains no reading of it that bears on this. The paste's hardware claims are unverified. | Unknown. Treat as an orthogonal axis. Do it after the loop has a job. |
| "Neuralese" zero-decode rollouts: append M continuous chunk vectors | A predicted continuous concept decoded worse than no concept; the true concept helped a lot while the predicted one hurt (LL 3.6). The deterministic head converges to the marginal. | Collides. Keep rollouts in bytes (seeds 8 and 9). |
| Outer-stage perturbation is a dead end | Not tested. Our evidence is that the outer path is where bypasses live (seeds 3, 4). | Neutral. |
| Looping the inner stage is "highest ROI" | Cost case: skipping the per-token loop skipped the part that earns (LL 3.11). The paste's cost argument (3 to 4x shorter than BPE) is arithmetic I did not check. | Partial (seed 1). |

Where the paste is right and our data agrees: the inner stage is where a loop is cheap; a silent chunk is only worth anything if a later loss reads it; and an answer-time gate that dechunks only the final chunks is a clean way to build the "draft then realize" split in seeds 8 and 9.

---

## C. Pitfalls to guard from day one

Taken from LL section 4 (measurement table), picked for a byte-level looped H-Net.

1. **Causal leaks through non-attention channels.** A GLA carry across loop iterations let positions see the future, and the "0.233 nats depth earned" was the leak (carry-off K1 4.622, K6 5.742) (`.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md`). Every H-Net piece with a scan is a candidate: the encoder and decoder SSMs, the router (it compares t with t-1), the EMA smoother, the upsampler's "repeat to the next boundary", and any chunk-level cache. Run a corrupt-future probe on every depth claim. List every op that mixes positions.
   - The causal rule matters: our boundary rule cuts after the first token of a run, because the generator cannot look ahead. The router's boundary for chunk c must not depend on bytes after the cut.
2. **Units.** Report bits per byte, never nats per chunk or per token. Different chunkings change the denominator (the paste states the same). A chunk-level K-curve is not comparable across chunk rates.
3. **Loss at init.** Check CE against ln 256 first. A CE above uniform made all contrasts nonsense (LL 4: CE 12.64 vs ln 49152 = 10.80). A loss that reads about log(V)/s at init (a multi-hot CE flattened into a single-hot kernel) is a wiring bug (Harvest G, L2). Check the label shift: a one-token shift trained copying at ppl ~1.0 (Harvest G, L3).
4. **Seeds set the error bar.** Same-code reruns differ 0.006 K1-K6 and 0.009 CE@6, against a 480-row interval of plus or minus 0.0006. Two seeds differ by 44 % of the mean K1-K6. Rounding-only changes decorrelate a pair in about 11 steps. A row bootstrap cannot see this.
5. **Frozen readers.** A frozen coda prefers no state to a wrong one (-4.64 vs +0.177 once only the reader adapts). Never grade a latent with a frozen reader (LL 3.5; `.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md`).
6. **Width-matched controls.** A trajectory-prefix K-curve (+0.034) was pad cells; the same-width control read +0.002 (LL 4). Any "extra chunks" arm needs a pad-chunk control.
7. **Depth-1-trained twin.** A K-curve ranks arms in CE@1 order. A broken depth-1 control inflates it. State the depth law: a clamped Poisson with mean equal to max still runs a third of rows at depths 1 to 5.
8. **Same-path instruments.** A bare-front probe gave slot rank 5.76; the trainer path gave 13.85, and a whole target grid was void. Instruments must call the model's own forward with its kwargs. For H-Net this means the instrument must use the same router, same chunk counts and same packing.
9. **Stream-mean probes.** Probing the HC stream mean hid the carrier constant (LL 3.1). Read per-position states, and check the share of energy in one shared vector.
10. **Hindsight oracles.** No fit that sees the target. A fitted-z "headroom" of 0.9 to 2.6 nats used the answer. Subtract a near-copy floor from any min-over-samples oracle.
11. **Unshared randomness.** Share dropout and noise masks across compared branches.
12. **bf16 readings.** Read maps in fp32. A gain hinge read 0.97 fused vs 0.884 in fp32. A reference that ran einsum in bf16 moved loss +10.9 %. Pad positions at h = 0 give an RMSNorm Jacobian near 1/eps, so mask them. `load_state_dict(strict=False)` silently dropped tensors: load through the trainer's loader.
13. **Dead branches.** A shape guard returned zeros when `S // m == 0`, so three of six blocks ran half-dead. H-Net's ragged chunks make this likely: a short sequence with zero or one chunk. Check per-block output sizes.
14. **Truncated BPTT.** A no-grad prefix left 27 to 57 % of samples with no loop gradient. Front-pad so every sample's last passes carry grad.
15. **Generation readouts.** Sample only what training supervised. Gen-PPL rewards repetition, so use a diversity guard. Greedy loops appeared in every pointer arm.
16. **Validation overlap.** Dedupe the eval against training. Probe rows were the first docs and trained on at steps 0 to 80.
17. **Non-determinism.** `index_add_` in a bag mean was non-identical in 20 of 20 repeats. Bit-reproducible mode costs 2.28x. Ragged chunk compaction in H-Net (scatter, prefix sum) is a similar risk.
18. **Systems.** Looped steps are launch-bound (31.3k launches vs 8.8k at equal FLOPs). Logged tok/s was a host rate before 2026-10-12 and read 1.5x high; use wall clock. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is mandatory on the 5090 for looped arms. Compiled flex attention recompiles on every new shape: ragged chunks make that worse (decode with eager). A byte gate proves only the path it runs. A fix in a gitignored script is not in the run.
19. **Retractions.** One home per number. Do not cite from LL section 6.
20. **Wrong tensor for a guard.** The EMA collapse guard sat on the predictor output, not the online encoder output the target pools. Put any guard on the tensor whose EMA is the target.

---

## Open questions

- Does the loop get a job at all on a chunk sequence? Seeds 8 and 9 are my best candidates, because they alone satisfy the commitment test by construction. Both are untested. The only draft design we wrote never ran.
- Which of "loop where positions are many" (seed 1, earns, costs) or "loop where positions are few" (seeds 6, 8, 9, saves) does Wolfe's goal need? Amortizing decoding favors the second. Our one equal wall-clock test favors the first for CE.
- Is the H-Net router's ratio loss itself a bypass? The ratio loss pushes the compression rate. I do not know whether chunk rate and loop share interact. The router's gradient also comes only through the STE-weighted confidence, so I would watch router entropy alongside the loop.
- Does H-Net's learned chunking solve the problem our hand rule solved (a causal unit the generator can reproduce)? We have no data on learned boundaries and loops.
- Can verification be trained without RL? Seed 10 uses CE on own samples. Whether that is enough to make acceptance rates move is open.
- What "too weak" means for the old span objective: weak in what sense? The records show targets met in one pass, no committed draft, and a decoder with a free token path inside the span. Which of these does Wolfe mean? The answer sets which seed to run first.
- Byte-level copy distances: our distance classes (33 to 256 tokens) were in tokens. In bytes they are different, and no measurement exists.
- Hyperbolic or Lorentz geometry: no lesson speaks to it. If wanted, plan it as its own experiment with its own control.

## Things I could not verify

- Every Cartesia, Latent Thought Flow and hardware number in the paste.
- That H-Net's main network reads only chunk vectors with no byte-level path other than the residual. I read the paper note, but I did not read H-Net code.
- Whether MORPH's Lorentz embedding was ever read for loop behavior. I found nothing in the lessons doc.
- The Lean statements beyond the line references the lessons doc gives (I checked that the files exist, not the theorem text).
- Whether the draft prereg was amended after 2026-10-28. It has empty Results.
