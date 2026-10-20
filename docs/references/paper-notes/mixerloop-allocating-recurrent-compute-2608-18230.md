# Paper note: Allocating Recurrent Compute in Looped Language Models (MixerLoop)

- **Title:** Allocating Recurrent Compute in Looped Language Models (Ruhai Lin, Yiyang Guo, Rui-Jie Zhu, Hao Ye, Jason K. Eshraghian; UC Santa Cruz)
- **arXiv or URL:** 2608.18230 (v1, 2026-08-18)
- **Local paper text:** docs/references/looping-depth/mixerloop-allocating-recurrent-compute-2608-18230/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It asks which part of a looped block earns the repeat. Our core repeats the whole 6-layer block (attention and FFN) and training speed (ms/step) is a top priority. If the FFN need not repeat, the loop gets cheaper. It also brings a readout-level instrument for "does pass r add something new", which is close to our K1-K6.

## 2. What the paper shows

Read in full (PDF text, main text and tables; appendix not present in the extract).

- 2026-10-20 (Sec 3.1, Table 1): MixerLoop repeats only the Gated DeltaNet token mixer T=4 times per physical layer and runs the dense FFN once. FullLoop (the LT2 style) repeats mixer and FFN. NoLoop runs both once. Same unique parameters (15.7M and 116.9M), same data, 52.43B processed tokens, seq 1024, 100k updates, batch 512, one checkpoint and one seed per arm.
- 2026-10-20 (Sec 3.3): FFNs are 61.2 to 61.3 % of per-layer projection FLOPs, so MixerLoop uses 54.1 % of FullLoop's recurrent-backbone projection FLOPs at T=4. End to end the saving is 33.9 % at 15M and 43.0 % at 110M, because the LM head runs once.
- 2026-10-20 (Table 2): CORE (22 tasks, chance-adjusted x100). 15M: NoLoop 5.01, FullLoop 5.52, MixerLoop 6.52. 110M: NoLoop 14.16, FullLoop 17.52, MixerLoop 15.56 (41.5 % of FullLoop's gain). Held-out NLL 15M: 2.995 / 2.936 / 2.946 (NoLoop / FullLoop / MixerLoop); 110M: 2.401 / 2.342 / 2.377. So FullLoop has the better NLL at both scales; MixerLoop wins CORE only at 15M.
- 2026-10-20 (Sec 4.3): prefill at length 1024 on one RTX 3090 Ti: 1.12x (15M) and 1.11x (110M) faster at batch 1; 1.34x (15M, batch 32) and 1.52x (110M, batch 16) throughput. Inference only. No training step time is reported.
- 2026-10-20 (Sec 3.4, 4.4, 4.5): Iterative Transport Rank (ITR) and marginal ITR. They restore a layer's contextual mixer passes one by one against a context-off mixer and read the change in sqrt(p) at the output. 15M: cumulative ITR 1.301 (MixerLoop) vs 1.305 (FullLoop); marginal ITR of passes 2 to 4 is 0.988, 0.871, 0.777; passes 2 to 4 add +0.321 nats of target log-prob (FullLoop +0.210). 110M: ITR 1.177 vs 1.293; later passes add +0.070 nats (FullLoop +0.056). A random-init 15M model reads ITR 1.068 and about -1.5e-4 nats. At 110M only 19 of 36 later layer-pass pairs of MixerLoop have marginal ITR above 0.5, and 30 of 36 add positive target log-prob (FullLoop 27 of 36).
- Failure modes: at 110M FullLoop is stronger on aggregate CORE and on Dyck, Repeat Copy, SQuAD, CoQA. Novelty "becomes more selective as the model scales". Single seed, no spread on any number.

Not verified: the context-off implementation beyond the text; any appendix; code (none cited in the extract).

## 3. Our speculation

- 2026-10-20 Speculation: this is a speed lever for our core only if our loop gain lives in the attention path. Our injection is a diagonal contractive map; the blocks are "quiet" (0.87 floor note). A test is cheap: loop only the attention sublayers of the 6-layer core, FFN once per pass-block, and read K1-K6 and ms/step against the full-block loop at matched steps.
- 2026-10-20 Speculation: the FFN is the part of our block that MORTAR and ternary already tax. Skipping its repeat could stack with the ternary core's weak per-pass map, or could remove the only nonlinearity that makes a pass differ from the last one. Either outcome is informative.
- 2026-10-20 Speculation: the ITR readout (restore pass r, measure the output change) is a per-pass, main-loss-free instrument. It fits our rule that only the main loss buys depth, because it measures a trained model and adds no loss.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: The paper's linear-attention (GDN) mixer carries a recurrent state across positions, so a repeat re-writes memory. Our attention is CCA+CSA+HCA over a read-only sequence at each pass. The mixer-only result may not transfer. Also their 110M FullLoop has a better NLL than MixerLoop, so "free" FLOP savings are not free in CE.
