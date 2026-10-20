# Paper note: Chain-of-Thought and Compressed Looped Transformers (memory-budget separation)

- **Title:** Chain-of-Thought and Compressed Looped Transformers: A Memory-Budget Separation (Haozhou Zhang, University of Idaho, single author)
- **arXiv or URL:** 2605.30757 (v1, 2026-05-29). Id, title and author read from the abs page and the PDF first page.
- **Local paper text:** docs/references/looping-depth/cot-vs-compressed-loops-2605-30757/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: Our span states and coarse states carry context in ONE pass ("necessary is not deep"). Span-CALM, the LXTUL x spiral panel and the particles filing all give a loop a small persistent state (slots, cells, K particles) next to a long token sequence. This paper is a clean theory of what such a loop can and cannot do, and it names the knob we vary: the number of persistent slots, not the number of passes.
- 2026-10-20: It is a theory paper with two small probes. It has no language-model result. Treat it as a design lens, not as evidence about OpenWebText loops.

## 2. What the paper shows

Read in full (main text, Section 5 figures and captions). Appendices B to F were not read line by line.

- 2026-10-20 (Defs 3.1 to 3.3): three reasoners. Compressed latent loop CLL[s,d,p,T]: only s slots persist, the n input tokens are read-only keys and values, memory M = s*d*p bits. Sequence-state loop SSL: all n positions are updated each pass (the standard looped Transformer), M = n*d*p. Chain of thought: M = Theta(l log |Sigma|) for l written tokens.
- 2026-10-20 (Lemma 4.1, Thm 4.2): a CLL runs in DSPACE(O(s d p + log n + log T)). More passes add time and a counter, not mutable memory. With s*d*p polylog(n) the loop is a polylog-space machine.
- 2026-10-20 (Prop 4.3): SSL is memory-rich, Theta(n) space. P-complete tasks (circuit value) sit inside it via Giannou 2023, Saunshi 2025, Li 2024.
- 2026-10-20 (Thm 4.4): assuming P is not contained in DSPACE(polylog n), no logspace-uniform CLL with s*d*p polylog(n) and log T polylog(n) decides a P-complete language. Polynomial-length CoT does. The result is asymptotic, conditional, and says nothing about SSLs or loops with scratch positions.
- 2026-10-20 (Section 5.1, Figure 1, Tables 1 and 2): pointer chasing on a functional graph with n = 16 cells, k concurrent chains, s in {1,2,4,8,16} slots, T = 16 passes, 10 seeds. Below the diagonal (s < k) mean exact-match accuracy is 0.041, maximum cell mean 0.129, std 0.00 to 0.02 for k in {2,4,8}. The SSL control reaches 0.999 for k = 2, 4, 8 and 0.936 at k = 1 (9 of 10 seeds near 1.00, one failed to train). Above the diagonal the picture is bimodal: k=1, s=1 splits 6 high seeds and 4 low; the s = 16 column is unstable (k=8, s=16 runs from near failure to near perfect). The authors read this as a routing-trainability problem, not a capacity limit. They state s is a register count here, not a bit budget (n = 16 makes log n small against d*p).
- 2026-10-20 (Section 5.2, Figure 2): associative recall with N key-value pairs, gated linear attention, state dimension m, 5 seeds. Accuracy rises monotonically with m, and the critical m (first to reach 0.9) moves right with N: N=8 median m_crit 8 (range 4 to 16), N=16 median 32 (16 to 64), N=24 median 64 (32 to 128). Softmax attention is at or above 0.999 everywhere. Vanilla linear attention stays low for N = 16, 24 at any m.
- 2026-10-20 (Section 6): the stated design rule is to specify the persistent state with the recurrence. KV-cache sharing across loop depth decouples memory from pass count but does not shrink state relative to context. The authors name the intermediate budgets between s*d*p and n*d*p, where practical looped LMs sit, as the open target. They also say success above the memory threshold is not automatic.

Not verified: Appendix proofs B to E; the Appendix F per-seed tables; the Sapunov 2026 Sudoku memory-token result (cited, not read).

## 3. Our speculation

- 2026-10-20 Speculation: the lens maps onto our arms by persistent state per pass. Plain Parcae loop: SSL (state at every token, n*d). Span-only Parcae over ~21-token spans: the span set is the state, so the state is smaller than the token sequence, but the span reads tokens only through the decoder or encoder. LXTUL slot loop with 4 cells: closest to a CLL (s = cells x slots). Particles: K copies of one state, so persistent bits grow by K but carry the same content unless identity differs.
- 2026-10-20 Speculation: this explains one part of "necessary is not deep". If a span state holds all the context a task needs in one pass, extra passes add time, not memory. Depth can then only pay on tasks whose working state exceeds what the slots hold. Web text may rarely demand that, which fits our flat K1-K6 on slot arms and the +0.0054 on the fan.
- 2026-10-20 Speculation: the s < k collapse shows a count of independent items per read, not a depth. Our fan (4 cells, set-reader decoder) has s = 4. A text target that needs more than 4 independent facts held at once is the analogue of k > s.
- 2026-10-20 Speculation: the bimodal above-threshold seeds (6 of 10 at k=1, s=1) match our "n=1 run comparisons are unreadable" memory. Any slot-count sweep needs several seeds per cell.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper claim: more passes do not add mutable memory in a compressed loop. Our measurement: the fan with 4 cells earns K1-K6 +0.0054 and the cell-count run holds CE at 2 cells (+0.0002) and 1 cell (+0.005, `lab/experiments/successes/2026-10-13-fan-variable-cell-count-v1.md`). This agrees in direction: cells are cheap to drop, and no arm gained above 4. It is not a test of the theorem, which needs a task with a known state demand.
- 2026-10-20: Our filing that particles with a persistent identity are used but add no depth fits the theory only loosely. K particles raise the bit count K-fold, but if all K hold the same content the effective memory does not rise. The paper offers no test of this.
