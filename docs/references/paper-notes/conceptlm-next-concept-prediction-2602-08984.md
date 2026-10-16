# Paper note: Next Concept Prediction in Discrete Latent Space Leads to Stronger Language Models (ConceptLM)

- **Title:** Next Concept Prediction in Discrete Latent Space Leads to Stronger Language Models (ConceptLM), LUMIA Lab, Shanghai Jiao Tong University and Shanghai AI Laboratory
- **arXiv or URL:** arXiv 2602.08984 (v1, submitted 2026-02-09). Code: https://github.com/LUMIA-Group/ConceptLM
- **Local paper text:** docs/references/looping-depth/conceptlm-next-concept-prediction/2602.08984.txt
- **Date first noted:** 2026-10-16

## 1. Why we reference it

- 2026-10-16: An external model's critique of our coarse-to-fine work leaned on this paper. We checked its claims against the text. The results are in section 4.
- 2026-10-16: It is a span-concept LM, like NCM and SpiralFormer. It pools token states into concepts with factor k (k=4 in the main runs). It then predicts the next concept over a VQ codebook.
- 2026-10-16: Table 18 (chunk size k) and Table 15 (insert depth) bear on two of our readings. One is "finer chunks use more depth". The other is "coarse compute is starved". The ablations use Pythia-160M on 30B Pile tokens. The paper does not say how many seeds each row uses.
- 2026-10-16: Overlap with LXTUL is low. ConceptLM has no slot loop and no core loop. Its concept module runs once per forward. Its codebook (SimVQ, a two-layer ReLU MLP on the codes) is the closest match to our codebook work.

## 2. What the paper shows

Read in full on 2026-10-16. Section, table, and equation numbers are given. Numbers are copied from the printed tables. Our arithmetic checks are marked.

- 2026-10-16 (Sec 2.2): Concepts are mean pools of adjacent token states. The concept sequence has length T/k. The text says "aggregating adjacent token-level ones with a mean-pooling". It does not use the words "non-overlapping" or "fixed window". T/k implies both.
- 2026-10-16 (Sec 2.2, Sec 3): Product quantization. Each concept splits into S segments. Each segment has its own codebook of N entries. The main runs use N=64 and S = number of heads (Sec 4.1).
- 2026-10-16 (Sec 2.2): The predicted concept is a weighted sum over codebook entries, not a hard sample. The text says "Rather than performing hard sampling ... the predicted segment representation is derived as a weighted sum". The weights come from a linear head. The paper calls them "a logits distribution". It does not write a softmax.
- 2026-10-16 (Sec 3, L_NCP): The loss is MSE between the predicted concept and the continuous concept state h_c. It is averaged over M = T/k steps. The paper writes no stop-gradient on the target. The text says "The context prediction gradients propagate back to the Encoder". Whether the target branch is detached is NOT FOUND.
- 2026-10-16 (Sec 3, L_VQ): The text says L_VQ is "strictly confined to the Quantizer module". The printed loss has stop-gradient on one term of each side. It still has a beta term that sends gradient to c, the encoder output, unless the code detaches c before the quantizer. The paper does not say which.
- 2026-10-16 (Sec 2.3, equation for h̃_t): The concept is broadcast k times and added to the token states. The first k-1 tokens get zero. The index is floor(t/(k-1)). The divisor is k-1, not k. The paper does not explain this. It means each concept index covers k-1 tokens, not k.
- 2026-10-16 (Sec 3, total loss): L = L_NTP + L_NCP + L_VQ. The terms are unweighted.
- 2026-10-16 (Sec 4.3, Table 3): Llama-3.1-8B, 9.6B continual tokens. Trained-8B avg 47.5. PM-8B avg 47.2. Ours-8B avg 47.6. The text reads the gain as +0.4 over PM and +0.1 over Trained-8B. The intro (lines 101-104) says "0.4" against "its token-level counterpart". The 0.4 is against PM-8B.
- 2026-10-16 (Sec 5.5, Table 6): Eight H200 GPUs. Llama-PM-8B training 80.0 h and ConceptLM-8B 71.0 h. That is 11.3% lower, and our arithmetic gives 11.25%. Inference on SQuAD-v2: 99.0 min and 86.0 min. That is 13.1% lower (our arithmetic: 13.13%). The paper gives the extra gain as IO and KV-cache savings. It does not measure those separately.
- 2026-10-16 (App C): The FLOP saving is about 4.69% at T=8192, d=4096. Our recompute from the printed totals gives 4.688%. The printed intermediate line "(6Td^2 + T^2 d)/4" does not reduce to the printed total "390Td^2 + 64.25T^2 d". The printed total gives 4.69%.
- 2026-10-16 (Sec 5.2, Table 4; Sec 5 default is Pythia-160M, 30B tokens): Avg PPL. LMTP 81.16. LNTP 69.28. LNTP+LVQ 75.64. LNTP+LNCP 76.30. LNTP+LNCP+LVQ 68.13. Avg Acc: LMTP 38.4, LNTP 39.6, LNTP+LNCP 39.5, full 40.3. The "LNTP" row is the ConceptLM architecture trained with the NTP loss only. It is not a plain token model.
- 2026-10-16 (Sec 5.4, Table 5): Avg PPL (printed): +2 token layers 59.80, +2 concept layers 68.13, +6 token 65.25, +6 concept 54.34, +10 token 75.20, +10 concept 58.21. Our arithmetic: the four printed components of +6 concept average 54.46, not 54.34. The four components of +2 token average 71.35, not 59.80. The printed "+3.22", "+7.91", "+16.99" annotations do not match the PPL column. Avg Acc: +2 concept 40.3, +6 concept 40.9. The paper's depth-collapse claim rests on this table.
- 2026-10-16 (App E.2, Table 15; Pythia-160M, 30B tokens): Avg PPL by insert layer. L0 63.25 (our arithmetic on the printed components: 62.50). L1 68.13. L3 62.09. L5 60.64. L7 64.34. L9 55.04. L11 59.79. Avg Acc: L0 40.5, L1 40.3, L3 39.6, L5 40.3, L7 40.3, L9 40.4, L11 40.3. The paper calls L9 "superior generalization". The accuracy gain over L1 is 0.1. The gain is in PPL. The table does not state the layer count of Pythia-160M.
- 2026-10-16 (App E.2, Sec 5.2 default): Sec B.2 says layers 0 and 1 are suboptimal and three-quarters depth is best. The main runs still use layer 1 (Sec 4.1, Table 9).
- 2026-10-16 (App E.4, Table 16, N ablation): Avg PPL by N. N=8 66.88, N=16 63.03, N=32 69.20, N=64 68.13, N=128 68.06, N=256 70.38. Avg Acc 40.0 to 40.4. The paper says N changes PPL much more than accuracy.
- 2026-10-16 (App E.4, Table 17, S ablation): Avg PPL. head/2 62.09, head*2 62.03, head*1 68.13, head*4 72.46. Acc 40.3 to 40.7 across most rows.
- 2026-10-16 (App E.5, Table 18, chunk size; Pythia-160M, 30B tokens): Chunk-2 Acc 40.7 / Avg PPL 59.80 / Lambada-S 158.19. Chunk-4 40.3 / 68.13 / 190.06. Chunk-8 40.2 / 65.81 / 179.41. Chunk-16 39.8 / 69.70 / 193.26. Our arithmetic matches each Avg PPL to rounding. Avg PPL is not monotone in k. Avg Acc falls as k grows.
- 2026-10-16 (Sec 4.2, Table 1; GPT-2, OWT trained): Avg PPL. 124M: base 101.94, PM 99.14, ConceptLM 84.75. 355M: 52.57, 52.55, 40.51. 774M: 36.35, 32.89, 29.61. 1.5B: 29.68, 30.01, 26.38.
- 2026-10-16 (Sec 4.2, Table 2; GPT-2 downstream Avg Acc): 124M base 38.0, PM 38.4, ConceptLM 39.1. 355M 40.6 / 41.4 / 42.3. 774M 42.8 / 42.8 / 44.0. 1.5B 43.9 / 43.5 / 45.1. Against PM, the gains are +0.7, +0.9, +1.2, +1.6.
- 2026-10-16 (Sec 4.2, Table 1 and Table 2; Pythia, Pile): Avg PPL. 70M: base 297.72, ContextLM 142.71, ConceptLM 137.70. 160M: 67.87, 43.09, 34.23. 410M: 17.84, 15.82, 15.31. Avg Acc. 70M: 34.6, 38.2, 38.2. 160M: 40.3, 42.0, 42.9. 410M: 47.5, 48.2, 48.3. The baseline and ContextLM rows are "sourced from Dai et al. [2025]" (Sec 4.2). On Pile PPL, ConceptLM is slightly worse than ContextLM at 70M (15.00 vs 14.96) and 410M (8.70 vs 8.67).
- 2026-10-16 (Sec 5.3, Figure 4): The paper says the gain over the PM baseline grows from training length 512 to 2048. The values come from a plot with no table. We have not checked them.
- 2026-10-16 (Sec 2.2 and App B.2, Table 9): GPT-2 main runs insert before the first layer (Sec 4.1, B.2). Pythia and Llama main runs insert after layer 1 (Sec 4.1, Table 9 says "before the second layer"). Layer-1 GPT-2 results are in Table 14.
- 2026-10-16 (Sec 4.2, GPT-2 baseline): Parameter-matched PM model. Two extra token layers "to match the parameter count of ConceptLM". The paper also says "we do not match these parameters". The codebooks have under 2M parameters at XL. So PM is close to, not exactly, parameter-matched.
- 2026-10-16 (Sec 4.3, Table 3 caption): PM-8B is Llama with two extra token layers (34 token layers). ConceptLM-8B is 32 token layers plus 2 concept layers. Parameter matching is approximate and not computed.
- 2026-10-16 (Sec 4.2, line 411-413): The paper says ConceptLM "requir[es] only 81% used by the standard models". The unit is not defined. NOT FOUND in the text.
- 2026-10-16 (whole paper): No seed counts or error bars appear in any table. NOT FOUND.
- 2026-10-16 (Sec 4.2, 5.1): No FLOP-matched baseline. No parameter-matched baseline for Pythia (only ContextLM, taken from Dai et al. 2025).

Answer to "same parameters or FLOPs as the NTP baseline?": Parameters, approximately, for GPT-2 (PM) and Llama (PM-8B). Not for Pythia. FLOPs, no. For Llama, ConceptLM uses about 4.69% fewer FLOPs than PM-8B by the paper's own count.

Headline result: On GPT-2 (Table 1, Table 2), ConceptLM has the lowest Avg PPL at all four sizes and the best Avg Acc against PM at all four sizes. On Pythia (Table 1, Table 2), ConceptLM has the lowest Avg PPL against the base and ContextLM at all three sizes. On Llama-3.1-8B (Table 3), the gain is +0.4 over PM-8B and +0.1 over Trained-8B. The abstract's "consistent performance gains" needs the 8B numbers in that context.

## 3. Our speculation

- 2026-10-16 Speculation: The paper's own reason (App B.2) for depth effects is that concept loss reaches the encoder. A deeper encoder then gives better concepts. Table 15 fits that story. Our coarse-to-fine arms could test the same thing at matched compute.
- 2026-10-16 Speculation: The index floor(t/(k-1)) may be a typo for floor(t/k), or it may be a deliberate shift. The released code will settle it. Check the code before we copy the formula.
- 2026-10-16 Speculation: Table 18 favors the finest chunk (k=2) on PPL here. That is one run per row on Pythia-160M. It does not show that finer chunks use more depth.
- 2026-10-16 Speculation: The paper does not measure starvation of coarse levels. It only shows that concept layers cost fewer FLOPs. So the paper gives no support for "coarse compute is starved" either way.

## 4. Corrections

- 2026-10-16: The claim list said C11: "The main experiments insert the concept module after layer 1." That is FALSE for GPT-2. The paper says GPT-2 main runs insert "before the first layer" (Sec 4.1; Sec B.2 repeats this). The paper's default in B.2 is "after the first Token Layer", but the GPT-2 comparison uses before the first layer. Table 9 says "before the first/second layer" for GPT-2. The claim is TRUE for Pythia and Llama only ("after the first layer", Sec 4.1; "before the second layer", Table 9). Layer-1 GPT-2 results are in Table 14 only.
- 2026-10-16: The external model said Table 18 shows "monotonic degradation with chunk size". Avg PPL is not monotone: Chunk-8 65.81 is better than Chunk-4 68.13 (Table 18). Avg Acc does fall as k grows (40.7, 40.3, 40.2, 39.8). Chunk-2 is best on both.

## 5. Where our findings disagree

- None recorded yet.
