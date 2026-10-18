# Paper note: Object-Centric Learning with Slot Attention

- **Title:** Object-Centric Learning with Slot Attention (Locatello et al., NeurIPS 2020)
- **arXiv or URL:** 2006.15055 (from memory; not checked against the export API yet)
- **Local paper text:** none archived
- **Date first noted:** 2026-10-18

## 1. Why we reference it

- 2026-10-18: The competitive pool of the coarse-particles arms (Parcae branch lxtul/particles, prereg
  `2026-10-18-parcae-coarse-particles-5k.md`) copies its central rule. K slots start from random draws, share one update
  map, and are exchangeable; the attention softmax is taken over the SLOT axis, so slots compete for each input and split
  the evidence. Our particles are K random-start copies of the chunks-16 loop; the competitive arm normalizes the group's
  token weights over particles, so each particle gets its own injection. The Lean particle work (MORPH 1b352c0f) says a
  shared injection cannot hold random-start particles apart, which is why we need a competition term in the forward pass.

## 2. What the paper shows

Seed note from memory. Section numbers not recorded; add them on the first full read.

- 2026-10-18: Mechanism as we use it: per iteration, softmax over slots of q_slot · k_input, then a weighted mean over
  inputs per slot (the weights renormalized over inputs), then a GRU and MLP update of each slot. Slots are drawn from a
  learned Gaussian at init and are permutation-equivariant.

## 3. Our speculation

- 2026-10-18 Speculation: competition gives depth a choice job. In the 1-D Lean model (`monotone_side_preserved`) an
  uncoupled monotone pass never moves a particle to another basin; coupling through the softmax over particles can.
- 2026-10-18 Speculation: the sharpness of the competition (entropy of the weights over particles divided by log K) plays
  the role of the Hopfield inverse temperature; we log it in every compete arm.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None recorded yet.
