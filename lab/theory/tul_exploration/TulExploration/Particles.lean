/-
Copyright (c) 2026 MORPH contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: MORPH theory agent (ParticleProver)
-/
import TulExploration.Particles.Collapse
import TulExploration.Particles.IWAE
import TulExploration.Particles.SetRead
import TulExploration.Particles.Credit
import TulExploration.Particles.Depth

/-!
# Coarse-level particles: K copies of the coarse loop from independent random starts

The design note's claims C1 to C6, split by file.

* `Particles/Collapse.lean` : P1 (a contractive pass collapses the particles; the softmax
  `log K` duplicate effect) and P2 (several basins need a non-contractive pass).
* `Particles/IWAE.lean` : P3 (the mixture read of exchangeable particles is monotone in `K`;
  role-indexed cells are not).
* `Particles/SetRead.lean` : P4 (the `log K` ceiling; the sequential Bayes read equals the
  span mixture; a switching reader and a value-mixing reader break the "between mixture and
  oracle" picture).
* `Particles/Credit.lean` : P5 (soft credit vs winner-take-all).
* `Particles/Depth.lean` : P6 (what depth buys in a multi-basin pass).
-/
