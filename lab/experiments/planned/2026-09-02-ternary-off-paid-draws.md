# Planned: ternary-off paid draws — is QAT load-bearing in the detonation?

Status: planned
Date: 2026-09-02 (frozen ~14:30, before any ternary-off run; Wolfe's
directive, paired with the checkpoint autopsy and the AdEMAMix history dig)

## Question

Does removing ternary QAT from the backbone change the paid-axis detonation
rate? Base rate: 6 of 8 unguarded paid draws detonated (R1 1/2, A2 1/2,
A2s 0/2, A2-20k 0/2), all with the beta1=0 explosion signature; the killed
20k retry showed sigma_max 2.44 on a core MLP at step 500 mid-explosion.

## Hypothesis (Wolfe's lean, shared): AdEMAMix is the root

The detonation is the known beta1=0 optimizer instability crossing the
loop's rho=1 manifold; ternary QAT is a constant across detonating and
clean regimes and is NOT load-bearing — its threshold staircase at most
modulates the rate. Counter-hypothesis: threshold-crossing channel flips
under raw beta1=0 steps are the discrete kicks that push sigma over 1, and
removing them collapses the rate.

## Method

Three sequential draws: config `tul_a2` + `model.ternary=false`, panel
flags, 2500 steps each (the death window closes by ~2040; every observed
detonation is unambiguous in the grad probe by step 1000). wandb
tul-a2-nt1/nt2/nt3. Verdict per draw from the probe, not survival alone:
DETONATED iff max(preclip/total over steps 200..2500) > 1e4; HEALTHY iff
< 1e3 throughout (between: judged manually and reported). Smoke condition
folded into draw 1: the "Ternary QAT ON" print must be ABSENT, embed QAT
may remain (int6 embeds stay on — one variable at a time).

Confound, named: ternary-off changes the effective model everywhere (48%
of params), so a rate change is evidence about the INTERACTION, not proof
of mechanism; the autopsy's threshold-churn measurement is the
mechanism-level check.

## Predictions (frozen)

- **P-N1.** Ternary-off detonation count over 3 draws <= 1 (rate collapses
  from ~0.71): 40%.
- **P-N2.** At least 2 of 3 draws detonate (ternary not load-bearing;
  AdEMAMix root stands): 45%. (Remaining 15%: exactly ambiguous 1-2 split
  patterns judged against the probe magnitudes.)

## Binding

P-N2-side outcome => quantization exonerated; the fix hunt goes to the
optimizer/guard (A2c cap, alpha_cap/t_beta3 retune per the history dig).
P-N1-side outcome => the ternary x beta1=0 x loop interaction is real;
the next arm tests ternary-on with a smoothed optimizer (beta1>0 or capped
alpha) before touching the deploy quant stack.
