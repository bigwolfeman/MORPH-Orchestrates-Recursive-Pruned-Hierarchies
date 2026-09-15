# P-E judgment: topic thread in the sampled continuations at 20k (by hand, 2026-09-15)

Rule (prereg 2026-09-14-arc-tul-code-20k, P-E): a sampled continuation keeps a topic
thread when it names an entity or subject from the context. Judged on the ten cuts of
`code_span_samples_*_20000.txt`, SAMPLE1 and SAMPLE2 both read, a cut counted when either
sample keeps the thread. "weak" = same domain, no named entity; counted as NO.

| cut | context | tul-code-20k | tul-code-rollout1-20k | strict ruler twin |
| --- | --- | --- | --- | --- |
| 1 | CNN / hospital / doctors | no (police) | no (police) | YES (hospital, health services) |
| 2 | doctors and nurses gone | no | no | no |
| 3 | Sanders campaign, delegates | no (Obama not in context) | no | YES (campaign) |
| 4 | trade deals, her husband, Michigan | no | no | YES (Clinton, Obama, primary) |
| 5 | Obama, consistency | no | YES (S2: "But Obama is the only one") | YES (Obama) |
| 6 | Obama or Hillary, guns | no | no | weak (Senate GOP policies) |
| 7 | BIGBANG concert, song, fireworks | no | weak (S1: "the sound") | no |
| 8 | BIGBANG last world tour | no | no (Clinton) | no |
| 9 | hate speech, debate, Clements | no | no | no |
| 10 | Seattle businesses, protest | no | no (NFL) | weak (cities) |
| **count** | | **0 / 10** | **1 / 10** | **4 / 10** |

P-E (rollout1 ≥ 3 of 10 AND tul-code ≥ 1 of 10): FAILS on both halves.
