# Phase 1, third run (Colab T4, 2026-11-28, commit b24e094)

**Data.** 28 training stories ×4, plus IADD for all four varieties after the country-name fix: 716–1277 packed examples per variety. Held-out: 224 examples, including 5 whole parallel stories.

## Held-out LM loss, backbone only → with CSAM + DAL

| | egyptian | gulf | levantine | maghrebi | msa |
|---|---|---|---|---|---|
| all held-out | 5.78 → 5.30 | 6.28 → 5.70 | 6.11 → 5.59 | 6.34 → 5.58 | 5.08 → 4.58 |
| held-out parallel stories | 6.14 → 4.97 | 6.09 → 4.92 | 6.23 → 5.03 | 6.48 → 5.26 | 5.14 → 4.13 |
| IADD | 5.72 → 5.35 | 6.30 → 5.79 | 6.10 → 5.64 | 6.33 → 5.62 | — |

**Held-out mean by step:** 5.430 (500), 5.374 (1000), 5.358 (1500), 5.352 (2000), 5.350 (2500), 5.349 (3000). The curve had flattened by about step 1500, so longer training alone is unlikely to help much.

## Samples
**What improved:** no repetition loops, and the varieties are clearly distinct:
- Egyptian: معندوش، كدة، عايز، ايه
- Levantine: هلأ، شو، هيك، منيح، مشان، هيدا (with some Gulf forms near the end)
- Maghrebi: ديال، غادي، بزاف، شنو، خاصك، راني باغية

**What did not:** the content is still social-media chatter rather than a story. The MSA sample drifts into a religious-history register. The Gulf sample stopped right after the opening, because the end-of-text token was sampled immediately.

**Reading.** The adapters learned dialect vocabulary, which comes mostly from IADD social-media text, but not narrative. The stories are a small share of the dialect data.

## Changes for run 4
- Source-balanced sampling, `--mix parallel=0.4,iadd=0.4,wikisource=0.2` (the default).
- `--min-new-tokens 120` (the end-of-text token is blocked before that).
- The reserved test stories are reported as `parallel_test/*`.
- **Planned run 5:** the same settings with `--no-csam`, an ablation that separates the contribution of CSAM from that of DAL.
