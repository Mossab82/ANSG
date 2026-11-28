# Phase 1, runs 6b and 7 (Colab T4, 2026-11-28): CSAM with options A + B, and ablation

The folder names `phase1_v6` / `phase1_v7_no_csam` come from the Colab command. This `phase1_v6` is not the diagnostic run in `../2026-11-28-colab-t4-run6-diagnostic/`.

## Settings (both runs)

Code at commit 753b12a, which predates the noise-character fix:
- 1500 steps, seed 42, 77 training stories.
- λ_align = 0.1 and λ_schema = 0.05 (§4.3.5).
- L_cultural on rule-based motif spans.
- Option B: `--csam-init-std 0.1 --csam-lr-mult 10`.
- Run 7 adds `--no-csam`. The batches are identical (separate RNG for motif pairs).

This is a post-submission reimplementation, not the dissertation's system.

## Results

- **CSAM still has no measurable effect.** CSAM+DAL and DAL-only differ by at most 0.004 nats on every held-out set.
  - Reserved test stories, CSAM+DAL vs DAL-only: egyptian 4.389 vs 4.388, gulf 4.344 vs 4.340, levantine 4.404 vs 4.404, maghrebi 4.447 vs 4.447, msa 3.783 vs 3.784.
  - Held-out mean at 500, 1000 and 1500 steps: 5.1791, 5.1328, 5.1284 vs 5.1786, 5.1326, 5.1280.
- **L_cultural stays at chance.**
  - It fell from 1.3995 (step 100) to 1.3758 (step 1300).
  - With H ≈ 0 for every pair, Eq. 4.18 gives 2·ln 2 ≈ 1.386. The bias matrices therefore barely separate in-span from out-of-span pairs.
- **The key-varying bias grew only to about 1 × 10⁻³** (average over 32 examples, about 4 × 10⁻³ on gold-schema examples), despite the 5× initial scale and the 10× learning rate. The maximum |bias| reached about 0.3, but it is almost all constant along the key axis.
- **No overfitting within 1500 steps.** The best checkpoint was the last.
- **Samples are still not stories.** The MSA sample now shows dialogue turns in «».

## Interpretation

These runs are consistent with the structural limit noted in D6 (`docs/THESIS_ALIGNED_ARCHITECTURE.md`):
- Eq. 4.13 makes Rel a function of positional encodings and the schema only.
- Motif spans fall at different positions in different texts, so a position-only H cannot tell an in-span pair from a random pair. The exception is the opening formula, which is always at the start.
- L_cultural therefore cannot fall much below chance.

This is evidence about this implementation. It shows nothing about the dissertation's original system or its reported results.

## Next (decision pending)

**Option C: content-dependent relevance.** Replace pos_j, pos_k in Eq. 4.13 with the token representations at positions j and k. This is a declared deviation from the equation as written. The same A + B settings and the same ablation would test it.
