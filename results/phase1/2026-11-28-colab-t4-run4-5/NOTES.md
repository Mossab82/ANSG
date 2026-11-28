# Phase 1, runs 4 and 5 (Colab T4, 2026-11-28): CSAM ablation

Both runs used the same code, data (62 parallel stories, of which 9 are the reserved test split), seed 42, 3000 steps, mix parallel=0.4 / iadd=0.4 / wikisource=0.2, and decoding settings.

- **Run 4 (`phase1_v4`):** CSAM + DAL.
- **Run 5 (`phase1_v5_no_csam`):** DAL only (`--no-csam`).

This is a post-submission reimplementation, not the dissertation's original system. These numbers say nothing about the original system's results.

## Held-out loss (lower is better)

| Set | n | Backbone | CSAM+DAL | DAL only |
|---|---|---|---|---|
| parallel_test/egyptian | 9 | 5.797 | 4.454 | 4.454 |
| parallel_test/gulf | 9 | 5.750 | 4.605 | 4.609 |
| parallel_test/levantine | 9 | 5.821 | 4.574 | 4.584 |
| parallel_test/maghrebi | 9 | 5.829 | 4.555 | 4.557 |
| parallel_test/msa | 9 | 5.223 | 3.946 | 3.938 |
| iadd/egyptian | 38 | 5.663 | 5.344 | 5.344 |
| iadd/gulf | 42 | 6.196 | 5.800 | 5.800 |
| iadd/levantine | 58 | 6.120 | 5.846 | 5.846 |
| iadd/maghrebi | 38 | 6.144 | 5.779 | 5.779 |
| wikisource/msa | 25 | 5.027 | 4.599 | 4.599 |

Held-out mean by step:

| Run | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| Run 4 | 5.2132 | 5.2005 | 5.2666 | 5.3261 | 5.3739 | 5.3849 |
| Run 5 | 5.2139 | 5.2005 | 5.2665 | 5.3259 | 5.3720 | 5.3835 |

## Findings

1. **DAL accounts for all of the improvement.**
   - Loss on the 9 reserved test stories drops by 1.15–1.35 nats in every variety, against the frozen backbone.
   - IADD and Wikisource improve by 0.3–0.4 nats.
2. **CSAM has no measurable effect in this implementation.**
   - Runs 4 and 5 differ by at most 0.01 nats on every set.
   - The two training curves are nearly identical. Run 4 is worse by 0.01 on levantine test and better by 0.008 on msa test.
   - Single runs, one seed: differences of this size are noise.
3. **Overfitting after step 1000.**
   - Training LM loss keeps falling (about 3.7 at step 2500), while held-out loss rises from 5.20 to 5.38.
   - The best checkpoint (step 1000) was restored, so the reported losses are the step-1000 adapters.
4. **Samples are still not stories.**
   - The register is forum or religious text, with drift and noise characters (U+0605, U+060A, U+0613).
   - One sample per variety and sampling noise make the SAS values of the two runs incomparable. Small logit differences change every sampled token after the first.

## Why CSAM is inert (analysis, not yet measured on the real backbone)

- **Eq. 4.13 depends only on positions and the schema.** The dissertation defines it as Rel = σ(vᵀ[pos_j; pos_k; c_s]), with pos_j and pos_k as positional encodings (§4.3.4). The j-term and the c_s-term are constant along the key axis, and softmax ignores any bias that is constant along that axis. Only the k-term can change attention.
- **The k-term is tiny at initialisation, and the LM loss alone barely trains it.**
  - The k-term is v_pos_k · pos_k with v ~ N(0, 0.02²), passed through σ at about −2, where σ′ ≈ 0.1, and scaled by β = 0.347.
  - The dissertation trains these parameters with the cultural alignment loss L_cultural (§4.3.5). Here L_cultural is disabled (λ_align = 0) because the corpus has no motif-span annotations.
- **Eq. 4.20 activation essentially never fires without teacher forcing.** With a random W_proj, cos(context, c_s) rarely exceeds τ = 0.5. CSAM is then only active on the parallel stories, which carry a gold value label.
- **Smoke test** (tiny random backbone, new `csam_diagnostics`): free activation rate 0.0; the key-varying part of the bias is about 0.001–0.0016 logits per layer, against a maximum |bias| of about 0.12. The next Colab run will report the same diagnostics on AraGPT2-base.

## Changes after this run

- `scripts/train_phase1.py` reports `csam_diagnostics` at the start and the end of training: free activation rate, and the key-varying bias per layer for heritage and rhetorical heads.
- Generation bans vocabulary items containing Arabic sign or format characters U+0600–U+061F, except ، ؛ ؟.

## Options to make CSAM testable (decision pending)

- **A (closest to the dissertation): enable L_cultural (Eq. 4.18).** This needs motif-span annotations: which token pairs should attend to each other, such as a proverb and the event it comments on. They could be added to the parallel stories by a generator and spot-checked.
- **B (engineering): larger initial scale for v_pos_k, or a separate learning rate for CSAM.** This keeps Eq. 4.13 unchanged.
- **C (deviation, would need to be declared): make relevance depend on token content** (hidden states instead of positional encodings). This changes Eq. 4.13.
