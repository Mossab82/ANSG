# Phase 1, run 6 (Colab T4, 2026-11-28): CSAM diagnostics on AraGPT2-base

Code as of commit 68d1316, *before* L_cultural and option B: CSAM + DAL, λ_align = 0, λ_schema = 0.1, 1000 steps, 77 training stories. Its purpose was to measure `csam_diagnostics` on the real backbone. This is a post-submission reimplementation, not the dissertation's system.

## CSAM diagnostics (32 held-out examples, 8 with a gold schema)

- **Free activation rate (Eq. 4.20 without teacher forcing): 0.0** at start and end. No schema ever passes τ = 0.5.
- **Key-varying part of the bias, averaged over all 32 examples:**
  - Start: 2–5 × 10⁻⁵.
  - End: 1.1–4.8 × 10⁻⁴.
  - Examples without a schema have zero bias, so among the 8 gold examples the values are about 4 times larger, at most about 2 × 10⁻³ logits.
- **Maximum |bias|:** 0.05 at start, 0.17–0.27 at end. Almost all of it is constant along the key axis, which softmax ignores.
- **Conclusion:** a key-varying bias of about 10⁻³ multiplies attention odds by about e^0.001 ≈ 1.001. On the real backbone too, CSAM cannot change attention in this configuration. This agrees with the analysis in `../2026-11-28-colab-t4-run4-5/NOTES.md`.

## Other observations

- **Held-out loss.** Held-out mean at step 1000 was 5.157 (run 4 at step 1000: 5.2005). Reserved test stories: egyptian 4.41, gulf 4.38, levantine 4.42, maghrebi 4.54, msa 3.80. This run had 77 training stories against 53 in run 4, so the runs are not directly comparable.
- **Noise characters despite the token ban.** `generation_banned_tokens` = 0: no single vocabulary item contains U+0600–U+061F, yet samples still contain ؄ ؅ ؉ ؊. AraGPT2 uses byte-level BPE, so these characters arise when two tokens join into one UTF-8 sequence. Fixed after this run: generation resamples any token whose addition creates such a character, and a test with a byte-level tokenizer covers it.
- **Samples are still not stories.** The religious storyteller openings (e.g. «وحدوا الله…»، «…بذكر النبي») lead into forum and political or religious text.
