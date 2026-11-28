# Phase 1, first run (Colab T4, 2026-11-28)

**Setup:**
- `scripts/train_phase1.py` at commit 0dcc857.
- Repository state at clone: 11 parallel stories; the lexicon had not yet been extended.
- Frozen AraGPT2-base, CSAM + DAL trained, 3000 steps, batch 8, seq 256, seed 42.
- Run time: about 21 minutes.

## Held-out language-model loss

| variety | backbone only | with CSAM + DAL | Δ | perplexity ratio |
|---|---|---|---|---|
| egyptian | 5.938 | 5.600 | −0.34 | 1.40× lower |
| gulf | 6.322 | 5.844 | −0.48 | 1.61× lower |
| levantine | 6.090 | 5.786 | −0.30 | 1.35× lower |
| maghrebi | 7.022 | 5.817 | −1.21 | 3.34× lower |
| msa | 5.902 | 4.761 | −1.14 | 3.13× lower |

**Reading these numbers:**
- The loss falls for every variety, and most for the varieties furthest from the backbone's training data (Maghrebi).
- **Limits:**
  - One seed.
  - 234 held-out examples.
  - Wikisource held-out paragraphs come from the same books as the training paragraphs.
  - Held-out parallel stories are synthetic and in the same style as the training ones.
  - Without an ablation, the gain cannot be split between CSAM and DAL.

## Generated samples
The samples are poor:
- **Repetition loops:** «شمس شمس …», «يحكي يحكي …», «تاع تاع …».
- **Social-media register:** text such as «يا وزير صحة», which comes from IADD.
- **Latin-script Arabizi and Algerian forms** (تاع، بصح) in the Maghrebi sample. IADD's MGH region includes Algeria and Tunisia.
- **Stray combining marks:** the prompts were diacritised, and AraGPT2 expects undiacritised text.

Dialect words do appear, for example the Levantine sample contains قاله، حدا، بيقدر.

## Fixes made after this run
1. Diacritics and tatweel are stripped from training text and prompts.
2. IADD is filtered:
   - Maghrebi is restricted to Morocco (MAR); Levantine to JOR/PSE/SYR/LBN; Egyptian to EGY.
   - Latin-script rows and rows with less than 95% Arabic letters are dropped.
   - Rows under 4 words are dropped.
3. Decoding uses a repetition penalty of 1.3 and blocks repeated 3-grams.
4. Held-out loss is reported by source and variety.
5. `--no-csam` and `--no-dal` flags allow an ablation.
6. SAS markers: ابي ("my father" vs Gulf "I want") and عندو (also Levantine) are excluded.

---
# Phase 1, second run (Colab T4, 2026-11-28, commit 777af54)

**Result: failed.** The final log lines reported by Mossab:
- The train LM loss reached **1.61**.
- Held-out loss on *parallel stories*: 10.6–12.5 across all five varieties. In the first run the per-variety figures, which mixed sources, were about 5.6–5.8.
- Wikisource MSA held-out: 4.46.
- Held-out IADD text appeared **only for Gulf**.

**Causes:**
1. **My bug.** The IADD country filter used codes (EGY, JOR, MAR…), but IADD stores full names. The datasets-server statistics list Palestine, Morocco, Egypt, Algeria, Lebanon, Iraq, Tunisia, Jordan, Syria and NA. As a result, every Egyptian, Levantine and Maghrebi IADD row was dropped. Gulf survived only because it has no country filter.
2. **Overfitting.** Without IADD, the 23 synthetic parallel stories, each repeated 20 times, dominated training. The model memorised them: the train loss fell to 1.6 while the loss on held-out stories rose.

**Fixes:**
- Country names corrected.
- The run now stops if any variety has no IADD training text.
- `--parallel-repeat` default lowered from 20 to 4.
- Held-out checks every 500 steps; the best adapters are restored at the end.
