# Seed Adherence Score (SAS)

Code: `src/ansg/evaluation/seed_adherence.py`. Tests: `tests/unit/test_seed_adherence.py`.

## What it measures
SAS answers one question: **did the story use the cultural elements it was asked to use?** Those elements are the ones that `build_story_prompt` sampled from `data/cultural_seeds`:
- the proverb;
- the traditional opening and closing;
- the character and setting;
- the dialect of the dialogue.

SAS does **not** measure:
- cultural authenticity;
- story quality or coherence;
- the dissertation's CMPI. SAS is a new, separate metric, and nothing is taken from the thesis numbers.

SAS is lexical. A story can paraphrase a proverb faithfully and still get a low proverb score. A story can also include the right words and still be bad.

## Equations
All components lie in [0, 1].

| Component | Definition |
|---|---|
| phrase recall | r(P, W) = LCS_tokens(P, W) / \|P\| |
| proverb | p = r(proverb, story) |
| opening | o = r(opening, first \|opening\| + 10 tokens) |
| closing | c = r(closing, last \|closing\| + 10 tokens) |
| entities | e = ½ · (r(character, story) + r(setting, story)) |
| dialect, dialect regions | d = h_t / (h_t + h_o), and 0 when no marker is found |
| dialect, MSA | d = 1 / (1 + h_o) |
| **score** | **SAS = 0.2·p + 0.2·o + 0.2·c + 0.2·e + 0.2·d** |

**Terms:**
- **LCS_tokens** is the longest common token subsequence, computed after normalisation. Normalisation removes diacritics and tatweel and unifies alef, ya, ta marbuta and hamza carriers. A leading و or ف on a token is ignored.
- **h_t** is the number of story tokens that are distinctive markers of the target region.
- **h_o** is the same count for the other regions.
- **Distinctive markers** are built in three steps:
  1. Take the tokens of a region's `dialect_markers`.
  2. Remove tokens that occur in any MSA phrase in the seed files, or in another region's markers.
  3. Remove a short declared list of standard or pan-dialect words (`NOT_DISTINCTIVE`).

**Weights.** The equal weights are a declared default, not an empirical result. If an element was not requested, its component is reported as `None` and the remaining weights are renormalised. Every component is reported separately next to the score.

## Known limitations
- The marker sets are small: 11–24 tokens per region. They come from seed files that a native speaker has not yet reviewed.
- The Levantine set mixes Jordanian, Palestinian, Syrian and Lebanese forms. The Gulf set mixes Najdi, Hijazi, Emirati, Kuwaiti and Omani forms.
- A story written fully in MSA gets d = 0 for a dialect region, even if it is otherwise good. This is intended, because the prompt asks for dialect dialogue.
- Exact lexical matching misses spelling variants that normalisation does not cover.
- **Coverage is low on narrative text.** On the first six parallel stories (about 90 words per variant), each variant matched only 0–3 distinctive markers, and several matched none. The seed markers are mostly greetings and questions. The dialect component d therefore needs larger marker lexicons before it can be trusted. One option is to derive them by frequency contrast from real dialect corpora such as IADD.
- **First extension (`data/cultural_seeds/lexicons/dialect_lexicon.json`).** Words exclusive to one variety across the 16 parallel stories were curated by hand into the lexicon. Region sizes are now 26–57 markers. Every dialect variant of those stories now has at least one marker. This coverage is **in-sample**: the lexicon was built from the same stories, so it says nothing yet about new text. The real test is new stories and generated text. The lexicon has not been reviewed by a native speaker.
- **First out-of-sample check (stories 17–21, not used to build the lexicon).**
  - 15 of the 20 dialect variants contain at least one target marker. Egyptian is weakest, at 3 of 5.
  - Two ambiguous Levantine markers produced cross-variety hits: عم (also "uncle") and وقتها (also Egyptian). Both were removed, and no cross-variety hits remain.
  - The commit message for that change says 16 of 20; the correct figure is 15 of 20.
- **Rebuild from training stories only (`scripts/build_dialect_lexicon.py`, 2026-11-28).**
  - Built from the 53 training stories. A word is kept if it occurs in at least 3 stories of one variety and in no other variety's or MSA version. A hand-curated exclusion list (with a reason per word) removes MSA words, words other varieties also use, spelling-only forms and the Maghrebi opening formula. The test split was not read.
  - Sizes: Egyptian 28, Levantine 24, Gulf 16, Maghrebi 93 (before merging with seed markers).
  - On the 9 reserved test stories, the share of dialect versions with at least one own marker:

    | | Egyptian | Levantine | Gulf | Maghrebi | Cross-variety hits |
    |---|---|---|---|---|---|
    | Old lexicon | 6/9 | 4/9 | 6/9 | 9/9 | 1 |
    | Rebuilt | 8/9 | 7/9 | 7/9 | 9/9 | 0 |

  - The 53 stories were ChatGPT-generated, so this measures agreement with one generator's dialect choices, not accuracy on real dialect text. Native review is still pending.
- **Cross-generator check (4 Gemini stories, parallel-0063–0066, scored with the lexicon built from ChatGPT stories only).**
  - Versions with at least one own marker: Egyptian 4/4, Levantine 2/4, **Gulf 0/4**, Maghrebi 4/4.
  - Gemini's Najdi uses words such as حنا، يم، طعس، كفو، شرى that the ChatGPT stories never used. The Gulf and Levantine markers therefore reflect one generator's vocabulary more than the dialect itself. This is direct evidence that d needs real dialect text or native review before it can be trusted.
  - Two false markers surfaced and were excluded: زي (also Jordanian/Palestinian) and كاين (after normalisation also MSA كائن).
  - Gemini batch 2 exposed two more homographs, now in NOT_DISTINCTIVE: وش (Gulf "what", Egyptian "face") and كاين (Moroccan "there is", MSA كائن after normalisation). Rebuilt with 62 training stories (Egyptian 29, Levantine 28, Gulf 17, Maghrebi 112); حل (Moroccan "open", MSA "solution") was excluded after it hit an MSA test story. Test coverage: 9/9, 7/9, 7/9, 9/9, 0 cross hits.
  - The lexicon was then rebuilt with all 57 training stories (both generators): Egyptian 27, Levantine 26, Gulf 16, Maghrebi 103. Test-split coverage is unchanged (8/9, 7/9, 7/9, 9/9, 0 cross hits).

## Validation plan (not yet done)
1. Generate stories with the chosen backbone from seed prompts covering all five regions.
2. Mossab rates each story blind on four 1–5 scales: dialect, language, coherence and culture. These are the same scales as the model-comparison review.
3. Report Spearman's ρ, with a bootstrap 95% CI, between SAS and its components on one side and the human ratings on the other. Report n.
4. Keep SAS only for what it correlates with. If d does not track the human dialect ratings, revise the marker sets before using d.

Until step 3 is done, SAS is a diagnostic of instruction adherence and nothing more.
