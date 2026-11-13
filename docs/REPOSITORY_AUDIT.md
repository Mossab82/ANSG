# Repository audit (Phase 1)

*Audit date: 2026-11-13. Audited revision: `c15e9ff` (default branch `master`).*
*Method: full read of every tracked file, `py_compile` of every script, `pyflakes`
for undefined names, and inspection of git history for credential exposure
(values were redacted during inspection and are not reproduced here).*

## 0. Critical limitation of this audit (resolved later the same day)

> **Update 2026-11-13:** the author supplied the dissertation as a Markdown export.
> It has since been transcribed: see `EXPERIMENTAL_SPECIFICATIONS.md` and
> `SPECIFICATION_CONFLICTS.md`. Two findings change §6 below: ArabicVerbNet and
> ArabicNameNet are **not** ANSG components in the thesis (C-RES-1), and the thesis'
> code repository is `mossab-ansg/ansg-framework`, not this repository (C-REPRO-1).
> The original text of this section is kept for provenance.

**At the time of the initial audit the dissertation PDF was not available in the execution environment.** It was
not attached to the session, is not tracked in the repository and was not
found on the file system. Consequently:

* No equation, hyperparameter, schema count, label inventory or section number
  was transcribed from the dissertation in this phase.
* The only dissertation-derived statements in this repository are those already
  written by the repository owner in `README.md` (commit `faaa3c0`) and
  `docs/legacy_security_migration.md` (commit `c15e9ff`), plus the component
  names given in the Phase 1 brief. They are quoted with that attribution and
  have **not** been re-verified against the thesis text.
* Every place where thesis content is required is marked `TO TRANSCRIBE` in the
  docs and `null`/`unverified` in the configs.

## 1. Inventory

| Path | Lines | Parses (py3.13) | What it actually is |
| --- | ---: | --- | --- |
| `Main.py` | 722 | yes (1 `SyntaxWarning`: `"C:\Program Files…"` escape) | Older copy of the preprocessing pipeline (subset of `preprocess.py`). |
| `preprocess.py` | 1105 | yes | Legacy story-splitting, Stanford segmentation/POS/parsing, n-grams, TF-IDF keywords, cosine similarity, K-means, polyglot/Wikipedia NER → MySQL. |
| `Verb-net.py` | 123 | yes | ArabicVerbNet XML → MySQL loader. |
| `Stopwordsar.py` | 392 | yes | Arabic stop-word set; format identical to spaCy v2 `spacy/lang/ar/stop_words.py` (MIT) — origin should be attributed. **Not imported anywhere.** |
| `wiki.py` | 27 | yes | Rewritten in `c15e9ff` into an opt-in HTTP client for an "ArabicNLP" service. Unrelated to Wikipedia despite the name. |
| `svm.py` | 174 | yes | Vector-space (TF-IDF/cosine) retrieval toy; **no SVM**. English `PorterStemmer` applied to Arabic; global `files` is never populated, so `main()` cannot work. |
| `test.py` | 72 | yes | TF-IDF toy over three Arabic paragraphs; **not a test**. |
| `tfidf.py` | 27 | yes | English sklearn TF-IDF toy. |
| `keywords.py` | 20 | yes | English tokenise/stem/TF-IDF toy. |
| `doc2vec.py` | 22 | **no** (`IndentationError`, line 8) | Unfinished gensim Doc2Vec snippet; `articles` undefined. |
| `README.md`, `docs/legacy_security_migration.md`, `.env.example`, `.gitignore` | — | — | Added by the owner on 2026-11-13. |

`src/ansg/`, `configs/` and `tests/` **did not exist** before this phase; the
README lists them as targets only.

## 2. Reusable elements

| Element | Where | Reuse decision |
| --- | --- | --- |
| ArabicVerbNet element/attribute usage (`MEMBERS/*@transname`, `Root|Deverbal|Participle@name`, `FRAMES/*/DESCRIPTION@primary`, `SYNTAX/NP|PREP@value`) | `Verb-net.py` | **Primary evidence for the resource format.** Re-implemented in `src/ansg/knowledge/arabic_verbnet.py`, including a `legacy_rows()` view matching the old `Arabic_Verbs` table. |
| Normalisation = `strip_harakat` + `strip_tashkeel` | `Main.py`, `preprocess.py` | Reproducible as `NormalizationConfig.legacy()`; no longer the default (lossy). |
| Story split = "title is 2nd line of a paragraph" heuristic, text split on `.` | `Stories_Split`, `Title_extractor` | Not reused (substring matching, first-occurrence `index`, string `"None"` sentinel). Replaced by configurable regex-based tale/night segmentation. |
| Sentence split on Arabic comma `،` | `Sentances_extractor` | Optional (`split_on_arabic_comma`). |
| Custom stop-word additions | `create_stop_wrods`, `SW_Remover` | Not needed in Phase 1 (no stop-word removal before generation). Kept in legacy files. |
| MySQL schema names (`Story_Narative.Naitave_Story`, `Nartive_Sentances`, `Nartive_Verbs`, `Narative_Ngram`, `Story_Keywords`, `VerbNet.Arabic_Verbs`, …) | all DB scripts | Documented only. **No DDL or dump is in the repo**, so table structure is known only from `INSERT`/`SELECT` statements. Not equated with ANHC. |

## 3. Defects found in legacy code (not fixed in place)

Historical files are preserved unchanged for provenance. Defects:

1. **Silent data loss — `Sentances_extractor`** (`Main.py:343`, `preprocess.py`):
   the sentiment and hash lines are commented out but `sentement['score']` and
   `sent_hash.hexdigest()` are still used. Every iteration raises `NameError`,
   swallowed by a bare `except` that prints "Duplicated Sentance". Unless
   another version of this function was used, `Nartive_Sentances` could not
   have been populated by this code.
2. `DF_IDF_D` references undefined `np`, `processed_title`, `words_count`,
   `doc_freq`, `N`, `doc`, `id` (dead code).
3. `ner()` (`preprocess.py`) iterates undefined `sent`; the `__main__` block uses
   undefined `DB`; `word_catagrizer` depends on undefined globals `words`,
   `tagged_data`.
4. `SW_Remover` removes items from a list while iterating it (skips tokens).
5. `Verb-net.py`: the database section (lines 89–124) is **outside** the
   `__main__` guard; frame parsing is nested inside the member loop and its
   NP/PREP results are discarded; `if not(d.attrib in syn)` compares a dict to
   strings; "category" = position in unsorted `glob` order (file-system
   dependent); rows with 0 deverbals go to `Ara_Verbs`, others to
   `Arabic_Verbs`.
6. `Doc_Cluster`: K-means without `random_state` → non-reproducible clusters;
   `cos_sim` uses `list.index(value)` (wrong on ties).
7. `story_text_extractor`: `if start in P` is substring matching and
   `Par.index(P)` returns the first equal paragraph.

## 4. Deprecated / non-portable dependencies

| Dependency | Problem |
| --- | --- |
| `nltk` Stanford wrappers (`StanfordSegmenter`, `StanfordPOSTagger`, `StanfordParser`, 2018 jars) | Deprecated in NLTK in favour of the CoreNLP server; require local Java + jars at hard-coded relative paths. |
| `polyglot` | Unmaintained; difficult to install on modern Python (PyICU, pycld2). `preprocess.py` downloads models **at import time** and loads a hard-coded `C:/Users/…` path, so it cannot be imported on another machine. |
| `wikipediaapi.Wikipedia('ar')` | Current releases require a `user_agent` argument; this call fails. |
| `sklearn …get_feature_names()` | Removed in scikit-learn 1.2 (`get_feature_names_out`). |
| `repustate` | Commercial API client. |
| `mysql.connector` | Requires a populated MySQL server whose schema is not in the repo. |
| `JAVA_HOME = C:\Program Files\Java\jdk-13.0.1` | Hard-coded Windows path. |

## 5. Security and data-governance findings

| # | Finding | Status |
| --- | --- | --- |
| S1 | Plaintext MySQL passwords and a Repustate API key existed in earlier revisions of `Main.py`, `preprocess.py`, `Verb-net.py` (confirmed in `git log -p`; values not reproduced). | Removed from HEAD in `c15e9ff`. **Still present in git history.** Provider-side rotation cannot be verified from here. |
| S2 | `preprocess.py` still hard-codes the MySQL **username** (`user='Mossab'`). | Not secret, but inconsistent with `ANSG_MYSQL_USER`. Left unchanged (historical file); documented. |
| S3 | Corpus text leaves the machine: `Verbs_Extractor` sends verbs to Repustate; `Wiki_NER`/`N_ANER` send n-grams to Wikipedia; `wiki.py` posts text to a third-party endpoint. | Relevant to copyright of licensed source texts. The new pipeline makes **no** network calls. |
| S4 | Import-time side effects (model downloads, DB writes). | New code has none; test-verified that importing `ansg.*` needs only PyYAML. |
| S5 | `.env.example` contains no secrets; `.gitignore` excludes `.env*`, `data/private/`, checkpoints. | OK. Added `data/processed/`. |

The new code adds: rejection of literal credentials in YAML (`ansg.utils.config`),
`${env:NAME}` indirection, and rejection of XML with inline DTD/ENTITY
declarations (`ansg.knowledge.arabic_verbnet`).

## 6. Repository vs. dissertation (as far as can be established without the PDF)

| Thesis component (per README / brief) | In repository before Phase 1 |
| --- | --- |
| ANHC corpus with DIA / MOTIF / MORAL annotations | No corpus, no annotation files, no schema. Legacy MySQL tables hold stories, sentences, verbs, n-grams, keywords — no dialect, motif or moral fields. |
| One Thousand and One Nights | Implied by test strings (Shahriyar/Shah Zaman) and `Input_Stroy.txt` (not in repo). Edition unknown. |
| ArabicNameNet | **No reference anywhere in code.** Legacy NER = polyglot `ner2.ar` + Wikipedia title look-ups. |
| ArabicVerbNet | `Verb-net.py` loader; XML files (`mm/*.xml`) not in repo. |
| CKG / CST-Onto | Nothing. |
| CSAM, DAL/LoRA, RL, reward model | Nothing. |
| CRAFT evaluation | Nothing. |
| Backbone / training code / checkpoints / logs | Nothing. |

Conclusion: the legacy code is an early **preprocessing and lexical-analysis
precursor**. No model, training, or evaluation implementation from the
dissertation is present, and none of the legacy outputs can be treated as the
original training data without additional evidence.
