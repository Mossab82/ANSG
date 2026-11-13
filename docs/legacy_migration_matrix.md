# Legacy-to-ANSG Migration Matrix

| Legacy file | Existing capability | Thesis linkage | Reuse plan | Reproducibility caveat |
|---|---|---|---|---|
| `Main.py` | Story extraction, paragraph and sentence splitting, N-grams, POS/NER | Ch. 3 preprocessing; Ch. 4 morphological conditioning | Preserve as historical script; export structured records via `ansg.legacy_bridge` | Legacy extraction != ANHC canonical annotations |
| `preprocess.py` | TF-IDF, cosine similarity, clustering, morphological analysis | Ch. 3 preprocessing and candidate cultural retrieval | Reuse methods only after test fixtures and dependency migration | Similarity features are not the CKG ontology |
| `Verb-net.py` | VerbNet XML parsing into SQL | Ch. 4 morpho-semantic conditioning | Optionally feed verb properties as features | This is not DAL, CSAM or CST-Onto |
| `Stopwordsar.py` | Arabic stopword definitions | Ch. 3 Arabic normalization | Review impacts on cultural rhetorical phrases | Do not remove culturally meaningful words blindly |
| `doc2vec.py` | Document embedding exploration | Ch. 3 corpus exploration | Reimplement and benchmark if needed | Older snippet is incomplete |
| `keywords.py`, `tfidf.py`, `svm.py` | Classical lexical and ranking utilities | Data baseline comparisons | Preserve with dependency pinning and unit tests | Not evidence for 847M ANSG architecture |
| `wiki.py` | External ArabicNLP HTTP client | Optional legacy extraction | Opt-in API call, credentials from environment | External API differs from ANHC processing protocol |
| `test.py` | Experimental lexical notebook-like script | Exploratory only | Preserve without running as a CI test until refactored | No ANSG evaluation claims |

## Data contracts

- **Legacy input:** SQL rows with `Story_ID`, `Story_Title`, `Story_Text`, no inferred annotation labels.
- **Annotated input:** requires explicit `DIA`, `MOTIF`, `MORAL` fields. The adapter checks dialect membership and types.
- **Target:** `NarrativeRecord` from `src/ansg/legacy_bridge.py`.
- **Validation still required:** actual ANHC ontology IDs, label coding manual, source licenses, dataset splits, privacy and decoding of expert annotations.

## Transition strategy

Keep the old files available for historical traceability; use `src/ansg/` modules in new experiments. Do not rely on database credentials, host-specific Java or Polyglot paths, or import-time model downloads in modern training. Replace such dependencies with validated adapters and explicit configuration.

This is a **software modernization and research reproduction project**, not a restoration of unavailable historical experiment files.
