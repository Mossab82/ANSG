# Thesis-to-code mapping

Legend — **Thesis ref.:** section given in `README.md`/brief (unverified) or
`TO TRANSCRIBE`. **Status:** `implemented` (code + tests), `interface`
(types/protocols only, no scientific computation), `absent`.

| Thesis component | Thesis ref. | Code | Tests | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Arabic text normalisation | Ch. 3 (§ TO TRANSCRIBE) | `src/ansg/data/normalization.py` | `tests/unit/test_normalization.py` | implemented | Steps used in thesis unknown; non-destructive defaults; legacy profile available. |
| Sentence/paragraph segmentation | Ch. 3 | `src/ansg/data/segmentation.py` | `tests/unit/test_segmentation.py` | implemented | Rule-based. No clitic segmentation (legacy used Stanford). |
| One Thousand and One Nights ingestion (story identification, nights, formulae) | Ch. 3 | `src/ansg/data/nights.py`, `arabic_numbers.py` | `test_nights.py`, `test_arabic_numbers.py` | implemented | Edition-specific patterns must be configured. Embedded-tale nesting not inferred. |
| ANHC record (DIA, MOTIF, MORAL, provenance, splits) | Ch. 3 | `src/ansg/data/records.py`, `anhc.py` | `test_records.py` | implemented (schema) | Label inventories & gold data not supplied. Schema id `ansg.anhc-record/0.1` is ours. |
| Dataset splitting | Ch. 3 / Ch. 5 | `src/ansg/data/splits.py` | `test_splits.py` | implemented | Ratios unverified; use `predefined` with the original partition when available. |
| Ingestion pipeline / manifests | — (engineering) | `src/ansg/data/pipeline.py`, `utils/provenance.py` | `tests/integration/test_pipeline.py` | implemented | |
| ArabicVerbNet | Ch. 3 / Ch. 4 (§ TO TRANSCRIBE) | `src/ansg/knowledge/arabic_verbnet.py` | `test_arabic_verbnet.py` | implemented (parser) | Format from legacy `Verb-net.py`. Real files not in repo. |
| ArabicNameNet | Ch. 3 (§ TO TRANSCRIBE) | `src/ansg/knowledge/arabic_namenet.py` | `test_arabic_namenet.py` | implemented (configurable adapter) | **Format unverified.** No legacy reference exists. |
| Cultural Knowledge Graph | Ch. 3/4 | `src/ansg/knowledge/ckg.py` | `test_ckg.py` | interface + data model | Graph data, counts, relation types not supplied. |
| CST-Onto | Ch. 3/4 | `ckg.load_ontology_rdf` | `test_ckg.py::test_rdf_ontology_loader` | implemented (loader) | Ontology file not supplied. |
| Schema activation scores | Ch. 4 | `ckg.SchemaActivator`, `ActivationScores` | `test_ckg.py` | interface | Thesis function TO TRANSCRIBE. |
| CSAM | Ch. 4 | — | — | absent | Dimensions disputed/unknown. |
| DAL / LoRA | Ch. 4 | — | — | absent | |
| Morphological conditioning | Ch. 4 | — | — | absent | |
| Backbone | Ch. 4 vs Ch. 5 vs App. A | — | — | absent | Contradiction preserved. |
| Training phases/stages | Ch. 4–5 vs App. A.26 | — | — | absent | |
| Cultural reward / RL | Ch. 5 vs App. A.3 | — | — | absent | |
| CRAFT | Ch. 6 | `src/ansg/evaluation/craft.py`, `configs/evaluation/craft.yaml` | `test_craft_interfaces.py` | interface | Metrics deliberately raise `MetricNotSpecifiedError`. |
| Human evaluation protocol | Ch. 6 | `craft.HumanEvaluationProtocol` | — | interface | All fields null. |

## Legacy code → new code

| Legacy | New | Relationship |
| --- | --- | --- |
| `Normalize_Text` | `normalization.normalize` + `NormalizationConfig.legacy()` | Re-implemented; lossy behaviour opt-in. |
| `Stories_Split`, `Title_extractor`, `story_text_extractor` | `nights.segment_stories` | Replaced (regex + offsets). |
| `Sentances_extractor` split on `،` | `segment_sentences(split_on_arabic_comma=True)` | Optional. |
| `Verb-net.py` | `arabic_verbnet.ArabicVerbNet` (+ `legacy_rows`) | Re-implemented; frames/roles now kept. |
| `ANER`, `Wiki_NER`, `N_ANER` | — | **Not** ported; not ArabicNameNet; network-dependent. |
| TF-IDF, cosine, K-means, n-grams, keywords | — | Not ported (no thesis requirement identified yet). |
