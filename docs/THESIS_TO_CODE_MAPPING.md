# Thesis-to-code mapping

Updated 2026-11-13 with references from the supplied dissertation. **Status:**
`implemented` (code + tests, faithful to the cited text), `implemented (choice)`
(implemented, but a choice the thesis leaves open is an explicit parameter),
`interface` (types/config only), `absent`.

| Thesis component | Thesis reference | Code | Tests | Status | Notes / conflicts |
| --- | --- | --- | --- | --- | --- |
| Normalisation (App. A.8 protocol) | Table A.8, Listing A.3 | `data/normalization.py` (`profile: thesis_appendix_a8`) | `test_thesis_profiles.py` | implemented (choice) | "except calligraphic" tatweel & dialect exceptions not representable (T-2); C-NORM-1 |
| Non-destructive default normalisation | — (engineering) | `data/normalization.py` | `test_normalization.py` | implemented | Default because of C-NORM-1 |
| Segmentation | §3.4 (segments), §5.4 | `data/segmentation.py` | `test_segmentation.py` | implemented | Thesis segment definition not given |
| Nights ingestion | §3.2.2, §3.2.4 | `data/nights.py`, `arabic_numbers.py` | `test_nights.py`, `test_arabic_numbers.py` | implemented | Mahdi/Bulaq/Calcutta II as separate sources; C-ANHC-8 |
| Formulaic expressions | §5.4.2, Table 5.4 | `nights.THESIS_TABLE_5_4_FORMULAE`, folded matching | `test_thesis_profiles.py` | implemented (choice) | 6 of 437 patterns; Arabic reconstructed from transliteration (T-1) |
| ANHC record (DIA, MOTIF, MORAL) | §3.5, Fig. 3.2, Table 3.7 | `data/records.py`, `anhc.py` | `test_records.py` | implemented | DIA (5) and MORAL (8) inventories configured; MOTIF codebook absent (C-ANHC-3, C-ANHC-4) |
| Annotation layers per App. A.7 | Table A.7 | — | — | absent | Different layer design (C-ANHC-6) |
| Splits 80/10/10, seed 42 | §5.4.4, §5.2.1 | `data/splits.py`, `configs/data/nights.yaml` | `test_splits.py` | implemented | |
| Cluster-based leakage control | §5.4.4 (TF-IDF + DBSCAN ε 0.3, min 2) | `splits.cluster_records_dbscan`, pipeline `cluster_dbscan` | `test_thesis_profiles.py` | implemented (choice) | metric & TF-IDF settings unstated (C-SPLIT-1) |
| Stratification targets | §5.4.4 | `splits.split_distribution` | — | interface (report only) | combination rule unstated |
| CRD | Eq. 3.1, §3.6.1 | `evaluation/corpus_metrics.py` | `test_crd.py` | implemented (choice) | lexicon not supplied; multi-word policy explicit |
| IAA statistics | §3.5.4, §3.6.2 | `evaluation/statistics.py` | `test_thesis_computations.py` | implemented | percentile vs BCa (C-EVAL-4) |
| CST-Onto loading | §3.1.4, App. A.19 | `knowledge/ckg.load_ontology_rdf` | `test_ckg.py` | implemented | ontology file not supplied |
| CKG + declared counts | §4.3.2, Table 4.8, A.16, A.2 | `knowledge/ckg.py`, `configs/knowledge/ckg.yaml` | `test_ckg.py`, `test_thesis_profiles.py` | implemented | four conflicting declarations kept (C-CKG-1/2) |
| Schema activation | Eq. 4.20, App. A.4 | `ckg.CosineThresholdActivator` | `test_thesis_profiles.py` | implemented (choice) | embeddings & context encoder supplied by caller (C-CSAM-5) |
| OWL2Vec* schema embeddings | §4.3.2, Eq. 4.8 | — | — | absent | Phase 2 |
| Cultural weight w_c | Eq. 4.2 | `knowledge/cultural_weight.py` | `test_thesis_computations.py` | implemented (choice) | Likert→[0,1] mapping unstated |
| Morphological pipeline (Eq. 4.3–4.7) | §4.2 | — | — | absent | C-MORPH-1 |
| CSAM (Eq. 4.9–4.19) | §4.3 | — | — | absent | C-CSAM-1…6 |
| DAL (Eq. 4.24–4.25) | §4.4 | — | — | absent | C-DAL-1/2 |
| Backbone | §4.1.6, Table 5.1, A.3 | `configs/specification/thesis_variants.yaml` | `test_config.py` | interface | C-MODEL-1 |
| Training phases | Ch. 3–5, A.26 | same | — | interface | C-TRAIN-1…5 |
| Reward / AHP | §4.5, §5.6.3, A.3, A.11, A.23 | `evaluation/ahp.py`; variants in yaml | `test_thesis_computations.py` | implemented (AHP) / interface (reward) | C-REWARD-1…6 |
| CRAFT metrics | Ch. 6 | `evaluation/craft.py`, `configs/evaluation/craft.yaml` | `test_craft_interfaces.py` | interface | none definable (C-EVAL-1/2) |
| Gap closure, Eq. A.1 | Eq. 6.1, A.10.2 | `evaluation/statistics.py` | `test_thesis_computations.py` | implemented | |
| Human evaluation protocol | §6.1.3, A.13.1, A.24 | `configs/evaluation/craft.yaml` | — | interface | C-EVAL-8 |
| ArabicVerbNet | not in ANSG (App. B.3.7 only) | `knowledge/arabic_verbnet.py` | `test_arabic_verbnet.py` | implemented (brief) | C-RES-1 |
| ArabicNameNet | not in ANSG (App. B.3.7 only) | `knowledge/arabic_namenet.py` | `test_arabic_namenet.py` | implemented (brief) | C-RES-1; format unknown |
| Arabic WordNet, Motif DB, Historical Lexicon | §4.2.5 | — | — | absent | resources not supplied |

## Legacy code → new code

| Legacy | New | Relationship |
| --- | --- | --- |
| `Normalize_Text` | `NormalizationConfig.legacy()` | Re-implemented; not the thesis protocol (A.8 keeps diacritics). |
| `Stories_Split`, `Title_extractor`, `story_text_extractor` | `nights.segment_stories` | Replaced. |
| `Sentances_extractor` split on `،` | `segment_sentences(split_on_arabic_comma=True)` | Optional. |
| `Verb-net.py` | `arabic_verbnet.ArabicVerbNet` (+ `legacy_rows`) | Re-implemented. |
| `ANER`, `Wiki_NER` | — | Not ported (not ArabicNameNet; network). |
| TF-IDF, cosine similarity, K-means | `splits.cluster_records_dbscan` (TF-IDF + DBSCAN) | The thesis uses TF-IDF + DBSCAN for splitting (§5.4.4); the legacy K-means code is unrelated. |
