# CRAFT evaluation specification

*Transcribed 2026-11-13 from the dissertation (Ch. 3 §3.6, Ch. 4 §4.5, Ch. 6, App. A.10–A.14, A.24).
Registry: `configs/evaluation/craft.yaml`. Conflicts: `SPECIFICATION_CONFLICTS.md` (C-EVAL-*).*

## 1. Bottom line

**No CRAFT metric can be computed from text with what the dissertation specifies.**
CRAFT ("Cultural Reasoning and Authenticity Fidelity Test") is described three
incompatible ways (C-EVAL-1). CulturalScore, ALDA, MFA and PCI have no equation or
algorithm. The central metric CMPI has two different expansions; its equation is defined as an AHP-weighted
composite (Ch. 5 §5.1 p.131; §8.7.4 p.223; implemented in `ansg.evaluation.cmpi`), but the
scorers of its four dimensions are not (C-EVAL-2). Therefore every
registry entry has status `incomplete`, and `ansg.evaluation.craft` refuses to
compute them. What *is* fully specified (statistical procedures, gap closure, CRD
up to two stated ambiguities) is implemented and tested.

## 2. What the thesis says CRAFT is

| Version | Location | Content |
| --- | --- | --- |
| (a) | Ch. 6 intro ("Evaluation Metrics") | Ontology-driven evaluation over 847 CST-Onto concepts; dimensions: virtue concepts, character archetypes, formulaic expressions; purpose: diagnose under-represented traditions. |
| (b) | §6.2.4 | "4 dimensions (CMPI, CulturalScore, ALDA, MFA)"; ontology + AHP + expert validation; κ ≈ 0.82. |
| (c) | Table 6.4 | Rhetoric, Formulaic, Metaphor, Prosody, Moral Framework scores + overall CMPI. |
| (d) | §4.1.1 RQ3; Table 4.15 | "multi-dimensional … derived through AHP with 12 experts"; "4-dimensional scoring (§6.3)". |

The brief's four dimensions (cultural authenticity, dialectal fidelity, linguistic
quality, narrative coherence) are used in code only as a grouping; they match the
reward components and human-evaluation dimensions, not a CRAFT definition.

## 3. Metric extraction table

| Metric | Expansion (location) | Definition in thesis | Range / direction | Status | Missing for implementation |
| --- | --- | --- | --- | --- | --- |
| CMPI | Cultural *Motif* Preservation Index (Ch. 4, Ch. 6, glossary); Cultural *Meaning* Preservation Index (Ch. 5 §5.1, Table A.2) | **Equation defined** (prose, AHP weights): CMPI = 0.35·rhetorical devices + 0.25·dialectal appropriateness + 0.25·narrative structure + 0.15·moral framing (§5.1 p.131; §8.7.4 p.223; Table A.2 p.268 'composite metric (0–1 scale)'); ANSG reference `ansg.evaluation.cmpi.ansg_cmpi`. Ch. 6 (p.160) calls it originally CULTURA's metric ('hierarchical schema activation scoring'), extended with 847 concepts and cross-dialect weighting; the CULTURA paper's metric is in fact a different 'Cultural Matching Performance Index' (C-EVAL-14). CULTURA/CST-Onto/MM-GTA metrics sharing the acronym are separate (`configs/evaluation/cmpi_variants.yaml`, C-EVAL-14..16). | 0–1, ↑ | incomplete (dimension scorers) | dimension scorers; relation of Table 6.4 dimensions and 'Overall CMPI' to the definition |
| CulturalScore | rhetorical sophistication (NarrativeMind) | sajʿ, jinās, ṭibāq presence/appropriateness; human ceiling 0.96. §3.6.1 uses the name for AHP-based expert ratings 0–100. | 0–1, ↑ | incomplete | detection method for each device; aggregation (C-EVAL-5) |
| MFA | Moral Framework Alignment (glossary, §6.2.4) | none | — | incomplete | everything |
| ALDA | Arabic Linguistic Dialectal Accuracy | Table 6.6 lists consistency %, authenticity (1–5 native rating), register accuracy %, code-switch quality (/5) | % or 0–1 (C-EVAL-6) | incomplete | formulas; dialect classifier; combination |
| PCI | Plot Coherence Index | none (used as reward, Table 4.1) | — | incomplete | everything |
| Table 6.4 dimensions | Rhetoric, Formulaic, Metaphor, Prosody, Moral Framework | none | 0–1 | incomplete | everything |
| BLEU-4, ROUGE-L, BERTScore (AraBERT), perplexity | standard | references, tokenisation, model versions, PPL normalisation not stated | — | incomplete | C-REWARD-3, C-REWARD-5 |
| AGAS, CCS, GER | glossary only | none | — | incomplete | never used in Ch. 3–6 |
| CRD | Cultural Reference Density | **Eq. 3.1** | per 1,000 tokens, ↑ | implemented with mandatory `policy` | lexicon (2,847 entries); multi-word rule; tokenizer |

## 4. Aggregation

No formula for a composite CRAFT score (Table 4.16 mentions a "Combined CRAFT score").
The AHP weight sets in the thesis belong to the **RL reward**, not to CRAFT, and
themselves conflict (C-REWARD-1, -2, -4). `aggregate()` therefore refuses to run
unless weights are supplied, flagged verified and given a thesis reference.

## 5. Fully specified procedures (implemented)

| Procedure | Thesis | Code |
| --- | --- | --- |
| Gap closure = (S − B)/(H − B) | Eq. 6.1 | `statistics.gap_closure` |
| Scale normalisation (Raw − min)/(max − min)·100 | Eq. A.1 | `statistics.normalize_scale` |
| Bonferroni | §6.2.1 (9 comparisons) vs Table A.11 (6) | `statistics.bonferroni_alpha` (n explicit) |
| Cohen's κ, Fleiss' κ | §3.5.4 | `statistics.cohens_kappa`, `fleiss_kappa` |
| Percentile bootstrap | §3.6.2 (1,000) — A.12 says BCa 10,000 (C-EVAL-4) | `statistics.bootstrap_ci` (BCa not implemented) |
| Cohen's d | §6.1.5 (variant unstated) | `statistics.cohens_d(variant=…)` required |
| AHP eigenvector + CR | §4.5.2, A.11, A.23 | `evaluation.ahp.ahp_weights` |
| CRD | Eq. 3.1 | `evaluation.corpus_metrics.cultural_reference_density` |

## 6. Human-evaluation protocol (three variants, C-EVAL-8)

Recorded in full in `configs/evaluation/craft.yaml: human_evaluation_variants`:
§6.1.3 (5 raters, 4 dimensions, 200 items/system, Fleiss κ 0.82, ICC 0.87, SD > 1.0
disagreement rule); A.13.1 (5 regional PhDs, 10% duplicates, κ > 0.7 target);
A.24 (30 evaluators, 3 dimensions, rubric Table A.22, error thresholds A.24.3,
mixed-effects + Tukey HSD). Likert anchors 1–5 (§6.1.3) and the rubric (Table A.22)
are complete enough to build rating forms.

## 7. Python interfaces

`ansg.evaluation.craft` (unchanged design): `CraftDimension`, `SpecStatus`,
`MetricSpec`, `EvaluationSample`, `MetricResult`, `CraftMetric` protocol,
`UnimplementedMetric`, `CraftRegistry.from_config`, `HumanEvaluationProtocol`,
guarded `aggregate`. Plus implemented `ansg.evaluation.{statistics,ahp,corpus_metrics}`.

## 8. Unit-test specifications for future metric implementations

A metric may move to `specified` only when the author supplies its definition
(equation + every symbol + reference data/model versions). Then:

1. **Reproduce a thesis number**: e.g. if CMPI is defined, recompute the §4.5.4 example inputs; where the thesis' own arithmetic is wrong (C-REWARD-6) test the recomputed value and document it.
2. **Range** — outputs within the stated range on random and edge inputs.
3. **Boundary cases** — empty text, single token, candidate = reference, unknown dialect, motif not in CKG.
4. **Invariances stated in the thesis** — e.g. diacritic handling per C-NORM-1 once decided.
5. **Monotonicity** where implied (adding a target motif must not lower a motif-coverage score).
6. **Determinism** — fixed seeds; pinned external model versions.
7. **Provenance** — `MetricSpec.thesis_reference` non-empty and `status == specified` before registration.
8. **Aggregation** — hand-computed composite equals `aggregate()` with verified weights.

Existing tests: `tests/unit/test_craft_interfaces.py` (registry transcribed, all
incomplete, refusal to compute), `test_thesis_computations.py` (Eq. 6.1, A.1, AHP,
Eq. 4.2), `test_crd.py` (Eq. 3.1 policies).
