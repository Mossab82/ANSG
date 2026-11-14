# Metrics: operational definitions

These are **operational definitions written for this project**. They are not established or published metrics, and are not presented as such. Related measures in the literature are cited in LITERATURE_REVIEW.md where they exist.

## 0. Input: the realised trace

Two annotators, blind to the condition, read each generated story and mark:
- which planned events are realised, and where;
- the narrative level of each realised event;
- the narrator of each passage;
- the points where a tale opens and closes.

This gives a realised trace with the same format as the controller's `Trace` (`ansg.hnc.metrics`).

- Disagreements are adjudicated, and κ is reported per field (§3).
- The controller's own trace is perfect by construction, so it is used only as a software test.

## 1. Deterministic measures (exact functions of the annotated trace or graphs)

| ID | Name | Definition | Code |
| --- | --- | --- | --- |
| M1 | Narrative Level Accuracy | Share of realised planned events annotated at the depth of their planned frame | `structural_metrics()["narrative_level_accuracy"]` |
| M2 | Story Resumption Accuracy | Share of closed embedded tales whose next realised step is the parent's `resumed_by` event, or the next sibling tale at the same anchor | `story_resumption_accuracy` |
| M3 | Narrator Consistency | Share of realised events whose annotated narrator equals the planned narrator of their frame | `narrator_consistency` |
| M4 | Causal Coherence | Number of causal and state errors in Event Graphs annotated from the text: cycles, violated or unestablished preconditions, prevented-yet-occurring events, contradictory relations. Reported per story and per event | `consistency_counts()["causal_violations"]` |
| M5 | Character Knowledge Consistency | 1 − (knowledge violations ÷ knowledge requirements) in the annotated graphs, counting premature, unsourced and told-by-ignorant cases | `character_knowledge_consistency` |
| M6 | Narrative Closure Accuracy | Share of embedded tales that close in LIFO order after all their events | `narrative_closure_accuracy` |
| M7 | Event Coverage | Share of planned events realised exactly once; lost, duplicated and unplanned events are listed | `event_coverage` |
| M8 | Story Originality (copy) | 8-gram containment and longest verbatim run against the training corpus and the heritage source | `ansg.evaluation.story_proxies.copy_overlap` |

M8 is deterministic but only a **partial** originality measure: it cannot detect paraphrase or plot borrowing.

## 2. Automatic proxies (`metric_status: proxy`, never primary)

| Proxy | Intended for | Does not measure |
| --- | --- | --- |
| Dialect-marker rate (unreviewed marker list) | Dialect presence | Dialect authenticity or naturalness |
| Plan surface coverage | Named characters and triggers appearing | Order or causality |
| distinct-n, repetition | Diversity | Semantic or plot diversity |
| Arabic-letter ratio | Script sanity | Language quality |
| LLM-as-judge (optional) | Pre-screening only | Must be calibrated against the human ratings, with agreement reported |

## 3. Human judgements (EXPERIMENT_PROTOCOL.md §6)

All items use a 5-point Likert scale with anchored descriptions.

| ID | Item (Arabic wording given to raters) |
| --- | --- |
| H-PLOT | منطقية الحبكة |
| H-TRANS | طبيعية انتقال الراوي بين القصص |
| H-TRACK | سهولة تتبع المستويات السردية |
| H-RETURN | وضوح العودة إلى القصة الأصلية |
| H-LANG | جودة اللغة |
| H-CULT | أصالة البيئة الثقافية (Cultural Fidelity) |
| H-DIAL | طبيعية الحوار واللهجة (Dialect Naturalness) |
| H-ORIG | أصالة القصة وعدم شبهة النقل (human complement of M8) |

**Agreement statistics:**

| Data | Statistic | Uncertainty |
| --- | --- | --- |
| Ordinal ratings | Krippendorff's α (ordinal) | 95% bootstrap CI over items |
| Annotation of structural fields | Fleiss/Cohen κ (`ansg.evaluation.statistics`) | — |

## 4. Validation error codes (software)

These codes reject a plan or flag an annotated graph. Each one is tested in `tests/unit/test_hnc.py`.

| Group | Codes |
| --- | --- |
| Structure and anchors | `HNC-E-ROOT`, `HNC-E-PARENT-UNKNOWN`, `HNC-E-CYCLE`, `HNC-E-DEPTH`, `HNC-E-DUPLICATE-EVENT`, `HNC-E-ANCHOR-UNKNOWN`, `HNC-E-RESUME-ORDER` |
| Narration and time | `HNC-E-NARRATOR-UNKNOWN`, `HNC-E-AUDIENCE`, `HNC-E-PERSPECTIVE`, `HNC-E-TIME-CONTEXT` |
| Function and isolation | `HNC-E-NO-FUNCTION`, `HNC-E-FOREIGN-ENTITY`, `HNC-E-CROSS-FRAME-RELATION` |
| Knowledge, from the Event Graph validator applied per frame with hearing steps | `E-KNOWLEDGE-PREMATURE`, `E-KNOWLEDGE-UNSOURCED`, `E-TOLD-BY-IGNORANT` |
