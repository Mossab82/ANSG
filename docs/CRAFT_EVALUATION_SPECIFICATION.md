# CRAFT evaluation specification

> **Status: template.** The dissertation (Chapter 6) was not available when
> this file was written, so **no metric, equation, normalisation rule or weight
> has been extracted yet**. The four dimensions below are those named in the
> project brief; their exact names and number in Chapter 6 are unverified.
> Nothing in this repository computes a CRAFT score.

## 1. Dimensions (from the brief — verify against Ch. 6)

| Dimension id (code) | Brief wording | Thesis section | Metrics |
| --- | --- | --- | --- |
| `cultural_authenticity` | cultural authenticity | TO TRANSCRIBE | TO TRANSCRIBE |
| `dialectal_fidelity` | dialectal fidelity | TO TRANSCRIBE | TO TRANSCRIBE |
| `linguistic_quality` | linguistic quality | TO TRANSCRIBE | TO TRANSCRIBE |
| `narrative_coherence` | narrative coherence | TO TRANSCRIBE | TO TRANSCRIBE |

## 2. Metric extraction table (fill one row per metric in Ch. 6)

| Metric | Dimension | Thesis ref. (§, Eq.) | Equation (verbatim LaTeX) | Inputs required | Range | Normalisation | Higher is better | Completeness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | `specified` / `incomplete` |

Rules:
1. Copy equations verbatim; do not "fix" them silently. Record suspected errors in a *Notes* column.
2. If any symbol, normalisation constant, tokenisation, reference set or model used inside a metric is undefined, mark the metric `incomplete` and list what is missing. Incomplete metrics are not implemented.
3. If a metric depends on an external model (e.g. a dialect classifier or an LM for perplexity), record the exact model and version stated in the thesis; if not stated, `incomplete`.

## 3. Aggregation

| Item | Value | Thesis ref. |
| --- | --- | --- |
| Is there a composite CRAFT score? | TO TRANSCRIBE | |
| Aggregation form (weighted sum, mean, other) | TO TRANSCRIBE | |
| Weights / coefficients | TO TRANSCRIBE | |
| Per-dimension normalisation before aggregation | TO TRANSCRIBE | |

`ansg.evaluation.craft.aggregate` refuses to run unless weights are marked
verified and carry a thesis reference.

## 4. Human evaluation protocol

| Item | Value | Thesis ref. |
| --- | --- | --- |
| Number and profile of raters | TO TRANSCRIBE | |
| Dialect background of raters | TO TRANSCRIBE | |
| Rating scale | TO TRANSCRIBE | |
| Dimensions rated | TO TRANSCRIBE | |
| Items per system / sampling | TO TRANSCRIBE | |
| Systems compared (baselines) | TO TRANSCRIBE | |
| Blinding / randomisation | TO TRANSCRIBE | |
| Agreement statistic and threshold | TO TRANSCRIBE | |
| Statistical tests | TO TRANSCRIBE | |

## 5. Python interfaces (implemented as types only)

Module `ansg.evaluation.craft`:

```python
class CraftDimension(str, Enum): CULTURAL_AUTHENTICITY | DIALECTAL_FIDELITY | LINGUISTIC_QUALITY | NARRATIVE_COHERENCE
class SpecStatus(str, Enum): UNVERIFIED | INCOMPLETE | SPECIFIED | IMPLEMENTED

@dataclass(frozen=True)
class MetricSpec:            # one row of the extraction table
    name; dimension; status; thesis_reference; equation; range; normalization; higher_is_better; requires; notes

@dataclass
class EvaluationSample:      # generated_text, prompt, reference_texts, target_dialect, target_motifs, metadata

@dataclass
class MetricResult:          # metric, value, per_sample, details

class CraftMetric(Protocol):
    spec: MetricSpec
    def __call__(self, samples: Sequence[EvaluationSample]) -> MetricResult: ...

class UnimplementedMetric:   # raises MetricNotSpecifiedError
class CraftRegistry:         # from_config(configs/evaluation/craft.yaml), status_report()
@dataclass(frozen=True)
class HumanEvaluationProtocol
def aggregate(scores, weights: AggregationWeights) -> float   # guarded
```

Expected inputs from other modules: generated text (Phase 3), CKG and
`ActivationScores` (`ansg.knowledge.ckg`) for cultural metrics, DIA labels
(`ansg.data.records`) for dialect metrics, ANHC test split as references.

## 6. Unit-test specifications for future metric implementations

For **every** metric moved to `specified`:

1. **Worked example from the thesis.** If Ch. 6 gives a numeric example, reproduce it exactly (`pytest.approx`, tolerance stated).
2. **Range test.** Random/edge inputs stay within the declared range.
3. **Boundary cases.** Empty text, single token, identical candidate and reference, no shared vocabulary, unknown dialect label, motif not in CKG.
4. **Invariance tests** stated in the thesis (e.g. diacritic-insensitivity if normalisation is specified; order-invariance over samples).
5. **Monotonicity sanity check** where the definition implies it (e.g. adding a target motif occurrence must not decrease a motif-coverage metric).
6. **Determinism.** Same inputs → identical outputs across two runs; any model-based metric pins model version and seed.
7. **Normalisation test.** Normalised score of the defined min/max inputs equals the stated bounds.
8. **Aggregation test.** With weights from the thesis, a hand-computed composite equals `aggregate()`.
9. **Provenance test.** `MetricSpec.thesis_reference` is non-empty and `status == SPECIFIED` before the implementation is registered.

Already implemented interface tests (`tests/unit/test_craft_interfaces.py`):
shipped config declares no metrics; unimplemented metrics raise;
aggregation refuses unverified weights.
