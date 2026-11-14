# Hypotheses (to be pre-registered before data collection)

Each hypothesis states the measure (METRICS.md), the comparison, the direction, and the result that would **falsify** it. Effect-size thresholds marked *[pre-register]* are fixed after the pilot (EXPERIMENT_PROTOCOL.md §7) and **before** the main study. They are not chosen after seeing results.

| ID | Hypothesis | Measure (unit: generated story) | Test | Falsified if |
| --- | --- | --- | --- | --- |
| **H1** (primary) | B > A on story resumption at depth ≥ 2 | Story Resumption Accuracy from blind annotation (M2) | Mixed-effects logistic regression: condition + depth + condition×depth, with random scenario and seed | The 95% CI of the B−A difference includes 0 or is negative |
| H2 | B > A on narrative-level accuracy and closure accuracy, and the gap grows with depth | M1, M6 | Condition×depth interaction | No positive interaction, and no main effect |
| H3 | B > C on character-knowledge consistency (the epistemic layer matters) | M5 knowledge violations per story | Poisson / negative-binomial mixed model | B is not lower than C |
| H4 | B > D on resumption and level accuracy (hierarchy matters beyond knowledge checks) | M1, M2 | As H1 | B − D CI includes 0 |
| H5 (non-inferiority) | B is not worse than A by more than δ on human-rated language quality and dialect naturalness | H-LANG, H-DIAL (Likert) | Cumulative-link mixed model; one-sided CI vs −δ, δ *[pre-register]* | The lower bound of B−A is below −δ |
| H6 (sanity) | At depth 1, B ≈ A on all structural measures | M1, M2, M6 | Equivalence (TOST), margin *[pre-register]* | B differs from A at depth 1. That would indicate a confound such as the prompt format, not nested-narrative control |
| H7 | Effects of H1 hold under R1 and under R2 | as H1 | condition×register | The effect reverses sign under one register |

**Multiplicity:**
- H1 is the single primary confirmatory test.
- H2–H7 are secondary, with Holm correction across them.
- Exploratory analyses are labelled as such.

**Not hypothesised:**
- that HNC improves "story quality" in general;
- that it reproduces any dissertation number;
- that a model has *learned* nesting.
