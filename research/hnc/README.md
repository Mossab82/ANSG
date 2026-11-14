# ANSG-HNC — Hierarchical Narrative Controller (research extension)

> **بالعربية:** هذا المجلد يجهّز أساساً لورقة بحثية مستقبلية عن امتداد اختياري اسمه HNC، يتحكم في القصص المتداخلة (قصة داخل قصة) أثناء التوليد. HNC **ليس جزءاً من معمارية ANSG في الأطروحة**. لا تُخلط نتائجه بنتائجها، ولا يعدّل أوزانها ولا معادلاتها. هو معطّل افتراضياً (`hnc_enabled: false`). لا توجد هنا أي نتيجة تجريبية حتى الآن.

**Status:** design, implementation and protocol only. **No experiment has been run and no result exists.** None of these files is a paper.

| File | Content |
| --- | --- |
| [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) | Research questions and scope |
| [HYPOTHESES.md](HYPOTHESES.md) | Testable, falsifiable hypotheses with decision rules |
| [ARCHITECTURE.md](ARCHITECTURE.md) | HNC design, its interfaces to ANSG, what it is and is not |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Conditions A–D, scenarios, generation, annotation, human evaluation |
| [METRICS.md](METRICS.md) | Operational metric definitions: deterministic, proxy, human |
| [ABLATION_PLAN.md](ABLATION_PLAN.md) | Ablations and the factorial design |
| [THREATS_TO_VALIDITY.md](THREATS_TO_VALIDITY.md) | Threats and mitigations |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | How to rerun everything; what to archive |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | Preliminary review and novelty assessment (no novelty claimed) |
| [scenarios/](scenarios/) | Original hierarchical plans for Aqaba, Alexandria, Kuwait City and Fes (draft, unreviewed) |

**Code:**
- `src/ansg/hnc/`: model, validation, controller, realize, metrics, integration.
- `scripts/hnc_generate.py` and `configs/hnc/default.yaml`.

**Tests:**
- `tests/unit/test_hnc.py`;
- `tests/integration/test_hnc_runtime.py`.

**Separation rule:**
- ANSG-Original reproduction work is tracked in `docs/THESIS_TO_CODE_TRACEABILITY.md`.
- Nothing in this folder may be cited as a dissertation result, and no HNC number may enter ANSG-Original result tables.
