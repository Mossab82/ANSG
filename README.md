# ANSG — Automated Arabic Narrative Story Generation

### A hybrid neuro-symbolic framework for culturally grounded Arabic storytelling

**ANSG** is a doctoral research framework for generating Arabic narratives that combine neural language generation with explicit cultural knowledge, dialect-aware adaptation, and culturally informed learning and evaluation.

The research investigates a central question: **How can a story-generation system produce fluent Arabic narratives while preserving the cultural motifs, moral frameworks, rhetorical conventions, and dialectal characteristics that give Arabic storytelling its identity?**

**Research area:** Arabic NLP · Computational Creativity · Neuro-Symbolic AI · Knowledge-Grounded Generation

**Dissertation:** *Automated Arabic Narrative Story Generation (ANSG): A Hybrid Neuro-Symbolic Architecture with Culturally-Grounded Reinforcement Learning*

**Researcher:** Mossab Ibrahim  
**Academic institution:** Universidad Complutense de Madrid (UCM)  
**Supervisors:** Gonzalo Méndez and Pablo Gervás

---

## Research overview

ANSG brings together three complementary levels of reasoning:

1. **Neural narrative generation** to model Arabic language, context and story continuity.
2. **Symbolic cultural grounding** to represent and activate relevant narrative schemas, motifs, relationships and moral knowledge.
3. **Cultural and linguistic adaptation** to guide generation toward intended traditions, dialects and narrative qualities.

Rather than treating cultural references as isolated keywords, the framework is designed to connect them to explicit knowledge structures and use those structures during narrative generation.

## Architecture

```text
                     Arabic story prompt / narrative context
                                      |
                                      v
                      Arabic preprocessing and analysis
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
       Cultural Knowledge Graph (CKG)          Dialect and narrative context
                  |                                       |
                  v                                       |
        Cultural schema activation                         |
                  |                                       |
                  v                                       v
       Cultural Schema Attention (CSAM)      Dialectal Adaptation Layers (DAL)
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
                         Neural narrative generator
                                      |
                                      v
                       Culturally grounded Arabic story
                                      |
                                      v
                  Evaluation, cultural analysis and feedback
```

*Conceptual research architecture; the executable integrations and supported configurations are documented in the development branches.*

### Core research components

| Component | Role in ANSG |
| --- | --- |
| **Arabic narrative corpus (ANHC)** | Organizes Arabic storytelling material and annotations, including **DIA** (dialect), **MOTIF** (cultural/narrative motifs) and **MORAL** (moral frameworks). |
| **Cultural Knowledge Graph (CKG)** | Represents cultural concepts, narrative relationships and the schemas used to ground generation. |
| **Cultural Schema Attention Mechanism (CSAM)** | Conditions neural representations on activated cultural schemas and relevant rhetorical/cultural information. |
| **Dialectal Adaptation Layers (DAL)** | Supports dialect-sensitive generation through parameter-efficient adaptation and dialect conditioning. |
| **Culturally grounded learning** | Investigates supervised and reinforcement-learning objectives for narrative fluency, coherence and cultural alignment. |
| **CRAFT and supporting evaluation** | Organizes the research's cultural-fidelity assessment alongside linguistic, narrative and statistical evaluation. |

## Cultural and linguistic resources

The research is informed by Arabic narrative heritage and structured linguistic knowledge:

- **One Thousand and One Nights (Alf Layla wa-Layla):** a major heritage narrative source used for studying traditional story structures, themes and storytelling conventions.
- **Arabic narrative annotations (ANHC):** dialect, motif and moral information used to analyze and condition narratives.
- **Cultural ontologies and knowledge graphs:** representations of cultural schemas and their relations.
- **ArabicNameNet and ArabicVerbNet:** auxiliary named-entity and verb-semantic resources explored for linguistic and knowledge processing; their exact contribution to the ANSG implementation is separately documented.

Dataset editions, licensing, annotation provenance and available exports matter for reproducibility. Source texts and private research datasets are **not automatically distributed** with this repository.

## Intended capabilities

ANSG is designed to support the following research workflows:

**Arabic story generation.** Generate narratives from a prompt while preserving context, continuity and selected cultural constraints.

**Cultural grounding and inspection.** Trace the cultural concepts and schemas activated for a narrative and inspect how they contribute to model conditioning.

**Dialect-sensitive narrative generation.** Study Arabic dialect and register conditioning, including the interaction between narrative context and DAL.

**Controlled comparisons.** Evaluate differences between model variants and component ablations, including cultural attention and dialect adaptation.

**Reproducible evaluation.** Compare outputs using documented linguistic, cultural and statistical measurements while retaining experiment configurations and provenance.

These describe the **research objectives and system architecture**; individual capabilities must be verified against the available model, trained checkpoints and data before being described as a completed demonstration.

## Repository organization

The project is being consolidated from earlier Arabic NLP utilities and independently developed ANSG research components.

| Location | Contents |
| --- | --- |
| [`docs/`](docs/) | Research governance, technical audits, architecture decisions and integration plan |
| [`tools/`](tools/) | Repository integrity, packaging and cross-branch verification tools |
| [`tests/`](tests/) | Automated validation and synthetic integration checks |
| [`src/ansg/`](src/ansg/) | **Data and knowledge layer** (no PyTorch dependency): One Thousand and One Nights ingestion, Arabic normalisation, ANHC records (DIA/MOTIF/MORAL), training-example adapter, ArabicVerbNet/ArabicNameNet adapters, CKG/CST-Onto interfaces, CRAFT specification registry and supported statistics. **Model layer** (`model` extra): CSAM, DAL, the trainable ANSG model, supervised/adversarial/PPO training steps, generation and a pretrained-backbone bridge |
| [`configs/`](configs/), [`scripts/`](scripts/), [`examples/`](examples/) | Data, knowledge, evaluation, model and dissertation-variant configurations; resource inspection, parameter-audit and synthetic end-to-end smoke scripts; synthetic demo inputs |
| `train.py`, `train_pipeline.py`, `evaluate.py` | Training, three-stage pipeline and generation entry points (synthetic demo data by default, labelled as such) |
| [`implementation/ansg-core-2026`](../../tree/implementation/ansg-core-2026) | History of the PyTorch research model, CSAM, DAL and training prototypes |
| [`claude/relaxed-knuth-q63k61`](../../tree/claude/relaxed-knuth-q63k61) | History of the Arabic data ingestion, ANHC records, CKG adapters and evaluation interfaces |
| [`chatgpt/fix-csam-schema-padding`](../../tree/chatgpt/fix-csam-schema-padding) | History of the CSAM masking and batching corrections (merged into the model branch as #9) |

**Development note:** The default branch currently retains the historical scripts and shared governance/integration tooling. Some newer components are in the linked development branches pending compatibility review and integration. The branches are not interchangeable runnable releases. The data and knowledge layer (Issue #6) and the PyTorch model with CSAM masking corrections (Issue #11) form one package; no trained ANSG checkpoint is distributed.

### Quick start

```bash
# Data and knowledge layer only (no PyTorch)
python -m pip install -e ".[test]"
python -m pytest                                # model tests are skipped explicitly
python tools/check_package_topology.py          # must report 0 findings

# Full stack: model, training and generation
python -m pip install -e ".[test,model]"
python -m pytest
python train.py --epochs 1 --output runs/smoke  # synthetic demo data, flagged as synthetic
python scripts/e2e_smoke.py --output runs/e2e   # synthetic: record -> CKG activation -> CSAM/DAL -> generation

# Real data (licensed corpus under data/private/)
ansg-ingest --config configs/data/nights.yaml
ansg-build-examples --records-dir data/processed/nights --ckg <your CKG JSON> --output data/processed/examples
```

Dissertation specifications and their inter-chapter conflicts are transcribed in [`docs/EXPERIMENTAL_SPECIFICATIONS.md`](docs/EXPERIMENTAL_SPECIFICATIONS.md) and [`docs/SPECIFICATION_CONFLICTS.md`](docs/SPECIFICATION_CONFLICTS.md); interfaces to the model are in [`docs/MODEL_INTEGRATION_INTERFACES.md`](docs/MODEL_INTEGRATION_INTERFACES.md).

## Research implementation and verification

This repository is maintained with an emphasis on traceability between **dissertation specifications, source code, test cases, and independently observed experimental results**.

Key engineering practices include:

- Explicit model and experimental configurations rather than silently choosing between alternative thesis specifications.
- Unit tests for tensor dimensions, attention masking, data contracts and generation/training procedures.
- Clear separation of **synthetic software checks** from real-corpus evaluation.
- Experiment provenance through source revisions, configurations, seeds and data/checkpoint identifiers.
- Documented limitations for metrics whose full computational definitions or required reference data are unavailable.

For the detailed engineering record, see:

- [Project governance](docs/PROJECT_GOVERNANCE.md)
- [Integration compatibility audit](docs/INTEGRATION_COMPATIBILITY_AUDIT.md)
- [Integration execution plan](docs/INTEGRATION_EXECUTION_PLAN.md)
- [Security migration notes](docs/legacy_security_migration.md)

### Experimental results

The dissertation reports evaluations of narrative quality, cultural alignment and dialectal behavior. A new repository run is treated as a **new experiment**, not as proof that historic thesis measurements have been reproduced.

Published dissertation results and independently rerun measurements should be compared with their dataset, metric definition, configuration and uncertainty made explicit. We do **not** present synthetic smoke-test outputs or placeholder evaluation scores as verified research results.

## Research and development roadmap

1. **Data foundation:** Arabic heritage ingestion, ANHC records, annotation provenance and CKG interfaces.
2. **Model integration:** connect cultural schema activation to CSAM and dialect conditioning to DAL.
3. **Training and generation:** move from small-scale functionality tests to appropriately configured Arabic language models and licensed training data.
4. **Evaluation:** implement and validate supported metrics, controlled comparisons and ablation studies.
5. **Research demonstration:** provide an auditable Arabic-language interface for story generation, cultural grounding and experimental inspection.

Development progress and review are tracked through [GitHub Issues](../../issues) and [Pull Requests](../../pulls).

## Citation and research use

When referencing this project, please cite the doctoral dissertation:

> Ibrahim, Mossab. *Automated Arabic Narrative Story Generation (ANSG): A Hybrid Neuro-Symbolic Architecture with Culturally-Grounded Reinforcement Learning*. Doctoral dissertation research, Universidad Complutense de Madrid.

The repository contains ongoing research software and should not be construed as an independently verified reproduction of every experiment reported in the dissertation.

## Licensing and responsible use

No general open-source license has been finalized for this repository. Copyright remains with the respective authors and rightsholders. Third-party narrative editions, lexical resources, annotations and checkpoints require their own appropriate usage permissions.

---

**ANSG explores how neural language models and structured cultural knowledge can work together to support authentic, dialect-aware Arabic narrative generation.**
