# ANSG — Automated Arabic Narrative Story Generation

Research development workspace for a thesis-grounded, reproducible implementation of **Automated Arabic Narrative Story Generation (ANSG)**.

> **Status:** implementation and verification in progress. Code currently in the root of this repository is preserved as legacy Arabic NLP research code and has **not** been established as the full ANSG architecture described in the dissertation.

## Research architecture

ANSG is described as a hybrid neural-symbolic framework integrating:
- Arabic narrative preprocessing and morphological conditioning
- Cultural Knowledge Graph (CKG) and schema activation
- Cultural Schema Attention Mechanism (CSAM)
- Dialectal Adaptation Layers (DAL / LoRA)
- Cultural alignment and reinforcement learning
- Reproducible evaluation of cultural authenticity and linguistic quality

The source of architectural requirements is the dissertation *Automated Arabic Narrative Story Generation (ANSG): A Hybrid Neuro-Symbolic Architecture with Culturally-Grounded Reinforcement Learning*, especially Chapters 3–6 and Appendix A.

## Planned layout

```text
src/ansg/data/
src/ansg/knowledge/
src/ansg/models/
src/ansg/training/
src/ansg/evaluation/
configs/
tests/
docs/
scripts/
```

As of Phase 1, `src/ansg/data/`, `src/ansg/knowledge/`, `src/ansg/evaluation/` (interfaces only), `configs/`, `tests/`, `docs/` and `scripts/` exist. `models/` and `training/` are still targets, **not** a claim that those modules exist.

## Phase 1: data and knowledge infrastructure

Implemented and tested on CPU (no model training):

- Offset-preserving Arabic normalisation (lossless defaults) and segmentation
- *One Thousand and One Nights* ingestion: tale identification, night-number parsing, frame-formula detection, provenance
- ANHC record schema with DIA / MOTIF / MORAL layers, annotation status, copyright-aware export, annotation import
- Deterministic group-level splits (hash-based or predefined partition)
- ArabicVerbNet XML parser, ArabicNameNet configurable adapter (format unverified), CKG + CST-Onto interface for CSAM
- CRAFT interfaces (no metrics implemented)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"          # add ,ontology for CST-Onto RDF loading
pytest
ansg-ingest --config configs/data/nights.yaml   # needs your licensed corpus file under data/private/
python scripts/inspect_resources.py             # reports which knowledge resources are supplied
```

No corpus, lexicon, ontology, annotation or checkpoint is bundled; test fixtures are synthetic (see `tests/fixtures/README.md`). Start with `docs/REPOSITORY_AUDIT.md`, `docs/SPECIFICATION_CONFLICTS.md` and `docs/IMPLEMENTATION_GAPS.md`.

The dissertation's specifications have been transcribed with section references (`docs/EXPERIMENTAL_SPECIFICATIONS.md`, `configs/specification/thesis_variants.yaml`). Where chapters disagree, all versions are kept and numbered in `docs/SPECIFICATION_CONFLICTS.md`; none has been chosen.

## Specification status

The dissertation contains differing experimental specifications that need verification before selecting the executable configuration (summary; the full register with ~60 entries is `docs/SPECIFICATION_CONFLICTS.md`):

| Topic | Reported alternatives |
| --- | --- |
| Backbone | GPT-2 Medium (Chapter 4); AraGPT2-large (Appendix A); 774M base parameters (Chapter 5) |
| Model parameters | 18M trainable modules (Chapter 4); 847M total / 73M additions (Chapter 5) |
| Training phases | Three phases in Chapters 4–5; four stages in Appendix A.26 |
| GPU environment | A100 40GB (Chapter 5); A100 80GB (Appendix A) |
| Reward weights | Three-term reward (Chapter 5); four-term reward (Appendix A.3) |

The aim is to evaluate these assertions programmatically and publish **new** experimental results together with their real execution dates. Do not interpret reproduction as evidence of when historical experiments were conducted.

## Existing code and security

The historical root-level Python scripts predate this new workspace. They include legacy preprocessing, database and Arabic NLP utilities; they have not been audited for current runtime compatibility. **Security alert:** at least some public legacy scripts contain hard-coded credentials or API tokens. Treat them as compromised: rotate/revoke them with their providers, remove them from tracked content *and* relevant public Git history, and replace with environment-variable-based configuration before using them.

## Next steps

1. Audit and secure existing scripts.
2. Establish thesis-to-code traceability.
3. Implement CSAM shape and parameter-count tests.
4. Add CKG, DAL and data pipelines.
5. Evaluate reproducibility before GPU-intensive training.

## License and data

No project license is selected yet. Do not redistribute copyrighted ANHC source texts, non-public annotations, private datasets, or model checkpoints without proper authorization.
