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

These directories are targets, **not** a claim that their modules already exist.

## Specification status

The dissertation contains differing experimental specifications that need verification before selecting the executable configuration:

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
