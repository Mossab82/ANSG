"""ANSG — Automated Arabic Narrative Story Generation.

Research implementation built in phases. Phase 1 (this package state) provides:

* ``ansg.data``       — Arabic normalisation, segmentation, One Thousand and One
  Nights ingestion, ANHC annotated-record schema, deterministic splitting.
* ``ansg.knowledge``  — ArabicVerbNet XML parser, ArabicNameNet adapter,
  Cultural Knowledge Graph (CKG) interface for CSAM.
* ``ansg.evaluation`` — CRAFT *interfaces only* (no metric is implemented yet;
  see ``docs/CRAFT_EVALUATION_SPECIFICATION.md``).

Nothing in this package is a historical artefact of the dissertation's
original experiments. See ``docs/IMPLEMENTATION_GAPS.md``.
"""

__version__ = "0.1.0.dev0"
