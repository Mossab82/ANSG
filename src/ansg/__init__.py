"""ANSG — Automated Arabic Narrative Story Generation.

One package, two layers (Issue #11):

* Data and knowledge layer — **no PyTorch dependency** (``pip install -e .``):
  ``ansg.data`` (Arabic normalisation, segmentation, One Thousand and One Nights
  ingestion, ANHC DIA/MOTIF/MORAL records, splits, training-example adapter),
  ``ansg.knowledge`` (ArabicVerbNet, ArabicNameNet, CKG / CST-Onto, schema
  activation), ``ansg.evaluation`` (CRAFT specification registry, supported
  statistics), ``ansg.utils``.
* Model layer — requires the ``model`` extra (``pip install -e ".[model]"``):
  ``ansg.config``, ``ansg.csam``, ``ansg.dal``, ``ansg.model``,
  ``ansg.training``, ``ansg.pretrained``, ``ansg.provenance``, plus the
  relocated former flat modules ``ansg.data.training_dataset``,
  ``ansg.evaluation.basic_metrics`` and ``ansg.knowledge.lexical_baseline``
  (re-exported lazily from their packages for backward compatibility).

Nothing in this package is a historical artefact of the dissertation's
original experiments, and no output of it is a dissertation result.
"""

__version__ = "0.1.0.dev0"
