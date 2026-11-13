"""Generation modes, readiness levels and the visible labels attached to every output.

Torch-free. The label of a generation is *derived* from verified evidence
(backbone provenance, cultural-module manifest); there is deliberately no way
for a caller to set it directly, so an untrained or synthetic model cannot be
presented as dissertation-level ANSG by a command-line flag.

Nothing produced by this package reproduces a dissertation result: every
record carries ``reproduces_dissertation = False`` unconditionally.
"""

from __future__ import annotations

from enum import Enum

RECORD_SCHEMA = "ansg.generation_record/1"
"""Schema tag of the JSON record written by ``ansg-generate``."""


class ReadinessLevel(str, Enum):
    """What the available evidence supports, weakest first (Issue #15, Deliverable 1)."""

    SMOKE_ONLY = "smoke_only"
    """Only tiny untrained/synthetic checkpoints (byte-level research model, synthetic stub)."""
    BACKBONE_ONLY = "backbone_only"
    """A verified pretrained Arabic causal LM with declared licence/provenance; no ANSG modules."""
    ANSG_UNTRAINED = "ansg_untrained"
    """ANSG cultural modules (CSAM/DAL) initialised but untrained."""
    ANSG_TRAINED_SYNTHETIC_ONLY = "ansg_trained_synthetic_only"
    """Cultural modules were optimised, but only on data declared synthetic."""
    TRAINED_ANSG = "trained_ansg"
    """Cultural modules trained on data declared real, with hashes and compatibility verified."""

    @property
    def rank(self) -> int:
        return list(ReadinessLevel).index(self)


class GenerationMode(str, Enum):
    """How a particular generation was produced."""

    SYNTHETIC_STUB = "synthetic_stub"
    BACKBONE_ONLY = "backbone_only"
    ANSG_UNTRAINED = "ansg_untrained"
    ANSG_TRAINED_SYNTHETIC_ONLY = "ansg_trained_synthetic_only"
    ANSG_TRAINED = "ansg_trained"


MODE_TO_LEVEL = {
    GenerationMode.SYNTHETIC_STUB: ReadinessLevel.SMOKE_ONLY,
    GenerationMode.BACKBONE_ONLY: ReadinessLevel.BACKBONE_ONLY,
    GenerationMode.ANSG_UNTRAINED: ReadinessLevel.ANSG_UNTRAINED,
    GenerationMode.ANSG_TRAINED_SYNTHETIC_ONLY: ReadinessLevel.ANSG_TRAINED_SYNTHETIC_ONLY,
    GenerationMode.ANSG_TRAINED: ReadinessLevel.TRAINED_ANSG,
}

LEVEL_DESCRIPTIONS = {
    ReadinessLevel.SMOKE_ONLY: "smoke only: tiny untrained or synthetic checkpoints; output is not Arabic narrative",
    ReadinessLevel.BACKBONE_ONLY: "backbone-only: verified pretrained Arabic causal LM, no ANSG cultural modules",
    ReadinessLevel.ANSG_UNTRAINED: "ANSG components initialised but untrained (cultural conditioning not meaningful)",
    ReadinessLevel.ANSG_TRAINED_SYNTHETIC_ONLY: "ANSG components trained on synthetic data only (pipeline check, no evidence)",
    ReadinessLevel.TRAINED_ANSG: "trained ANSG checkpoint: cultural modules trained on declared real data, hashes verified",
}

BANNERS = {
    GenerationMode.SYNTHETIC_STUB:
        "SYNTHETIC STUB — tiny randomly initialised test model; the text is meaningless test output.",
    GenerationMode.BACKBONE_ONLY:
        "BACKBONE-ONLY GENERATION — pretrained Arabic language model WITHOUT ANSG cultural modules "
        "(no CSAM, no DAL). This is not ANSG output and not a dissertation result.",
    GenerationMode.ANSG_UNTRAINED:
        "UNTRAINED RESEARCH PROTOTYPE — CSAM/DAL are randomly initialised. Cultural and dialectal "
        "conditioning is NOT meaningful. Not dissertation-level ANSG; not a dissertation result.",
    GenerationMode.ANSG_TRAINED_SYNTHETIC_ONLY:
        "RESEARCH PROTOTYPE — cultural modules were trained on SYNTHETIC data only. No cultural or "
        "dialectal capability is claimed. Not a dissertation result.",
    GenerationMode.ANSG_TRAINED:
        "ANSG with trained cultural modules (training provenance recorded in this file). This is a new "
        "generation, not a reproduction of the dissertation's reported results.",
}


def classify_mode(*, backbone_synthetic: bool, cultural_training: dict | None, cultural_loaded: bool) -> GenerationMode:
    """Derive the mode from evidence. ``cultural_training`` is the ``training`` block of a
    validated cultural-module manifest (``None`` for freshly initialised modules)."""
    if backbone_synthetic:
        return GenerationMode.SYNTHETIC_STUB
    if not cultural_loaded:
        return GenerationMode.BACKBONE_ONLY
    if not cultural_training or not is_trained(cultural_training):
        return GenerationMode.ANSG_UNTRAINED
    if cultural_training.get("data_kind") != "real":
        return GenerationMode.ANSG_TRAINED_SYNTHETIC_ONLY
    return GenerationMode.ANSG_TRAINED


def is_trained(training: dict) -> bool:
    """A manifest counts as trained only with status, positive steps and data hashes."""
    steps = training.get("optimizer_steps")
    return (training.get("status") == "trained"
            and isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
            and bool(training.get("data_sha256")))


def label_block(mode: GenerationMode, *, dialect_requested: bool = False) -> dict:
    """The label written into every generation record and printed to stderr."""
    level = MODE_TO_LEVEL[mode]
    warnings = []
    if dialect_requested and mode is not GenerationMode.ANSG_TRAINED:
        warnings.append("A dialect id was passed to a DAL that is not trained on real data: "
                        "no dialect adaptation is claimed.")
    return {
        "mode": mode.value,
        "readiness_level": level.value,
        "description": LEVEL_DESCRIPTIONS[level],
        "banner": BANNERS[mode],
        "cultural_conditioning_claimed": mode is GenerationMode.ANSG_TRAINED,
        "dialect_adaptation_claimed": mode is GenerationMode.ANSG_TRAINED and dialect_requested,
        "dissertation_result": False,
        "reproduces_dissertation": False,
        "warnings": warnings,
    }


def validate_label(label: dict, *, backbone_synthetic: bool, cultural_loaded: bool,
                   cultural_training: dict | None) -> None:
    """Reject a label stronger than its evidence (guards hand-edited or stale JSON)."""
    expected = classify_mode(backbone_synthetic=backbone_synthetic, cultural_training=cultural_training,
                             cultural_loaded=cultural_loaded)
    if label.get("mode") != expected.value:
        raise ValueError(f"label mode {label.get('mode')!r} is not supported by the evidence "
                         f"(expected {expected.value!r})")
    if label.get("reproduces_dissertation") is not False or label.get("dissertation_result") is not False:
        raise ValueError("no generation record may claim to reproduce a dissertation result")
    if label.get("cultural_conditioning_claimed") and expected is not GenerationMode.ANSG_TRAINED:
        raise ValueError("cultural conditioning claimed without a trained cultural-module checkpoint")
    if label.get("dialect_adaptation_claimed") and expected is not GenerationMode.ANSG_TRAINED:
        raise ValueError("dialect adaptation claimed without a trained DAL")
