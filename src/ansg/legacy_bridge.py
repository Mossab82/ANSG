"""Adapters to reuse legacy Arabic NLP outputs as optional ANSG inputs.

See dissertation Chapter 3 (ANHC annotations) and Chapter 4 (morphology).
This adapter does not assert that legacy token labels are ANHC gold annotations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import re

ARABIC_DIACRITICS = re.compile("[ً-ٰٟ]")
TATWEEL = "ـ"
VALID_DIALECTS = frozenset({"msa", "egyptian", "levantine", "gulf", "maghrebi"})


def normalize_arabic(text: str, *, remove_diacritics: bool = False) -> str:
    """Normalize whitespace and tatweel; optionally strip tashkeel.

    More aggressive hamza/alef normalization is not applied by default because
    normalization policy must be validated against dissertation Appendix A.8.
    """
    text = text.replace(TATWEEL, "")
    if remove_diacritics:
        text = ARABIC_DIACRITICS.sub("", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class NarrativeRecord:
    story_id: str
    text: str
    dialect: str | None = None
    motif_ids: tuple[str, ...] = ()
    moral_labels: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.story_id or not self.text.strip():
            raise ValueError("story_id and text must be non-empty")
        if self.dialect is not None and self.dialect not in VALID_DIALECTS:
            raise ValueError(f"Unknown dialect: {self.dialect}")


def legacy_sql_row_to_record(row: Mapping[str, Any]) -> NarrativeRecord:
    """Parse an extracted story; do not invent cultural annotations.

    Accepts database keys used by old Naitave_Story data or normalized aliases.
    Dataset-wide schema and rights validation must happen before training.
    """
    story_id = row.get("Story_ID", row.get("story_id"))
    title = row.get("Story_Title", row.get("title", ""))
    text = row.get("Story_Text", row.get("text", ""))
    if story_id is None:
        raise ValueError("Missing Story_ID")
    return NarrativeRecord(story_id=str(story_id), text=normalize_arabic(str(text)),
                           metadata={"title": str(title), "source": "legacy_sql"})


def annotated_row_to_record(row: Mapping[str, Any]) -> NarrativeRecord:
    """Load explicit ANHC-style DIA / MOTIF / MORAL annotation fields.

    Does not infer cultural gold labels; corpus protocol needs separate validation.
    """
    story_id = row.get("story_id", row.get("Story_ID"))
    text = row.get("text", row.get("Story_Text"))
    dialect = row.get("DIA")
    motifs = row.get("MOTIF", ())
    morals = row.get("MORAL", ())
    if story_id is None or text is None:
        raise ValueError("Both story_id and text are required")
    if not isinstance(motifs, (list, tuple)) or not isinstance(morals, (list, tuple)):
        raise TypeError("MOTIF and MORAL must be lists or tuples, not raw strings")
    return NarrativeRecord(
        story_id=str(story_id), text=normalize_arabic(str(text)),
        dialect=None if dialect is None else str(dialect).lower(),
        motif_ids=tuple(str(x) for x in motifs),
        moral_labels=tuple(str(x) for x in morals),
        metadata={"source": "explicit_annotations"},
    )
