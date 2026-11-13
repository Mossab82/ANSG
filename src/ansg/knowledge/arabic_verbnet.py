"""Parser for ArabicVerbNet class files (VerbNet-style XML).

Format evidence
---------------
The structure parsed here is taken from what the legacy ``Verb-net.py``
actually read from the original resource files (``mm/*.xml``):

* ``MEMBERS/<member>`` elements whose verb form is the ``transname``
  attribute, with child elements ``Root``, ``Deverbal`` (0..n) and
  ``Participle``, each carrying a ``name`` attribute;
* ``FRAMES/<frame>`` elements with ``DESCRIPTION@primary`` and ``SYNTAX``
  children ``NP@value`` and ``PREP@value``.

It additionally supports the standard VerbNet elements that the legacy code
ignored — ``VNCLASS@ID``, ``THEMROLES/THEMROLE`` (+ ``SELRESTRS``),
``EXAMPLES``, ``SEMANTICS/PRED/ARGS/ARG``, ``LEX``, ``VERB``, ``ADJ``/``ADV``
and recursive ``SUBCLASSES/VNSUBCLASS`` — *when present*. Missing elements
yield empty fields; nothing is synthesised.

Differences from the legacy script (fixed here, see docs/IMPLEMENTATION_GAPS.md):
frame parsing was nested inside the member loop and the parsed NP/PREP lists
were discarded; ``if not(d.attrib in syn)`` compared a dict against strings;
the "category" was the file's position in ``glob`` order (non-deterministic
across file systems); bare ``except`` swallowed all errors; database writes
ran at import time.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

_DTD_RE = re.compile(rb"<!(?:DOCTYPE[^>]*\[|ENTITY)", re.I)


class UnsafeXMLError(ValueError):
    """Raised for XML containing inline DTD/entity declarations."""


@dataclass
class SelectionalRestriction:
    value: str
    type: str
    logic: str | None = None


@dataclass
class ThematicRole:
    type: str
    restrictions: list[SelectionalRestriction] = field(default_factory=list)


@dataclass
class VerbMember:
    verb: str | None
    attributes: dict[str, str]
    root: str | None = None
    deverbals: list[str] = field(default_factory=list)
    participles: list[str] = field(default_factory=list)


@dataclass
class SyntaxElement:
    tag: str  # NP, VERB, PREP, LEX, ADJ, ADV, ...
    value: str | None
    restrictions: list[SelectionalRestriction] = field(default_factory=list)


@dataclass
class PredicateArg:
    type: str | None
    value: str | None


@dataclass
class SemanticPredicate:
    value: str
    negated: bool
    args: list[PredicateArg] = field(default_factory=list)


@dataclass
class Frame:
    primary: str | None
    secondary: str | None
    description_number: str | None
    examples: list[str] = field(default_factory=list)
    syntax: list[SyntaxElement] = field(default_factory=list)
    semantics: list[SemanticPredicate] = field(default_factory=list)

    @property
    def noun_phrase_roles(self) -> list[str]:
        return [s.value for s in self.syntax if s.tag == "NP" and s.value]

    @property
    def prepositions(self) -> list[str]:
        return [s.value for s in self.syntax if s.tag == "PREP" and s.value]


@dataclass
class VerbClass:
    class_id: str | None
    source_file: str
    parent_id: str | None
    members: list[VerbMember] = field(default_factory=list)
    roles: list[ThematicRole] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    subclass_ids: list[str] = field(default_factory=list)

    def inherited_roles(self, index: "ArabicVerbNet") -> list[ThematicRole]:
        """Own roles plus those of ancestor classes (VerbNet inheritance)."""
        roles = list(self.roles)
        parent = index.classes.get(self.parent_id) if self.parent_id else None
        while parent is not None:
            roles.extend(r for r in parent.roles if r.type not in {x.type for x in roles})
            parent = index.classes.get(parent.parent_id) if parent.parent_id else None
        return roles


def _safe_parse(path: Path) -> ET.Element:
    data = path.read_bytes()
    if _DTD_RE.search(data):
        raise UnsafeXMLError(f"{path}: inline DTD/ENTITY declarations are not accepted")
    return ET.fromstring(data)


def _children(elem: ET.Element | None, tag: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [c for c in elem if c.tag.upper() == tag.upper()]


def _child(elem: ET.Element, tag: str) -> ET.Element | None:
    found = _children(elem, tag)
    return found[0] if found else None


def _restrictions(elem: ET.Element | None) -> list[SelectionalRestriction]:
    out: list[SelectionalRestriction] = []
    if elem is None:
        return out
    for container in _children(elem, "SELRESTRS") + _children(elem, "SYNRESTRS"):
        logic = container.attrib.get("logic")
        for r in container:
            if r.tag.upper() in {"SELRESTR", "SYNRESTR"}:
                out.append(SelectionalRestriction(r.attrib.get("Value", r.attrib.get("value", "")),
                                                  r.attrib.get("type", ""), logic))
    return out


def _parse_member(e: ET.Element, verb_attributes: tuple[str, ...]) -> VerbMember:
    verb = next((e.attrib[a] for a in verb_attributes if e.attrib.get(a)), None)
    m = VerbMember(verb=verb, attributes=dict(e.attrib))
    for t in e:
        tag = t.tag.lower()
        name = t.attrib.get("name")
        if tag == "root":
            m.root = name
        elif tag == "deverbal" and name:
            m.deverbals.append(name)
        elif tag == "participle" and name:
            m.participles.append(name)
    return m


def _parse_frame(f: ET.Element) -> Frame:
    desc = _child(f, "DESCRIPTION")
    frame = Frame(
        primary=desc.attrib.get("primary") if desc is not None else None,
        secondary=desc.attrib.get("secondary") if desc is not None else None,
        description_number=desc.attrib.get("descriptionNumber") if desc is not None else None,
    )
    for ex_container in _children(f, "EXAMPLES"):
        frame.examples.extend((ex.text or "").strip() for ex in _children(ex_container, "EXAMPLE"))
    for syn in _children(f, "SYNTAX"):
        for s in syn:
            frame.syntax.append(SyntaxElement(s.tag.upper(), s.attrib.get("value"), _restrictions(s)))
    for sem in _children(f, "SEMANTICS"):
        for p in _children(sem, "PRED"):
            args = [PredicateArg(a.attrib.get("type"), a.attrib.get("value"))
                    for args_el in _children(p, "ARGS") for a in _children(args_el, "ARG")]
            frame.semantics.append(SemanticPredicate(
                p.attrib.get("value", ""), p.attrib.get("bool") == "!", args))
    return frame


def _parse_class(
    elem: ET.Element, source: str, parent_id: str | None, verb_attributes: tuple[str, ...]
) -> Iterator[VerbClass]:
    vc = VerbClass(class_id=elem.attrib.get("ID"), source_file=source, parent_id=parent_id)
    for members in _children(elem, "MEMBERS"):
        vc.members.extend(_parse_member(m, verb_attributes) for m in members)
    for roles in _children(elem, "THEMROLES"):
        vc.roles.extend(ThematicRole(r.attrib.get("type", ""), _restrictions(r))
                        for r in _children(roles, "THEMROLE"))
    for frames in _children(elem, "FRAMES"):
        vc.frames.extend(_parse_frame(f) for f in frames)
    subs: list[VerbClass] = []
    for container in _children(elem, "SUBCLASSES"):
        for sub in _children(container, "VNSUBCLASS"):
            subs.extend(_parse_class(sub, source, vc.class_id, verb_attributes))
    vc.subclass_ids = [s.class_id for s in subs if s.parent_id == vc.class_id and s.class_id]
    yield vc
    yield from subs


@dataclass
class ArabicVerbNet:
    classes: dict[str, VerbClass]
    files: list[str]

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | os.PathLike[str]],
        verb_attributes: tuple[str, ...] = ("transname", "name"),
    ) -> "ArabicVerbNet":
        """Parse files in sorted path order (deterministic)."""
        classes: dict[str, VerbClass] = {}
        files = sorted(str(p) for p in paths)
        for fp in files:
            root = _safe_parse(Path(fp))
            top = root
            if root.tag.upper() not in {"VNCLASS", "VNSUBCLASS"}:
                inner = _child(root, "VNCLASS")
                top = inner if inner is not None else root
            for vc in _parse_class(top, Path(fp).name, None, verb_attributes):
                key = vc.class_id or f"{Path(fp).stem}#{len(classes)}"
                if key in classes:
                    raise ValueError(f"duplicate class id {key!r} in {fp}")
                vc.class_id = key
                classes[key] = vc
        return cls(classes=classes, files=files)

    @classmethod
    def from_directory(cls, directory: str | os.PathLike[str], pattern: str = "*.xml",
                       **kw: tuple[str, ...]) -> "ArabicVerbNet":
        paths = list(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no files matching {pattern} in {directory}")
        return cls.from_paths(paths, **kw)

    # ---------------------------------------------------------------- queries
    def members(self) -> Iterator[tuple[VerbClass, VerbMember]]:
        for vc in self.classes.values():
            for m in vc.members:
                yield vc, m

    def classes_for_verb(self, verb: str) -> list[VerbClass]:
        return [vc for vc, m in self.members() if m.verb == verb]

    def verbs_by_root(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for _, m in self.members():
            if m.root and m.verb:
                out.setdefault(m.root, [])
                if m.verb not in out[m.root]:
                    out[m.root].append(m.verb)
        return out

    def stats(self) -> dict[str, int]:
        return {
            "files": len(self.files),
            "classes": len(self.classes),
            "members": sum(len(c.members) for c in self.classes.values()),
            "frames": sum(len(c.frames) for c in self.classes.values()),
            "roles": sum(len(c.roles) for c in self.classes.values()),
            "members_with_root": sum(1 for _, m in self.members() if m.root),
        }

    def legacy_rows(self) -> list[dict[str, str | None]]:
        """Rows shaped like the legacy MySQL ``Arabic_Verbs`` table.

        ``Catagory`` (sic) is the 1-based index of the source file in sorted
        order — the legacy code used unsorted ``glob`` order, so historical
        category numbers are not guaranteed to match.
        """
        file_index = {f: i + 1 for i, f in enumerate(sorted(Path(f).name for f in self.files))}
        rows = []
        for vc, m in self.members():
            row: dict[str, str | None] = {"Verb": m.verb, "Root": m.root,
                                          "Pariciple": m.participles[0] if m.participles else None,
                                          "Catagory": str(file_index[vc.source_file])}
            for i in range(3):
                row[f"Deverbal_{i + 1}"] = m.deverbals[i] if i < len(m.deverbals) else None
            rows.append(row)
        return rows
