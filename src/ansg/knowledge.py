"""CKG data structures and deterministic context-conditioned schema retrieval."""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Motif:
    id: int
    label: str
    category: str
    keywords: list[str] = field(default_factory=list)
    relations: list[int] = field(default_factory=list)

class CulturalKnowledgeGraph:
    def __init__(self, motifs):
        self.motifs = {m.id: m for m in motifs}
        if len(self.motifs) != len(motifs):
            raise ValueError("Duplicate motif identifiers")
        for m in motifs:
            for related in m.relations:
                if related not in self.motifs:
                    raise ValueError(f"Unknown motif relation: {related}")

    @classmethod
    def from_json(cls, path):
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls([Motif(**entry) for entry in raw['motifs']])

    @staticmethod
    def _tokens(text):
        return set(re.findall(r'[\w\u0600-\u06ff]+', text.casefold(), flags=re.UNICODE))

    def activate(self, text, top_k=8, min_score=0.0):
        """Transparent lexical overlap BASELINE, not thesis embedding-based CKG retrieval."""
        tokens = self._tokens(text)
        scored = []
        for motif in self.motifs.values():
            terms = self._tokens(' '.join([motif.label] + motif.keywords))
            score = len(tokens & terms) / math.sqrt(max(1, len(terms)))
            if score > min_score:
                scored.append((motif.id, float(score)))
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:top_k]
