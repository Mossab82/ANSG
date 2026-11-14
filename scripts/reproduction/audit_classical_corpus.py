#!/usr/bin/env python3
"""Is data/corpora/classical_arabic the thesis's ANHC? Measure it against the thesis's own claims.

Deterministic, offline, stdlib + ansg. Output: evidence/corpus/classical_corpus_audit.json.

    python scripts/reproduction/audit_classical_corpus.py           # write + print summary
    python scripts/reproduction/audit_classical_corpus.py --check   # fail if the committed JSON is stale
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CORPUS = ROOT / "data" / "corpora" / "classical_arabic"
OUT = ROOT / "evidence" / "corpus" / "classical_corpus_audit.json"
EVIDENCE = ROOT / "evidence" / "external" / "external_evidence.json"

# Thesis claims (Tesis_Ibrahim; pages from PDF footers)
# Split of the pilot as merged in PR #25 (kept to document its leakage).
PILOT_SPLIT_V1 = {
    "train": {"arabian_nights-001", "arabian_nights-002", "arabian_nights-003", "arabian_nights-004",
              "kalila_dimna-006", "kalila_dimna-007", "kalila_dimna-008", "kalila_dimna-009", "kalila_dimna-010",
              "kalila_dimna-011", "kalila_dimna-012", "kalila_dimna-013", "kalila_dimna-014", "kalila_dimna-017"},
    "validation": {"kalila_dimna-015", "kalila_dimna-016", "kalila_dimna-018"},
    "test": {"kalila_dimna-019", "kalila_dimna-020"},
}
THESIS = {
    "anhc_total_tokens": {"value": 70_404_829, "ref": "App. A.20 Table A.17, p. 289 (70.4M in §3, Table 3.2 p. 56)"},
    "anhc_narratives": {"value": [55_600, 5_247], "ref": "A.20 Table A.17 p. 289 (55,600) vs §5.4.4 p. 143 (5,247)"},
    "alf_layla_mahdi_tokens": {"value": 8_200_000, "ref": "§3.2.2, p. 57 ('Mahdi critical edition provides 8.2M tokens')"},
    "kalila_tokens": {"value": 3_100_000, "ref": "§3.2.2, p. 57"},
    "classical_tokens": {"value": 24_600_000, "ref": "Table 3.2, p. 56"},
    "nights_witnesses": {"value": ["Bulaq 1835", "Calcutta II 1839-42", "Mahdi 1984"], "ref": "§3.2.4, p. 58"},
}


def containment(train_texts, test_text, n=8):
    from ansg.data.leakage import ngram_overlap
    return ngram_overlap(train_texts, {"x": test_text}, n=n).per_document[0]["containment"] or 0.0


def build() -> dict:
    rows = [json.loads(l) for l in (CORPUS / "stories.jsonl").read_text("utf-8").splitlines() if l.strip()]
    by = {r["id"]: r for r in rows}
    words = {k: len(r["text"].split()) for k, r in by.items()}
    nights_ids = [k for k in by if by[k]["corpus"] == "arabian_nights"]
    kalila_ids = [k for k in by if by[k]["corpus"] == "kalila_dimna"]
    total = sum(words.values())

    pairs = []
    for a in by:
        for b in by:
            if a != b:
                c = containment([by[b]["text"]], by[a]["text"])
                if c > 0.02:
                    pairs.append({"contained": a, "in": b, "containment_8gram": round(c, 6)})

    spec = importlib.util.spec_from_file_location("vcc", ROOT / "scripts" / "validate_classical_corpus.py")
    vcc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vcc)
    def contamination(train, splits):
        texts = [by[i]["text"] for i in sorted(train)]
        res = {name: {i: round(containment(texts, by[i]["text"]), 6) for i in sorted(ids)} for name, ids in splits}
        return res, sorted(i for split in res.values() for i, c in split.items() if c > 0.5)
    pilot_v1, leaked_v1 = contamination(PILOT_SPLIT_V1["train"], (("validation", PILOT_SPLIT_V1["validation"]),
                                                                  ("test", PILOT_SPLIT_V1["test"])))
    pilot, leaked = contamination(vcc.TRAIN, (("validation", vcc.VALIDATION), ("test", vcc.TEST)))

    sinbad = "arabian_nights-005"
    sinbad_c = containment([by[i]["text"] for i in nights_ids if i != sinbad], by[sinbad]["text"])
    ali_baba = sum(len(re.findall("علي بابا", r["text"])) for r in rows)

    coverage = {}
    seg_path = CORPUS / "story_segments.json"
    if seg_path.exists():
        segs = json.loads(seg_path.read_text("utf-8"))["segments"]
        for s in segs:
            coverage.setdefault(s["source_record_id"], set()).update(s.get("nights") or [])
    nights_covered = sorted(set().union(*coverage.values())) if coverage else []
    anomalies = {}
    if EVIDENCE.exists():
        for p in json.loads(EVIDENCE.read_text("utf-8"))["wikisource"]["pages"]:
            seq = p.get("night_sequence")
            if seq and (seq.get("regressions") or seq.get("gaps")):
                anomalies[p["id"]] = {"regressions": seq["regressions"][:20], "gaps": seq["gaps"][:20]}

    nights_words = sum(words[i] for i in nights_ids)
    kalila_words = sum(words[i] for i in kalila_ids)
    return {
        "schema": "ansg.classical_corpus_audit/1",
        "corpus": "data/corpora/classical_arabic (20 Wikisource pages, pinned revisions)",
        "thesis_claims": THESIS,
        "measured": {
            "records": len(rows), "words_total": total,
            "words_alf_layla_pages": nights_words, "words_kalila_pages": kalila_words,
            "unit_note": "whitespace words; the thesis reports tokens after preprocessing with an unspecified "
                         "tokenizer, so ratios below are indicative only",
        },
        "ratios": {
            "corpus_words_over_anhc_tokens_pct": round(100 * total / THESIS["anhc_total_tokens"]["value"], 4),
            "alf_layla_words_over_claimed_mahdi_tokens_pct": round(100 * nights_words / 8_200_000, 3),
            "kalila_words_over_claimed_kalila_tokens_pct": round(100 * kalila_words / 3_100_000, 3),
        },
        "nights_covered_by_segments": {"count": len(nights_covered),
                                       "ranges": _ranges(nights_covered)},
        "night_sequence_anomalies": anomalies,
        "duplicate_text_pairs_8gram_gt_2pct": pairs,
        "pilot_split_v1_contamination_8gram": pilot_v1,
        "pilot_split_v1_leaked_pages": leaked_v1,
        "pilot_split_contamination_8gram": pilot,
        "pilot_split_leaked_pages": leaked,
        "sinbad_page_containment_in_other_nights_pages": round(sinbad_c, 6),
        "ali_baba_mentions": ali_baba,
        "verdict": {
            "equivalent_to_anhc": False,
            "reasons": [
                "size: about 0.28% of the claimed ANHC token count",
                "sources: an unnamed Wikisource edition of selected Alf Layla parts and Kalila chapters "
                "(Wikisource category 'texts of unknown source' on the Kalila pages); none of the three "
                "witnesses named in §3.2.4, no Maqamat/Aghani/Jahiz, no modern or contemporary material",
                "no dialectal text and no DIA/MOTIF/MORAL annotations",
                "page records are not narratives: tales cross page boundaries (part 1 ends mid-tale)",
            ],
        },
    }


def _ranges(nums):
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append([start, prev])
            start = prev = n
    if start is not None:
        out.append([start, prev])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    audit = build()
    payload = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text("utf-8") != payload:
            print(f"{OUT} is stale: rerun scripts/reproduction/audit_classical_corpus.py", file=sys.stderr)
            return 1
        print("classical corpus audit is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload, encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("measured", "ratios", "pilot_split_v1_leaked_pages", "pilot_split_leaked_pages",
                                            "sinbad_page_containment_in_other_nights_pages", "ali_baba_mentions",
                                            "nights_covered_by_segments")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
