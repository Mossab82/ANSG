# Is `data/corpora/classical_arabic` the thesis's ANHC? — No.

**Measurements:** `evidence/corpus/classical_corpus_audit.json`, regenerated with `python scripts/reproduction/audit_classical_corpus.py`.
**Source checks:** CI evidence run 43564658606 (`evidence/external/external_evidence.json`).

| Property | Thesis ANHC (claimed) | Repository corpus (measured) |
| --- | --- | --- |
| Size | 70,404,829 tokens (A.20, p. 289) | 195,197 whitespace words, about **0.28%**. Units differ: tokens vs words |
| Units | 55,600 or 5,247 narratives (conflict D-DATA-1) | 20 Wikisource pages; 31 automatically segmented stories |
| Alf Layla | Mahdi 1984 critical edition (8.2M tokens) plus Būlāq 1835 and Calcutta II witnesses (§3.2.2–3.2.4, p. 57–58) | One unnamed Wikisource transcription; 4 parts plus the Sinbad page; 162,029 words; nights 20–130, 531–560, 983–1001 (147 nights) |
| Kalīla wa-Dimna | 3.1M tokens; edition unspecified | 15 chapters, 33,168 words. Wikisource categorises the pages as "نصوص مجهولة المصدر" (texts of unknown source) |
| Classical adab (Maqāmāt, Aghānī, Jāḥiẓ: 13.3M tokens) | yes | absent |
| Modern (18.3M) and contemporary (27.5M) material | yes | absent |
| Dialects (Table 3.3) | 5 varieties, 59.2% non-MSA | none: classical prose; DIA unannotated |
| Annotations | DIA / MOTIF / MORAL tri-layer on 23M tokens | none |
| Splits | 80/10/10 with TF-IDF + DBSCAN clusters, seed 42 (§5.4.4) | none for ANHC. The supplementary pilot uses split v2 (page groups) |
| Text reproducibility | unavailable | **20/20 pages reproduce byte for byte** from their pinned revision ids (CI evidence) |
| Licence | CC BY-NC-SA 4.0 per A.20.2; not obtainable | Wikisource site licence CC BY-SA 4.0. The underlying texts are pre-modern, but the edition is unknown. **Owner review pending.** |

**Conclusion:**
- The repository corpus is a small, public, classical-prose collection.
- It cannot stand in for ANHC in any reproduction of Chapter 6. It must never be called ANHC.
- It is registered as a **supplementary experiment** resource (`wikisource_classical_arabic_pages`).
- Ali Baba does not occur in it (0 mentions of "علي بابا").

## Story-level records

`data/corpora/classical_arabic/story_records.json` holds 31 records. Each record has:
- the pinned revision and permalink;
- the block span and SHA-256 of the story text;
- boundary evidence;
- nesting evidence;
- dialect `null` / `unannotated`;
- split group;
- rights.

No text is duplicated in this file. Main limits:

- **Nesting.** All tale headings in the Nights pages are level 2, so no frame/embedded relation is asserted. Textual cues are kept only as candidates. One of them (Merchant and Jinni under Donkey and Bull) is known to be wrong, which is why cues are never accepted automatically.
- **Unassigned text.** Text before a page's first title continues the previous page's tale (part 2: 381 words; part 3: 3,711 words). It is reported per page, not silently dropped.
- **Sinbad's voyages.** Only the 2nd and 5th voyages have headings, so the other voyages stay inside neighbouring segments.
- **Kalīla chapters.** Each page is treated as one chapter unit. Embedded fables are not marked in the source and are not extracted.
