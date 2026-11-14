# Dialect resource audit — 2026-11-14

**Status:** resource metadata only. No copyrighted stories, lyrics, tweets, MADAR sentences or modern novel passages were copied into this repository. These resources are candidates for controlled dialect evaluation/adaptation of the existing **DAL**, not extra ANSG thesis components.

## Priorities

1. **MADAR** ([official dataset](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/)): 25 city dialects, including Amman, Salt, Alexandria, Cairo and Fes. Strong for sentence-level control and held-out evaluations; weak for long narrative coherence. **Official license** grants internal research/evaluation only; it forbids redistribution and modification without permission. No contents will be committed.
2. **NADI** ([official shared task](https://nadi.dlnlp.ai/)): country/province dialect identification; check edition, download agreements and train/test isolation.
3. **QADI** ([dataset card](https://huggingface.co/datasets/Abdelrahman-Rezk/Arabic_Dialect_Identification)): 540k tweets across 18 countries; classify geographic language, not long-form storytelling.
4. **DART** ([LREC 2018](https://www.lrec-conf.org/proceedings/lrec2018/summaries/1013.html)): about 25k dialect-annotated tweets, **not** dramatic scripts or stories.
5. **PADIC** ([Arabic NLP Observatory](https://sites.aub.edu.lb/arabicnlp/tools/)): parallel dialect resources, with access and licensing still requiring verification.
6. **Habibi** ([LREC 2020](https://aclanthology.org/2020.lrec-1.165/)): song lyrics, **not** folk narratives. Avoid full-text model training or redistribution without confirmed permissions.
7. **Masader** ([registry](https://arbml.github.io/masader/)): large searchable index; individual dataset licenses control their contents.

## Not cleared

- Wattpad, modern published novels, scripts and subtitles: publication online is **not** a license to copy, redistribute or train. Negotiate direct author permission and verify publisher rights.
- A supposed exact **Shami Corpus** for open licensed long-form Levantine stories remains unverified; require persistent research link.
- No confirmed locally appropriate *Aqaba/Jordanian southern* narrative corpus. **MADAR Amman or Salt is not a substitute for Aqaba dialect**.

## Scientific separation

Keep: `ANSG-original` data/protocol for thesis verification; `classical_arabic` for independent heritage narrative pilot; `dialect_sources` metadata for existing DAL evaluation.

A future dialect corpus should carry `license_id`, `permission_scope`, `corpus_version`, `language_variety`, `country`, `city`, `genre`, `narrator_style`, `split` and SHA256. Dialogue-level labels must be distinguished from narration text. Human native-speaker evaluations are needed for dialect naturalness. This audit does not validate new dialect performance or claim any result of the dissertation.
