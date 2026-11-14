# OpenITI heritage sources: metadata only, status `pending`

The audit report (in Arabic) is [docs/ar/OPENITI_HERITAGE_SOURCE_AUDIT.md](../../../docs/ar/OPENITI_HERITAGE_SOURCE_AUDIT.md).

| File | Content |
| --- | --- |
| `openiti_metadata_evidence.json` | Raw evidence collected in CI by `.github/workflows/openiti-metadata-evidence.yml` and `scripts/openiti/collect_openiti_metadata.py`. It holds Zenodo release records, the RELEASE-tag metadata rows, the pinned KITAB catalogue rows, the repository commits, the version `.yml` fields and SHA-256 fingerprints. |
| `openiti_heritage_registry.json` | The registry, built offline by `scripts/openiti/build_openiti_heritage_registry.py` (`--check` verifies it is current). |

- **No text.** This folder contains no OpenITI text. The release archive was not downloaded, and texts were hashed in memory only.
- **Not ANHC.** These are candidate sources for the classical adab family named in the thesis (§3.2.2, p. 57). They are not the ANHC corpus and do not reproduce it.
- **Rights.** The releases are licensed CC BY-NC-SA 4.0. Every decision on use, redistribution and admission stays `pending` until the owner reviews it.
