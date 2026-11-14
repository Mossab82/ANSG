# Rights gate

No data is used for model training until its row in `data/sources/source_registry.json` (generated from `data/sources/build_source_registry.py`) has:
- `permission_status: "verified"`; and
- `"model_training"` in `allowed_uses`.

`scripts/preflight_heritage_pilot.py --require-rights` enforces this, and the GPU workflow runs it before any Kaggle push.

## Current status

| Resource | Status | What is needed |
| --- | --- | --- |
| `wikisource_classical_arabic_pages` (the pilot corpus) | `pending_owner_review` | see the checklist below |
| `madar` | `requires_license_agreement` | a local copy under the CMU/NYUAD licence, internal research and evaluation only. Never redistribute or commit it. Further distribution or modification needs CMU's permission |
| ANHC, CKG, weights, CRAFT, human ratings | `unavailable` | the artefacts themselves (DOIs and repositories return 404) |

## Checklist for the Wikisource pilot corpus

Recorded so far by the evidence workflow:
- site licence CC BY-SA 4.0;
- Kalīla pages categorised "texts of unknown source";
- no licence template on the pages.

Still to decide:

1. **Identify the edition** behind each page, or accept "unknown edition" explicitly.
2. **Confirm the public-domain status** of the underlying text in the relevant jurisdiction(s).
3. **Confirm that training a model on CC BY-SA transcriptions** is acceptable for the intended use and publication. Decide how attribution and share-alike apply to released adapters and outputs.
4. **Record the decision.** Edit the row in `build_source_registry.py`:
   - `permission_status="verified"`;
   - `allowed_uses` including `"model_training"`;
   - a `notes` entry with reviewer, date and scope.

   Then regenerate the registry and commit. CI fails if the JSON and the builder disagree.

Only after step 4 can the GPU workflow pass its preflight.
