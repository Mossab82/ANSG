# Example files — SYNTHETIC

`demo.jsonl` and `ckg_demo.json` come from the model branch (`implementation/ansg-core-2026`)
and are **synthetic smoke-test inputs**. They are not ANHC data, not CST-Onto/CKG data and
not annotations from the dissertation. Runs that use them (`python train.py`,
`python train_pipeline.py`) record `"data_label": "synthetic"` in their manifest.

Known limitation (audit I-DATA-5): `demo.jsonl` refers to schema ids 4, 5 and 6, which
`ckg_demo.json` does not define (it has ids 1–3). The ids are only model inputs for the
smoke test; they correspond to no cultural concept.

Real training examples are produced from licensed ANHC split files with
`ansg-build-examples` (`src/ansg/data/training_examples.py`), which keeps missing dialect
and schema labels unknown instead of defaulting them.
