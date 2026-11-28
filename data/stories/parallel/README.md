# Parallel dialect stories

One JSON object per line: the same short story in `msa`, `egyptian`, `levantine`, `gulf` and `maghrebi`, with its value concept.

Records are **synthetic** (`generated_by`) and stay `pending_native_review` until a native speaker checks them.

Uses:
- dialect-adapter (DAL) training data, mixed with real dialect text such as IADD (CC-BY-4.0);
- a fixed evaluation set, because the content is the same across dialects and only the dialect varies.

Before training on model-generated text, check the generator's terms of use.

## Train/test split
- Records with `"split": "test"` are a **reserved test set**. They are never trained on: `train_phase1.py` reports their loss as `parallel_test/<variety>`. They are also excluded when the dialect lexicon (`data/cultural_seeds/lexicons/`) is rebuilt.
- Records without a `split` field are training data.
- Test set (closed at 9 stories): parallel-0039, 0040, 0042, 0043 and 0044–0048. parallel-0041 was moved to train because it nearly duplicates parallel-0031. All later stories are training data.
