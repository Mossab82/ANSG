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

## Stories not added
- Batch 11, value الكرم (caravan in the desert, a girl named رُبى shares scarce food with a stranger and is rewarded with a shorter route): not added, because its plot nearly duplicates test story parallel-0048. Training on it would leak the test set.
- Batch 11, value العدل (parallel-0052) was added. Its palace-and-puzzle frame resembles test story parallel-0047, but the plot differs (a label puzzle, not a measuring puzzle). One edit: MSA «القاضي نُهى» → «القاضية نُهى», to match the four dialect versions.
