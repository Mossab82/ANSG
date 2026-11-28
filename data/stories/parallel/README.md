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
- Batch 12 (parallel-0053–0057) was added in full. The closest test stories were checked by plot. The value الفزعة والتعاون (0055: balancing cargo on a ship) shares the sea setting and ropes with test story 0046 (rescuing a boat near rocks), but the plot differs. The value حفظ الجيرة (0057: tents in a desert storm) differs from test story 0044.
- Batch 13 (parallel-0058–0062) was added in full. The value الرحمة (0062: a girl in a caravan frees a cloud stuck in the sand) shares a common folk motif with test story 0039: a rescued creature later shows the way. The setting, characters and plot differ, and the lexical overlap is near zero, so it was kept.
  Two spelling fixes: 0058 MSA «اسمه وثّابًا» → «اسمه وثّاب»; 0059 Egyptian «دلتّه» → «دلّته».
- Batch 13 Gulf revision: the generator re-sent the five Gulf fields. 0059 and 0061 changed («يديدة»→«جديدة», «مب»→«مو»); 0058, 0060 and 0062 were identical. Each changed record has an `edits` note.
- Gemini batch 1 (parallel-0063–0066, `generated_by: Gemini`). Field names were mapped (levantine_jordanian→levantine, gulf_najdi→gulf, moroccan→maghrebi); `genre`, `setting` and the batch's own id (`batch_id`) are kept. Its story_03 (الفزعة والتعاون: a boat pushed onto harbour rocks, rescued with ropes by other boats) was **not added**, because it nearly duplicates test story parallel-0046. Note: in story_01 the hero is ثامر in MSA but تامر in all dialects, including Najdi Gulf, which normally keeps ث.
- Gemini batch 2 (parallel-0067–0071) was added in full; no plot overlaps a test story. Known issues, kept for now:
  - All versions are 78–94 words, below the requested 110–140.
  - The Gulf versions overuse «ما هو» + participle (e.g. «ما هو متأخر»، «ما هو مغير»، «ما هو مخلوف»), which sounds unnatural in Najdi. The prompt's instruction "use ما هو/مو instead of مب" probably caused this. Revised Gulf fields arrived and were applied to 0067 and 0069–0071 (0068 was unchanged). The only «ما هو» left is the natural «ما هو بالسرعة» in 0067.
  - 0067 (الحلم) retells Aesop's hare and tortoise. Its plot illustrates perseverance more than forbearance, and it treats السلحفاة as masculine.
  - 0069 (الوفاء بالعهد): the captain sails through a storm against the navigator's advice to keep a promise, so the story praises taking a risk with the crew.
