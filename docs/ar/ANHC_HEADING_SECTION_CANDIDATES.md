# مرشحات المقاطع ذات العناوين في ANHC المباشر

أضفنا مستخرجاً حتمياً يلتقط الأسطر العربية المنفردة التي تبدأ بعناوين بنيوية مثل «الفصل»، «الباب»، «الحكاية»، «قصة»، «المشهد»، و«الجزء». يكتب المستخرج إزاحات Unicode نصف مفتوحة إلى النص المصدر وبصمة النص ومعرف مجموعة العمل والتقسيم الموجودين؛ لا ينسخ النص ولا يغيّر التقسيم ولا يضيف أي وسم DIA أو MOTIF أو MORAL.

## نتيجة التشغيل الحالي

- عولجت 951 سجلاً مصدرّياً: 941 مرشحاً من نطاق الكتب السردية الأساسية و10 فصول من كليلة ودمنة.
- ظهرت عناوين بنيوية في 478 سجلاً، نتج منها 8,838 مقطعاً مرشحاً للمراجعة.
- تحتوي المقاطع على 16,976,538 كلمة بحسب الفصل بالمسافات و27,639,186 قطعة من نموذج SentencePiece المباشر، أي 65.49% من قطع النصوص المدخلة عند حساب التقطيع المنفصل للمقاطع.
- تبقى النصوص قبل أول عنوان، والكتب التي لم يظهر فيها عنوان مطابق، ضمن حاويات المصدر. لا تُسقط ولا تُحتسب كأنها قُسمت.
- بقيت تغطية الوسوم الثلاثية البشرية صفرًا. المخرجات ليست 8,838 قصة مؤكدة ولا إعادة إنتاج لحدود الرسالة.

**مراجعة المحتوى شرط قبل اعتماد أي مقطع كقصة.** عنوان «الفصل الأول» يظهر كذلك في كتب غير قصصية، مثل «حقائق الإسلام وأباطيل خصومه»؛ لذلك نوع الصف في المخرجات هو «مقطع بنيوي مرشح»، لا «قصة مؤكدة». ولا يستنتج البرنامج علاقات الحكاية داخل الحكاية.

## إعادة البناء

بعد توليد ملفات المدونة والتقسيم والمرمّز المباشر وفق `data/corpora/anhc_direct_build/release_v1/README.md`:

```bash
python scripts/segment_story_boundary_candidates.py \
  --corpus build/anhc/release_v1/corpora/narrative_candidates/train.jsonl.gz \
  --corpus build/anhc/release_v1/corpora/narrative_candidates/validation.jsonl.gz \
  --corpus build/anhc/release_v1/corpora/narrative_candidates/test.jsonl.gz \
  --split-map build/anhc/release_v1/split_assignments.jsonl \
  --tokenizer-model build/anhc/tokenizer/anhc_direct_spm32.model \
  --output build/anhc/release_v1/story_boundary_candidates.jsonl \
  --report build/anhc/release_v1/story_boundary_candidates_report.json
```

يحفظ JSONL إحداثيات المقطع والعنوان والصف المصدر وبصمته ومجموعة منع التسرب وتقسيم train/validation/test، مع علامات مراجعة المحتوى والحدود. افتراض حدّ فاصل عند كل عنوان بنيوي **خيار فرز مباشر** وليس طريقة TF-IDF/DBSCAN الواردة في الرسالة.

## المصدر

- مصدر الكتب: [The Arabic E-Book Corpus, University of Gothenburg](https://doi.org/10.5878/7rbh-gy93).
- مواصفة الأرقام التاريخية ومقارنة البناء المباشر: [مطابقة ANHC مع الرسالة](ANHC_THESIS_ALIGNMENT.md).
