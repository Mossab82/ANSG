# حالة نموذج اللغة والتدريب

## ما هو موجود

| الجزء | الموجود في المستودع | دلالته |
|---|---|---|
| نموذج بحثي مستقل | `src/ansg/model.py` | Decoder صغير يدمج CSAM وDAL في نموذج PyTorch قابل للتدريب؛ ليس AraGPT2 ولا يحمل أوزاناً مدرّبة. |
| جسر نموذج مسبق التدريب | `src/ansg/pretrained.py` | يحمّل نموذج Hugging Face ويضيف تكييفاً ثقافياً وDAL بعد الـbackbone عند مدخل رأس الإخراج. لا يحقن CSAM داخل انتباه طبقات GPT، ولذلك لا يطابق وحده معمارية الرسالة. |
| تدريب إشرافي | `scripts/train_cultural_modules.py` | يدرّب الوحدات الثقافية على تنبؤ الرمز التالي مع backbone مجمّد. عند بيانات `real` يشترط اجتياز خطة الجاهزية نفسها ومطابقة مسارات الأمثلة والمفردات والـbackbone والتفويض مع مدخلات CLI. |
| خطوات تدريب منفردة | `src/ansg/training.py` | توجد دوال للتحديث الإشرافي، والمميّز/التحديث الخصومي، وPPO. هي لبنات منفصلة وليست مشغّل مراحل الرسالة. |
| ملف إعادة البناء | `configs/training/ansg_direct_profile.yaml` | يبقي `training_ready: false` و`training_start_permitted: false` بسبب نقص المدخلات وتعارضات الرسالة. |

## ما لم يُبنَ أو يُتحقق منه بعد

- لا يوجد checkpoint أصلي موثق ولا tokenizer التاريخي ذي 32 ألف قطعة.
- لا توجد ملفات ANHC المرخّصة والنسخة الموسومة ذات 23 مليون رمز وتقسيمات الرسالة الأصلية.
- لا يوجد ملف CKG/CST-Onto الأصلي أو معرّفات المخططات اللازمة لبناء أمثلة تدريب الرسالة.
- لا يوجد تشغيل متسلسل موحد للمراحل الثلاث، ولا سجلات أو checkpoints أو نتائج من إعادة تشغيل تاريخية.
- مراحل استخدام بيانات التدريب نفسها متعارضة بين الفصل الثالث والفصل الخامس؛ وتفاصيل الخصومي وCMPI وحدود تطبيع المكافآت غير مكتملة.

- وسم التدريب بأنه `real` يتطلب الآن ملف تفويض بحالة `approved`، وترخيص/سماح لكل مصدر، وبصمات SHA-256 للمصادر والملفات المشتقة. القالب في `configs/training/training_authorization.template.json` حالته `pending` ولا يفتح التدريب.

## فحص الجاهزية قبل التشغيل

أُضيف `scripts/check_training_readiness.py` لقراءة خطة JSON صريحة ومقارنتها بإعداد إعادة البناء وملفاتها:

```bash
python scripts/check_training_readiness.py \
  --plan configs/training/training_run_plan.json \
  --json

python scripts/train_cultural_modules.py --backbone DIR --backbone-sha256 DIGEST \
  --examples data/processed/examples/train.examples.jsonl \
  --schema-vocabulary data/processed/examples/schema_vocabulary.json \
  --data-kind real --authorization-manifest data/private/training_authorization.json \
  --run-plan configs/training/training_run_plan.json \
  --profile configs/training/approved_reconstruction_profile.yaml \
  --steps 2000 --output checkpoints/cultural.pt
```

يفحص الاختيارات المتعارضة والملفات وبصماتها، ويتطلب أدوار الأمثلة المرخّصة وcheckpoint النموذج ومفردات المخططات. وعند `data_kind=real` يشترط ملف تفويض approved وبصمته المثبتة في الخطة، ويتحقق من بصمات المصادر والملفات المشتقة وتطابق مساراتها مع مدخلات التشغيل عبر `scripts/training_authorization.py`. لا يبدأ الفحص التدريب ولا يجيز إعادة إنتاج تاريخية. الخطة المثال في `configs/training/training_run_plan.template.json` معلّقة عمداً؛ ومع الإعداد الحالي ستظل النتيجة `blocked`. وحتى اجتياز الفحص، النطاق الوحيد الذي يغطيه مدرّب موجود هو fine-tuning للوحدات الثقافية، وليس مراحل الرسالة الثلاث.

## مسار الإكمال القابل للتدقيق

1. تثبيت backbone وtokenizer بإصدارات وبصمات وترخيص واضح، وتسوية تعارض اختيار النموذج في C-MODEL-1.
2. إتاحة ملفات ANHC وCKG والوسوم والتقسيمات الفعلية، والتحقق من hashes وحقوق استخدامها.
3. تثبيت التعريفات المتعارضة كخيارات تشغيل معلنة أو استعادتها من مواد الرسالة الأصلية؛ لا يعتمد المشغّل خياراً صامتاً.
4. استكمال حقن CSAM وDAL في طبقات backbone وفق اختيار موثق، ثم وصل مراحل التدريب مع تسجيل البيانات والبذور والمقاييس والأوزان.
5. وسم أي تشغيل جديد بوضوح كتجربة إعادة بناء، لا كإعادة إنتاج تاريخية.

إلى حين تحقق المدخلات، الملف التنفيذي يرفض وصف البناء الحالي بأنه تدريب الرسالة. لم يُشغّل تدريب أو GPU ضمن هذا العمل.
