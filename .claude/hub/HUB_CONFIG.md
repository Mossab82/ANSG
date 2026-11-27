# Personal AI Hub — إعداد Claude

دور Claude في المنظومة: **المنفذ التقني المساند** (مراجعة الكود وتنفيذ التعديلات المصرح بها،
ومراجعة النتائج البحثية مقابل المصادر، وسد الفجوات عند الطلب فقط). ChatGPT هو المتابع اليومي
والكاتب الرئيسي للسجلات؛ Gemini للمراجعة الأسبوعية التكميلية.

## مواقع Google Drive (للقراءة)

| العنصر | المعرّف |
|---|---|
| المركز الرئيسي | `1j7yil0b7bC7MnL8sqwvh9hwBuSPkDbB3` |
| 06 - Agent Instructions | `1APPI8Kscbbmh6uLMYWQCYMSHLBAuDhfJ` |
| 02 - Career Agent / Career_Tracker.xlsx | `1Q1kTumX0zSVittYWNrePGIqa5siuI4a-` / `1uWOI4rN2fr8-11T5FUvtbdDEOANdEFOu` |
| 04 - Academic Agent / Academic_Tracker.xlsx | `1wBr8urzmWs7ohzgvr0AjTSjoaVK25DNM` / `1-x7uiEr--TX7Xs4Dr3RbJZupTeoQvVg7` |
| 05 - Dashboard and Logs / Master_Dashboard.xlsx | `1fvOST9djMW5XQKMjQMmb3qV_SnXNuQ3e` / `1wKJmLxxbN9zfEtHvfkXx7fDXoFFbGosU` |
| 03 - Education Agent - Non-sensitive | `19m3T7eEPh7_lnN5fzxLpSdgp83K86z3K` (قراءة أسماء فقط ما لم يطلب المستخدم ملفًا محددًا) |

## ممنوع الوصول

- `01 - Health Agent - Private` (`1NodvHOUozcfkGQ3gEwfp2ohUeArYczXQ`) و`Health_Agent.md`: لا قراءة ولا بحث داخلها.
- أي بيانات طلبة أو موظفين أو أسرار أنظمة (AqabaIMS وغيره). اسم المجلد ليس آلية تحكم في الوصول.

## أعمدة السجلات (للالتزام بها في التقارير)

- **Career_Tracker**: تاريخ التحقق، الوظيفة، الجهة، الدولة / عن بعد، الرابط الرسمي، الموعد النهائي، الراتب المعلن، التطابق /100، الحالة، الخطوة التالية.
- **Academic_Tracker**: تاريخ التحقق، الفرصة / البحث، النوع، الجهة، الرابط الرسمي، الموعد، التمويل / الرسوم، الأهلية، الحالة، الخطوة التالية.
- **Master_Dashboard**: الوكيل، الدورية، آخر تشغيل مؤكد، الحالة، المهام المفتوحة، الخطوة التالية.

## قواعد الكتابة

- Claude **لا يكتب** في السجلات المشتركة؛ ينتج تقارير مستقلة بصيغة صفوف جاهزة للنسخ.
- أي كتابة إلى Drive (مثل تقرير تجريبي في Dashboard and Logs) تتم فقط بعد موافقة صريحة في الجلسة نفسها، وبملف جديد مستقل، دون الكتابة فوق ملف قائم.
- لا API مدفوعة ولا خدمات بالأرصدة (Firecrawl، TinyFish ممنوعة في `.claude/settings.json`). البحث عبر WebSearch/WebFetch المضمّنة في الاشتراك فقط.
- لا إرسال بريد أو طلبات أو مشاركة ملفات.
