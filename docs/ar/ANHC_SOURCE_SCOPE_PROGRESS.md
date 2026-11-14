# ANHC source integration, novelty, and build status

## What makes ANHC novel in the thesis

The thesis frames ANHC's novelty as a **combination**: large-scale Arabic narrative text, five-way dialect coverage, three annotation layers (DIA, MOTIF, MORAL), and direct architectural coupling to CSAM, DAL, and culturally grounded RL. Its Chapter 3 source strategy spans three periods and several narrative families; MADAR appears in the dialect-resource landscape, while CAMeL Tools supports morphology-aware preprocessing and root-aware tokenization. These roles matter to the evidence: a toolkit helps prepare and normalize source text, a dialect corpus contributes dialect evidence, and story collections provide narrative content.

The thesis reports these source strata:

| Thesis source stratum | Reported tokens / segments | Named materials |
| --- | ---: | --- |
| Classical, pre-1800 | 24.6M / 296K | Mahdi Alf Layla edition (8.2M), Kalila wa-Dimna (3.1M), classical adab (13.3M) |
| Modern, 1800–2000 | 18.3M / 224K | Mahfouz, Taha Hussein, canonical adab |
| Contemporary, post-2000 | 27.5M / 327K | Blogs, transcribed oral tales, regional collections |
| Total | 70.4M / 847K | Five dialect families and three annotation layers |

MADAR should remain visible in the provenance and novelty map because the thesis positions it as a major dialect resource and reports a CNN-based DIA classifier with 94.3% accuracy on the MADAR benchmark (§5.4.1). Its released examples are translated travel-domain sentences, so they contribute dialect supervision/benchmark evidence, not long-form story plots. The thesis calls the corpus out at 2.1M in its resource comparison; the direct build keeps this separate from the narrative-text count. CAMeL Tools is documented as the morphology/root-analysis toolkit, including CALIMA MSA and Egyptian databases and lexicon extensions; it contributes processing, not corpus tokens.

## What we have locally verified

| Scope | Measure | Meaning |
| --- | ---: | --- |
| Complete Arabic E-Book Corpus | 76,486,597 provider-reported tokens | All 1,745 books; provider-native count, not story-only |
| Same complete corpus | 107,029,217 local SentencePiece pieces | Direct-build SPM32 counter; not comparable one-to-one to the provider count |
| Narrative candidates from that corpus | 82,705,621 local SentencePiece pieces | Includes adjacent categories such as history and biography; boundary review needed |
| Four user-supplied dialect files | 66,247 lines / 1,668,817 pieces | Short dialect evidence; not verified stories and not publicly copied |
| Human-adjudicated triple coverage | 0 tokens | DIA/MOTIF/MORAL coverage must all be adjudicated before it counts |

The locally available Alf Layla files contain six Arabic-original volumes from the Arabic E-Book Corpus; the edition is not established as the thesis's 8.2M-token Mahdi critical edition. Kalila is a separate 1816 Wikisource edition; it is not the 3.1M-token source family described by the thesis. We preserve these as transparent direct-build sources and do not pass them off as exact historical editions.

The Jordanian Layla Witheeb collection is a useful supplementary contemporary dialect-story source: 109 speakers read and retell one adapted story. Its repetitions form one story-cycle group. It is recorded for text-only intake, but no count is reported until its transcripts are acquired and inspected.

## Evidence and completion status

The current build demonstrates comparable corpus scale and usable classical/dialect resource plumbing, but does not yet reconstruct every thesis-named edition or its historical source manifest. The thesis reports 70,404,829 tokens and 23M triple-annotated tokens; this branch must derive corresponding totals from actual, attributed records using a pinned tokenizer and human-adjudicated spans. We keep the novelty claim grounded in the integration design while making every locally verified source and every remaining provenance gap explicit.

## قيد استعادة النسخة التاريخية

تنص الرسالة في §3.7.1 على أن corpus الكامل ووسومه متاحان بطلب وفق ترخيص بحثي غير تجاري، بينما تسمي عينة 10% (7 ملايين رمز) في هذا الموضع برقم DOI غير معيّن. ويورد الملحق A.20 رقم Zenodo مختلفاً للعينة. لم نعثر على ملفات ANHC أو عينة الوسوم في مستودعات GitHub أو Zenodo عبر المعرفات المشار إليها. لذلك لا يمكن استخراج 23 مليون وسم محكّم من النسخة التاريخية دون الوصول إلى ملفاتها الأصلية؛ ملفات مصادر بديلة أو وسوم آلية لا تثبت ذلك الادعاء.
