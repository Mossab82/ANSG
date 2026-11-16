#!/usr/bin/env python3
"""Run a small, non-training ANSG story-planning and proxy-metric pilot."""
from __future__ import annotations

import json
from pathlib import Path

from ansg.data.records import AnnotationSource, AnnotationStatus
from ansg.evaluation.story_proxies import summarise
from ansg.narrative.events import (
    Entity, EntityKind, Event, EventGraph, Modality, Participant, Proposition,
    Relation, RelationKind, ValueAnnotation,
)
from ansg.narrative.plan import StoryPlan, build_prompt, load_settings, validate_plan
from ansg.narrative.validation import knowledge_item


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "data/corpora/anhc_direct_build/alf_layla/story_candidates/alf_layla_story_candidates_001.jsonl"
SOURCE_ID = "hindawi-27482482-b003"

FLAWED_DRAFT = """في العقبة، سمع الملك سالم إشاعة أن أهل الحي يسرقون ماء القناة. فورًا أمر الوزير عودة أن يغلق القناة عنهم. بعد ذلك فحص الملك القناة فوجد حجرًا يسدها، مع أن الناس كانوا يشربون منها. قال الوزير: اتركها مغلقة حتى يتساوى الناس في الشرب. لم يسمع الملك أهل الحي، لكنه أعلن أنه وزع الماء بالعدل. وكانت العبرة أن الحاكم العادل لا يحتاج إلى دليل ولا إلى سماع الناس."""

FINAL_STORY = """في سنةٍ قلّ فيها المطر، لاحظ الملك سالم أن الماء في القناة صار أضعف من المعتاد. وصلته شكاوى من أهل الحي، فاستشار الوزير عودة وقال له: «شو رأيك؟ نسمع من الناس ونفحص مجرى القناة قبل ما نحكم؟»

راح سالم وعودة إلى طرف القناة، وهناك وجدوا حجرًا كبيرًا عالقًا عند المدخل، فكان يمنع الماء من الوصول إلى الحقول. نصح عودة الملك وقال: «هسّا عرفنا السبب. إذا أزلنا الحجر ووزعنا الماء بالدور، ما بنظلم حدا».

أزال العمّال الحجر، وفتحوا المجرى، ثم وزّعوا الماء بالدور وسجّلوا أوقات السقاية على لوحة عند البئر. اجتمع أهل الحي واختاروا ممثلين يتابعون الجدول، واتفقوا أن يخبّروا بعضهم إذا تغيّر تدفق الماء.

بعد أيام رجعت المياه إلى مجراها، ووصلت الحصص إلى الحقول بالتساوي. شكر الملك الوزير وأهل الحي على تعاونهم، وقال: «العدل ما بيقوم على إشاعة؛ بدّه دليل، وسماع للطرفين، ومشاركة بالقرار».

ومن يومها صار أهل العقبة يفحصون القناة معًا قبل ما يتهموا أحدًا، ويقسمون الماء بالدور. وعرفوا أن النصيحة النافعة تبدأ بالسؤال، وأن الإنصاف يحفظ حق الجماعة."""


def build_plan(*, repaired: bool) -> StoryPlan:
    setting = "aqaba"
    entities = [
        Entity("king", "سالم", EntityKind.CHARACTER, "protagonist"),
        Entity("minister", "عودة", EntityKind.CHARACTER, "helper"),
        Entity("residents", "أهل العقبة", EntityKind.GROUP, "community"),
        Entity("canal", "القناة", EntityKind.OBJECT),
        Entity("stone", "الحجر", EntityKind.OBJECT),
        Entity("aqaba", "العقبة", EntityKind.PLACE),
    ]

    def ev(eid: str, trigger: str, index: int, gloss: str, *, parts=(), pre=(), effects=(),
           required=(), gained=(), values=()):
        return Event(eid, trigger, None, index, list(parts), Modality.PLANNED,
                     preconditions=list(pre), effects=list(effects),
                     knowledge_required=list(required), knowledge_gained=list(gained),
                     values=list(values), gloss=gloss)

    blocked = Proposition("blocked", ("canal",))
    consultation = ValueAnnotation("deliberation", 1, "king", status=AnnotationStatus.AUTOMATIC,
                                    source=AnnotationSource.AUTOMATIC)
    justice = ValueAnnotation("justice", 1, "king", status=AnnotationStatus.AUTOMATIC,
                              source=AnnotationSource.AUTOMATIC)
    if repaired:
        events = [
            ev("e1", "لاحظ", 0, "سالم يلاحظ ضعف الماء في القناة", parts=[Participant("Agent", "king"), Participant("Theme", "canal")],
               gained=[knowledge_item("king", "low_flow", "canal", channel="perception")]),
            ev("e2", "فحص", 1, "سالم وعودة يفحصان القناة ويكتشفان الحجر", parts=[Participant("Agent", "king"), Participant("Agent", "minister"), Participant("Theme", "canal"), Participant("Instrument", "stone")],
               effects=[blocked], gained=[knowledge_item("king", "blocked", "canal", channel="perception"),
                                          knowledge_item("minister", "blocked", "canal", channel="perception")]),
            ev("e3", "استشار", 2, "سالم يستشير عودة بعد معرفة سبب ضعف الماء", parts=[Participant("Agent", "king"), Participant("Recipient", "minister")],
               required=[knowledge_item("king", "blocked", "canal")], values=[consultation]),
            ev("e4", "نصح", 3, "عودة يقترح إزالة الحجر وتوزيع الماء بالدور", parts=[Participant("Agent", "minister"), Participant("Recipient", "king")],
               required=[knowledge_item("minister", "blocked", "canal")]),
            ev("e5", "أزال", 4, "العمّال يزيلون الحجر فيعود الماء إلى المجرى", parts=[Participant("Agent", "residents"), Participant("Theme", "stone"), Participant("Location", "canal")],
               pre=[blocked], effects=[Proposition("blocked", ("canal",), True), Proposition("flowing", ("canal",))]),
            ev("e6", "وزّع", 5, "سالم وأهل الحي يوزعون الماء بالدور", parts=[Participant("Agent", "king"), Participant("Agent", "residents"), Participant("Theme", "canal")],
               pre=[Proposition("flowing", ("canal",))], effects=[Proposition("fair_access", ("residents",))], values=[justice]),
            ev("e7", "اتفق", 6, "أهل الحي يتفقون على جدول معلن ومراجعة مشتركة", parts=[Participant("Agent", "residents")],
               values=[ValueAnnotation("shared_responsibility", 1, "residents", status=AnnotationStatus.AUTOMATIC,
                                       source=AnnotationSource.AUTOMATIC)]),
        ]
        relations = [
            Relation(RelationKind.BEFORE, "e1", "e2"),
            Relation(RelationKind.BEFORE, "e2", "e3"),
            Relation(RelationKind.BEFORE, "e3", "e4"),
            Relation(RelationKind.MOTIVATES, "e4", "e5"),
            Relation(RelationKind.BEFORE, "e2", "e5"),
            Relation(RelationKind.BEFORE, "e5", "e6"),
            Relation(RelationKind.BEFORE, "e6", "e7"),
        ]
    else:
        events = [
            ev("e1", "أوصى", 0, "عودة يوصي بإغلاق القناة اعتمادًا على إشاعة", parts=[Participant("Agent", "minister"), Participant("Theme", "canal")],
               required=[knowledge_item("minister", "blocked", "canal")]),
            ev("e2", "فحص", 1, "سالم وعودة يفحصان القناة ويكتشفان الحجر", parts=[Participant("Agent", "king"), Participant("Agent", "minister"), Participant("Theme", "canal"), Participant("Instrument", "stone")],
               effects=[blocked], gained=[knowledge_item("king", "blocked", "canal", channel="perception"),
                                          knowledge_item("minister", "blocked", "canal", channel="perception")]),
            ev("e3", "أغلق", 2, "سالم يغلق القناة قبل التحقق من الشكوى", parts=[Participant("Agent", "king"), Participant("Theme", "canal")],
               required=[knowledge_item("king", "blocked", "canal")], effects=[Proposition("flowing", ("canal",), True)]),
        ]
        relations = [Relation(RelationKind.BEFORE, "e1", "e2"), Relation(RelationKind.BEFORE, "e2", "e3")]

    graph = EventGraph("pilot-king-adviser-aqaba", None, entities, events, relations)
    return StoryPlan(setting, "القناة والعدل", graph, {"narration": "msa", "dialogue": "dialect"}, 180,
                     notes="Original pilot inspired only by the advisor-and-ruler motif in an ANHC source candidate; not a retelling.")


def main() -> None:
    settings = load_settings(ROOT / "configs/generation/settings.yaml")
    rows = [json.loads(line) for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = next(row for row in rows if row["candidate_id"] == SOURCE_ID)
    source_text = source["text"]

    flawed = build_plan(repaired=False)
    flawed_report = validate_plan(flawed, settings)
    repaired = build_plan(repaired=True)
    repaired_report = validate_plan(repaired, settings)
    prompt = build_prompt(repaired, settings)

    source_meta = {k: source.get(k) for k in ("candidate_id", "title_candidate", "source", "span", "work_group", "candidate_scope", "status", "annotations")}
    metrics = summarise(FINAL_STORY, sources=[source_text], plan_graph=repaired.graph,
                        markers=settings["aqaba"].dialect_markers)
    report = {
        "experiment": "ANSG stepwise story pilot",
        "execution_mode": "plan-validation-and-deterministic-proxy-metrics; no language model or training",
        "source_candidate": source_meta,
        "source_text_sha256": source["source"]["source_text_sha256"],
        "source_annotation_state": source.get("annotations"),
        "source_boundary_status": source.get("status"),
        "provisional_pilot_tags": {
            "DIA": {"value": "Levantine setting; planned register = MSA narration + dialect dialogue", "status": "experimenter-provisional; not source annotation"},
            "MOTIF": {"value": "ruler-and-adviser consultation; rumor resolved by evidence", "status": "experimenter-provisional"},
            "MORAL": {"value": ["justice", "deliberation", "shared responsibility"], "status": "experimenter-provisional"},
        },
        "flawed_draft": FLAWED_DRAFT,
        "flawed_plan_validation": flawed_report.to_dict(),
        "repaired_plan_validation": repaired_report.to_dict(),
        "generation_prompt": prompt,
        "final_story_manual_realisation": FINAL_STORY,
        "proxy_metrics": metrics,
        "interpretation": {
            "planning": "The intentionally premature advice is rejected because the adviser acts on a fact learned only in a later event.",
            "cultural_reasoning": "The fairness/deliberation values are explicitly authored pilot tags; CSAM weights, CKG retrieval, and RL are not executed.",
            "dialect": "The plan requests Levantine/DAL with MSA narration and dialect dialogue. Marker rate is a presence proxy only; setting config is marked unreviewed_draft.",
            "realisation": "The final text is manually authored against the validated plan because no verified Arabic backbone/checkpoint is present locally.",
        },
        "model_and_training": {"local_backbone_found": False, "torch_available": False, "csam_dal_checkpoint_found": False, "ppo_executed": False},
    }
    (OUT / "pilot_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flawed.graph.write(OUT / "plan_flawed.event_graph.json")
    repaired.graph.write(OUT / "plan_repaired.event_graph.json")
    (OUT / "generation_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (OUT / "story_flawed.txt").write_text(FLAWED_DRAFT + "\n", encoding="utf-8")
    (OUT / "story_repaired.txt").write_text(FINAL_STORY + "\n", encoding="utf-8")
    print(json.dumps({
        "source_candidate": SOURCE_ID,
        "flawed_ok": flawed_report.ok,
        "flawed_errors": [i.code for i in flawed_report.errors],
        "repaired_ok": repaired_report.ok,
        "repaired_errors": [i.code for i in repaired_report.errors],
        "repaired_warnings": [i.code for i in repaired_report.warnings],
        "proxy_metrics": metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
