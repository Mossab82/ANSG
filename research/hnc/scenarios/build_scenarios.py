#!/usr/bin/env python
"""Build the HNC test scenarios (hierarchical story PLANS) as JSON.

These are ORIGINAL plans written by the engineering reviewer for the experiment
protocol (research/hnc/EXPERIMENT_PROTOCOL.md §4). They contain no heritage text
and no generated text, are not data and are not results. Settings follow
configs/generation/settings.yaml (status: unreviewed_draft); a native-speaker /
cultural review is required before the scenarios are used with human raters.

Every embedded tale has a stated function, a time context, and conveys knowledge
that the frame story uses when it resumes (checked by ansg.hnc.validation).

    python research/hnc/scenarios/build_scenarios.py      # rewrites the JSON files next to this script
"""
from __future__ import annotations

import sys
from pathlib import Path

from ansg.data.records import AnnotationStatus
from ansg.hnc.model import HierarchicalPlan, NarrativeFrame
from ansg.narrative.events import (Entity, EntityKind, Event, EventGraph, Modality, Participant, Proposition,
                                   Relation, RelationKind)
from ansg.narrative.validation import knowledge_item

HERE = Path(__file__).resolve().parent


def P(spec: str, *args: str) -> Proposition:
    return Proposition(spec.lstrip("!"), tuple(args), spec.startswith("!"))


def ent(i, name, kind="character", role=None):
    return Entity(i, name, EntityKind(kind), role)


def ev(i, trigger, n, gloss, parts, pre=(), eff=(), kreq=()):
    return Event(i, trigger, None, n, [Participant(r, e) for r, e in parts], Modality.PLANNED, gloss=gloss,
                 preconditions=list(pre), effects=list(eff),
                 knowledge_required=[knowledge_item(who, p.predicate, *p.args) for who, p in kreq])


def chain(*ids, kind=RelationKind.BEFORE):
    return [Relation(kind, a, b, status=AnnotationStatus.SILVER) for a, b in zip(ids, ids[1:])]


def frame(fid, entities, events, relations, initial=(), **kw):
    return NarrativeFrame(id=fid, graph=EventGraph(f"plan:{fid}", None, entities, events, relations,
                                                   initial_state=list(initial)), **kw)


# --------------------------------------------------------------------------- Aqaba (2 levels)
def aqaba() -> HierarchicalPlan:
    root = frame("aq-frame", [ent("odeh", "عودة", role="protagonist"), ent("eid", "عيد", role="grandfather"),
                              ent("boat", "القارب", "object"), ent("gulf", "خليج العقبة", "place")],
                 [ev("aq1", "قرر", 0, "عودة يقرر بيع قارب أبيه القديم ليشتري سيارة",
                     [("Agent", "odeh"), ("Theme", "boat")], pre=[P("has", "odeh", "boat")],
                     eff=[P("wants_to_sell", "odeh", "boat")]),
                  ev("aq2", "جلس", 1, "الجد عيد يجلس مع حفيده عودة على شاطئ خليج العقبة عند الغروب",
                     [("Agent", "eid"), ("Recipient", "odeh"), ("Location", "gulf")]),
                  ev("aq3", "تراجع", 2, "عودة يتراجع عن بيع القارب ويقرر إصلاحه",
                     [("Agent", "odeh"), ("Theme", "boat")], kreq=[("odeh", P("saved_lives", "boat"))],
                     eff=[P("!wants_to_sell", "odeh", "boat")]),
                  ev("aq4", "أصلح", 3, "عودة وجده يصلحان القارب ويخرجان به للصيد معًا",
                     [("Agent", "odeh"), ("Theme", "boat")], pre=[P("has", "odeh", "boat")])],
                 chain("aq1", "aq2", "aq3", "aq4"), initial=[P("has", "odeh", "boat")])
    storm = frame("aq-storm", [ent("eid_young", "عيد", role="narrator as young man"),
                               ent("fishers", "الصيادون", "group"), ent("storm", "العاصفة", "abstract"),
                               ent("old_boat", "القارب", "object")],
                  [ev("st1", "خرج", 0, "عيد الشاب يخرج للصيد بالقارب في ليلة هادئة",
                      [("Agent", "eid_young"), ("Theme", "old_boat")], eff=[P("at_sea", "eid_young")]),
                   ev("st2", "هبت", 1, "تهب عاصفة مفاجئة فتنقلب قوارب الصيادين الآخرين",
                      [("Theme", "storm"), ("Patient", "fishers")], pre=[P("at_sea", "eid_young")],
                      eff=[P("in_danger", "fishers")]),
                   ev("st3", "أنقذ", 2, "عيد يجدّف نحوهم في الظلام وينقذهم جميعًا بقاربه الصغير",
                      [("Agent", "eid_young"), ("Patient", "fishers"), ("Instrument", "old_boat")],
                      pre=[P("in_danger", "fishers")], eff=[P("!in_danger", "fishers"), P("rescued", "fishers")])],
                  chain("st1", "st2") + chain("st2", "st3", kind=RelationKind.ENABLES),
                  parent="aq-frame", narrator="eid", audience=["odeh"], perspective="first_person",
                  narrator_in_frame="eid_young", time_context="شباب الجد عيد قبل عقود",
                  function="إقناع الحفيد بقيمة القارب ووفاء أهل البحر بعضهم لبعض",
                  opened_by="aq2", resumed_by="aq3", conveys=[P("saved_lives", "boat")])
    return HierarchicalPlan("aqaba", "قارب الجد", [root, storm])


# --------------------------------------------------------------------------- Alexandria (2 levels)
def alexandria() -> HierarchicalPlan:
    root = frame("alx-frame", [ent("zeinab", "زينب", role="bakery owner"), ent("hassan", "حسن", role="apprentice"),
                               ent("mahmoud", "محمود", role="neighbour"), ent("nadia", "نادية", role="elder"),
                               ent("ledger", "دفتر الحسابات", "object"), ent("bahari", "بحري", "place")],
                 [ev("al1", "اتهم", 0, "محمود يتهم حسن بأخذ نقود من صندوق مخبز زينب في بحري",
                     [("Agent", "mahmoud"), ("Patient", "hassan"), ("Location", "bahari")],
                     eff=[P("accused", "hassan")]),
                  ev("al2", "طلبت", 1, "نادية تطلب من الجميع أن يسمعوا حكاية قبل أن يحكموا",
                     [("Agent", "nadia"), ("Recipient", "mahmoud")]),
                  ev("al3", "راجع", 2, "محمود يوافق على مراجعة دفتر الحسابات قبل الحكم",
                     [("Agent", "mahmoud"), ("Theme", "ledger")],
                     kreq=[("mahmoud", P("hasty_accusation_wrongs", "innocents"))], eff=[P("checked", "ledger")]),
                  ev("al4", "اعتذر", 3, "يظهر الدفتر أن محمود أخطأ في العد فيعتذر لحسن",
                     [("Agent", "mahmoud"), ("Recipient", "hassan")],
                     pre=[P("checked", "ledger"), P("accused", "hassan")], eff=[P("!accused", "hassan")])],
                 chain("al1", "al2", "al3") + chain("al3", "al4", kind=RelationKind.ENABLES))
    key = frame("alx-key", [ent("sayed", "سيد", role="fisherman"), ent("youth", "عبد الرحمن", role="accused youth"),
                            ent("key", "المفتاح", "object"), ent("anfoushi", "الأنفوشي", "place")],
                [ev("ky1", "فقد", 0, "صياد اسمه سيد يفقد مفتاح قاربه في الأنفوشي",
                    [("Agent", "sayed"), ("Theme", "key"), ("Location", "anfoushi")], eff=[P("lost", "key")]),
                 ev("ky2", "اتهم", 1, "سيد يتهم جاره الشاب عبد الرحمن بسرقته أمام الناس",
                    [("Agent", "sayed"), ("Patient", "youth")], pre=[P("lost", "key")], eff=[P("accused", "youth")]),
                 ev("ky3", "وجد", 2, "بعد أيام يجد سيد المفتاح في جيب معطفه، وقد ترك الشاب الحي حزينًا",
                    [("Agent", "sayed"), ("Theme", "key")], pre=[P("lost", "key")], eff=[P("!lost", "key")])],
                chain("ky1", "ky2", "ky3"),
                parent="alx-frame", narrator="nadia", audience=["mahmoud", "zeinab"], perspective="third_person",
                time_context="قبل ثلاثين عامًا في حي الأنفوشي", function="تحذير من التسرع في الاتهام",
                opened_by="al2", resumed_by="al3", conveys=[P("hasty_accusation_wrongs", "innocents")])
    return HierarchicalPlan("alexandria", "المفتاح الضائع", [root, key])


# --------------------------------------------------------------------------- Kuwait City (3 levels)
def kuwait() -> HierarchicalPlan:
    root = frame("kw-frame", [ent("fahad", "فهد", role="protagonist"), ent("latifa", "لطيفة", role="mother"),
                              ent("abdullah", "عبد الله", role="partner"),
                              ent("market", "سوق المباركية", "place")],
                 [ev("kf1", "وعد", 0, "فهد يعد شريكه عبد الله بتسليم البضاعة في موعدها",
                     [("Agent", "fahad"), ("Recipient", "abdullah"), ("Location", "market")],
                     eff=[P("promised", "fahad")]),
                  ev("kf2", "تردد", 1, "يعرض تاجر آخر على فهد ثمنًا أعلى فيتردد",
                     [("Experiencer", "fahad")], eff=[P("tempted", "fahad")]),
                  ev("kf3", "جلست", 2, "أمه لطيفة تلاحظ تردده وتجلس معه في البيت",
                     [("Agent", "latifa"), ("Recipient", "fahad")]),
                  ev("kf4", "سلّم", 3, "فهد يسلّم البضاعة لعبد الله كما وعد",
                     [("Agent", "fahad"), ("Recipient", "abdullah")], pre=[P("promised", "fahad")],
                     kreq=[("fahad", P("family_keeps_word", "family"))])],
                 chain("kf1", "kf2", "kf3", "kf4"))
    grand = frame("kw-grand", [ent("jassim", "جاسم", role="grandfather"), ent("buyer", "تاجر اللؤلؤ", role="buyer"),
                               ent("nokhada", "يوسف", role="ship captain")],
                  [ev("kg1", "وعد", 0, "الجد جاسم يعد تاجر اللؤلؤ بلؤلؤة نادرة من موسم الغوص",
                      [("Agent", "jassim"), ("Recipient", "buyer")], eff=[P("promised", "jassim")]),
                   ev("kg2", "تردد", 1, "يعرض غريب على جاسم ثمنًا أعلى فيتردد",
                      [("Experiencer", "jassim")], eff=[P("tempted", "jassim")]),
                   ev("kg3", "جمع", 2, "النوخذة يوسف يجمع البحارة ليلًا على ظهر السفينة",
                      [("Agent", "nokhada"), ("Recipient", "jassim")]),
                   ev("kg4", "وفى", 3, "جاسم يفي بوعده للتاجر الأول",
                      [("Agent", "jassim"), ("Recipient", "buyer")], pre=[P("promised", "jassim")],
                      kreq=[("jassim", P("word_saves", "keeping_promises"))])],
                  chain("kg1", "kg2", "kg3", "kg4"),
                  parent="kw-frame", narrator="latifa", audience=["fahad"], perspective="third_person",
                  time_context="أيام الغوص على اللؤلؤ في زمن الجد جاسم",
                  function="تذكير فهد بأن العائلة تفي بوعدها مهما كان الإغراء",
                  opened_by="kf3", resumed_by="kf4", conveys=[P("family_keeps_word", "family")])
    diver = frame("kw-diver", [ent("diver", "غواص شاب", role="diver"), ent("crew", "الطاقم", "group")],
                  [ev("kd1", "وعد", 0, "غواص شاب يعد الطاقم بأن يعود قبل الغروب",
                      [("Agent", "diver"), ("Recipient", "crew")], eff=[P("promised", "diver")]),
                   ev("kd2", "وجد", 1, "يجد الغواص محارًا كثيرًا يغريه بالبقاء",
                      [("Agent", "diver")], eff=[P("tempted", "diver")]),
                   ev("kd3", "عاد", 2, "يعود في موعده فيرى الطاقم العاصفة قادمة ويبحرون قبلها",
                      [("Agent", "diver"), ("Recipient", "crew")], pre=[P("promised", "diver")],
                      eff=[P("safe", "crew")])],
                  chain("kd1", "kd2", "kd3"),
                  parent="kw-grand", narrator="nokhada", audience=["jassim"], perspective="third_person",
                  time_context="زمن أقدم من زمن جاسم", function="بيان أن الوفاء بالعهد أنقذ طاقمًا كاملًا",
                  opened_by="kg3", resumed_by="kg4", conveys=[P("word_saves", "keeping_promises")])
    return HierarchicalPlan("kuwait", "اللؤلؤة الموعودة", [root, grand, diver])


# --------------------------------------------------------------------------- Fes (2 levels)
def fes() -> HierarchicalPlan:
    root = frame("fs-frame", [ent("khadija", "خديجة", role="protagonist"), ent("idriss", "إدريس", role="uncle"),
                              ent("chouara", "دار الدباغة شوارة", "place"), ent("bab", "باب بوجلود", "place")],
                 [ev("fs1", "قررت", 0, "خديجة تقرر ترك عملها في دار الدباغة شوارة والرحيل عن فاس",
                     [("Agent", "khadija"), ("Location", "chouara")], eff=[P("wants_to_leave", "khadija")]),
                  ev("fs2", "دعا", 1, "عمها إدريس يدعوها إلى كأس شاي قرب باب بوجلود",
                     [("Agent", "idriss"), ("Recipient", "khadija"), ("Location", "bab")]),
                  ev("fs3", "بقيت", 2, "خديجة تقرر البقاء وتعلُّم الصنعة كي تعلّمها لغيرها",
                     [("Agent", "khadija")], kreq=[("khadija", P("craft_needs_heirs", "craft"))],
                     eff=[P("!wants_to_leave", "khadija")])],
                 chain("fs1", "fs2", "fs3"))
    weaver = frame("fs-weaver", [ent("idriss_young", "إدريس", role="narrator as young man"),
                                 ent("hamid", "حميد", role="master weaver"), ent("workshop", "الورشة", "object")],
                   [ev("fw1", "رحل", 0, "إدريس الشاب يترك ورشة النسيج ويرحل",
                       [("Agent", "idriss_young"), ("Source", "workshop")], eff=[P("away", "idriss_young")]),
                    ev("fw2", "أغلق", 1, "المعلم حميد يغلق الورشة لأنه لم يجد من يخلفه",
                       [("Agent", "hamid"), ("Patient", "workshop")], eff=[P("closed", "workshop")]),
                    ev("fw3", "عاد", 2, "يعود إدريس فيعيد فتح الورشة ويعلّم الصغار",
                       [("Agent", "idriss_young"), ("Theme", "workshop")], pre=[P("closed", "workshop")],
                       eff=[P("!closed", "workshop"), P("!away", "idriss_young")])],
                   chain("fw1", "fw2") + chain("fw2", "fw3", kind=RelationKind.ENABLES),
                   parent="fs-frame", narrator="idriss", audience=["khadija"], perspective="first_person",
                   narrator_in_frame="idriss_young", time_context="شباب إدريس",
                   function="إقناع خديجة بأن الصنعة تحتاج من يرثها", opened_by="fs2", resumed_by="fs3",
                   conveys=[P("craft_needs_heirs", "craft")])
    return HierarchicalPlan("fes", "من يرث الصنعة", [root, weaver])


# --------------------------------------------------------------------------- stress: N-level chain
def nested_chain(depth: int, setting: str = "fes") -> HierarchicalPlan:
    """A chain of tellers, each telling the next tale, each lesson used by its listener (stress test)."""
    frames = []
    for lvl in range(1, depth + 1):
        fid = f"ch{lvl}"
        teller, listener = f"t{lvl}", f"l{lvl}"
        ents = [ent(listener, f"السامع {lvl}"), ent(teller, f"الراوي {lvl}")]
        kreq = [(listener, P("lesson", f"level{lvl + 1}"))] if lvl < depth else []
        events = [ev(f"{fid}a", "سأل", 0, f"السامع {lvl} يسأل عن أمر يحيّره", [("Agent", listener)]),
                  ev(f"{fid}b", "جلس", 1, f"الراوي {lvl} يجلس ليحكي", [("Agent", teller), ("Recipient", listener)]),
                  ev(f"{fid}c", "فهم", 2, f"السامع {lvl} يفهم ما كان يحيّره", [("Experiencer", listener)], kreq=kreq)]
        kw = {}
        if lvl > 1:
            prev = f"ch{lvl - 1}"
            kw = dict(parent=prev, narrator=f"t{lvl - 1}", audience=[f"l{lvl - 1}"], perspective="third_person",
                      time_context=f"زمن أقدم من المستوى {lvl - 1}", function=f"جواب حيرة السامع {lvl - 1}",
                      opened_by=f"{prev}b", resumed_by=f"{prev}c", conveys=[P("lesson", f"level{lvl}")])
        frames.append(frame(fid, ents, events, chain(f"{fid}a", f"{fid}b", f"{fid}c"), **kw))
    return HierarchicalPlan(setting, f"سلسلة من {depth} مستويات", frames)


def root_only(plan: HierarchicalPlan) -> HierarchicalPlan:
    """Depth-1 control: the frame story alone, without its embedded tales and without the knowledge
    those tales would have conveyed (used for the 'one level' condition of the protocol)."""
    import copy
    root = copy.deepcopy(plan.root())
    conveyed = {p.key() for f in plan.frames if f.parent == root.id for p in f.conveys}
    for e in root.graph.events:
        e.knowledge_required = [k for k in e.knowledge_required if k.proposition.key() not in conveyed]
    return HierarchicalPlan(plan.setting, plan.title, [root], dict(plan.register), plan.target_words,
                            "depth-1 control derived from the nested scenario")


SCENARIOS = {"aqaba_2level": aqaba, "alexandria_2level": alexandria, "kuwait_3level": kuwait, "fes_2level": fes,
             "fes_4level_stress": lambda: nested_chain(4),
             "aqaba_1level": lambda: root_only(aqaba()), "alexandria_1level": lambda: root_only(alexandria()),
             "kuwait_1level": lambda: root_only(kuwait()), "fes_1level": lambda: root_only(fes())}


def main() -> int:
    for name, build in SCENARIOS.items():
        (HERE / f"{name}.json").write_text(build().to_json() + "\n", encoding="utf-8")
    print(f"wrote {len(SCENARIOS)} scenarios to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
