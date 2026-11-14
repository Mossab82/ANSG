# Event Graph specification (`ansg.event_graph/1`)

Code: `src/ansg/narrative/` (torch-free). Tests: `tests/unit/test_event_graph.py`,
`tests/unit/test_story_plan.py`. Review context: `docs/ar/ANSG_SCIENTIFIC_REVIEW_2026-11.md`.

The Event Graph is the semantic layer between heritage text and generation.

- **Analysis:** annotators (or later a model) represent a tale as events with
  participants, causal and temporal relations, world-state changes, character knowledge
  and values. Every claim is grounded in a character span of the source text.
- **Generation:** the same structure, with `planned` events and no text, is the
  *plan* of an original story. Plans are checked with the same validator before any text
  is generated.

Nothing in this specification is a dissertation result. The dissertation's ANHC
MOTIF/MORAL layers and CST-Onto codebook are not available (C-ANHC-3), so value labels
and narrative roles are free labels validated against an inventory only when one is
supplied.

## 1. Sources the design relies on

| Element | Basis | Status in this repository |
| --- | --- | --- |
| Thematic roles (Agent, Patient, Theme, Recipient, Location, Instrument, Experiencer, …) and selectional restrictions | VerbNet (Kipper Schuler 2005) and ArabicVerbNet (Mousser 2010), via the existing parser `ansg.knowledge.arabic_verbnet` | **No ArabicVerbNet data is bundled.** The parser reads files the owner supplies. Tests use a synthetic XML fixture |
| Preconditions and effects as fluents | STRIPS-style state transitions (Fikes & Nilsson 1971) | Implemented as a three-valued simulation (§4.4) |
| Narrative roles (protagonist, helper, donor, …) | Free labels, for example Propp-style functions | No inventory is supplied, so labels are not validated |
| Causal and temporal links | Annotated relations. Arabic connectives (فـ، لما، لأن، حتى …) are offered only as cues | Cue finder is a heuristic (`extraction.causal_cue_candidates`) |
| Annotation status | ANHC record model (`AnnotationStatus`, `AnnotationSource`) | Automatic candidates can never be silver or gold |

## 2. Data model

`EventGraph(record_id, text_sha256, entities, events, relations, initial_state, initial_knowledge, metadata)`

- **`text_sha256`** pins the exact `NarrativeRecord.text` the offsets refer to. It is
  `null` only for plans.
- **`Entity`**: `id`, `name`, `kind` (character, group, supernatural, animal, object, place,
  abstract), `narrative_role`, `aliases`, `mentions` (spans).
- **`Event`**:
  - `trigger`: the verb as written; `evidence`: its clause span; `discourse_index`.
  - `participants`: role, entity and an explicit mention span, *or* an `implicit_reason`
    (Arabic pro-drop, an anaphor). When both are given, the span is a pronoun or
    demonstrative resolved by coreference.
  - `modality`: actual, negated, hypothetical, reported or planned.
  - `story_time` (optional) and `verbnet_class` (optional).
  - `preconditions` and `effects`: propositions; a negated effect deletes a fluent.
  - `knowledge_required` and `knowledge_gained`: items of (entity, proposition, channel,
    source). The channel is initial, perception, told, inference or revelation.
  - `values`: label, polarity ±1, holder, evidence and status.
  - `status`, `source`; `gloss` (a description, required for plans).
- **`Relation`**: BEFORE, CAUSES, ENABLES, MOTIVATES or PREVENTS, with a `cue` span and
  `status`. CAUSES, ENABLES and MOTIVATES also imply BEFORE in story time.

JSON uses `ensure_ascii=False`. `EventGraph.from_dict(to_dict())` round-trips exactly
(tested). `from_dict` raises `ValueError` for a `null` entity mention. A mention is
always a span. Only participant `evidence` may be `null`, and that means implicit.

## 3. How a graph is made

1. `extraction.verb_trigger_candidates` marks tokens equal to a listed ArabicVerbNet
   member form (with و/ف stripped). `causal_cue_candidates` marks connectives. Both
   return status `automatic`, with spans and **no participants**. Inflected forms match
   only if the resource lists them, so recall is unknown. No morphological analyser is
   assumed.
   - The و/ف stripping is applied to a folded *copy* of the token, for lookup only.
     The candidate's `trigger` is the token as written, and its evidence is
     `text[start:end]`. Neither the record text nor the VerbNet member list is
     changed.
   - A verb candidate's evidence is its whole sentence: from the previous `. ! ? ؟` or
     line break to the next one, keeping the closing punctuation. Leading and trailing
     whitespace and line breaks are excluded. A sentence can be wider than the event's
     clause, so annotators narrow it.
2. Annotators complete participants, preconditions and effects, knowledge, values and
   relations, and set the status:
   - `silver`: a single annotator;
   - `gold`: adjudicated.

   Agreement is reported with the existing statistics module (κ).
3. `validate_event_graph(graph, text, verbnet=...)` must report **no errors** before a
   graph is used for training or evaluation. Warnings form the review queue.

## 4. Validation (error catalogue)

Errors (`E-…`) mean the graph contradicts its text or itself. Warnings (`W-…`) mean a
claim is not yet supported. Every error code listed below, including the plan codes
in §4.7, has at least one test that triggers it. A guard test checks this.

### 4.1 Grounding — "the event is in the text"

| Code | Condition |
| --- | --- |
| E-TEXT-HASH | the graph was annotated on a different text |
| E-EVIDENCE-SPAN / E-EVIDENCE-TEXT | the span is out of range, or `text[start:end]` differs from the stored text |
| E-EVIDENCE-MISSING | a non-planned event has no evidence |
| E-TRIGGER-NOT-IN-EVIDENCE | the trigger does not occur in its clause (diacritic- and hamza-insensitive), or the trigger is empty or only whitespace. "Occurs" is a substring test on folded text (§5) |
| E-MENTION-MISMATCH | an entity mention does not name the entity (matching rule below) |
| E-UNKNOWN-ENTITY | a participant, a knowledge item (required, gained or initial), the `source_entity` of a gained knowledge item, or a value holder refers to an entity id that is not in the graph |
| E-DUPLICATE-ID | two entities or two events share an id |

**How a mention is matched to a name** (`validation.mention_matches`, used by
E-MENTION-MISMATCH and E-PARTICIPANT-MISMATCH):

1. Both sides are folded for comparison only: diacritics, dagger alef and tatweel
   removed; alef, alef maqsura and teh marbuta normalised. The record text and the
   stored `Evidence.text` are never changed.
2. Proclitics (و ف ب ل ك, at most two) are removed **only from the running-text
   mention**, and only while at least three letters remain. Entity names and aliases
   are citation forms, so their first letter is part of the word and is never removed.
   So `وزير`, `كريم`, `فارس` and `لبنى` keep their first letter, and `ريم` does **not**
   name `كريم`.
3. The article `ال` is optional on both sides (`الوزير` names `وزير`, and `فارس` names
   `الفارس`). Contracted `لل` is covered: `للصياد` names `الصياد`.
4. Multi-word names match word by word and must have the same number of words.
5. The rule changes neither the entity's names and aliases (the candidate forms) nor
   the source text. It decides only whether a mention names one of them.

### 4.2 Participants — "the agent is the right one"

| Code | Condition |
| --- | --- |
| E-PARTICIPANT-MISMATCH | an explicit role mention does not name the assigned entity, after clitics, article and diacritics are stripped |
| E-PARTICIPANT-OUTSIDE-EVENT | a role mention lies outside the event's clause |
| E-AGENT-INANIMATE | the Agent or Experiencer is not animate |
| E-ROLE-NOT-IN-CLASS / E-SELRESTR / E-VERBNET-CLASS-UNKNOWN | with an ArabicVerbNet index: the role is not defined for the class (including inherited roles), violates its ±animate restriction, or the class is unknown |
| W-IMPLICIT-PARTICIPANT / W-ANAPHORIC-PARTICIPANT | a pro-drop or coreference assignment needs human review |

### 4.3 Relations — "causality does not contradict itself"

| Code | Condition |
| --- | --- |
| E-TEMPORAL-CYCLE | BEFORE, CAUSES, ENABLES and MOTIVATES edges (plus `story_time` order) form a cycle, for example an effect that causes its own cause |
| E-STORY-TIME-CONFLICT | explicit time stamps contradict a temporal edge |
| E-RELATION-CONTRADICTION | the same pair is linked by both PREVENTS and CAUSES/ENABLES |
| E-PREVENTED-EVENT-OCCURRED | an actual event prevents another actual event |
| E-CAUSED-EVENT-DID-NOT-OCCUR | an event caused or enabled by another is negated |
| E-RELATION-SELF / E-RELATION-UNKNOWN-EVENT | structural errors in a relation |
| W-CAUSE-NO-CUE / W-ENABLES-WITHOUT-MECHANISM / W-CAUSE-FROM-NONACTUAL | a causal claim has no textual cue, no state or knowledge mechanism, or a non-actual cause. W-CAUSE-NO-CUE is raised only for CAUSES/ENABLES without a cue whose status is neither `silver` nor `gold`, and only when text grounding is expected (`require_evidence=True`). Plans have no text, so they never receive it |

### 4.4 World state

Events are replayed in a story-time order that respects every temporal edge. Ties are
broken by `story_time`, then discourse order, then id, so flashbacks are handled by
explicit BEFORE edges. Only `actual` and `planned` events change the world; reported,
hypothetical and negated events do not.

| Code | Condition |
| --- | --- |
| E-PRECONDITION-VIOLATED | a precondition is false at that point |
| W-PRECONDITION-UNSUPPORTED | a precondition is never established. Under the closed world used for plans this becomes **E-PRECONDITION-NOT-ESTABLISHED** |
| E-EFFECT-CONFLICT | one event both adds and deletes a fluent |

### 4.5 Character knowledge — "nobody knows what they have not learned"

| Code | Condition |
| --- | --- |
| E-KNOWLEDGE-PREMATURE | a character acts on a proposition that it learns only *later* in story time |
| E-KNOWLEDGE-UNSOURCED | a character acts on a proposition it never learns |
| E-TOLD-BY-IGNORANT | the teller did not know what it tells |
| E-TOLD-WITHOUT-SOURCE | a `told` item has no teller |

A `told` item whose teller is not an entity of the graph is reported once, as
E-UNKNOWN-ENTITY (§4.1). It is not reported as E-TOLD-BY-IGNORANT, because nothing can
be said about what a missing entity knew.

Animate participants in observer roles (Agent, Experiencer, Recipient, Patient,
Beneficiary) learn the effects of events they take part in. This rule is configurable
(`participants_observe_effects`).

### 4.6 Values

| Code | Condition |
| --- | --- |
| W-VALUE-LABEL-UNKNOWN | the label is not in the supplied inventory |
| E-VALUE-POLARITY | polarity is not ±1 |
| E-UNKNOWN-ENTITY | the holder is not a known entity |

### 4.7 Story plans (`plan.validate_plan`)

A plan is validated in two steps:

1. `validate_event_graph(graph, None, config=ValidationConfig(require_evidence=False,
   closed_world=True))` runs. Grounding is skipped. W-NO-SOURCE-TEXT and
   W-CAUSE-NO-CUE do not arise. All structural, relation, state and knowledge checks
   apply. Under the closed world, a precondition that no earlier event or initial
   state establishes is **E-PRECONDITION-NOT-ESTABLISHED**. Planned events change the
   world, so a planned event that PREVENTS another planned event is
   E-PREVENTED-EVENT-OCCURRED.
2. Plan-level checks are added. An unknown setting does **not** stop them; only the
   two setting-dependent warnings are skipped.

| Code | Severity | Condition | Validator behaviour |
| --- | --- | --- | --- |
| E-PLAN-SETTING-UNKNOWN | error | `plan.setting` is not a key of the loaded settings (`configs/generation/settings.yaml`) | the other plan checks still run; W-PLAN-NAME/PLACE-OUTSIDE-SETTING are skipped |
| E-PLAN-TARGET-WORDS | error | `target_words` is not a positive integer: 0, negative, `bool` or a string. `StoryPlan.from_dict` converts the JSON value with `int()`, so a numeric string there becomes an integer | blocks `build_prompt` |
| E-PLAN-REGISTER | error | `register.narration` or `register.dialogue` is not `msa` or `dialect` | one issue per bad key |
| E-PLAN-EMPTY | error | the plan has no events | — |
| E-PLAN-MODALITY | error | an event's modality is not `planned` | one issue per event (`ref` = event id) |
| E-PLAN-NO-CHARACTER | error | no entity of kind `character` or `group` | — |
| W-PLAN-NO-GLOSS | warning | a planned event has no `gloss` | `build_prompt` falls back to the trigger |
| W-PLAN-NAME-OUTSIDE-SETTING | warning | a `character` name matches none of the setting's `names` (§4.1 rule, with the plan name in the mention role) | only when the setting lists names |
| W-PLAN-PLACE-OUTSIDE-SETTING | warning | a `place` name matches none of the setting's `places` | only when the setting lists places |

The name and place lists are an `unreviewed_draft` (see the settings file). A warning
here is a prompt for review. It says nothing about cultural authenticity.

`build_prompt(plan, settings)` raises `ValueError` listing every error code when the
plan has any error. Warnings do not block. Events are listed in the validator's
story-time order, and relations (except BEFORE) are listed in Arabic.

`build_prompt` does **not** list what each character knows. Knowledge items are
propositions with Latin predicate and argument ids, such as `owner(box, b)`, and they
are not rendered in Arabic. The prompt therefore gives only the general rule that no
character acts on information before learning it. It adds no character knowledge or
story facts beyond what the plan states.

## 5. Limits (stated, not hidden)

- Grounding verifies that a mention *names* the entity. It does not verify the
  *semantic* role; whether the named entity is really the agent needs gold annotation.
  Implicit and anaphoric assignments are always queued for review.
- The simulation uses one linear extension of the partial order. Unordered events are
  ordered by discourse position, so annotate BEFORE edges where order matters.
- Knowledge is propositional and binary (knows p / knows not-p). Degrees of belief,
  deception and nested beliefs (A knows that B knows) are not modelled yet.
- The validator checks consistency, not plausibility or quality.
- Mention matching is **approximate**. No morphological analysis is performed, and
  none is claimed. Removing proclitics from the mention can still mis-segment a word
  whose first letter is a root letter. For example, `كمال` in the text is accepted as
  naming an entity called `مال`, because it is read as ك + `مال`. The reverse case is
  avoided: names are never stripped, so `ريم` does not name `كريم`. Remaining
  ambiguities need annotator review.
- Trigger grounding (E-TRIGGER-NOT-IN-EVIDENCE) is a **substring** test on folded text.
  A trigger contained inside a longer word of its clause, such as `قال` inside
  `انتقال`, is accepted. The test proves only that the string occurs, not that the
  verb is the clause's predicate.
- `build_prompt` gives no per-character knowledge (§4.7) until propositions have an
  Arabic rendering.
