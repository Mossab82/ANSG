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
(tested).

## 3. How a graph is made

1. `extraction.verb_trigger_candidates` marks tokens equal to a listed ArabicVerbNet
   member form (with و/ف stripped). `causal_cue_candidates` marks connectives. Both
   return status `automatic`, with spans and **no participants**. Inflected forms match
   only if the resource lists them, so recall is unknown. No morphological analyser is
   assumed.
2. Annotators complete participants, preconditions and effects, knowledge, values and
   relations, and set the status:
   - `silver`: a single annotator;
   - `gold`: adjudicated.

   Agreement is reported with the existing statistics module (κ).
3. `validate_event_graph(graph, text, verbnet=...)` must report **no errors** before a
   graph is used for training or evaluation. Warnings form the review queue.

## 4. Validation (error catalogue)

Errors (`E-…`) mean the graph contradicts its text or itself. Warnings (`W-…`) mean a
claim is not yet supported. Every error code listed below has at least one test that
triggers it.

### 4.1 Grounding — "the event is in the text"

| Code | Condition |
| --- | --- |
| E-TEXT-HASH | the graph was annotated on a different text |
| E-EVIDENCE-SPAN / E-EVIDENCE-TEXT | the span is out of range, or `text[start:end]` differs from the stored text |
| E-EVIDENCE-MISSING | a non-planned event has no evidence |
| E-TRIGGER-NOT-IN-EVIDENCE | the trigger does not occur in its clause (diacritic- and hamza-insensitive) |
| E-MENTION-MISMATCH | an entity mention does not name the entity |

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
| W-CAUSE-NO-CUE / W-ENABLES-WITHOUT-MECHANISM / W-CAUSE-FROM-NONACTUAL | a causal claim has no textual cue, no state or knowledge mechanism, or a non-actual cause |

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

Animate participants in observer roles (Agent, Experiencer, Recipient, Patient,
Beneficiary) learn the effects of events they take part in. This rule is configurable
(`participants_observe_effects`).

### 4.6 Values

| Code | Condition |
| --- | --- |
| W-VALUE-LABEL-UNKNOWN | the label is not in the supplied inventory |
| E-VALUE-POLARITY | polarity is not ±1 |
| E-UNKNOWN-ENTITY | the holder is not a known entity |

## 5. Limits (stated, not hidden)

- Grounding verifies that a mention *names* the entity. It does not verify the
  *semantic* role; whether the named entity is really the agent needs gold annotation.
  Implicit and anaphoric assignments are always queued for review.
- The simulation uses one linear extension of the partial order. Unordered events are
  ordered by discourse position, so annotate BEFORE edges where order matters.
- Knowledge is propositional and binary (knows p / knows not-p). Degrees of belief,
  deception and nested beliefs (A knows that B knows) are not modelled yet.
- The validator checks consistency, not plausibility or quality.
