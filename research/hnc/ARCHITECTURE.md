# HNC architecture

## 1. Principle: an optional, external controller

HNC decides **what the generator is asked to write, and in which order**. It does not change:
- the generator's weights;
- CSAM, DAL or the backbone;
- any ANSG equation.

It has **no trainable parameters** and imports no deep-learning library; this is tested. It is off by default:

```yaml
# configs/hnc/default.yaml
hnc_enabled: false
```

With `hnc_enabled: false`, `ansg.hnc.integration.generate()` calls the original `ansg.generation.runtime.run_generation()` with the caller's request unchanged and only adds `{"hnc": {"enabled": false}}` to the record. A test checks that the records are otherwise equal, including the token-id hash (`tests/integration/test_hnc_runtime.py`).

## 2. Representation (`ansg.hnc.model`)

A `HierarchicalPlan` is a tree of `NarrativeFrame`s, one frame per story level. Each frame holds:

| Field | Meaning |
| --- | --- |
| `graph` | A planned **Event Graph** (`ansg.narrative`): entities, events, thematic roles, causal and temporal relations, preconditions and effects, knowledge requirements. Each frame has its **own entities and its own story-time axis** |
| `narrator`, `audience` | Parent-frame characters who tell and hear the tale (narrator identity) |
| `perspective` | First or third person; `narrator_in_frame` links the first-person narrator to the character inside the tale |
| `time_context` | The tale's time, declared, never shared with the parent's time axis |
| `opened_by`, `resumed_by` | Parent events between which the tale is told. `resumed_by` must directly follow `opened_by` |
| `function`, `conveys` | Why the tale is told, and what its audience learns at closure. The parent story **must use** that knowledge at or after resumption, otherwise the plan is rejected (HNC-E-NO-FUNCTION) |

## 3. Validation (`ansg.hnc.validation`)

Before anything is generated, the plan is checked:

- **Structure:** one root, no cycles, depth limit, unique event ids.
- **Anchors:** the parent resumes directly after the tale is told.
- **Narration:** the narrator is a known animate character, the audience is non-empty, first-person tales name their narrator.
- **Time:** every embedded frame declares its time context; relations cannot link events of different frames.
- **Isolation:** a frame cannot mention entities that exist only in another frame.
- **Knowledge across levels:** every parent graph gets a "hearing" step between `opened_by` and `resumed_by`, in which the audience is *told* `conveys` by the narrator. The existing Event Graph validator then flags:
  - a character acting on a tale it did not hear (`E-KNOWLEDGE-UNSOURCED`);
  - a character acting on it before hearing it (`E-KNOWLEDGE-PREMATURE`);
  - a narrator telling what it cannot know (`E-TOLD-BY-IGNORANT`).

## 4. Control (`ansg.hnc.controller`)

The `NarrativeController` is a stack machine with the operations `start`, `emit(event)`, `enter(frame)`, `close()` and `finish()`. Its invariants are:

- an event is emitted only in its frame's planned story order;
- a tale is entered only right after its `opened_by` event, and only once;
- a tale closes only when all its events and sub-tales were told; its audience then learns `conveys`;
- after closure the parent **must** resume with `resumed_by`, unless a sibling tale anchored at the same event comes next;
- `finish` requires every frame to be closed and every event emitted exactly once, so no event is lost.

Every operation is recorded in a `Trace`.

## 5. What HNC is *not*

| Kind | Example in this repository | What it shows |
| --- | --- | --- |
| Programmatic control | `NarrativeController` | That the *requests* to the generator follow a legal nesting. It says nothing about the text |
| Planning | `HierarchicalPlan` + validation | That the *plan* is coherent. Plans here are hand-written; no planner is learned |
| Learning | — | Nothing. A model that "learned nesting" would have to show the effect **without** the controller |

Whether the generated *text* respects the levels is measured on human annotation of that text (METRICS.md). The controller's own trace is perfect by construction and is never reported as evidence.

## 6. Generation (`ansg.hnc.realize`)

| Condition | What runs |
| --- | --- |
| A | `hnc_enabled: false`. One call with a flat prompt that describes the whole nested plan in prose |
| B | Full HNC. One call per segment, where a segment is the consecutive events of one frame between transitions. The prompt carries the nesting path (who tells whom), narrator, perspective, time context, the tail of the story so far, the segment's events, and an explicit return instruction when the parent resumes |
| C | As B, but the plan is validated **without** the knowledge layer |
| D | Knowledge validation of the flattened plan, then one call with the flat prompt. No stack and no resumption control |

**Control data never reaches the text:**
- prompts contain names and descriptions, never internal ids (tested);
- the story is the concatenation of generated segments;
- `control_leaks()` flags any internal id that appears in the output.

## 7. Interfaces with ANSG components (all switchable, none modified)

| Component | Use by HNC | Status |
| --- | --- | --- |
| Event Graph (`ansg.narrative`) | Representation of every frame; validator reused unchanged | implemented |
| Narrative Generator (`ansg.generation.runtime`) | `generate_fn` adapter: one `run_generation` per segment, seed = seed + segment, one record per call | implemented |
| CSAM / DAL | Used exactly as the runtime uses them when a cultural checkpoint or `--untrained-cultural-modules` is passed. HNC passes the same request each time | implemented as pass-through |
| CSAM / DAL per frame | Different schema nodes or dialect per frame (e.g. a Moroccan narrator inside an Egyptian frame) would only need per-segment request parameters, not an architectural change | **future work, not implemented** |
| CKG | Frames could carry CKG schema node ids for activation per level | **future work** (needs a real CKG export) |

**Constraint.** Anything that would need CSAM or DAL to *see* the narrative level would be an architectural change to ANSG. One example is a level embedding inside attention. Such a change is **not** done here; it is recorded as a proposal for owner approval.
