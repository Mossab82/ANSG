# Research questions

## Context

Classical Arabic storytelling, and the *One Thousand and One Nights* above all, relies on **frame narratives**: a character in one story tells another story, which may contain further stories. A nested tale is told *for a reason*: to ransom a life, to warn, to persuade. When it ends, the enclosing story resumes, changed by what its listeners heard.

Large language models are known to struggle with global coherence in long narratives (see LITERATURE_REVIEW.md). Nested narration adds specific demands:
- keeping several story contexts apart, each with its own narrator, characters and time;
- returning to the right context after a tale closes;
- making sure characters know only what they have witnessed or been told.

## Primary question (RQ1)

> For Arabic stories with embedded tales, does generation under an optional, external hierarchical narrative controller (HNC) — which manages a stack of narrative frames, enforces closure and resumption, and validates cross-level character knowledge in the story plan — yield higher **structural fidelity**, i.e. correct narrative levels, correct resumption of the enclosing story and correct closure of embedded tales, than the same generator and the same plan without the controller (ANSG-Original path, `hnc_enabled=false`)?

## Secondary questions

- **RQ2 — depth.** How does any effect from RQ1 change with nesting depth (1, 2, 3, ≥4 levels)? At depth 1 no effect is expected.
- **RQ3 — components.** Which part of HNC carries the effect?
  - hierarchical management alone (condition C);
  - knowledge validation alone (condition D);
  - both together (condition B).
- **RQ4 — character knowledge.** Does validating cross-level knowledge in the plan reduce knowledge-leakage errors in the generated text, where a character acts on information from a tale it never heard?
- **RQ5 — cost to language and culture.** Does segment-wise controlled generation harm, as rated by native speakers:
  - fluency;
  - naturalness of transitions between narrators;
  - cultural fidelity;
  - dialect naturalness?

  This is tested as non-inferiority.
- **RQ6 — register.** Do the effects hold under both register policies?
  - R1: MSA narration with local-dialect dialogue.
  - R2: dialect narration and dialogue.

## Out of scope

- Reproducing the dissertation's ANSG results; that is ANSG-Original work.
- Learning nested narration inside model weights. HNC is programmatic control; see ARCHITECTURE.md §5.
- Claims about *quality* of heritage imitation. The scenarios are original and must not reproduce heritage text.
