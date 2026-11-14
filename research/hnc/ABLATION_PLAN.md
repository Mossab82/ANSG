# Ablation plan

| Ablation | Contrast | Question |
| --- | --- | --- |
| Remove the whole controller | B vs A | Does HNC help at all (H1, H2)? |
| Remove the knowledge layer | B vs C | Does cross-level knowledge validation reduce leakage (H3)? |
| Remove hierarchical management | B vs D | Is the stack and resumption control needed beyond knowledge checks (H4)? |
| Remove the resumption instruction | B vs B−R | Is the explicit return instruction in the resume prompt what drives M2? (exploratory; implement as a `segment_prompt` flag before the study) |
| Remove story-so-far context | B (`context_chars=600`) vs B (`context_chars=0`) | Does continuity text matter for M3 and H-TRANS? (exploratory) |
| Depth | depth 1 vs 2 vs 3 vs ≥4 | Does the effect grow with nesting (H2), and is it absent at depth 1 (H6)? |
| Register | R1 vs R2 | Robustness across register policies (H7) |
| Cultural modules | backbone-only vs trained CSAM/DAL (only if trained modules exist) | Interaction with ANSG components. **Not** an evaluation of ANSG-Original |

**Rules:**
- One factor changes at a time; everything else (checkpoint, seeds, plans, token budget) is held constant.
- Each ablation is pre-declared as confirmatory (H1–H7) or exploratory, and exploratory results are labelled as such.
- HNC ablations are never mixed with ANSG-Original ablations (thesis §6.5).
