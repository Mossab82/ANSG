# Preliminary literature review and novelty assessment

**Status: preliminary.** This review was compiled on 2026-11-14 from a small number of web searches plus background knowledge. Each entry carries a **verification** column, and nothing here is a systematic review. **No novelty is claimed.** §5 lists what must be done before any novelty statement is made in a paper.

## 1. Narrative levels and embedding (theory)

| Work | Relevance | Verification |
| --- | --- | --- |
| G. Genette, *Narrative Discourse: An Essay in Method* (Cornell UP, 1980; French *Figures III*, 1972) | Narrative levels (extradiegetic, intradiegetic, metadiegetic) and narrators; the vocabulary of HNC's frames | Standard reference; bibliographic details not re-checked in this session |
| M.-L. Ryan, "Stacks, Frames and Boundaries, or Narrative as Computer Language", *Poetics Today* 11(4), 1990, pp. 873–899; reprinted in B. Richardson (ed.), *Narrative Dynamics* (2002) | **Describes narrative embedding with the computer-science notion of a stack.** HNC's narrative stack is therefore **not** a new idea | Volume, issue and pages taken from the author's own bibliography page; not checked against the journal |

## 2. Embedded stories in computational narrative

| Work | Relevance | Verification |
| --- | --- | --- |
| P. Gervás, "A Model of Interpretation of Embedded Stories", Text2Story workshop at ECIR 2021, CEUR-WS Vol-2860, pp. 23–31 | A model for representing and interpreting stories told inside stories. Later papers describe it as using a **stack** for changing interpretation contexts. Prof. Gervás co-supervises this dissertation | Listing confirmed (dblp/webis); full text not read in this session |
| Later Gervás papers on embedded discourses: modal, anaphoric and false (insincere) embedded stories (2022); frame vs embedded story (2024 WNU); adapting Propp's morphology to embedded stories and multiple plot lines (ICCC 2025) | Directly related: embedded stories as anaphoric references, alternative views, insincere tales | Found by search (e.g. aclanthology.org/2024.wnu-1.1, computationalcreativity.net/iccc25/papers/iccc25-gervas2025adapting.pdf); titles and venues must be confirmed from the papers |
| L. Berov, character-focused narrative models (AAAI doctoral consortium) | Proposes modelling characters' embedded narratives as a graph linking plot generation and tellability | Found by search; details unverified |

## 3. Planning, hierarchy and coherence in story generation

| Work | Relevance | Verification |
| --- | --- | --- |
| M. O. Riedl & R. M. Young, "Narrative Planning: Balancing Plot and Character", *JAIR* 39 (2010) 217–268, doi:10.1613/jair.2989 | Causal plot plus character intentionality (IPOCL); basis for causal and character-consistency checks | Verified (JAIR / DOI) |
| A. Fan, M. Lewis, Y. Dauphin, "Hierarchical Neural Story Generation", ACL 2018 | Premise-to-story hierarchy. "Hierarchical" there means coarse-to-fine, **not** nested narration | Not re-verified in this session |
| L. Yao et al., "Plan-and-Write", AAAI 2019; L. J. Martin et al., event representations for story generation, AAAI 2018 | Plan- and event-based generation | Not re-verified in this session |
| K. Yang et al., Re3: recursive reprompting and revision (arXiv:2210.06774) | Plan-guided long-story generation by injecting plan parts into prompts. **Closest in mechanism** to HNC's segment prompting | arXiv verified; venue (EMNLP 2022) not confirmed here |
| K. Yang, D. Klein, N. Peng, Y. Tian, DOC: Detailed Outline Control (arXiv:2212.10077) | Hierarchical *outlines* with a controller that keeps passages aligned to the outline | arXiv verified; venue (ACL 2023) not confirmed here |
| Recent LLM long-story work: StoryWriter multi-agent (arXiv:2506.16445); StoryTeller with storyline and entity knowledge graph (arXiv:2506.02347); SCORE with episode summaries and item tracking (arXiv:2503.23512); DOME dynamic hierarchical outline; "Lost in Stories: Consistency Bugs in Long Story Generation by LLMs" (arXiv:2603.05890); *A Survey on LLMs for Story Generation* (Findings of EMNLP 2025) | Global coherence and consistency tracking; none found that targets nested frame narratives | Found by search; not read |

## 4. Preliminary novelty assessment (to be confirmed)

**Not novel:**
- a narrative stack (Ryan 1990);
- stack-based handling of embedded-story contexts in computational models (Gervás 2021 onward, from the same supervision team);
- plan- or outline-guided segment prompting (Re3, DOC);
- causal and character-intention checks (Riedl & Young 2010).

**Candidate contributions.** Each is to be checked against the full texts of the works above and a systematic search:
1. Generation-time control of **nested** narration for LLM generators, evaluated **against the same generator without control**, with ablations of hierarchy versus knowledge validation.
2. **Cross-level epistemic validation:** a listener knows a tale's content only through an explicit hearing step, enforced on the plan with an event-graph validator.
3. A **functional-coupling constraint:** an embedded tale must convey knowledge that the frame story uses at resumption, so that embedding counts as a narrative function and not mere decoration.
4. An **Arabic, multi-setting, multi-register** evaluation design (Aqaba, Alexandria, Kuwait City, Fes; MSA narration vs dialect).

How HNC relates to the Gervás line of work (interpretation versus generation; representation choices) must be discussed with that author before any claim is made.

## 5. Required before writing a paper

- [ ] Systematic search, with queries and dates recorded, in:
  - ACL Anthology (INLG, WNU, ACL/EMNLP/NAACL);
  - AAAI/AIIDE, ICCC, CMN, Text2Story;
  - Scopus / Web of Science.

  Search terms: "embedded narrative", "frame story", "story within a story", "nested narrative", "narrative levels", "metadiegetic", "theory of mind story generation", "Arabian Nights computational".
- [ ] Read the full texts of the Gervás papers (2021–2025) and of Ryan (1990).
- [ ] Check Arabic NLP venues (ArabicNLP / WANLP) for Arabic story generation and dialect work.
- [ ] Confirm every citation against the publisher record.
