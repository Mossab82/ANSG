# ANSG — Engineering Governance and Integration Plan

**Project owner:** Dissertation author. **Technical coordination and code review:** ChatGPT in the current collaboration. **Implementation contributor:** Claude, working only on scoped issues.

## Source of truth and provenance
1. The dissertation (Chapters 3–6 and Appendix A) is the research specification. Existing GitHub files are implementation evidence, not evidence of historic training outcomes.
2. Changes must cite exact thesis sections, equations, tables, and any deliberate engineering deviation. Unspecified algorithms (especially CRAFT/CMPI) remain explicitly unresolved.
3. Do not fabricate historical logs, model weights, corpus records, annotation labels, experiment dates, metrics, commits, or checksums.
4. Do not commit restricted datasets, copyrighted text, credentials, or original source PDFs unless licensing permits it.

## Branch strategy
- `master`: protected conceptual integration target; merge only reviewed work. Branch-protection rules must be separately configured by the repository administrator.
- `implementation/ansg-core-2026`: existing PyTorch model/training work (PR #1).
- `claude/relaxed-knuth-q63k61`: existing data, knowledge, evaluation-interface and documentation work (historical PR #2 is closed without merge).
- **Next task branch**: `claude/integration-audit` (create from latest `master`; do not mutate either existing implementation branch).
- Work is submitted through one scoped PR per GitHub issue. Do not merge your own PR without a research and technical review.

## Integration priority
**P0 Security:** rotate exposed provider credentials; do not mistake redaction of current branch for key revocation or sensitive-history removal.
**P1 Contract audit:** compare modules and packaging in both implementation branches; enumerate collisions, imports, entrypoints, configuration and dependency conflicts; produce a file-by-file merge plan.
**P2 Data interfaces:** validate ANHC (DIA/MOTIF/MORAL), Nights editions, ArabicNameNet/VerbNet provenance and CKG contract against actual source formats before training.
**P3 Model integration:** implement adapter connecting an explicit schema activation contract to CSAM and dialect metadata to DAL with deterministic shape/gradient tests.
**P4 Evaluation:** preserve explicit unknowns for CRAFT metrics; implement only metrics with fully supported formulae, test them independently; do not silently replace CMPI.
**P5 Real training:** run synthetic smoke tests followed by licensed-data and GPU experiments, reporting actual configuration/date/seed/hardware.

## Pull request acceptance checklist
- [ ] Scoped issue linked and branch base identified
- [ ] Description of changed files, design decisions and thesis references
- [ ] No existing feature silently removed, renamed or overwritten
- [ ] Compatibility with both branches documented with reproducible commands
- [ ] Tests include positive, negative, and integration cases; results are real CI artifacts
- [ ] Python imports and package topology have no collisions
- [ ] No secrets, copyrighted data, hard-coded absolute paths, silent network downloads
- [ ] Security checks and CI pass on PR head
- [ ] Experimental outputs labeled *synthetic*, *real*, or *not run*
- [ ] Unresolved scientific discrepancies documented and not portrayed as solved

## Collaboration reporting format
Each Claude delivery must include:
- Issue + PR URL and commit SHA
- Implemented files, executable commands, test results, known limitations
- Traceability table (claim -> dissertation section -> code line -> test)
- Decisions requested from dissertation author, explicitly scoped
- No claims of a full ANSG reconstruction until training and evaluation are independently verified.

## Immediate decision
Do **not** merge PR #1 or the closed PR #2 blindly. First complete cross-branch compatibility audit, preserve both code histories, and propose an integration PR with isolated tests.
