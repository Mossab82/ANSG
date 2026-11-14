# Reproducibility

## Software tests (no GPU, no download)

```bash
python -m pip install -e ".[test]"                  # torch-free layer: HNC, plans, validation, metrics
python -m pytest tests/unit/test_hnc.py tests/unit/test_event_graph.py tests/unit/test_story_plan.py
python -m pip install -e ".[test,model,pretrained]"  # runtime tests on a locally generated synthetic stub
python -m pytest tests/integration/test_hnc_runtime.py
```

The acceptance criteria and the tests that pin them:

| Criterion | Test |
| --- | --- |
| 1. Disabling HNC preserves the original path | `test_disabled_hnc_reproduces_the_original_path_exactly`, `test_disabled_hnc_is_one_untouched_call` |
| 2. An embedded tale keeps the parent context | `test_embedded_tale_keeps_parent_context_and_returns_to_correct_level` |
| 3. Return to the correct level | the same test, plus `test_sibling_tales_at_the_same_anchor_then_resume` |
| 4. No knowledge without a justifying event | `test_knowledge_does_not_cross_levels_without_an_event`, `test_inner_tale_cannot_know_frame_events` |
| 5. No event loss | `test_schedule_replays_and_loses_no_event`, `test_illegal_transitions`, `test_metrics_detect_*` |
| 6. HNC does not change weights | `test_hnc_does_not_change_model_weights`, `test_hnc_has_no_trainable_parameters_and_no_torch` |
| 7. Outputs state whether HNC was on | `test_hnc_enabled_generates_per_segment_with_resumption`, `test_disabled_hnc_reproduces_the_original_path_exactly` |
| 8. Existing suite still passes | full `pytest` in CI (both jobs) |

## Scenarios

The scenarios are deterministic:

```bash
python research/hnc/scenarios/build_scenarios.py
```

This regenerates the committed JSON files. `test_committed_scenarios_match_their_builder_and_validate` fails if they drift.

## Experiment runs (owner, GPU outside CI)

```bash
python scripts/hnc_generate.py --hplan research/hnc/scenarios/<scenario>.json --variant {A,B,C,D} \
    --register {r1,r2} --backbone <local dir> --backbone-sha256 <digest> --seed <s> \
    [--chat-template] --output runs/hnc/<scenario>_<variant>_<register>_<seed>.json
```

Each run record (`ansg.hnc_run/1`) contains:
- `hnc.enabled`, the variant and the full HNC config;
- the plan SHA-256, setting, register and the settings' review status;
- the trace (B/C), the segments with prompt and text SHA-256, and control-leak checks;
- **one complete `ansg-generate` record per call**: backbone digest, tokenizer digest, software versions, git commit, seed, sampling configuration.

Records are created once and never overwritten.

## What to archive with a paper

- The git commit, `configs/hnc/default.yaml`, the scenario JSON files and their SHA-256.
- The backbone directory digest and licence (`ansg_backbone.json`).
- Every run record.
- Annotation guidelines, raw annotations with adjudication, rating data with consent forms (anonymised), and the frozen analysis code.

**Never archive or report as HNC evidence:**
- the controller's own trace;
- synthetic-stub runs;
- results from the heritage pilot's night-level split.
