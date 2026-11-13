#!/usr/bin/env python
"""Cross-branch contract probe (Issue #3) — runs WITHOUT merging either branch.

Loads ``src/ansg`` of ``implementation/ansg-core-2026`` as ``ansg_core`` and of
``claude/relaxed-knuth-q63k61`` as ``ansg_data`` (two import aliases, so the
module/package collisions documented in the audit cannot occur), then drives
the target flow on **synthetic** fixtures only:

    Nights text -> ingestion/normalisation -> ANHC record -> explicit (synthetic)
    DIA label -> CKG activation -> schema ids + mask -> CSAM -> DAL ->
    generation -> evaluation reported separately

Every step records PASS / FAIL / FINDING. A FINDING is an expected,
documented incompatibility (``docs/INTEGRATION_COMPATIBILITY_AUDIT.md``); it
does not fail the probe. Any unexpected exception fails it (exit 1).

All annotations created here are labelled ``synthetic`` and
``AnnotationSource.AUTOMATIC``; nothing is a dissertation result.

Usage::

    python tools/integration_probe.py --core <worktree of ansg-core-2026> \\
        --data <worktree of relaxed-knuth-q63k61> [--json out.json]

Requires torch (for the core model) and PyYAML.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

RESULTS: list[dict[str, Any]] = []


def load_alias(alias: str, pkg_dir: Path):
    spec = importlib.util.spec_from_file_location(alias, pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def step(name: str, contract: str):
    def deco(fn: Callable[[dict], Any]):
        def run(ctx: dict) -> None:
            try:
                status, detail = fn(ctx)
            except Exception as exc:  # unexpected
                status, detail = "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
            RESULTS.append({"step": name, "contract": contract, "status": status, "detail": detail})
        return run
    return deco


# --------------------------------------------------------------------------- steps
@step("1 ingest synthetic Nights fixture", "C-DATA-1")
def s_ingest(ctx):
    nights = importlib.import_module("ansg_data.data.nights")
    records_mod = importlib.import_module("ansg_data.data.records")
    prov = records_mod.SourceProvenance(source_id="synthetic", title="synthetic fixture",
                                        copyright_status=records_mod.CopyrightStatus.PUBLIC_DOMAIN,
                                        redistributable=True)
    recs, rep = nights.ingest_file(ctx["data_root"] / "tests/fixtures/nights/synthetic_nights.txt", prov)
    ctx["records"] = [r for r in recs if r.narrative.title]  # the two titled tales
    return "PASS", {"n_records": len(recs), "titled": len(ctx["records"])}


@step("2 core text normaliser vs data normaliser", "C-DATA-3")
def s_norm(ctx):
    lb = importlib.import_module("ansg_core.legacy_bridge")
    text = ctx["records"][0].text
    core_norm = lb.normalize_arabic(text)
    lost_newlines = text.count("\n") - core_norm.count("\n")
    return ("FINDING" if lost_newlines else "PASS",
            {"newlines_in_record": text.count("\n"), "newlines_after_core_normalize": core_norm.count("\n"),
             "note": "core normalize_arabic collapses all whitespace; paragraph/night boundaries are lost"})


@step("3 attach explicit synthetic DIA label", "C-DATA-2")
def s_dia(ctx):
    rec_mod = importlib.import_module("ansg_data.data.records")
    labels = rec_mod.LabelInventory.from_mapping({"dia": ["MSA", "Egyptian", "Levantine", "Gulf", "Maghrebi"]})
    for r, lab in zip(ctx["records"], ["MSA", "Levantine"]):
        r.dia.append(rec_mod.DialectAnnotation(span=None, annotator_id=None,
                                               source=rec_mod.AnnotationSource.AUTOMATIC,
                                               attributes={"synthetic": True}, label=lab))
        r.annotation_status["dia"] = rec_mod.AnnotationStatus.AUTOMATIC
        errors = [i for i in rec_mod.validate_record(r, labels) if i.severity == "error"]
        if errors:
            return "FAIL", [e.message for e in errors]
    core_cfg = importlib.import_module("ansg_core.config")
    lb = importlib.import_module("ansg_core.legacy_bridge")
    raw = [r.dia[0].label for r in ctx["records"]]
    direct_ok = [lab in lb.VALID_DIALECTS for lab in raw]
    mapping = {lab: lab.lower() for lab in raw}  # adapter rule proposed in the plan
    ctx["dialect_ids"] = [core_cfg.DIALECTS.index(mapping[lab]) for lab in raw]
    return ("FINDING" if not all(direct_ok) else "PASS",
            {"data_labels": raw, "accepted_by_core_without_mapping": direct_ok,
             "core_DIALECTS": list(core_cfg.DIALECTS), "dialect_ids_after_casefold": ctx["dialect_ids"]})


@step("4 CKG activation -> schema ids (0 reserved for padding)", "C-CKG-1")
def s_ckg(ctx):
    ckg = importlib.import_module("ansg_data.knowledge.ckg")
    g = ckg.CulturalKnowledgeGraph.from_json(ctx["data_root"] / "tests/fixtures/ckg/synthetic_ckg.json")
    index = g.index()
    act = ckg.LexicalTriggerEvidence()
    ids, weights = [], []
    for r in ctx["records"]:
        res = act.activate(r.text, g)
        ranked = sorted(res.scores, key=lambda k: (-res.scores[k], k))
        ids.append([index[k] + 1 for k in ranked] or [])  # +1: core uses 0 as pad/no-schema
        weights.append([res.scores[k] for k in ranked])
    ctx["schema_count"] = len(index) + 1
    ctx["schema_lists"] = ids
    empty = [i for i, x in enumerate(ids) if not x]
    detail = {"index_starts_at": min(index.values()), "schema_lists": ids, "weights": weights,
              "records_without_any_activation": empty,
              "note": "data-branch index() starts at 0, which core treats as padding; +1 offset applied"}
    return ("FINDING" if empty else "PASS"), detail


@step("5 build core batch (ByteTokenizer, schema mask)", "C-CSAM-1")
def s_batch(ctx):
    import torch
    data_mod = importlib.import_module("ansg_core.data")
    tok = data_mod.ByteTokenizer()
    rows = []
    for r, sids, did in zip(ctx["records"], ctx["schema_lists"], ctx["dialect_ids"]):
        rows.append({"tokens": tok.encode(r.text)[:64], "schema_ids": sids or [0], "dialect_id": did})
    batch = data_mod.collate(rows)
    mask = torch.zeros_like(batch["schema_ids"], dtype=torch.bool)
    for i, row in enumerate(rows):
        mask[i, :len(row["schema_ids"])] = row["schema_ids"] != [0]
    ctx["rows"], ctx["batch"], ctx["mask"] = rows, batch, mask
    return "PASS", {"input_ids": list(batch["input_ids"].shape), "schema_ids": batch["schema_ids"].tolist(),
                    "mask": mask.tolist(), "dialect_ids": batch["dialect_ids"].tolist(),
                    "note": "core collate() returns no schema mask; mask built by the probe"}


@step("6 core ANSG forward/backward (CSAM + DAL) on synthetic batch", "C-CSAM-1/C-DAL-1")
def s_model(ctx):
    import torch
    cfg_mod = importlib.import_module("ansg_core.config")
    model_mod = importlib.import_module("ansg_core.model")
    cfg = cfg_mod.ModelConfig(context_length=64, schema_count=max(ctx["schema_count"], 4))
    b = ctx["batch"]

    def grads_for(schema_ids, dialect_ids):
        torch.manual_seed(0)
        model = model_mod.ANSG(cfg)
        with torch.no_grad():  # emulate a post-step state: zero-initialised LoRA "up" blocks gate gradients at init
            model.dal.up.normal_(std=0.02)
        out = model(b["input_ids"], schema_ids, dialect_ids, labels=b["labels"])
        out["loss"].backward()
        g = lambda p: bool(p.grad is not None and p.grad.abs().sum() > 0)  # noqa: E731
        return model, out, {"csam.query_proj": g(model.csam.query_proj.weight),
                            "csam.heritage_bias": g(model.csam.heritage_bias.weight),
                            "pairwise.relevance_h": g(model.pairwise_cultural_bias.relevance_h.weight),
                            "dal.up": g(model.dal.up), "dal.gate": g(model.dal.gate[0].weight)}

    model, out, one = grads_for(b["schema_ids"], b["dialect_ids"])            # pipeline batch (1 schema/row)
    two_ids = torch.tensor([[1, 2]] * b["input_ids"].shape[0])
    _, _, two = grads_for(two_ids, b["dialect_ids"])                           # 2 schemas/row
    _, _, soft = grads_for(two_ids, None)                                      # no explicit dialect
    ctx["model"], ctx["cfg"] = model, cfg
    finite = bool(torch.isfinite(out["loss"]))
    detail = {"logits": list(out["logits"].shape), "loss_finite": finite,
              "nonzero_grad_1_schema_explicit_dialect": one,
              "nonzero_grad_2_schemas_explicit_dialect": two,
              "nonzero_grad_2_schemas_no_dialect": soft,
              "notes": ["dal.down has zero grad at init because dal.up is zero-initialised (standard LoRA) - expected",
                        "with one active schema CSAM softmax is over a single key, so query/key projections and the "
                        "beta/gamma schema biases receive no gradient",
                        "dal.up is set to small random values here (as after one update); at initialisation it is "
                        "zero and no DAL parameter except dal.up receives gradient",
                        "explicit dialect_ids replace the gate with a one-hot, so the gate is trained only when "
                        "dialect labels are absent"]}
    if not finite or list(out["logits"].shape[:2]) != list(b["input_ids"].shape):
        return "FAIL", detail
    degenerate = (not one["csam.query_proj"]) and two["csam.query_proj"]
    gate_unused = (not two["dal.gate"]) and soft["dal.gate"]
    return ("FINDING" if degenerate or gate_unused else "PASS"), detail


@step("7 schema padding leakage in core model", "C-CSAM-2")
def s_padding(ctx):
    import torch
    data_mod = importlib.import_module("ansg_core.data")
    model = ctx["model"].eval()
    # Fixed non-zero ids so that a real id can never coincide with the pad value 0.
    a = dict(ctx["rows"][0], schema_ids=[1])
    b = dict(ctx["rows"][1], schema_ids=[1, 2])
    alone, together = data_mod.collate([a]), data_mod.collate([a, b])
    n = len(a["tokens"])
    with torch.no_grad():
        la = model(alone["input_ids"], alone["schema_ids"], alone["dialect_ids"])["logits"][0, :n]
        lb = model(together["input_ids"][:, :], together["schema_ids"], together["dialect_ids"])["logits"][0, :n]
    diff = float((la - lb).abs().max())
    return ("FINDING" if diff > 1e-5 else "PASS",
            {"max_abs_logit_diff_same_example": diff,
             "note": "zero-padded schema slots are attended (no mask in ANSG.forward / PairwiseCulturalBias)"})


@step("8 generation", "C-GEN-1")
def s_generate(ctx):
    import torch
    model = ctx["model"]
    b = ctx["batch"]
    torch.manual_seed(0)
    out = model.generate(b["input_ids"][:, :8], b["schema_ids"], b["dialect_ids"], max_new_tokens=4)
    return ("PASS" if out.shape == (b["input_ids"].shape[0], 12) else "FAIL"), {"generated_shape": list(out.shape),
            "note": "untrained synthetic model; output text is meaningless and not evaluated"}


@step("9 evaluation reported separately (CRAFT registry)", "C-EVAL-1")
def s_eval(ctx):
    craft = importlib.import_module("ansg_data.evaluation.craft")
    utils_cfg = importlib.import_module("ansg_data.utils.config")
    core_eval = importlib.import_module("ansg_core.evaluation")
    reg = craft.CraftRegistry.from_config(utils_cfg.load_yaml_config(ctx["data_root"] / "configs/evaluation/craft.yaml"))
    try:
        reg.get("CMPI")([craft.EvaluationSample("synthetic-0", "نص")])
        refused = False
    except craft.MetricNotSpecifiedError:
        refused = True
    core_value = core_eval.weighted_cmpi(0.5, 0.5, 0.5, 0.5)
    return ("FINDING", {"data_branch_CMPI_refused": refused,
                        "core_weighted_cmpi_on_dummy_inputs": core_value,
                        "note": "core exposes a function named weighted_cmpi (Ch.5 §5.1 weights); data branch "
                                "registers CMPI as undefined (C-EVAL-2). Same name, different scientific status."})


@step("10 thesis configuration records agree across branches", "I-PKG-7")
def s_configs(ctx):
    import yaml
    a3 = json.loads((ctx["core_root"] / "configs/thesis_appendix_a3_reported.json").read_text(encoding="utf-8"))
    c5 = json.loads((ctx["core_root"] / "configs/thesis_ch5_reported.json").read_text(encoding="utf-8"))
    v = yaml.safe_load((ctx["data_root"] / "configs/specification/thesis_variants.yaml").read_text(encoding="utf-8"))
    a3y = v["backbone"]["appendix_a3_listing_a2"]
    tp = v["training_protocol_variants"]
    checks = {
        "A.3 model": (a3["model_name"], a3y["model"]),
        "A.3 total params": (a3["reported_total_parameters"], a3y["parameters_total"]),
        "A.3 hidden": (a3["hidden_size"], a3y["hidden_size"]),
        "A.3 heads": (a3["transformer_heads"], a3y["num_attention_heads"]),
        "beta": (a3["beta_heritage"], v["csam"]["beta"]),
        "gamma": (a3["gamma_rhetorical"], v["csam"]["gamma"]),
        "lora rank": (a3["lora_rank"], v["dal"]["rank"]),
        "lora alpha": (a3["lora_alpha"], v["dal"]["lora_alpha"]),
        "A.3 phase1 epochs": (a3["phase1_supervised_epochs"], tp["appendix_a3_listing_a2"]["phase1_supervised"]["epochs"]),
        "A.3 phase1 lr": (a3["phase1_learning_rate"], tp["appendix_a3_listing_a2"]["phase1_supervised"]["lr"]),
        "A.3 rl lr": (a3["rl_learning_rate"], tp["appendix_a3_listing_a2"]["phase3_rl"]["lr"]),
        "A.3 ppo epochs": (a3["ppo_epochs"], tp["appendix_a3_listing_a2"]["phase3_rl"]["ppo_epochs"]),
        "A.3 reward": (a3["reward_weights"], v["reward_variants"]["appendix_a3_a11"]["total"]),
        "Ch.5 total": (c5["reported_total_parameters"], v["backbone"]["ch5_table_5_1"]["parameters_total"]),
        "Ch.5 base": (c5["reported_base_parameters"], v["backbone"]["ch5_table_5_1"]["base_transformer"]),
        "Ch.5 cultural": (c5["reported_cultural_parameters"], v["backbone"]["ch5_table_5_1"]["cultural_modules"]),
        "Ch.5 reward": (c5["reward_weights"], v["reward_variants"]["ch5_5_5_3_and_5_9"]["total"]),
        "Ch.5 transformers": (c5["transformers"], v["hardware_software_variants"]["ch5_5_2"]["transformers"]),
    }
    mismatches = {k: {"core": a, "data": b} for k, (a, b) in checks.items() if a != b}
    # alpha_base 0.370 appears in data branch only as prose (A.25 semantics), not as a field.
    alpha_text = "0.370" in v["csam"]["appendix_a25_code_semantics"]
    detail = {"compared": len(checks), "mismatches": mismatches, "alpha_base_0.370_in_data_register": alpha_text}
    return ("PASS" if not mismatches and alpha_text else "FINDING"), detail


STEPS = [s_ingest, s_norm, s_dia, s_ckg, s_batch, s_model, s_padding, s_generate, s_eval, s_configs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ANSG cross-branch contract probe (synthetic data only)")
    ap.add_argument("--core", required=True, type=Path, help="worktree of implementation/ansg-core-2026")
    ap.add_argument("--data", required=True, type=Path, help="worktree of claude/relaxed-knuth-q63k61")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    load_alias("ansg_core", args.core / "src" / "ansg")
    load_alias("ansg_data", args.data / "src" / "ansg")
    ctx: dict[str, Any] = {"data_root": args.data, "core_root": args.core}
    for s in STEPS:
        s(ctx)
        if RESULTS[-1]["status"] == "FAIL":
            break
    report = {"synthetic_data_only": True, "results": RESULTS}
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    for r in RESULTS:
        print(f"{r['status']:8} {r['step']}  [{r['contract']}]")
    return 1 if any(r["status"] == "FAIL" for r in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
