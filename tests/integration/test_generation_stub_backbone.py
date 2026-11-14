"""End-to-end ``ansg-generate`` contract on a tiny SYNTHETIC backbone (Issue #15).

The backbone is created locally by ``ansg.generation.stub`` (random GPT-2-style
weights, byte-level BPE trained on a few synthetic sentences). Nothing is
downloaded. All generated text is meaningless test output.

Some tests copy the stub and rewrite its ``ansg_backbone.json`` to
``"synthetic": false`` ("declared-real stub"). That exists ONLY to exercise the
labelling gate's non-synthetic branches; it is never a pretrained model and is
deleted with the temporary directory. Likewise ``data_kind="real"`` below is a
gate-logic test on synthetic text, never evidence.
"""
import json
import shutil

import pytest

torch = pytest.importorskip("torch", reason="model extra (PyTorch) not installed")
pytest.importorskip("transformers", reason="pretrained extra (transformers) not installed")
pytestmark = pytest.mark.model

from ansg.data.training_examples import SchemaVocabulary  # noqa: E402
from ansg.generation import cli  # noqa: E402
from ansg.generation.artifacts import BACKBONE_PROVENANCE_FILE, cultural_manifest_path, digest_of, file_inventory  # noqa: E402
from ansg.generation.cultural import load_example_rows, save_cultural_modules, train_cultural_modules  # noqa: E402
from ansg.generation.readiness import assess  # noqa: E402
from ansg.generation.runtime import GenerationError, build_cultural, load_backbone  # noqa: E402
from ansg.generation.stub import make_stub_backbone  # noqa: E402
from ansg.knowledge.ckg import CulturalKnowledgeGraph  # noqa: E402
from ansg.utils.provenance import sha256_file  # noqa: E402

PROMPT = "كان يا ما كان في قديم الزمان"
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    root = tmp_path_factory.mktemp("stub") / "backbone"
    make_stub_backbone(root, seed=0)
    return root


@pytest.fixture(scope="module")
def declared_real(stub, tmp_path_factory):
    root = tmp_path_factory.mktemp("declared") / "backbone"
    shutil.copytree(stub, root)
    prov = json.loads((root / BACKBONE_PROVENANCE_FILE).read_text())
    prov.update(name="TEST ONLY: synthetic stub declared non-synthetic to exercise the gate", synthetic=False)
    (root / BACKBONE_PROVENANCE_FILE).write_text(json.dumps(prov, ensure_ascii=False))
    return root


@pytest.fixture(scope="module")
def schema(tmp_path_factory):
    """Synthetic Nights fixture -> ANHC records -> ansg-build-examples (vocabulary, examples, manifest)."""
    from pathlib import Path

    from ansg.data.anhc import write_jsonl
    from ansg.data.nights import ingest_file
    from ansg.data.records import CopyrightStatus, SourceProvenance
    from ansg.data.training_examples import main as build_examples
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    base = tmp_path_factory.mktemp("examples")
    prov = SourceProvenance(source_id="synthetic", title="synthetic Nights fixture",
                            copyright_status=CopyrightStatus.PUBLIC_DOMAIN, redistributable=True)
    records, _ = ingest_file(fixtures / "nights" / "synthetic_nights.txt", prov)
    write_jsonl([r for r in records if r.narrative.title], base / "records" / "train.jsonl")
    ckg = fixtures / "ckg" / "synthetic_ckg.json"
    assert build_examples(["--records-dir", str(base / "records"), "--ckg", str(ckg),
                           "--output", str(base / "out")]) == 0
    vocab = SchemaVocabulary.from_graph(CulturalKnowledgeGraph.from_json(ckg), source_path=ckg)
    return {"dir": base / "out", "vocab": base / "out" / "schema_vocabulary.json", "ckg": ckg,
            "examples": base / "out" / "train.examples.jsonl", "node_id": vocab.node_to_id["motif:test_magic_ring"]}


def run(args, capsys):
    rc = cli.main(args)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def gen(backbone, tmp_path, capsys, *extra, name="gen.json"):
    out = tmp_path / name
    rc, _, err = run(["--backbone", str(backbone), "--prompt", PROMPT, "--max-new-tokens", "12",
                      "--device", "cpu", "--output", str(out), *extra], capsys)
    assert rc == 0, err
    return json.loads(out.read_text(encoding="utf-8")), err


# ----------------------------------------------------------------- backbone-only path
def test_stub_generation_is_labelled_synthetic_and_reproducible(stub, tmp_path, capsys):
    a, err = gen(stub, tmp_path, capsys, "--num-samples", "2", name="a.json")
    b, _ = gen(stub, tmp_path, capsys, "--num-samples", "2", name="b.json")
    assert a["label"]["mode"] == "synthetic_stub" and "SYNTHETIC STUB" in err
    assert a["label"]["reproduces_dissertation"] is False
    assert a["backbone"]["hash_verified"] is True and a["backbone"]["synthetic"] is True
    assert a["prompt"]["text"] == PROMPT and a["prompt"]["tokenizer_roundtrip_exact"] is True
    assert [s["seed"] for s in a["samples"]] == [42, 43]
    assert [s["token_ids_sha256"] for s in a["samples"]] == [s["token_ids_sha256"] for s in b["samples"]]
    assert a["samples"][0]["token_ids_sha256"] != a["samples"][1]["token_ids_sha256"]
    assert all(s["text"].startswith(PROMPT) for s in a["samples"])
    assert a["tokenizer"]["aligned"] and a["generation_config"]["top_p"] == 0.95


def test_json_stdout_is_utf8_arabic(stub, capsys):
    rc, out, _ = run(["--backbone", str(stub), "--prompt", PROMPT, "--max-new-tokens", "3", "--device", "cpu",
                      "--json"], capsys)
    assert rc == 0 and PROMPT in out and "\\u06" not in out  # not ASCII-escaped
    assert json.loads(out)["schema"] == "ansg.generation_record/1"


def test_greedy_decoding_is_deterministic_across_seeds(stub, tmp_path, capsys):
    a, _ = gen(stub, tmp_path, capsys, "--temperature", "0", "--seed", "1", name="g1.json")
    b, _ = gen(stub, tmp_path, capsys, "--temperature", "0", "--seed", "2", name="g2.json")
    assert a["generation_config"]["decoding"] == "greedy"
    assert a["samples"][0]["token_ids_sha256"] == b["samples"][0]["token_ids_sha256"]


def test_backbone_hash_is_enforced(stub, tmp_path, capsys):
    copy = tmp_path / "bb"
    shutil.copytree(stub, copy)
    (copy / "config.json").write_text((copy / "config.json").read_text() + "\n")
    rc, _, err = run(["--backbone", str(copy), "--prompt", PROMPT, "--device", "cpu"], capsys)
    assert rc == 2 and "digest mismatch" in err
    prov = json.loads((copy / BACKBONE_PROVENANCE_FILE).read_text())
    del prov["directory_digest"]
    (copy / BACKBONE_PROVENANCE_FILE).write_text(json.dumps(prov))
    rc, _, err = run(["--backbone", str(copy), "--prompt", PROMPT, "--device", "cpu"], capsys)
    assert rc == 2 and "no expected backbone digest" in err
    digest = digest_of(file_inventory(copy))
    rec, _ = gen(copy, tmp_path, capsys, "--backbone-sha256", digest, name="v.json")
    assert rec["backbone"]["hash_verified"] and rec["backbone"]["expected_digest_source"] == "command line"
    rec, _ = gen(copy, tmp_path, capsys, "--allow-unverified-backbone", name="u.json")
    assert rec["backbone"]["hash_verified"] is False
    assert any("WITHOUT an expected digest" in w for w in rec["label"]["warnings"])


def test_hub_ids_are_not_downloaded(capsys):
    rc, _, err = run(["--backbone", "aubmindlab/aragpt2-base", "--prompt", PROMPT], capsys)
    assert rc == 2 and "automatic downloads are disabled" in err


def test_invalid_requests(stub, capsys):
    base = ["--backbone", str(stub), "--prompt", PROMPT, "--device", "cpu"]
    for extra, msg in [(["--dialect", "Egyptian"], "backbone-only run"),
                       (["--schema-node", "motif:test_magic_ring"], "backbone-only run"),
                       (["--untrained-cultural-modules"], "--schema-vocabulary"),
                       (["--max-new-tokens", "500"], "exceeds the backbone context"),
                       (["--top-p", "0"], "top_p")]:
        rc, _, err = run(base + extra, capsys)
        assert rc == 2 and msg in err, (extra, err)


def test_backbone_only_label_on_declared_real_stub(declared_real, tmp_path, capsys):
    rec, err = gen(declared_real, tmp_path, capsys)
    assert rec["label"]["mode"] == "backbone_only" and "BACKBONE-ONLY" in err
    assert rec["cultural_modules"] is None and rec["label"]["cultural_conditioning_claimed"] is False


# ----------------------------------------------------------------- cultural modules
def test_untrained_bridge_reproduces_the_backbone_exactly(declared_real, schema):
    """Bridge v2: attaching untrained CSAM/DAL must not change the pretrained distribution."""
    bb = load_backbone(declared_real, device=CPU, dtype=torch.float32)
    cult = build_cultural(bb, schema_vocabulary=schema["vocab"], checkpoint=None, init_seed=3)
    ids = torch.tensor([bb.tokenizer(PROMPT)["input_ids"]])
    with torch.no_grad():
        plain = bb.model(ids).logits
        for sid, dia in ((torch.tensor([[schema["node_id"]]]), torch.tensor([1])),
                         (torch.tensor([[0]]), None)):
            bridged = cult.model(ids, sid, dia)["logits"]
            assert torch.equal(bridged, plain)


def _activate(cult):
    """Give the modules non-trivial weights (as training would) so the bridge is not a no-op."""
    with torch.no_grad():
        cult.model.cultural_gate.fill_(0.5)
        cult.model.dal.up.normal_(std=0.05)


def test_kv_cache_matches_full_forward(declared_real, schema):
    bb = load_backbone(declared_real, device=CPU, dtype=torch.float32)
    cult = build_cultural(bb, schema_vocabulary=schema["vocab"], checkpoint=None, init_seed=3)
    _activate(cult)
    ids = torch.tensor([bb.tokenizer(PROMPT)["input_ids"]])
    sid = torch.tensor([[schema["node_id"]]])
    dia = torch.tensor([1])
    with torch.no_grad():
        full = cult.model(ids, sid, dia)["logits"][0, -1]
        first = cult.model(ids[:, :-1], sid, dia, use_cache=True)
        step = cult.model(ids[:, -1:], sid, dia, past_key_values=first["past_key_values"], use_cache=True)
        plain = bb.model(ids).logits[0, -1]
    torch.testing.assert_close(step["logits"][0, -1], full, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(full, plain)  # active modules do change the distribution


def test_cultural_gate_receives_gradient_at_initialisation(declared_real, schema):
    bb = load_backbone(declared_real, device=CPU, dtype=torch.float32)
    cult = build_cultural(bb, schema_vocabulary=schema["vocab"], checkpoint=None, init_seed=3)
    ids = torch.tensor([bb.tokenizer(PROMPT)["input_ids"]])
    out = cult.model(ids, torch.tensor([[schema["node_id"]]]), torch.tensor([2]), labels=ids)
    out["loss"].backward()
    assert cult.model.cultural_gate.grad is not None and cult.model.cultural_gate.grad.abs() > 0
    assert cult.model.dal.up.grad.abs().sum() > 0


def test_untrained_cultural_modules_are_labelled_untrained(declared_real, schema, tmp_path, capsys):
    rec, err = gen(declared_real, tmp_path, capsys, "--untrained-cultural-modules", "--schema-vocabulary",
                   str(schema["vocab"]), "--schema-node", "motif:test_magic_ring", "--dialect", "Egyptian")
    assert rec["label"]["mode"] == "ansg_untrained" and "UNTRAINED RESEARCH PROTOTYPE" in err
    assert rec["label"]["dialect_adaptation_claimed"] is False and "no dialect adaptation" in err
    cm = rec["cultural_modules"]
    assert cm["source"].startswith("freshly initialised") and cm["dialect_id"] == 1
    assert cm["schema_ids"] == [schema["node_id"]] and cm["manifest"] is None
    rc, _, err = run(["--backbone", str(declared_real), "--prompt", PROMPT, "--untrained-cultural-modules",
                      "--schema-vocabulary", str(schema["vocab"]), "--schema-node", "motif:absent",
                      "--max-new-tokens", "5"], capsys)
    assert rc == 2 and "not in the schema vocabulary" in err


def _train(backbone, schema, dest, data_kind, steps=2):
    bb = load_backbone(backbone, device=CPU, dtype=torch.float32)
    cult = build_cultural(bb, schema_vocabulary=schema["vocab"], checkpoint=None, init_seed=0)
    rows = load_example_rows(schema["examples"], cult.schema_vocabulary.schema_count)
    record = train_cultural_modules(cult, bb, rows, steps=steps, batch_size=2, max_length=64, seed=0)
    assert len(record["loss_per_step"]) == steps
    return save_cultural_modules(cult, bb, dest, training=record, data_kind=data_kind,
                                 data_files={"examples": schema["examples"], "schema_vocabulary": schema["vocab"]})


def test_synthetic_training_roundtrip_and_readiness(declared_real, schema, tmp_path, capsys):
    ckpt = tmp_path / "cultural.pt"
    manifest = _train(declared_real, schema, ckpt, "synthetic")
    assert manifest["training"]["status"] == "trained" and manifest["checkpoint_sha256"] == sha256_file(ckpt)
    with pytest.raises(GenerationError, match="refusing to overwrite"):
        _train(declared_real, schema, ckpt, "synthetic")
    rec, err = gen(declared_real, tmp_path, capsys, "--cultural-checkpoint", str(ckpt), "--schema-vocabulary",
                   str(schema["vocab"]), "--schema-node", "motif:test_magic_ring")
    assert rec["label"]["mode"] == "ansg_trained_synthetic_only" and "SYNTHETIC data only" in err
    assert rec["cultural_modules"]["checkpoint"]["sha256"] == manifest["checkpoint_sha256"]
    report = assess(backbone=str(declared_real), cultural_checkpoint=str(ckpt), schema_vocabulary=str(schema["vocab"]),
                    ckg=str(schema["ckg"]), examples_dir=str(schema["dir"]), generation_records=[str(tmp_path / "gen.json")])
    assert report["level"] == "ansg_trained_synthetic_only", report["checks"]
    assert report["generation_exercised"] is True


def test_gate_logic_for_declared_real_training(declared_real, schema, tmp_path, capsys):
    """Gate logic only: synthetic text declared 'real' inside a temp dir. Never evidence."""
    ckpt = tmp_path / "cultural.pt"
    _train(declared_real, schema, ckpt, "real", steps=1)
    rec, err = gen(declared_real, tmp_path, capsys, "--cultural-checkpoint", str(ckpt), "--schema-vocabulary",
                   str(schema["vocab"]), "--dialect", "Gulf")
    assert rec["label"]["mode"] == "ansg_trained" and rec["label"]["dialect_adaptation_claimed"] is True
    assert rec["label"]["reproduces_dissertation"] is False and "not a reproduction" in err
    report = assess(backbone=str(declared_real), cultural_checkpoint=str(ckpt), schema_vocabulary=str(schema["vocab"]),
                    ckg=str(schema["ckg"]), examples_dir=str(schema["dir"]))
    assert report["level"] == "trained_ansg", report["checks"]


def test_cultural_checkpoint_verification(declared_real, stub, schema, tmp_path, capsys):
    ckpt = tmp_path / "c.pt"
    _train(declared_real, schema, ckpt, "synthetic", steps=1)
    args = ["--prompt", PROMPT, "--device", "cpu", "--max-new-tokens", "5", "--cultural-checkpoint", str(ckpt),
            "--schema-vocabulary", str(schema["vocab"])]
    # The stub has the same weights as `declared_real` (only the excluded sidecar differs), so it
    # is accepted; a stub with other random weights is a different backbone and is refused.
    assert run(["--backbone", str(stub)] + args, capsys)[0] == 0
    other = tmp_path / "other"
    make_stub_backbone(other, seed=1)
    rc, _, err = run(["--backbone", str(other)] + args, capsys)
    assert rc == 2 and "different backbone" in err
    tampered = tmp_path / "t.pt"
    shutil.copy(ckpt, tampered)
    shutil.copy(cultural_manifest_path(ckpt), cultural_manifest_path(tampered))
    with open(tampered, "ab") as fh:
        fh.write(b"\0")
    tampered_args = ["--backbone", str(declared_real), "--prompt", PROMPT, "--device", "cpu", "--max-new-tokens", "5",
                     "--cultural-checkpoint", str(tampered), "--schema-vocabulary", str(schema["vocab"])]
    rc, _, err = run(tampered_args, capsys)
    assert rc == 2 and "SHA-256 differs" in err
    cultural_manifest_path(tampered).unlink()
    rc, _, err = run(tampered_args, capsys)
    assert rc == 2 and "manifest not found" in err
    rc, _, err = run(["--backbone", str(declared_real), "--untrained-cultural-modules"] + args, capsys)
    assert rc == 2 and "not both" in err


def test_generation_opens_no_network_connection(stub, tmp_path):
    """A socket audit hook in a fresh interpreter sees no connection during load + generate."""
    import subprocess
    import sys
    code = (
        "import sys, json\n"
        "events = []\n"
        "sys.addaudithook(lambda e, a: events.append(e) if e in ('socket.connect', 'socket.getaddrinfo') else None)\n"
        "from ansg.generation.cli import main\n"
        f"rc = main(['--backbone', {str(stub)!r}, '--prompt', 'كان', '--max-new-tokens', '4', '--device', 'cpu'])\n"
        "print(json.dumps({'rc': rc, 'events': events}))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={k: v for k, v in __import__("os").environ.items() if not k.startswith("HF_")})
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {"rc": 0, "events": []}


# ----------------------------------------------------------------- plans and chat templates
def _plan_file(tmp_path, *, coherent=True):
    from ansg.narrative.events import Entity, EntityKind, Event, EventGraph, Modality, Participant, Proposition, \
        Relation, RelationKind
    from ansg.narrative.plan import StoryPlan
    ents = [Entity("a", "سالم", EntityKind.CHARACTER, "protagonist"), Entity("box", "صندوق", EntityKind.OBJECT)]
    evs = [Event("p1", "وجد", None, 0, [Participant("Agent", "a"), Participant("Theme", "box")], Modality.PLANNED,
                 gloss="سالم يجد صندوقًا", effects=[Proposition("has", ("a", "box"))]),
           Event("p2", "فتح", None, 1, [Participant("Agent", "a"), Participant("Patient", "box")], Modality.PLANNED,
                 gloss="سالم يفتح الصندوق", preconditions=[Proposition("has", ("a", "box"))])]
    rels = [Relation(RelationKind.ENABLES, "p1", "p2") if coherent else Relation(RelationKind.BEFORE, "p2", "p1")]
    path = tmp_path / ("plan.json" if coherent else "bad_plan.json")
    path.write_text(json.dumps(StoryPlan("aqaba", "الصندوق", EventGraph("plan", None, ents, evs, rels)).to_dict(),
                               ensure_ascii=False), encoding="utf-8")
    return path


def test_generation_from_a_validated_plan(tmp_path, capsys):
    stub = tmp_path / "long_stub"
    make_stub_backbone(stub, seed=0, context=1024)   # the rendered plan prompt is long in byte-level BPE
    out = tmp_path / "plan_gen.json"
    rc, _, err = run(["--backbone", str(stub), "--plan", str(_plan_file(tmp_path)), "--max-new-tokens", "4",
                      "--device", "cpu", "--output", str(out)], capsys)
    assert rc == 0, err
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["plan"]["setting"] == "aqaba" and rec["plan"]["validation"]["errors"] == 0
    assert "العقبة" in rec["prompt"]["text"] and rec["label"]["mode"] == "synthetic_stub"
    rc, _, err = run(["--backbone", str(stub), "--plan", str(_plan_file(tmp_path, coherent=False)),
                      "--max-new-tokens", "4", "--device", "cpu"], capsys)
    assert rc == 2 and "E-PRECONDITION-NOT-ESTABLISHED" in err   # opens the box before finding it


def test_chat_template_is_explicit_and_recorded(stub, tmp_path, capsys):
    rc, _, err = run(["--backbone", str(stub), "--prompt", PROMPT, "--chat-template", "--device", "cpu"], capsys)
    assert rc == 2 and "no chat template" in err
    rc, _, err = run(["--backbone", str(stub), "--prompt", PROMPT, "--system-prompt", "x", "--device", "cpu",
                      "--max-new-tokens", "3"], capsys)
    assert rc == 2 and "requires --chat-template" in err
    copy = tmp_path / "chat_stub"
    shutil.copytree(stub, copy)
    cfg = json.loads((copy / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg["chat_template"] = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}\n{% endfor %}[assistant] "
    (copy / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    rec, _ = gen(copy, tmp_path, capsys, "--chat-template", "--system-prompt", "أنت راوٍ عربي",
                 "--backbone-sha256", digest_of(file_inventory(copy)), "--max-new-tokens", "4", name="chat.json")
    assert rec["prompt"]["chat_template"] is True and rec["prompt"]["system_prompt"] == "أنت راوٍ عربي"
    assert rec["prompt"]["tokenizer_roundtrip_exact"] is True
