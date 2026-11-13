import pytest

from ansg.utils.config import ConfigError, load_yaml_config


def test_literal_secret_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("db:\n  password: hunter2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="credential"):
        load_yaml_config(p)


@pytest.mark.parametrize("key", ["tokenizer", "max_new_tokens", "secretary_note", "author"])
def test_non_secret_keys_not_flagged(tmp_path, key):
    p = tmp_path / "c.yaml"
    p.write_text(f"{key}: some value\n", encoding="utf-8")
    assert load_yaml_config(p) == {key: "some value"}


@pytest.mark.parametrize("key", ["api_key", "ApiKey", "db_password", "hf-token", "client_secret", "aws_access_key"])
def test_secret_keys_flagged(tmp_path, key):
    p = tmp_path / "c.yaml"
    p.write_text(f"{key}: literal\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml_config(p)


def test_env_reference_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("ANSG_TEST_TOKEN", "abc")
    p = tmp_path / "c.yaml"
    p.write_text("api_token: ${env:ANSG_TEST_TOKEN}\nother: ${env:ANSG_UNSET_VAR}\n", encoding="utf-8")
    assert load_yaml_config(p) == {"api_token": "abc", "other": None}


@pytest.mark.parametrize("path", ["configs/data/nights.yaml", "configs/knowledge/resources.yaml",
                                  "configs/knowledge/ckg.yaml", "configs/evaluation/craft.yaml",
                                  "configs/specification/thesis_variants.yaml"])
def test_shipped_configs_load(path):
    assert isinstance(load_yaml_config(path), dict)


def test_thesis_variants_keep_conflicts_unresolved():
    v = load_yaml_config("configs/specification/thesis_variants.yaml")
    assert {"ch4_4_1_6", "ch5_table_5_1", "appendix_a3_listing_a2"} <= set(v["backbone"])
    assert len(v["reward_variants"]) >= 5
    assert len(v["training_protocol_variants"]) >= 5
