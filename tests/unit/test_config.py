import pytest

from ansg.utils.config import ConfigError, load_yaml_config


def test_literal_secret_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("db:\n  password: hunter2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="credential"):
        load_yaml_config(p)


def test_env_reference_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("ANSG_TEST_TOKEN", "abc")
    p = tmp_path / "c.yaml"
    p.write_text("api_token: ${env:ANSG_TEST_TOKEN}\nother: ${env:ANSG_UNSET_VAR}\n", encoding="utf-8")
    assert load_yaml_config(p) == {"api_token": "abc", "other": None}


@pytest.mark.parametrize("path", ["configs/data/nights.yaml", "configs/knowledge/resources.yaml",
                                  "configs/knowledge/ckg.yaml", "configs/evaluation/craft.yaml"])
def test_shipped_configs_load(path):
    assert isinstance(load_yaml_config(path), dict)
