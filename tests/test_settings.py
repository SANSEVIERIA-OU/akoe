# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import os

from akoe.settings import load_env, parse_env


def test_parse_env_handles_comments_quotes_and_export():
    text = '# comment\nHF_TOKEN="abc"\nexport HF_HOME=/data/hf\nEMPTY=\nBAD LINE\n'
    assert parse_env(text) == {"HF_TOKEN": "abc", "HF_HOME": "/data/hf", "EMPTY": ""}


def test_load_env_never_overrides_and_skips_empty(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AKOE_T1=file\nAKOE_T2=file\nAKOE_T3=\n")
    monkeypatch.setenv("AKOE_T1", "already")
    monkeypatch.delenv("AKOE_T2", raising=False)
    monkeypatch.delenv("AKOE_T3", raising=False)
    loaded = load_env(env)
    assert loaded == ["AKOE_T2"]
    assert os.environ["AKOE_T1"] == "already" and os.environ["AKOE_T2"] == "file"
    assert "AKOE_T3" not in os.environ
    monkeypatch.delenv("AKOE_T2")
