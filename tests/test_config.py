"""WATCH_DETAIL resolution and frame_cap mapping."""
from __future__ import annotations

import config


def test_default_detail_is_balanced(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_env_overrides_detail(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "efficient")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "efficient"


def test_invalid_detail_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "bogus")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_get_config_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    cfg = config.get_config()
    assert set(cfg) == {"detail", "whisper", "config_file"}
    assert cfg["whisper"] == "auto"


def test_whisper_preference_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_WHISPER", raising=False)
    env = tmp_path / ".env"
    env.write_text("WATCH_WHISPER=MLX\n")
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    assert config.get_config()["whisper"] == "mlx"  # case-insensitive


def test_whisper_preference_env_overrides_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("WATCH_WHISPER=mlx\n")
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    monkeypatch.setenv("WATCH_WHISPER", "openai")
    assert config.get_config()["whisper"] == "openai"


def test_whisper_preference_invalid_falls_back_to_auto(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("WATCH_WHISPER", "deepgram")
    assert config.get_config()["whisper"] == "auto"


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100
