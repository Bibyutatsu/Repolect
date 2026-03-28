"""
Tests for the Repolect configuration system.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from repolect.config import load_config, DEFAULT_CONFIG


# ── load_config ───────────────────────────────────────────────────────────────


def test_load_config_returns_defaults_when_no_file(tmp_path):
    """With no config file, load_config should return DEFAULT_CONFIG values."""
    fake_global = tmp_path / ".repolect" / "config.yaml"
    fake_local = tmp_path / "project" / ".repolect" / "config.yaml"

    with patch("repolect.config.GLOBAL_CONFIG_FILE", fake_global):
        cfg = load_config(repo_root=str(tmp_path / "project"))

    assert cfg["provider"] == DEFAULT_CONFIG["provider"]
    assert cfg["model_name"] == DEFAULT_CONFIG["model_name"]


def test_load_config_reads_global_file(tmp_path):
    config_dir = tmp_path / ".repolect"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text("provider: openai-compatible\nmodel_name: gpt-4o\n")

    with patch("repolect.config.GLOBAL_CONFIG_FILE", config_file):
        cfg = load_config(repo_root=str(tmp_path))

    assert cfg["provider"] == "openai-compatible"
    assert cfg["model_name"] == "gpt-4o"


def test_local_config_overrides_global(tmp_path):
    """Per-repo config.yaml should override global settings."""
    global_dir = tmp_path / ".repolect"
    global_dir.mkdir()
    global_file = global_dir / "config.yaml"
    global_file.write_text("provider: ollama\nmodel_name: qwen3.5:4b\n")

    repo_dir = tmp_path / "myproject"
    repo_dir.mkdir()
    local_dir = repo_dir / ".repolect"
    local_dir.mkdir()
    local_file = local_dir / "config.yaml"
    local_file.write_text("model_name: gpt-4o-mini\n")

    with patch("repolect.config.GLOBAL_CONFIG_FILE", global_file):
        cfg = load_config(repo_root=str(repo_dir))

    # Global: ollama; local overrides model_name
    assert cfg["provider"] == "ollama"
    assert cfg["model_name"] == "gpt-4o-mini"


def test_env_var_overrides_config(tmp_path):
    """Environment variables should override both global and local config."""
    config_dir = tmp_path / ".repolect"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text("provider: ollama\n")

    with patch("repolect.config.GLOBAL_CONFIG_FILE", config_file):
        with patch.dict(os.environ, {"REPOLECT_PROVIDER": "openai-compatible"}):
            cfg = load_config(repo_root=str(tmp_path))

    assert cfg["provider"] == "openai-compatible"


def test_load_config_has_all_default_keys(tmp_path):
    """Returned config must always contain all DEFAULT_CONFIG keys."""
    with patch("repolect.config.GLOBAL_CONFIG_FILE", tmp_path / "nonexistent.yaml"):
        cfg = load_config(repo_root=str(tmp_path))

    for key in DEFAULT_CONFIG:
        assert key in cfg, f"Missing expected config key: {key}"
