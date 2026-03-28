"""
Tests for git utilities — pure logic helpers only (no git subprocess calls).
"""
import pytest
from pathlib import Path
from repolect.git_utils import is_git_repo, get_repo_name, find_repo_root, get_file_hash


def test_is_git_repo_false_for_empty_tmp(tmp_path):
    """A fresh tmp dir has no .git — should return False."""
    assert is_git_repo(str(tmp_path)) is False


def test_is_git_repo_true_with_dotgit(tmp_path):
    """A directory with .git/ should be detected as a repo."""
    (tmp_path / ".git").mkdir()
    assert is_git_repo(str(tmp_path)) is True


def test_find_repo_root_walks_up(tmp_path):
    """find_repo_root should walk up to find a .git directory."""
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    root = find_repo_root(str(nested))
    assert root == str(tmp_path)


def test_find_repo_root_returns_none_when_no_git(tmp_path):
    result = find_repo_root(str(tmp_path))
    assert result is None


def test_get_repo_name_returns_directory_name(tmp_path):
    """Without git remote, should fall back to the directory name."""
    name = get_repo_name(str(tmp_path))
    assert name == tmp_path.name


def test_get_repo_name_strips_trailing_slash():
    """Path with trailing slash should still give the right name."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        name = get_repo_name(d.rstrip("/") + "/")
        assert name == os.path.basename(d.rstrip("/"))


def test_get_file_hash_is_deterministic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = get_file_hash(str(f))
    h2 = get_file_hash(str(f))
    assert h1 == h2
    assert len(h1) == 16


def test_get_file_hash_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello")
    f2.write_text("world")
    assert get_file_hash(str(f1)) != get_file_hash(str(f2))


def test_get_file_hash_returns_empty_for_missing_file():
    result = get_file_hash("/nonexistent/path/file.txt")
    assert result == ""
