# tests/conftest.py
"""
Shared pytest fixtures for Repolect test suite.
"""
import pytest
from repolect.models import CodeNode, TreeMeta


@pytest.fixture
def sample_node():
    """A simple CodeNode for use across tests."""
    return CodeNode(
        node_id="0001",
        title="AuthService",
        kind="class",
        path="src/auth.py",
        line_start=10,
        line_end=80,
        summary="Handles user authentication via JWT",
        language="python",
    )


@pytest.fixture
def sample_tree(sample_node):
    """A small two-level tree."""
    child = CodeNode(
        node_id="0001.001",
        title="login",
        kind="method",
        path="src/auth.py",
        line_start=20,
        line_end=35,
        summary="Validates credentials and returns JWT",
        language="python",
    )
    sample_node.children = [child]
    return sample_node


@pytest.fixture
def sample_meta():
    """A minimal TreeMeta for use across tests."""
    return TreeMeta(
        repo_name="test_repo",
        repo_path="/tmp/test_repo",
        repo_id=TreeMeta.make_repo_id("/tmp/test_repo"),
        git_commit="abc123",
        indexed_at="2026-01-01T00:00:00Z",
        node_count=5,
        file_count=2,
    )
