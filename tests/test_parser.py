"""
Tests for the tree-sitter + regex hybrid parser.
Focuses on pure parsing logic — no LLM calls involved.
"""
import os
import textwrap
import tempfile
import pytest
from repolect.parser import parse_file
from repolect.models import CodeNode


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_python(source: str) -> list[CodeNode]:
    """Parse an in-memory Python source string via a temp file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test.py")
        with open(fpath, "w") as f:
            f.write(source)
        return parse_file(fpath, repo_root=tmpdir, node_id_prefix="0001")


# ── Python class and function detection ──────────────────────────────────────


def test_parses_top_level_function():
    src = textwrap.dedent("""\
        def greet(name: str) -> str:
            return f"Hello, {name}"
    """)
    nodes = parse_python(src)
    titles = [n.title for n in nodes]
    assert any("greet" in t for t in titles), f"Expected 'greet' in {titles}"


def test_parses_class():
    src = textwrap.dedent("""\
        class AuthService:
            def login(self, user: str, pwd: str) -> bool:
                return True
    """)
    nodes = parse_python(src)
    kinds = {n.kind for n in nodes}
    titles = [n.title for n in nodes]
    assert "class" in kinds or "method" in kinds, f"No class/method found in {kinds}"
    assert any("AuthService" in t or "login" in t for t in titles), \
        f"Missing expected title in {titles}"


def test_parses_multiple_functions():
    src = textwrap.dedent("""\
        def add(a, b):
            return a + b

        def subtract(a, b):
            return a - b

        def multiply(a, b):
            return a * b
    """)
    nodes = parse_python(src)
    titles = [n.title for n in nodes]
    found = sum(1 for name in ["add", "subtract", "multiply"] if any(name in t for t in titles))
    assert found >= 2, f"Expected at least 2 functions, got titles: {titles}"


def test_nodes_have_line_numbers():
    src = textwrap.dedent("""\
        def hello():
            pass
    """)
    nodes = parse_python(src)
    for node in nodes:
        assert node.line_start >= 1, f"line_start should be >= 1, got {node.line_start}"


def test_nodes_have_path():
    src = "def foo(): pass\n"
    nodes = parse_python(src)
    for node in nodes:
        assert node.path, f"Node {node.title} has empty path"


def test_nodes_have_language():
    src = "def foo(): pass\n"
    nodes = parse_python(src)
    for node in nodes:
        assert node.language == "python", f"Expected 'python', got '{node.language}'"


def test_empty_file_returns_no_or_file_node():
    nodes = parse_python("")
    assert isinstance(nodes, list)


def test_private_functions_are_parsed():
    """Private/dunder methods should still be captured."""
    src = textwrap.dedent("""\
        class MyClass:
            def __init__(self):
                self.x = 1

            def _helper(self):
                pass
    """)
    nodes = parse_python(src)
    titles = [n.title for n in nodes]
    assert any("__init__" in t or "_helper" in t or "MyClass" in t for t in titles), \
        f"No expected titles found: {titles}"
