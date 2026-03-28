"""
Tests for Repolect core data models: CodeNode, Relation, TreeMeta, SearchResult.
"""
import json
import pytest
from repolect.models import CodeNode, Relation, TreeMeta, SearchResult


# ── Relation ──────────────────────────────────────────────────────────────────


def test_relation_to_dict_and_back():
    rel = Relation(source_id="0001", target_id="0002", kind="CALLS", label="calls login()")
    d = rel.to_dict()
    assert d["source_id"] == "0001"
    assert d["kind"] == "CALLS"
    restored = Relation.from_dict(d)
    assert restored == rel


def test_relation_empty_label():
    rel = Relation(source_id="a", target_id="b", kind="IMPORTS")
    assert rel.label == ""


# ── CodeNode ──────────────────────────────────────────────────────────────────


def make_node(node_id="0001", title="MyClass", kind="class", path="src/foo.py") -> CodeNode:
    return CodeNode(node_id=node_id, title=title, kind=kind, path=path)


def test_code_node_defaults():
    node = make_node()
    assert node.summary == ""
    assert node.children == []
    assert node.relations == []
    assert node.line_start == 0


def test_code_node_to_dict_round_trip():
    child = make_node(node_id="0001.001", title="my_method", kind="method", path="src/foo.py")
    rel = Relation(source_id="0001", target_id="0002", kind="CALLS")
    parent = make_node()
    parent.children = [child]
    parent.relations = [rel]
    parent.summary = "A test class"

    d = parent.to_dict()
    assert len(d["children"]) == 1
    assert d["children"][0]["node_id"] == "0001.001"
    assert len(d["relations"]) == 1

    restored = CodeNode.from_dict(d)
    assert restored.node_id == "0001"
    assert restored.summary == "A test class"
    assert len(restored.children) == 1
    assert restored.children[0].node_id == "0001.001"
    assert len(restored.relations) == 1


def test_code_node_json_round_trip():
    node = make_node()
    node.summary = "Top-level auth module"
    json_str = json.dumps(node.to_dict())
    restored = CodeNode.from_dict(json.loads(json_str))
    assert restored.summary == "Top-level auth module"


def test_flat_iter_single_node():
    node = make_node()
    nodes = list(node.flat_iter())
    assert len(nodes) == 1
    assert nodes[0] is node


def test_flat_iter_with_children():
    root = make_node(node_id="0001", kind="file")
    c1 = make_node(node_id="0001.001", kind="class")
    c2 = make_node(node_id="0001.002", kind="function")
    c1_child = make_node(node_id="0001.001.001", kind="method")
    c1.children = [c1_child]
    root.children = [c1, c2]

    all_ids = [n.node_id for n in root.flat_iter()]
    assert all_ids == ["0001", "0001.001", "0001.001.001", "0001.002"]


def test_get_node_map():
    root = make_node(node_id="0001", kind="file")
    child = make_node(node_id="0001.001", kind="class")
    root.children = [child]
    node_map = root.get_node_map()
    assert "0001" in node_map
    assert "0001.001" in node_map
    assert node_map["0001.001"].title == "MyClass"


def test_from_dict_ignores_extra_keys():
    """from_dict should be robust to unknown keys (forward compatibility)."""
    d = make_node().to_dict()
    d["unknown_future_key"] = "some_value"
    # Should not raise
    node = CodeNode.from_dict(d)
    assert node.node_id == "0001"


# ── TreeMeta ──────────────────────────────────────────────────────────────────


def make_meta(**kwargs) -> TreeMeta:
    defaults = dict(
        repo_name="my_repo",
        repo_path="/home/user/my_repo",
        repo_id="abc123",
        git_commit="deadbeef",
        indexed_at="2026-01-01T00:00:00Z",
    )
    defaults.update(kwargs)
    return TreeMeta(**defaults)


def test_tree_meta_defaults():
    meta = make_meta()
    assert meta.node_count == 0
    assert meta.repolect_version == "0.1.0"
    assert meta.graph_backend == "networkx"


def test_tree_meta_round_trip():
    meta = make_meta(node_count=42, file_count=10, language_stats={"python": 8, "js": 2})
    d = meta.to_dict()
    restored = TreeMeta.from_dict(d)
    assert restored.node_count == 42
    assert restored.language_stats == {"python": 8, "js": 2}


def test_tree_meta_from_dict_ignores_unknown_keys():
    meta = make_meta()
    d = meta.to_dict()
    d["future_field"] = "future_value"
    # Should not raise
    restored = TreeMeta.from_dict(d)
    assert restored.repo_name == "my_repo"


def test_make_repo_id_is_deterministic_and_short():
    repo_id = TreeMeta.make_repo_id("/home/user/my_project")
    assert len(repo_id) == 16
    # Same path → same ID
    assert TreeMeta.make_repo_id("/home/user/my_project") == repo_id


def test_make_repo_id_normalizes_slashes():
    id_unix = TreeMeta.make_repo_id("/home/user/project")
    id_win = TreeMeta.make_repo_id("\\home\\user\\project")
    assert id_unix == id_win


def test_make_repo_id_strips_trailing_slash():
    assert TreeMeta.make_repo_id("/home/user/project/") == TreeMeta.make_repo_id("/home/user/project")


# ── SearchResult ──────────────────────────────────────────────────────────────


def test_search_result_format_for_llm():
    node = make_node(title="login", kind="function", path="src/auth.py")
    node.summary = "Validates user credentials"
    node.language = "python"
    node.line_start = 10
    node.line_end = 40
    result = SearchResult(
        node=node,
        relevance_score=8.5,
        reasoning="directly implements login logic",
        source_snippet="def login(user, pwd): ...",
    )
    formatted = result.format_for_llm()
    assert "login" in formatted
    assert "src/auth.py" in formatted
    assert "Validates user credentials" in formatted
    assert "def login" in formatted


def test_search_result_format_with_related():
    node = make_node(title="login")
    related = make_node(node_id="0002", title="UserService", kind="class", path="src/users.py")
    related.summary = "Handles user CRUD operations"
    result = SearchResult(
        node=node,
        relevance_score=7.0,
        reasoning="core entry point",
        source_snippet="def login(): ...",
        related_nodes=[related],
    )
    formatted = result.format_for_llm()
    assert "UserService" in formatted
    assert "Handles user CRUD" in formatted
