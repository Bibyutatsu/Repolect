"""
Repolect — Core Data Models
 
Defines CodeNode (hierarchical semantic tree), Relation (graph edges),
TreeMeta (index metadata), and SearchResult (query output).
 
The tree (tree.json) stores structure + LLM summaries.  The graph
(NetworkX or FalkorDB) stores CALLS/IMPORTS/EXTENDS relations.
Both stores are built at index time; the LLM reasons over summaries
at search time.
"""
 
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json
import hashlib
 
 
NodeKind = Literal["repo", "module", "file", "class", "function", "method", "doc", "interface"]
RelationKind = Literal["CALLS", "IMPORTS", "EXTENDS", "IMPLEMENTS", "DOCS", "DEFINES"]
 
 
@dataclass
class Relation:
    source_id: str
    target_id: str          # node_id or "external:module_name" for unresolvable
    kind: RelationKind
    label: str = ""         # optional human label e.g. "calls login() on line 42"
 
    def to_dict(self) -> dict:
        return asdict(self)
 
    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(**d)
 
 
@dataclass
class CodeNode:
    """
    One node in the semantic tree. Every file, class, function, and doc
    is a node with an LLM-generated summary that enables vectorless reasoning.
    """
    node_id: str            # dot-notation: "0001", "0001.003", "0001.003.007"
    title: str              # human-readable: "AuthService", "src/auth/", "login()"
    kind: NodeKind
    path: str               # relative path from repo root (empty for repo node)
    line_start: int = 0
    line_end: int = 0
    summary: str = ""       # LLM-generated. The KEY field for vectorless search.
    signature: str = ""     # function/class signature if applicable
    language: str = ""      # "python", "typescript", etc.
    git_hash: str = ""      # SHA256 of source at index time (change detection)
    doc_ref: Optional[str] = None   # node_id of paired documentation node
    relations: list[Relation] = field(default_factory=list)
    children: list["CodeNode"] = field(default_factory=list)
 
    # --- Extended metadata ---
    change_frequency: int = 0       # how many times this node changed across git history
    complexity_score: float = 0.0   # cyclomatic complexity (future)
    usage_count: int = 0            # how many other nodes CALL/IMPORT this one
 
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "kind": self.kind,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "summary": self.summary,
            "signature": self.signature,
            "language": self.language,
            "git_hash": self.git_hash,
            "doc_ref": self.doc_ref,
            "change_frequency": self.change_frequency,
            "complexity_score": self.complexity_score,
            "usage_count": self.usage_count,
            "relations": [r.to_dict() for r in self.relations],
            "children": [c.to_dict() for c in self.children],
        }
 
    @classmethod
    def from_dict(cls, d: dict) -> "CodeNode":
        relations = [Relation.from_dict(r) for r in d.pop("relations", [])]
        children_raw = d.pop("children", [])
        node = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        node.relations = relations
        node.children = [CodeNode.from_dict(c) for c in children_raw]
        return node
 
    def flat_iter(self):
        """Depth-first iterator over this node and all descendants."""
        yield self
        for child in self.children:
            yield from child.flat_iter()
 
    def get_node_map(self) -> dict[str, "CodeNode"]:
        """Build flat {node_id: node} dict for O(1) lookups."""
        return {node.node_id: node for node in self.flat_iter()}
 
    def __repr__(self):
        return f"CodeNode({self.kind}:{self.node_id} '{self.title}')"
 
 
from .version import __version__


@dataclass
class TreeMeta:
    """Metadata stored alongside the tree. Used for change detection and registry."""
    repo_name: str
    repo_path: str              # absolute path
    repo_id: str                # SHA256 of normalized absolute path (collision-safe)
    git_commit: str             # HEAD at index time
    indexed_at: str             # ISO timestamp
    node_count: int = 0
    git_branch: str = ""        # branch name at index time
    file_count: int = 0
    language_stats: dict = field(default_factory=dict)   # {"python": 42, "js": 10}
    repolect_version: str = __version__
    index_duration_seconds: float = 0.0
    embeddings_enabled: bool = False
    graph_backend: str = "networkx"
 
    def to_dict(self) -> dict:
        return asdict(self)
 
    @classmethod
    def from_dict(cls, d: dict) -> "TreeMeta":
        # Backward compat: ignore keys not in our dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
 
    @staticmethod
    def make_repo_id(absolute_path: str) -> str:
        normalized = str(absolute_path).replace("\\", "/").rstrip("/").lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
 
 
@dataclass
class SearchResult:
    """Returned by the tree search engine."""
    node: CodeNode
    relevance_score: float          # 0–10, assigned by LLM reasoning
    reasoning: str                  # why the LLM thinks this is relevant
    source_snippet: str             # actual lines of code
    related_nodes: list[CodeNode] = field(default_factory=list)  # 1-hop neighbors
 
    def format_for_llm(self) -> str:
        """Format this result for inclusion in an LLM prompt."""
        lines = [
            f"=== {self.node.title} ===",
            f"File: {self.node.path}  Lines: {self.node.line_start}–{self.node.line_end}",
            f"Type: {self.node.kind}  Language: {self.node.language}",
            f"Summary: {self.node.summary}",
            "",
            self.source_snippet,
        ]
        if self.related_nodes:
            lines.append("\nRelated components:")
            for rel_node in self.related_nodes[:3]:
                lines.append(f"  - {rel_node.title} ({rel_node.path}): {rel_node.summary[:80]}...")
        return "\n".join(lines)
 