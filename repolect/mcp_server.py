"""
Repolect — MCP Server
Exposes Repolect tools via the Model Context Protocol (MCP) for integration
with AI editors like Claude Code, Cursor, Windsurf, etc.
 
Uses the FastMCP decorator API from mcp.server.fastmcp.
 
Install: pip install repolect[mcp]
"""
 
from __future__ import annotations
 
import json
import logging
import os
from pathlib import Path
from typing import Any
 
try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Context cache — keyed by (repo_path, branch), with mtime-based invalidation
# ---------------------------------------------------------------------------
 
_ctx_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
 
 
def _get_repo_root() -> str | None:
    from .git_utils import detect_repo_root
    explicit = os.environ.get("REPOLECT_REPO")
    if explicit and Path(explicit).exists():
        return explicit
    return detect_repo_root(".")
 
 
def _resolve_repo(repo: str | None) -> str:
    """Resolve a repo name/path/id to an absolute repo_root path."""
    if repo:
        p = Path(repo)
        if p.is_dir():
            return str(p.resolve())
        from .storage import find_repo
        entry = find_repo(repo)
        if entry:
            return entry["repo_path"]
        raise RuntimeError(f"Repository '{repo}' not found in registry.")
    root = _get_repo_root()
    if not root:
        raise RuntimeError(
            "Cannot detect repo root. Set REPOLECT_REPO or run from a project directory."
        )
    return root
 
 
def _auto_detect_branch(repo_root: str) -> str | None:
    """Auto-detect the branch for index loading.
 
    Priority:
      1. Current git branch (if an index exists for it)
      2. First available branch under .repolect/branches/
      3. None (legacy non-branch index)
    """
    from .git_utils import get_current_branch
    from .storage import get_index_dir
 
    current = get_current_branch(repo_root)
    if current:
        idx = get_index_dir(repo_root, branch=current)
        if (idx / "tree.json").exists():
            return current
 
    branches_dir = Path(repo_root) / ".repolect" / "branches"
    if branches_dir.is_dir():
        for d in sorted(branches_dir.iterdir()):
            if d.is_dir() and (d / "tree.json").exists():
                return d.name
 
    return None
 
 
def _load_context(
    repo: str | None = None, branch: str | None = None,
):
    """Load tree, provider, graph, and embedder.
 
    Results are cached per (repo_path, branch).  On subsequent calls the
    tree.json mtime is checked — if it changed (e.g. after ``repolect sync``),
    the cache entry is refreshed automatically.
 
    When *branch* is None, auto-detects the current git branch and falls
    back to scanning .repolect/branches/ for the first available index.
    """
    repo_root = _resolve_repo(repo)
 
    from .storage import load_tree, get_index_dir
 
    if branch is None:
        branch = _auto_detect_branch(repo_root)
 
    cache_key = (repo_root, branch)
 
    index_dir = get_index_dir(repo_root, branch=branch)
    tree_path = index_dir / "tree.json"
    current_mtime = tree_path.stat().st_mtime if tree_path.exists() else 0.0
 
    cached = _ctx_cache.get(cache_key)
    if cached and cached.get("_mtime") == current_mtime:
        return (
            cached["repo_root"], cached["root"],
            cached["provider"], cached["graph_db"],
            cached["embedder"],
        )
 
    from .summarizer import get_provider
    from .graph_db import GraphDB
 
    root = load_tree(repo_root, branch=branch)
 
    provider = None
    try:
        provider = get_provider()
    except Exception:
        pass
 
    graph_db = None
    if (index_dir / "graph.pkl").exists() or (index_dir / "graph.db").exists():
        try:
            graph_db = GraphDB.open(index_dir)
        except Exception:
            pass
 
    embedder = None
    if graph_db and graph_db.has_embeddings():
        try:
            from .embedder import get_embedder
            embedder = get_embedder()
        except Exception:
            pass
 
    _ctx_cache[cache_key] = dict(
        repo_root=repo_root, root=root, provider=provider,
        graph_db=graph_db, embedder=embedder, _mtime=current_mtime,
    )
    return repo_root, root, provider, graph_db, embedder
 
 
def _read_source(file_path: Path, start: int, end: int) -> str:
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(0, start - 1)
        e = min(len(lines), end)
        return "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines[s:e], start=s))
    except (IOError, OSError):
        return ""
 
 
# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
 
if HAS_MCP:
    mcp = FastMCP(
        "repolect",
        instructions=(
            "Repolect provides semantic code intelligence via a hierarchical "
            "summary tree and a knowledge graph.\n\n"
            "**Before coding:** Use plan_change(description) to get a structured "
            "change plan (what to modify, read, and test).\n"
            "**Before creating new code:** Use find_similar(description) to find "
            "an existing implementation to use as a template.\n"
            "**Before editing:** Use get_conventions(node_id) to match local style.\n"
            "**After editing:** Use scope_test(node_ids) to find the minimal test set.\n"
            "**For renaming:** Use rename(old, new) for a coordinated rename plan.\n\n"
            "Other tools: tree_search (semantic search), get_node (360-degree symbol "
            "view with relations/callers/callees), trace_flow, impact_analysis, "
            "diff_analysis, graph_query."
        ),
    )
 
    # ── Tools ─────────────────────────────────────────────────────────────
 
    @mcp.tool(description="Search the codebase semantically using tree-based LLM reasoning")
    def tree_search(
        query: str, max_results: int = 5,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Find code relevant to a natural-language query.
 
        Returns an LLM-synthesised answer with (file:line) citations.
        Requires a running LLM provider (Ollama or openai-compatible).
        """
        repo_root, root, provider, graph_db, embedder = _load_context(repo, branch)
        if not provider:
            return (
                "No LLM provider available. Start Ollama or configure "
                "~/.repolect/config.yaml with an openai-compatible endpoint."
            )
        from .search import TreeSearcher, Explainer
 
        searcher = TreeSearcher(root, repo_root, provider, graph_db=graph_db, embedder=embedder)
        results = searcher.search(query, max_results=max_results)
        if not results:
            return "No relevant code found."
        explainer = Explainer(provider)
        answer = explainer.explain(query, results)
        refs = "\n".join(
            f"  - {r.node.title} ({r.node.node_id}, {r.node.path}:{r.node.line_start})"
            for r in results[:5]
        )
        return (
            f"{answer}\n\n"
            f"Referenced nodes:\n{refs}\n\n"
            "Next: Use get_node(node_id) for full source. "
            "Use trace_flow() for execution paths."
        )
 
    @mcp.tool(description="Get full details of a code node: source, summary, relations, callers, callees")
    def get_node(
        node_id: str,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Retrieve a node's metadata, summary, source code, callers, and callees
        in a single call — a 360-degree view of a symbol.
 
        Node IDs look like '0001', '0001.002', '0001.002.003'.
        """
        repo_root, root, _, graph_db, _ = _load_context(repo, branch)
        node_map = root.get_node_map()
        if node_id not in node_map:
            return f"Node '{node_id}' not found."
        node = node_map[node_id]
 
        _DEP_KINDS = {"IMPORTS", "CALLS", "EXTENDS", "IMPLEMENTS"}
        _STRUCT_KINDS = {"CONTAINS", "DEFINES"}
 
        span = node.line_end - node.line_start
        if node.path and node.kind == "file" and span > 60:
            source = _read_source(Path(repo_root) / node.path, node.line_start, node.line_start + 50)
            child_sigs = "\n".join(
                f"  - {c.title} ({c.kind}, line {c.line_start})"
                for c in node.children[:25]
            )
            remaining = span - 50
            source += f"\n... ({remaining} more lines)\n\nChildren:\n{child_sigs}"
        elif node.path and span <= 1 and node.kind in ("function", "method", "class"):
            parent = _find_parent(root, node_id)
            if parent and parent.path:
                ctx_start = max(node.line_start - 3, parent.line_start)
                ctx_end = min(node.line_start + 25, parent.line_end)
                source = _read_source(Path(repo_root) / parent.path, ctx_start, ctx_end)
                source += f"\n(Note: line range {node.line_start}-{node.line_end} seems narrow; showing surrounding context)"
            else:
                source = _read_source(Path(repo_root) / node.path, max(1, node.line_start - 3), node.line_start + 25)
                source += f"\n(Note: showing extended context around line {node.line_start})"
        elif node.path:
            source = _read_source(Path(repo_root) / node.path, node.line_start, node.line_end)
        else:
            source = ""
 
        deps: list[str] = []
        structure: list[str] = []
        callers: list[str] = []
 
        for r in node.relations:
            if r.target_id in node_map:
                entry = f"{r.kind} -> {node_map[r.target_id].title} ({r.target_id})"
                if r.kind in _DEP_KINDS:
                    deps.append(entry)
                elif r.kind in _STRUCT_KINDS:
                    structure.append(entry)
                else:
                    deps.append(entry)
 
        used_by = [
            n for n in node_map.values()
            if any(rel.target_id == node_id for rel in n.relations)
        ][:8]
        for n in used_by:
            rel_kinds = [rel.kind for rel in n.relations if rel.target_id == node_id]
            kind_str = f" [{','.join(set(rel_kinds))}]" if rel_kinds else ""
            callers.append(f"{n.title} ({n.node_id}){kind_str}")
 
        if graph_db:
            try:
                out_neighbors = graph_db.get_neighbors(node_id, direction="out")
                seen_out = {r.target_id for r in node.relations}
                for nb in out_neighbors[:8]:
                    nid = nb.get("node_id", "")
                    if nid and nid not in seen_out and nid in node_map:
                        rel_type = nb.get("_rel_type", "RELATED")
                        entry = f"{rel_type} -> {node_map[nid].title} ({nid})"
                        if rel_type in _DEP_KINDS:
                            deps.append(entry)
                        elif rel_type in _STRUCT_KINDS:
                            structure.append(entry)
                        else:
                            deps.append(entry)
            except Exception:
                pass
            try:
                in_neighbors = graph_db.get_neighbors(node_id, direction="in")
                seen_in = {n.node_id for n in used_by}
                for nb in in_neighbors[:8]:
                    nid = nb.get("node_id", "")
                    if nid and nid not in seen_in and nid in node_map:
                        rel_type = nb.get("_rel_type", "")
                        kind_str = f" [{rel_type}]" if rel_type else ""
                        callers.append(f"{node_map[nid].title} ({nid}){kind_str}")
            except Exception:
                pass
 
        parts = [
            f"**{node.title}** ({node.kind})",
            f"Path: {node.path}  Lines: {node.line_start}-{node.line_end}",
            f"Language: {node.language}  ID: {node.node_id}",
            f"Summary: {node.summary}",
        ]
        if node.signature:
            parts.append(f"Signature: {node.signature}")
        if source.strip():
            parts.append(f"\n```{node.language}\n{source}\n```")
        elif node.children and node.kind in ("repo", "module"):
            child_list = "\n".join(
                f"  - **{c.title}** ({c.kind}, {c.node_id})"
                for c in node.children[:20]
            )
            parts.append(f"\n**Children** ({len(node.children)}):\n{child_list}")
        if deps:
            parts.append(f"\n**Dependencies** ({len(deps)}):")
            for d in deps[:12]:
                parts.append(f"  - {d}")
        if structure:
            parts.append(f"\n**Structure** ({len(structure)}):")
            for s in structure[:12]:
                parts.append(f"  - {s}")
        if callers:
            parts.append(f"\n**Called by** ({len(callers)}):")
            for c in callers[:10]:
                parts.append(f"  - {c}")
 
        return "\n".join(parts)
 
    @mcp.tool(description="Explain why a file or function exists in the codebase context")
    def explain_node(
        node_id: str,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Walk the tree upward and graph inward to explain a node's role.
 
        Requires a running LLM provider.
        """
        if not node_id or not node_id.strip():
            return "Error: node_id is required."
        repo_root, root, provider, graph_db, _ = _load_context(repo, branch)
        if not provider:
            return "No LLM provider available."
        from .search import TreeSearcher
 
        searcher = TreeSearcher(root, repo_root, provider, graph_db=graph_db)
        node_map = root.get_node_map()
        node = node_map.get(node_id)
        if not node:
            return f"Node '{node_id}' not found."
        explanation = searcher.explain_node(node_id)
        hints = ["Next: Use tree_search() to find related components."]
        if node:
            callers = [
                n for n in node_map.values()
                if any(r.target_id == node_id for r in n.relations)
            ][:3]
            if callers:
                caller_str = ", ".join(f"{c.title} ({c.node_id})" for c in callers)
                hints.append(f"Callers: {caller_str}")
            callees = [
                node_map[r.target_id]
                for r in node.relations
                if r.kind == "CALLS" and r.target_id in node_map
            ][:3]
            if callees:
                callee_str = ", ".join(f"{c.title} ({c.node_id})" for c in callees)
                hints.append(f"Calls: {callee_str}")
        return f"{explanation}\n\n" + "\n".join(hints)
 
    @mcp.tool(description="Trace execution flow from an entry point following CALLS edges")
    def trace_flow(
        entry_point: str, max_depth: int = 5,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Follow the CALLS chain from a starting function.
 
        Pass a node_id (e.g. '0002.013.002') or a natural-language query
        (e.g. 'main', 'handle_request').  If the resolved node is a file,
        traces from each function defined in it.  Requires a running LLM
        provider for natural-language resolution.
        """
        repo_root, root, provider, graph_db, _ = _load_context(repo, branch)
        if not provider:
            return "No LLM provider available."
        from .search import TreeSearcher
 
        searcher = TreeSearcher(root, repo_root, provider, graph_db=graph_db)
        result = searcher.trace_flow(entry_point, max_depth=max_depth)
        return (
            f"{result}\n\n"
            "Next: Use get_node() on any step for source code details."
        )
 
    @mcp.tool(description="Run a Cypher query against the code knowledge graph")
    def graph_query(
        cypher: str,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Execute a Cypher query and return tabular results.
 
        Schema:
          Node label: CodeNode
          Node properties: node_id, title, kind, file_path, summary, line_start, line_end
          Edge types: CALLS, IMPORTS, CONTAINS, EXTENDS, IMPLEMENTS
 
        Examples:
          MATCH (n:CodeNode) WHERE n.kind = 'function' RETURN n.title, n.file_path LIMIT 10
          MATCH (a)-[:CALLS]->(b) RETURN a.title, b.title LIMIT 20
          MATCH (n)-[r]->(m) RETURN type(r), count(r)
        """
        _, _, _, graph_db, _ = _load_context(repo, branch)
        if not graph_db:
            return "No graph database available. Re-run 'repolect analyze' to build the graph."
        if not cypher.strip():
            return "No Cypher query provided."
        import re as _re_guard
        _MUTATING = _re_guard.compile(
            r"\b(DELETE|DETACH\s+DELETE|CREATE|SET|REMOVE|MERGE|DROP)\b", _re_guard.IGNORECASE,
        )
        if _MUTATING.search(cypher):
            return "Refused: graph_query is read-only. Mutating operations (DELETE, CREATE, SET, REMOVE, MERGE, DROP) are not allowed."
        try:
            results = graph_db.cypher(cypher)
            if not results:
                return "(no results)"
            import re as _re
            headers: list[str] = []
            ret_match = _re.search(r"(?i)\bRETURN\b(.+?)(?:\bORDER\b|\bLIMIT\b|\bSKIP\b|$)", cypher)
            if ret_match:
                raw_cols = ret_match.group(1).strip()
                cols = [c.strip().rstrip(",") for c in raw_cols.split(",")]
                for col in cols:
                    alias_match = _re.search(r"(?i)\bAS\s+(\w+)", col)
                    if alias_match:
                        headers.append(alias_match.group(1))
                    else:
                        headers.append(col.split("(")[0].strip() if "(" in col else col)
            lines = []
            if results and headers and len(headers) == len(results[0]):
                sep = " | "
                lines.append(sep.join(str(h) for h in headers))
                lines.append(sep.join("---" for _ in headers))
            for row in results[:50]:
                lines.append(" | ".join(str(cell) for cell in row))
            if len(results) > 50:
                lines.append(f"... ({len(results) - 50} more rows)")
            lines.append(f"\n({len(results)} row(s) returned)")
            return "\n".join(lines)
        except Exception as e:
            return f"Query error: {e}"
 
    @mcp.tool(description="Show blast radius: what depends on a given node")
    def impact_analysis(
        node_id: str, max_hops: int = 3,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Find all nodes that depend on the target (reverse CALLS/IMPORTS traversal).
 
        Returns dependents grouped by hop distance with dependency chains
        showing *why* each node is affected.
        """
        _, root, _, graph_db, _ = _load_context(repo, branch)
        if not graph_db:
            return "No graph database available."
        node_map = root.get_node_map()
        if node_id not in node_map:
            return f"Node '{node_id}' not found."
        target = node_map[node_id]
        deps = graph_db.get_reverse_dependencies(
            node_id, max_hops=max_hops, rel_types=["CALLS", "IMPORTS"],
        )
        mem_callers = [
            n for n in node_map.values()
            if any(r.target_id == node_id and r.kind in ("CALLS", "IMPORTS") for r in n.relations)
        ]
        if not deps and not mem_callers:
            return f"No dependents found for '{target.title}'."
        if not deps or (len(deps) < 5 and len(mem_callers) > len(deps)):
            graph_ids = {nid for nid, _ in deps}
            prod_callers = [n for n in mem_callers if not _is_test_file(n.path)]
            test_callers = [n for n in mem_callers if _is_test_file(n.path)]
            for n in prod_callers:
                if n.node_id not in graph_ids and n.node_id != node_id:
                    deps.append((n.node_id, 1))
                    graph_ids.add(n.node_id)
            for n in test_callers[:10]:
                if n.node_id not in graph_ids and n.node_id != node_id:
                    deps.append((n.node_id, 1))
                    graph_ids.add(n.node_id)
            deps.sort(key=lambda x: x[1])
        if not deps:
            return f"No dependents found for '{target.title}'."
 
        by_hop: dict[int, list[tuple[str, int]]] = {}
        for nid, hop in deps:
            by_hop.setdefault(hop, []).append((nid, hop))
 
        lines = [f"Impact analysis for **{target.title}** ({target.path}):\n"]
        for hop_dist in sorted(by_hop):
            lines.append(f"**Hop {hop_dist}** ({len(by_hop[hop_dist])} node(s)):")
            for nid, _ in by_hop[hop_dist]:
                n = node_map.get(nid)
                if not n:
                    lines.append(f"  - {nid}")
                    continue
                chain = ""
                if hop_dist > 1:
                    try:
                        path = graph_db.find_shortest_path(node_id, nid)
                        if path and len(path) > 2:
                            mid = [node_map[p].title for p in path[1:-1] if p in node_map]
                            chain = f"  via {' -> '.join(mid)}"
                    except Exception:
                        pass
                label = ""
                if _is_test_file(n.path):
                    label = " [test]"
                elif n.path and Path(n.path).name in ("cli.py", "__main__.py", "main.py"):
                    label = " [entrypoint]"
 
                detail = ""
                if n.kind == "file" and n.children:
                    child_callers = [
                        c for c in n.children
                        if c.kind in ("function", "method", "class")
                        and any(r.target_id == node_id or r.kind == "CALLS" for r in c.relations)
                    ][:3]
                    if child_callers:
                        detail = f"  (specifically: {', '.join(c.title for c in child_callers)})"
 
                lines.append(f"  - {n.title} ({n.kind}, {n.path}){chain}{label}{detail}")
            lines.append("")
 
        lines.append(f"{len(deps)} node(s) affected within {max_hops} hops.")
        return "\n".join(lines)
 
    @mcp.tool(description="Show which functions/classes changed since a git ref and their blast radius")
    def diff_analysis(
        ref: str = "HEAD~1", with_impact: bool = True,
        committed_only: bool = False,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Map git changes to affected code symbols and optionally show impact.
 
        Args:
            ref: Git ref to diff against (default: HEAD~1).
            with_impact: If True, also show downstream dependents.
            committed_only: If True, only show committed changes (ref..HEAD).
                If False (default), includes uncommitted working-tree changes.
        """
        repo_root, root, _, graph_db, _ = _load_context(repo, branch)
        from .git_utils import get_changed_line_ranges, is_git_repo, get_current_commit
        from .storage import load_meta
        from .tree_builder import map_changes_to_nodes
 
        if not is_git_repo(repo_root):
            return "This repository does not have git — diff analysis requires git."
 
        changed_ranges = get_changed_line_ranges(
            repo_root, ref=ref, committed_only=committed_only,
        )
        if not changed_ranges:
            return f"No changes detected since {ref}."
 
        preamble: list[str] = []
        meta = load_meta(repo_root, branch=branch)
        if meta and meta.git_commit:
            current_commit = get_current_commit(repo_root)
            if current_commit != meta.git_commit and current_commit != "no-git":
                preamble.append(
                    f"**Warning:** Index was built at commit {meta.git_commit[:8]}, "
                    f"current HEAD is {current_commit[:8]}. Run `repolect sync` "
                    "for accurate symbol mapping.\n"
                )
 
        affected = map_changes_to_nodes(changed_ranges, root)
        node_map = root.get_node_map()
        lines = list(preamble)
        lines.append(f"Changes since {ref}:")
        lines.append(
            f"{len(changed_ranges)} file(s) modified, "
            f"{len(affected)} symbol(s) affected:\n"
        )
 
        if affected:
            for node in affected:
                lines.append(f"  - {node.title} ({node.kind}, {node.path}:{node.line_start})")
        else:
            lines.append("No function/class-level symbols overlapped changed lines.")
            lines.append("Changed files:")
            for fpath, ranges in sorted(changed_ranges.items()):
                file_node = None
                for n in node_map.values():
                    if n.kind == "file" and n.path and n.path.replace("\\", "/") == fpath.replace("\\", "/"):
                        file_node = n
                        break
                range_str = ", ".join(f"L{s}-{e}" for s, e in ranges[:5])
                if len(ranges) > 5:
                    range_str += f", +{len(ranges) - 5} more"
                nid_str = f" ({file_node.node_id})" if file_node else ""
                lines.append(f"  - {fpath}{nid_str} [{range_str}]")
 
        if with_impact and affected and graph_db:
            lines.append(f"\nBlast radius:")
            all_impacted: set[str] = set()
            affected_ids = {node.node_id for node in affected}
            for node in affected:
                deps = graph_db.get_reverse_dependencies(
                    node.node_id, max_hops=3, rel_types=["CALLS", "IMPORTS"],
                )
                for nid, hop in deps:
                    if nid not in all_impacted and nid not in affected_ids:
                        all_impacted.add(nid)
                        n = node_map.get(nid)
                        if n:
                            label = " [test]" if _is_test_file(n.path) else ""
                            lines.append(f"    [hop {hop}] {n.title} ({n.kind}, {n.path}){label}")
            if not all_impacted:
                intra_callers: list[str] = []
                for node in affected:
                    callers = [
                        n for n in node_map.values()
                        if any(r.target_id == node.node_id and r.kind == "CALLS" for r in n.relations)
                        and n.node_id not in affected_ids
                    ]
                    for c in callers[:5]:
                        if c.node_id not in affected_ids:
                            label = " [test]" if _is_test_file(c.path) else ""
                            intra_callers.append(
                                f"    {c.title} ({c.kind}, {c.path}:{c.line_start}) calls {node.title}{label}"
                            )
                if intra_callers:
                    lines.append("  (in-memory callers — graph had no cross-file hits):")
                    lines.extend(intra_callers[:10])
                else:
                    lines.append("  No downstream dependents found.")
 
        return "\n".join(lines)
 
    @mcp.tool(description="List all repositories indexed by Repolect")
    def list_repos() -> str:
        """Show name, file count, node count, and path for every indexed repo."""
        from .storage import list_repos as _list_repos
 
        repos = _list_repos()
        if not repos:
            return "No repositories indexed yet."
        lines = [f"{'Name':<25} {'Files':>6} {'Nodes':>7}  Path"]
        for r in repos:
            lines.append(
                f"{r.get('repo_name', '?'):<25} {r.get('file_count', '?'):>6} "
                f"{r.get('node_count', '?'):>7}  {r.get('repo_path', '')}"
            )
        lines.append(
            "\nNext: Use repo_summary() to get an overview of any listed repo."
        )
        return "\n".join(lines)
 
    @mcp.tool(description="Get the top-level overview of the current codebase")
    def repo_summary(
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Return the repo-level summary, stats, and top-level modules."""
        _, root, _, _, _ = _load_context(repo, branch)
        node_map = root.get_node_map()
        file_count = sum(1 for n in node_map.values() if n.kind == "file")
        fn_count = sum(1 for n in node_map.values() if n.kind in ("function", "method"))
        class_count = sum(1 for n in node_map.values() if n.kind == "class")
 
        lang_counts: dict[str, int] = {}
        for n in node_map.values():
            if n.kind == "file" and n.language:
                lang_counts[n.language] = lang_counts.get(n.language, 0) + 1
        lang_str = ", ".join(
            f"{lang}: {cnt}" for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1])[:5]
        )
 
        def _trunc_sentence(text: str, limit: int = 150) -> str:
            if len(text) <= limit:
                return text
            dot = text.rfind(".", 0, limit)
            return text[:dot + 1] if dot > 20 else text[:limit] + "..."
 
        modules = "\n".join(
            f"  - **{c.title}** ({c.node_id}): {_trunc_sentence(c.summary)}"
            for c in root.children[:15]
        )
        stats = f"Stats: {file_count} files, {class_count} classes, {fn_count} functions"
        if lang_str:
            stats += f"\nLanguages: {lang_str}"
        return (
            f"**{root.title}**\n\n"
            f"{root.summary}\n\n"
            f"{stats}\n\n"
            f"Top-level modules:\n{modules}\n\n"
            "Next: Use tree_search(query) to find specific code."
        )
 
    # ── Agent-Assist Tools ────────────────────────────────────────────────
 
    def _is_test_file(path: str) -> bool:
        """Heuristic: does this path look like a test file?"""
        if not path:
            return False
        parts = Path(path).parts
        name = Path(path).name
        in_test_dir = any(p in ("tests", "test", "__tests__", "spec") for p in parts)
        test_name = (
            name.startswith("test_") or name.endswith("_test.py")
            or name.startswith("test.") or name.endswith(".test.ts")
            or name.endswith(".test.js") or name.endswith(".spec.ts")
            or name.endswith(".spec.js")
        )
        return in_test_dir or test_name
 
    def _find_parent(root, target_id: str):
        """Walk the tree to find the parent CodeNode of target_id."""
        for node in root.flat_iter():
            for child in node.children:
                if child.node_id == target_id:
                    return node
        return None
 
    def _find_test_file_for(node_path: str, node_map: dict, repo_root: str | None = None) -> str | None:
        """Heuristic: find a test file that corresponds to a source file.
 
        Checks the index first, then falls back to filesystem if repo_root is
        provided (catches test files created after the last index build).
        """
        if not node_path:
            return None
        p = Path(node_path)
        stem = p.stem
        candidates = [
            f"tests/test_{stem}.py",
            f"test/test_{stem}.py",
            f"tests/{stem}_test.py",
            f"{p.parent}/test_{stem}.py",
            f"{p.parent}/{stem}_test.py",
        ]
        for n in node_map.values():
            if n.path and any(n.path.endswith(c) or n.path == c for c in candidates):
                return n.path
        for n in node_map.values():
            if n.path and _is_test_file(n.path) and stem in Path(n.path).stem:
                return n.path
        if repo_root:
            for cand in candidates:
                cand_path = Path(repo_root) / cand
                if cand_path.exists():
                    return cand
        return None
 

    @mcp.tool(description="Find which tests to run after modifying specific code nodes")
    def scope_test(
        node_ids: str,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Given node IDs of modified symbols, find the minimal set of tests
        to run, ranked by confidence tier.
     
        Args:
            node_ids: Comma-separated node IDs or a JSON array
                (e.g. '0002.006,0002.010' or '["0002.006","0002.010"]').
     
        Returns three tiers:
            - MUST RUN: tests directly importing/calling modified code (hop 1)
            - SHOULD RUN: tests transitively dependent or heuristic match (hop 2)
            - CONSIDER: tests at the edge of the blast radius (hop 3)
        """
        repo_root, root, _, graph_db, _ = _load_context(repo, branch)
        if not graph_db:
            return "No graph database available. Re-run 'repolect analyze' to build the graph."
        node_map = root.get_node_map()
     
        ids: list[str] = []
        raw = node_ids.strip()
        if raw.startswith("["):
            try:
                ids = [str(x).strip() for x in json.loads(raw) if x]
            except (json.JSONDecodeError, ValueError):
                pass
        if not ids:
            ids = [nid.strip() for nid in raw.split(",") if nid.strip()]
        if not ids:
            return "No node IDs provided. Pass comma-separated node IDs or a JSON array."
     
        missing = [nid for nid in ids if nid not in node_map]
        if missing:
            return f"Node(s) not found: {', '.join(missing)}"
     
        tiers: dict[str, int] = {}
        for nid in ids:
            deps = graph_db.get_reverse_dependencies(
                nid, max_hops=3, rel_types=["CALLS", "IMPORTS"],
            )
            for dep_id, hop in deps:
                n = node_map.get(dep_id)
                if n and _is_test_file(n.path):
                    if dep_id not in tiers or hop < tiers[dep_id]:
                        tiers[dep_id] = hop
     
        for nid in ids:
            n = node_map.get(nid)
            if n:
                tp = _find_test_file_for(n.path, node_map, repo_root=repo_root)
                if tp:
                    matched = False
                    for tn in node_map.values():
                        if tn.path == tp and tn.kind == "file":
                            if tn.node_id not in tiers:
                                tiers[tn.node_id] = 2
                            matched = True
                            break
                    if not matched:
                        tiers[f"_fs:{tp}"] = 2
     
        if not tiers:
            return "No tests found in the blast radius of the modified nodes."
     
        tier_labels = {1: "MUST RUN", 2: "SHOULD RUN", 3: "CONSIDER"}
        lines = [f"Test scope for {len(ids)} modified node(s):\n"]
        for tier_hop in (1, 2, 3):
            indexed_tests = sorted(
                [(nid, node_map[nid]) for nid, h in tiers.items() if h == tier_hop and nid in node_map],
                key=lambda x: x[1].path,
            )
            fs_tests = sorted(
                [(nid, nid[4:]) for nid, h in tiers.items() if h == tier_hop and nid.startswith("_fs:")],
                key=lambda x: x[1],
            )
            all_tests = indexed_tests + [(nid, None) for nid, _ in fs_tests]
            if all_tests:
                lines.append(f"**{tier_labels[tier_hop]}** ({len(all_tests)} test(s)):")
                for nid, n in indexed_tests:
                    if n.kind == "file":
                        lines.append(f"  - {n.path}")
                    else:
                        lines.append(f"  - {n.path} ({n.title})")
                for nid, fpath in fs_tests:
                    lines.append(f"  - {fpath} (not yet indexed)")
                lines.append("")
     
        total = len(tiers)
        lines.append(f"Total: {total} test(s) across {len(ids)} modified node(s).")
        return "\n".join(lines)
     
    @mcp.tool(description="Find an existing implementation similar to what you want to create")
    def find_similar(
        description: str,
        kind: str = "any",
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Search for existing code that does something similar to what you plan
        to build. Returns the closest match with full source, relations, associated
        test file, and an LLM-generated guide on which parts are template vs
        domain-specific.
     
        Args:
            description: What you want to create (e.g. "a REST endpoint for deleting users").
            kind: Optional filter: "endpoint", "test", "handler", "class", "function", or "any".
        """
        if not description or not description.strip():
            return "Error: description is required. Describe what you want to find."
        repo_root, root, provider, graph_db, embedder = _load_context(repo, branch)
        if not provider:
            return "No LLM provider available."
        from .search import TreeSearcher
     
        kind_hint = f" (looking for a {kind})" if kind != "any" else ""
        query = f"existing implementation of: {description}{kind_hint}"
        searcher = TreeSearcher(root, repo_root, provider, graph_db=graph_db, embedder=embedder)
        results = searcher.search(query, max_results=8)
        if not results:
            return "No similar implementations found."
     
        kind_mismatch = False
        if kind != "any":
            _kind_map = {
                "endpoint": ("function", "method"),
                "handler": ("function", "method"),
                "test": ("function",),
                "class": ("class",),
                "function": ("function", "method"),
                "method": ("method",),
                "file": ("file",),
            }
            allowed = _kind_map.get(kind, (kind,))
            filtered = [r for r in results if r.node.kind in allowed]
            if filtered:
                results = filtered
            else:
                kind_mismatch = True
     
        seen_paths: set[str] = set()
        deduped: list = []
        for r in results:
            key = (r.node.title, r.node.path)
            if key not in seen_paths:
                seen_paths.add(key)
                deduped.append(r)
        results = deduped[:3]
     
        node_map = root.get_node_map()
        best = results[0]
        node = best.node
        source = best.source_snippet
        source_lines = source.splitlines()
        if len(source_lines) <= 3 and node.path:
            extended = _read_source(
                Path(repo_root) / node.path,
                max(1, node.line_start - 2),
                node.line_start + 40,
            )
            if extended:
                source = extended
                source_lines = source.splitlines()
        if len(source_lines) > 80:
            source = "\n".join(source_lines[:80]) + f"\n... ({len(source_lines) - 80} more lines)"
     
        relations_lines = []
        for r in node.relations[:8]:
            target_title = node_map[r.target_id].title if r.target_id in node_map else r.target_id
            relations_lines.append(f"  - {r.kind} -> {target_title}")
        if graph_db:
            try:
                neighbors = graph_db.get_neighbors(node.node_id, direction="both")
                seen = {r.target_id for r in node.relations}
                for nb in neighbors[:8]:
                    nid = nb.get("node_id", "")
                    if nid and nid not in seen and nid in node_map:
                        rel_type = nb.get("_rel_type", "RELATED")
                        relations_lines.append(f"  - {rel_type} -> {node_map[nid].title}")
            except Exception:
                pass
     
        test_file = _find_test_file_for(node.path, node_map, repo_root=repo_root)
     
        template_guide = provider.complete(
            f"""You are analyzing code to help someone create something new.
     
    They want to create: "{description}"
    The closest existing implementation is `{node.title}` ({node.kind}) at {node.path}.
     
    Summary: {node.summary}
     
    Source (may be truncated):
    {source[:3000]}
     
    In 3-5 bullet points, explain:
    1. Which parts of this code are structural/boilerplate that should be copied as-is
    2. Which parts are domain-specific and need to be replaced
    3. Any patterns or conventions this code follows that the new code should match
     
    Reference function/class/variable names from the source, NOT line numbers (the source shown may be truncated).
    Be concise and specific.""",
            max_tokens=400,
        )
     
        lines = [
            f"**Closest match: {node.title}** ({node.kind})",
            f"Path: {node.path}  Lines: {node.line_start}-{node.line_end}",
            f"Similarity: {best.relevance_score:.1f}/10",
        ]
        if kind_mismatch:
            lines.append(f"Note: No {kind}-level matches found; showing best overall match.")
        lines.extend([
            f"Summary: {node.summary}\n",
            f"```{node.language}",
            source,
            "```\n",
        ])
        if relations_lines:
            lines.append("**Relations:**")
            lines.extend(relations_lines)
            lines.append("")
        if test_file:
            lines.append(f"**Test file:** {test_file}")
        else:
            lines.append("**Test file:** (none found)")
        lines.append(f"\n**How to adapt this:**\n{template_guide}")
     
        if len(results) > 1:
            lines.append("\n**Other candidates:**")
            for r in results[1:]:
                lines.append(
                    f"  - {r.node.title} ({r.node.path}) — score {r.relevance_score:.1f}"
                )
     
        return "\n".join(lines)
     
    @mcp.tool(description="Extract coding conventions and patterns from a module or file's neighborhood")
    def get_conventions(
        node_id: str,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Before modifying or creating code near a given node, call this tool to
        understand the local conventions: error handling, naming, architecture
        patterns, import style, async vs sync, etc.
     
        Returns a concise "when-in-Rome" guide for that area of the codebase.
        """
        repo_root, root, provider, graph_db, _ = _load_context(repo, branch)
        if not provider:
            return "No LLM provider available."
        node_map = root.get_node_map()
        if node_id not in node_map:
            return f"Node '{node_id}' not found."
        target = node_map[node_id]
     
        parent = _find_parent(root, node_id)
        if not parent:
            parent = root
     
        siblings = [
            c for c in parent.children
            if c.node_id != node_id and c.kind in ("file", "class", "function", "method")
        ][:12]
     
        community_peers: list = []
        if graph_db:
            try:
                neighbors = graph_db.get_neighbors(node_id, direction="both")
                for nb in neighbors[:12]:
                    nid = nb.get("node_id", "")
                    if nid and nid != node_id and nid in node_map:
                        n = node_map[nid]
                        if n.kind in ("file", "class", "function", "method"):
                            community_peers.append(n)
            except Exception:
                pass
     
        all_neighbors = {s.node_id: s for s in siblings}
        for peer in community_peers:
            if peer.node_id not in all_neighbors:
                all_neighbors[peer.node_id] = peer
        neighborhood = list(all_neighbors.values())[:12]
     
        if not neighborhood:
            return f"No neighboring code found around '{target.title}' to extract conventions from."
     
        summaries = []
        for n in neighborhood:
            summaries.append(f"- {n.title} ({n.kind}, {n.path}): {n.summary[:200]}")
        summaries_text = "\n".join(summaries)
     
        source_samples = []
        sample_nodes = sorted(neighborhood, key=lambda n: n.path or "")[:4]
        for n in sample_nodes:
            if n.path and n.line_start:
                src = _read_source(Path(repo_root) / n.path, n.line_start, min(n.line_end, n.line_start + 30))
                if src:
                    source_samples.append(f"--- {n.title} ({n.path}) ---\n{src}")
        samples_text = "\n\n".join(source_samples) if source_samples else "(no source samples available)"
     
        conventions = provider.complete(
            f"""You are analyzing a code neighborhood to extract conventions and patterns.
     
    Target: `{target.title}` ({target.kind}) at {target.path}
    Module: {parent.title} — {parent.summary[:200]}
     
    Neighboring code in this area:
    {summaries_text}
     
    Source samples from nearby code:
    {samples_text}
     
    Extract the coding conventions for this area. Cover:
    1. **Error handling**: How do errors get raised/caught? Custom exception classes?
    2. **Naming conventions**: Function/class/variable naming style
    3. **Import style**: How are imports organized?
    4. **Architecture patterns**: Factory, repository, middleware, decorator, etc.
    5. **Async vs sync**: Which style is used?
    6. **Return patterns**: What do functions return? Dicts, dataclasses, tuples?
    7. **Logging**: How is logging done?
    8. **Any other patterns** specific to this module
     
    Be concise — 1-2 sentences per category. Skip categories that aren't visible in the samples. This guide will be used by a coding agent to write new code that matches the local style.""",
            max_tokens=500,
        )
     
        lines = [
            f"**Conventions for area around `{target.title}`**",
            f"Module: {parent.title}",
            f"Based on {len(neighborhood)} neighboring nodes\n",
            conventions,
        ]
        return "\n".join(lines)
     
    @mcp.tool(description="Plan which files to modify, read, and test for a proposed change")
    def plan_change(
        description: str,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Given a natural-language feature description, produce a structured
        change plan before writing any code.
     
        Returns:
            - ADD: files where new code should be inserted
            - MODIFY: existing code that needs changes, in suggested order
            - READ_ONLY: files needed for context but no changes required
            - TEST_AFTER: tests to run after the change
            - Estimated scope (small/medium/large)
        """
        if not description or not description.strip():
            return "Error: description is required. Provide a natural-language description of the change."
        repo_root, root, provider, graph_db, embedder = _load_context(repo, branch)
        if not provider:
            return "No LLM provider available."
        if not graph_db:
            return "No graph database available. Re-run 'repolect analyze' to build the graph."
        from .search import TreeSearcher
     
        node_map = root.get_node_map()
        searcher = TreeSearcher(root, repo_root, provider, graph_db=graph_db, embedder=embedder)
        results = searcher.search(description, max_results=5)
        if not results:
            return "Could not find relevant entry points for this change."
     
        entry_ids = [r.node.node_id for r in results]
     
        surface: dict[str, tuple[int, str]] = {}
        for eid in entry_ids:
            surface[eid] = (0, "entry point")
            for rel in ("IMPORTS", "CALLS"):
                try:
                    neighbors = graph_db.get_neighbors(eid, direction="out", rel_type=rel)
                    for nb in neighbors:
                        nid = nb.get("node_id", "")
                        if nid and nid in node_map and nid not in surface:
                            surface[nid] = (1, f"{rel} from {node_map[eid].title}")
                except Exception:
                    pass
     
        meaningful = {
            nid: info for nid, info in surface.items()
            if node_map[nid].kind in ("function", "method", "class", "file")
            and not _is_test_file(node_map[nid].path)
        }
     
        by_path: dict[str, list[str]] = {}
        for nid in meaningful:
            p = node_map[nid].path
            by_path.setdefault(p, []).append(nid)
        for path, nids in by_path.items():
            if len(nids) > 1:
                file_nids = [n for n in nids if node_map[n].kind == "file"]
                child_nids = [n for n in nids if node_map[n].kind != "file"]
                if child_nids and file_nids:
                    for fn in file_nids:
                        meaningful.pop(fn, None)
     
        _VALID_CATS = {"ADD", "MODIFY", "READ_ONLY", "UNAFFECTED"}
        batch_size = 10
        node_items = sorted(meaningful.items(), key=lambda x: x[1][0])[:30]
        classifications: dict[str, tuple[str, str]] = {}
     
        for i in range(0, len(node_items), batch_size):
            batch = node_items[i:i + batch_size]
            node_descriptions = "\n".join(
                f'{idx+1}. node_id="{nid}" title="{node_map[nid].title}" '
                f'kind={node_map[nid].kind} path={node_map[nid].path} '
                f'reason_in_surface="{info[1]}" '
                f'summary="{node_map[nid].summary[:200]}"'
                for idx, (nid, info) in enumerate(batch)
            )
            raw = provider.complete(
                f"""You are planning a code change. The proposed change is:
    "{description}"
     
    For each node below, classify it as one of:
    - ADD: new code must be added to this file (existing code stays the same)
    - MODIFY: existing code in this specific node must change
    - READ_ONLY: useful as reference/context but no changes needed
    - UNAFFECTED: not relevant to this change
     
    Use ADD when the feature requires inserting new functions/classes into an existing file.
    Use MODIFY only when a node's existing code must be edited (not just imported from).
    Files that are IMPORTED or CALLED by the change target are almost always READ_ONLY.
    Utility modules, config loaders, and infrastructure code should be READ_ONLY unless the change description explicitly requires editing them.
    When in doubt, prefer READ_ONLY over MODIFY.
     
    Nodes:
    {node_descriptions}
     
    Return ONLY a JSON array like:
    [{{"node_id": "...", "category": "ADD", "reason": "brief reason"}}]
    Return ONLY the JSON array, no other text.""",
                max_tokens=600,
            )
            try:
                arr_str = raw[raw.find("["):raw.rfind("]") + 1]
                arr = json.loads(arr_str)
                for item in arr:
                    nid = str(item.get("node_id", ""))
                    cat = str(item.get("category", "UNAFFECTED")).upper()
                    reason = str(item.get("reason", ""))
                    if nid in node_map and cat in _VALID_CATS:
                        classifications[nid] = (cat, reason)
            except (json.JSONDecodeError, ValueError):
                for nid, info in batch:
                    if info[0] == 0:
                        classifications[nid] = ("ADD", "entry point for the change")
                    else:
                        classifications[nid] = ("READ_ONLY", "dependency of modified code")
     
        for nid, info in node_items:
            if nid not in classifications:
                classifications[nid] = (
                    ("ADD", "entry point") if info[0] == 0 else ("READ_ONLY", "dependency")
                )
     
        add_ids = [nid for nid, (cat, _) in classifications.items() if cat == "ADD"]
        modify_ids = [nid for nid, (cat, _) in classifications.items() if cat == "MODIFY"]
        readonly_ids = [nid for nid, (cat, _) in classifications.items() if cat == "READ_ONLY"]
        touched_ids = add_ids + modify_ids
     
        ordered_modify = []
        remaining = set(modify_ids)
        while remaining:
            progress = False
            for nid in list(remaining):
                deps = set()
                for rel in ("IMPORTS", "CALLS"):
                    for nb in graph_db.get_neighbors(nid, direction="out", rel_type=rel):
                        dep = nb.get("node_id", "")
                        if dep in remaining:
                            deps.add(dep)
                deps.discard(nid)
                if not deps:
                    ordered_modify.append(nid)
                    remaining.discard(nid)
                    progress = True
            if not progress:
                ordered_modify.extend(sorted(remaining))
                break
     
        test_tiers: dict[str, int] = {}
        for nid in touched_ids:
            deps = graph_db.get_reverse_dependencies(
                nid, max_hops=3, rel_types=["CALLS", "IMPORTS"],
            )
            for dep_id, hop in deps:
                n = node_map.get(dep_id)
                if n and _is_test_file(n.path):
                    if dep_id not in test_tiers or hop < test_tiers[dep_id]:
                        test_tiers[dep_id] = hop
     
        for nid in touched_ids:
            n = node_map.get(nid)
            if n:
                tp = _find_test_file_for(n.path, node_map, repo_root=repo_root)
                if tp:
                    matched = False
                    for tn in node_map.values():
                        if tn.path == tp and tn.kind == "file":
                            if tn.node_id not in test_tiers:
                                test_tiers[tn.node_id] = 2
                            matched = True
                            break
                    if not matched:
                        test_tiers[f"_fs:{tp}"] = 2
     
        touched_files = {node_map[nid].path for nid in touched_ids if nid in node_map}
        total_files = len(touched_files)
        scope = "small" if total_files <= 3 else ("medium" if total_files <= 8 else "large")
     
        lines = [
            f"**Change Plan: {description}**",
            f"Scope: **{scope}** ({total_files} file(s) to touch)\n",
        ]
     
        if add_ids:
            lines.append(f"**ADD** ({len(add_ids)} file(s) where new code goes):")
            for idx, nid in enumerate(add_ids, 1):
                n = node_map[nid]
                _, reason = classifications.get(nid, ("ADD", ""))
                lines.append(f"  {idx}. [{nid}] {n.title} ({n.path}:{n.line_start}) — {reason}")
            lines.append("")
     
        if ordered_modify:
            lines.append(f"**MODIFY** ({len(ordered_modify)} node(s), in suggested order):")
            for idx, nid in enumerate(ordered_modify, 1):
                n = node_map[nid]
                _, reason = classifications.get(nid, ("MODIFY", ""))
                lines.append(f"  {idx}. [{nid}] {n.title} ({n.path}:{n.line_start}) — {reason}")
            lines.append("")
     
        if readonly_ids:
            lines.append(f"**READ_ONLY context** ({len(readonly_ids)} node(s)):")
            for nid in readonly_ids[:10]:
                n = node_map[nid]
                _, reason = classifications.get(nid, ("READ_ONLY", ""))
                lines.append(f"  - [{nid}] {n.title} ({n.path}) — {reason}")
            if len(readonly_ids) > 10:
                lines.append(f"  ... and {len(readonly_ids) - 10} more")
            lines.append("")
     
        if test_tiers:
            tier_labels = {1: "MUST RUN", 2: "SHOULD RUN", 3: "CONSIDER"}
            lines.append(f"**TEST_AFTER** ({len(test_tiers)} test(s)):")
            for tier_hop in (1, 2, 3):
                indexed = [
                    (nid, node_map[nid]) for nid, h in test_tiers.items()
                    if h == tier_hop and nid in node_map
                ]
                for nid, n in sorted(indexed, key=lambda x: x[1].path):
                    lines.append(f"  [{tier_labels[tier_hop]}] {n.path}")
                fs_only = [
                    (nid, nid[4:]) for nid, h in test_tiers.items()
                    if h == tier_hop and nid.startswith("_fs:")
                ]
                for _, fpath in sorted(fs_only, key=lambda x: x[1]):
                    lines.append(f"  [{tier_labels[tier_hop]}] {fpath} (not yet indexed)")
            lines.append("")
        else:
            lines.append("**TEST_AFTER:** No tests found in blast radius.\n")
     
        return "\n".join(lines)
     
    @mcp.tool(description="Multi-file coordinated rename with graph + text search")
    def rename(
        old_name: str,
        new_name: str,
        repo: str | None = None, branch: str | None = None,
    ) -> str:
        """Find all references to a symbol name (structural via graph + textual
        via file search) and return a coordinated rename plan with confidence
        tags (graph-confirmed vs text-search-only).
     
        Does NOT modify files — returns the plan for the agent to execute.
     
        Args:
            old_name: Current symbol name (e.g. "get_user", "UserService").
            new_name: Desired new name.
        """
        if not old_name or not old_name.strip():
            return "Error: old_name cannot be empty."
        if not new_name or not new_name.strip():
            return "Error: new_name cannot be empty."
        if old_name.strip() == new_name.strip():
            return "Error: old_name and new_name are identical."
        import re as _re
        repo_root, root, _, graph_db, _ = _load_context(repo, branch)
        node_map = root.get_node_map()
     
        graph_refs: dict[str, set[str]] = {}  # file_path -> set of node titles
     
        matching_nodes = [
            n for n in node_map.values()
            if n.title == old_name or n.title.endswith(f".{old_name}")
            or n.title.endswith(f":{old_name}") or old_name in n.title
        ]
     
        for node in matching_nodes:
            if node.path:
                graph_refs.setdefault(node.path, set()).add(node.title)
     
        if graph_db:
            for node in matching_nodes:
                try:
                    rev_deps = graph_db.get_reverse_dependencies(
                        node.node_id, max_hops=2, rel_types=["CALLS", "IMPORTS"],
                    )
                    for dep_id, _ in rev_deps:
                        dep = node_map.get(dep_id)
                        if dep and dep.path:
                            graph_refs.setdefault(dep.path, set()).add(dep.title)
                except Exception:
                    pass
     
                try:
                    neighbors = graph_db.get_neighbors(node.node_id, direction="in")
                    for nb in neighbors:
                        nid = nb.get("node_id", "")
                        if nid and nid in node_map:
                            nb_node = node_map[nid]
                            if nb_node.path:
                                graph_refs.setdefault(nb_node.path, set()).add(nb_node.title)
                except Exception:
                    pass
     
        text_refs: dict[str, list[int]] = {}  # file_path -> list of line numbers
        repo_path = Path(repo_root)
        pattern = _re.compile(_re.escape(old_name))
        scanned_paths: set[str] = set()
        for node in node_map.values():
            if not node.path or node.path in scanned_paths:
                continue
            scanned_paths.add(node.path)
            fpath = repo_path / node.path
            if not fpath.exists():
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if pattern.search(line):
                        text_refs.setdefault(node.path, []).append(i)
            except (IOError, OSError):
                pass
     
        all_files = sorted(set(list(graph_refs.keys()) + list(text_refs.keys())))
        if not all_files:
            return f"No references to '{old_name}' found in the codebase."
     
        lines = [
            f"**Rename Plan: `{old_name}` -> `{new_name}`**",
            f"Found references in {len(all_files)} file(s):\n",
        ]
     
        graph_confirmed_count = 0
        text_only_count = 0
        for fpath in all_files:
            in_graph = fpath in graph_refs
            in_text = fpath in text_refs
            if in_graph and in_text:
                tag = "GRAPH+TEXT"
                graph_confirmed_count += 1
            elif in_graph:
                tag = "GRAPH-ONLY"
                graph_confirmed_count += 1
            else:
                tag = "TEXT-ONLY"
                text_only_count += 1
     
            text_line_info = ""
            previews: list[str] = []
            if in_text:
                text_line_info = f" (lines: {', '.join(str(l) for l in text_refs[fpath][:10])})"
                if len(text_refs[fpath]) > 10:
                    text_line_info = text_line_info[:-1] + f", ... +{len(text_refs[fpath]) - 10} more)"
                fpath_full = repo_path / fpath
                try:
                    file_lines = fpath_full.read_text(encoding="utf-8", errors="replace").splitlines()
                    for ln in text_refs[fpath][:5]:
                        if 0 < ln <= len(file_lines):
                            previews.append(f"    L{ln}: {file_lines[ln-1].strip()[:80]}")
                except (IOError, OSError):
                    pass
     
            lines.append(f"  [{tag}] {fpath}{text_line_info}")
            lines.extend(previews)
     
        lines.append(f"\nSummary: {graph_confirmed_count} graph-confirmed, {text_only_count} text-only.")
        lines.append(
            "\nGraph-confirmed references are high-confidence (structural dependency). "
            "Text-only references may include comments, strings, or docs — review before renaming."
        )
        return "\n".join(lines)
     
    # ── Resources ─────────────────────────────────────────────────────────
     
    @mcp.resource(
        "repolect://tree",
        name="Semantic Tree",
        description="Full hierarchical semantic tree as JSON",
        mime_type="application/json",
    )
    def resource_tree() -> str:
        _, root, _, _, _ = _load_context()
        return json.dumps(root.to_dict(), indent=2, ensure_ascii=False)
     
    @mcp.resource(
        "repolect://summary",
        name="Repo Summary",
        description="Top-level codebase overview with module list",
        mime_type="text/plain",
    )
    def resource_summary() -> str:
        _, root, _, _, _ = _load_context()  # resources use default repo/branch
        modules = "\n".join(
            f"  - {c.title}: {c.summary[:100]}" for c in root.children[:15]
        )
        return f"{root.title}\n\n{root.summary}\n\nModules:\n{modules}"
     
    # ── Prompts ───────────────────────────────────────────────────────────
     
    @mcp.prompt(
        name="code_search_guide",
        description="Step-by-step guide for using Repolect tools to answer code questions",
    )
    def prompt_code_search_guide() -> str:
        return (
            "You have access to Repolect, a code intelligence system.  "
            "Follow this workflow to answer questions about the codebase:\n\n"
            "1. Start with repo_summary() to understand the high-level structure.\n"
            "2. Use tree_search(query) to find relevant code by meaning.\n"
            "3. Use get_node(node_id) to read the full source of interesting results.\n"
            "4. Use get_node(node_id) to see full source, summary, callers, and callees.\n"
            "5. Use explain_node(node_id) if you need to understand *why* something exists.\n"
            "6. Use trace_flow(entry_point) to follow execution paths from a function.\n"
            "7. Use graph_query(cypher) for advanced structural queries.\n\n"
            "Tips:\n"
            "- Node IDs look like '0001', '0001.002', '0001.002.003' (dot-separated hierarchy).\n"
            "- tree_search uses LLM reasoning, not keyword matching — phrase queries naturally.\n"
            "- Always cite sources as (file:line) when answering questions."
        )
     
    @mcp.prompt(
        name="explain_codebase",
        description="Generate a structured explanation of the entire codebase",
    )
    def prompt_explain_codebase() -> str:
        return (
            "Using Repolect tools, generate a structured codebase explanation:\n\n"
            "1. Call repo_summary() to get the top-level overview.\n"
            "2. For each major module, call tree_search() with the module name.\n"
            "3. For key components, call get_node(node_id) for a 360-degree view.\n"
            "4. For important functions, call explain_node() for context.\n\n"
            "Organize your explanation as:\n"
            "- **Overview**: What the system does and who uses it.\n"
            "- **Architecture**: Major modules and how they connect.\n"
            "- **Key Components**: Important classes/functions and their roles.\n"
            "- **Data Flow**: How data moves through the system.\n"
            "- **Entry Points**: Where execution starts (CLI, API, etc.)."
        )
     

# ---------------------------------------------------------------------------
# Entry point — called by cli.py
# ---------------------------------------------------------------------------

def start_server():
    """Start the MCP stdio server."""
    if not HAS_MCP:
        raise ImportError(
            "MCP package not installed. Run: pip install repolect[mcp]\n"
            "Or: pip install mcp"
        )
    mcp.run(transport="stdio")