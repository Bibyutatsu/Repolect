"""
Repolect — Tree Builder
Orchestrates the full indexing pipeline:
  scan → parse → structure → link relations → build graph
  (summaries added separately by Summarizer)
 
The key design decision: build the STRUCTURE first, then fill in summaries.
This separates the fast (parsing) from the slow (LLM calls).
"""
 
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from collections import defaultdict
from .models import CodeNode, Relation
from .parser import parse_file, parse_doc_file, detect_language, is_doc_file, DOC_EXTENSIONS, extract_file_imports
from .storage import load_ignore_patterns
from .git_utils import get_file_hash, get_git_log_frequency
 
logger = logging.getLogger(__name__)
 
 
def scan_repo(repo_root: str | Path, show_progress: bool = True) -> list[Path]:
    """
    Walk the repository and return all indexable file paths.
    Respects .repolectignore and default ignore patterns.
    """
    repo_root = Path(repo_root).resolve()
    ignore_patterns = load_ignore_patterns(repo_root)
 
    # Build pathspec matcher if pathspec is available
    matcher = _build_matcher(ignore_patterns)
 
    indexable = []
    for root_dir, dirs, files in os.walk(repo_root):
        root_path = Path(root_dir)
        rel_root = root_path.relative_to(repo_root)
 
        # Filter out ignored directories in-place (prevents os.walk from descending)
        dirs[:] = [
            d for d in dirs
            if not _is_ignored(str(rel_root / d) + "/", matcher, ignore_patterns)
            and not d.startswith(".")
        ]
 
        for filename in files:
            file_path = root_path / filename
            rel_path = file_path.relative_to(repo_root)
 
            if _is_ignored(str(rel_path), matcher, ignore_patterns):
                continue
 
            lang = detect_language(file_path)
            is_doc = is_doc_file(file_path)
 
            if lang or is_doc:
                indexable.append(file_path)
 
    return sorted(indexable)
 
 
def build_raw_tree(
    repo_root: str | Path,
    repo_name: str,
    show_progress: bool = True,
    parse_workers: int | None = None,
    progress_callback = None,
    files: list[Path] | None = None,
) -> CodeNode:
    """
    Build the complete CodeNode tree without summaries.
 
    Args:
        files: Pre-scanned file list. If None, scan_repo() is called internally.
               Pass this when the CLI has already scanned to show early file counts.
 
    Structure:
      RepoNode
      ├── ModuleNode (each directory with code)
      │   ├── FileNode (each source file)
      │   │   ├── ClassNode
      │   │   │   └── MethodNode
      │   │   └── FunctionNode
      │   └── DocNode (README, etc.)
      └── ...
    """
    repo_root = Path(repo_root).resolve()
    if files is None:
        files = scan_repo(repo_root, show_progress)
        if show_progress:
            print(f"  Found {len(files)} indexable files")
 
    # Group files by their parent directory
    files_by_dir: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        rel_dir = str(f.relative_to(repo_root).parent)
        files_by_dir[rel_dir].append(f)
 
    # Build the tree bottom-up
    # Counter for node IDs — simple global counter, dot-notation built during assembly
    module_nodes = []
    module_counter = 0
    parse_workers = _resolve_worker_count(parse_workers, "REPOLECT_PARSE_WORKERS", max(4, (os.cpu_count() or 4)))
    futures: dict[str, object] = {}
 
    with ThreadPoolExecutor(max_workers=parse_workers) as executor:
        preview_module_counter = 0
        for rel_dir, dir_files in sorted(files_by_dir.items()):
            preview_module_counter += 1
            module_id = f"{preview_module_counter:04d}"
            preview_file_counter = 0
 
            for file_path in sorted(dir_files):
                preview_file_counter += 1
                file_id = f"{module_id}.{preview_file_counter:03d}"
                futures[file_id] = executor.submit(
                    _parse_indexable_file,
                    file_path,
                    repo_root,
                    file_id,
                )
 
        if progress_callback:
            completed = 0
            total_files = len(files)
            import concurrent.futures
            for future in concurrent.futures.as_completed(futures.values()):
                completed += 1
                try:
                    res = future.result()
                    rel_path = res.get("rel_path", "")
                    progress_callback(completed, total_files, rel_path)
                except Exception:
                    pass
 
    for rel_dir, dir_files in sorted(files_by_dir.items()):
        module_counter += 1
        module_id = f"{module_counter:04d}"
 
        file_nodes = []
        file_counter = 0
 
        for file_path in sorted(dir_files):
            file_counter += 1
            file_id = f"{module_id}.{file_counter:03d}"
            parsed = futures[file_id].result()
 
            if parsed["is_doc"]:
                doc_node = parsed["doc_node"]
                if doc_node:
                    doc_node.node_id = file_id
                    _enrich_with_git(doc_node, file_path, repo_root)
                    file_nodes.append(doc_node)
                continue
 
            file_node = CodeNode(
                node_id=file_id,
                title=file_path.name,
                kind="file",
                path=parsed["rel_path"],
                line_start=1,
                line_end=parsed["line_end"],
                language=parsed["language"],
                git_hash=parsed["git_hash"],
                children=parsed["symbol_nodes"],
            )
            _enrich_with_git(file_node, file_path, repo_root)
            extract_file_imports(file_path, repo_root, file_node)
            file_nodes.append(file_node)
 
        if not file_nodes:
            continue
 
        # Determine module title (make it readable)
        module_title = rel_dir if rel_dir != "." else repo_name
        module_path = rel_dir if rel_dir != "." else ""
 
        module_node = CodeNode(
            node_id=module_id,
            title=module_title,
            kind="module",
            path=module_path,
            children=file_nodes,
        )
        module_nodes.append(module_node)
 
    # Root repo node
    root = CodeNode(
        node_id="0000",
        title=repo_name,
        kind="repo",
        path="",
        children=module_nodes,
    )
 
    # Resolve relations across the whole tree
    _resolve_relations(root)
 
    # Cross-file CALLS: match function calls to symbols in imported files
    _resolve_cross_file_calls(root, repo_root)
 
    # Compute usage counts (how many nodes import/call each node)
    _compute_usage_counts(root)
 
    return root
 
 
def _resolve_relations(root: CodeNode) -> None:
    """
    Attempt to resolve IMPORTS relations from "external:module_name"
    to actual node_ids where possible.
    
    This is intentionally best-effort — unresolvable imports stay as "external:*"
    """
    node_map = root.get_node_map()
 
    # Build a reverse map: path → node_id for file resolution
    path_to_node: dict[str, str] = {}
    for node in node_map.values():
        if node.kind in ("file", "module") and node.path:
            path_to_node[node.path] = node.node_id
            # Also index without extension
            p = Path(node.path)
            if p.name:  # Guard: PosixPath('.').with_suffix('') crashes
                path_to_node[str(p.with_suffix(""))] = node.node_id
                path_to_node[p.stem] = node.node_id
 
    for node in node_map.values():
        for relation in node.relations:
            if relation.target_id.startswith("external:"):
                module_name = relation.target_id[len("external:"):]
                # Try to resolve to an internal node
                resolved = (
                    path_to_node.get(module_name)
                    or path_to_node.get(module_name.replace(".", "/"))
                    or path_to_node.get(module_name.split(".")[-1])
                )
                if resolved:
                    relation.target_id = resolved
 
 
def _resolve_cross_file_calls(root: CodeNode, repo_root: Path) -> None:
    """Create cross-file CALLS relations using IMPORTS as disambiguation.
 
    For each file that has resolved IMPORTS (pointing to internal file nodes),
    scan every function/method body for calls to symbols defined in those
    imported files and add CALLS relations.
    """
    import re as _re
    node_map = root.get_node_map()
 
    # Build global symbol index: symbol_name -> [(node_id, file_node_id)]
    global_symbols: dict[str, list[tuple[str, str]]] = defaultdict(list)
    file_of: dict[str, str] = {}
    for node in node_map.values():
        if node.kind in ("function", "method", "class") and node.path:
            parent_file_id = _find_parent_file_id(root, node.node_id)
            if parent_file_id:
                global_symbols.setdefault(node.title, []).append(
                    (node.node_id, parent_file_id)
                )
                file_of[node.node_id] = parent_file_id
 
    if not global_symbols:
        return
 
    added = 0
    for file_node in node_map.values():
        if file_node.kind != "file" or not file_node.path:
            continue
 
        imported_file_ids: set[str] = set()
        for rel in file_node.relations:
            if rel.kind == "IMPORTS" and not rel.target_id.startswith("external:"):
                imported_file_ids.add(rel.target_id)
 
        if not imported_file_ids:
            continue
 
        symbols_in_imports: dict[str, str] = {}
        for sym_name, entries in global_symbols.items():
            for nid, fid in entries:
                if fid in imported_file_ids:
                    symbols_in_imports[sym_name] = nid
                    break
 
        if not symbols_in_imports:
            continue
 
        file_path = repo_root / file_node.path
        try:
            source_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except (IOError, OSError):
            continue
 
        for child in file_node.children:
            if child.kind not in ("function", "method"):
                continue
            if child.line_start <= 0 or child.line_end <= 0:
                continue
 
            body = "\n".join(source_lines[child.line_start - 1 : child.line_end])
            existing_targets = {r.target_id for r in child.relations if r.kind == "CALLS"}
 
            for sym_name, target_id in symbols_in_imports.items():
                if target_id == child.node_id or target_id in existing_targets:
                    continue
                pattern = rf"(?<!\w){_re.escape(sym_name)}\s*\("
                if _re.search(pattern, body):
                    child.relations.append(Relation(
                        source_id=child.node_id,
                        target_id=target_id,
                        kind="CALLS",
                        label=f"{child.title} calls {sym_name} (cross-file)",
                    ))
                    existing_targets.add(target_id)
                    added += 1
 
    if added:
        logger.info("Resolved %d cross-file CALLS relations", added)
 
 
def _find_parent_file_id(root: CodeNode, target_id: str) -> str | None:
    """Find the file node_id that contains a given symbol node."""
    for node in root.flat_iter():
        if node.kind == "file":
            for child in node.children:
                if child.node_id == target_id:
                    return node.node_id
    return None
 
 
def _compute_usage_counts(root: CodeNode) -> None:
    """Count how many other nodes have a relation pointing to each node."""
    counts: dict[str, int] = defaultdict(int)
 
    for node in root.flat_iter():
        for relation in node.relations:
            if not relation.target_id.startswith("external:"):
                counts[relation.target_id] += 1
 
    node_map = root.get_node_map()
    for node_id, count in counts.items():
        if node_id in node_map:
            node_map[node_id].usage_count = count
 
 
def _enrich_with_git(node: CodeNode, file_path: Path, repo_root: Path) -> None:
    """Add git-derived metadata to a node."""
    try:
        node.change_frequency = get_git_log_frequency(repo_root, file_path)
    except Exception:
        node.change_frequency = 0
 
 
def _count_lines(file_path: Path) -> int:
    try:
        return sum(1 for _ in open(file_path, encoding="utf-8", errors="replace"))
    except (IOError, OSError):
        return 0
 
 
def _build_matcher(patterns: list[str]):
    """Build a pathspec matcher if pathspec is available."""
    try:
        import pathspec
        return pathspec.PathSpec.from_lines("gitignore", patterns)
    except ImportError:
        return None
 
 
def _is_ignored(rel_path: str, matcher, patterns: list[str]) -> bool:
    if matcher:
        return matcher.match_file(rel_path)
    # Simple fallback without pathspec
    for pattern in patterns:
        if pattern.rstrip("/") in rel_path:
            return True
        if rel_path.endswith(pattern.lstrip("*")):
            return True
    return False
 
 
def count_nodes(root: CodeNode) -> int:
    return sum(1 for _ in root.flat_iter())
 
 
def count_files(root: CodeNode) -> int:
    return sum(1 for n in root.flat_iter() if n.kind == "file")
 
 
def get_language_stats(root: CodeNode) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for node in root.flat_iter():
        if node.kind == "file" and node.language:
            stats[node.language] += 1
    return dict(stats)
 
 
# ── Incremental sync support ─────────────────────────────────────────────────
 
def find_stale_nodes(
    root: CodeNode, repo_root: str | Path,
) -> tuple[list[str], list[str]]:
    """Compare current file hashes to stored hashes.
 
    Returns:
        (stale, deleted) — stale are changed files, deleted are missing from disk.
    """
    stale: list[str] = []
    deleted: list[str] = []
    repo_root = Path(repo_root).resolve()
 
    for node in root.flat_iter():
        if node.kind not in ("file", "doc") or not node.path:
            continue
        current_hash = get_file_hash(repo_root / node.path)
        if not current_hash:
            deleted.append(node.node_id)
        elif current_hash != node.git_hash:
            stale.append(node.node_id)
 
    return stale, deleted
 
 
def find_orphan_nodes(root: CodeNode, repo_root: str | Path) -> list[str]:
    """Identify tree nodes whose files are no longer in the scan result.
 
    Catches files added to .gitignore or .repolectignore since last index.
    """
    repo_root = Path(repo_root).resolve()
    current_files = {
        str(p.relative_to(repo_root))
        for p in scan_repo(repo_root, show_progress=False)
    }
    orphans: list[str] = []
    for node in root.flat_iter():
        if node.kind in ("file", "doc") and node.path and node.path not in current_files:
            orphans.append(node.node_id)
    return orphans
 
 
def get_ancestor_ids(root: CodeNode, node_id: str) -> list[str]:
    """Return node_ids of all ancestors of a given node (for summary roll-up)."""
    ancestors = []
 
    def walk(current: CodeNode, target: str, path: list[str]) -> bool:
        if current.node_id == target:
            ancestors.extend(path)
            return True
        for child in current.children:
            if walk(child, target, path + [current.node_id]):
                return True
        return False
 
    walk(root, node_id, [])
    return ancestors
 
 
# ── Git-diff to node mapping ─────────────────────────────────────────────────
 
def map_changes_to_nodes(
    changed_ranges: dict[str, list[tuple[int, int]]],
    root: CodeNode,
) -> list[CodeNode]:
    """Map changed line ranges to affected CodeNodes.
 
    Args:
        changed_ranges: {relative_path: [(start, end), ...]} from git_utils.get_changed_line_ranges.
        root: The root CodeNode of the semantic tree.
 
    Returns:
        List of CodeNodes whose line ranges overlap with changed lines.
    """
    norm_ranges = {k.replace("\\", "/"): v for k, v in changed_ranges.items()}
 
    affected: list[CodeNode] = []
    seen: set[str] = set()
 
    for node in root.flat_iter():
        if node.kind in ("repo", "module") or not node.path:
            continue
        node_path = node.path.replace("\\", "/")
        file_ranges = norm_ranges.get(node_path)
        if not file_ranges:
            continue
        if node.kind == "file":
            continue
        for rng_start, rng_end in file_ranges:
            if node.line_start <= rng_end and node.line_end >= rng_start:
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    affected.append(node)
                break
 
    return affected
 
 
# ── Graph building ──────────────────────────────────────────────────────────
 
def build_graph(root: CodeNode, graph_db) -> None:
    """
    Populate the graph database from the semantic tree.
    Adds every node and all relations (CONTAINS, IMPORTS, CALLS) as edges.
 
    Two-pass strategy: add ALL nodes first, then ALL edges.
    This ensures edge targets exist in the graph before edges reference them.
 
    Args:
        root: Root CodeNode of the semantic tree.
        graph_db: A GraphDB instance (from graph_db.py).
    """
    if graph_db is None:
        return
 
    node_count = 0
    edge_count = 0
    node_map = root.get_node_map()
 
    # Pass 1: Add all nodes
    for node in root.flat_iter():
        props = {
            "title": node.title,
            "kind": node.kind,
            "file_path": node.path or "",
            "language": node.language or "",
            "line_start": node.line_start,
            "line_end": node.line_end,
            "summary": (node.summary or "")[:500],
        }
        graph_db.add_node(node.node_id, **props)
        node_count += 1
 
    # Pass 2: Add all edges
    for node in root.flat_iter():
        # CONTAINS edges (parent → child)
        for child in node.children:
            graph_db.add_edge(node.node_id, child.node_id, "CONTAINS",
                              label=f"{node.title} contains {child.title}")
            edge_count += 1
 
        # Explicit relations (IMPORTS, CALLS, etc.)
        for rel in node.relations:
            if not rel.target_id.startswith("external:"):
                # Only add edges to nodes that exist in our tree
                if rel.target_id in node_map:
                    graph_db.add_edge(rel.source_id, rel.target_id, rel.kind,
                                      label=rel.label or "")
                    edge_count += 1
 
    # Community detection (Louvain) — store community_id on each graph node
    try:
        communities = graph_db.detect_communities()
        for nid, comm_id in communities.items():
            graph_db.add_node(nid, community_id=comm_id)
        logger.info("Detected %d communities", len(set(communities.values())))
    except NotImplementedError:
        logger.info("Community detection not available on %s backend", graph_db.backend_name)
 
    graph_db.save()
    logger.info("Graph built: %d nodes, %d edges", node_count, edge_count)
 
 
def reparse_stale_files(
    root: CodeNode,
    repo_root: str | Path,
    stale_node_ids: list[str],
    graph_db=None,
    parse_workers: int | None = None,
) -> list[str]:
    """
    Re-parse stale files and update their children in the tree.
    Also refreshes graph edges for affected nodes.
 
    Returns list of all node_ids that need re-summarization
      (stale nodes + their ancestors).
    """
    repo_root = Path(repo_root).resolve()
    node_map = root.get_node_map()
    all_affected = set(stale_node_ids)
    old_child_ids_map: dict[str, set[str]] = {}
    parsed_results: dict[str, list[CodeNode]] = {}
    parse_workers = _resolve_worker_count(parse_workers, "REPOLECT_PARSE_WORKERS", max(4, (os.cpu_count() or 4)))
 
    futures: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=parse_workers) as executor:
        for node_id in stale_node_ids:
            node = node_map.get(node_id)
            if not node or node.kind != "file" or not node.path:
                continue
            file_path = repo_root / node.path
            if file_path.exists():
                futures[node_id] = executor.submit(parse_file, file_path, repo_root, node_id)
 
        for node_id, future in futures.items():
            nodes = future.result()
            parsed_results[node_id] = nodes
 
    for node_id in stale_node_ids:
        node = node_map.get(node_id)
        if not node or node.kind not in ("file", "doc") or not node.path:
            continue
 
        file_path = repo_root / node.path
 
        if not file_path.exists():
            node.children = []
            node.summary = "[file deleted]"
            if graph_db:
                graph_db.delete_edges_for_node(node_id)
            parent_id = _find_parent_id(root, node_id)
            if parent_id:
                parent_node = node_map.get(parent_id)
                if parent_node:
                    parent_node.children = [
                        c for c in parent_node.children if c.node_id != node_id
                    ]
            continue
 
        # Update hash
        node.git_hash = get_file_hash(file_path)
 
        if node.kind == "doc":
            node.summary = ""  # will be re-summarized
            continue
 
        old_child_ids = {c.node_id for c in node.children}
        old_child_ids_map[node_id] = old_child_ids
        new_children = parsed_results.get(node_id, [])
        node.children = new_children
        node.summary = ""  # mark for re-summarization
        node.line_end = _count_lines(file_path)
 
        node.relations = [r for r in node.relations if r.kind != "IMPORTS"]
        extract_file_imports(file_path, repo_root, node)
 
        for child in node.children:
            all_affected.add(child.node_id)
 
        ancestors = get_ancestor_ids(root, node_id)
        all_affected.update(ancestors)
 
    _resolve_relations(root)
    _resolve_cross_file_calls(root, repo_root)
    _compute_usage_counts(root)
 
    if graph_db:
        refreshed_node_map = root.get_node_map()
        for node_id in stale_node_ids:
            node = refreshed_node_map.get(node_id)
            if not node or node.kind not in ("file", "doc"):
                continue
 
            for old_id in old_child_ids_map.get(node_id, set()):
                graph_db.delete_node(old_id)
            graph_db.delete_edges_for_node(node_id)
 
            if node.summary == "[file deleted]":
                graph_db.delete_node(node_id)
                continue
 
            graph_db.add_node(
                node_id,
                title=node.title,
                kind=node.kind,
                file_path=node.path,
                language=node.language or "",
                line_start=node.line_start,
                line_end=node.line_end,
                summary=(node.summary or "")[:500],
            )
 
            parent_id = _find_parent_id(root, node_id) or root.node_id
            graph_db.add_edge(parent_id, node_id, "CONTAINS", label=f"contains {node.title}")
 
            for child in node.children:
                graph_db.add_node(
                    child.node_id,
                    title=child.title,
                    kind=child.kind,
                    file_path=child.path,
                    language=child.language or "",
                    line_start=child.line_start,
                    line_end=child.line_end,
                    summary=(child.summary or "")[:500],
                )
 
            for child in node.children:
                graph_db.add_edge(node_id, child.node_id, "CONTAINS", label=f"{node.title} contains {child.title}")
                for rel in child.relations:
                    if not rel.target_id.startswith("external:") and rel.target_id in refreshed_node_map:
                        graph_db.add_edge(rel.source_id, rel.target_id, rel.kind, label=rel.label or "")
 
            for rel in node.relations:
                if not rel.target_id.startswith("external:") and rel.target_id in refreshed_node_map:
                    graph_db.add_edge(rel.source_id, rel.target_id, rel.kind, label=rel.label or "")
 
        graph_db.save()
 
    return list(all_affected)
 
 
def _resolve_worker_count(explicit: int | None, env_name: str, default: int) -> int:
    value = explicit
    if value is None:
        raw = os.environ.get(env_name, "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = default
    if value is None:
        value = default
    return max(1, value)
 
 
def _parse_indexable_file(file_path: Path, repo_root: Path, file_id: str):
    if is_doc_file(file_path):
        return {
            "is_doc": True,
            "doc_node": parse_doc_file(file_path, repo_root, f"{file_id}.doc"),
            "rel_path": str(file_path.relative_to(repo_root)),
        }
 
    nodes = parse_file(file_path, repo_root, file_id)
    return {
        "is_doc": False,
        "symbol_nodes": nodes,
        "rel_path": str(file_path.relative_to(repo_root)),
        "line_end": _count_lines(file_path),
        "language": detect_language(file_path) or "",
        "git_hash": get_file_hash(file_path),
    }
 
 
def _find_parent_id(root: CodeNode, target_id: str) -> str | None:
    """Find the parent node_id of a given node."""
    for node in root.flat_iter():
        for child in node.children:
            if child.node_id == target_id:
                return node.node_id
    return None
 