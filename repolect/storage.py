"""
Repolect — Storage Layer
Manages .repolect/ per-repo and ~/.repolect/registry.json global registry.
 
Design: everything is plain JSON. No database. The tree.json IS the database.
A 10,000 node tree compresses to ~2MB JSON — fast to load, easy to inspect.
"""
 
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
from .models import CodeNode, TreeMeta  # noqa: E402
 
 
# ── Directory layout ────────────────────────────────────────────────────────
 
REPOLECT_DIR = ".repolect"
BRANCHES_DIR = "branches"
TREE_FILE = "tree.json"
META_FILE = "meta.json"
CACHE_DIR = "cache"
LLM_CACHE_DB = "llm_cache.sqlite"
IGNORE_FILE = ".repolectignore"
CONTEXT_FILE = "REPOLECT.md"
 
GLOBAL_DIR = Path.home() / ".repolect"
REGISTRY_FILE = GLOBAL_DIR / "registry.json"
 
 
# ── Per-repo operations ──────────────────────────────────────────────────────
 
def get_index_dir(repo_root: str | Path, branch: str | None = None) -> Path:
    """Return the index directory for a repo, optionally scoped to a branch.
 
    With branch=None: .repolect/  (legacy / top-level)
    With branch set:  .repolect/branches/<branch>/
    """
    base = Path(repo_root) / REPOLECT_DIR
    if branch:
        return base / BRANCHES_DIR / branch
    return base
 
 
def get_cache_dir(repo_root: str | Path) -> Path:
    """Shared LLM cache directory (not branch-scoped)."""
    return Path(repo_root) / REPOLECT_DIR / CACHE_DIR
 
 
def ensure_index_dir(repo_root: str | Path, branch: str | None = None) -> Path:
    index_dir = get_index_dir(repo_root, branch=branch)
    index_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(repo_root)
    return index_dir
 
 
def save_tree(root: CodeNode, repo_root: str | Path, branch: str | None = None) -> Path:
    """Serialize the full tree to the branch-scoped index directory."""
    index_dir = ensure_index_dir(repo_root, branch=branch)
    tree_path = index_dir / TREE_FILE
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(root.to_dict(), f, indent=2, ensure_ascii=False)
    return tree_path
 
 
def load_tree(repo_root: str | Path, branch: str | None = None) -> CodeNode:
    """Load the tree from the branch-scoped index directory."""
    tree_path = get_index_dir(repo_root, branch=branch) / TREE_FILE
    if not tree_path.exists():
        raise FileNotFoundError(
            f"No index found at {tree_path}. Run `repolect analyze` first."
        )
    with open(tree_path, encoding="utf-8") as f:
        data = json.load(f)
    return CodeNode.from_dict(data)
 
 
def save_meta(meta: TreeMeta, repo_root: str | Path, branch: str | None = None) -> Path:
    index_dir = ensure_index_dir(repo_root, branch=branch)
    meta_path = index_dir / META_FILE
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2)
    return meta_path
 
 
def load_meta(repo_root: str | Path, branch: str | None = None) -> TreeMeta | None:
    meta_path = get_index_dir(repo_root, branch=branch) / META_FILE
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    return TreeMeta.from_dict(data)
 
 
def tree_exists(repo_root: str | Path, branch: str | None = None) -> bool:
    return (get_index_dir(repo_root, branch=branch) / TREE_FILE).exists()
 
 
def list_branches(repo_root: str | Path) -> list[str]:
    """Return branch names that have an index stored."""
    branches_dir = Path(repo_root) / REPOLECT_DIR / BRANCHES_DIR
    if not branches_dir.exists():
        return []
    return sorted(
        d.name for d in branches_dir.iterdir()
        if d.is_dir() and (d / TREE_FILE).exists()
    )
 
 
def migrate_legacy_index(repo_root: str | Path, branch: str) -> bool:
    """Move a legacy flat .repolect/ index into .repolect/branches/<branch>/.
 
    Returns True if migration happened.
    """
    base = Path(repo_root) / REPOLECT_DIR
    legacy_tree = base / TREE_FILE
    branches_dir = base / BRANCHES_DIR
 
    if not legacy_tree.exists() or branches_dir.exists():
        return False
 
    target = branches_dir / branch
    target.mkdir(parents=True, exist_ok=True)
 
    for name in (TREE_FILE, META_FILE, "graph.pkl", "graph.db"):
        src = base / name
        if src.exists():
            shutil.move(str(src), str(target / name))
 
    return True
 
 
def clean_index(repo_root: str | Path, branch: str | None = None) -> None:
    """Remove the index. If branch is set, only that branch's index."""
    if branch:
        index_dir = get_index_dir(repo_root, branch=branch)
        if index_dir.exists():
            shutil.rmtree(index_dir)
    else:
        index_dir = get_index_dir(repo_root)
        if index_dir.exists():
            shutil.rmtree(index_dir)
 
 
_MARKER_START = "<!-- repolect:start -->"
_MARKER_END = "<!-- repolect:end -->"
 
 
def write_context_file(meta: TreeMeta, root: CodeNode, repo_root: str | Path, graph_db=None) -> Path:
    """Write REPOLECT.md with prescriptive agent rules and marker-based upsert.
 
    If the file already exists with repolect markers, only the section between
    markers is replaced — user-written content outside is preserved.
    """
    lang_list = ", ".join(f"{lang} ({count})" for lang, count in meta.language_stats.items())
    git_display = f"`{meta.git_commit[:8]}`" if meta.git_commit else "_no git_"
 
    graph_stats = ""
    if graph_db:
        graph_stats = (
            f"- **Graph**: {graph_db.node_count()} nodes, "
            f"{graph_db.edge_count()} edges ({graph_db.backend_name})\n"
            f"- **Embeddings**: {'enabled' if meta.embeddings_enabled else 'disabled'}"
        )
 
    community_map = _build_community_map(graph_db, root) if graph_db else ""
 
    generated = f"""{_MARKER_START}
# Repolect — Repository Intelligence
 
{root.summary}
 
## Index Stats
 
- **{meta.node_count} symbols** indexed across **{meta.file_count} files**
- **Languages**: {lang_list or "unknown"}
- **Last indexed**: {meta.indexed_at[:10]}
- **Git commit**: {git_display}
{graph_stats}
 
## Always Do
 
1. **Before implementing any feature or change**, call `plan_change(description)` to get a structured roadmap (ADD / MODIFY / READ_ONLY / TEST_AFTER).
2. **Before creating new code**, call `find_similar(description)` to find an existing template to copy from.
3. **Before modifying code in an unfamiliar area**, call `get_conventions(node_id)` to match local style.
4. **Before committing**, call `diff_analysis()` to verify blast radius of your changes.
5. **After modifying code**, call `scope_test(node_ids)` with changed node IDs to identify tests to run.
 
## Never Do
 
- Never skip `plan_change` for multi-file changes — it prevents missed files and wrong edit order.
- Never modify a widely-used symbol without first calling `impact_analysis` — know the blast radius.
- Never commit without running `diff_analysis` — it catches unintended downstream effects.
 
## When Debugging
 
```
tree_search("error message or symptom")  →  Find suspect code
get_node(node_id)                        →  Full source + callers + callees
trace_flow(entry_point)                  →  Execution chain from entry point
impact_analysis(node_id)                 →  What else is affected
scope_test(node_ids)                     →  Tests to verify the fix
```
 
## When Refactoring
 
```
plan_change("describe the refactoring")  →  What to modify, read, test
get_conventions(node_id)                 →  Match target area's patterns
impact_analysis(node_id)                 →  Blast radius before changing
[implement]                              →  Follow MODIFY list in order
scope_test(node_ids)                     →  Minimal test set
diff_analysis()                          →  Final safety check
```
 
## Tool Quick Reference
 
| Tool | When to Use |
|------|-------------|
| `plan_change` | Before any feature/change — structured roadmap |
| `find_similar` | Before creating new code — find a template |
| `get_conventions` | Before modifying — match local style |
| `tree_search` | Find code by meaning — "how does X work?" |
| `get_node` | Full symbol details: source + callers + callees |
| `trace_flow` | Follow execution flow from an entry point |
| `explain_node` | Narrative explanation of why a symbol exists |
| `impact_analysis` | Blast radius — what depends on a symbol |
| `diff_analysis` | Pre-commit — map changes to affected symbols |
| `scope_test` | After changes — minimal test set to run |
| `rename` | Multi-file rename plan with confidence tags |
| `graph_query` | Custom Cypher queries against the knowledge graph |
| `repo_summary` | High-level codebase overview |
| `list_repos` | List all indexed repositories |
{community_map}
## Top-Level Modules
 
{_format_modules(root)}
 
## MCP Config
 
```json
{{
  "mcpServers": {{
    "repolect": {{
      "command": "repolect",
      "args": ["mcp"]
    }}
  }}
}}
```
{_MARKER_END}"""
 
    context_path = Path(repo_root) / CONTEXT_FILE
    if context_path.exists():
        existing = context_path.read_text(encoding="utf-8")
        if _MARKER_START in existing and _MARKER_END in existing:
            before = existing[:existing.index(_MARKER_START)]
            after = existing[existing.index(_MARKER_END) + len(_MARKER_END):]
            content = before + generated + after
        else:
            content = generated
    else:
        content = generated
 
    with open(context_path, "w", encoding="utf-8") as f:
        f.write(content)
    return context_path
 
 
def _build_community_map(graph_db, root: CodeNode) -> str:
    """Build a community map section from Louvain detection results."""
    from collections import defaultdict
 
    try:
        communities = graph_db.detect_communities()
    except Exception:
        return ""
 
    if not communities:
        return ""
 
    node_map = root.get_node_map()
    groups: dict[int, list[str]] = defaultdict(list)
    for nid, comm_id in communities.items():
        groups[comm_id].append(nid)
 
    if len(groups) < 2:
        return ""
 
    lines = ["\n## Codebase Areas (Communities)\n"]
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    used_labels: set[str] = set()
 
    for comm_id, member_ids in sorted_groups[:15]:
        nodes = [node_map[nid] for nid in member_ids if nid in node_map]
        if not nodes:
            continue
 
        non_test = [n for n in nodes if not (n.path and ("test_" in n.path or "/tests/" in n.path))]
        if len(non_test) < 2:
            continue
 
        file_nodes = [n for n in non_test if n.kind == "file" and n.title]
        label = None
        if file_nodes:
            stems: dict[str, int] = defaultdict(int)
            for fn in file_nodes:
                stem = fn.title.split(".")[0]
                stems[stem] += 1
            for candidate in sorted(stems, key=stems.get, reverse=True):
                if candidate not in used_labels:
                    label = candidate
                    break
        if not label:
            class_nodes = [n for n in non_test if n.kind == "class" and n.title]
            if class_nodes:
                candidate = class_nodes[0].title
                if candidate not in used_labels:
                    label = candidate
        if not label:
            label = f"area-{comm_id}"
 
        used_labels.add(label)
        key_symbols = [n.title for n in non_test if n.kind in ("class", "function") and n.title][:5]
        sym_str = ", ".join(f"`{s}`" for s in key_symbols) if key_symbols else ""
 
        lines.append(f"- **{label}** ({len(non_test)} symbols): {sym_str}")
 
    if len(lines) < 2:
        return ""
 
    lines.append("")
    return "\n".join(lines)
 
 
def _format_modules(root: CodeNode) -> str:
    lines = []
    for child in root.children[:10]:
        lines.append(f"- **{child.title}** ({child.path}): {child.summary[:100]}...")
    if len(root.children) > 10:
        lines.append(f"- *...and {len(root.children) - 10} more*")
    return "\n".join(lines)
 
 
# ── Global registry ──────────────────────────────────────────────────────────
 
def _load_registry() -> list[dict]:
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        return []
    with open(REGISTRY_FILE, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
 
 
def _save_registry(entries: list[dict]) -> None:
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
 
 
def register_repo(repo_path: str | Path, meta: TreeMeta) -> None:
    """Add or update a repo entry in the global registry."""
    repo_path = str(Path(repo_path).resolve())
    entries = _load_registry()
 
    existing = None
    for e in entries:
        if e.get("repo_path") == repo_path:
            existing = e
            break
 
    indexed_branches = list_branches(repo_path)
 
    if existing:
        existing.update({
            "repo_id": meta.repo_id,
            "repo_name": meta.repo_name,
            "git_commit": meta.git_commit,
            "git_branch": getattr(meta, "git_branch", ""),
            "indexed_at": meta.indexed_at,
            "node_count": meta.node_count,
            "file_count": meta.file_count,
            "graph_backend": meta.graph_backend,
            "embeddings_enabled": meta.embeddings_enabled,
            "branches": indexed_branches,
        })
    else:
        entries.append({
            "repo_id": meta.repo_id,
            "repo_name": meta.repo_name,
            "repo_path": repo_path,
            "git_commit": meta.git_commit,
            "git_branch": getattr(meta, "git_branch", ""),
            "indexed_at": meta.indexed_at,
            "node_count": meta.node_count,
            "file_count": meta.file_count,
            "graph_backend": meta.graph_backend,
            "embeddings_enabled": meta.embeddings_enabled,
            "branches": indexed_branches,
        })
    _save_registry(entries)
 
 
def unregister_repo(repo_path: str | Path) -> bool:
    repo_path = str(Path(repo_path).resolve())
    entries = _load_registry()
    before = len(entries)
    entries = [e for e in entries if e.get("repo_path") != repo_path]
    if len(entries) < before:
        _save_registry(entries)
        return True
    return False
 
 
def list_repos() -> list[dict]:
    """Return all registered repos, filtering out ones that no longer exist."""
    entries = _load_registry()
    valid = []
    for entry in entries:
        path = entry.get("repo_path", "")
        if not Path(path).exists():
            continue
        branches = list_branches(path)
        has_legacy = tree_exists(path, branch=None)
        if branches or has_legacy:
            entry["branches"] = branches
            valid.append(entry)
    if len(valid) != len(entries):
        _save_registry(valid)
    return valid
 
 
def find_repo(name_or_id: str) -> dict | None:
    """Find a repo by name or repo_id. Returns None if not found or ambiguous."""
    entries = list_repos()
    # Exact repo_id match
    for e in entries:
        if e.get("repo_id") == name_or_id:
            return e
    # Exact name match
    matches = [e for e in entries if e.get("repo_name") == name_or_id]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Ambiguous — return None, caller should handle with repo_id
        return None
    # Fuzzy name match
    matches = [e for e in entries if name_or_id.lower() in e.get("repo_name", "").lower()]
    if len(matches) == 1:
        return matches[0]
    return None
 
 
# ── .repolectignore ────────────────────────────────────────────────────────
 
DEFAULT_IGNORE_PATTERNS = [
    # Dependencies
    "node_modules/", ".venv/", "venv/", "env/", ".env/",
    "__pycache__/", "*.pyc", "*.pyo",
    # Build outputs
    "dist/", "build/", "out/", "target/", ".next/", ".nuxt/",
    "*.min.js", "*.bundle.js",
    # Generated files
    "*.generated.*", "*.pb.go", "*_pb2.py",
    # Media and data
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.ico", "*.svg",
    "*.pdf", "*.zip", "*.tar.gz",
    # Lock files (usually not useful for understanding)
    "package-lock.json", "yarn.lock", "Pipfile.lock", "poetry.lock",
    # Hidden dirs
    ".git/", ".github/", ".vscode/", ".idea/",
    # Our own output
    ".repolect/",
]
 
 
def load_ignore_patterns(repo_root: str | Path) -> list[str]:
    """Load .repolectignore, .gitignore, and merge with defaults."""
    repo_root = Path(repo_root)
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    for ignore_name in (IGNORE_FILE, ".gitignore"):
        ignore_path = repo_root / ignore_name
        if ignore_path.exists():
            try:
                with open(ignore_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and line not in patterns:
                            patterns.append(line)
            except (IOError, OSError):
                pass
    return patterns
 
 
# ── LLM call-level disk cache ────────────────────────────────────────────────
 
class LLMDiskCache:
    """SQLite-backed cache for LLM completions.
 
    Caches at the raw LLM call level: every ``BaseLLM.complete()`` call is
    keyed by ``(provider, model, max_tokens, prompt)``.  SQLite WAL mode
    ensures each INSERT is immediately durable — if the process is killed,
    all completed calls survive on disk.
 
    Thread-safe via an internal lock around writes.
    """
 
    def __init__(self, repo_root: str | Path):
        cache_dir = get_cache_dir(repo_root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / LLM_CACHE_DB
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(k TEXT PRIMARY KEY, v TEXT NOT NULL, created_at TEXT)"
        )
        self._conn.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
 
    @staticmethod
    def make_key(provider: str, model: str, max_tokens: int, prompt: str) -> str:
        raw = f"{provider}::{model}::{max_tokens}::{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()
 
    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT v FROM cache WHERE k = ?", (key,)
        ).fetchone()
        if row:
            self.hits += 1
            return row[0]
        self.misses += 1
        return None
 
    def put(self, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (k, v, created_at) VALUES (?, ?, ?)",
                (key, value, now),
            )
            self._conn.commit()
 
    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()
 
    def close(self) -> None:
        self._conn.close()
 
    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return row[0] if row else 0
 
    def purge_errors(self, sentinel: str = "[summary unavailable:") -> int:
        """Delete cached values that are LLM error responses."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache WHERE v LIKE ?", (sentinel + "%",)
            )
            self._conn.commit()
            return cursor.rowcount
 
    def bulk_insert(self, entries: dict[str, str]) -> int:
        """Insert multiple key-value pairs at once. Returns count inserted."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO cache (k, v, created_at) VALUES (?, ?, ?)",
                [(k, v, now) for k, v in entries.items()],
            )
            self._conn.commit()
        return len(entries)
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def _ensure_gitignored(repo_root: str | Path) -> None:
    """Make sure .repolect is in .gitignore."""
    gitignore_path = Path(repo_root) / ".gitignore"
    entry = ".repolect/"
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        if entry not in content and REPOLECT_DIR not in content:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Repolect\n{entry}\n{CONTEXT_FILE}\n")
    else:
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(f"# Repolect\n{entry}\n{CONTEXT_FILE}\n")
 