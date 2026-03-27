"""
Install Repolect skills into editor-specific directories.
 
Detects which editors are present (Cursor, Claude Code) and copies
skill files accordingly:
  - Cursor:     .cursor/rules/repolect-<name>.mdc
  - Claude Code: .claude/skills/repolect/<name>.md
 
Static skills are shipped with the package in repolect/skills/.
Generated community skills are produced by generate_community_skills().
"""
 
from __future__ import annotations
 
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any
 
logger = logging.getLogger(__name__)
 
STATIC_SKILLS_DIR = Path(__file__).parent / "skills"
 
CURSOR_RULES_DIR = ".cursor/rules"
CLAUDE_SKILLS_DIR = ".claude/skills/repolect"
 
SKILL_FILES = [
    "exploring.md",
    "debugging.md",
    "planning.md",
    "refactoring.md",
    "reviewing.md",
]
 
 
def _detect_editors(repo_root: Path) -> list[str]:
    """Detect which editors have config in this repo or globally."""
    editors: list[str] = []
    if (repo_root / ".cursor").exists() or (Path.home() / ".cursor").exists():
        editors.append("cursor")
    if (repo_root / ".claude").exists() or (Path.home() / ".claude").exists():
        editors.append("claude")
    if not editors:
        editors.append("cursor")
    return editors
 
 
def _read_skill(name: str) -> tuple[dict[str, str], str]:
    """Read a skill file and parse YAML frontmatter + body."""
    path = STATIC_SKILLS_DIR / name
    text = path.read_text(encoding="utf-8")
    frontmatter: dict[str, str] = {}
    body = text
 
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    frontmatter[key.strip()] = val.strip()
            body = parts[2].strip()
 
    return frontmatter, body
 
 
def _to_cursor_mdc(frontmatter: dict[str, str], body: str) -> str:
    """Convert a skill file into Cursor .mdc format."""
    desc = frontmatter.get("description", "")
    globs = frontmatter.get("globs", "")
    always = frontmatter.get("alwaysApply", "false")
    return f"""---
description: {desc}
globs: {globs}
alwaysApply: {always}
---
 
{body}
"""
 
 
def _to_claude_skill(frontmatter: dict[str, str], body: str) -> str:
    """Convert a skill file into Claude Code skill format."""
    name = frontmatter.get("name", "repolect-skill")
    desc = frontmatter.get("description", "")
    return f"""---
name: {name}
description: "{desc}"
---
 
{body}
"""
 
 
def install_static_skills(repo_root: str | Path) -> list[str]:
    """Install static skill files into detected editor directories.
 
    Returns list of installed file paths (relative to repo root).
    """
    repo_root = Path(repo_root)
    editors = _detect_editors(repo_root)
    installed: list[str] = []
 
    for editor in editors:
        for skill_name in SKILL_FILES:
            frontmatter, body = _read_skill(skill_name)
            stem = skill_name.replace(".md", "")
 
            if editor == "cursor":
                target_dir = repo_root / CURSOR_RULES_DIR
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"repolect-{stem}.mdc"
                target.write_text(_to_cursor_mdc(frontmatter, body), encoding="utf-8")
                installed.append(str(target.relative_to(repo_root)))
 
            elif editor == "claude":
                target_dir = repo_root / CLAUDE_SKILLS_DIR
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{stem}.md"
                target.write_text(_to_claude_skill(frontmatter, body), encoding="utf-8")
                installed.append(str(target.relative_to(repo_root)))
 
    logger.info("Installed %d static skills for editors: %s", len(installed), ", ".join(editors))
    return installed
 
 
def install_generated_skill(
    repo_root: str | Path,
    name: str,
    frontmatter: dict[str, str],
    body: str,
) -> list[str]:
    """Install a single generated community skill file.
 
    Returns list of installed file paths.
    """
    repo_root = Path(repo_root)
    editors = _detect_editors(repo_root)
    installed: list[str] = []
 
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
 
    for editor in editors:
        if editor == "cursor":
            target_dir = repo_root / CURSOR_RULES_DIR
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"repolect-area-{slug}.mdc"
            target.write_text(_to_cursor_mdc(frontmatter, body), encoding="utf-8")
            installed.append(str(target.relative_to(repo_root)))
 
        elif editor == "claude":
            target_dir = repo_root / CLAUDE_SKILLS_DIR / "generated"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{slug}.md"
            target.write_text(_to_claude_skill(frontmatter, body), encoding="utf-8")
            installed.append(str(target.relative_to(repo_root)))
 
    return installed
 
 
def clean_generated_skills(repo_root: str | Path) -> int:
    """Remove previously generated community skills (before regenerating)."""
    repo_root = Path(repo_root)
    removed = 0
 
    cursor_dir = repo_root / CURSOR_RULES_DIR
    if cursor_dir.exists():
        for f in cursor_dir.glob("repolect-area-*.mdc"):
            f.unlink()
            removed += 1
 
    claude_gen_dir = repo_root / CLAUDE_SKILLS_DIR / "generated"
    if claude_gen_dir.exists():
        shutil.rmtree(claude_gen_dir)
        removed += 1
 
    return removed
 
 
def generate_community_skills(
    repo_root: str | Path,
    graph_db: Any,
    node_map: dict[str, Any],
    provider: Any | None = None,
    min_symbols: int = 3,
    max_communities: int = 20,
) -> list[str]:
    """Generate per-community skill files from Louvain community detection.
 
    Uses LLM (if available) to synthesize descriptions from node summaries.
    Falls back to structural-only output if no provider.
 
    Returns list of installed file paths.
    """
    repo_root = Path(repo_root)
 
    try:
        communities = graph_db.detect_communities()
    except (NotImplementedError, Exception) as e:
        logger.warning("Community detection failed, skipping skill generation: %s", e)
        return []
 
    if not communities:
        return []
 
    groups: dict[int, list[str]] = defaultdict(list)
    for nid, comm_id in communities.items():
        groups[comm_id].append(nid)
 
    clean_generated_skills(repo_root)
    installed: list[str] = []
 
    sorted_communities = sorted(groups.items(), key=lambda x: -len(x[1]))
    significant = [(cid, members) for cid, members in sorted_communities if len(members) >= min_symbols]
    significant = significant[:max_communities]
 
    used_labels: set[str] = set()
    comm_labels: dict[int, str] = {}
    comm_nodes: dict[int, list[Any]] = {}
    for comm_id, member_ids in significant:
        nodes = [node_map[nid] for nid in member_ids if nid in node_map]
        if not nodes:
            continue
        label = _derive_community_label(nodes, used_labels)
        comm_labels[comm_id] = label
        comm_nodes[comm_id] = nodes
 
    for comm_id in list(comm_labels.keys()):
        nodes = comm_nodes[comm_id]
        label = comm_labels[comm_id]
        member_ids = groups[comm_id]
        files = _get_community_files(nodes)
        entry_points = _get_entry_points(nodes)
        cross_community = _get_cross_community_edges(graph_db, member_ids, communities, comm_labels)
 
        description = _synthesize_description(nodes, label, provider)
 
        test_files = [n for n in nodes if _is_test_node(n)]
        non_test_nodes = [n for n in nodes if not _is_test_node(n)]
 
        body = _render_community_skill(
            label=label,
            node_count=len(non_test_nodes),
            file_count=len(files),
            description=description,
            files=files,
            entry_points=entry_points,
            cross_community=cross_community,
            test_files=test_files,
        )
 
        fm = {
            "name": f"repolect-area-{label}",
            "description": f"Use when working with code in the {label} area. {len(non_test_nodes)} symbols across {len(files)} files.",
            "globs": "",
            "alwaysApply": "false",
        }
 
        paths = install_generated_skill(repo_root, label, fm, body)
        installed.extend(paths)
 
    logger.info("Generated %d community skills", len(installed))
    return installed
 
 
def _derive_community_label(nodes: list[Any], used_labels: set[str] | None = None) -> str:
    """Derive a unique human-readable label from the dominant file/module."""
    if used_labels is None:
        used_labels = set()
 
    file_nodes = [n for n in nodes if n.kind == "file" and n.title]
    if file_nodes:
        name_counts: dict[str, int] = defaultdict(int)
        for fn in file_nodes:
            stem = fn.title.replace(".py", "").replace(".ts", "").replace(".js", "")
            stem = stem.replace(".tsx", "").replace(".jsx", "").replace(".go", "")
            name_counts[stem] += 1
        best_file = max(name_counts, key=name_counts.get)
        if best_file not in used_labels:
            used_labels.add(best_file)
            return best_file
 
    class_nodes = [n for n in nodes if n.kind == "class" and n.title]
    if class_nodes:
        label = class_nodes[0].title.lower().replace(" ", "-")
        if label not in used_labels:
            used_labels.add(label)
            return label
 
    path_counts: dict[str, int] = defaultdict(int)
    for n in nodes:
        if n.path:
            parts = Path(n.path).parts
            for p in parts:
                if p not in (".", "..", "src", "lib", "tests", "test"):
                    path_counts[p] += 1
 
    if path_counts:
        candidates = sorted(path_counts.items(), key=lambda x: -x[1])
        for name, _ in candidates:
            stem = name.replace(".py", "").replace(".ts", "").replace(".js", "")
            if stem not in used_labels:
                used_labels.add(stem)
                return stem
 
    fallback = f"area-{len(used_labels) + 1}"
    used_labels.add(fallback)
    return fallback
 
 
def _get_community_files(nodes: list[Any]) -> list[dict[str, Any]]:
    """Get unique files in the community with their symbol counts."""
    file_symbols: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        if n.path and n.kind in ("function", "method", "class"):
            file_symbols[n.path].append(n.title)
 
    return [
        {"path": path, "symbols": symbols}
        for path, symbols in sorted(file_symbols.items())
    ]
 
 
def _get_entry_points(nodes: list[Any]) -> list[dict[str, str]]:
    """Identify entry points: exported symbols, high usage_count, specific kinds."""
    candidates = []
    for n in nodes:
        if n.kind in ("function", "method", "class") and not _is_test_node(n):
            score = getattr(n, "usage_count", 0) or 0
            if n.kind == "class":
                score += 5
            if n.kind == "function":
                score += 2
            candidates.append((score, n))
 
    candidates.sort(key=lambda x: -x[0])
    return [
        {"name": n.title, "kind": n.kind, "path": n.path or "", "line": n.line_start}
        for _, n in candidates[:8]
    ]
 
 
def _get_cross_community_edges(
    graph_db: Any,
    member_ids: list[str],
    communities: dict[str, int],
    comm_labels: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Find edges from this community to other communities."""
    member_set = set(member_ids)
    cross: dict[str, int] = defaultdict(int)
 
    for nid in member_ids:
        try:
            neighbors = graph_db.get_neighbors(nid, direction="out", rel_type="CALLS")
            for nb in neighbors:
                target_id = nb.get("node_id", "")
                if target_id and target_id not in member_set and target_id in communities:
                    other_comm = communities[target_id]
                    label = (comm_labels or {}).get(other_comm, f"area-{other_comm}")
                    cross[label] += 1
        except Exception:
            continue
 
    return [
        {"area": area, "connections": count}
        for area, count in sorted(cross.items(), key=lambda x: -x[1])[:10]
    ]
 
 
def _is_test_node(n: Any) -> bool:
    """Check if a node is a test file/function."""
    if n.path and ("test_" in n.path or "/tests/" in n.path or "test." in n.path):
        return True
    if n.title and n.title.startswith("test_"):
        return True
    return False
 
 
def _synthesize_description(
    nodes: list[Any],
    label: str,
    provider: Any | None,
) -> str:
    """Generate a 2-3 sentence description of the community using LLM or fallback."""
    summaries = []
    for n in nodes:
        if n.summary and n.kind in ("file", "class", "function") and not _is_test_node(n):
            summaries.append(f"- {n.title} ({n.kind}): {n.summary[:150]}")
    summaries = summaries[:15]
 
    if not provider or not summaries:
        kinds = defaultdict(int)
        for n in nodes:
            if not _is_test_node(n):
                kinds[n.kind] += 1
        def _plural(k: str, c: int) -> str:
            if k == "class":
                return f"{c} classes" if c != 1 else "1 class"
            return f"{c} {k}s" if c != 1 else f"1 {k}"
        kind_str = ", ".join(_plural(k, count) for k, count in sorted(kinds.items(), key=lambda x: -x[1]))
        return f"Contains {kind_str}."
 
    summaries_text = "\n".join(summaries)
    try:
        description = provider.complete(
            f"You are describing a functional area of a codebase called '{label}'. "
            f"Given these symbol summaries, write 2-3 sentences explaining what this area does "
            f"and its role in the system. Be specific and technical.\n\n{summaries_text}",
            max_tokens=200,
        )
        return description.strip()
    except Exception:
        return f"Functional area containing code related to {label}."
 
 
def _render_community_skill(
    label: str,
    node_count: int,
    file_count: int,
    description: str,
    files: list[dict],
    entry_points: list[dict],
    cross_community: list[dict],
    test_files: list[Any],
) -> str:
    """Render a community skill as markdown body."""
    lines = [
        f"# {label.replace('-', ' ').title()}",
        f"{node_count} symbols | {file_count} files",
        "",
        "## What This Area Does",
        "",
        description,
        "",
        "## When to Use",
        "",
    ]
 
    file_paths = [f["path"] for f in files[:5]]
    if file_paths:
        lines.append(f"- Working with code in {', '.join(f'`{p}`' for p in file_paths)}")
 
    ep_names = [ep["name"] for ep in entry_points[:5]]
    if ep_names:
        lines.append(f"- Understanding how {', '.join(ep_names)} work")
 
    lines.append(f"- Modifying {label.replace('-', ' ')}-related functionality")
    lines.append("")
 
    if files:
        lines.append("## Key Files")
        lines.append("")
        lines.append("| File | Symbols |")
        lines.append("|------|---------|")
        for f in files[:12]:
            syms = ", ".join(f["symbols"][:5])
            if len(f["symbols"]) > 5:
                syms += f" +{len(f['symbols']) - 5} more"
            lines.append(f"| `{f['path']}` | {syms} |")
        lines.append("")
 
    if entry_points:
        lines.append("## Entry Points")
        lines.append("")
        for ep in entry_points[:8]:
            lines.append(f"- **`{ep['name']}`** ({ep['kind']}) -- `{ep['path']}:{ep['line']}`")
        lines.append("")
 
    if cross_community:
        lines.append("## Connected Areas")
        lines.append("")
        lines.append("| Area | Connections |")
        lines.append("|------|-------------|")
        for cc in cross_community[:8]:
            lines.append(f"| {cc['area']} | {cc['connections']} calls |")
        lines.append("")
 
    if test_files:
        lines.append("## Tests")
        lines.append("")
        test_paths = sorted(set(n.path for n in test_files if n.path))[:5]
        for tp in test_paths:
            lines.append(f"- `{tp}`")
        lines.append("")
 
    lines.extend([
        "## How to Explore",
        "",
        f'- `tree_search(query="how does {label.replace("-", " ")} work?")`',
        f"- `get_node(node_id=\"...\")` on any entry point above",
        f"- `impact_analysis(node_id=\"...\")` before modifying entry points",
        "",
    ])
 
    return "\n".join(lines)
 