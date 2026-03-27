"""
Repolect — Parser Layer
 
Two-layer parsing strategy:
  Layer 1: tree-sitter (required, handles all languages with grammar support)
  Layer 2: Regex enhancer (catches symbols tree-sitter misses: lambdas, etc.)
 
tree-sitter-languages is a required dependency. If it is not installed,
parse_file() raises RuntimeError immediately.
 
Language-agnostic: the same CodeNode structure is produced for all languages.
"""
 
from __future__ import annotations
 
import logging
import re
from pathlib import Path
 
from .git_utils import get_file_hash
from .models import CodeNode, Relation
 
logger = logging.getLogger(__name__)
 
# ── Language detection ───────────────────────────────────────────────────────
 
EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
    ".md": "markdown",
    ".rst": "rst",
}
 
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
 
 
def detect_language(file_path: str | Path) -> str | None:
    suffix = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(suffix)
 
 
def is_doc_file(file_path: str | Path) -> bool:
    suffix = Path(file_path).suffix.lower()
    name = Path(file_path).name.upper()
    return suffix in DOC_EXTENSIONS or name in {
        "README", "CHANGELOG", "LICENSE", "CONTRIBUTING", "AUTHORS",
    }
 
 
# ── Main entry point ─────────────────────────────────────────────────────────
 
def parse_file(
    file_path: str | Path,
    repo_root: str | Path,
    node_id_prefix: str,
) -> list[CodeNode]:
    """Parse a source file and return a flat list of CodeNode stubs.
 
    Summaries are empty -- they get filled by the Summarizer later.
 
    Strategy:
      1. tree-sitter (required) — extracts functions, classes, methods, imports
      2. Regex enhancer — adds lambda assignments and any symbols tree-sitter
         cannot express (runs on every file, merges without duplicates)
 
    Returns an empty list if the file cannot be parsed or has no detectable language.
    """
    file_path = Path(file_path)
    language = detect_language(file_path)
 
    if not language:
        return []
 
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (IOError, OSError):
        return []
 
    git_hash = get_file_hash(file_path)
    rel_path = str(file_path.relative_to(Path(repo_root)))
 
    nodes = _parse_with_treesitter(source, language, rel_path, git_hash, node_id_prefix)
 
    extra = _enhance_with_regex(source, language, rel_path, git_hash, node_id_prefix, nodes)
    if extra:
        nodes.extend(extra)
 
    _extract_call_relations(source, language, nodes)
 
    return nodes
 
 
# ── Tree-sitter parsing ──────────────────────────────────────────────────────
 
def _parse_with_treesitter(
    source: str, language: str, rel_path: str, git_hash: str, prefix: str,
) -> list[CodeNode]:
    """Tree-sitter based parsing — required, most accurate."""
    import warnings
    try:
        from tree_sitter_languages import get_language, get_parser
    except ImportError:
        raise RuntimeError(
            "tree-sitter-languages is required but not installed. "
            "Run: pip install 'tree-sitter>=0.21,<0.22' 'tree-sitter-languages>=1.10.2'"
        )
 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
        lang_obj = get_language(language)
        parser = get_parser(language)
    tree = parser.parse(bytes(source, "utf8"))
 
    lines = source.splitlines()
    counter = [0]
 
    def make_id() -> str:
        counter[0] += 1
        return f"{prefix}.{counter[0]:03d}"
 
    queries = _get_queries(language)
    raw_captures: list[tuple[str, str, int, int, str]] = []
 
    for query_str in queries:
        try:
            query = lang_obj.query(query_str)
            captures = query.captures(tree.root_node)
        except Exception:
            continue
 
        for node, capture_name in captures:
            name = _extract_name(node, source, language, capture_name)
            if not name:
                continue
 
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            signature = _extract_signature(node, lines, language)
 
            kind = _classify_capture(node, capture_name, language)
            if not kind:
                continue
 
            raw_captures.append((name, kind, start_line, end_line, signature))
 
    seen: set[tuple[str, int]] = set()
    nodes: list[CodeNode] = []
 
    for name, kind, start_line, end_line, signature in raw_captures:
        key = (name, start_line)
        if key in seen:
            continue
        seen.add(key)
 
        nodes.append(CodeNode(
            node_id=make_id(),
            title=name,
            kind=kind,
            path=rel_path,
            line_start=start_line,
            line_end=end_line,
            signature=signature,
            language=language,
            git_hash=git_hash,
        ))
 
    return nodes
 
 
def _get_queries(language: str) -> list[str]:
    """Return tree-sitter queries for symbol extraction per language.
 
    Captures bind to the **full definition node** (not the name identifier)
    so that start_point/end_point span the entire body.  The name is
    extracted separately in _extract_name via child_by_field_name.
    """
    queries = {
        "python": [
            "(function_definition) @function",
            "(class_definition) @class",
            "(import_statement) @import",
            "(import_from_statement) @import_from",
        ],
        "javascript": [
            "(function_declaration) @function",
            "(class_declaration) @class",
            "(method_definition) @method",
            "(arrow_function) @function",
            "(import_statement) @import",
        ],
        "typescript": [
            "(function_declaration) @function",
            "(class_declaration) @class",
            "(method_definition) @method",
            "(interface_declaration) @interface",
            "(import_statement) @import",
        ],
        "java": [
            "(method_declaration) @method",
            "(class_declaration) @class",
            "(interface_declaration) @interface",
            "(import_declaration) @import",
        ],
        "go": [
            "(function_declaration) @function",
            "(method_declaration) @method",
            "(type_declaration) @class",
        ],
        "rust": [
            "(function_item) @function",
            "(impl_item) @class",
            "(struct_item) @class",
            "(trait_item) @interface",
        ],
    }
    return queries.get(language, [
        "(function_definition) @function",
        "(class_definition) @class",
    ])
 
 
def _classify_capture(node, capture_name: str, language: str) -> str | None:
    """Determine the CodeNode kind for a capture, using parent context for Python.
 
    For Python, a function_definition inside a class_definition body is
    classified as 'method' rather than 'function'.  This avoids the old
    duplicate-capture bug where decorated defs created both a @function
    and a @method entry.
    """
    if capture_name in ("import", "import_from"):
        return None
 
    if capture_name in ("class", "interface"):
        return capture_name
 
    if capture_name == "method":
        return "method"
 
    if capture_name == "function":
        if language == "python":
            if _is_inside_class(node):
                return "method"
        return "function"
 
    return None
 
 
def _is_inside_class(node) -> bool:
    """Walk up the tree-sitter AST to check if a node lives inside a class body.
 
    The captured node is now the full function_definition (not the identifier).
    Walk from the parent upward looking for a class_definition ancestor.
    """
    current = node.parent
    while current is not None:
        if current.type == "class_definition":
            return True
        if current.type == "function_definition":
            return False
        current = current.parent
    return False
 
 
def _extract_name(node, source: str, language: str, capture_name: str) -> str:
    """Extract the symbol name from a captured definition node.
 
    The captured node is the full definition (e.g. function_definition),
    so we look up the ``name`` child field to get the identifier text.
    Falls back to scanning the node text for common patterns.
    """
    if capture_name not in ("function", "method", "class", "interface"):
        return ""
 
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf8") if hasattr(name_node, "text") else ""
 
    # Fallback for nodes without a ``name`` field (e.g. arrow_function, impl_item)
    text = node.text.decode("utf8") if hasattr(node, "text") else ""
    if not text:
        return ""
 
    if capture_name == "function":
        m = re.match(r"(?:async\s+)?(?:function\s+)?(\w+)\s*[=(]", text)
        if m:
            return m.group(1)
    if capture_name == "class":
        m = re.match(r"(?:impl|struct|class|type)\s+(\w+)", text)
        if m:
            return m.group(1)
 
    return ""
 
 
def _extract_signature(node, lines: list[str], language: str) -> str:
    """Extract the first meaningful line of a definition as its signature."""
    start = node.start_point[0]
    if start >= len(lines):
        return ""
    sig = lines[start].strip()
    if not sig and start + 1 < len(lines):
        sig = lines[start + 1].strip()
    return sig[:200] if len(sig) > 200 else sig
 
 
def extract_file_imports(
    file_path: str | Path,
    repo_root: str | Path,
    file_node: CodeNode,
) -> None:
    """Extract IMPORTS relations from a source file and attach them to the file node.
 
    Supports Python, JavaScript, TypeScript, Java, Go, and Rust.
    Uses regex-based extraction so tree-sitter re-parsing is not needed.
    """
    file_path = Path(file_path)
    language = detect_language(file_path)
    if not language:
        return
 
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (IOError, OSError):
        return
 
    patterns = _get_import_patterns(language)
    seen_modules: set[str] = set()
 
    for pattern, group_idx in patterns:
        for match in re.finditer(pattern, source, re.MULTILINE):
            module = match.group(group_idx).strip()
            if not module or module in seen_modules:
                continue
            seen_modules.add(module)
            file_node.relations.append(Relation(
                source_id=file_node.node_id,
                target_id=f"external:{module}",
                kind="IMPORTS",
                label=match.group(0).strip()[:80],
            ))
 
 
def _get_import_patterns(language: str) -> list[tuple[str, int]]:
    """Return (regex_pattern, group_index) pairs for extracting imported module names."""
    if language == "python":
        return [
            (r"^\s*from\s+([\w.]+)\s+import", 1),
            (r"^\s*import\s+([\w.]+)", 1),
        ]
    if language in ("javascript", "typescript"):
        return [
            (r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]""", 1),
            (r"""import\s+['"]([^'"]+)['"]""", 1),
            (r"""require\(\s*['"]([^'"]+)['"]\s*\)""", 1),
        ]
    if language == "java":
        return [
            (r"^\s*import\s+([\w.]+)", 1),
        ]
    if language == "go":
        return [
            (r"""^\s*import\s+"([^"]+)""", 1),
            (r"""^\s+"([^"]+)""", 1),
        ]
    if language == "rust":
        return [
            (r"^\s*use\s+([\w:]+)", 1),
        ]
    return []
 
 
# ── Regex enhancer ───────────────────────────────────────────────────────────
 
def _enhance_with_regex(
    source: str,
    language: str,
    rel_path: str,
    git_hash: str,
    prefix: str,
    existing_nodes: list[CodeNode],
) -> list[CodeNode]:
    """Find symbols that tree-sitter cannot express (e.g. lambda assignments).
 
    Runs AFTER tree-sitter on every file. Only adds nodes that don't overlap
    with existing tree-sitter nodes (checked by name + line proximity).
    """
    existing_index: set[tuple[str, int]] = set()
    for n in existing_nodes:
        existing_index.add((n.title, n.line_start))
 
    lines = source.splitlines()
    extra: list[CodeNode] = []
    max_existing_id = max(
        (int(n.node_id.rsplit(".", 1)[-1]) for n in existing_nodes),
        default=0,
    )
    counter = [max_existing_id + 500]
 
    def make_id() -> str:
        counter[0] += 1
        return f"{prefix}.{counter[0]:03d}"
 
    def _already_captured(name: str, line: int) -> bool:
        for delta in range(-2, 3):
            if (name, line + delta) in existing_index:
                return True
        return False
 
    patterns = _get_regex_enhancer_patterns(language)
 
    for i, line_text in enumerate(lines):
        for pattern, kind in patterns:
            m = re.match(pattern, line_text)
            if not m:
                continue
            name = m.group(m.lastindex)
            if not name or len(name) < 2:
                continue
            line_num = i + 1
            if _already_captured(name, line_num):
                continue
 
            end_line = _find_block_end(lines, i, language)
            extra.append(CodeNode(
                node_id=make_id(),
                title=name,
                kind=kind,
                path=rel_path,
                line_start=line_num,
                line_end=end_line,
                signature=line_text.strip()[:200],
                language=language,
                git_hash=git_hash,
            ))
            existing_index.add((name, line_num))
            break
 
    return extra
 
 
def _get_regex_enhancer_patterns(language: str) -> list[tuple[str, str]]:
    """Patterns for symbols that tree-sitter cannot capture."""
    if language == "python":
        return [
            # Lambda assignments: my_fn = lambda x: x+1
            (r"^\s*(\w+)\s*=\s*lambda\b", "function"),
        ]
    return []
 
 
# ── Call relation extraction ────────────────────────────────────────────────
 
def _extract_call_relations(source: str, language: str, nodes: list[CodeNode]) -> None:
    """Extract CALLS relations by detecting function calls within node bodies.
 
    Handles homonyms (multiple functions with the same name in one file,
    e.g. across different classes) by tracking each candidate's parent
    class and line position, then disambiguating via ``self.`` context
    and proximity.
    """
    if not nodes:
        return
 
    callable_kinds = {"function", "method", "class"}
    symbol_entries: dict[str, list[tuple[str, str | None, int]]] = {}
    for n in nodes:
        if n.kind not in callable_kinds:
            continue
        parent_class = _find_enclosing_class(n, nodes)
        symbol_entries.setdefault(n.title, []).append(
            (n.node_id, parent_class, n.line_start)
        )
 
    if not symbol_entries:
        return
 
    lines = source.splitlines()
 
    for node in nodes:
        if node.kind not in ("function", "method"):
            continue
        if node.line_start <= 0 or node.line_end <= 0:
            continue
 
        body_lines = lines[node.line_start - 1 : node.line_end]
        body = "\n".join(body_lines)
        caller_class = _find_enclosing_class(node, nodes)
 
        for symbol_name, candidates in symbol_entries.items():
            pattern = rf"(?<!\w){re.escape(symbol_name)}\s*\("
            if not re.search(pattern, body):
                continue
 
            target_id = _pick_best_candidate(
                candidates, node.node_id, caller_class, node.line_start,
                is_self_call=bool(re.search(rf"self\.{re.escape(symbol_name)}\s*\(", body)),
            )
            if target_id and target_id != node.node_id:
                node.relations.append(Relation(
                    source_id=node.node_id,
                    target_id=target_id,
                    kind="CALLS",
                    label=f"{node.title} calls {symbol_name}",
                ))
 
 
def _find_enclosing_class(node: CodeNode, all_nodes: list[CodeNode]) -> str | None:
    """Return the title of the class that encloses *node*, or None."""
    for n in all_nodes:
        if n.kind == "class" and n.line_start <= node.line_start and n.line_end >= node.line_end:
            return n.title
    return None
 
 
def _pick_best_candidate(
    candidates: list[tuple[str, str | None, int]],
    caller_id: str,
    caller_class: str | None,
    caller_line: int,
    is_self_call: bool,
) -> str | None:
    """Pick the best target from homonym candidates.
 
    Preference order:
      1. Skip self (caller_id)
      2. If ``self.foo()`` — prefer the candidate in the same class
      3. Otherwise prefer the candidate closest in the file
    """
    viable = [(nid, cls, ln) for nid, cls, ln in candidates if nid != caller_id]
    if not viable:
        return None
    if len(viable) == 1:
        return viable[0][0]
    if is_self_call and caller_class:
        same_class = [v for v in viable if v[1] == caller_class]
        if same_class:
            return same_class[0][0]
    viable.sort(key=lambda v: abs(v[2] - caller_line))
    return viable[0][0]
 
 
# ── Block-end heuristic ──────────────────────────────────────────────────────
 
def _find_block_end(lines: list[str], start: int, language: str) -> int:
    if language == "python":
        start_indent = len(lines[start]) - len(lines[start].lstrip())
        for i in range(start + 1, min(start + 200, len(lines))):
            if lines[i].strip() == "":
                continue
            indent = len(lines[i]) - len(lines[i].lstrip())
            if indent <= start_indent and lines[i].strip():
                return i
    else:
        depth = 0
        for i in range(start, min(start + 500, len(lines))):
            depth += lines[i].count("{") - lines[i].count("}")
            if depth < 0 or (i > start and depth == 0):
                return i + 1
    return min(start + 50, len(lines))
 
 
# ── Doc file parsing ─────────────────────────────────────────────────────────
 
def parse_doc_file(file_path: str | Path, repo_root: str | Path, node_id: str) -> CodeNode | None:
    file_path = Path(file_path)
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (IOError, OSError):
        return None
 
    if not content.strip():
        return None
 
    rel_path = str(file_path.relative_to(Path(repo_root)))
    lines = content.splitlines()
 
    return CodeNode(
        node_id=node_id,
        title=file_path.name,
        kind="doc",
        path=rel_path,
        line_start=1,
        line_end=len(lines),
        language="markdown" if file_path.suffix == ".md" else "text",
        git_hash=get_file_hash(file_path),
    )
 