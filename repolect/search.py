"""
Repolect — Tree Search Engine
Hybrid retrieval: LLM tree reasoning + optional vector search.
 
Algorithm:
  1. ROOT PROBE: LLM reads repo summary + module list → picks relevant modules
  2. BRANCH DESCENT: LLM reads module contents → picks relevant files/classes
  3. LEAF RETRIEVAL: Fetch actual source for selected nodes + 1-hop neighbors
  4. VECTOR BOOST (optional): If embeddings are available, vector-search results
     are merged with tree results — nodes found by both methods get a score boost.
"""
 
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Callable
from .models import CodeNode, SearchResult
from .summarizer import BaseLLM
 
logger = logging.getLogger(__name__)
 
 
class TreeSearcher:
    """
    Hybrid searcher: LLM tree reasoning + optional vector similarity.
    Uses a graph database for neighbor expansion and vector search when available.
    """
 
    def __init__(
        self,
        root: CodeNode,
        repo_root: str | Path,
        provider: BaseLLM,
        graph_db=None,
        embedder=None,
    ):
        self.root = root
        self.repo_root = Path(repo_root).resolve()
        self.provider = provider
        self.graph_db = graph_db
        self.embedder = embedder
        self._node_map = root.get_node_map()
 
    def search(
        self,
        query: str,
        max_results: int = 5,
        verbose: bool = False,
    ) -> list[SearchResult]:
        """Main search entry point. Returns ranked SearchResults."""
        if verbose:
            print(f"  🔍 Searching: {query}")
 
        # Step 0 (optional): Vector pre-search
        vector_hits: dict[str, float] = {}
        if self.embedder and self.graph_db and self.graph_db.has_embeddings():
            vector_hits = self._vector_search(query, max_results, verbose)
 
        # Step 1: Root probe
        relevant_modules = self._probe_root(query, verbose)
 
        if not relevant_modules:
            if verbose:
                print("  ⚠️  No relevant modules found")
            if not vector_hits:
                return []
            # Fall through to use vector-only results
            return self._build_vector_only_results(vector_hits, max_results, verbose)
 
        # Step 2: Branch descent
        candidate_nodes = self._descend_branches(query, relevant_modules, verbose)
 
        if not candidate_nodes:
            if verbose:
                print("  ⚠️  No relevant nodes found in modules")
            if not vector_hits:
                return []
            return self._build_vector_only_results(vector_hits, max_results, verbose)
 
        # Step 3: Leaf retrieval + neighbor expansion + vector boost
        results = self._retrieve_leaves(query, candidate_nodes, max_results, verbose, vector_hits)
 
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)[:max_results]
 
    def _vector_search(
        self, query: str, max_results: int, verbose: bool
    ) -> dict[str, float]:
        """Embed the query and run cosine similarity search against the graph."""
        try:
            vecs = self.embedder.embed_documents([query])
            if not vecs or not vecs[0]:
                return {}
            hits = self.graph_db.vector_search(vecs[0], top_k=max_results * 2)
            if verbose and hits:
                top_labels = ", ".join(
                    f"{nid}({sim:.2f})" for nid, sim in hits[:5]
                )
                print(f"  🧲 Vector search found {len(hits)} candidates: {top_labels}")
            return dict(hits)
        except Exception as e:
            logger.debug("Vector search failed: %s", e)
            return {}
 
    def _build_vector_only_results(
        self, vector_hits: dict[str, float], max_results: int, verbose: bool
    ) -> list[SearchResult]:
        """Build SearchResults from vector hits only (fallback when tree search finds nothing)."""
        results = []
        sorted_hits = sorted(vector_hits.items(), key=lambda x: x[1], reverse=True)
        for node_id, sim in sorted_hits[:max_results]:
            if node_id not in self._node_map:
                continue
            node = self._node_map[node_id]
            source = self._read_source(node)
            related = self._get_related_nodes(node)
            score = sim * 10.0
            results.append(SearchResult(
                node=node,
                relevance_score=score,
                reasoning=f"Found by vector search (similarity {sim:.2f})",
                source_snippet=source,
                related_nodes=related,
            ))
        if verbose and results:
            print(f"  🧲 Using {len(results)} vector-only results (tree search found nothing)")
        return results
 
    def _prioritize_children(self, children: list[CodeNode], limit: int) -> list[CodeNode]:
        """Prioritize children by usage_count so high-traffic nodes aren't truncated."""
        if len(children) <= limit:
            return children
        scored = []
        for c in children:
            child_usage = sum(n.usage_count for n in c.flat_iter())
            scored.append((child_usage, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]
 
    def _probe_root(self, query: str, verbose: bool) -> list[CodeNode]:
        """Step 1: Ask LLM which top-level modules are relevant."""
        if not self.root.children:
            return []
 
        candidates = self._prioritize_children(self.root.children, 30)
        module_list = "\n".join(
            f'  {{"title": "{c.title}", "node_id": "{c.node_id}", "summary": "{c.summary[:200]}"}}'
            for c in candidates
        )
 
        prompt = f"""You are navigating a code repository to answer this question:
"{query}"
 
Repository overview: {self.root.summary}
 
Top-level modules:
[
{module_list}
]
 
Which modules are most likely to contain the answer?
Return ONLY a JSON array of node_ids, e.g.: ["0001", "0003"]
Return at most 4 modules. If none seem relevant, return [].
Return ONLY the JSON array, no other text."""
 
        response = self.provider.complete(prompt, max_tokens=self.provider.max_reasoning_tokens)
 
        if verbose:
            print(f"  📂 Root probe selected: {response[:100]}")
 
        node_ids = _parse_json_list(response)
        return [self._node_map[nid] for nid in node_ids if nid in self._node_map]
 
    def _descend_branches(
        self, query: str, modules: list[CodeNode], verbose: bool
    ) -> list[tuple[CodeNode, float]]:
        """Step 2: For each relevant module, pick relevant files/components."""
        candidates = []
 
        for module in modules:
            if not module.children:
                continue
 
            top_children = self._prioritize_children(module.children, 35)
            children_list = "\n".join(
                f'  {{"node_id": "{c.node_id}", "title": "{c.title}", "kind": "{c.kind}", '
                f'"summary": "{c.summary[:180]}"}}'
                for c in top_children
            )
 
            prompt = f"""Finding code to answer: "{query}"
 
Module `{module.title}`: {module.summary}
 
Components in this module:
[
{children_list}
]
 
Which components directly help answer the question?
Return ONLY a JSON array like:
[{{"node_id": "0001.002", "score": 8.5}}, {{"node_id": "0001.003", "score": 6.0}}]
Scores are 1-10. Return at most 5. Return [] if none are relevant.
Return ONLY the JSON array, no other text."""
 
            response = self.provider.complete(prompt, max_tokens=self.provider.max_reasoning_tokens)
 
            if verbose:
                print(f"  📄 Module `{module.title}` selected: {response[:100]}")
 
            scored = _parse_scored_list(response)
            for node_id, score in scored:
                if node_id in self._node_map:
                    candidates.append((self._node_map[node_id], score))
 
        return candidates
 
    def _retrieve_leaves(
        self,
        query: str,
        candidates: list[tuple[CodeNode, float]],
        max_results: int,
        verbose: bool,
        vector_hits: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """Step 3: Fetch source for candidates + expand to neighbors.
 
        Nodes that also appear in vector_hits get a score boost.
        Vector-only nodes (not in tree candidates) are appended at the end.
        """
        results = []
        seen_ids: set[str] = set()
        vector_hits = vector_hits or {}
 
        top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:max_results * 2]
 
        for node, score in top_candidates:
            seen_ids.add(node.node_id)
            source = self._read_source(node)
            related = self._get_related_nodes(node)
 
            if node.node_id in vector_hits:
                boost = vector_hits[node.node_id] * 2.0
                score = min(score + boost, 10.0)
                reasoning = f"Tree search ({score:.1f}) + vector boost ({boost:.1f})"
            else:
                reasoning = f"Selected by tree search with score {score:.1f}"
 
            results.append(SearchResult(
                node=node,
                relevance_score=score,
                reasoning=reasoning,
                source_snippet=source,
                related_nodes=related,
            ))
 
        # Append vector-only results that tree search missed
        for node_id, sim in sorted(vector_hits.items(), key=lambda x: x[1], reverse=True):
            if node_id in seen_ids or node_id not in self._node_map:
                continue
            if len(results) >= max_results * 2:
                break
            node = self._node_map[node_id]
            source = self._read_source(node)
            related = self._get_related_nodes(node)
            score = sim * 10.0 * 0.8  # slight discount vs tree-confirmed results
            results.append(SearchResult(
                node=node,
                relevance_score=score,
                reasoning=f"Found by vector search (similarity {sim:.2f})",
                source_snippet=source,
                related_nodes=related,
            ))
 
        return results
 
    def _get_related_nodes(self, node: CodeNode) -> list[CodeNode]:
        """Get 1-hop neighbors — uses graph_db when available, falls back to in-memory relations."""
        related = []
        seen_ids: set[str] = set()
 
        # Primary: graph database neighbors
        if self.graph_db:
            try:
                neighbors = self.graph_db.get_neighbors(node.node_id, direction="both")
                for nb in neighbors[:8]:
                    nid = nb.get("node_id", "")
                    if nid and nid in self._node_map and nid not in seen_ids:
                        seen_ids.add(nid)
                        related.append(self._node_map[nid])
            except Exception as e:
                logger.debug("Graph neighbor lookup failed for %s: %s", node.node_id, e)
 
        # Fallback / supplement: in-memory relations
        if len(related) < 5:
            for relation in node.relations[:5]:
                target_id = relation.target_id
                if (not target_id.startswith("external:")
                        and target_id in self._node_map
                        and target_id not in seen_ids):
                    seen_ids.add(target_id)
                    related.append(self._node_map[target_id])
 
        return related[:8]
 
    def _read_source(self, node: CodeNode) -> str:
        """Read the source lines for a node."""
        if not node.path:
            return ""
        file_path = self.repo_root / node.path
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            s = max(0, node.line_start - 1)
            e = min(len(lines), node.line_end)
            snippet = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines[s:e], start=s))
            # Cap at ~100 lines to keep prompts manageable
            snippet_lines = snippet.splitlines()
            if len(snippet_lines) > 100:
                snippet = "\n".join(snippet_lines[:100]) + f"\n... ({len(snippet_lines) - 100} more lines)"
            return snippet
        except (IOError, OSError):
            return ""
 
    def explain_node(self, node_id: str, query: str | None = None) -> str:
        """
        NOVEL FEATURE: "Why is this here?"
        Explain a node's role by walking the tree upward + graph lookups.
        Includes a source snippet to ground the LLM and reduce hallucination.
        """
        if node_id not in self._node_map:
            return f"Node {node_id} not found."
 
        node = self._node_map[node_id]
 
        ancestors = self._get_ancestors(node_id)
        ancestor_context = " → ".join(a.title for a in ancestors) + f" → {node.title}"
 
        used_by: list[CodeNode] = []
        calls_out: list[CodeNode] = []
        if self.graph_db:
            try:
                in_neighbors = self.graph_db.get_neighbors(node_id, direction="in")
                for nb in in_neighbors[:5]:
                    nid = nb.get("node_id", "")
                    if nid and nid in self._node_map:
                        used_by.append(self._node_map[nid])
            except Exception:
                pass
            try:
                out_neighbors = self.graph_db.get_neighbors(
                    node_id, direction="out", rel_type="CALLS",
                )
                for nb in out_neighbors[:5]:
                    nid = nb.get("node_id", "")
                    if nid and nid in self._node_map:
                        calls_out.append(self._node_map[nid])
            except Exception:
                pass
 
        if not used_by:
            used_by = [
                n for n in self._node_map.values()
                if any(r.target_id == node_id for r in n.relations)
            ][:5]
        if not calls_out:
            calls_out = [
                self._node_map[r.target_id]
                for r in node.relations
                if r.kind == "CALLS" and r.target_id in self._node_map
            ][:5]
 
        used_by_text = ""
        if used_by:
            used_by_text = f"\nCalled by: {', '.join(f'{n.title} ({n.kind})' for n in used_by)}"
        calls_text = ""
        if calls_out:
            calls_text = f"\nCalls: {', '.join(f'{n.title} ({n.kind})' for n in calls_out)}"
 
        source_snippet = self._read_source(node)
        source_lines = source_snippet.splitlines()[:30]
        source_text = "\n".join(source_lines) if source_lines else "(source not available)"
 
        prompt = f"""Explain the role of `{node.title}` in the codebase.
 
Path in system: {ancestor_context}
Summary: {node.summary}
{used_by_text}{calls_text}
{"Relations: " + ", ".join(f"{r.kind} {r.target_id}" for r in node.relations[:5]) if node.relations else ""}
 
Source (first 30 lines):
{source_text}
 
{"Question context: " + query if query else ""}
 
Explain in 2-3 sentences: what this component does, why it exists, and how it fits into the larger system. Be specific about the role — mention concrete callers/callees if available."""
 
        return self.provider.complete(prompt, max_tokens=self.provider.max_reasoning_tokens)
 
    def _get_ancestors(self, target_id: str) -> list[CodeNode]:
        """Walk tree to find ancestor chain."""
        def walk(node: CodeNode, target: str, path: list[CodeNode]) -> list[CodeNode] | None:
            if node.node_id == target:
                return path
            for child in node.children:
                result = walk(child, target, path + [node])
                if result is not None:
                    return result
            return None
 
        result = walk(self.root, target_id, [])
        return result or []
 
    def trace_flow(self, entry_point_query: str, max_depth: int = 5) -> str:
        """Trace execution flow from a starting function.
 
        Accepts a node_id (e.g. '0002.013.002') for direct lookup or a
        natural-language query for semantic search.  Follows CALLS edges via
        graph and in-memory relations.  Returns structured graph-derived
        output using pre-indexed summaries rather than LLM narration.
        """
        entry = None
        if re.match(r"^\d+(\.\d+)*$", entry_point_query) and entry_point_query in self._node_map:
            entry = self._node_map[entry_point_query]
        else:
            q_lower = entry_point_query.strip().lower()
            title_matches = [
                n for n in self._node_map.values()
                if n.kind in ("function", "method", "class")
                and n.title.lower() == q_lower
            ]
            if not title_matches:
                title_matches = [
                    n for n in self._node_map.values()
                    if n.kind in ("function", "method", "class")
                    and q_lower in n.title.lower()
                ]
            if title_matches:
                entry = title_matches[0]
            else:
                results = self.search(entry_point_query, max_results=3)
                if results:
                    for r in results:
                        if r.node.kind in ("function", "method"):
                            entry = r.node
                            break
                    if not entry:
                        entry = results[0].node
        if not entry:
            return "Could not find an entry point matching that query."
 
        start_nodes = [entry]
        if entry.kind == "file":
            fn_children = [c for c in entry.children if c.kind in ("function", "method")]
            if fn_children:
                _ENTRY_HINTS = {"main", "run", "start", "cli", "app", "serve", "execute", "handle"}
                preferred = [
                    c for c in fn_children
                    if any(h in c.title.lower() for h in _ENTRY_HINTS)
                    or not c.title.startswith("_")
                ]
                has_callers = []
                for c in (preferred or fn_children):
                    caller_count = sum(
                        1 for n in self._node_map.values()
                        if any(r.target_id == c.node_id and r.kind == "CALLS" for r in n.relations)
                    )
                    has_callers.append((c, caller_count))
                has_callers.sort(key=lambda x: -x[1])
                start_nodes = [c for c, _ in has_callers[:3]]
 
        visited: set[str] = set()
        trees: list[dict] = []
 
        def follow_calls(node: CodeNode, depth: int = 0) -> dict:
            if depth > max_depth or node.node_id in visited:
                return {"node": node, "depth": depth, "children": [], "truncated": node.node_id in visited}
            visited.add(node.node_id)
 
            call_targets: set[str] = set()
            if self.graph_db:
                try:
                    neighbors = self.graph_db.get_neighbors(
                        node.node_id, direction="out", rel_type="CALLS",
                    )
                    for nb in neighbors:
                        nid = nb.get("node_id", "")
                        if nid:
                            call_targets.add(nid)
                except Exception:
                    pass
 
            for rel in node.relations:
                if rel.kind == "CALLS" and rel.target_id in self._node_map:
                    call_targets.add(rel.target_id)
 
            subtrees = []
            for target_id in sorted(call_targets):
                if target_id in self._node_map:
                    subtrees.append(follow_calls(self._node_map[target_id], depth + 1))
 
            return {"node": node, "depth": depth, "children": subtrees, "truncated": False}
 
        for start in start_nodes:
            trees.append(follow_calls(start))
 
        if not trees or all(t["node"].node_id in visited and not t["children"] for t in trees):
            if not trees:
                return f"No execution flow found from `{entry.title}`."
 
        header = entry.title if len(start_nodes) == 1 else f"{entry.title} ({len(start_nodes)} entry points)"
        lines = [f"Execution flow from `{header}` ({entry.path}):\n"]
        node_count = 0
 
        def render_tree(tree: dict, prefix: str = "") -> None:
            nonlocal node_count
            if node_count >= 25:
                return
            node_count += 1
            n = tree["node"]
            trunc_mark = " (↻ cycle)" if tree["truncated"] else ""
            lines.append(
                f"{prefix}├─ **{n.title}** "
                f"({n.path}:{n.line_start}){trunc_mark}"
            )
            if n.summary and not tree["truncated"]:
                lines.append(f"{prefix}│  {n.summary[:120]}")
            child_prefix = prefix + "│  "
            for child in tree["children"]:
                if node_count >= 25:
                    lines.append(f"{child_prefix}└─ ... (truncated)")
                    break
                render_tree(child, child_prefix)
 
        for tree in trees:
            render_tree(tree)
 
        total = len(visited)
        if total > 25:
            lines.append(f"\n... {total - 25} more node(s) in the call graph.")
        max_depth_seen = max((t["depth"] for t in trees), default=0) if trees else 0
        has_children = any(t["children"] for t in trees)
        lines.append(f"\nTotal: {total} unique node(s) traced.")
        if not has_children and total > 0:
            lines.append(
                "(No outgoing CALLS edges found — call graph may be sparse. "
                "Use graph_query() to check CALLS edges, or get_node() to read source.)"
            )
        return "\n".join(lines)
 
    def impact_analysis(
        self, node_id: str, max_hops: int = 3,
    ) -> list[tuple[CodeNode, int]]:
        """Find all nodes that depend on the target via reverse CALLS/IMPORTS.
 
        Returns (CodeNode, hop_distance) pairs sorted by distance.
        """
        if not self.graph_db:
            return []
        deps = self.graph_db.get_reverse_dependencies(
            node_id, max_hops=max_hops, rel_types=["CALLS", "IMPORTS"],
        )
        results = []
        for nid, hop in deps:
            if nid in self._node_map:
                results.append((self._node_map[nid], hop))
        return results
 
    def graph_query(self, cypher: str) -> list[list]:
        """Passthrough Cypher query to the graph database."""
        if not self.graph_db:
            return []
        try:
            return self.graph_db.cypher(cypher)
        except Exception as e:
            logger.error("Graph query failed: %s", e)
            return []
 
 
# ── Explainer (synthesis layer) ──────────────────────────────────────────────
 
class Explainer:
    """
    Takes search results and synthesizes a final answer with citations.
    This is the last step in the query pipeline.
    """
 
    def __init__(self, provider: BaseLLM):
        self.provider = provider
 
    def _build_prompt(self, query: str, results: list[SearchResult]) -> str:
        sections = [r.format_for_llm() for r in results[:5]]
        context = "\n\n---\n\n".join(sections)
        return f"""Answer this question about the codebase: "{query}"
 
Relevant code sections:
 
{context}
 
Provide a clear, specific answer.
- Cite sources as (filename:line_number) inline
- If the code is the answer (e.g., "where is X"), show the key snippet
- If something is unclear or not shown, say so
- Be concise — 3-5 sentences for simple questions, more for complex ones
 
Answer:"""
 
    def explain(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return "No relevant code found for that query."
        return self.provider.complete(self._build_prompt(query, results), max_tokens=self.provider.max_reasoning_tokens)
 
    def stream_explain(self, query: str, results: list[SearchResult]):
        """Stream the answer token by token."""
        if not results:
            return iter(["No relevant code found for that query."])
        return self.provider.stream_complete(self._build_prompt(query, results), max_tokens=self.provider.max_reasoning_tokens)
 
 
# ── JSON parsing helpers ─────────────────────────────────────────────────────
 
def _extract_json_array(text: str) -> str | None:
    """Find the outermost JSON array in text using bracket counting.
 
    Handles nested arrays/objects correctly, unlike non-greedy regex.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
 
 
def _parse_json_list(text: str) -> list[str]:
    """Parse a JSON array of strings from LLM output."""
    raw = _extract_json_array(text)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return [str(x) for x in parsed if x]
    except (json.JSONDecodeError, TypeError):
        pass
    return []
 
 
def _parse_scored_list(text: str) -> list[tuple[str, float]]:
    """Parse [{node_id: "...", score: N}, ...] from LLM output."""
    raw = _extract_json_array(text)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        result = []
        for item in parsed:
            if isinstance(item, dict):
                node_id = str(item.get("node_id", ""))
                score = float(item.get("score", 5.0))
                if node_id:
                    result.append((node_id, score))
        return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []
 