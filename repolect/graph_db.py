"""
Graph database adapter for Repolect.
 
Provides a unified interface over two backends:
  - NetworkX (always available, default) — uses grand-cypher for Cypher queries
  - FalkorDBLite (optional, Cypher-native, requires `pip install falkordblite`)
 
Usage:
    graph = GraphDB.open(index_dir)          # auto-selects best backend
    graph = GraphDB.open(index_dir, "networkx")  # force backend
    graph.add_node("n1", title="MyClass", kind="class", file_path="src/main.py")
    graph.add_edge("n1", "n2", "CALLS", label="calls helper")
    neighbors = graph.get_neighbors("n1", direction="out")
    graph.save()
    graph.close()
"""
 
from __future__ import annotations
 
import json
import logging
import pickle
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
 
import networkx as nx
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------
 
class _Backend(ABC):
    """Interface that each graph backend must implement.
 
    All backends provide the same public API so that GraphDB can delegate
    transparently. New backends should subclass this and implement every
    abstract method.
    """
 
    # -- Node ops --
 
    @abstractmethod
    def add_node(self, node_id: str, **props: Any) -> None: ...
 
    @abstractmethod
    def get_node(self, node_id: str) -> dict | None: ...
 
    @abstractmethod
    def node_exists(self, node_id: str) -> bool: ...
 
    @abstractmethod
    def delete_node(self, node_id: str) -> None: ...
 
    # -- Edge ops --
 
    @abstractmethod
    def add_edge(self, src: str, dst: str, rel_type: str, **props: Any) -> None: ...
 
    @abstractmethod
    def get_neighbors(
        self, node_id: str, direction: str = "both", rel_type: str | None = None
    ) -> list[dict]: ...
 
    @abstractmethod
    def get_edges(self, node_id: str, direction: str = "both") -> list[dict]: ...
 
    @abstractmethod
    def delete_edges_for_node(self, node_id: str) -> None: ...
 
    # -- Bulk ops --
 
    @abstractmethod
    def bulk_add_nodes(self, nodes: list[dict]) -> None:
        """Add multiple nodes. Each dict must have 'node_id' key."""
        ...
 
    @abstractmethod
    def bulk_add_edges(self, edges: list[dict]) -> None:
        """Add multiple edges. Each dict must have 'src', 'dst', 'rel_type'."""
        ...
 
    # -- Query ops --
 
    @abstractmethod
    def find_shortest_path(self, src: str, dst: str) -> list[str]: ...
 
    @abstractmethod
    def get_k_hop_neighbors(self, node_id: str, k: int = 2) -> list[str]: ...
 
    @abstractmethod
    def get_most_connected(self, top_n: int = 10) -> list[tuple[str, int]]: ...
 
    @abstractmethod
    def cypher(self, query: str, params: dict | None = None) -> list[list]: ...
 
    @abstractmethod
    def get_subgraph(self, node_ids: list[str]) -> Any:
        """Return a subgraph induced by the given node IDs."""
        ...
 
    @abstractmethod
    def detect_communities(self) -> dict[str, int]:
        """Run community detection and return {node_id: community_id}."""
        ...
 
    @abstractmethod
    def get_reverse_dependencies(
        self, node_id: str, max_hops: int = 3, rel_types: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        """Return nodes that depend on the given node (reverse traversal).
 
        Returns (node_id, hop_distance) pairs sorted by distance.
        ``rel_types`` filters to specific relation kinds (e.g. ["CALLS", "IMPORTS"]).
        """
        ...
 
    # -- Bulk retrieval --
 
    @abstractmethod
    def get_all_nodes(self) -> list[dict]:
        """Return every node as a dict with at least ``node_id`` plus stored properties."""
        ...
 
    @abstractmethod
    def get_all_edges(self) -> list[dict]:
        """Return every edge as ``{src, dst, rel_type, ...extra props}``."""
        ...
 
    # -- Vector ops --
 
    @abstractmethod
    def has_embeddings(self) -> bool:
        """Return True if at least one node has a stored embedding."""
        ...
 
    @abstractmethod
    def vector_search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Return (node_id, similarity) pairs sorted by descending cosine similarity."""
        ...
 
    # -- Lifecycle --
 
    @abstractmethod
    def node_count(self) -> int: ...
 
    @abstractmethod
    def edge_count(self) -> int: ...
 
    @abstractmethod
    def clear(self) -> None: ...
 
    @abstractmethod
    def save(self) -> None: ...
 
    @abstractmethod
    def close(self) -> None: ...
 
    @abstractmethod
    def export_json(self, path: Path | None = None) -> None:
        """Export graph to a human-readable JSON file. No-op if unsupported."""
        ...
 
    @property
    @abstractmethod
    def backend_name(self) -> str: ...
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _gc_to_rows(gc_result: dict) -> list[list]:
    """Convert grand-cypher columnar {col: [vals]} to row-based [[val, val], ...]."""
    if not gc_result:
        return []
    columns = list(gc_result.keys())
    if not columns:
        return []
    n_rows = len(gc_result[columns[0]])
    if n_rows == 0:
        return []
    return [[gc_result[col][i] for col in columns] for i in range(n_rows)]
 
 
# ---------------------------------------------------------------------------
# NetworkX backend (always available)
# ---------------------------------------------------------------------------
 
class _NetworkXBackend(_Backend):
    """Persistent MultiDiGraph with grand-cypher for Cypher queries.
 
    Nodes are stored with ``__labels__={"CodeNode"}`` and edges with
    ``__labels__={rel_type}`` so that grand-cypher's ``:Label`` syntax works.
    The ``rel_type`` attribute is also kept for native NetworkX filtering.
    """
 
    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._pkl_path = index_dir / "graph.pkl"
        self._json_path = index_dir / "graph.json"
        self._lock = threading.Lock()
 
        if self._pkl_path.exists():
            with open(self._pkl_path, "rb") as f:
                self._G: nx.MultiDiGraph = pickle.load(f)
            logger.info("Loaded NetworkX graph: %d nodes, %d edges",
                        self._G.number_of_nodes(), self._G.number_of_edges())
        else:
            self._G = nx.MultiDiGraph()
 
    # -- Node ops --
 
    def add_node(self, node_id: str, **props: Any) -> None:
        with self._lock:
            self._G.add_node(node_id, __labels__={"CodeNode"}, **props)
 
    def get_node(self, node_id: str) -> dict | None:
        if node_id not in self._G:
            return None
        data = dict(self._G.nodes[node_id])
        data.pop("__labels__", None)
        data["node_id"] = node_id
        return data
 
    def node_exists(self, node_id: str) -> bool:
        return node_id in self._G
 
    def delete_node(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._G:
                self._G.remove_node(node_id)
 
    # -- Edge ops --
 
    def add_edge(self, src: str, dst: str, rel_type: str, **props: Any) -> None:
        with self._lock:
            self._G.add_edge(src, dst, __labels__={rel_type}, rel_type=rel_type, **props)
 
    def get_neighbors(
        self, node_id: str, direction: str = "both", rel_type: str | None = None
    ) -> list[dict]:
        if node_id not in self._G:
            return []
        results = []
 
        if direction in ("out", "both"):
            for _, dst, data in self._G.out_edges(node_id, data=True):
                if rel_type and data.get("rel_type") != rel_type:
                    continue
                entry = dict(self._G.nodes.get(dst, {}))
                entry.pop("__labels__", None)
                entry["node_id"] = dst
                entry["_rel_type"] = data.get("rel_type", "")
                entry["_direction"] = "out"
                results.append(entry)
 
        if direction in ("in", "both"):
            for src_node, _, data in self._G.in_edges(node_id, data=True):
                if rel_type and data.get("rel_type") != rel_type:
                    continue
                entry = dict(self._G.nodes.get(src_node, {}))
                entry.pop("__labels__", None)
                entry["node_id"] = src_node
                entry["_rel_type"] = data.get("rel_type", "")
                entry["_direction"] = "in"
                results.append(entry)
 
        return results
 
    def get_edges(self, node_id: str, direction: str = "both") -> list[dict]:
        if node_id not in self._G:
            return []
        edges = []
 
        if direction in ("out", "both"):
            for src, dst, data in self._G.out_edges(node_id, data=True):
                d = {k: v for k, v in data.items() if k != "__labels__"}
                edges.append({"src": src, "dst": dst, **d})
 
        if direction in ("in", "both"):
            for src, dst, data in self._G.in_edges(node_id, data=True):
                d = {k: v for k, v in data.items() if k != "__labels__"}
                edges.append({"src": src, "dst": dst, **d})
 
        return edges
 
    def delete_edges_for_node(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self._G:
                return
            out_edges = list(self._G.out_edges(node_id, keys=True))
            in_edges = list(self._G.in_edges(node_id, keys=True))
            self._G.remove_edges_from(out_edges)
            self._G.remove_edges_from(in_edges)
 
    # -- Bulk ops (single lock acquisition) --
 
    def bulk_add_nodes(self, nodes: list[dict]) -> None:
        with self._lock:
            for node in nodes:
                node = dict(node)
                nid = node.pop("node_id")
                self._G.add_node(nid, __labels__={"CodeNode"}, **node)
 
    def bulk_add_edges(self, edges: list[dict]) -> None:
        with self._lock:
            for edge in edges:
                edge = dict(edge)
                src = edge.pop("src")
                dst = edge.pop("dst")
                rt = edge.pop("rel_type")
                self._G.add_edge(src, dst, __labels__={rt}, rel_type=rt, **edge)
 
    # -- Query ops --
 
    def find_shortest_path(self, src: str, dst: str) -> list[str]:
        try:
            return nx.shortest_path(self._G, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
 
    def get_k_hop_neighbors(self, node_id: str, k: int = 2) -> list[str]:
        if node_id not in self._G:
            return []
        visited: set[str] = set()
        frontier = {node_id}
        for _ in range(k):
            next_frontier: set[str] = set()
            for n in frontier:
                for neighbor in set(self._G.successors(n)) | set(self._G.predecessors(n)):
                    if neighbor not in visited and neighbor != node_id:
                        next_frontier.add(neighbor)
            visited.update(next_frontier)
            frontier = next_frontier
        return list(visited)
 
    def get_most_connected(self, top_n: int = 10) -> list[tuple[str, int]]:
        degrees = sorted(self._G.degree(), key=lambda x: x[1], reverse=True)
        return degrees[:top_n]
 
    def cypher(self, query: str, params: dict | None = None) -> list[list]:
        """Run a Cypher query on the NetworkX graph.
 
        Uses grand-cypher for pattern matching (MATCH/WHERE/RETURN) and falls
        back to native NetworkX for COUNT aggregations, which grand-cypher does
        not handle correctly on MultiDiGraph.
 
        Natively handled (fast path):
          - MATCH (n...) RETURN count(n)
          - MATCH ()-[r]->() RETURN count(r)
          - MATCH ()-[r:TYPE]->() RETURN count(r)
 
        Delegated to grand-cypher (everything else):
          - MATCH (n:CodeNode {kind: 'function'}) RETURN n.title, n.file_path
          - MATCH (a)-[:CALLS]->(b) RETURN a.title, b.title
          - MATCH ... WHERE ... ORDER BY ... LIMIT ...
        """
        q = query.strip()
        q_lower = q.lower()
 
        # --- Native fast paths for COUNT (broken in grand-cypher on MultiDiGraph) ---
 
        if "count(n)" in q_lower and re.search(r"match\s*\(n", q_lower):
            return [[self._G.number_of_nodes()]]
 
        if "count(r)" in q_lower and "-[r" in q_lower:
            m = re.search(r"-\[r:(\w+)\]->", q)
            if m:
                rt = m.group(1)
                count = sum(
                    1 for _, _, d in self._G.edges(data=True)
                    if d.get("rel_type") == rt
                )
                return [[count]]
            return [[self._G.number_of_edges()]]
 
        # --- Delegate to grand-cypher for everything else ---
 
        try:
            from grandcypher import GrandCypher
            result = GrandCypher(self._G).run(q)
            return _gc_to_rows(result)
        except Exception as e:
            logger.warning("Cypher query failed on NetworkX backend: %s — %s", q, e)
            return []
 
    def get_subgraph(self, node_ids: list[str]) -> nx.MultiDiGraph:
        """Return an induced subgraph for the given node IDs."""
        valid_ids = [nid for nid in node_ids if nid in self._G]
        return self._G.subgraph(valid_ids).copy()
 
    def detect_communities(self) -> dict[str, int]:
        if self._G.number_of_nodes() < 2:
            return {n: 0 for n in self._G.nodes()}
        simple = nx.Graph(self._G.to_undirected())
        communities = nx.community.louvain_communities(simple, seed=42)
        mapping: dict[str, int] = {}
        for comm_id, members in enumerate(communities):
            for node_id in members:
                mapping[node_id] = comm_id
        return mapping
 
    def get_reverse_dependencies(
        self, node_id: str, max_hops: int = 3, rel_types: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        if node_id not in self._G:
            return []
        results: dict[str, int] = {}
        frontier = {node_id}
        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for n in frontier:
                for pred in self._G.predecessors(n):
                    if pred == node_id or pred in results:
                        continue
                    if rel_types:
                        edges = self._G.get_edge_data(pred, n)
                        if not edges:
                            continue
                        if not any(d.get("rel_type") in rel_types for d in edges.values()):
                            continue
                    results[pred] = hop
                    next_frontier.add(pred)
            frontier = next_frontier
            if not frontier:
                break
        return sorted(results.items(), key=lambda x: x[1])
 
    # -- Bulk retrieval --
 
    def get_all_nodes(self) -> list[dict]:
        nodes = []
        for nid, data in self._G.nodes(data=True):
            entry = {k: v for k, v in data.items() if k != "__labels__" and k != "embedding"}
            entry["node_id"] = nid
            nodes.append(entry)
        return nodes
 
    def get_all_edges(self) -> list[dict]:
        edges = []
        for src, dst, data in self._G.edges(data=True):
            entry = {k: v for k, v in data.items() if k != "__labels__"}
            entry["src"] = src
            entry["dst"] = dst
            edges.append(entry)
        return edges
 
    # -- Vector ops --
 
    def has_embeddings(self) -> bool:
        return any("embedding" in data for _, data in self._G.nodes(data=True))
 
    def vector_search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        import math
 
        q = query_embedding
        q_norm = math.sqrt(sum(x * x for x in q))
        if q_norm == 0:
            return []
 
        scores: list[tuple[str, float]] = []
        for nid, data in self._G.nodes(data=True):
            vec = data.get("embedding")
            if vec is None:
                continue
            dot = sum(a * b for a, b in zip(q, vec))
            v_norm = math.sqrt(sum(x * x for x in vec))
            if v_norm == 0:
                continue
            scores.append((nid, dot / (q_norm * v_norm)))
 
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
 
    # -- Lifecycle --
 
    def node_count(self) -> int:
        return self._G.number_of_nodes()
 
    def edge_count(self) -> int:
        return self._G.number_of_edges()
 
    def clear(self) -> None:
        with self._lock:
            self._G.clear()
 
    def save(self) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self._pkl_path, "wb") as f:
                pickle.dump(self._G, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    def export_json(self, path: Path | None = None) -> None:
        target = path or self._json_path
        try:
            data = nx.node_link_data(self._G)
            with open(target, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass
 
    def close(self) -> None:
        self.save()
        self.export_json()
 
    @property
    def backend_name(self) -> str:
        return "networkx"
 
 
# ---------------------------------------------------------------------------
# FalkorDBLite backend (optional)
# ---------------------------------------------------------------------------
 
class _FalkorDBBackend(_Backend):
    """FalkorDBLite backend using embedded Redis + FalkorDB graph engine."""
 
    def __init__(self, index_dir: Path) -> None:
        from redislite.falkordb_client import FalkorDB  # type: ignore[import-untyped]
 
        self._index_dir = index_dir
        self._db_path = str(index_dir / "graph.db")
        self._db = FalkorDB(self._db_path)
        self._graph = self._db.select_graph("repolect")
        self._vector_index_created = False
 
        try:
            self._graph.query(
                "CREATE INDEX IF NOT EXISTS FOR (n:CodeNode) ON (n.node_id)"
            )
        except Exception:
            pass
 
    def _q(self, query: str, params: dict | None = None) -> Any:
        if params:
            return self._graph.query(query, params)
        return self._graph.query(query)
 
    def _ensure_vector_index(self, dim: int) -> None:
        """Create the HNSW vector index on first embedding insertion."""
        if self._vector_index_created:
            return
        try:
            self._graph.query(
                f"CREATE VECTOR INDEX FOR (n:CodeNode) ON (n.embedding) "
                f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
            )
        except Exception:
            pass  # index may already exist
        self._vector_index_created = True
 
    # -- Node ops --

    def add_node(self, node_id: str, **props: Any) -> None:
        params: dict[str, Any] = {"node_id": node_id}
        set_parts: list[str] = []
        for k, v in props.items():
            if v is None:
                continue
            if k == "embedding":
                self._ensure_vector_index(len(v))
                vec_str = ", ".join(str(x) for x in v)
                set_parts.append(f"n.embedding = vecf32([{vec_str}])")
                continue
            param_key = f"p_{k}"
            set_parts.append(f"n.{k} = ${param_key}")
            params[param_key] = v

        query = "MERGE (n:CodeNode {node_id: $node_id})"
        if set_parts:
            query += " SET " + ", ".join(set_parts)
        self._q(query, params)

    def get_node(self, node_id: str) -> dict | None:
        result = self._q(
            "MATCH (n:CodeNode {node_id: $nid}) RETURN n",
            {"nid": node_id},
        )
        if result.result_set:
            node = result.result_set[0][0]
            return dict(node.properties) if hasattr(node, "properties") else {"node_id": node_id}
        return None

    def node_exists(self, node_id: str) -> bool:
        result = self._q(
            "MATCH (n:CodeNode {node_id: $nid}) RETURN count(n)",
            {"nid": node_id},
        )
        return result.result_set[0][0] > 0

    def delete_node(self, node_id: str) -> None:
        self._q("MATCH (n:CodeNode {node_id: $nid}) DETACH DELETE n", {"nid": node_id})

    # -- Edge ops --

    def add_edge(self, src: str, dst: str, rel_type: str, **props: Any) -> None:
        prop_parts = []
        params: dict[str, Any] = {"src": src, "dst": dst}
        for k, v in props.items():
            if v is None:
                continue
            param_key = f"p_{k}"
            prop_parts.append(f"{k}: ${param_key}")
            params[param_key] = v
        prop_str = "{" + ", ".join(prop_parts) + "}" if prop_parts else ""
        safe_type = rel_type.upper().replace(" ", "_").replace("-", "_")
        self._q(
            f"MATCH (a:CodeNode {{node_id: $src}}), (b:CodeNode {{node_id: $dst}}) "
            f"CREATE (a)-[:{safe_type}{prop_str}]->(b)",
            params,
        )

    def get_neighbors(
        self, node_id: str, direction: str = "both", rel_type: str | None = None
    ) -> list[dict]:
        results = []
        rel_filter = f":{rel_type}" if rel_type else ""

        if direction in ("out", "both"):
            q = f"MATCH (n:CodeNode {{node_id: $nid}})-[r{rel_filter}]->(m:CodeNode) RETURN m, type(r)"
            r = self._q(q, {"nid": node_id})
            for row in r.result_set:
                entry = dict(row[0].properties) if hasattr(row[0], "properties") else {}
                entry["_rel_type"] = row[1]
                entry["_direction"] = "out"
                results.append(entry)

        if direction in ("in", "both"):
            q = f"MATCH (n:CodeNode {{node_id: $nid}})<-[r{rel_filter}]-(m:CodeNode) RETURN m, type(r)"
            r = self._q(q, {"nid": node_id})
            for row in r.result_set:
                entry = dict(row[0].properties) if hasattr(row[0], "properties") else {}
                entry["_rel_type"] = row[1]
                entry["_direction"] = "in"
                results.append(entry)

        return results

    def get_edges(self, node_id: str, direction: str = "both") -> list[dict]:
        edges = []

        if direction in ("out", "both"):
            r = self._q(
                "MATCH (a:CodeNode {node_id: $nid})-[r]->(b:CodeNode) "
                "RETURN a.node_id, type(r), b.node_id, properties(r)",
                {"nid": node_id},
            )
            for row in r.result_set:
                edges.append({"src": row[0], "rel_type": row[1], "dst": row[2], **row[3]})

        if direction in ("in", "both"):
            r = self._q(
                "MATCH (a:CodeNode)-[r]->(b:CodeNode {node_id: $nid}) "
                "RETURN a.node_id, type(r), b.node_id, properties(r)",
                {"nid": node_id},
            )
            for row in r.result_set:
                edges.append({"src": row[0], "rel_type": row[1], "dst": row[2], **row[3]})

        return edges

    def delete_edges_for_node(self, node_id: str) -> None:
        self._q(
            "MATCH (n:CodeNode {node_id: $nid})-[r]-() DELETE r",
            {"nid": node_id},
        )

    # -- Bulk ops --

    def bulk_add_nodes(self, nodes: list[dict]) -> None:
        batch_size = 100
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            self._q(
                "UNWIND $nodes AS nd "
                "CREATE (n:CodeNode {node_id: nd.node_id, title: nd.title, kind: nd.kind, "
                "file_path: nd.file_path, language: nd.language, "
                "line_start: nd.line_start, line_end: nd.line_end, summary: nd.summary})",
                {"nodes": batch},
            )

    def bulk_add_edges(self, edges: list[dict]) -> None:
        for edge in edges:
            edge = dict(edge)
            src = edge.pop("src")
            dst = edge.pop("dst")
            rt = edge.pop("rel_type")
            self.add_edge(src, dst, rt, **edge)

    # -- Query ops --

    def find_shortest_path(self, src: str, dst: str) -> list[str]:
        try:
            r = self._q(
                "MATCH p=shortestPath((a:CodeNode {node_id: $s})-[*]-(b:CodeNode {node_id: $d})) "
                "RETURN [n IN nodes(p) | n.node_id]",
                {"s": src, "d": dst},
            )
            if r.result_set:
                return r.result_set[0][0]
        except Exception:
            pass
        return []

    def get_k_hop_neighbors(self, node_id: str, k: int = 2) -> list[str]:
        try:
            r = self._q(
                f"MATCH (n:CodeNode {{node_id: $nid}})-[*1..{k}]-(m:CodeNode) "
                "WHERE m.node_id <> $nid RETURN DISTINCT m.node_id",
                {"nid": node_id},
            )
            return [row[0] for row in r.result_set]
        except Exception:
            return []

    def get_most_connected(self, top_n: int = 10) -> list[tuple[str, int]]:
        try:
            r = self._q(
                "MATCH (n:CodeNode)-[r]-() "
                "RETURN n.node_id, count(r) AS deg ORDER BY deg DESC LIMIT $lim",
                {"lim": top_n},
            )
            return [(row[0], row[1]) for row in r.result_set]
        except Exception:
            return []

    def cypher(self, query: str, params: dict | None = None) -> list[list]:
        r = self._q(query, params)
        return r.result_set

    def get_subgraph(self, node_ids: list[str]) -> dict:
        """Return a subgraph as {nodes: [...], edges: [...]} for the given IDs."""
        if not node_ids:
            return {"nodes": [], "edges": []}
        nodes = []
        for nid in node_ids:
            n = self.get_node(nid)
            if n:
                nodes.append(n)
        edges = []
        try:
            r = self._q(
                "UNWIND $ids AS sid "
                "UNWIND $ids AS did "
                "MATCH (a:CodeNode {node_id: sid})-[r]->(b:CodeNode {node_id: did}) "
                "RETURN a.node_id, type(r), b.node_id",
                {"ids": node_ids},
            )
            for row in r.result_set:
                edges.append({"src": row[0], "rel_type": row[1], "dst": row[2]})
        except Exception:
            pass
        return {"nodes": nodes, "edges": edges}

    def detect_communities(self) -> dict[str, int]:
        count_r = self._q("MATCH (n:CodeNode) RETURN count(n)")
        node_count = count_r.result_set[0][0] if count_r.result_set else 0
        if node_count == 0:
            return {}

        r = self._q("MATCH (n:CodeNode) RETURN n.node_id")
        node_ids = [row[0] for row in r.result_set]
        if len(node_ids) < 2:
            return {nid: 0 for nid in node_ids}

        r = self._q(
            "MATCH (a:CodeNode)-[]->(b:CodeNode) RETURN a.node_id, b.node_id"
        )
        edges = [(row[0], row[1]) for row in r.result_set]

        G = nx.Graph()
        G.add_nodes_from(node_ids)
        G.add_edges_from(edges)

        communities = nx.community.louvain_communities(G, seed=42)
        mapping: dict[str, int] = {}
        for comm_id, members in enumerate(communities):
            for node_id in members:
                mapping[node_id] = comm_id
        return mapping

    def get_reverse_dependencies(
        self, node_id: str, max_hops: int = 3, rel_types: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        results: list[tuple[str, int]] = []
        rel_filter = ""
        if rel_types:
            rel_filter = ":" + "|".join(rel_types)
        for hop in range(1, max_hops + 1):
            try:
                r = self._q(
                    f"MATCH (n:CodeNode {{node_id: $nid}})<-[{rel_filter}*{hop}..{hop}]-(m:CodeNode) "
                    "WHERE m.node_id <> $nid "
                    "RETURN DISTINCT m.node_id",
                    {"nid": node_id},
                )
                for row in r.result_set:
                    mid = row[0]
                    if not any(mid == existing[0] for existing in results):
                        results.append((mid, hop))
            except Exception:
                break
        return sorted(results, key=lambda x: x[1])

    # -- Bulk retrieval --

    def get_all_nodes(self) -> list[dict]:
        r = self._q("MATCH (n:CodeNode) RETURN n")
        nodes = []
        for row in r.result_set:
            node = row[0]
            if hasattr(node, "properties"):
                entry = dict(node.properties)
                entry.pop("embedding", None)
            else:
                continue
            nodes.append(entry)
        return nodes

    def get_all_edges(self) -> list[dict]:
        r = self._q(
            "MATCH (a:CodeNode)-[r]->(b:CodeNode) "
            "RETURN a.node_id, type(r), b.node_id, properties(r)"
        )
        edges = []
        for row in r.result_set:
            entry = dict(row[3]) if row[3] else {}
            entry["src"] = row[0]
            entry["rel_type"] = row[1]
            entry["dst"] = row[2]
            edges.append(entry)
        return edges

    # -- Vector ops --

    def has_embeddings(self) -> bool:
        try:
            r = self._q(
                "MATCH (n:CodeNode) WHERE n.embedding IS NOT NULL RETURN count(n)"
            )
            return r.result_set[0][0] > 0 if r.result_set else False
        except Exception:
            return False

    def vector_search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        try:
            vec_str = ", ".join(str(x) for x in query_embedding)
            r = self._q(
                f"CALL db.idx.vector.queryNodes('CodeNode', 'embedding', {top_k}, "
                f"vecf32([{vec_str}])) YIELD node, score "
                f"RETURN node.node_id, score"
            )
            # FalkorDB returns cosine *distance* (0 = identical); convert to similarity
            return [(row[0], 1.0 - float(row[1])) for row in r.result_set]
        except Exception as e:
            logger.debug("FalkorDB vector search failed: %s", e)
            return []

    # -- Lifecycle --

    def node_count(self) -> int:
        r = self._q("MATCH (n:CodeNode) RETURN count(n)")
        return r.result_set[0][0] if r.result_set else 0

    def edge_count(self) -> int:
        r = self._q("MATCH ()-[r]->() RETURN count(r)")
        return r.result_set[0][0] if r.result_set else 0

    def clear(self) -> None:
        self._q("MATCH (n) DETACH DELETE n")

    def save(self) -> None:
        pass

    def export_json(self, path: Path | None = None) -> None:
        """Export graph to JSON. Queries all nodes and edges from FalkorDB."""
        target = path or (self._index_dir / "graph.json")
        try:
            nodes = self.get_all_nodes()
            edges = self.get_all_edges()
            data = {"nodes": nodes, "edges": edges}
            with open(target, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass

    @property
    def backend_name(self) -> str:
        return "falkordb"

# ---------------------------------------------------------------------------
# Public API — GraphDB facade
# ---------------------------------------------------------------------------

class GraphDB:
    """Unified graph database interface.

    Auto-selects the best available backend:
    1. FalkorDBLite (if installed and starts successfully)
    2. NetworkX (always available)

    All public methods delegate to the active backend.
    """

    def __init__(self, backend: _Backend) -> None:
        self._backend = backend

    @classmethod
    def open(cls, index_dir: str | Path, backend: str | None = None) -> "GraphDB":
        """Open or create a graph database in the given directory.

        Args:
            index_dir: Path to the .repolect/ index directory.
            backend: Force a specific backend ("falkordb" or "networkx").
                If None, auto-detect best available.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        if backend == "networkx":
            return cls(_NetworkXBackend(index_dir))

        if backend == "falkordb":
            return cls(_try_falkordb(index_dir))

        try:
            fb = _try_falkordb(index_dir)
            logger.info("Using FalkorDBLite backend")
            return cls(fb)
        except Exception as e:
            logger.info("FalkorDBLite not available (%s), using NetworkX backend", e)
            return cls(_NetworkXBackend(index_dir))

    # -- Delegate all public methods to backend --

    def add_node(self, node_id: str, **props: Any) -> None:
        self._backend.add_node(node_id, **props)

    def get_node(self, node_id: str) -> dict | None:
        return self._backend.get_node(node_id)

    def node_exists(self, node_id: str) -> bool:
        return self._backend.node_exists(node_id)

    def delete_node(self, node_id: str) -> None:
        self._backend.delete_node(node_id)

    def add_edge(self, src: str, dst: str, rel_type: str, **props: Any) -> None:
        self._backend.add_edge(src, dst, rel_type, **props)

    def get_neighbors(
        self, node_id: str, direction: str = "both", rel_type: str | None = None
    ) -> list[dict]:
        return self._backend.get_neighbors(node_id, direction, rel_type)

    def get_edges(self, node_id: str, direction: str = "both") -> list[dict]:
        return self._backend.get_edges(node_id, direction)

    def delete_edges_for_node(self, node_id: str) -> None:
        self._backend.delete_edges_for_node(node_id)

    def bulk_add_nodes(self, nodes: list[dict]) -> None:
        self._backend.bulk_add_nodes(nodes)

    def bulk_add_edges(self, edges: list[dict]) -> None:
        self._backend.bulk_add_edges(edges)

    def find_shortest_path(self, src: str, dst: str) -> list[str]:
        return self._backend.find_shortest_path(src, dst)

    def get_k_hop_neighbors(self, node_id: str, k: int = 2) -> list[str]:
        return self._backend.get_k_hop_neighbors(node_id, k)

    def get_most_connected(self, top_n: int = 10) -> list[tuple[str, int]]:
        return self._backend.get_most_connected(top_n)

    def cypher(self, query: str, params: dict | None = None) -> list[list]:
        return self._backend.cypher(query, params)

    def get_subgraph(self, node_ids: list[str]) -> Any:
        return self._backend.get_subgraph(node_ids)

    def detect_communities(self) -> dict[str, int]:
        return self._backend.detect_communities()

    def get_reverse_dependencies(
        self, node_id: str, max_hops: int = 3, rel_types: list[str] | None = None,
    ) -> list[tuple[str, int]]:
        return self._backend.get_reverse_dependencies(node_id, max_hops, rel_types)

    def get_all_nodes(self) -> list[dict]:
        return self._backend.get_all_nodes()

    def get_all_edges(self) -> list[dict]:
        return self._backend.get_all_edges()

    def has_embeddings(self) -> bool:
        return self._backend.has_embeddings()

    def vector_search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        return self._backend.vector_search(query_embedding, top_k)

    def node_count(self) -> int:
        return self._backend.node_count()

    def edge_count(self) -> int:
        return self._backend.edge_count()

    def clear(self) -> None:
        self._backend.clear()

    def save(self) -> None:
        self._backend.save()

    def export_json(self, path: Path | None = None) -> None:
        self._backend.export_json(path)

    def close(self) -> None:
        self._backend.close()

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    def __enter__(self) -> "GraphDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

# ---------------------------------------------------------------------------
# Helper to instantiate FalkorDB with timeout protection
# ---------------------------------------------------------------------------

def _try_falkordb(index_dir: Path) -> _FalkorDBBackend:
    """Try to create a FalkorDB backend with a startup timeout."""
    import concurrent.futures

    def _create() -> _FalkorDBBackend:
        return _FalkorDBBackend(index_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_create)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            raise RuntimeError("FalkorDBLite startup timed out (>10s)")
        except ImportError:
            raise RuntimeError("FalkorDBLite not installed")