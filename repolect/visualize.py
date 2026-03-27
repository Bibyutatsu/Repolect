"""
Repolect — Interactive graph visualization (Streamlit + pyvis).
 
Launch via: repolect viz
"""
 
from __future__ import annotations
 
import sys
import tempfile
from pathlib import Path
 
import streamlit as st
from pyvis.network import Network
 
# ---------------------------------------------------------------------------
# Color palette for node kinds
# ---------------------------------------------------------------------------
 
KIND_COLORS = {
    "repo": "#6366f1",       # indigo
    "module": "#8b5cf6",     # violet
    "file": "#3b82f6",       # blue
    "class": "#10b981",      # emerald
    "function": "#f59e0b",   # amber
    "method": "#f97316",     # orange
    "interface": "#06b6d4",  # cyan
    "doc": "#64748b",        # slate
}
 
KIND_ICONS = {
    "repo": "📦", "module": "📁", "file": "📄", "class": "🔷",
    "function": "🔹", "method": "🔸", "interface": "🔶", "doc": "📝",
}
 
EDGE_COLORS = {
    "CONTAINS": "#94a3b8",   # light slate
    "CALLS": "#ef4444",      # red
    "IMPORTS": "#a855f7",    # purple
    "EXTENDS": "#14b8a6",    # teal
    "IMPLEMENTS": "#06b6d4", # cyan
    "DEFINES": "#f59e0b",    # amber
    "DOCS": "#64748b",       # slate
}
 
# ---------------------------------------------------------------------------
# Data loading (cached so it survives Streamlit reruns)
# ---------------------------------------------------------------------------
 
@st.cache_resource
def load_graph(index_dir: str, backend: str | None = None):
    """Open the GraphDB and pull all nodes/edges into plain lists."""
    from repolect.graph_db import GraphDB
    graph = GraphDB.open(index_dir, backend=backend)
    nodes = graph.get_all_nodes()
    edges = graph.get_all_edges()
    actual_backend = graph.backend_name
    n_count = graph.node_count()
    e_count = graph.edge_count()
    graph.close()
    return nodes, edges, actual_backend, n_count, e_count
 
 
@st.cache_resource
def load_meta(repo_root: str, branch: str):
    """Load TreeMeta for the sidebar stats."""
    from repolect.storage import load_meta as _load_meta
    return _load_meta(repo_root, branch=branch)
 
 
# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
 
def build_pyvis(
    nodes: list[dict],
    edges: list[dict],
    *,
    selected_kinds: set[str],
    search_query: str,
    focus_node: str | None,
    hop_depth: int,
    height: str = "700px",
) -> Network:
    """Build a pyvis Network from node/edge lists with filtering applied."""
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="#e2e8f0",
        select_menu=False,
        filter_menu=False,
    )
 
    net.barnes_hut(
        gravity=-4000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
        damping=0.09,
    )
 
    node_map = {n["node_id"]: n for n in nodes}
 
    # Determine which nodes to show
    if focus_node and focus_node in node_map:
        visible_ids = _get_neighborhood(focus_node, edges, node_map, hop_depth)
    else:
        visible_ids = set(node_map.keys())
 
    search_lower = search_query.strip().lower()
 
    kept_ids: set[str] = set()
    for nid in visible_ids:
        n = node_map[nid]
        kind = n.get("kind", "")
        if kind not in selected_kinds:
            continue
        if search_lower:
            title = (n.get("title") or "").lower()
            fpath = (n.get("file_path") or "").lower()
            if search_lower not in title and search_lower not in fpath:
                continue
        kept_ids.add(nid)
 
    for nid in kept_ids:
        n = node_map[nid]
        kind = n.get("kind", "")
        color = KIND_COLORS.get(kind, "#94a3b8")
        icon = KIND_ICONS.get(kind, "•")
        label = n.get("title", nid)
        fpath = n.get("file_path", "")
        summary = n.get("summary", "")
 
        size = {"repo": 40, "module": 28, "file": 22, "class": 20,
                "function": 14, "method": 12, "interface": 18, "doc": 10}.get(kind, 14)
 
        tooltip = f"{icon} {label}\nKind: {kind}"
        if fpath:
            tooltip += f"\nPath: {fpath}"
        lines = n.get("line_start"), n.get("line_end")
        if lines[0]:
            tooltip += f"\nLines: {lines[0]}–{lines[1]}"
        if summary:
            tooltip += f"\n\n{summary[:300]}"
 
        net.add_node(
            nid,
            label=label,
            title=tooltip,
            color=color,
            size=size,
            shape="dot",
            borderWidth=2,
            borderWidthSelected=4,
        )
 
    for e in edges:
        src, dst = e.get("src", ""), e.get("dst", "")
        if src not in kept_ids or dst not in kept_ids:
            continue
        rel = e.get("rel_type", "CONTAINS")
        color = EDGE_COLORS.get(rel, "#475569")
        width = 1 if rel == "CONTAINS" else 2.5
        net.add_edge(
            src, dst,
            title=rel,
            color=color,
            width=width,
            arrows="to",
            smooth={"type": "curvedCW", "roundness": 0.1} if rel != "CONTAINS" else True,
        )
 
    return net
 
 
def _get_neighborhood(
    center: str,
    edges: list[dict],
    node_map: dict[str, dict],
    depth: int,
) -> set[str]:
    """BFS from center node up to `depth` hops (follows edges in both directions)."""
    visited: set[str] = {center}
    frontier: set[str] = {center}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for e in edges:
            src, dst = e.get("src", ""), e.get("dst", "")
            if src in frontier and dst in node_map and dst not in visited:
                next_frontier.add(dst)
            if dst in frontier and src in node_map and src not in visited:
                next_frontier.add(src)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break
    return visited
 
 
# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
 
def _parse_cli_args() -> tuple[str, str]:
    """Extract --repo-root and --branch from sys.argv (passed after Streamlit's --)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--branch", default="")
    args, _ = parser.parse_known_args()
    return args.repo_root, args.branch
 
 
def main() -> None:
    st.set_page_config(
        page_title="Repolect — Graph Explorer",
        page_icon="🧠",
        layout="wide",
    )
 
    if "repo_root" not in st.session_state:
        root, br = _parse_cli_args()
        st.session_state["repo_root"] = root
        st.session_state["branch"] = br
 
    repo_root = st.session_state.get("repo_root")
    branch = st.session_state.get("branch", "")
 
    if not repo_root:
        st.error("No repository configured. Launch via `repolect viz`.")
        st.stop()
 
    from repolect.storage import get_index_dir
    index_dir = str(get_index_dir(repo_root, branch=branch or None))
    meta = load_meta(repo_root, branch or None)
 
    preferred_backend = meta.graph_backend if meta else None
    nodes, edges, backend, n_count, e_count = load_graph(index_dir, backend=preferred_backend)
 
    if not nodes:
        st.warning("Graph is empty. Run `repolect analyze` first.")
        st.stop()
 
    # -- Sidebar controls --
    with st.sidebar:
        st.title("🧠 Repolect")
        st.caption("Graph Explorer")
 
        if meta:
            st.markdown(f"**Repo:** {meta.repo_name}")
            if meta.git_branch:
                st.markdown(f"**Branch:** `{meta.git_branch}`")
            st.markdown(f"**Backend:** {backend}")
            st.markdown(f"**Nodes:** {n_count:,}  |  **Edges:** {e_count:,}")
            if meta.language_stats:
                lang_str = ", ".join(
                    f"{lang}: {count}" for lang, count in
                    sorted(meta.language_stats.items(), key=lambda x: -x[1])[:6]
                )
                st.markdown(f"**Languages:** {lang_str}")
            st.divider()
 
        st.subheader("Filters")
 
        all_kinds = sorted({n.get("kind", "") for n in nodes})
        selected_kinds = set()
        for kind in all_kinds:
            icon = KIND_ICONS.get(kind, "•")
            if st.checkbox(f"{icon} {kind}", value=True, key=f"kind_{kind}"):
                selected_kinds.add(kind)
 
        st.divider()
 
        search_query = st.text_input("🔍 Search nodes", placeholder="function name, file path...")
 
        st.divider()
        st.subheader("Focus mode")
 
        node_titles = {n["node_id"]: f"{KIND_ICONS.get(n.get('kind',''),'•')} {n.get('title', n['node_id'])}" for n in nodes}
        focus_options = ["(show all)"] + [
            f"{nid} — {node_titles[nid]}" for nid in sorted(node_titles.keys())
        ]
        focus_selection = st.selectbox("Center on node", focus_options, index=0)
        focus_node: str | None = None
        if focus_selection != "(show all)":
            focus_node = focus_selection.split(" — ")[0]
 
        hop_depth = st.slider("Hop depth", min_value=1, max_value=5, value=2, disabled=focus_node is None)
 
    # -- Main content --
    net = build_pyvis(
        nodes, edges,
        selected_kinds=selected_kinds,
        search_query=search_query,
        focus_node=focus_node,
        hop_depth=hop_depth,
        height="750px",
    )
 
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html_path = f.name
 
    html_content = Path(html_path).read_text()
    st.components.v1.html(html_content, height=780, scrolling=False)
 
    # -- Legend --
    cols = st.columns(len(KIND_COLORS))
    for i, (kind, color) in enumerate(KIND_COLORS.items()):
        icon = KIND_ICONS.get(kind, "•")
        cols[i].markdown(
            f'<span style="color:{color}; font-size:1.1em;">●</span> {icon} {kind}',
            unsafe_allow_html=True,
        )
 
    edge_legend = " &nbsp;|&nbsp; ".join(
        f'<span style="color:{c};">━</span> {r}'
        for r, c in EDGE_COLORS.items()
        if any(e.get("rel_type") == r for e in edges)
    )
    if edge_legend:
        st.markdown(f"**Edges:** {edge_legend}", unsafe_allow_html=True)
 
 
if __name__ == "__main__":
    main()
 