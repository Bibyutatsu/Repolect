"""
Repolect — CLI
Primary commands: analyze, ask, why, tree, sync, graph, list, mcp
"""
 
from __future__ import annotations
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
 
import click
 
 
def _init_provider():
    """Initialize LLM provider using config.yaml → env → auto-detect."""
    from .summarizer import get_provider
    from .config import load_config
    config = load_config()
    try:
        return get_provider(config=config)
    except RuntimeError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)
 
 
def _resolve_embeddings_flag() -> bool:
    """Decide whether to run embeddings.
    
    Priority: env var → config file.
    """
    raw = os.environ.get("REPOLECT_EMBEDDINGS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # Check config: embeddings are ON if an embedding provider is configured
    from .config import load_config
    config = load_config()
    embed_provider = config.get("embedding_provider", "").strip()
    embed_model = config.get("embedding_model", "").strip()
    if embed_provider and embed_model:
        return True
    click.echo(
        "⚠ Embedding not configured in config.yaml "
        "(embedding_provider / embedding_model are empty). "
        "Running without embeddings.",
        err=True,
    )
    return False
 
 
@click.group()
@click.version_option("0.1.0")
def cli():
    """Repolect — Vectorless code intelligence for any codebase."""
    pass
 
 
# ── Primary commands ─────────────────────────────────────────────────────
 
@cli.command()
@click.option("--force", is_flag=True, help="Re-index even if nothing changed")
@click.option("--no-git", is_flag=True, help="Skip git metadata")
@click.option("--path", "target_path", default=".", help="Directory to index (default: current dir)")
@click.option("--parse-workers", default=None, type=int, help="Parallel parser worker count")
@click.option("--num-workers", default=None, type=int, help="Parallel LLM worker count (overrides provider default)")
@click.option("--graph-backend", default=None, type=click.Choice(["networkx", "falkordb"]), help="Force graph backend")
@click.option("--quiet", is_flag=True, help="Suppress progress bars")
@click.option("--no-cache", is_flag=True, help="Ignore summary cache, re-summarize everything from scratch")
@click.option("--branch", "branches", multiple=True, help="Analyze specific branch(es). Repeatable.")
@click.option("--all-branches", is_flag=True, help="Analyze all local branches")
@click.option("--skills", is_flag=True, help="Generate repo-specific community skills from detected communities")
def analyze(force, no_git, target_path, parse_workers, num_workers, graph_backend, quiet, no_cache, branches, all_branches, skills):
    """Index a repository into a semantic tree + knowledge graph."""
    from .git_utils import (
        detect_repo_root, is_git_repo, get_current_commit, get_repo_name,
        get_current_branch, get_all_branches, checkout_branch,
    )
    from .storage import migrate_legacy_index
 
    repo_root = detect_repo_root(target_path)
    has_git = is_git_repo(repo_root)
    current_branch = get_current_branch(repo_root) if has_git else ""
 
    if has_git:
        migrated = migrate_legacy_index(repo_root, current_branch)
        if migrated:
            click.echo(f"  Migrated legacy index to branch '{current_branch}'")
 
    if all_branches and has_git:
        target_branches = get_all_branches(repo_root)
        if not target_branches:
            target_branches = [current_branch]
    elif branches:
        target_branches = list(branches)
    else:
        target_branches = [current_branch] if has_git else [""]
 
    original_branch = current_branch
    try:
        for br in target_branches:
            if has_git and br and br != get_current_branch(repo_root):
                click.echo(f"\n{'='*60}")
                click.echo(f"Switching to branch: {br}")
                if not checkout_branch(repo_root, br):
                    click.echo(f"  Failed to checkout '{br}' — skipping", err=True)
                    continue
            _analyze_one(
                repo_root, br if has_git else "",
                force=force, no_git=no_git,
                parse_workers=parse_workers, num_workers=num_workers,
                graph_backend=graph_backend, quiet=quiet, no_cache=no_cache,
                generate_skills=skills,
            )
    finally:
        if has_git and len(target_branches) > 1 and get_current_branch(repo_root) != original_branch:
            checkout_branch(repo_root, original_branch)
            click.echo(f"\nRestored branch: {original_branch}")
 
 
def _analyze_one(
    repo_root: str,
    branch: str,
    *,
    force: bool,
    no_git: bool,
    parse_workers: int | None,
    num_workers: int | None,
    graph_backend: str | None,
    quiet: bool,
    no_cache: bool,
    generate_skills: bool = False,
):
    """Core analyze logic for a single branch."""
    from tqdm import tqdm
    from .git_utils import is_git_repo, get_current_commit, get_repo_name
    from .storage import (
        load_meta, save_tree, save_meta, register_repo,
        write_context_file, tree_exists, ensure_index_dir,
        LLMDiskCache,
    )
    from .tree_builder import scan_repo, build_raw_tree, build_graph, count_nodes, count_files, get_language_stats
    from .summarizer import Summarizer, summarize_tree
    from .graph_db import GraphDB
    from .models import TreeMeta
 
    has_git = is_git_repo(repo_root)
    repo_name = get_repo_name(repo_root) if has_git else Path(repo_root).name
    git_commit = get_current_commit(repo_root) if (has_git and not no_git) else ""
    embeddings_enabled = _resolve_embeddings_flag()

    if not force and tree_exists(repo_root, branch=branch):
        existing_meta = load_meta(repo_root, branch=branch)
        if existing_meta and git_commit and existing_meta.git_commit == git_commit:
            click.echo(f"✓ Index is up to date for {branch or 'repo'} (commit {git_commit[:8]}). Use --force to re-index.")
            return
 
    total_phases = 5 if not embeddings_enabled else 6
    phase = 0
 
    def _phase(label: str) -> None:
        nonlocal phase
        phase += 1
        click.echo(f"\n[{phase}/{total_phases}] {label}")
 
    branch_label = f" [{branch}]" if branch else ""
    click.echo(f"\nIndexing {repo_name}{branch_label}...")
    start_time = time.time()
 
    llm = _init_provider()
    click.echo(f"  Provider: {llm.provider_name}" + (f" ({llm.model})" if hasattr(llm, "model") else ""))
 
    ok, msg = llm.health_check()
    if not ok:
        click.echo(f"\nLLM provider is not reachable: {msg}", err=True)
        click.echo("Check that your provider (Ollama, OpenAI, etc.) is running and accessible.", err=True)
        sys.exit(1)
    click.echo("  LLM health check: OK")
 
    # ── Phase 1: Scan ──
    _phase("Scanning repository...")
    files = scan_repo(repo_root, show_progress=False)
    lang_preview: dict[str, int] = {}
    from .parser import detect_language
    for f in files:
        lang = detect_language(f) or "other"
        lang_preview[lang] = lang_preview.get(lang, 0) + 1
    lang_str = ", ".join(f"{l}: {c}" for l, c in sorted(lang_preview.items(), key=lambda x: -x[1]))
    click.echo(f"      Found {len(files)} files ({lang_str})")
 
    # ── Phase 2: Parse ──
    _phase("Parsing files...")
    if not quiet:
        with tqdm(total=len(files), desc="      Parsing", unit="file", dynamic_ncols=True) as parse_bar:
            def on_parse_progress(current, total, filename):
                parse_bar.set_postfix_str(filename[-40:])
                parse_bar.n = current
                parse_bar.update(0)
 
            root = build_raw_tree(
                repo_root, repo_name, show_progress=False,
                parse_workers=parse_workers,
                progress_callback=on_parse_progress, files=files,
            )
    else:
        root = build_raw_tree(
            repo_root, repo_name, show_progress=False,
            parse_workers=parse_workers, files=files,
        )
 
    node_count = count_nodes(root)
    file_count = count_files(root)
    lang_stats = get_language_stats(root)
    click.echo(f"      {node_count} symbols in {file_count} files")
 
    # ── Phase 3: Summarize ──
    _phase("Generating summaries...")
    cache = None
    if not no_cache:
        cache = LLMDiskCache(repo_root)
        purged = cache.purge_errors()
        if purged:
            click.echo(f"      Purged {purged} cached error(s) from previous runs")
        llm.enable_cache(cache)
        click.echo(f"      LLM cache: {len(cache)} entries in SQLite")
 
    setup_msg = llm.parallel_setup_message()
    if setup_msg:
        click.echo(f"      {setup_msg}")
 
    workers = num_workers or llm.num_workers
    click.echo(f"      Workers: {workers}")
 
    summarizer = Summarizer(llm)
 
    if not quiet:
        with tqdm(total=node_count, desc="      Summarizing", unit="node", dynamic_ncols=True) as bar:
            last = [0]
            def on_summary_progress(current, total, title, summary):
                if summary and summary.startswith("[summary unavailable:"):
                    bar.write(f"      ⚠ Error summarizing '{title}': {summary}")
                
                postfix = title[-30:] if title else ""
                if cache:
                    postfix += f" | H:{cache.hits} M:{cache.misses}"
                bar.set_postfix_str(postfix)
                bar.update(current - last[0])
                last[0] = current
            summarize_tree(root, repo_root, summarizer, progress_callback=on_summary_progress, max_workers=num_workers)
    else:
        summarize_tree(root, repo_root, summarizer, max_workers=num_workers)
 
    if cache:
        click.echo(f"      Cache: {cache.hits} hits, {cache.misses} misses")
        cache.close()
        llm.disable_cache()
 
    # ── Phase 4: Build graph ──
    _phase("Building knowledge graph...")
    index_dir = ensure_index_dir(repo_root, branch=branch)
    graph = GraphDB.open(index_dir, backend=graph_backend)
    graph.clear()
    build_graph(root, graph)
    click.echo(f"      {graph.node_count()} nodes, {graph.edge_count()} edges ({graph.backend_name})")
 
    # ── Phase 5 (optional): Embeddings ──
    embed_count = 0
    if embeddings_enabled:
        _phase("Generating embeddings...")
        from .embedder import get_embedder
        embedder = get_embedder()
        if embedder is not None:
            eok, emsg = embedder.health_check()
            if not eok:
                click.echo(f"\nEmbedding provider is not reachable: {emsg}", err=True)
                click.echo("Check that your embedding provider is running. Skipping embeddings.", err=True)
            else:
                embed_count = embedder.embed_tree(root, graph)
                click.echo(f"      Embedded {embed_count} nodes")
        else:
            click.echo("⚠ No embedding provider configured — skipping embeddings", err=True)
 
    # ── Phase N: Save ──
    _phase("Saving index...")
    duration = time.time() - start_time
 
    meta = TreeMeta(
        repo_name=repo_name,
        repo_path=str(repo_root),
        repo_id=TreeMeta.make_repo_id(repo_root),
        git_commit=git_commit,
        git_branch=branch,
        indexed_at=datetime.now(timezone.utc).isoformat(),
        node_count=node_count,
        file_count=file_count,
        language_stats=lang_stats,
        index_duration_seconds=round(duration, 1),
        embeddings_enabled=embeddings_enabled and embed_count > 0,
        graph_backend=graph.backend_name,
    )
 
    save_tree(root, repo_root, branch=branch)
    save_meta(meta, repo_root, branch=branch)
    register_repo(repo_root, meta)
    write_context_file(meta, root, repo_root, graph_db=graph)
 
    from .skill_installer import install_static_skills, generate_community_skills
    static_installed = install_static_skills(repo_root)
    if static_installed:
        click.echo(f"      Installed {len(static_installed)} agent skills")
 
    if generate_skills:
        node_map = root.get_node_map()
        gen_installed = generate_community_skills(
            repo_root, graph, node_map, provider=llm,
        )
        if gen_installed:
            click.echo(f"      Generated {len(gen_installed)} community skills")

    from .git_utils import ensure_gitignored
    ensure_gitignored(repo_root, [".claude/", ".cursor/", ".agents/", "REPOLECT.md"])

    graph.close()
 
    click.echo(f"\nDone in {duration:.0f}s -- {node_count} symbols indexed across {file_count} files")
    click.echo(f"Run 'repolect ask <question>' to query")
 
 
@cli.command()
@click.argument("query")
@click.option("--repo", default=None, help="Repo name or ID (if not in a repo)")
@click.option("-q", "--quiet", is_flag=True, help="Suppress search steps and stream; print only the final answer")
@click.option("--max-results", default=5, help="Max code sections to retrieve")
def ask(query, repo, quiet, max_results):
    """Ask a question about the codebase. Uses tree search + optional vector search."""
    from .storage import load_tree
    from .search import TreeSearcher, Explainer
 
    verbose = not quiet
 
    repo_root = _resolve_repo(repo)
    branch = _get_branch()
    root = load_tree(repo_root, branch=branch)
    llm = _init_provider()
 
    graph = _open_graph(repo_root)
 
    embedder = None
    if graph and graph.has_embeddings():
        from .embedder import get_embedder
        embedder = get_embedder()
        if embedder is None:
            click.echo(
                "⚠ Embeddings exist in graph but no embedding provider configured. "
                "Vector search disabled for this query.",
                err=True,
            )
        elif verbose:
            click.echo("  ✓ Embeddings detected — vector search enabled")
    elif verbose:
        click.echo("  ℹ No embeddings found. Configure embedding_provider/embedding_model in config.yaml and run 'repolect sync'.")
 
    searcher = TreeSearcher(root, repo_root, llm, graph_db=graph, embedder=embedder)
    results = searcher.search(query, max_results=max_results, verbose=verbose)
 
    if not results:
        click.echo("No relevant code found. Try rephrasing the question.")
        if graph:
            graph.close()
        return
 
    explainer = Explainer(llm)
 
    if verbose:
        click.echo("\n── Answer ──\n")
        for chunk in explainer.stream_explain(query, results):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        click.echo("\n")
 
        click.echo("── Sources ──")
        for r in results:
            click.echo(f"  [{r.relevance_score:.1f}] {r.node.title} ({r.node.path}:{r.node.line_start})")
    else:
        answer = explainer.explain(query, results)
        click.echo(f"\n{answer}\n")
 
    if graph:
        graph.close()
 
 
@cli.command()
@click.argument("path_or_query")
@click.option("--repo", default=None)
def why(path_or_query, repo):
    """Explain why a file or function exists in the codebase context."""
    from .storage import load_tree
    from .search import TreeSearcher
 
    repo_root = _resolve_repo(repo)
    branch = _get_branch()
    root = load_tree(repo_root, branch=branch)
    llm = _init_provider()
    graph = _open_graph(repo_root)
    searcher = TreeSearcher(root, repo_root, llm, graph_db=graph)
 
    # Try to find by path first, then by name search
    node_map = root.get_node_map()
    target_node = None
 
    # Check if it's a file path
    for node in node_map.values():
        if node.path and (node.path.endswith(path_or_query) or path_or_query in node.path):
            target_node = node
            break
 
    if not target_node:
        # Search by name
        results = searcher.search(path_or_query, max_results=1)
        if results:
            target_node = results[0].node
 
    if not target_node:
        click.echo(f"❌ Could not find '{path_or_query}'")
        if graph:
            graph.close()
        return
 
    explanation = searcher.explain_node(target_node.node_id)
    click.echo(f"\n{target_node.title} ({target_node.path})\n")
    click.echo(explanation)
    if graph:
        graph.close()
 
 
@cli.command()
@click.option("--depth", default=3, help="Tree depth to display")
@click.option("--repo", default=None)
def tree(depth, repo):
    """Display the semantic tree structure."""
    from .storage import load_tree
    repo_root = _resolve_repo(repo)
    branch = _get_branch()
    root = load_tree(repo_root, branch=branch)
    _print_tree(root, depth, 0)
 
 
@cli.command()
@click.option("--repo", default=None)
@click.option("--parse-workers", default=None, type=int, help="Parallel parser worker count")
@click.option("--num-workers", default=None, type=int, help="Parallel LLM worker count (overrides provider default)")
@click.option("--quiet", is_flag=True, help="Suppress progress bars")
@click.option("--no-cache", is_flag=True, help="Ignore summary cache, re-summarize everything from scratch")
def sync(repo, parse_workers, num_workers, quiet, no_cache):
    """Re-index only changed files (fast incremental update)."""
    from tqdm import tqdm
    from .git_utils import is_git_repo, get_current_commit
    from .storage import load_tree, load_meta, save_tree, save_meta, register_repo, ensure_index_dir, LLMDiskCache, migrate_legacy_index, write_context_file
    from .tree_builder import find_stale_nodes, find_orphan_nodes, reparse_stale_files
    from .summarizer import Summarizer, summarize_tree
    from .graph_db import GraphDB
 
    repo_root = _resolve_repo(repo)
    branch = _get_branch()
    if branch:
        migrate_legacy_index(repo_root, branch)
    root = load_tree(repo_root, branch=branch)
    meta = load_meta(repo_root, branch=branch)
    embeddings_enabled = _resolve_embeddings_flag()
 
    start_time = time.time()
 
    # ── Phase 1: Detect changes ──
    stale_ids, deleted_ids = find_stale_nodes(root, repo_root)
    orphan_ids = find_orphan_nodes(root, repo_root)
    removed_ids = list(set(deleted_ids) | set(orphan_ids))
    all_stale = list(set(stale_ids) | set(removed_ids))
 
    graph = _open_graph(repo_root)
 
    needs_full_embed = (
        embeddings_enabled
        and graph is not None
        and not graph.has_embeddings()
    )
 
    if not all_stale and not needs_full_embed:
        click.echo("Nothing changed since last index.")
        if graph:
            graph.close()
        return
 
    has_work = bool(all_stale)
    total_phases = (3 if has_work else 0) + (1 if embeddings_enabled else 0) + 1  # +1 for save
    phase = 0
 
    def _phase(label: str) -> None:
        nonlocal phase
        phase += 1
        click.echo(f"\n[{phase}/{total_phases}] {label}")
 
    to_update: set[str] = set()
 
    if has_work:
        _phase("Detecting changes...")
        if stale_ids:
            click.echo(f"      {len(stale_ids)} files changed")
        if deleted_ids:
            click.echo(f"      {len(deleted_ids)} files deleted")
        if orphan_ids:
            click.echo(f"      {len(orphan_ids)} files newly ignored (gitignore/repolectignore)")
 
        llm = None
        if stale_ids:
            llm = _init_provider()
            ok, msg = llm.health_check()
            if not ok:
                click.echo(f"\nLLM provider is not reachable: {msg}", err=True)
                click.echo("Check that your provider (Ollama, OpenAI, etc.) is running and accessible.", err=True)
                if graph:
                    graph.close()
                sys.exit(1)
 
        # ── Re-parse ──
        _phase("Re-parsing changed files...")
        to_update = reparse_stale_files(root, repo_root, all_stale, graph_db=graph, parse_workers=parse_workers)
        if removed_ids:
            to_update = [nid for nid in to_update if nid not in set(removed_ids)]
        click.echo(f"      {len(stale_ids)} changed, {len(removed_ids)} removed, {len(to_update)} nodes to re-summarize")
 
        # ── Re-summarize ──
        if to_update and llm:
            _phase("Re-summarizing affected nodes...")
            cache = None
            if not no_cache:
                cache = LLMDiskCache(repo_root)
                purged = cache.purge_errors()
                if purged:
                    click.echo(f"      Purged {purged} cached error(s) from previous runs")
                llm.enable_cache(cache)
                click.echo(f"      LLM cache: {len(cache)} entries in SQLite")
 
            setup_msg = llm.parallel_setup_message()
            if setup_msg:
                click.echo(f"      {setup_msg}")
 
            workers = num_workers or llm.num_workers
            click.echo(f"      Workers: {workers}")
 
            summarizer = Summarizer(llm)
 
            if not quiet:
                with tqdm(total=len(to_update), desc="      Summarizing", unit="node", dynamic_ncols=True) as bar:
                    last = [0]
                    def on_progress(current, total, title, summary):
                        if summary and summary.startswith("[summary unavailable:"):
                            bar.write(f"      ⚠ Error summarizing '{title}': {summary}")
                        
                        postfix = title[-30:] if title else ""
                        if cache:
                            postfix += f" | H:{cache.hits} M:{cache.misses}"
                        bar.set_postfix_str(postfix)
                        bar.update(current - last[0])
                        last[0] = current
                    summarize_tree(root, repo_root, summarizer, only_node_ids=to_update, progress_callback=on_progress, max_workers=num_workers)
            else:
                summarize_tree(root, repo_root, summarizer, only_node_ids=to_update, max_workers=num_workers)
 
            if cache:
                click.echo(f"      Cache: {cache.hits} hits, {cache.misses} misses")
                cache.close()
                llm.disable_cache()
 
    # ── Optional: Embeddings ──
    if embeddings_enabled and graph:
        _phase("Updating embeddings...")
        from .embedder import get_embedder
        embedder = get_embedder()
        if embedder is not None:
            eok, emsg = embedder.health_check()
            if not eok:
                click.echo(f"\nEmbedding provider is not reachable: {emsg}", err=True)
                click.echo("Check that your embedding provider is running. Skipping embeddings.", err=True)
            elif needs_full_embed:
                count = embedder.embed_tree(root, graph)
                click.echo(f"      Embedded {count} nodes (full backfill)")
            elif to_update:
                node_map = root.get_node_map()
                embed_nodes = [node_map[nid] for nid in to_update if nid in node_map]
                if embed_nodes:
                    embedder.embed_nodes(embed_nodes, graph)
                    click.echo(f"      Re-embedded {len(embed_nodes)} nodes")
        else:
            click.echo("⚠ No embedding provider configured — skipping embeddings", err=True)
 
    # ── Save ──
    _phase("Saving index...")
    save_tree(root, repo_root, branch=branch)
    if meta:
        if is_git_repo(repo_root):
            meta.git_commit = get_current_commit(repo_root)
        meta.git_branch = branch
        meta.indexed_at = datetime.now(timezone.utc).isoformat()
        meta.embeddings_enabled = embeddings_enabled
        if graph:
            meta.graph_backend = graph.backend_name
        save_meta(meta, repo_root, branch=branch)
        register_repo(repo_root, meta)
        write_context_file(meta, root, repo_root, graph_db=graph)
 
    from .skill_installer import install_static_skills
    static_installed = install_static_skills(repo_root)
    if static_installed:
        click.echo(f"      Refreshed {len(static_installed)} agent skills")

    from .git_utils import ensure_gitignored
    ensure_gitignored(repo_root, [".claude/", ".cursor/", ".agents/", "REPOLECT.md"])
 
    if graph:
        graph.close()
 
    duration = time.time() - start_time
    parts = []
    if stale_ids:
        parts.append(f"{len(stale_ids)} changed")
    if removed_ids:
        parts.append(f"{len(removed_ids)} removed")
    if to_update:
        parts.append(f"{len(to_update)} re-summarized")
    click.echo(f"\nDone in {duration:.0f}s -- {', '.join(parts) or 'no changes'}")
 
 
@cli.command("list")
def list_cmd():
    """Show all indexed repositories."""
    from .storage import list_repos
    repos = list_repos()
    if not repos:
        click.echo("No repositories indexed yet. Run 'repolect analyze' in a repo.")
        return
 
    click.echo(f"\n{'Name':<22} {'Files':>5} {'Nodes':>6} {'Graph':>8} {'Embed':>6}  {'Indexed':<11}  {'Branches':<20}  Path")
    click.echo("─" * 120)
    for r in repos:
        name = r.get("repo_name", "unknown")[:21]
        files = r.get("file_count", "?")
        nodes = r.get("node_count", "?")
        graph_be = r.get("graph_backend", "?")[:7]
        embed = "✓" if r.get("embeddings_enabled") else "—"
        indexed = r.get("indexed_at", "")[:10]
        branches = r.get("branches", [])
        branches_str = ", ".join(branches[:5]) if branches else "—"
        if len(branches) > 5:
            branches_str += f" +{len(branches) - 5}"
        path = r.get("repo_path", "")
        click.echo(f"{name:<22} {files:>5} {nodes:>6} {graph_be:>8} {embed:>6}  {indexed:<11}  {branches_str:<20}  {path}")
 
 
@cli.command()
@click.argument("cypher_query")
@click.option("--repo", default=None)
def graph(cypher_query, repo):
    """Run a Cypher query against the code knowledge graph."""
    repo_root = _resolve_repo(repo)
    gdb = _open_graph(repo_root)
    if not gdb:
        click.echo("❌ No graph found. Run 'repolect analyze' first.", err=True)
        sys.exit(1)
 
    try:
        results = gdb.cypher(cypher_query)
        if not results:
            click.echo("(no results)")
        else:
            for row in results[:50]:
                formatted = "  ".join(str(cell) for cell in row)
                click.echo(formatted)
            if len(results) > 50:
                click.echo(f"... ({len(results) - 50} more rows)")
    except Exception as e:
        click.echo(f"❌ Query error: {e}", err=True)
    finally:
        gdb.close()
 
 
@cli.command()
@click.option("--repo", default=None)
def communities(repo):
    """Show detected code communities (Louvain clustering)."""
    from .storage import load_tree
 
    repo_root = _resolve_repo(repo)
    branch = _get_branch()
    root = load_tree(repo_root, branch=branch)
    gdb = _open_graph(repo_root)
 
    if not gdb:
        click.echo("No graph found. Run 'repolect analyze' first.", err=True)
        sys.exit(1)
 
    try:
        mapping = gdb.detect_communities()
    except NotImplementedError as e:
        click.echo(f"Not supported: {e}", err=True)
        gdb.close()
        sys.exit(1)
 
    node_map = root.get_node_map()
    from collections import defaultdict
    groups: dict[int, list[str]] = defaultdict(list)
    for nid, comm_id in mapping.items():
        groups[comm_id].append(nid)
 
    click.echo(f"\nDetected {len(groups)} communities:\n")
    for comm_id in sorted(groups.keys()):
        members = groups[comm_id]
        titled = []
        for nid in sorted(members):
            n = node_map.get(nid)
            if n and n.kind in ("function", "method", "class", "file"):
                titled.append(f"{n.title} ({n.kind})")
        if not titled:
            continue
        click.echo(f"  Community {comm_id} ({len(titled)} nodes):")
        for t in titled[:15]:
            click.echo(f"    - {t}")
        if len(titled) > 15:
            click.echo(f"    ... and {len(titled) - 15} more")
        click.echo()
 
    gdb.close()
 
 
@cli.command()
@click.argument("symbol")
@click.option("--repo", default=None)
@click.option("--max-hops", default=3, type=int, help="Maximum traversal depth")
def impact(symbol, repo, max_hops):
    """Show blast radius: what depends on a given file or symbol."""
    from .storage import load_tree
 
    repo_root = _resolve_repo(repo)
    branch = _get_branch()
    root = load_tree(repo_root, branch=branch)
    gdb = _open_graph(repo_root)
 
    if not gdb:
        click.echo("No graph found. Run 'repolect analyze' first.", err=True)
        sys.exit(1)
 
    node_map = root.get_node_map()
    target = None
    for node in node_map.values():
        if node.title == symbol or (node.path and node.path.endswith(symbol)):
            target = node
            break
 
    if not target:
        for node in node_map.values():
            if symbol.lower() in node.title.lower():
                target = node
                break
 
    if not target:
        click.echo(f"Could not find '{symbol}'. Try the exact name or path.", err=True)
        gdb.close()
        sys.exit(1)
 
    deps = gdb.get_reverse_dependencies(target.node_id, max_hops=max_hops, rel_types=["CALLS", "IMPORTS"])
 
    click.echo(f"\nImpact analysis for **{target.title}** ({target.path}):\n")
    if not deps:
        click.echo("  No dependents found.")
    else:
        click.echo(f"  {len(deps)} node(s) affected within {max_hops} hops:\n")
        for nid, hop in deps:
            n = node_map.get(nid)
            if n:
                click.echo(f"    [hop {hop}] {n.title} ({n.kind}, {n.path})")
            else:
                click.echo(f"    [hop {hop}] {nid}")
    click.echo()
    gdb.close()
 
 
@cli.command("diff")
@click.option("--ref", default="HEAD~1", help="Git ref to diff against (default: HEAD~1)")
@click.option("--repo", default=None)
@click.option("--with-impact", is_flag=True, help="Also show blast radius for each changed symbol")
@click.option("--max-hops", default=3, type=int, help="Max hops for impact analysis")
def diff_cmd(ref, repo, with_impact, max_hops):
    """Show which functions/classes changed since a git ref."""
    from .storage import load_tree
    from .git_utils import get_changed_line_ranges, is_git_repo
    from .tree_builder import map_changes_to_nodes
 
    repo_root = _resolve_repo(repo)
    if not is_git_repo(repo_root):
        click.echo("This command requires a git repository.", err=True)
        sys.exit(1)
 
    branch = _get_branch()
    root = load_tree(repo_root, branch=branch)
 
    changed_ranges = get_changed_line_ranges(repo_root, ref=ref)
    if not changed_ranges:
        click.echo(f"No changes detected since {ref}.")
        return
 
    affected = map_changes_to_nodes(changed_ranges, root)
    click.echo(f"\nChanges since {ref}:\n")
    click.echo(f"  {len(changed_ranges)} file(s) modified, {len(affected)} symbol(s) affected:\n")
 
    for node in affected:
        click.echo(f"  - {node.title} ({node.kind}, {node.path}:{node.line_start})")
 
    if with_impact and affected:
        gdb = _open_graph(repo_root)
        if gdb:
            click.echo(f"\nBlast radius (up to {max_hops} hops):\n")
            all_impacted: set[str] = set()
            for node in affected:
                deps = gdb.get_reverse_dependencies(
                    node.node_id, max_hops=max_hops, rel_types=["CALLS", "IMPORTS"],
                )
                if deps:
                    node_map = root.get_node_map()
                    for nid, hop in deps:
                        if nid not in all_impacted:
                            all_impacted.add(nid)
                            n = node_map.get(nid)
                            if n:
                                click.echo(f"    [hop {hop}] {n.title} ({n.kind}, {n.path})")
            if not all_impacted:
                click.echo("    No downstream dependents found.")
            gdb.close()
        else:
            click.echo("\nNo graph available for impact analysis. Run 'repolect analyze' first.", err=True)
    click.echo()
 
 
@cli.command()
def mcp():
    """Start MCP server for AI editor integration (Claude Code, Cursor, Windsurf, etc.)."""
    _start_mcp_server()
 
 
@cli.command()
@click.option("--repo", default=None, help="Repo name or ID (if not in a repo)")
@click.option("--port", default=8501, type=int, help="Streamlit server port")
def viz(repo, port):
    """Launch interactive graph visualization in the browser."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        click.echo("Visualization requires extra dependencies. Install with:\n  pip install repolect[viz]", err=True)
        sys.exit(1)
 
    repo_root = _resolve_repo(repo)
    branch = _get_branch()
 
    from .storage import get_index_dir
    index_dir = get_index_dir(repo_root, branch=branch)
    has_graph = (index_dir / "graph.db").exists() or (index_dir / "graph.pkl").exists()
    if not has_graph:
        click.echo("❌ No graph found. Run 'repolect analyze' first.", err=True)
        sys.exit(1)
 
    viz_script = str(Path(__file__).parent / "visualize.py")
    click.echo(f"Launching graph explorer for {Path(repo_root).name}...")
    click.echo(f"  http://localhost:{port}")
 
    import subprocess
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", viz_script,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--",
            "--repo-root", repo_root,
            "--branch", branch,
        ],
        check=False,
    )
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def _get_branch() -> str:
    """Return the current git branch, or '' if not in a git repo."""
    from .git_utils import is_git_repo, get_current_branch, detect_repo_root
    repo_root = detect_repo_root(".")
    if is_git_repo(repo_root):
        return get_current_branch(repo_root)
    return ""
 
 
def _resolve_repo(repo_name: str | None) -> str:
    """Resolve to an absolute repo root path. Works with or without git."""
    from .git_utils import detect_repo_root
    from .storage import find_repo, tree_exists, migrate_legacy_index
 
    if repo_name:
        entry = find_repo(repo_name)
        if not entry:
            click.echo(f"❌ Repo '{repo_name}' not found. Run 'repolect list'.", err=True)
            sys.exit(1)
        return entry["repo_path"]
 
    repo_root = detect_repo_root(".")
    branch = _get_branch()
 
    if branch:
        migrate_legacy_index(repo_root, branch)
 
    if not tree_exists(repo_root, branch=branch):
        click.echo("❌ No index found. Run 'repolect analyze' first.", err=True)
        sys.exit(1)
    return repo_root
 
 
def _open_graph(repo_root: str, branch: str | None = None):
    """Open the graph DB for a repo/branch, or return None if not available."""
    from .graph_db import GraphDB
    from .storage import get_index_dir
 
    if branch is None:
        branch = _get_branch()
 
    index_dir = get_index_dir(repo_root, branch=branch)
    graph_pkl = index_dir / "graph.pkl"
    graph_db_file = index_dir / "graph.db"
 
    if graph_pkl.exists() or graph_db_file.exists():
        try:
            return GraphDB.open(index_dir)
        except Exception:
            pass
    return None
 
 
def _print_tree(node, max_depth: int, current_depth: int) -> None:
    if current_depth > max_depth:
        return
 
    icons = {"repo": "📦", "module": "📁", "file": "📄", "class": "🔷", "function": "🔹",
             "method": "🔸", "interface": "🔶", "doc": "📝"}
    icon = icons.get(node.kind, "•")
    indent = "  " * current_depth
    summary_preview = f" — {node.summary[:60]}..." if node.summary else ""
 
    click.echo(f"{indent}{icon} {node.title}{summary_preview}")
 
    if current_depth < max_depth:
        for child in node.children:
            _print_tree(child, max_depth, current_depth + 1)
 
 
def _start_mcp_server() -> None:
    """Start the MCP stdio server."""
    try:
        from .mcp_server import start_server
        start_server()
    except ImportError:
        click.echo("MCP support not installed. Run: pip install repolect[mcp]", err=True)
 
 
if __name__ == "__main__":
    cli()
 