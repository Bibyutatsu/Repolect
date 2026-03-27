"""
Repolect — Hybrid Tree + Graph Code Intelligence
Build a hierarchical semantic tree and knowledge graph of any codebase,
use LLM reasoning to navigate it, and optionally generate embeddings.
"""

from .models import CodeNode, Relation, TreeMeta, SearchResult, NodeKind, RelationKind
from .storage import save_tree, load_tree, save_meta, load_meta, tree_exists, clean_index
from .storage import register_repo, unregister_repo, list_repos, find_repo
from .storage import load_ignore_patterns, write_context_file
from .graph_db import GraphDB
from .config import load_config, save_config, get_config_value, config_exists
from .summarizer import BaseLLM, OllamaProvider, OpenAICompatibleProvider, get_provider, register_provider
from .search import TreeSearcher, Explainer
from .embedder import BaseEmbedder, OllamaEmbedder, OpenAICompatibleEmbedder, get_embedder
from .tree_builder import build_raw_tree, build_graph, reparse_stale_files
from .git_utils import detect_repo_root, is_git_repo

__version__ = "0.1.0"

__all__ = [
    # Models
    "CodeNode",
    "Relation",
    "TreeMeta",
    "SearchResult",
    "NodeKind",
    "RelationKind",
    # Storage
    "save_tree",
    "load_tree",
    "save_meta",
    "load_meta",
    "tree_exists",
    "clean_index",
    "register_repo",
    "unregister_repo",
    "list_repos",
    "find_repo",
    "load_ignore_patterns",
    "write_context_file",
    # Graph
    "GraphDB",
    # Config
    "load_config",
    "save_config",
    "get_config_value",
    "config_exists",
    # LLM Providers
    "BaseLLM",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "get_provider",
    "register_provider",
    # Search
    "TreeSearcher",
    "Explainer",
    # Embeddings
    "BaseEmbedder",
    "OllamaEmbedder",
    "OpenAICompatibleEmbedder",
    "get_embedder",
    # Tree Builder
    "build_raw_tree",
    "build_graph",
    "reparse_stale_files",
    # Git Utils
    "detect_repo_root",
    "is_git_repo",
]
