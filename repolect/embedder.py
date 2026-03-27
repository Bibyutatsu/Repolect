"""
Repolect — Embedder
Generates embeddings via Ollama's embedding API.
 
Uses the configured embedding model (default: qwen3-embedding:0.6b) running
in the same Ollama instance used for summarization. No subprocess isolation
needed — Ollama runs as a separate daemon process.
 
Usage:
    from repolect.embedder import get_embedder
    embedder = get_embedder()  # reads config, returns a BaseEmbedder subclass
    embedder.embed_tree(root, graph_db, progress_callback=my_cb)
"""
 
from __future__ import annotations
 
from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Any, Callable
 
from .models import CodeNode
 
logger = logging.getLogger(__name__)
 
# Safety threshold — warn user if tree is very large
_MAX_NODES_WARN = 100_000
_DEFAULT_MODEL = "qwen3-embedding:0.6b"
_DEFAULT_BATCH_SIZE = 64
 
 
class BaseEmbedder(ABC):
    """Abstract base for generating embeddings with built-in batching logic."""
    model: str
    batch_size: int
 
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings for a batch of strings."""
        pass
 
    def health_check(self) -> tuple[bool, str]:
        """Quick connectivity test. Returns (ok, message)."""
        result = self.embed_documents(["test"])
        if result is None:
            return False, "Embedding provider is not reachable"
        return True, "OK"
 
    def embed_tree(
        self,
        root: CodeNode,
        graph_db,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        return self.embed_nodes(list(root.flat_iter()), graph_db, progress_callback=progress_callback)
 
    def embed_nodes(
        self,
        nodes: list[CodeNode],
        graph_db,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Generate and store embeddings for a list of nodes, processing in batches to respect memory constraints."""
        items: list[tuple[str, str]] = []
        for node in nodes:
            text = _node_to_text(node)
            if text:
                items.append((node.node_id, text))
 
        if not items:
            logger.warning("No embeddable content found in the tree")
            return 0
 
        if len(items) > _MAX_NODES_WARN:
            logger.warning(
                "Large tree detected (%d nodes). Embedding may take a while.", len(items)
            )
 
        node_ids = [nid for nid, _ in items]
        texts = [text for _, text in items]
 
        logger.info(
            "Embedding %d nodes with model %s (batch_size=%d)",
            len(texts), getattr(self, "model", "unknown"), getattr(self, "batch_size", _DEFAULT_BATCH_SIZE)
        )
 
        stored = 0
        total = len(texts)
 
        # Process in batches to ensure strict RAM boundaries (1-2GB handling capability)
        batch_size = getattr(self, "batch_size", _DEFAULT_BATCH_SIZE)
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_texts = texts[batch_start:batch_end]
            batch_ids = node_ids[batch_start:batch_end]
 
            vectors = self.embed_documents(batch_texts)
            if vectors is None:
                logger.error("Embedding batch %d-%d failed", batch_start, batch_end)
                continue
 
            for i, (node_id, vec) in enumerate(zip(batch_ids, vectors)):
                try:
                    graph_db.add_node(node_id, embedding=vec)
                    stored += 1
                except Exception as e:
                    logger.debug("Failed to store embedding for %s: %s", node_id, e)
 
                global_idx = batch_start + i
                if progress_callback and (global_idx % 50 == 0 or global_idx == total - 1):
                    progress_callback(global_idx + 1, total, node_id)
 
        logger.info("Stored %d embeddings in graph", stored)
        graph_db.save()
        return stored
 
 
class OllamaEmbedder(BaseEmbedder):
    """Generates embeddings via Ollama python SDK."""
 
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.batch_size = batch_size
 
    def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        try:
            from ollama import Client
            client = Client(host=self.host)
            response = client.embed(model=self.model, input=texts)
            return response.get("embeddings", [])
        except Exception as e:
            logger.error("Ollama embedding failed: %s", e)
            return None
 
 
class OpenAICompatibleEmbedder(BaseEmbedder):
    """Generates embeddings via any OpenAI-compatible embedding API."""
 
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
 
    def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        try:
            import requests
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
 
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            return embeddings
        except Exception as e:
            logger.error("OpenAI-compatible embedding failed: %s", e)
            return None
 
 
# ── Factory ──────────────────────────────────────────────────────────────────
 
def get_embedder(config: dict | None = None):
    """Create the right embedder based on config.
 
    Args:
        config: Config dict (from config.load_config). If None, loads automatically.
 
    Returns:
        OllamaEmbedder or OpenAICompatibleEmbedder instance, or None if no embedding configured.
    """
    if config is None:
        from .config import load_config
        config = load_config()
 
    embed_provider = config.get("embedding_provider", "").strip()
    embed_model = config.get("embedding_model", "").strip()
 
    if not embed_provider or not embed_model:
        logger.warning(
            "Embedding not configured (embedding_provider / embedding_model "
            "missing in config). Proceeding without embeddings."
        )
        return None
 
    if embed_provider == "ollama":
        host = config.get("embedding_base_url", "").strip()
        if not host:
            host = config.get("base_url", "http://localhost:11434")
        return OllamaEmbedder(model=embed_model, host=host)
 
    elif embed_provider == "openai-compatible":
        base_url = config.get("embedding_base_url", "").strip()
        if not base_url:
            base_url = config.get("base_url", "https://api.openai.com/v1")
        api_key = config.get("api_key", "")
        return OpenAICompatibleEmbedder(model=embed_model, base_url=base_url, api_key=api_key)
 
    logger.warning("Unknown embedding provider: %s", embed_provider)
    return None
 
 
# ── Shared helpers ───────────────────────────────────────────────────────────
 
def _node_to_text(node: CodeNode) -> str:
    """Convert a CodeNode to embeddable text."""
    parts = []
 
    if node.title:
        parts.append(node.title)
 
    if node.kind:
        parts.append(f"({node.kind})")
 
    if node.summary:
        parts.append(node.summary)
    elif node.signature:
        parts.append(node.signature)
 
    text = " ".join(parts).strip()
    # Truncate to ~512 tokens (~2000 chars for most models)
    return text[:2000] if text else ""