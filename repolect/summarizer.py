"""
Repolect — Summarizer
Build summaries bottom-up so that every node in the tree has an
LLM-generated summary. This enables vectorless tree search — the LLM
reasons over summaries, not raw code.
 
Design:
  1. Leaf nodes first (functions, methods, classes) — read actual source
  2. File nodes — synthesize from children summaries
  3. Module nodes — synthesize from file summaries
  4. Repo root — one-paragraph codebase overview
 
This is the most expensive step (LLM calls per node) but runs ONCE.
Incremental sync only re-summarizes changed nodes.
 
Provider resolution (config.yaml driven):
  REPOLECT_PROVIDER env → config.yaml provider field → auto-detect Ollama → error
 
Only two providers are supported:
  - ollama        (local, default)
  - openai-compatible  (any OpenAI-compatible API endpoint)
"""
 
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import logging
import os
from pathlib import Path
from typing import Callable, Iterator
from .models import CodeNode
 
logger = logging.getLogger(__name__)
 
# Preferred Ollama code models in priority order (install.sh default: qwen3.5:4b)
_OLLAMA_CODE_MODELS = [
    "qwen3.5:4b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:3b",
    "codellama:7b",
    "deepseek-coder:6.7b",
    "llama3.1:8b",
    "mistral:7b",
]
 
# ── Provider abstraction ─────────────────────────────────────────────────────
 
class BaseLLM(ABC):
    """Abstract base — swap providers without changing Summarizer logic.
 
    Subclasses must implement ``call`` and ``chat``.  They *should* override
    ``num_workers`` and ``parallel_setup_message`` so that the summarization
    pipeline can pick the right concurrency and give the user actionable
    guidance when the server needs reconfiguration.
    """
 
    provider_name: str = "base"
    _cache = None  # Optional LLMDiskCache instance
 
    # ── abstract ─────────────────────────────────────────────────────────
 
    @abstractmethod
    def call(self, message: str, stream: bool = False, max_tokens: int = 200) -> str | Iterator[str]:
        pass
 
    @abstractmethod
    def chat(self, messages: list[dict], stream: bool = False, max_tokens: int = 200) -> str | Iterator[str]:
        pass
 
    # ── parallelism ──────────────────────────────────────────────────────
 
    @property
    def num_workers(self) -> int:
        """Max concurrent ``complete()`` calls this provider supports.
 
        Override in subclasses.  The summarization pipeline uses this as the
        default thread-pool size.  The CLI ``--num-workers`` flag takes
        precedence when provided.
        """
        return 1
 
    def parallel_setup_message(self) -> str | None:
        """Return a user-facing hint if the server needs reconfiguration
        to enable parallel inference.  Return ``None`` when no action is
        needed.  Called once by the CLI before summarization starts.
        """
        return None
 
    # ── error detection ─────────────────────────────────────────────────
 
    _ERROR_SENTINEL = "[summary unavailable:"
 
    def _is_error_response(self, text: str) -> bool:
        return text.startswith(self._ERROR_SENTINEL)
 
    # ── health check ──────────────────────────────────────────────────
 
    def health_check(self) -> tuple[bool, str]:
        """Quick connectivity test. Returns (ok, message)."""
        result = str(self.call("Say OK", stream=False, max_tokens=5))
        if self._is_error_response(result):
            return False, result
        return True, "OK"
 
    # ── caching ──────────────────────────────────────────────────────────
 
    def enable_cache(self, cache) -> None:
        """Attach an LLMDiskCache so all complete() calls are cached."""
        self._cache = cache
 
    def disable_cache(self) -> None:
        self._cache = None
 
    def complete(self, prompt: str, max_tokens: int = 200) -> str:
        """Cached completion: checks disk cache before calling the LLM."""
        if self._cache is not None:
            from .storage import LLMDiskCache
            key = LLMDiskCache.make_key(
                self.provider_name,
                getattr(self, "model", ""),
                max_tokens,
                prompt,
            )
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            result = str(self.call(prompt, stream=False, max_tokens=max_tokens))
            if result and result.strip() and not self._is_error_response(result):
                self._cache.put(key, result)
            return result
        return str(self.call(prompt, stream=False, max_tokens=max_tokens))
 
    def stream_complete(self, prompt: str, max_tokens: int = 200) -> Iterator[str]:
        """Streaming completion with cache support (tee pattern).
 
        Cache hit  → yield the cached string as a single chunk.
        Cache miss → stream from LLM, accumulate, write to cache after exhaustion.
        No cache   → pure pass-through.
        """
        if self._cache is not None:
            from .storage import LLMDiskCache
            key = LLMDiskCache.make_key(
                self.provider_name,
                getattr(self, "model", ""),
                max_tokens,
                prompt,
            )
            cached = self._cache.get(key)
            if cached is not None:
                yield cached
                return
            chunks: list[str] = []
            for chunk in self.call(prompt, stream=True, max_tokens=max_tokens):
                chunks.append(chunk)
                yield chunk
            full_text = "".join(chunks)
            if full_text and full_text.strip() and not self._is_error_response(full_text):
                self._cache.put(key, full_text)
            return
        yield from self.call(prompt, stream=True, max_tokens=max_tokens)
 
    # ── misc ─────────────────────────────────────────────────────────────
 
    def clone(self) -> "BaseLLM":
        return self
 
    @classmethod
    def is_available(cls, **kwargs) -> bool:
        """Return True if this provider can be instantiated right now."""
        return False
 
 
_PROVIDER_REGISTRY: dict[str, type[BaseLLM]] = {}
_EXTERNAL_PROVIDER_SPEC: str | None = None
 
 
def register_provider(name: str, provider_cls: type[BaseLLM]) -> None:
    _PROVIDER_REGISTRY[name.lower().strip()] = provider_cls
 
 
def _load_external_provider() -> None:
    global _EXTERNAL_PROVIDER_SPEC
    spec = os.environ.get("REPOLECT_PROVIDER_CLASS", "").strip()
    if not spec or spec == _EXTERNAL_PROVIDER_SPEC:
        return
 
    module_name, _, attr_name = spec.partition(":")
    if not module_name or not attr_name:
        raise RuntimeError("REPOLECT_PROVIDER_CLASS must be in the form package.module:ClassName")
 
    module = importlib.import_module(module_name)
    provider_cls = getattr(module, attr_name)
    if not isinstance(provider_cls, type) or not issubclass(provider_cls, BaseLLM):
        raise RuntimeError("REPOLECT_PROVIDER_CLASS must point to a BaseLLM subclass")
 
    register_provider(getattr(provider_cls, "provider_name", attr_name).lower(), provider_cls)
    _EXTERNAL_PROVIDER_SPEC = spec
 
 
class OllamaProvider(BaseLLM):
    """Ollama local LLM — completely free, private. Uses native ollama package."""
    provider_name = "ollama"
 
    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.1, max_tokens: int = 200):
        try:
            import ollama as _ollama  # noqa: F401
        except ImportError:
            raise ImportError("Run: pip install 'repolect[ollama]'")
 
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or self.auto_select_model() or "qwen3.5:4b"
        self.temperature = float(temperature)
        self.default_max_tokens = int(max_tokens)
        self._client = None
        self._detected_workers: int | None = None
 
    def _get_client(self):
        if self._client is None:
            from ollama import Client
            self._client = Client(host=self.host)
        return self._client
 
    # ── parallelism ──────────────────────────────────────────────────────
 
    @property
    def num_workers(self) -> int:
        if self._detected_workers is not None:
            return self._detected_workers
        self._detected_workers = self._detect_parallel_slots()
        return self._detected_workers
 
    def _detect_parallel_slots(self) -> int:
        """Probe Ollama to discover OLLAMA_NUM_PARALLEL.
 
        Sends one baseline request, then two concurrent requests.
        If the pair completes in roughly the same wall-time as one call
        the server is running with NUM_PARALLEL >= 2.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor
 
        def _ping() -> float:
            t0 = time.monotonic()
            try:
                self._get_client().chat(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    think=False,
                    options={"num_predict": 1},
                )
            except Exception:
                pass
            return time.monotonic() - t0
 
        try:
            baseline = _ping()
 
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(lambda _: _ping(), range(2)))
            pair_time = time.monotonic() - t0
 
            if pair_time < baseline * 1.6:
                return max(2, int(pair_time / baseline * 2 + 0.5))
        except Exception:
            pass
        return 1
 
    def parallel_setup_message(self) -> str | None:
        if self.num_workers <= 1:
            return (
                "Ollama is processing requests sequentially (OLLAMA_NUM_PARALLEL=1).\n"
                "      To speed up summarization ~4x, restart Ollama with:\n"
                "        launchctl setenv OLLAMA_NUM_PARALLEL 4  # macOS\n"
                "        # or: OLLAMA_NUM_PARALLEL=4 ollama serve"
            )
        return None
 
    # ── core LLM calls ───────────────────────────────────────────────────
 
    def call(self, message: str, stream: bool = False, max_tokens: int = 200) -> str | Iterator[str]:
        return self.chat([{"role": "user", "content": message}], stream=stream, max_tokens=max_tokens)
 
    def chat(self, messages: list[dict], stream: bool = False, max_tokens: int | None = None) -> str | Iterator[str]:
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=messages,
                stream=stream,
                think=False,
                options={"num_predict": max_tokens, "temperature": self.temperature},
            )
            if stream:
                return (chunk.message.content for chunk in response)
            return response.message.content.strip()
        except Exception as e:
            err_msg = f"[summary unavailable: {e}]"
            if stream:
                return iter([err_msg])
            return err_msg
 
    # ── discovery ────────────────────────────────────────────────────────
 
    @classmethod
    def is_available(cls, **kwargs) -> bool:
        try:
            from ollama import Client
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            Client(host=host).list()
            return True
        except Exception:
            return False
 
    @classmethod
    def auto_select_model(cls) -> str | None:
        try:
            from ollama import Client
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            resp = Client(host=host).list()
            models = {m.model for m in resp.models}
            for preferred in _OLLAMA_CODE_MODELS:
                if preferred in models:
                    logger.info("Auto-selected Ollama model: %s", preferred)
                    return preferred
            if models:
                first = sorted(models)[0]
                logger.info("No preferred code model found, using: %s", first)
                return first
        except Exception:
            pass
        return None
 
    def clone(self) -> "OllamaProvider":
        return OllamaProvider(model=self.model, host=self.host,
                              temperature=self.temperature,
                              max_tokens=self.default_max_tokens)
 
 
register_provider("ollama", OllamaProvider)
 
 
class OpenAICompatibleProvider(BaseLLM):
    """Generic OpenAI-compatible provider using requests (no openai SDK needed)."""
    provider_name = "openai-compatible"
 
    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1",
                 api_key: str = "", temperature: float = 0.1, max_tokens: int = 200,
                 timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.default_max_tokens = int(max_tokens)
        self.timeout = int(timeout)
 
    @property
    def num_workers(self) -> int:
        return 8
 
    def call(self, message: str, stream: bool = False, max_tokens: int = 200) -> str | Iterator[str]:
        return self.chat([{"role": "user", "content": message}], stream=stream, max_tokens=max_tokens)
 
    def chat(self, messages: list[dict], stream: bool = False, max_tokens: int | None = None) -> str | Iterator[str]:
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if stream:
            return self._chat_stream(messages, max_tokens)
        
        try:
            import requests
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
 
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[summary unavailable: {e}]"
 
    def _chat_stream(self, messages: list[dict], max_tokens: int) -> Iterator[str]:
        import requests
        import json
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
 
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "stream": True,
                },
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                        data = json.loads(decoded[6:])
                        if data['choices'] and 'delta' in data['choices'][0]:
                            if 'content' in data['choices'][0]['delta']:
                                yield data['choices'][0]['delta']['content']
        except Exception as e:
            yield f"[summary unavailable: {e}]"
 
    @classmethod
    def is_available(cls, **kwargs) -> bool:
        return False
 
    def clone(self) -> "OpenAICompatibleProvider":
        return OpenAICompatibleProvider(model=self.model, base_url=self.base_url,
                                        api_key=self.api_key,
                                        temperature=self.temperature,
                                        max_tokens=self.default_max_tokens,
                                        timeout=self.timeout)
 
 
register_provider("openai-compatible", OpenAICompatibleProvider)
 
 
def get_provider(config: dict | None = None) -> BaseLLM:
    """Resolve the best available LLM provider.
 
    Resolution order (config.yaml driven, no CLI flags):
      1. REPOLECT_PROVIDER env var  (for CI / scripts)
      2. config.yaml ``provider`` field
      3. Auto-detect Ollama
      4. Error
    """
    _load_external_provider()
    provider_map = dict(_PROVIDER_REGISTRY)
 
    # 1. Explicit env override
    env_provider = os.environ.get("REPOLECT_PROVIDER", "").lower().strip()
    if env_provider and env_provider in provider_map:
        logger.info("Using provider from REPOLECT_PROVIDER: %s", env_provider)
        return provider_map[env_provider]()
 
    # 2. Config file
    if config is None:
        try:
            from .config import load_config
            config = load_config()
        except Exception:
            config = {}
 
    config_provider = config.get("provider", "").strip().lower()
    if config_provider and config_provider in provider_map:
        config_kwargs: dict = {}
        config_model = config.get("model_name", "").strip()
        config_base_url = config.get("base_url", "").strip()
        config_api_key = config.get("api_key", "").strip()
        config_temperature = float(config.get("temperature", "0.1").strip())
        config_max_tokens = int(config.get("max_tokens", "200").strip())
        config_timeout = int(config.get("timeout", "60").strip())
 
        if config_model:
            config_kwargs["model"] = config_model
        config_kwargs["temperature"] = config_temperature
        config_kwargs["max_tokens"] = config_max_tokens
 
        if config_provider == "openai-compatible":
            if config_base_url:
                config_kwargs["base_url"] = config_base_url
            if config_api_key:
                config_kwargs["api_key"] = config_api_key
            config_kwargs["timeout"] = config_timeout
            logger.info("Using config provider: openai-compatible (%s)", config_base_url)
            return provider_map[config_provider](**config_kwargs)
 
        if config_provider == "ollama" and config_base_url:
            config_kwargs["host"] = config_base_url
 
        logger.info("Using config provider: %s", config_provider)
        try:
            return provider_map[config_provider](**config_kwargs)
        except Exception as e:
            logger.warning("Config provider '%s' failed: %s, trying auto-detect", config_provider, e)
 
    # 3. Auto-detect Ollama
    ollama_cls = provider_map.get("ollama")
    if ollama_cls and ollama_cls.is_available():
        logger.info("Auto-detected provider: ollama")
        return ollama_cls()
 
    raise RuntimeError(
        "No LLM provider available. Options:\n"
        "  • Start Ollama: ollama serve  (then pull a model: ollama pull qwen3.5:4b)\n"
        "  • Edit ~/.repolect/config.yaml to configure a provider\n"
        "    Supported providers: ollama, openai-compatible"
    )
 
 
# ── Summarizer ───────────────────────────────────────────────────────────────
 
class Summarizer:
    """
    Generates LLM summaries for every node in the tree, bottom-up.
    This is the core "magic" that makes vectorless search possible.
    """
 
    def __init__(self, provider: BaseLLM, rate_limit_delay: float = 0.0):
        self.provider = provider
        self.delay = rate_limit_delay
 
    def summarize_leaf(self, node: CodeNode, source_snippet: str) -> str:
        """
        Summarize a leaf node (function, method, class body).
        Prefers signature + truncated source to minimise tokens sent to local LLMs.
        """
        if getattr(node, "signature", None):
            content = f"Signature:\n{node.signature}\n\nSource:\n{source_snippet[:800]}"
        else:
            content = source_snippet[:1000]
 
        prompt = f"""Summarize this {node.kind} named `{node.title}` in 1 sentence.
Focus strictly on its core purpose and inputs/outputs.
Do not use introductory filler phrases. State facts directly.
 
```{node.language}
{content}
```
 
Summary (1 sentence):"""
 
        return self.provider.complete(prompt, max_tokens=100)
 
    def summarize_file(self, node: CodeNode) -> str:
        """Summarize a file from its children's summaries — no source reading needed."""
        if not node.children:
            return f"Empty {node.language} file."
 
        children_list = "\n".join(
            f"  - {c.title} ({c.kind}): {c.summary[:100] or '[no summary]'}"
            for c in node.children[:20]
        )
 
        prompt = f"""Summarize the file `{node.title}` based on its contents.
 
Components:
{children_list}
 
Write 1-2 sentences covering: (1) the file's overall purpose, (2) its key responsibilities.
Do not list the components — synthesize them into a coherent description.
 
Summary:"""
 
        return self.provider.complete(prompt, max_tokens=150)
 
    def summarize_module(self, node: CodeNode) -> str:
        """Summarize a directory/module from its files' summaries."""
        if not node.children:
            return f"Empty module at {node.path}."
 
        children_list = "\n".join(
            f"  - {c.title}: {c.summary[:120] or '[no summary]'}"
            for c in node.children[:15]
        )
 
        prompt = f"""Summarize the module/directory `{node.title}` based on its files.
 
Files:
{children_list}
 
Write 2-3 sentences covering the module's purpose and its role in the larger system.
Be specific. Avoid phrases like "this module contains" or "this directory has".
 
Summary:"""
 
        return self.provider.complete(prompt, max_tokens=150)
 
    def summarize_doc(self, node: CodeNode, content: str) -> str:
        """Summarize a documentation file (README, etc.)."""
        prompt = f"""Summarize this documentation file `{node.title}` in 2-3 sentences.
Cover: what the project/section does, who it's for, and any key usage patterns mentioned.
 
Content (first 2500 chars):
{content[:2500]}
 
Summary:"""
 
        return self.provider.complete(prompt, max_tokens=150)
 
    def summarize_repo(self, root: CodeNode) -> str:
        """Generate the top-level repository overview."""
        modules_list = "\n".join(
            f"  - {c.title}: {c.summary[:150] or '[no summary]'}"
            for c in root.children[:12]
        )
 
        lang_info = ""
        all_langs = set()
        for node in root.flat_iter():
            if node.language:
                all_langs.add(node.language)
        if all_langs:
            lang_info = f"Languages: {', '.join(sorted(all_langs))}\n"
 
        prompt = f"""Write a 3-sentence overview of the codebase named `{root.title}`.
{lang_info}
Top-level modules:
{modules_list}
 
Cover: (1) what the system does, (2) its main architectural components, (3) who would use it.
Be specific and concrete. Avoid vague words like "comprehensive" or "robust".
 
Overview:"""
 
        return self.provider.complete(prompt, max_tokens=250)
 
 
# ── Pipeline ─────────────────────────────────────────────────────────────────
 
def summarize_tree(
    root: CodeNode,
    repo_root: str | Path,
    summarizer: Summarizer,
    only_node_ids: list[str] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    max_workers: int | None = None,
) -> None:
    """Bottom-up summarization of the entire tree.
 
    Caching is handled transparently at the ``BaseLLM.complete()`` layer
    via ``LLMDiskCache`` — every LLM call is cached individually in SQLite,
    committed on each write, and survives process restarts.
 
    Args:
        root: The root CodeNode (modified in place).
        repo_root: Absolute path to the repository.
        summarizer: LLM summarizer instance.
        only_node_ids: If set, only re-summarize these nodes (incremental sync).
        progress_callback: Optional (current, total, title) -> None.
    """
    repo_root = Path(repo_root).resolve()
 
    if only_node_ids is not None:
        target_ids = set(only_node_ids)
    else:
        target_ids = None
 
    ordered = list(_postorder(root))
    targeted = [n for n in ordered if target_ids is None or n.node_id in target_ids]
    total = len(targeted)
    current = 0
 
    leaf_kinds = {"doc", "class", "interface", "function", "method"}
    stages = [
        [n for n in targeted if n.kind in leaf_kinds or (n.kind == "file" and not n.children)],
        [n for n in targeted if n.kind == "file" and n.children],
        [n for n in targeted if n.kind == "module"],
        [n for n in targeted if n.kind == "repo"],
    ]
 
    provider_default = summarizer.provider.num_workers
    resolved_workers = _resolve_worker_count(max_workers, "REPOLECT_SUMMARY_WORKERS", provider_default)
 
    for stage in stages:
        if not stage:
            continue
 
        if resolved_workers <= 1 or len(stage) == 1:
            for node in stage:
                node.summary = _summarize_node(node, repo_root, summarizer)
                current += 1
                if progress_callback:
                    progress_callback(current, total, node.title, node.summary)
            continue
 
        with ThreadPoolExecutor(max_workers=min(resolved_workers, len(stage))) as executor:
            future_map = {
                executor.submit(_summarize_node, node, repo_root, summarizer): node
                for node in stage
            }
            for future in as_completed(future_map):
                node = future_map[future]
                node.summary = future.result()
                current += 1
                if progress_callback:
                    progress_callback(current, total, node.title, node.summary)
 
 
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
 
 
def _summarize_node(node: CodeNode, repo_root: Path, summarizer: Summarizer) -> str:
    if node.kind == "repo":
        return summarizer.summarize_repo(node)
 
    if node.kind == "module":
        return summarizer.summarize_module(node)
 
    if node.kind == "doc":
        source = _read_source(repo_root / node.path)
        return summarizer.summarize_doc(node, source)
 
    if node.kind == "file":
        if node.children:
            return summarizer.summarize_file(node)
        source = _read_source(repo_root / node.path)
        if source.strip():
            return summarizer.summarize_leaf(node, source)
        return f"Empty {node.language} file."
 
    if node.kind in ("class", "interface", "function", "method"):
        source = _read_source_lines(repo_root / node.path, node.line_start, node.line_end)
        return summarizer.summarize_leaf(node, source)
 
    return f"{node.kind}: {node.title}"
 
 
def _postorder(node: CodeNode):
    """Post-order traversal: children before parents (leaves first)."""
    for child in node.children:
        yield from _postorder(child)
    yield node
 
 
def _read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (IOError, OSError):
        return ""
 
 
def _read_source_lines(path: Path, start: int, end: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # Add a few lines of context before/after
        s = max(0, start - 3)
        e = min(len(lines), end + 3)
        snippet = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines[s:e], start=s))
        return snippet
    except (IOError, OSError):
        return ""
 