# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-03-28

### Added
- **Zsh/Shell Robustness** — Refactored `install.sh` to correctly detect the user's login shell via `$SHELL`, ensuring PATH updates always land in `.zshrc` regardless of how the installer is invoked.
- **Immediate Path Availability** — The installer now automatically symlinks binaries into `/usr/local/bin` for instant access without a `source` command.
- **Improved Error Messages** — Added specific instructions for `pipx inject` in CLI errors for users who installed via the script.

### Changed
- **MCP Config Safety** — The `repolect mcp` auto-configuration now safely merges with existing `mcp.json` files instead of potentially overwriting other servers.
- **Naming Consistency** — Completed the formal transition from "RepoBrain" to "Repolect" across all manifests, configurations, and internal metadata.
- **Default Exclusions** — Added standard `.gitignore` coverage for `.venv`, `__pycache__`, and editor-specific artifacts.

## [0.1.1] - 2026-03-28

### Fixed
- Duplicate GitHub Actions CI runs by refining push/pull_request triggers
- Project metadata in `pyproject.toml` (links and author attribution)

## [0.1.0] - 2026-03-28

### Added
- **Hierarchical Semantic Tree** — bottom-up LLM summarization of every symbol
- **Knowledge Graph** — `CALLS`, `IMPORTS`, `EXTENDS`, `IMPLEMENTS` relations
- **Vectorless Search** — LLM tree reasoning in O(log N) steps, no vector DB needed
- **MCP Server** — 14 tools for AI coding agents (Cursor, Claude Code, Windsurf)
- **Interactive Installer** — `install.sh` with Ollama setup and shell PATH config
- **Incremental Sync** — fast re-indexing of changed files only
- **Impact Analysis** — blast radius tracing for any symbol
- **Community Detection** — Louvain-based functional cluster discovery
- **Agent Skills** — auto-installed workflow skills for Cursor and Claude Code
- **Streamlit Visualizer** — interactive graph explorer
- **Test Suite** — baseline tests for models, config, parser, and git utilities

### Fixed
- Thread-safe SQLite caching for parallel LLM summarization
- Unicode non-printable character errors in parser output
- Initial public release as `Repolect`

### Changed
- LLM providers refactored to unified `BaseLLM`/`BaseEmbedder` abstract classes
- Token limits now configurable via `max_summarization_tokens` / `max_reasoning_tokens`
- `install.sh` updated with Conda-style shell initialization markers
