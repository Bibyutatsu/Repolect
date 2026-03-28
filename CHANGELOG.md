# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Legacy `RepoBrain` branding replaced with `Repolect`

### Changed
- LLM providers refactored to unified `BaseLLM`/`BaseEmbedder` abstract classes
- Token limits now configurable via `max_summarization_tokens` / `max_reasoning_tokens`
- `install.sh` updated with Conda-style shell initialization markers
