---
name: repolect-exploring
description: Use when navigating unfamiliar code, onboarding to a new codebase, or answering "how does X work?" questions.
globs:
alwaysApply: false
---
 
# Exploring Unfamiliar Code with Repolect
 
## When to Use
 
- First time working in a codebase or module
- User asks "how does X work?" or "where is Y handled?"
- Need to understand execution flow or architecture
- Onboarding to an unfamiliar area
 
## Workflow
 
```
Step 1: Orient
  repo_summary()  →  High-level overview, module list, language breakdown
 
Step 2: Search by meaning
  tree_search(query="how does authentication work?")  →  LLM-guided semantic results with node IDs
 
Step 3: Deep-dive into specific symbols
  get_node(node_id="...")  →  Full source + callers + callees + summary in one call
 
Step 4: Follow execution flow
  trace_flow(entry_point="...")  →  Call tree with summaries, cross-file resolution, cycle detection
 
Step 5: Understand purpose
  explain_node(node_id="...")  →  Narrative explanation of why this exists and how it fits
```
 
## Key Principles
 
- Start broad (repo_summary), narrow with search (tree_search), then drill in (get_node).
- tree_search understands meaning, not just text — ask questions naturally.
- get_node is your 360-degree view: source + callers + callees replaces reading file + grepping.
- trace_flow follows calls across files — use it to understand pipelines and data flow.
- Use the `node_id` from any tool's output to chain into the next tool.
 
## Pro Tips
 
- tree_search works without embeddings — it uses LLM reasoning over the summary tree.
- trace_flow detects cycles and marks them with `↻ cycle` — safe on recursive code.
- For structural questions ("which files import X?", "most connected nodes"), use `graph_query` with Cypher.
 