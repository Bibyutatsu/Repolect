---
name: repolect-debugging
description: Use when tracing bugs, investigating errors, understanding failure paths, or diagnosing unexpected behavior.
globs:
alwaysApply: false
---
 
# Debugging with Repolect
 
## When to Use
 
- Investigating a bug report or error trace
- Tracing how data flows through a failing path
- Understanding what calls a broken function
- Diagnosing unexpected behavior
 
## Workflow
 
```
Step 1: Find the suspect
  tree_search(query="where is the JWT token validated?")
  → Semantic search finds relevant code by meaning
 
Step 2: Get full context
  get_node(node_id="...")
  → Source code + callers + callees + summary — see who calls this and what it calls
 
Step 3: Trace the execution path
  trace_flow(entry_point="...")
  → Full call tree from entry point, with cross-file resolution and cycle detection
 
Step 4: Check blast radius
  impact_analysis(node_id="...", max_hops=3)
  → Everything that depends on the broken symbol — what else might be affected
 
Step 5: Find tests
  scope_test(node_ids="...")
  → Which tests cover this code — run them to verify your fix
```
 
## Key Principles
 
- get_node shows callers (who triggers this?) and callees (what does it depend on?) in one call.
- trace_flow reveals the full execution chain — follow it to find where data gets corrupted or lost.
- impact_analysis after finding the bug tells you what else might have the same problem.
- Use graph_query for custom structural questions: "which functions call X but not Y?"
 
## Pro Tips
 
- trace_flow accepts both node_id and natural language — `trace_flow(entry_point="handle_login")` works.
- impact_analysis marks `[test]` and `[entrypoint]` nodes — quickly identify test coverage gaps.
- For "what changed recently?", use `diff_analysis(ref="HEAD~5")` to correlate with when the bug appeared.
- Chain tools: tree_search → get_node → trace_flow → impact_analysis builds a complete picture.
 