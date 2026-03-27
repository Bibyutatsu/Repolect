---
name: repolect-reviewing
description: Use before committing changes, during code review, or when verifying that changes are safe to merge.
globs:
alwaysApply: false
---
 
# Reviewing Changes with Repolect
 
## When to Use
 
- Pre-commit safety check
- Reviewing a PR or branch diff
- Verifying that changes don't break downstream code
- Assessing risk before merging
 
## Workflow
 
```
Step 1: See what changed
  diff_analysis(ref="HEAD~1", with_impact=True)
  → All changed symbols + their downstream blast radius
 
Step 2: Check blast radius of high-risk changes
  impact_analysis(node_id="<changed symbol>", max_hops=3)
  → Multi-hop dependency map: Hop 1 = WILL break, Hop 2+ = LIKELY affected
 
Step 3: Identify tests to run
  scope_test(node_ids="<all changed node IDs>")
  → MUST RUN (direct dependents) + SHOULD RUN (transitive dependents)
 
Step 4: Verify understanding of changes
  get_node(node_id="<changed symbol>")
  → Full source + callers + callees — verify callers won't break
```
 
## Key Principles
 
- diff_analysis is the single most important pre-commit tool — it maps git changes to affected code symbols.
- Use `ref="main"` for branch-level comparison, `ref="HEAD~1"` for last-commit comparison.
- impact_analysis on changed symbols reveals downstream breakage you might not have considered.
- scope_test gives you the minimal test set — run exactly these, not the entire suite.
 
## Pro Tips
 
- diff_analysis with `with_impact=True` combines change detection + blast radius in one call.
- `committed_only=True` on diff_analysis ignores unstaged changes — useful mid-development.
- For large PRs, call impact_analysis on the highest-risk changed symbols (entry points, widely-used functions).
- If scope_test returns no tests, that's a signal: the changed code may lack test coverage.
 