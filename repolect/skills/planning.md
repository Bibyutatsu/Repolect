---
name: repolect-planning
description: Use before implementing any feature, change, or multi-file modification. Always call plan_change before writing code.
globs:
alwaysApply: false
---
 
# Planning Changes with Repolect
 
## When to Use
 
- Before implementing any feature or change
- User asks to add, modify, or refactor something
- Any task that touches more than one file
- Starting a non-trivial coding task
 
## Workflow
 
```
Step 1: Plan the change
  plan_change(description="Add rate limiting to API endpoints")
  → Structured plan: ADD (new files), MODIFY (ordered), READ_ONLY (context), TEST_AFTER (tiered)
 
Step 2: Find a template
  find_similar(description="a middleware that intercepts requests")
  → Best matching implementation + source + "How to adapt" guidance
 
Step 3: Match local style
  get_conventions(node_id="<node_id from plan>")
  → Error handling, naming, imports, architecture patterns for that area
 
Step 4: Implement
  Follow the MODIFY list in order. Use get_node() on each file for full context.
 
Step 5: Verify
  scope_test(node_ids="<changed node IDs>")
  → Minimal test set: MUST RUN (direct dependents) + SHOULD RUN (transitive)
```
 
## Key Principles
 
- ALWAYS call plan_change first — it replaces 5-8 rounds of search/read/explore with one structured roadmap.
- The MODIFY list is ordered by suggested implementation sequence — follow it.
- find_similar shows what to copy as-is vs what to replace — saves time on boilerplate.
- get_conventions ensures your code matches the local style, not generic patterns.
- scope_test after implementation tells you exactly which tests to run, not "run everything."
 
## Pro Tips
 
- plan_change output includes `node_id`s — chain directly into get_node or get_conventions.
- find_similar accepts a `kind` filter: `"function"`, `"class"`, `"method"`, `"test"`, `"file"`.
- TEST_AFTER from plan_change includes both MUST RUN (directly affected) and SHOULD RUN (indirectly affected).
- For renaming, use the dedicated `rename` tool instead — it handles multi-file reference tracking.
 