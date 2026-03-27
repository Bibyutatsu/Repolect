---
name: repolect-refactoring
description: Use when renaming, extracting, restructuring, or making safe changes to existing code architecture.
globs:
alwaysApply: false
---
 
# Refactoring with Repolect
 
## When to Use
 
- Renaming a function, class, or variable across files
- Extracting code into a new module or function
- Restructuring architecture (moving files, splitting classes)
- Any change where breaking dependencies is the main risk
 
## Workflow
 
```
Step 1: Understand what depends on it
  impact_analysis(node_id="...", max_hops=3)
  → Full blast radius: what breaks if you change this symbol
 
Step 2: Match local style
  get_conventions(node_id="...")
  → Ensure refactored code matches the conventions of the target area
 
Step 3: For renames — get the full reference map
  rename(old_name="parse_file", new_name="analyze_source")
  → Every reference across all files, tagged by confidence:
    GRAPH+TEXT (high confidence), GRAPH-ONLY (structural), TEXT-ONLY (review carefully)
 
Step 4: Implement the refactoring
  Use get_node() on each affected file for full context before editing.
 
Step 5: Verify nothing broke
  scope_test(node_ids="<all changed node IDs>")
  → Minimal test set covering your changes
 
Step 6: Pre-commit check
  diff_analysis(with_impact=True)
  → Map all changes to affected symbols + downstream blast radius
```
 
## Key Principles
 
- ALWAYS check impact_analysis before refactoring — know the blast radius first.
- The rename tool is a plan, not an executor — it shows all references, you make the edits.
- GRAPH+TEXT references are safe to rename mechanically. TEXT-ONLY references need human review (may be in comments or strings).
- get_conventions ensures the refactored code fits the destination area's patterns.
- diff_analysis is the final safety net — run it before committing.
 
## Pro Tips
 
- For "extract function" refactors: use impact_analysis on the code being extracted, then get_conventions on the target file.
- scope_test accepts comma-separated node_ids — pass all changed symbols at once for the combined test set.
- graph_query can answer structural questions during refactoring: "MATCH (a)-[:CALLS]->(b) WHERE b.title = 'old_name' RETURN a.title, a.file_path"
 