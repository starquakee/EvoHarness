# Ralph Agent Instructions

You are an autonomous coding agent working in EvoHarness. Implement exactly one story per iteration.

## Required Reading

1. Read `tasks/prd-istratde-population.md`, root `prd.json`, and `progress.txt`.
2. Read `README.md`, `docs/EXPERIMENTS.md`, and the source files named by the current story.
3. Treat the PRD's Decisions, Non-Goals, Validation, and Risks as hardened. Do not reopen them unless the implementation proves a contradiction; record any contradiction instead of silently changing scope.

## Story Loop

1. Check that the current branch is `ralph/istratde-population`. If not, create or check out that branch from the current branch without discarding changes.
2. Select the lowest-priority-number story with `passes: false`. Do not implement a later story opportunistically.
3. If the next story is US-007 and any external precondition is missing, stop and report the blocker. Do not run a real benchmark, mark the story passed, or emit the completion token.
4. Implement only the selected story, following existing Python naming, dataclass, concurrency, and test conventions.
5. Run the required validation. Do not mark a story passed while any required check fails.
6. Update the corresponding checkboxes in `tasks/prd-istratde-population.md` and set only that story's `passes` to `true` in root `prd.json` after validation succeeds.
7. Append an entry to `progress.txt` with implementation, files, exact validation evidence, and durable gotchas. Record failures honestly.
8. Commit the completed story with `feat: [Story ID] - [Story Title]`, then push the current branch to `origin`. Do not commit broken code, secrets, `.env`, credentials, run artifacts, temporary homes, or logs containing sensitive data.

## Hard Constraints

- Preserve existing `agg` and `seq` behavior, defaults, namespaces, and resume semantics.
- Population mode uses `N=k`, `G=2`, `k>=2`; strategies remain fixed per slot across generations.
- Population budgets are `R=4k` and `P=2k+2`; batch comparisons count as sessions and deterministic trials count as rollouts.
- Optimization and survivor selection must never call `adapter.score`, import `core.judge`, read rubrics, or use grader verdicts.
- Comparator uncertainty preserves the parent. A `baseN` survivor is legal in population mode and must be reported honestly.
- Do not add JAX, PyTorch, EvoX, copied GPL iStratDE code, or unrelated dependencies.
- Do not change continuous F into a pretend text operation; continuous F arithmetic and arithmetic recombination are out of scope.
- Do not add mid-generation checkpoint recovery or extra population CLI knobs.
- Preserve unrelated user changes. Do not modify third_party source for this feature.
- Do not claim real LAB improvement from unit tests or the deterministic Demo.

## Required Validation

The repository has no separate static typechecker configured. For the PRD's typecheck criterion, use Python 3.12 import/compile validation for touched modules in addition to tests; do not introduce a new typechecking dependency.

Run the full focused suite after each story because it is small and covers shared budget/resume behavior:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 \
  python -m unittest discover -s tests -p 'test_workflow.py' -v
```

Run narrower tests first when useful. For US-006, also run and repeat the real-Python Demo:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 \
  python -m harness2.cli.demo --output "$(mktemp -d)/pop-demo" --k 2 --mode population
```

Use a persistent temporary output root for the repeat check and verify that the second run adds no execute/Proposer calls. Inspect summary JSON for strategy, lineage, comparison, budget, and cost records.

## Progress Format

```text
## [Date/Time] - [Story ID]
- Implemented ...
- Files changed:
  - `path/to/file`
- Validation:
  - `command` — result
- **Learnings for future iterations:**
  - Durable gotchas or reusable patterns
---
```

## Stop Condition

After completing a story, check `prd.json`. Emit `<promise>COMPLETE</promise>` on its own line only when every story, including the real LAB comparison, is genuinely complete and validated. In the current environment US-007 is blocked, so do not emit that token. If stories remain, end normally after committing and pushing the completed story.
