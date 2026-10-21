# PRD: iStratDE-Inspired Discrete Population Recursion

## Introduction

EvoHarness currently supports parallel (`agg`) and sequential (`seq`) task-level recursion. This run adds a third mode, `population` (on-disk token `pop`), inspired by iStratDE's individual-level fixed strategy diversity. The mode evolves complete Harness file trees through structured edits, deterministic component crossover, execution probes, and answer-free survivor selection.

The first version must preserve the existing modes and Harness²'s ground-truth-free optimization boundary. The implementation phase can finish locally with unit tests and a deterministic Demo. A real LAB score improvement remains a separate validation goal and is currently blocked by the absence of a Linux environment.

## References

- `README.md`: Harness components, recursion overview, installation, and safety boundary.
- `docs/EXPERIMENTS.md`: budgets, reproducibility, task partitions, judging, reporting, and network restrictions.
- `docs/BENCHMARKS.md`: LAB setup and independent judging contract.
- `harness2/core/recursion.py`: existing Propose–Probe–Compose execution and shared helpers.
- `harness2/core/tree.py`: `Tree` representation, validation, diff, and diversity helpers.
- `harness2/core/selection.py`: anonymous answer-free final selection and adoption reflection.
- `harness2/core/schedule.py`: shared base executions, mode dispatch, summaries, Libraries, and resume.
- `harness2/core/ledger.py`: rollout/session accounting.
- `harness2/core/spec.py`: `RunView` carries no score; `BenchmarkAdapter.score` is reserved for post-selection judging.
- `harness2/core/judge.py`: independent measurement after selection.
- `harness2/demo.py` and `tests/test_workflow.py`: deterministic real-Python integration harness and existing regression patterns.
- iStratDE paper: <https://arxiv.org/abs/2602.01147>.
- iStratDE reference repository: <https://github.com/EMI-Group/istratde>. Do not copy its GPL-3.0 source into this Apache-2.0 project.

## Decisions

1. The public mode name is `population`; its on-disk token is `pop`. Documentation calls it an “iStratDE-inspired discrete population mode,” not the original continuous iStratDE algorithm.
2. Existing `agg` and `seq` behavior, default mode selection, namespaces, result files, and resume behavior must remain compatible. Demo `both` continues to mean `agg+seq` only.
3. Population mode fixes `N=k` individuals and `G=2` generations. It rejects `k<2`. No extra population CLI knobs are added in this version.
4. Each slot receives a deterministic fixed strategy at initialization. Replacing a slot's Harness never changes its strategy.
5. Strategies cover parent selection (`current`, `random`, `best`, `pbest`), one or two donors, edit scope, fixed CR, and component-level binomial or exponential crossover. Strategy version 1 does not include continuous F arithmetic or arithmetic recombination.
6. A structural difference is represented by parent/donor file diffs and execution contrasts. Text is never converted into a numeric vector for arithmetic.
7. Every generation freezes its parent snapshot before parallel trial generation. Survivors change only after all trials in that generation finish.
8. Each slot receives one LLM trial per generation. Each slot also receives exactly one deterministic component-crossover trial across the two generations, distributed deterministically across disjoint extra-trial slots. Thus there are `2k` LLM trials plus `k` deterministic trials, or `3k` trial rollouts in total.
9. Including `k` shared base rollouts, population mode has `R=4k`. It records `2k` proposal sessions plus two batch comparison sessions, so `P=2k+2<=3k`; at `k=2`, `P=3k` exactly. Final selection and reflection remain outside the Ledger as in existing modes.
10. One batch selector session per generation compares each slot's parent with its one or two trials using only answer-free task and artifact evidence. It never calls `adapter.score` or imports `core.judge`.
11. A comparator failure, malformed answer, missing pair result, or invalid trial preserves the parent conservatively. Judge failure semantics remain unrelated: a missing judge result is never a zero.
12. Surrogate fitness is optimization-local lineage evidence, not a benchmark score. Initial individuals have fitness 0. A challenger accepted over its parent inherits the parent's fitness plus 1; a retained parent keeps its fitness. Best/pbest ties are resolved deterministically. Fitness is never added to `RunView`.
13. Final selection receives exactly the `k` surviving labels from `ModeResult.finals`. Because greedy selection may retain an initial parent, a `baseN` label can legally survive in `pop`; this is an explicit difference from the agg/seq composed-final pool and must be represented honestly in selection and reports.
14. Existing final selection and adoption reflection are reused. Each mode keeps a separate experience Library.
15. Resume keeps the existing complete-task semantics. Deterministic strategies, labels, run ids, and Proposer cache allow an incomplete task to restart its population procedure; generation-level checkpoint resume is out of scope.
16. `POPULATION_STRATEGY_VERSION` enters run identity only when `pop` is selected. Existing agg/seq run identity is not invalidated.
17. The implementation uses the Python standard library. JAX, PyTorch, EvoX, and copied iStratDE source are not introduced.
18. Real LAB improvement requires equal or lower resource use and a higher independently judged test score. A negative result must be reported as negative; it must never be hidden by repeated test-set tuning.

## Goals

- Add a working discrete population recursion mode with fixed per-individual strategies.
- Preserve or improve Harness quality without exceeding the parallel mode's rollout and model-session budgets.
- Keep all optimization and survivor selection free of benchmark answers, rubrics, grader verdicts, and judge scores.
- Produce auditable strategy, parent, donor, trial, survivor, fitness, cost, and lineage records.
- Preserve existing parallel/sequential behavior and complete-task resume.
- Validate the local implementation with deterministic tests and real Python Demo executions.
- When a Linux environment becomes available, run a preregistered same-budget LAB comparison and report the result honestly.

## Non-Goals

- No model-weight training or fine-tuning.
- No direct numeric differential mutation of prompt, skill, script, or other text.
- No use of benchmark judge scores during proposal, probing, survivor comparison, final selection, reflection, or Library updates.
- No adaptive strategy rewriting based on success history; individual strategy profiles remain fixed.
- No continuous F semantics, arithmetic text recombination, mid-generation checkpoint resume, or user-configurable population/generation counts in version 1.
- No JAX/PyTorch/EvoX dependency and no GPU-utilization claim.
- No change to default `parallel` behavior or the meaning of Demo `both`.
- No real LAB execution on the current macOS-only environment and no claim that local Demo success proves benchmark improvement.
- No repeated test-set tuning to manufacture a positive result.

## User Stories

### US-001: Discrete strategies and budget primitives

**Description:** As a researcher, I want deterministic population strategies and explicit budget helpers so that experiments are reproducible and auditable.

**Acceptance Criteria:**
- [x] Add `harness2/core/population.py` with fixed strategy generation, donor selection, component crossover, lineage records, and pop budget helpers.
- [x] Identical task/mode/k/version/slot inputs reproduce identical strategies across calls and generations.
- [x] Crossover returns a new Tree and never mutates an input Tree.
- [x] The deterministic extra-trial schedule assigns exactly one deterministic trial per slot across two generations and `3k` total trial rollouts.
- [x] Tests prove `R=4k`, `P=2k+2`, `P=3k` at k=2, and `P<3k` at k>2.
- [x] Focused unittest command passes.

### US-002: Answer-free batch survivor comparison

**Description:** As a researcher, I want one anonymous comparator call per generation so that survivor selection has a quality signal without using a benchmark judge.

**Acceptance Criteria:**
- [ ] Add batch comparison support in `harness2/core/selection.py` and `harness2/core/prompts.py`, with unique generation roots and session labels.
- [ ] The comparator receives anonymized task/source/artifact evidence and returns one explicit winner for every requested slot.
- [ ] Invalid trials, missing winners, parser failures, and comparator exceptions preserve the relevant parent and are recorded.
- [ ] Multi-generation calls do not delete each other's context or collide in the Proposer cache.
- [ ] Tests mock `adapter.score` and prove it is never called; comparison costs and attempts are returned for Ledger recording.
- [ ] Focused unittest command passes.

### US-003: Two-generation population recursion

**Description:** As a researcher, I want a complete population recursion implementation so that fixed strategies can evolve Harnesses under a bounded budget.

**Acceptance Criteria:**
- [ ] Add `run_population` and dispatch for `pop` in `harness2/core/recursion.py` while leaving agg/seq behavior unchanged.
- [ ] Each generation freezes parents, generates and executes all trials, then performs batch survivor selection; no thread writes survivors early.
- [ ] Record fixed strategies, parents, donors, trials, survivors, surrogate fitness, comparison outcomes, and lineage in `ModeResult`.
- [ ] Update `harness2/core/ledger.py` for mode-aware budgets and include comparison sessions, costs, and attempts.
- [ ] Tests cover budget identities, invalid proposal retry, failed-trial parent retention, exactly k final labels, and judge isolation.
- [ ] Focused unittest command passes.

### US-004: Scheduling, final selection, Library, and resume integration

**Description:** As a maintainer, I want population mode to use the existing task lifecycle so that results and experience remain consistent with other modes.

**Acceptance Criteria:**
- [ ] Update `harness2/core/schedule.py` and `harness2/core/selection.py` to pass the existing selector into population recursion and the actual final labels into final-pool construction.
- [ ] Preserve existing agg/seq call signatures and final pools; pop explicitly permits a surviving base label and records it as base.
- [ ] Chosen trees, materialized Harnesses, fingerprints, summary records, and selection records work for every possible survivor label, including `baseN`.
- [ ] Per-mode Libraries remain isolated, and restored Libraries contain only earlier tasks from the same mode.
- [ ] A complete second run adds no execute/Proposer calls; a changed adopted Harness or missing adopted artifact causes a rerun.
- [ ] Focused unittest command passes.

### US-005: CLI, run identity, and reporting

**Description:** As an experimenter, I want consistent CLI and reporting support so that population runs cannot be confused with other experiments.

**Acceptance Criteria:**
- [ ] Update mode parsing/help in `harness2/cli/_common.py`, `harness2/cli/run_sweep.py`, `harness2/cli/demo.py`, and `harness2/cli/judge.py` as applicable.
- [ ] `population` and `pop` resolve to the same token; defaults and Demo `both` do not gain a new mode.
- [ ] Population with `k<2` fails before model calls with an actionable error.
- [ ] Add an independent population strategy version to `harness2/config.py` and include it in identity only for population runs.
- [ ] Update `harness2/core/report.py` so pop finals/base survivors are classified honestly and k=1 agg/seq noise-floor logic never assumes every two-mode report contains both agg and seq.
- [ ] CLI dry-run, namespace, judge/report parsing, and old-default regression tests pass.

### US-006: Deterministic end-to-end Demo and documentation

**Description:** As a maintainer, I want a credential-free end-to-end check and accurate documentation so that the new mode can be verified without external services.

**Acceptance Criteria:**
- [ ] Extend `harness2/demo.py` so scripted strategies produce distinguishable edits and batch comparisons select valid outputs using source evidence without a benchmark judge.
- [ ] `python -m harness2.cli.demo --k 2 --mode population` completes two real-Python tasks, writes all population audit records, and resumes without new calls on repeat.
- [ ] Update `README.md` and `docs/EXPERIMENTS.md` with mode behavior, budgets, base-survivor semantics, resume boundary, citations, and judge isolation.
- [ ] Full unittest suite passes, and existing agg/seq Demo behavior still passes.
- [ ] Documentation does not claim real LAB improvement.

### US-007: Same-budget LAB comparison and independent evaluation

**Description:** As a researcher, I want a preregistered LAB comparison so that population mode is accepted only if it improves independent test performance within budget.

**Acceptance Criteria:**
- [ ] BLOCKED until a supported Linux host, LAB dependencies, model credentials/quota, executor network restrictions, and a user-approved sample/cost cap are available.
- [ ] Before execution, record the train/test split, task stream, seed, models, sample size, cost cap, and stopping rule; test results cannot be used for repeated tuning.
- [ ] Run `parallel,population` together at `k=2` on the same tasks so base executions, executor, improver, selector, judge, and network policy are shared.
- [ ] Verify actual population rollouts/sessions do not exceed parallel budgets and no judge ran during optimization.
- [ ] Run independent `harness2-judge` and `harness2-report`; save scores, ungraded rows, actual costs, timings, and failures.
- [ ] Mark this story passed only if the independent population test score exceeds parallel without exceeding the agreed budget. Otherwise leave it failed and record the negative result.

## Functional Requirements

- FR-1: `schedule.mode_token` must accept `population` and `pop`; all other existing tokens retain their meanings.
- FR-2: Population strategies must be derived from stable versioned seeds and stored in summary records.
- FR-3: Parent and donor selection must read only the frozen generation snapshot.
- FR-4: All generated Trees must pass existing Harness and adapter validation before execution; invalid proposals use existing retry/fallback behavior.
- FR-5: Deterministic crossover must operate on named Harness files/components without mutating source Trees.
- FR-6: Every trial must have a unique deterministic label and run id; duplicate Ledger labels remain an error.
- FR-7: The batch comparator must never receive grader outputs, rubrics, answer keys, or judge scores and must use existing leak checks.
- FR-8: Every requested slot must have a recorded survivor and an auditable reason, including conservative fallback reasons.
- FR-9: Population Ledger accounting must include batch comparisons and use `R=4k`, `P=2k+2`; existing agg/seq accounting remains `R=4k`, `P=3k`.
- FR-10: `ModeResult.finals` must contain exactly k surviving labels, and every final label must have a Tree, materialized Harness, and RunView.
- FR-11: Final selection, reflection, and Library behavior must not call `adapter.score`.
- FR-12: Population run identity must include its strategy version without changing identity records for agg/seq-only runs.
- FR-13: Reports must distinguish selected base survivors from non-base pop finals and must not apply agg/seq-specific warnings to pop.
- FR-14: Repeating a completed task must use existing resume behavior without new model or executor calls.
- FR-15: All experiment documentation must report actual calls and costs rather than treating equal `k` alone as proof of equal resource use.

## Validation

Run from the repository root with Python 3.12:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 \
  python -m unittest discover -s tests -p 'test_workflow.py' -v

PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 \
  python -m harness2.cli.demo --output "$(mktemp -d)/pop-demo" --k 2 --mode population

scripts/ralph/ralph.sh --agent kimi --dry-run
```

Also verify:

- Existing agg/seq budget, failure, Library, and resume tests still pass.
- The population Demo is repeated against the same output root and makes no new calls after completion.
- Summary records contain fixed strategies, complete lineage, comparison sessions, and actual costs.
- `adapter.score` is never called by optimization code.
- `parallel,population` CLI dry-run/report paths do not assume `seq` exists.
- US-007 remains unpassed while its external preconditions or positive independent result are missing.

## Risks and Open Questions

- **No Linux host:** real LAB validation is blocked. Do not substitute an unsupported local run or claim improvement from tests.
- **Model credentials and cost cap:** the Linux host, model access, sample size, and maximum approved cost still need to be supplied before US-007.
- **Selector bias:** repeated answer-free comparison can favor outputs that please the selector. Keep the selector fixed, anonymize candidates, preserve parent on uncertainty, and report this limitation.
- **Noisy executions:** a single probe can produce a lucky result. Version 1 records this limitation; repeated probe averaging requires a separately approved budget and design.
- **Budget gaming:** deterministic trials still consume executor rollouts and must be counted. Comparison sessions, retries, final selection, and reflection must remain visible in cost reports.
- **Base survivor semantics:** population can legitimately retain base; reports and docs must not treat this as an out-of-pool error.
- **Resume scope:** incomplete tasks restart the population procedure using deterministic state and caches; generation-level recovery is not promised.
- **Negative result:** if the preregistered comparison does not improve, stop and report it. Do not tune on the test split or mark US-007 passed.
