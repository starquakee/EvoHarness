# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel, sequential, and fixed-strategy discrete population recursion.

Every mode schedules 4k rollouts, including shared base executions. Invalid proposals
receive one retry before inheriting the current harness.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .. import config
from . import context, population, prompts, selection
from . import tree as harness_tree
from .ledger import Ledger
from .proposer import Proposer, ProposerError
from .spec import BenchmarkAdapter, RunView, resolve_prompts

STAGE_ATTEMPTS = 2
TRANSPORT_ATTEMPTS = config.TRANSPORT_ATTEMPTS
PHASE_WIDTH = config.PHASE_WIDTH


@dataclass
class Stage:
    """An improvement stage, including retries, its harness, and decision notes."""

    label: str
    dim: str
    tree: harness_tree.Tree
    kept_base: bool = False
    fell_back: bool = False

    error: str = ""
    usd: float = 0.0
    attempts: int = 0
    cached: bool = False
    notes: str = ""


@dataclass
class ModeResult:
    mode: str
    task_id: str
    k: int
    stages: list[Stage] = field(default_factory=list)
    trees: dict[str, harness_tree.Tree] = field(default_factory=dict)
    rollouts: dict[str, RunView] = field(default_factory=dict)
    finals: list[str] = field(default_factory=list)
    ledger: Ledger | None = None
    similarity: dict | None = None
    errors: list[str] = field(default_factory=list)
    strategies: list[population.PopulationStrategy] = field(default_factory=list)
    lineage: list[population.LineageRecord] = field(default_factory=list)
    comparisons: list[selection.BatchComparison] = field(default_factory=list)
    fitness: dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.finals) and not self.errors

    @property
    def fell_back(self) -> list[str]:
        return [s.label for s in self.stages if s.fell_back]

    def to_dict(self) -> dict:
        out = {
            "mode": self.mode,
            "task_id": self.task_id,
            "k": self.k,
            "finals": self.finals,
            "ok": self.ok,
            "fell_back": self.fell_back,
            "kept_base": [s.label for s in self.stages if s.kept_base],
            "stages": [
                {
                    "label": s.label,
                    "dim": s.dim,
                    "fell_back": s.fell_back,
                    "kept_base": s.kept_base,
                    "attempts": s.attempts,
                    "usd": s.usd,
                    "cached": s.cached,
                    "notes": s.notes,
                    "error": (s.error or "")[:400],
                }
                for s in self.stages
            ],
            "similarity": self.similarity,
            "ledger": self.ledger.to_dict() if self.ledger else None,
            "errors": self.errors,
        }
        if self.mode == "pop":
            out.update(
                {
                    "strategies": [strategy.to_dict() for strategy in self.strategies],
                    "lineage": [record.to_dict() for record in self.lineage],
                    "comparisons": [comparison.to_dict() for comparison in self.comparisons],
                    "fitness": {str(slot): value for slot, value in sorted(self.fitness.items())},
                }
            )
        return out


def _accept_tree(
    surface,
    adapter: BenchmarkAdapter,
    new: harness_tree.Tree,
    fallback: harness_tree.Tree,
    label: str,
) -> tuple[bool, str]:
    """Repair then validate a proposed tree: (kept_base, "") if accepted, else the error."""
    triggered = harness_tree.autotrigger(surface, new)
    if triggered:
        print(f"  [{label}] auto-triggered skill(s): {', '.join(triggered)}", flush=True)
    changed = harness_tree.diff_summary(surface, fallback, new)
    kept = not (changed["added"] or changed["removed"] or changed["modified"])
    try:
        harness_tree.validate(
            surface,
            new,
            allow_subagents=adapter.spec.allow_subagents,
            reserved_scripts=harness_tree.RESERVED_SCRIPTS | adapter.spec.reserved_scripts,
        )
        adapter.validate_tree(new)
        context.assert_tree_has_no_grader_verdicts(new)
    except ValueError as exc:
        return kept, str(exc)
    return kept, ""


def _run_stage(
    *,
    label: str,
    dim: str,
    job: str,
    fallback: harness_tree.Tree,
    adapter: BenchmarkAdapter,
    prop: Proposer,
    ctx_dir: Path,
    task_id: str,
    rollouts: Sequence[RunView],
    harness_dir: Path | None,
    library_dir: Path | None,
    pool: Sequence[tuple[str, Path, RunView | None]] = (),
    decisions: dict[str, str] | None = None,
    extra_validation: Callable[[harness_tree.Tree], None] | None = None,
) -> Stage:
    """Build evidence, propose an edit, and retry an invalid proposal once."""
    surface = adapter.substrate.surface()

    try:
        tree = context.build(
            ctx_dir / label,
            adapter,
            task_id,
            rollouts,
            harness_dir=harness_dir,
            library_dir=library_dir,
            pool=pool,
            decisions=decisions,
        )
        mission = context.mission(tree, job=job)
    except Exception as exc:
        return Stage(
            label=label,
            dim=dim,
            tree=dict(fallback),
            fell_back=True,
            error=f"could not build evidence: {type(exc).__name__}: {exc}"[:400],
            attempts=0,
        )

    err = ""
    spent_usd = 0.0
    spent_calls = 0
    all_cached = True
    for attempt in range(1, STAGE_ATTEMPTS + 1):
        try:
            reply = prop.call(
                mission
                + (
                    f"\n\n# YOUR PREVIOUS ATTEMPT WAS REJECTED\n\n{err}\n\n"
                    f"Fix it in `context/harness/` and reply again."
                    if err
                    else ""
                ),
                tree.root,
                f"{label}.a{attempt}",
                harvest="context/harness",
                max_attempts=TRANSPORT_ATTEMPTS,
            )
        except ProposerError as exc:
            err = str(exc)
            spent_calls += exc.attempts or TRANSPORT_ATTEMPTS
            spent_usd += exc.cost_usd
            all_cached = False
            break
        spent_usd += reply.cost_usd
        spent_calls += 0 if reply.cached else max(1, reply.attempts)
        all_cached = all_cached and reply.cached

        if reply.workspace is None:
            err = (
                "the harness directory did not come back from the session; edit the files "
                "in `context/harness/` in place"
            )
            continue
        rejected: list[str] = []
        new = harness_tree.read_tree(surface, reply.workspace, rejected=rejected)
        if not new:
            err = "`context/harness/` came back empty -- edit the files, do not delete them all"
            continue
        if rejected:
            err = (
                "these files are off the editable surface and were not applied: "
                + ", ".join(sorted(rejected)[:6])
                + ". Component names must be lowercase letters, digits, `_` or `-` "
                f"(e.g. `{surface.skill_dir}/doc_review/SKILL.md`, "
                f"`scripts/check_totals.py`)."
            )
            continue

        try:
            if extra_validation is not None:
                extra_validation(new)
        except ValueError as exc:
            err = str(exc)
        else:
            kept, err = _accept_tree(surface, adapter, new, fallback, label)
        if err:
            harness_tree.write_tree(surface, new, tree.root / "harness")
            continue
        notes = re.search(r"<notes>(.*?)</notes>", reply.text, re.S)
        return Stage(
            label=label,
            dim=dim,
            tree=new,
            kept_base=kept,
            usd=spent_usd,
            attempts=spent_calls,
            cached=all_cached,
            notes=notes.group(1).strip() if notes else reply.text.strip(),
        )

    return Stage(
        label=label,
        dim=dim,
        tree=dict(fallback),
        fell_back=True,
        error=err[:400],
        usd=spent_usd,
        attempts=spent_calls,
        cached=all_cached,
    )


def _record_sessions(ledger: Ledger, stages: Sequence[Stage]) -> None:
    """Record stages in stable order after their concurrent phase finishes."""
    for stage in stages:
        ledger.session(stage.label, stage.usd, cached=stage.cached, attempts=stage.attempts)


def _roll(
    adapter: BenchmarkAdapter,
    ledger: Ledger,
    task_id: str,
    label: str,
    tree: harness_tree.Tree | None,
    run_id_of: Callable[[str], str],
    work_dir: Path,
    *,
    record: bool = True,
) -> RunView:
    """Materialise a harness and roll it once. A dead rollout is recorded, never raised."""
    hdir = None
    if tree is not None:
        hdir = harness_tree.write_tree(
            adapter.substrate.surface(), tree, work_dir / "harnesses" / label
        )
    run_id = run_id_of(label)
    try:
        rd = adapter.execute(task_id, hdir, run_id)
        exists = Path(rd).exists()
        rv = RunView(
            label=label,
            run_id=run_id,
            run_dir=rd,
            ok=exists,
            delivered=exists and adapter.delivered(rd),
            error="" if exists else "adapter returned a missing run directory",
        )
    except Exception as exc:
        rv = RunView(
            label=label,
            run_id=run_id,
            run_dir=work_dir / "missing",
            ok=False,
            delivered=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    if record:
        ledger.rollout(label)
    return rv


def _roll_many(
    adapter: BenchmarkAdapter,
    ledger: Ledger,
    task_id: str,
    items: Sequence[tuple[str, harness_tree.Tree]],
    run_id_of,
    work_dir: Path,
    *,
    width: int,
) -> dict[str, RunView]:
    """Roll independent harnesses concurrently, then record them in the caller's order.

    The ledger is written after the pool drains, so a concurrent run records the same sequence
    of labels as a serial one.
    """
    out: dict[str, RunView] = {}
    if not items:
        return out
    n = max(1, min(len(items), width))
    if n == 1:
        for label, tree in items:
            out[label] = _roll(
                adapter, ledger, task_id, label, tree, run_id_of, work_dir, record=True
            )
        return out
    with ThreadPoolExecutor(max_workers=n) as pool:
        got = list(
            pool.map(
                lambda it: _roll(
                    adapter, ledger, task_id, it[0], it[1], run_id_of, work_dir, record=False
                ),
                items,
            )
        )
    for (label, _tree), rv in zip(items, got):
        ledger.rollout(label)
        out[label] = rv
    return out


def run_parallel(
    task_id: str,
    k: int,
    base: Sequence[RunView],
    *,
    adapter: BenchmarkAdapter,
    prop: Proposer,
    work_dir: Path,
    run_id_of: Callable[[str], str],
    base_tree: harness_tree.Tree,
    library_dir: Path | None = None,
    propose_workers: int = PHASE_WIDTH,
    rollout_workers: int = PHASE_WIDTH,
) -> ModeResult:
    """WIDTH: 2k iid candidates from shared evidence, then k INDEPENDENT compositions."""
    surface = adapter.substrate.surface()
    ledger = Ledger(mode="agg", task_id=task_id, k=k)
    for r in base:
        ledger.rollout(r.label)
    result = ModeResult(mode="agg", task_id=task_id, k=k, ledger=ledger)
    result.rollouts.update({r.label: r for r in base})
    ctx_dir = work_dir / "context"
    benchmark_prompts = resolve_prompts(adapter, task_id)
    base_dir = harness_tree.write_tree(surface, base_tree, work_dir / "harnesses" / "base")

    specs = [(f"{d.lower()}{i}", d, i) for i in range(1, k + 1) for d in ("INV", "DOM")]
    with ThreadPoolExecutor(max_workers=min(len(specs), propose_workers)) as pool_x:
        stages = list(
            pool_x.map(
                lambda s: _run_stage(
                    label=s[0],
                    dim=s[1],
                    job=prompts.propose(benchmark_prompts, s[1], surface, k=k, index=s[2]),
                    fallback=base_tree,
                    adapter=adapter,
                    prop=prop,
                    ctx_dir=ctx_dir,
                    task_id=task_id,
                    rollouts=base,
                    harness_dir=base_dir,
                    library_dir=library_dir,
                ),
                specs,
            )
        )
    _record_sessions(ledger, stages)
    for stage in stages:
        result.stages.append(stage)
        result.trees[stage.label] = stage.tree
    result.rollouts.update(
        _roll_many(
            adapter,
            ledger,
            task_id,
            [(stage.label, stage.tree) for stage in stages],
            run_id_of,
            work_dir,
            width=rollout_workers,
        )
    )

    pool_ev = [
        (stage.label, work_dir / "harnesses" / stage.label, result.rollouts.get(stage.label))
        for stage in stages
    ]
    agg_specs = [f"agg{i}" for i in range(1, k + 1)]
    with ThreadPoolExecutor(max_workers=min(len(agg_specs), propose_workers)) as pool_x:
        aggs = list(
            pool_x.map(
                lambda lbl: _run_stage(
                    label=lbl,
                    dim="AGG",
                    job=prompts.aggregate(benchmark_prompts, surface, k=k, index=int(lbl[3:])),
                    fallback=base_tree,
                    adapter=adapter,
                    prop=prop,
                    ctx_dir=ctx_dir,
                    task_id=task_id,
                    rollouts=base,
                    harness_dir=base_dir,
                    library_dir=library_dir,
                    pool=pool_ev,
                    decisions={stage.label: stage.notes or stage.error for stage in stages},
                ),
                agg_specs,
            )
        )
    _record_sessions(ledger, aggs)
    for stage in aggs:
        result.stages.append(stage)
        result.trees[stage.label] = stage.tree
        result.finals.append(stage.label)
    result.rollouts.update(
        _roll_many(
            adapter,
            ledger,
            task_id,
            [(stage.label, stage.tree) for stage in aggs],
            run_id_of,
            work_dir,
            width=rollout_workers,
        )
    )

    result.similarity = harness_tree.diversity({s.label: s.tree for s in aggs})
    ledger.assert_identity()
    return result


def run_sequential(
    task_id: str,
    k: int,
    base: Sequence[RunView],
    *,
    adapter: BenchmarkAdapter,
    prop: Proposer,
    work_dir: Path,
    run_id_of: Callable[[str], str],
    base_tree: harness_tree.Tree,
    library_dir: Path | None = None,
    propose_workers: int = 2,
    rollout_workers: int = PHASE_WIDTH,
) -> ModeResult:
    """DEPTH: k nested passes over the same shared base evidence."""
    surface = adapter.substrate.surface()
    ledger = Ledger(mode="seq", task_id=task_id, k=k)
    for r in base:
        ledger.rollout(r.label)
    result = ModeResult(mode="seq", task_id=task_id, k=k, ledger=ledger)
    result.rollouts.update({r.label: r for r in base})
    ctx_dir = work_dir / "context"
    benchmark_prompts = resolve_prompts(adapter, task_id)

    current_harness = dict(base_tree)
    history: list[tuple[str, Path, RunView | None]] = []
    for t in range(1, k + 1):
        current_label = f"c{t - 1}" if t > 1 else "base"
        current_dir = harness_tree.write_tree(
            surface, current_harness, work_dir / "harnesses" / current_label
        )
        specs = [(f"inv{t}", "INV"), (f"dom{t}", "DOM")]
        with ThreadPoolExecutor(max_workers=min(2, propose_workers)) as pool_x:
            stages = list(
                pool_x.map(
                    lambda s: _run_stage(
                        label=s[0],
                        dim=s[1],
                        job=prompts.propose(
                            benchmark_prompts,
                            s[1],
                            surface,
                            k=k,
                            index=t,
                            round_t=(t if k > 1 else None),
                        ),
                        fallback=current_harness,
                        adapter=adapter,
                        prop=prop,
                        ctx_dir=ctx_dir,
                        task_id=task_id,
                        rollouts=base,
                        harness_dir=current_dir,
                        library_dir=library_dir,
                        pool=list(history),
                        decisions={
                            stage.label: stage.notes or stage.error for stage in result.stages
                        },
                    ),
                    specs,
                )
            )
        _record_sessions(ledger, stages)
        for stage in stages:
            result.stages.append(stage)
            result.trees[stage.label] = stage.tree
        result.rollouts.update(
            _roll_many(
                adapter,
                ledger,
                task_id,
                [(stage.label, stage.tree) for stage in stages],
                run_id_of,
                work_dir,
                width=rollout_workers,
            )
        )

        round_pool = list(history) + [
            (stage.label, work_dir / "harnesses" / stage.label, result.rollouts.get(stage.label))
            for stage in stages
        ]
        ck = f"c{t}"

        comp_job = (
            prompts.aggregate(benchmark_prompts, surface, k=1, index=1)
            if k == 1
            else prompts.compose_round(benchmark_prompts, surface, t=t, k=k)
        )
        comp = _run_stage(
            label=ck,
            dim="COMPOSE",
            job=comp_job,
            fallback=current_harness,
            adapter=adapter,
            prop=prop,
            ctx_dir=ctx_dir,
            task_id=task_id,
            rollouts=base,
            harness_dir=current_dir,
            library_dir=library_dir,
            pool=round_pool,
            decisions={stage.label: stage.notes or stage.error for stage in result.stages},
        )
        _record_sessions(ledger, [comp])
        result.stages.append(comp)
        result.trees[ck] = comp.tree
        result.rollouts[ck] = _roll(adapter, ledger, task_id, ck, comp.tree, run_id_of, work_dir)

        history = round_pool + [(ck, work_dir / "harnesses" / ck, result.rollouts[ck])]
        current_harness = comp.tree

    result.finals = [f"c{index}" for index in range(1, k + 1)]
    ledger.assert_identity()
    return result


@dataclass(frozen=True)
class _PopulationPlan:
    strategy: population.PopulationStrategy
    parent_slot: int
    parent: str
    donors: tuple[str, ...]
    parent_fitness: int
    llm_label: str
    deterministic_label: str = ""

    @property
    def slot(self) -> int:
        return self.strategy.slot

    @property
    def trials(self) -> tuple[str, ...]:
        return tuple(
            label for label in (self.llm_label, self.deterministic_label) if label
        )


def run_population(
    task_id: str,
    k: int,
    base: Sequence[RunView],
    *,
    adapter: BenchmarkAdapter,
    prop: Proposer,
    work_dir: Path,
    run_id_of: Callable[[str], str],
    base_tree: harness_tree.Tree,
    library_dir: Path | None = None,
    selector: Proposer | None = None,
    propose_workers: int = PHASE_WIDTH,
    rollout_workers: int = PHASE_WIDTH,
    strategy_version: str | None = None,
) -> ModeResult:
    """Run two frozen generations with one LLM trial per slot and k crossover trials."""
    k = int(k)
    strategy_version = str(
        config.POPULATION_STRATEGY_VERSION if strategy_version is None else strategy_version
    )
    strategies = population.strategies_for(task_id, "pop", k, strategy_version)
    expected_base = [f"base{slot}" for slot in range(1, k + 1)]
    if [rollout.label for rollout in base] != expected_base:
        raise ValueError(f"population mode needs ordered base rollouts {expected_base}")

    surface = adapter.substrate.surface()
    work_dir = Path(work_dir)
    ledger = Ledger(mode="pop", task_id=task_id, k=k)
    result = ModeResult(
        mode="pop",
        task_id=task_id,
        k=k,
        ledger=ledger,
        strategies=list(strategies),
        lineage=population.initial_lineage(k),
        fitness={slot: 0 for slot in range(1, k + 1)},
    )
    for rollout in base:
        ledger.rollout(rollout.label)
        result.rollouts[rollout.label] = rollout
        result.trees[rollout.label] = dict(base_tree)
        harness_tree.write_tree(
            surface, base_tree, work_dir / "harnesses" / rollout.label
        )

    benchmark_prompts = resolve_prompts(adapter, task_id)
    comparator = selector or prop
    deterministic_slots = population.deterministic_trial_schedule(
        task_id, "pop", k, strategy_version
    )
    snapshot = {slot: base[slot - 1].label for slot in range(1, k + 1)}
    fitness = {slot: 0 for slot in range(1, k + 1)}

    for generation in (1, 2):
        parent_snapshot = dict(snapshot)
        fitness_snapshot = dict(fitness)
        generation_trees = {
            label: dict(result.trees[label]) for label in set(parent_snapshot.values())
        }
        generation_rollouts = dict(result.rollouts)
        plans = []
        for strategy in strategies:
            parent_slot, donor_slots = population.choose_parent_and_donor_slots(
                strategy, parent_snapshot, fitness_snapshot, generation=generation
            )
            llm_label = f"g{generation}s{strategy.slot}"
            deterministic_label = (
                llm_label + "x" if strategy.slot in deterministic_slots[generation] else ""
            )
            plans.append(
                _PopulationPlan(
                    strategy=strategy,
                    parent_slot=parent_slot,
                    parent=parent_snapshot[parent_slot],
                    donors=tuple(parent_snapshot[slot] for slot in donor_slots),
                    parent_fitness=fitness_snapshot[parent_slot],
                    llm_label=llm_label,
                    deterministic_label=deterministic_label,
                )
            )

        def run_llm(plan: _PopulationPlan) -> Stage:
            parent_tree = generation_trees[plan.parent]
            evidence_labels = tuple(dict.fromkeys((plan.parent, *plan.donors)))
            evidence = [generation_rollouts[label] for label in evidence_labels]
            pool = [
                (
                    label,
                    work_dir / "harnesses" / label,
                    generation_rollouts.get(label),
                )
                for label in evidence_labels
            ]
            return _run_stage(
                label=plan.llm_label,
                dim="POP",
                job=prompts.population_propose(
                    benchmark_prompts,
                    surface,
                    strategy=plan.strategy,
                    generation=generation,
                    parent=plan.parent,
                    donors=plan.donors,
                ),
                fallback=parent_tree,
                adapter=adapter,
                prop=prop,
                ctx_dir=work_dir / "context",
                task_id=task_id,
                rollouts=evidence,
                harness_dir=work_dir / "harnesses" / plan.parent,
                library_dir=library_dir,
                pool=pool,
                extra_validation=lambda new: population.validate_edit_scope(
                    surface, parent_tree, new, plan.strategy
                ),
            )

        with ThreadPoolExecutor(
            max_workers=min(len(plans), propose_workers)
        ) as pool_x:
            stages = list(pool_x.map(run_llm, plans))
        _record_sessions(ledger, stages)
        result.stages.extend(stages)

        trial_trees = {
            plan.llm_label: stage.tree for plan, stage in zip(plans, stages)
        }
        invalid: dict[str, str] = {}
        for plan in plans:
            if not plan.deterministic_label:
                continue
            parent_tree = generation_trees[plan.parent]
            crossed = dict(parent_tree)
            try:
                crossed = population.component_crossover(
                    surface,
                    parent_tree,
                    [generation_trees[label] for label in plan.donors],
                    plan.strategy,
                    generation=generation,
                )
                population.validate_edit_scope(
                    surface, parent_tree, crossed, plan.strategy
                )
                _kept, error = _accept_tree(
                    surface,
                    adapter,
                    crossed,
                    parent_tree,
                    plan.deterministic_label,
                )
                if error:
                    raise ValueError(error)
            except ValueError as exc:
                invalid[plan.deterministic_label] = str(exc)[:400]
            trial_trees[plan.deterministic_label] = crossed

        result.trees.update(trial_trees)
        roll_items = [
            (label, trial_trees[label])
            for plan in plans
            for label in plan.trials
            if label not in invalid
        ]
        rolled = _roll_many(
            adapter,
            ledger,
            task_id,
            roll_items,
            run_id_of,
            work_dir,
            width=rollout_workers,
        )
        for label, error in invalid.items():
            harness_tree.write_tree(
                surface, trial_trees[label], work_dir / "harnesses" / label
            )
            rolled[label] = RunView(
                label=label,
                run_id=run_id_of(label),
                run_dir=work_dir / "missing" / label,
                ok=False,
                delivered=False,
                error=f"invalid deterministic trial: {error}",
            )
            ledger.rollout(label)
        result.rollouts.update(rolled)

        comparison = selection.run_batch_comparison(
            task_id,
            k,
            generation,
            [
                selection.SlotCandidates(plan.slot, plan.parent, plan.trials)
                for plan in plans
            ],
            result.rollouts,
            adapter=adapter,
            prop=comparator,
            work_dir=work_dir,
        )
        ledger.session(
            comparison.session,
            comparison.usd,
            cached=comparison.cached,
            attempts=comparison.attempts,
        )
        result.comparisons.append(comparison)

        outcomes = {outcome.slot: outcome for outcome in comparison.outcomes}
        next_snapshot: dict[int, str] = {}
        next_fitness: dict[int, int] = {}
        generation_lineage = []
        for plan in plans:
            outcome = outcomes[plan.slot]
            accepted = outcome.accepted
            next_snapshot[plan.slot] = outcome.survivor
            next_fitness[plan.slot] = population.survivor_fitness(
                plan.parent_fitness, accepted
            )
            generation_lineage.append(
                population.LineageRecord(
                    slot=plan.slot,
                    generation=generation,
                    parent=plan.parent,
                    donors=plan.donors,
                    trial=plan.llm_label,
                    trial_kind="llm",
                    deterministic_trial=plan.deterministic_label,
                    survivor=outcome.survivor,
                    fitness=next_fitness[plan.slot],
                    outcome=outcome.outcome,
                    reason=outcome.reason or outcome.error,
                )
            )
        result.lineage.extend(generation_lineage)
        snapshot = next_snapshot
        fitness = next_fitness
        result.fitness = dict(fitness)

    result.finals = [snapshot[slot] for slot in range(1, k + 1)]
    ledger.assert_identity()
    return result


def run_mode(mode: str, *a, **kw) -> ModeResult:
    """Dispatch on the on-disk mode token: agg, seq, or pop."""
    if mode == "agg":
        return run_parallel(*a, **kw)
    if mode == "seq":
        return run_sequential(*a, **kw)
    if mode == "pop":
        return run_population(*a, **kw)
    raise ValueError(
        f"unknown mode {mode!r}; expected 'agg' (parallel), 'seq' (sequential), "
        f"or 'pop' (population)"
    )
