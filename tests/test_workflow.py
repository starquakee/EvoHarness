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

"""Focused regressions for the paper's execution, evidence, and adoption contracts."""

import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.cli import _common as cli_common
from harness2.cli import demo as demo_cli
from harness2.cli import judge as judge_cli
from harness2.cli import report as report_cli
from harness2.cli import run_sweep
from harness2.core import context, judge, population, recursion, report, runmark, schedule, selection, tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import RunView, resolve_channel_model
from harness2.demo import BASE_SCRIPT, DemoAdapter, DemoProposer


def _comparison_rollout(root, label, total, *, ok=True, delivered=True):
    run_dir = root / "comparison-rollouts" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    if delivered:
        (run_dir / "answer.txt").write_text(f"Total: {total}\n", encoding="utf-8")
    else:
        (run_dir / "answer.txt").unlink(missing_ok=True)
    (run_dir / "trajectory.md").write_text(
        f"[OBSERVATION]\nTotal: {total}\n" if ok else "[ERROR] execution failed\n",
        encoding="utf-8",
    )
    return RunView(
        label=label,
        run_id=f"run-{label}",
        run_dir=run_dir,
        ok=ok,
        delivered=delivered,
        error="" if ok else "execution failed",
    )


class _BatchProposer(DemoProposer):
    def __init__(self, spec):
        super().__init__(spec)
        self.batch_calls = []
        self.responder = None

    def source_winners(self, mission, context_dir):
        groups = re.findall(r"^slot (\d+): ([A-Z, ]+)$", mission, re.M)
        winners = []
        for slot, candidate_ids in groups:
            candidates = candidate_ids.split(", ")
            winner = next(
                (
                    candidate_id
                    for candidate_id in candidates
                    if "Total: 85.15"
                    in (context_dir / "candidates" / f"{candidate_id}.md").read_text()
                ),
                candidates[0],
            )
            winners.append((int(slot), winner))
        return winners

    def call(self, mission, context_dir, label, **kwargs):
        if not label.startswith("compare."):
            return super().call(mission, context_dir, label, **kwargs)
        context_dir = Path(context_dir)
        self.batch_calls.append((label, mission, context_dir))
        if self.responder is not None:
            text = self.responder(mission, context_dir)
        else:
            text = "\n".join(
                f'<winner slot="{slot}" choice="{choice}">The total matches the source.</winner>'
                for slot, choice in self.source_winners(mission, context_dir)
            )
        reply = Reply(label, text, True, cost_usd=1.25, attempts=2)
        self._calls.append(reply)
        return reply


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root)
        self.proposer = DemoProposer(self.adapter.spec.channels["proposer"])

    def run_stream(self, k=2, proposer=None):
        return schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            k,
            adapter=self.adapter,
            prop=proposer or self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library" / str(k),
        )

    def test_recursion_budget_and_resume(self):
        with patch.dict(config.PROPOSER_MODEL_BY_CHANNEL, {"opencode": "", "claude_cli": ""}):
            unresolved = resolve_channel_model(None, None)
            self.assertEqual(unresolved.model, "")
            with self.assertRaises(ValueError):
                Proposer(unresolved)
            with self.assertRaises(ValueError):
                resolve_channel_model("opencode", "")
        for k in (1, 2):
            with self.subTest(k=k):
                result = self.run_stream(k)
                self.assertEqual((result["done"], result["failed"]), (2, 0))
                for task_id in self.adapter.tasks():
                    summary = json.loads(
                        (
                            self.adapter.spec.runs_root
                            / f"demo-oc-agg{k}"
                            / task_id
                            / "summary.json"
                        ).read_text()
                    )
                    for mode in ("agg", "seq"):
                        ledger = summary["modes"][mode]["ledger"]
                        self.assertEqual((ledger["R"], ledger["P"]), (4 * k, 3 * k))
                        selection = summary["selections"][mode]
                        expected = "Total: 85.15\n" if task_id == "ledger-1" else "Total: 20.25\n"
                        self.assertEqual(
                            self.adapter.deliverable_text(Path(selection["chosen_run_dir"])),
                            expected,
                        )
                executions, calls = self.adapter.execution_count, len(self.proposer._calls)
                resumed = self.run_stream(k)
                self.assertEqual(
                    (self.adapter.execution_count, len(self.proposer._calls)), (executions, calls)
                )
                self.assertEqual(
                    [row["status"] for row in resumed["tasks"]], ["skipped", "skipped"]
                )
        self.assertFalse(
            any(
                reply.label.endswith(".agg1") or reply.label.endswith(".seq1")
                for reply in self.proposer._calls
                if reply.label.startswith("select.")
            )
        )
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        chosen = json.loads(summary_path.read_text())["selections"]["seq"]
        (work / "harnesses" / chosen["chosen_label"] / "scripts/summarize.py").write_text(
            "changed after selection\n"
        )
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])
        chosen = json.loads(summary_path.read_text())["selections"]["agg"]
        (Path(chosen["chosen_run_dir"]) / "answer.txt").unlink()
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])

    def test_complete_evidence_and_library_prefix(self):
        (self.adapter.task_files("ledger-1") / "reference_answer.txt").write_text('"pass_rate": 1')
        self.run_stream()
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        reflection = work / "reflection/seq2"
        required = [
            "source/ledger.csv",
            "rollouts/base1.md",
            "rollouts/c2.md",
            "pool/inv1/harness.diff",
            "decisions/c1.md",
            "adoption.json",
            "base_harness/scripts/summarize.py",
            "harness/scripts/summarize.py",
        ]
        self.assertEqual([name for name in required if not (reflection / name).is_file()], [])
        self.assertTrue((work / "context/inv2/pool/c1/rollout.md").is_file())
        self.assertEqual(list(work.rglob("reference_answer.txt")), [])
        next_context = work.parent / "ledger-2/context/inv1/library/harnesses"
        self.assertEqual([path.name for path in next_context.iterdir()], ["ledger-1"])
        with self.assertRaises(ValueError):
            context.assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": '"pass_rate": 1'})
        previous_sessions = []

        def run_opencode(command, **kwargs):
            database = Path(kwargs["env"]["XDG_DATA_HOME"]) / "opencode.db"
            previous_sessions.append(database.exists())
            database.parent.mkdir(parents=True, exist_ok=True)
            database.write_text("private evidence from this session")
            return Mock(stdout='{"type":"text","part":{"text":"done"}}', stderr="", returncode=0)

        proposer = Proposer(self.proposer.spec)
        with (
            patch("harness2.core.proposer._opencode_dir", return_value=self.root),
            patch("harness2.core.proposer.subprocess.run", side_effect=run_opencode),
        ):
            for index in range(2):
                proposer.call("Read this task only", None, f"isolation{index}")
        self.assertEqual(previous_sessions, [False, False])

    def test_invalid_proposal_retries_with_feedback_and_accumulates_cost(self):
        class RepairProposer(DemoProposer):
            feedback = ""

            def call(self, mission, context_dir, label, **kwargs):
                script = Path(context_dir) / "harness/scripts/summarize.py"
                if label.endswith(".a1"):
                    script.write_text("def broken(:\n")
                    return Reply(
                        label,
                        "invalid edit",
                        True,
                        cost_usd=1,
                        attempts=1,
                        workspace=script.parent.parent,
                    )
                self.feedback = mission
                script.write_text(BASE_SCRIPT)
                reply = super().call(mission, context_dir, label, **kwargs)
                reply.cost_usd = 2
                return reply

        proposer = RepairProposer(self.proposer.spec)
        stage = recursion._run_stage(
            label="inv1",
            dim="INV",
            job="Repair the script",
            adapter=self.adapter,
            prop=proposer,
            ctx_dir=self.root / "context",
            task_id="ledger-1",
            rollouts=[],
            fallback=tree.read_tree(self.adapter.substrate.surface(), self.adapter.base_harness()),
            harness_dir=self.adapter.base_harness(),
            library_dir=None,
        )
        self.assertEqual((stage.fell_back, stage.attempts, stage.usd), (False, 2, 3))
        self.assertIn("PREVIOUS ATTEMPT WAS REJECTED", proposer.feedback)

    def test_no_entry_stays_out_of_library_on_resume(self):
        class NoExperience(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("experience."):
                    if label.endswith("seq1"):
                        raise ProposerError("reflection failed", attempts=2, cost_usd=2)
                    return Reply(label, "NO_ENTRY", True, cost_usd=1)
                return super().call(mission, context_dir, label, **kwargs)

        proposer = NoExperience(self.proposer.spec)
        self.run_stream(k=1, proposer=proposer)
        self.run_stream(k=1, proposer=proposer)
        for mode in ("agg", "seq"):
            library = Library(
                self.root / "library/1" / mode, "accounting", self.adapter.substrate.surface()
            )
            self.assertEqual(library.entries(), [])
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg1/ledger-1/summary.json"
        selections = json.loads(summary_path.read_text())["selections"]
        self.assertEqual([selections[mode]["usd"] for mode in ("agg", "seq")], [1, 2])

    def test_failed_selection_is_never_judged(self):
        class FailedSelector(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("select."):
                    raise ProposerError("selection failed", attempts=2, cost_usd=3)
                return super().call(mission, context_dir, label, **kwargs)

        self.run_stream(proposer=FailedSelector(self.proposer.spec))
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertEqual([summary["selections"][mode]["usd"] for mode in ("agg", "seq")], [3, 3])
        self.adapter.score = Mock(return_value={"n_passed": 1, "n_criteria": 1})
        rows = judge.judge_selections(
            self.adapter,
            list(summary["selections"].values()),
        )
        self.adapter.score.assert_not_called()
        self.assertEqual((rows[0].graded, rows[0].n_passed), (False, None))

    def test_population_strategies_and_donor_selection_are_deterministic(self):
        for k in (2, 5):
            strategies = population.strategies_for("ledger-1", "pop", k, "strategy-v1")
            self.assertEqual(strategies, population.strategies_for("ledger-1", "pop", k, "strategy-v1"))
            self.assertEqual([strategy.slot for strategy in strategies], list(range(1, k + 1)))
            for strategy in strategies:
                for _generation in (1, 2):
                    self.assertEqual(
                        population.strategy_for("ledger-1", "pop", k, "strategy-v1", strategy.slot),
                        strategy,
                    )
                self.assertIn(strategy.parent_selection, population.PARENT_SELECTIONS)
                self.assertIn(strategy.edit_scope, population.EDIT_SCOPES)
                self.assertIn(strategy.crossover, population.CROSSOVER_KINDS)
                self.assertIn(strategy.cr, population.CR_VALUES)
                self.assertIn(strategy.donor_count, (1,) if k == 2 else (1, 2))

        base = population.strategy_for("ledger-1", "pop", 5, "strategy-v1", 2)
        snapshot = {slot: f"base{slot}" for slot in range(1, 6)}
        fitness = {1: 0, 2: 3, 3: 1, 4: 7, 5: 7}

        def profile(**changes):
            return population.PopulationStrategy(**(base.to_dict() | changes))

        current = profile(parent_selection="current", donor_count=2)
        selected = population.choose_parent_and_donors(
            current, snapshot, fitness, generation=1
        )
        self.assertEqual(selected, population.choose_parent_and_donors(
            current, snapshot, fitness, generation=1
        ))
        parent, donors = selected
        self.assertEqual(parent, snapshot[current.slot])
        self.assertEqual(len(donors), 2)
        self.assertEqual(len(set(donors)), 2)
        self.assertNotIn(parent, donors)

        best = profile(parent_selection="best")
        parent, _ = population.choose_parent_and_donors(best, snapshot, fitness, generation=2)
        self.assertEqual(parent, "base4")
        for selection in ("random", "pbest"):
            strategy = profile(parent_selection=selection)
            first = population.choose_parent_and_donors(strategy, snapshot, fitness, generation=2)
            second = population.choose_parent_and_donors(strategy, snapshot, fitness, generation=2)
            self.assertEqual(first, second)

    def test_population_crossover_copies_components_without_mutating_inputs(self):
        surface = self.adapter.substrate.surface()
        parent = {
            "systemprompt.md": "parent prompt",
            "scripts/summarize.py": "parent summary",
            "scripts/other.py": "parent other",
        }
        donor = {
            "systemprompt.md": "donor prompt",
            "scripts/summarize.py": "donor summary",
            "scripts/other.py": "donor other",
        }
        parent_before, donor_before = dict(parent), dict(donor)
        base = population.strategy_for("ledger-1", "pop", 2, "strategy-v1", 1)

        for kind in population.CROSSOVER_KINDS:
            strategy = population.PopulationStrategy(
                **(base.to_dict() | {"crossover": kind, "edit_scope": "all"})
            )
            crossed = population.component_crossover(
                surface, parent, [donor], strategy, generation=1
            )
            self.assertIsNot(crossed, parent)
            self.assertNotEqual(crossed, parent)
            self.assertEqual(
                crossed,
                population.component_crossover(surface, parent, [donor], strategy, generation=1),
            )
            self.assertEqual(parent, parent_before)
            self.assertEqual(donor, donor_before)

        scoped = population.PopulationStrategy(
            **(base.to_dict() | {"edit_scope": "scripts", "crossover": "binomial"})
        )
        crossed = population.component_crossover(surface, parent, [donor], scoped, generation=1)
        self.assertEqual(crossed["systemprompt.md"], parent["systemprompt.md"])

    def test_population_schedule_lineage_and_budget_identities(self):
        for k in (2, 3, 6):
            schedule = population.deterministic_trial_schedule(
                "ledger-1", "pop", k, "strategy-v1"
            )
            self.assertEqual(
                schedule,
                population.deterministic_trial_schedule("ledger-1", "pop", k, "strategy-v1"),
            )
            self.assertEqual(sorted(schedule), [1, 2])
            self.assertEqual(
                sorted(slot for slots in schedule.values() for slot in slots),
                list(range(1, k + 1)),
            )
            self.assertEqual(population.expected_trial_rollouts(k), 3 * k)
            self.assertEqual(population.expected_rollouts(k), 4 * k)
            self.assertEqual(population.expected_sessions(k), 2 * k + 2)
            if k == 2:
                self.assertEqual(population.expected_sessions(k), 3 * k)
            else:
                self.assertLess(population.expected_sessions(k), 3 * k)
            initial = population.initial_lineage(k)
            self.assertEqual(
                [record.survivor for record in initial],
                [f"base{i}" for i in range(1, k + 1)],
            )
            self.assertEqual([record.fitness for record in initial], [0] * k)
            json.dumps([record.to_dict() for record in initial])

        self.assertEqual(population.survivor_fitness(4, False), 4)
        self.assertEqual(population.survivor_fitness(4, True), 5)
        with self.assertRaisesRegex(ValueError, "k >= 2"):
            population.expected_rollouts(1)

    def _fixed_population_strategies(self, k):
        return tuple(
            population.PopulationStrategy(
                **(
                    strategy.to_dict()
                    | {
                        "parent_selection": "current",
                        "donor_count": 1,
                        "edit_scope": "scripts",
                    }
                )
            )
            for strategy in population.strategies_for("ledger-1", "pop", k, "1")
        )

    def _population_result(self, proposer, k=2):
        task_id = "ledger-1"
        base = []
        for slot in range(1, k + 1):
            label = f"base{slot}"
            run_id = f"population-{k}/{label}"
            run_dir = self.adapter.execute(task_id, None, run_id)
            base.append(
                RunView(
                    label=label,
                    run_id=run_id,
                    run_dir=run_dir,
                    ok=True,
                    delivered=self.adapter.delivered(run_dir),
                )
            )
        base_tree = tree.read_tree(
            self.adapter.substrate.surface(), self.adapter.base_harness()
        )
        return recursion.run_mode(
            "pop",
            task_id,
            k,
            base,
            adapter=self.adapter,
            prop=proposer,
            selector=proposer,
            work_dir=self.root / f"population-work-{k}",
            run_id_of=lambda label: f"population-{k}/{label}",
            base_tree=base_tree,
        )

    def test_population_recursion_budget_lineage_and_judge_isolation(self):
        self.adapter.score = Mock()
        for k in (2, 3):
            with self.subTest(k=k):
                proposer = _BatchProposer(self.proposer.spec)
                result = self._population_result(proposer, k=k)
                self.assertTrue(result.ok)
                self.assertEqual((result.ledger.R, result.ledger.P), (4 * k, 2 * k + 2))
                self.assertEqual(len(result.rollouts), 4 * k)
                self.assertEqual(len(result.stages), 2 * k)
                self.assertEqual(len(result.finals), k)
                self.assertEqual(len(result.strategies), k)
                self.assertEqual(len(result.comparisons), 2)
                self.assertEqual(len(result.lineage), 3 * k)
                self.assertEqual(result.ledger.session_usd, 2.5)
                self.assertTrue(all(comparison.ok for comparison in result.comparisons))
                generated = [record for record in result.lineage if record.generation]
                self.assertEqual(sum(len(record.trials) for record in generated), 3 * k)
                self.assertEqual(
                    sum(bool(record.deterministic_trial) for record in generated), k
                )
                for label in result.finals:
                    self.assertIn(label, result.trees)
                    self.assertIn(label, result.rollouts)
                    self.assertTrue(
                        (self.root / f"population-work-{k}/harnesses" / label).is_dir()
                    )
                record = result.to_dict()
                self.assertEqual(record["ledger"]["expected_P"], 2 * k + 2)
                self.assertEqual(len(record["lineage"]), 3 * k)
                self.assertEqual(len(record["comparisons"]), 2)
                self.assertEqual(len(record["fitness"]), k)
                json.dumps(record)
        self.adapter.score.assert_not_called()

    def test_population_invalid_proposal_retries_without_extra_session(self):
        class RepairPopulationProposer(_BatchProposer):
            def call(self, mission, context_dir, label, **kwargs):
                script = Path(context_dir) / "harness/scripts/summarize.py"
                if label == "g1s1.a1":
                    script.write_text("def broken(:\n", encoding="utf-8")
                    reply = Reply(
                        label,
                        "invalid edit",
                        True,
                        cost_usd=1,
                        attempts=1,
                        workspace=script.parent.parent,
                    )
                    self._calls.append(reply)
                    return reply
                if label == "g1s1.a2":
                    script.write_text(BASE_SCRIPT, encoding="utf-8")
                reply = super().call(mission, context_dir, label, **kwargs)
                if label == "g1s1.a2":
                    reply.cost_usd = 2
                return reply

        proposer = RepairPopulationProposer(self.proposer.spec)
        with patch.object(
            population,
            "strategies_for",
            return_value=self._fixed_population_strategies(2),
        ):
            result = self._population_result(proposer)
        stage = next(stage for stage in result.stages if stage.label == "g1s1")
        self.assertEqual((stage.fell_back, stage.attempts, stage.usd), (False, 2, 3))
        self.assertEqual(result.ledger.labels("session").count("g1s1"), 1)
        self.assertEqual((result.ledger.R, result.ledger.P), (8, 6))
        self.assertIn("g1s1.a1", [reply.label for reply in proposer._calls])
        self.assertIn("g1s1.a2", [reply.label for reply in proposer._calls])

    def test_population_failed_trial_preserves_parent_within_budget(self):
        original_execute = self.adapter.execute

        def fail_one_trial(task_id, harness_dir, run_id, **kwargs):
            if run_id.endswith("/g1s1"):
                raise RuntimeError("trial execution lost")
            return original_execute(task_id, harness_dir, run_id, **kwargs)

        proposer = _BatchProposer(self.proposer.spec)
        with (
            patch.object(
                population,
                "strategies_for",
                return_value=self._fixed_population_strategies(2),
            ),
            patch.object(self.adapter, "execute", side_effect=fail_one_trial),
        ):
            result = self._population_result(proposer)
        outcome = result.comparisons[0].outcomes[0]
        self.assertFalse(result.rollouts["g1s1"].ok)
        self.assertEqual(outcome.survivor, "base1")
        self.assertEqual(outcome.outcome, "parent_fallback")
        self.assertIn("g1s1", outcome.error)
        self.assertEqual((result.ledger.R, result.ledger.P), (8, 6))
        self.assertEqual(len(result.finals), 2)

    def _batch_inputs(self):
        rollouts = {
            label: _comparison_rollout(self.root, label, total)
            for label, total in (
                ("base1", "105.25"),
                ("trial1", "85.15"),
                ("trial1b", "105.25"),
                ("base2", "85.15"),
                ("trial2", "105.25"),
            )
        }
        slots = [
            selection.SlotCandidates(1, "base1", ("trial1", "trial1b")),
            selection.SlotCandidates(2, "base2", ("trial2",)),
        ]
        return slots, rollouts

    def test_batch_comparison_is_anonymous_and_generation_scoped(self):
        (self.adapter.task_files("ledger-1") / "reference_answer.txt").write_text(
            '"pass_rate": 1', encoding="utf-8"
        )
        slots, rollouts = self._batch_inputs()
        proposer = _BatchProposer(self.proposer.spec)
        self.adapter.score = Mock()

        first = selection.run_batch_comparison(
            "ledger-1",
            2,
            1,
            slots,
            rollouts,
            adapter=self.adapter,
            prop=proposer,
            work_dir=self.root / "work",
        )
        self.assertTrue(first.ok)
        self.assertEqual(first.survivors, {1: "trial1", 2: "base2"})
        self.assertEqual((first.usd, first.attempts), (1.25, 2))
        self.assertEqual(first.session, "compare.pop2.g1")
        self.assertEqual(len(first.id_map), 5)
        self.assertTrue(all(outcome.reason for outcome in first.outcomes))
        first_label, first_mission, first_context = proposer.batch_calls[0]
        self.assertTrue((first_context / "source/ledger.csv").is_file())
        self.assertFalse(list(first_context.rglob("reference_answer.txt")))
        self.assertNotIn('"pass_rate"', first_mission)
        for label in rollouts:
            self.assertNotIn(label, first_mission)
        first_record = (first.root / "comparison.json").read_text()
        self.assertEqual(json.loads(first_record)["attempts"], 2)

        second = selection.run_batch_comparison(
            "ledger-1",
            2,
            2,
            slots,
            rollouts,
            adapter=self.adapter,
            prop=proposer,
            work_dir=self.root / "work",
        )
        self.assertTrue(second.ok)
        self.assertEqual(second.survivors, first.survivors)
        self.assertNotEqual(first.root, second.root)
        self.assertEqual(
            [label for label, _mission, _context in proposer.batch_calls],
            ["compare.pop2.g1", "compare.pop2.g2"],
        )
        self.assertNotEqual(first_label, second.session)
        self.assertEqual((first.root / "comparison.json").read_text(), first_record)
        self.assertEqual(len(list((first.root / "candidates").glob("*.md"))), 5)
        self.adapter.score.assert_not_called()

    def test_batch_comparison_invalid_trial_preserves_only_relevant_parent(self):
        slots, rollouts = self._batch_inputs()
        rollouts["trial1"] = _comparison_rollout(
            self.root, "trial1", "0", ok=False, delivered=False
        )
        proposer = _BatchProposer(self.proposer.spec)
        self.adapter.score = Mock()

        result = selection.run_batch_comparison(
            "ledger-1",
            2,
            1,
            slots,
            rollouts,
            adapter=self.adapter,
            prop=proposer,
            work_dir=self.root / "invalid",
        )
        self.assertEqual(result.survivors, {1: "base1", 2: "base2"})
        self.assertEqual(result.outcomes[0].outcome, "parent_fallback")
        self.assertIn("invalid trial", result.outcomes[0].error)
        self.assertEqual(result.outcomes[1].outcome, "parent")
        self.assertNotIn("slot 1:", proposer.batch_calls[0][1])
        self.assertEqual(len(proposer.batch_calls), 1)
        self.adapter.score.assert_not_called()

    def test_batch_comparison_missing_winner_preserves_relevant_parent(self):
        slots, rollouts = self._batch_inputs()
        proposer = _BatchProposer(self.proposer.spec)

        def only_first_slot(mission, context_dir):
            slot, choice = proposer.source_winners(mission, context_dir)[0]
            return f'<winner slot="{slot}" choice="{choice}">Source match.</winner>'

        proposer.responder = only_first_slot
        result = selection.run_batch_comparison(
            "ledger-1",
            2,
            1,
            slots,
            rollouts,
            adapter=self.adapter,
            prop=proposer,
            work_dir=self.root / "missing",
        )
        self.assertEqual(result.survivors, {1: "trial1", 2: "base2"})
        self.assertEqual(result.outcomes[0].outcome, "accepted")
        self.assertEqual(result.outcomes[1].outcome, "parent_fallback")
        self.assertIn("no winner", result.outcomes[1].error)
        self.assertEqual((result.usd, result.attempts), (1.25, 2))

    def test_batch_comparison_failures_preserve_parents_and_return_costs(self):
        def raise_comparator(_mission, _context_dir):
            raise ProposerError("comparator unavailable", attempts=3, cost_usd=2.5)

        cases = (
            ("malformed", lambda _mission, _context: "not a valid reply", 1.25, 2),
            ("exception", raise_comparator, 2.5, 3),
        )
        self.adapter.score = Mock()
        for name, responder, usd, attempts in cases:
            with self.subTest(name=name):
                slots, rollouts = self._batch_inputs()
                proposer = _BatchProposer(self.proposer.spec)
                proposer.responder = responder
                result = selection.run_batch_comparison(
                    "ledger-1",
                    2,
                    1,
                    slots,
                    rollouts,
                    adapter=self.adapter,
                    prop=proposer,
                    work_dir=self.root / name,
                )
                self.assertEqual(result.survivors, {1: "base1", 2: "base2"})
                self.assertEqual(
                    [outcome.outcome for outcome in result.outcomes],
                    ["parent_fallback", "parent_fallback"],
                )
                self.assertEqual((result.usd, result.attempts), (usd, attempts))
                self.assertTrue(result.error)
                self.assertTrue((result.root / "comparison.json").is_file())
        self.adapter.score.assert_not_called()

    def test_batch_comparison_uncertain_or_unknown_choice_preserves_parent(self):
        slots, rollouts = self._batch_inputs()
        proposer = _BatchProposer(self.proposer.spec)
        proposer.responder = lambda _mission, _context: (
            '<winner slot="1" choice="UNCERTAIN">Too close.</winner>'
            '<winner slot="2" choice="ZZ">Not an assigned candidate.</winner>'
        )
        result = selection.run_batch_comparison(
            "ledger-1",
            2,
            1,
            slots,
            rollouts,
            adapter=self.adapter,
            prop=proposer,
            work_dir=self.root / "uncertain",
        )
        self.assertEqual(result.survivors, {1: "base1", 2: "base2"})
        self.assertIn("uncertain", result.outcomes[0].error)
        self.assertIn("unknown candidate", result.outcomes[1].error)

    def test_population_selection_pool_uses_actual_survivors(self):
        rollouts = {
            label: _comparison_rollout(self.root, label, "85.15")
            for label in ("base1", "base2", "g2s1", "agg1", "agg2", "c1", "c2")
        }
        base = [rollouts["base1"], rollouts["base2"]]
        pop = selection.selection_pool(
            "pop", 2, rollouts, base, finals=["g2s1", "base2"]
        )
        self.assertEqual(list(pop), ["g2s1", "base2"])
        for mode, expected in (("agg", ["agg1", "agg2"]), ("seq", ["c1", "c2"])):
            self.assertEqual(list(selection.selection_pool(mode, 2, rollouts, base)), expected)
        with self.assertRaisesRegex(ValueError, "actual ModeResult.finals"):
            selection.selection_pool("pop", 2, rollouts, base)

    def test_population_schedule_base_survivor_library_and_resume(self):
        class PopulationSelector(_BatchProposer):
            def __init__(self, spec):
                super().__init__(spec)
                self.responder = self.preserve_parents

            def preserve_parents(self, mission, _context_dir):
                slots = re.findall(r"^slot (\d+): [A-Z, ]+$", mission, re.M)
                return "\n".join(
                    f'<winner slot="{slot}" choice="UNCERTAIN">'
                    "No source-supported improvement.</winner>"
                    for slot in slots
                )

            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("select.pop"):
                    reply = Reply(
                        label,
                        "<why>Retain the surviving reference output.</why>"
                        "```choice\nA\n```",
                        True,
                        cost_usd=0.5,
                    )
                elif label.startswith("experience."):
                    reply = Reply(
                        label,
                        "<experience>\n"
                        "cue: When a signed ledger needs reconciliation\n"
                        "lesson: Check the script's treatment of credits and debits.\n"
                        "evidence: The recorded probe contrasts show which amounts were included.\n"
                        "confidence: high for the observed arithmetic contrast\n"
                        "</experience>",
                        True,
                        cost_usd=0.25,
                    )
                else:
                    return super().call(mission, context_dir, label, **kwargs)
                self._calls.append(reply)
                return reply

        selector = PopulationSelector(self.proposer.spec)
        modes = ("agg", "seq", "pop")
        library_root = self.root / "library" / "population-integration"
        self.adapter.score = Mock()

        def run_stream():
            return schedule.run_domain(
                "accounting",
                self.adapter.tasks(),
                2,
                adapter=self.adapter,
                prop=self.proposer,
                runs_root=self.adapter.spec.runs_root,
                library_root=library_root,
                modes=modes,
                selector_prop=selector,
            )

        result = run_stream()
        self.assertEqual((result["done"], result["failed"]), (2, 0))
        self.assertEqual(len(selector.batch_calls), 4)
        self.assertFalse(
            any(reply.label.startswith("compare.") for reply in self.proposer._calls)
        )

        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        summary = json.loads(summary_path.read_text())
        pop_mode = summary["modes"]["pop"]
        self.assertEqual(len(pop_mode["finals"]), 2)
        self.assertTrue(all(label.startswith("base") for label in pop_mode["finals"]))
        self.assertEqual((pop_mode["ledger"]["R"], pop_mode["ledger"]["P"]), (8, 6))
        pop_selection = summary["selections"]["pop"]
        self.assertTrue(pop_selection["ok"])
        self.assertTrue(pop_selection["chosen_is_base"])
        self.assertTrue(pop_selection["chosen_tree_is_base"])
        self.assertIn(pop_selection["chosen_label"], pop_mode["finals"])

        pop_work = self.adapter.spec.runs_root / "demo-oc-pop2" / "ledger-1"
        harness_dir = pop_work / "harnesses" / pop_selection["chosen_label"]
        self.assertTrue(tree.is_tree(self.adapter.substrate.surface(), harness_dir))
        self.assertEqual(
            pop_selection["harness_fingerprint"], runmark.fingerprint(harness_dir)
        )
        selection_record = json.loads(
            (pop_work / "selection/pop2/selection.json").read_text()
        )
        self.assertEqual(selection_record["chosen_label"], pop_selection["chosen_label"])
        self.assertTrue(selection_record["chosen_is_base"])
        self.assertEqual(
            selection_record["chosen_run_dir"], pop_selection["chosen_run_dir"]
        )
        restored_context = (
            self.adapter.spec.runs_root
            / "demo-oc-pop2/ledger-2/context/g1s1/library/harnesses"
        )
        self.assertEqual([path.name for path in restored_context.iterdir()], ["ledger-1"])

        executions = self.adapter.execution_count
        calls = (len(self.proposer._calls), len(selector._calls))
        restored = []
        original_add = Library.add

        def record_add(library, task_id, harness, **kwargs):
            restored.append((library.root.name, task_id, library.entries(), kwargs["label"]))
            return original_add(library, task_id, harness, **kwargs)

        with patch.object(Library, "add", record_add):
            resumed = run_stream()
        self.assertEqual([row["status"] for row in resumed["tasks"]], ["skipped", "skipped"])
        self.assertEqual(self.adapter.execution_count, executions)
        self.assertEqual((len(self.proposer._calls), len(selector._calls)), calls)
        for mode, prefix in (("agg", "agg"), ("seq", "c"), ("pop", "base")):
            rows = [row for row in restored if row[0] == mode]
            self.assertEqual(
                [(task_id, before) for _mode, task_id, before, _label in rows],
                [("ledger-1", []), ("ledger-2", ["ledger-1"])],
            )
            self.assertTrue(all(label.startswith(prefix) for *_rest, label in rows))

        harness_file = harness_dir / "scripts/summarize.py"
        harness_file.write_text("changed after selection\n", encoding="utf-8")
        repaired = run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])
        self.assertGreater(self.adapter.execution_count, executions)

        summary = json.loads(summary_path.read_text())
        adopted_run = Path(summary["selections"]["pop"]["chosen_run_dir"])
        (adopted_run / "answer.txt").unlink()
        executions = self.adapter.execution_count
        repaired = run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])
        self.assertGreater(self.adapter.execution_count, executions)
        self.adapter.score.assert_not_called()

    def test_population_mode_parsing_and_demo_defaults(self):
        self.assertEqual(schedule.mode_token("population"), "pop")
        self.assertEqual(schedule.mode_token("pop"), "pop")
        self.assertEqual(cli_common.parse_modes("population"), ("pop",))
        self.assertEqual(cli_common.parse_modes("pop,parallel"), ("agg", "pop"))
        self.assertEqual(cli_common.parse_modes("sequential,population"), ("seq", "pop"))
        self.assertEqual(cli_common.parse_modes(cli_common.MODES_DEFAULT), ("agg",))
        self.assertEqual(cli_common.mode_label("pop"), "population")
        self.assertIn("population", run_sweep._build_parser().format_help())

        completed = {"done": 0, "failed": 0, "tasks": []}
        with patch.object(demo_cli, "run_domain", return_value=completed) as run_domain:
            for mode, expected in (
                ("both", ("agg", "seq")),
                ("population", ("pop",)),
                ("pop", ("pop",)),
            ):
                with self.subTest(mode=mode), patch("sys.stdout", new_callable=io.StringIO):
                    status = demo_cli.main(
                        ["--output", str(self.root / "demo-cli"), "--mode", mode]
                    )
                self.assertEqual(status, 0)
                self.assertEqual(run_domain.call_args.kwargs["modes"], expected)

    def test_population_rejects_small_k_before_execution_or_model_calls(self):
        with (
            patch.object(demo_cli, "DemoAdapter") as adapter_type,
            patch.object(demo_cli, "DemoProposer") as proposer_type,
        ):
            for mode in ("population", "pop"):
                with self.subTest(mode=mode), patch("sys.stderr", new_callable=io.StringIO) as err:
                    with self.assertRaises(SystemExit) as caught:
                        demo_cli.main(["--k", "1", "--mode", mode])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--k must be at least 2 for population mode", err.getvalue())
            adapter_type.assert_not_called()
            proposer_type.assert_not_called()

        with (
            patch.object(cli_common, "make_adapter") as make_adapter,
            patch.object(run_sweep, "Proposer") as proposer_type,
            patch("sys.stderr", new_callable=io.StringIO) as err,
        ):
            status = run_sweep.main(
                ["--bench", "jb", "--k", "1", "--mode", "population", "--dry-run"]
            )
        self.assertEqual(status, 1)
        self.assertIn("--k must be at least 2", err.getvalue())
        make_adapter.assert_not_called()
        proposer_type.assert_not_called()

        with patch.object(self.adapter, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "population mode requires k >= 2"):
                schedule.run_task(
                    "ledger-1",
                    1,
                    adapter=self.adapter,
                    prop=self.proposer,
                    runs_root=self.adapter.spec.runs_root,
                    modes=("pop",),
                )
        execute.assert_not_called()
        self.assertEqual(self.proposer._calls, [])

    def test_population_dry_run_identity_and_configured_strategy_version(self):
        with (
            patch.object(cli_common, "make_adapter", return_value=self.adapter),
            patch.object(run_sweep, "Proposer") as proposer_type,
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            status = run_sweep.main(
                [
                    "--bench",
                    "jb",
                    "--k",
                    "2",
                    "--mode",
                    "parallel,population",
                    "--dry-run",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("modes=parallel,population", out.getvalue())
        self.assertIn("namespace[parallel]: demo-oc-agg2", out.getvalue())
        self.assertIn("namespace[population]: demo-oc-pop2", out.getvalue())
        proposer_type.assert_not_called()
        self.assertEqual(self.adapter.execution_count, 0)

        legacy_args = run_sweep._build_parser().parse_args(["--bench", "jb"])
        legacy_dir = self.root / "identity-legacy"
        run_sweep._check_identity(self.adapter, legacy_args, legacy_dir)
        legacy = json.loads((legacy_dir / "config.json").read_text())
        self.assertNotIn("population_strategy_version", legacy)
        run_sweep._check_identity(self.adapter, legacy_args, legacy_dir)

        pop_args = run_sweep._build_parser().parse_args(
            ["--bench", "jb", "--k", "2", "--mode", "parallel,pop"]
        )
        pop_dir = self.root / "identity-pop"
        run_sweep._check_identity(self.adapter, pop_args, pop_dir)
        pop_identity = json.loads((pop_dir / "config.json").read_text())
        self.assertEqual(
            pop_identity["population_strategy_version"], config.POPULATION_STRATEGY_VERSION
        )
        with patch.object(config, "POPULATION_STRATEGY_VERSION", "different-version"):
            with self.assertRaisesRegex(ValueError, "different models or tasks"):
                run_sweep._check_identity(self.adapter, pop_args, pop_dir)

        proposer = _BatchProposer(self.proposer.spec)
        with patch.object(config, "POPULATION_STRATEGY_VERSION", "configured-test-version"):
            result = self._population_result(proposer)
        self.assertEqual(
            {strategy.version for strategy in result.strategies}, {"configured-test-version"}
        )

    def test_population_judge_report_base_survivor_and_comparison_costs(self):
        selection_record = {
            "task_id": "ledger-1",
            "mode": "pop",
            "k": 2,
            "ok": True,
            "chosen_label": "base1",
            "chosen_run_id": "demo-oc-pop2/ledger-1/base1",
            "chosen_run_dir": str(self.root / "selected-base"),
            "chosen_is_base": True,
            "usd": 0.25,
        }
        summary = {
            "task_id": "ledger-1",
            "selections": {"pop": selection_record},
            "modes": {
                "pop": {
                    "ok": True,
                    "stages": [{"usd": 1.5}],
                    "comparisons": [{"usd": 1.25}, {"usd": 0.75}],
                    "fell_back": [],
                }
            },
        }
        ns_dir = cli_common.runs_root_for(self.adapter) / cli_common.summary_namespace(
            self.adapter, 2, seed=None
        )
        task_dir = ns_dir / "ledger-1"
        task_dir.mkdir(parents=True)
        (task_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        judged = []

        def fake_judge(_adapter, selections, *, parallel):
            judged.extend(selections)
            return [
                judge.Scored(
                    task_id="ledger-1",
                    mode=selections[0]["mode"],
                    k=2,
                    label="base1",
                    run_id=selections[0]["chosen_run_id"],
                    chosen_is_base=True,
                    n_passed=1,
                    n_criteria=1,
                    all_pass=True,
                )
            ]

        with (
            patch.object(cli_common, "make_adapter", return_value=self.adapter),
            patch.object(judge_cli.config, "validate_models"),
            patch.object(judge_cli.J, "judge_selections", side_effect=fake_judge),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            status = judge_cli.main(["--bench", "jb", "--k", "2", "--mode", "population"])
        self.assertEqual(status, 0)
        self.assertEqual([row["mode"] for row in judged], ["pop"])
        self.assertTrue((ns_dir / "judged.json").is_file())

        with (
            patch.object(cli_common, "make_adapter", return_value=self.adapter),
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            status = report_cli.main(["--bench", "jb", "--k", "2", "--mode", "pop"])
        self.assertEqual(status, 0)
        rendered = out.getvalue()
        self.assertIn("population k=2", rendered)
        self.assertIn("final 0 / candidate 0 / base 1", rendered)
        self.assertIn("$3.75", rendered)
        self.assertNotIn("outside the published pool", rendered)
        self.assertNotIn("WARNING", rendered)

        non_base = judge.Scored(
            task_id="ledger-2",
            mode="pop",
            k=2,
            label="g2s1",
            run_id="run-g2s1",
            chosen_is_base=False,
            n_passed=1,
            n_criteria=1,
        )
        self.assertEqual(report._kind(non_base), "final")

    def test_population_demo_end_to_end_audit_and_no_call_resume(self):
        output = self.root / "population-demo"
        score = Mock()
        with (
            patch.object(DemoAdapter, "score", score, create=True),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            status = demo_cli.main(
                ["--output", str(output), "--k", "2", "--mode", "population"]
            )
        self.assertEqual(status, 0)
        first = json.loads((output / "demo.json").read_text())
        self.assertEqual(
            (first["done"], first["failed"], first["execution_count"], first["editor_calls"]),
            (2, 0, 16, 16),
        )
        self.assertEqual([row["status"] for row in first["tasks"]], ["ok", "ok"])

        summaries = {}
        for task_id in self.adapter.tasks():
            summary_path = output / f"runs/demo-oc-agg2/{task_id}/summary.json"
            summary = json.loads(summary_path.read_text())
            summaries[task_id] = summary
            pop = summary["modes"]["pop"]
            self.assertTrue(pop["ok"])
            self.assertEqual(len(pop["strategies"]), 2)
            self.assertEqual(len(pop["lineage"]), 6)
            self.assertEqual(len(pop["comparisons"]), 2)
            self.assertEqual(len(pop["finals"]), 2)
            self.assertEqual(
                (pop["ledger"]["R"], pop["ledger"]["expected_R"]), (8, 8)
            )
            self.assertEqual(
                (pop["ledger"]["P"], pop["ledger"]["expected_P"]), (6, 6)
            )
            self.assertEqual(pop["ledger"]["session_attempts"], 6)
            self.assertEqual(pop["ledger"]["usd"], 0.0)
            self.assertTrue(all(stage["attempts"] == 1 for stage in pop["stages"]))
            self.assertTrue(all("usd" in stage for stage in pop["stages"]))

            pop_work = output / f"runs/demo-oc-pop2/{task_id}"
            for strategy in pop["strategies"]:
                if strategy["edit_scope"] not in ("all", "scripts"):
                    continue
                slot = strategy["slot"]
                scripts = [
                    (pop_work / f"harnesses/g{generation}s{slot}/scripts/summarize.py")
                    .read_text()
                    for generation in (1, 2)
                ]
                self.assertNotEqual(scripts[0], scripts[1])
                self.assertTrue(all("if amount > 0" not in script for script in scripts))
            for generation, comparison in enumerate(pop["comparisons"], 1):
                self.assertEqual(comparison["session"], f"compare.pop2.g{generation}")
                self.assertEqual((comparison["usd"], comparison["attempts"]), (0.0, 1))
                record_path = pop_work / f"comparisons/pop2/g{generation}/comparison.json"
                record = json.loads(record_path.read_text())
                self.assertEqual(record["session"], comparison["session"])
                self.assertEqual(len(record["outcomes"]), 2)
                self.assertTrue(all(outcome["reason"] for outcome in record["outcomes"]))

            selection = summary["selections"]["pop"]
            self.assertTrue(selection["ok"])
            self.assertIn(selection["chosen_label"], pop["finals"])
            self.assertEqual(selection["usd"], 0.0)

        chosen = summaries["ledger-1"]["selections"]["pop"]
        self.assertFalse(chosen["chosen_is_base"])
        self.assertEqual(
            (Path(chosen["chosen_run_dir"]) / "answer.txt").read_text(), "Total: 85.15\n"
        )
        retained = summaries["ledger-2"]["selections"]["pop"]
        self.assertTrue(retained["chosen_is_base"])
        self.assertTrue(retained["chosen_tree_is_base"])
        self.assertIn("No candidate matched", retained["why"])
        score.assert_not_called()

        with patch("sys.stdout", new_callable=io.StringIO):
            status = demo_cli.main(
                ["--output", str(output), "--k", "2", "--mode", "population"]
            )
        self.assertEqual(status, 0)
        resumed = json.loads((output / "demo.json").read_text())
        self.assertEqual((resumed["execution_count"], resumed["editor_calls"]), (0, 0))
        self.assertEqual([row["status"] for row in resumed["tasks"]], ["skipped", "skipped"])
        score.assert_not_called()

    def test_report_noise_floor_requires_parallel_and_sequential(self):
        parallel = report.Summary(mode="agg", k=1, n=1, score=50, chose_final=1)
        sequential = report.Summary(mode="seq", k=1, n=1, score=55, chose_final=1)
        population = report.Summary(mode="pop", k=1, n=1, score=55, chose_final=1)
        mixed = report.render([parallel, population])
        self.assertNotIn("NOISE FLOOR", mixed)
        both = report.render([parallel, sequential])
        self.assertIn("NOISE FLOOR", both)


if __name__ == "__main__":
    unittest.main()
