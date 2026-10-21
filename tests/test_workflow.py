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

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.core import context, judge, population, recursion, schedule, tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import resolve_channel_model
from harness2.demo import BASE_SCRIPT, DemoAdapter, DemoProposer


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


if __name__ == "__main__":
    unittest.main()
