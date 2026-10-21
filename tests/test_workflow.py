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
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.core import context, judge, population, recursion, schedule, selection, tree
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
            winner = next(
                candidate_id
                for candidate_id in candidate_ids.split(", ")
                if "Total: 85.15"
                in (context_dir / "candidates" / f"{candidate_id}.md").read_text()
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


if __name__ == "__main__":
    unittest.main()
