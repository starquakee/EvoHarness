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

"""Deterministic fixed-strategy primitives for discrete population recursion."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .tree import SurfaceSpec, Tree

GENERATIONS = 2
PARENT_SELECTIONS = ("current", "random", "best", "pbest")
EDIT_SCOPES = ("all", "always_on", "skills", "scripts", "subagents", "plugins")
CROSSOVER_KINDS = ("binomial", "exponential")
CR_VALUES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class PopulationStrategy:
    task_id: str
    mode: str
    k: int
    version: str
    slot: int
    parent_selection: str
    donor_count: int
    edit_scope: str
    cr: float
    crossover: str

    def to_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class LineageRecord:
    slot: int
    generation: int
    parent: str = ""
    donors: tuple[str, ...] = ()
    trial: str = ""
    trial_kind: str = ""
    survivor: str = ""
    fitness: int = 0
    outcome: str = "pending"
    reason: str = ""

    def to_dict(self) -> dict:
        out = {key: value for key, value in self.__dict__.items()}
        out["donors"] = list(self.donors)
        return out


def _validate_k(k: int) -> int:
    k = int(k)
    if k < 2:
        raise ValueError(f"population mode requires k >= 2, got {k}")
    return k


def _validate_generation(generation: int) -> int:
    generation = int(generation)
    if generation not in range(1, GENERATIONS + 1):
        raise ValueError(f"population generation must be 1 or 2, got {generation}")
    return generation


def _rng(*parts: object) -> random.Random:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def strategy_for(task_id: str, mode: str, k: int, version: str, slot: int) -> PopulationStrategy:
    k = _validate_k(k)
    slot = int(slot)
    if not 1 <= slot <= k:
        raise ValueError(f"population slot must be in 1..{k}, got {slot}")
    rng = _rng("strategy", task_id, mode, k, str(version), slot)
    return PopulationStrategy(
        task_id=str(task_id),
        mode=str(mode),
        k=k,
        version=str(version),
        slot=slot,
        parent_selection=rng.choice(PARENT_SELECTIONS),
        donor_count=rng.choice((1, 2)) if k >= 3 else 1,
        edit_scope=rng.choice(EDIT_SCOPES),
        cr=rng.choice(CR_VALUES),
        crossover=rng.choice(CROSSOVER_KINDS),
    )


def strategies_for(task_id: str, mode: str, k: int, version: str) -> tuple[PopulationStrategy, ...]:
    k = _validate_k(k)
    return tuple(strategy_for(task_id, mode, k, version, slot) for slot in range(1, k + 1))


def choose_parent_and_donors(
    strategy: PopulationStrategy,
    snapshot: Mapping[int, str],
    fitness: Mapping[int, int],
    *,
    generation: int,
) -> tuple[str, tuple[str, ...]]:
    """Select from one frozen slot->label snapshot, never from partially updated survivors."""
    generation = _validate_generation(generation)
    slots = sorted(snapshot)
    expected = list(range(1, strategy.k + 1))
    if slots != expected or set(fitness) != set(expected):
        raise ValueError(f"population snapshot and fitness must contain exactly slots 1..{strategy.k}")
    if strategy.parent_selection not in PARENT_SELECTIONS:
        raise ValueError(f"unknown parent selection {strategy.parent_selection!r}")
    if strategy.donor_count not in (1, 2):
        raise ValueError(f"population donor count must be 1 or 2, got {strategy.donor_count}")

    rng = _rng(
        "parent-and-donors",
        strategy.task_id,
        strategy.mode,
        strategy.k,
        strategy.version,
        strategy.slot,
        generation,
    )
    ranked = sorted(slots, key=lambda slot: (-int(fitness[slot]), snapshot[slot], slot))
    if strategy.parent_selection == "current":
        parent_slot = strategy.slot
    elif strategy.parent_selection == "random":
        parent_slot = rng.choice(slots)
    elif strategy.parent_selection == "best":
        parent_slot = ranked[0]
    else:
        pbest_count = max(1, (len(slots) + 4) // 5)
        parent_slot = rng.choice(ranked[:pbest_count])

    donor_slots = [slot for slot in slots if slot != parent_slot]
    if len(donor_slots) < strategy.donor_count:
        raise ValueError(
            f"population k={strategy.k} cannot supply {strategy.donor_count} distinct donors"
        )
    donors = rng.sample(donor_slots, strategy.donor_count)
    return snapshot[parent_slot], tuple(snapshot[slot] for slot in donors)


def _in_scope(surface: SurfaceSpec, rel: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "always_on":
        return rel in surface.always_on
    if scope == "skills":
        return rel.startswith(surface.skill_dir.strip("/") + "/")
    if scope == "scripts":
        return rel.startswith("scripts/") or "/scripts/" in rel
    if scope == "subagents":
        return rel.startswith("sub_agents/")
    if scope == "plugins":
        return surface.plugin_dir is not None and rel.startswith(surface.plugin_dir.strip("/") + "/")
    raise ValueError(f"unknown population edit scope {scope!r}")


def component_crossover(
    surface: SurfaceSpec,
    parent: Tree,
    donors: Sequence[Tree],
    strategy: PopulationStrategy,
    *,
    generation: int,
) -> Tree:
    """Copy whole named file components from donors without mutating any input tree."""
    generation = _validate_generation(generation)
    if strategy.crossover not in CROSSOVER_KINDS:
        raise ValueError(f"unknown population crossover {strategy.crossover!r}")
    if not 0 < strategy.cr <= 1:
        raise ValueError(f"population CR must be in (0, 1], got {strategy.cr}")
    if strategy.donor_count not in (1, 2):
        raise ValueError(f"population donor count must be 1 or 2, got {strategy.donor_count}")
    if len(donors) != strategy.donor_count:
        raise ValueError(
            f"strategy for slot {strategy.slot} needs {strategy.donor_count} donors, "
            f"got {len(donors)}"
        )

    rng = _rng(
        "component-crossover",
        strategy.task_id,
        strategy.mode,
        strategy.k,
        strategy.version,
        strategy.slot,
        generation,
    )
    paths = sorted(set(parent).union(*(set(donor) for donor in donors)))
    eligible = [rel for rel in paths if _in_scope(surface, rel, strategy.edit_scope)]
    out = dict(parent)
    if not eligible:
        return out

    if strategy.crossover == "binomial":
        selected = {rel for rel in eligible if rng.random() < strategy.cr}
        selected.add(eligible[rng.randrange(len(eligible))])
    else:
        start = rng.randrange(len(eligible))
        length = 1
        while length < len(eligible) and rng.random() < strategy.cr:
            length += 1
        selected = {eligible[(start + offset) % len(eligible)] for offset in range(length)}

    for rel in sorted(selected):
        donor = donors[rng.randrange(len(donors))]
        if rel in donor:
            out[rel] = donor[rel]
        else:
            out.pop(rel, None)
    return out


def deterministic_trial_schedule(
    task_id: str, mode: str, k: int, version: str
) -> dict[int, tuple[int, ...]]:
    """Assign each slot's sole deterministic crossover trial to one of the two generations."""
    k = _validate_k(k)
    slots = list(range(1, k + 1))
    _rng("deterministic-trials", task_id, mode, k, str(version)).shuffle(slots)
    return {
        generation: tuple(slots[generation - 1 :: GENERATIONS])
        for generation in range(1, GENERATIONS + 1)
    }


def initial_lineage(k: int) -> list[LineageRecord]:
    k = _validate_k(k)
    return [
        LineageRecord(slot=slot, generation=0, survivor=f"base{slot}", outcome="initial")
        for slot in range(1, k + 1)
    ]


def survivor_fitness(parent_fitness: int, accepted: bool) -> int:
    return int(parent_fitness) + int(bool(accepted))


def expected_trial_rollouts(k: int) -> int:
    return 3 * _validate_k(k)


def expected_rollouts(k: int) -> int:
    return 4 * _validate_k(k)


def expected_sessions(k: int) -> int:
    return 2 * _validate_k(k) + GENERATIONS
