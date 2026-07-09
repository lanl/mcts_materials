"""Core dataclasses shared across the planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumericRange:
    minimum: float | None
    maximum: float | None
    units: str | None = None

    @property
    def midpoint(self) -> float | None:
        if self.minimum is None and self.maximum is None:
            return None
        if self.minimum is None:
            return self.maximum
        if self.maximum is None:
            return self.minimum
        return (self.minimum + self.maximum) / 2.0


@dataclass(frozen=True)
class PrecursorRecord:
    formula: str
    class_name: str
    elements: tuple[str, ...]


@dataclass(frozen=True)
class OperationRecord:
    verb: str
    temperature_c: NumericRange | None = None
    time_h: NumericRange | None = None
    atmosphere: str | None = None
    source_label: str | None = None


@dataclass(frozen=True)
class RouteRecord:
    route_id: str
    source_doi: str
    modality: str
    target_formula: str
    target_elements: tuple[str, ...]
    target_class: str
    precursors: tuple[PrecursorRecord, ...]
    operations: tuple[OperationRecord, ...]
    reaction_string: str
    paragraph_excerpt: str
    source_dataset: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningProblem:
    target_formula: str
    modality: str = "solid_state"
    max_precursors: int = 6


@dataclass(frozen=True)
class JudgeResult:
    score: float
    notes: tuple[str, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ScoreBreakdown:
    stoich: float
    precursor: float
    thermo: float
    retrieval: float
    condition: float
    llm: float
    cost: float
    hazard: float
    complexity: float
    total: float


@dataclass(frozen=True)
class PlannedRoute:
    target_formula: str
    modality: str
    precursors: tuple[PrecursorRecord, ...]
    operations: tuple[OperationRecord, ...]
    evidence_dois: tuple[str, ...]
    analog_targets: tuple[str, ...]
    score: ScoreBreakdown
    judge: JudgeResult
    mcts_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Action:
    kind: str
    label: str
    prior: float
    payload: Any


@dataclass(frozen=True)
class PlanningState:
    problem: PlanningProblem
    target_elements: tuple[str, ...]
    target_class: str
    stage: str = "precursors"
    precursors: tuple[PrecursorRecord, ...] = field(default_factory=tuple)
    operations: tuple[OperationRecord, ...] = field(default_factory=tuple)
    evidence_dois: tuple[str, ...] = field(default_factory=tuple)
    analog_targets: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_terminal(self) -> bool:
        return self.stage == "terminal"
