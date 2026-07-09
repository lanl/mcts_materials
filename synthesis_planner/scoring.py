"""Hard constraints, heuristics, and deterministic judge logic."""

from __future__ import annotations

from statistics import mean

from .formula import required_target_elements
from .schema import JudgeResult, PlannedRoute, PlanningState, PrecursorRecord, RouteRecord, ScoreBreakdown


def evaluate_state(state: PlanningState, analogs: list[tuple[float, RouteRecord]]) -> PlannedRoute:
    required = required_target_elements(state.problem.target_formula)
    coverage = {element for precursor in state.precursors for element in precursor.elements}

    stoich = 1.0 if required <= coverage else 0.0
    retrieval = max((score for score, _ in analogs), default=0.0) / 10.0
    precursor = _precursor_score(state.precursors, state.problem.target_formula)
    condition = _condition_score(state)
    thermo = _thermo_score(state)
    judge = _judge_route(state)
    complexity = len(state.operations) / 10.0
    cost = max(0.0, (len(state.precursors) - 2) * 0.1)
    hazard = _hazard_score(state)

    total = (
        1.2 * stoich
        + 1.0 * precursor
        + 0.6 * thermo
        + 1.0 * retrieval
        + 0.8 * condition
        + 1.0 * judge.score
        - 0.4 * cost
        - 0.5 * hazard
        - 0.3 * complexity
    )

    return PlannedRoute(
        target_formula=state.problem.target_formula,
        modality=state.problem.modality,
        precursors=state.precursors,
        operations=state.operations,
        evidence_dois=state.evidence_dois,
        analog_targets=state.analog_targets,
        score=ScoreBreakdown(
            stoich=stoich,
            precursor=precursor,
            thermo=thermo,
            retrieval=retrieval,
            condition=condition,
            llm=judge.score,
            cost=cost,
            hazard=hazard,
            complexity=complexity,
            total=total,
        ),
        judge=judge,
        mcts_value=total,
    )


def _precursor_score(precursors: tuple[PrecursorRecord, ...], target_formula: str) -> float:
    classes = {precursor.class_name for precursor in precursors}
    target_lower = target_formula.lower()
    score = 0.4
    if "oxide" in classes:
        score += 0.2
    if "carbonate" in classes or "nitrate" in classes:
        score += 0.1
    if "sulfide" in target_lower and "sulfide" in classes:
        score += 0.2
    if "nitride" in target_lower and "halide" not in classes:
        score += 0.05
    return min(score, 1.0)


def _condition_score(state: PlanningState) -> float:
    heating = [operation for operation in state.operations if operation.verb == "heat"]
    if not heating:
        return 0.0
    temperatures = [op.temperature_c.midpoint for op in heating if op.temperature_c and op.temperature_c.midpoint is not None]
    score = 0.5
    if temperatures:
        avg_temp = mean(temperatures)
        if 650.0 <= avg_temp <= 1250.0:
            score += 0.25
        if state.target_class == "oxide" and avg_temp >= 750.0:
            score += 0.15
    if any(operation.verb == "mix" for operation in state.operations):
        score += 0.1
    return min(score, 1.0)


def _thermo_score(state: PlanningState) -> float:
    classes = {precursor.class_name for precursor in state.precursors}
    if state.target_class == "oxide" and classes & {"oxide", "carbonate", "nitrate", "hydroxide"}:
        return 0.7
    if state.target_class == "phosphate" and classes & {"oxide", "carbonate"}:
        return 0.55
    if state.target_class in {"sulfide", "nitride"}:
        return 0.45
    return 0.5


def _hazard_score(state: PlanningState) -> float:
    score = 0.0
    formulas = [precursor.formula for precursor in state.precursors]
    if any("NH4" in formula or "NO3" in formula for formula in formulas):
        score += 0.15
    if any("Cl" in formula or "Br" in formula for formula in formulas):
        score += 0.1
    if state.target_class == "sulfide":
        score += 0.2
    return min(score, 1.0)


def _judge_route(state: PlanningState) -> JudgeResult:
    notes = []
    flags = []
    formulas = [precursor.formula for precursor in state.precursors]
    heating = [operation for operation in state.operations if operation.verb == "heat"]
    temperatures = [op.temperature_c.midpoint for op in heating if op.temperature_c and op.temperature_c.midpoint is not None]
    atmospheres = [op.atmosphere for op in heating if op.atmosphere]

    score = 0.6
    if any("CO3" in formula or "NO3" in formula for formula in formulas):
        if temperatures and min(temperatures) < 600.0:
            flags.append("decomposition_risk")
            notes.append("Low-temperature calcination may leave carbonates or nitrates incompletely decomposed.")
            score -= 0.15
        else:
            notes.append("The route includes decomposable salt precursors with a plausible calcination window.")
            score += 0.1

    if state.target_class in {"sulfide", "nitride"} and any(atm and "air" in atm.lower() for atm in atmospheres):
        flags.append("atmosphere_mismatch")
        notes.append("Air heating is a poor fit for oxygen-sensitive target chemistry.")
        score -= 0.2

    if len(required_target_elements(state.problem.target_formula)) >= 3 and not any(operation.source_label == "anneal" for operation in state.operations):
        flags.append("limited_diffusion_support")
        notes.append("Multicomponent solid-state targets often benefit from regrinding and a second heat treatment.")
        score -= 0.1
    else:
        notes.append("The route includes enough thermal processing to be plausible as a first-pass solid-state recipe.")
        score += 0.1

    if not any(operation.verb == "mix" for operation in state.operations):
        flags.append("missing_mixing")
        notes.append("A practical solid-state route should include explicit mixing or grinding.")
        score -= 0.2

    score = max(0.0, min(score, 1.0))
    return JudgeResult(score=score, notes=tuple(notes), flags=tuple(flags))
