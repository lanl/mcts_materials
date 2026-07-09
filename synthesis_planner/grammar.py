"""Solid-state synthesis grammar for MCTS expansion."""

from __future__ import annotations

from collections import Counter
from statistics import median

from .schema import Action, OperationRecord, PlanningState, PrecursorRecord, RouteRecord


def expand_state(state: PlanningState, analogs: list[tuple[float, RouteRecord]], candidate_precursor_sets: list[tuple[float, tuple[PrecursorRecord, ...]]]) -> list[Action]:
    if state.stage == "precursors":
        return [
            Action(
                kind="set_precursors",
                label=", ".join(precursor.formula for precursor in precursors),
                prior=max(0.1, float(score)),
                payload=precursors,
            )
            for score, precursors in candidate_precursor_sets
        ]

    if state.stage == "preparation":
        return [
            Action("set_preparation", "mix -> grind", 0.9, _prep_ops("grind")),
            Action("set_preparation", "mix -> ball_mill", 0.6, _prep_ops("ball_mill")),
            Action("set_preparation", "mix -> grind -> pelletize", 0.55, _prep_ops("grind", include_shape=True)),
        ]

    if state.stage == "heating":
        return _heating_actions(analogs)

    if state.stage == "finalize":
        return [
            Action("finalize", "terminate", 1.0, ()),
            Action("finalize", "slow cool", 0.45, (OperationRecord(verb="cool", source_label="slow cool"),)),
            Action("finalize", "quench", 0.35, (OperationRecord(verb="quench", source_label="quench"),)),
        ]

    return []


def apply_action(state: PlanningState, action: Action, analogs: list[tuple[float, RouteRecord]]) -> PlanningState:
    if action.kind == "set_precursors":
        top_dois = tuple(route.source_doi for _, route in analogs[:5] if route.source_doi)
        top_targets = tuple(route.target_formula for _, route in analogs[:5])
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="preparation",
            precursors=tuple(action.payload),
            operations=state.operations,
            evidence_dois=top_dois,
            analog_targets=top_targets,
        )

    if action.kind == "set_preparation":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="heating",
            precursors=state.precursors,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )

    if action.kind == "set_heating":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="finalize",
            precursors=state.precursors,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )

    if action.kind == "finalize":
        return PlanningState(
            problem=state.problem,
            target_elements=state.target_elements,
            target_class=state.target_class,
            stage="terminal",
            precursors=state.precursors,
            operations=state.operations + tuple(action.payload),
            evidence_dois=state.evidence_dois,
            analog_targets=state.analog_targets,
        )

    raise ValueError(f"Unknown action kind: {action.kind}")


def rollout_completion(state: PlanningState, analogs: list[tuple[float, RouteRecord]], candidate_precursor_sets: list[tuple[float, tuple[PrecursorRecord, ...]]], rng) -> PlanningState:
    current = state
    while not current.is_terminal:
        actions = expand_state(current, analogs, candidate_precursor_sets)
        if not actions:
            break
        total_prior = sum(action.prior for action in actions)
        threshold = rng.random() * total_prior
        cumulative = 0.0
        chosen = actions[-1]
        for action in actions:
            cumulative += action.prior
            if cumulative >= threshold:
                chosen = action
                break
        current = apply_action(current, chosen, analogs)
    return current


def _prep_ops(grinding_label: str, include_shape: bool = False) -> tuple[OperationRecord, ...]:
    ops = [
        OperationRecord(verb="mix", source_label="mix"),
        OperationRecord(verb=grinding_label, source_label=grinding_label.replace("_", " ")),
    ]
    if include_shape:
        ops.append(OperationRecord(verb="shape", source_label="pelletize"))
    return tuple(ops)


def _heating_actions(analogs: list[tuple[float, RouteRecord]]) -> list[Action]:
    temperatures = []
    durations = []
    atmospheres = Counter()
    multi_step_examples = []
    for _, route in analogs:
        heating_ops = [operation for operation in route.operations if operation.verb == "heat"]
        if heating_ops:
            for op in heating_ops:
                if op.temperature_c and op.temperature_c.midpoint is not None:
                    temperatures.append(op.temperature_c.midpoint)
                if op.time_h and op.time_h.midpoint is not None:
                    durations.append(op.time_h.midpoint)
                if op.atmosphere:
                    atmospheres[op.atmosphere] += 1
            if len(heating_ops) >= 2:
                multi_step_examples.append(tuple(heating_ops[:2]))

    median_temp = round(median(temperatures), 1) if temperatures else 900.0
    median_time = round(median(durations), 1) if durations else 8.0
    atmosphere = atmospheres.most_common(1)[0][0] if atmospheres else "air"
    default_step = (
        OperationRecord(
            verb="heat",
            temperature_c=_range(median_temp),
            time_h=_range(median_time, units="h"),
            atmosphere=atmosphere,
            source_label="calcine",
        ),
    )
    staged_step = (
        OperationRecord(
            verb="heat",
            temperature_c=_range(max(600.0, median_temp - 120.0)),
            time_h=_range(max(2.0, median_time - 2.0), units="h"),
            atmosphere=atmosphere,
            source_label="calcine",
        ),
        OperationRecord(verb="mix", source_label="regrind"),
        OperationRecord(
            verb="heat",
            temperature_c=_range(min(1300.0, median_temp + 80.0)),
            time_h=_range(median_time, units="h"),
            atmosphere=atmosphere,
            source_label="anneal",
        ),
    )

    actions = [
        Action("set_heating", "single heat step", 1.0, default_step),
        Action("set_heating", "calcine -> regrind -> anneal", 0.85, staged_step),
    ]
    if multi_step_examples:
        example = multi_step_examples[0]
        actions.append(Action("set_heating", "literature-style multistep", 0.95, example))
    return actions


def _range(midpoint: float, units: str = "C"):
    from .schema import NumericRange

    return NumericRange(midpoint, midpoint, units)
