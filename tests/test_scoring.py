from synthesis_planner.formula import infer_target_class, parse_formula
from synthesis_planner.schema import PlanningProblem, PlanningState, PrecursorRecord
from synthesis_planner.scoring import evaluate_state


def test_evaluate_state_rewards_complete_oxide_route(processed_data):
    problem = PlanningProblem(target_formula="BaTiO3")
    state = PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(),
    )
    route = evaluate_state(state, [])
    assert route.score.stoich == 1.0
    assert route.score.precursor > 0.0


def test_judge_flags_low_temperature_decomposition_risk():
    from synthesis_planner.schema import NumericRange, OperationRecord

    problem = PlanningProblem(target_formula="BaTiO3")
    state = PlanningState(
        problem=problem,
        target_elements=tuple(sorted(parse_formula(problem.target_formula))),
        target_class=infer_target_class(problem.target_formula),
        stage="terminal",
        precursors=(
            PrecursorRecord("BaCO3", "carbonate", ("Ba", "C", "O")),
            PrecursorRecord("TiO2", "oxide", ("Ti", "O")),
        ),
        operations=(
            OperationRecord("mix", source_label="mix"),
            OperationRecord("heat", temperature_c=NumericRange(500.0, 500.0, "C"), source_label="calcine"),
        ),
    )
    route = evaluate_state(state, [])
    assert "decomposition_risk" in route.judge.flags
