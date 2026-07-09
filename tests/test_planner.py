from synthesis_planner.planner import SynthesisPlanner
from synthesis_planner.schema import PlanningProblem


def test_planner_generates_ranked_routes(sample_raw_data, processed_data):
    planner = SynthesisPlanner(sample_raw_data, processed_data)
    routes = planner.plan(
        PlanningProblem(target_formula="BaTiO3"),
        iterations=25,
        top_k=3,
        rollout_count=3,
        seed=7,
    )
    assert routes
    assert routes[0].target_formula == "BaTiO3"
    assert any(precursor.formula == "BaCO3" for precursor in routes[0].precursors)
    assert routes[0].score.total > 0.0
