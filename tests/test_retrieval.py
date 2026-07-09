from synthesis_planner.datasets import load_processed_routes
from synthesis_planner.retrieval import RetrievalIndex


def test_retrieval_prefers_exact_target(processed_data):
    routes = load_processed_routes(processed_data, "solid_state")
    index = RetrievalIndex(routes)
    results = index.retrieve("BaTiO3", top_k=2)
    assert results[0][1].target_formula == "BaTiO3"


def test_candidate_precursor_sets_include_exact_literature_set(processed_data):
    routes = load_processed_routes(processed_data, "solid_state")
    index = RetrievalIndex(routes)
    analogs = index.retrieve("BaTiO3", top_k=2)
    candidates = index.candidate_precursor_sets("BaTiO3", analogs, max_sets=5)
    formulas = {tuple(sorted(precursor.formula for precursor in precursor_set)) for _, precursor_set in candidates}
    assert ("BaCO3", "TiO2") in formulas
