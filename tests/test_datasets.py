from synthesis_planner.datasets import load_processed_routes


def test_prepare_processed_data_creates_solid_and_solution_routes(processed_data):
    solid_routes = load_processed_routes(processed_data, "solid_state")
    solution_routes = load_processed_routes(processed_data, "hydrothermal")

    assert len(solid_routes) == 2
    assert len(solution_routes) == 1
    assert solid_routes[0].target_formula == "BaTiO3"
    assert solution_routes[0].modality == "hydrothermal"


def test_precursor_classes_are_normalized(processed_data):
    solid_routes = load_processed_routes(processed_data, "solid_state")
    classes = {precursor.class_name for precursor in solid_routes[0].precursors}
    assert "carbonate" in classes
    assert "oxide" in classes
