import pytest

from synthesis_planner.formula import infer_target_class, parse_formula, required_target_elements


def test_parse_simple_formula():
    assert parse_formula("BaTiO3") == {"Ba": 1.0, "Ti": 1.0, "O": 3.0}


def test_parse_parentheses_and_hydrate():
    parsed = parse_formula("La(NO3)3·6H2O")
    assert parsed["La"] == 1.0
    assert parsed["N"] == 3.0
    assert parsed["O"] == 15.0
    assert parsed["H"] == 12.0


def test_required_target_elements_ignore_oxygen_and_hydrogen():
    assert required_target_elements("BaTiO3") == {"Ba", "Ti"}


@pytest.mark.parametrize(
    ("formula", "target_class"),
    [
        ("BaTiO3", "oxide"),
        ("LiFePO4", "phosphate"),
        ("ZnS", "sulfide"),
        ("GaN", "nitride"),
        ("CsPbBr3", "halide"),
    ],
)
def test_infer_target_class(formula, target_class):
    assert infer_target_class(formula) == target_class
