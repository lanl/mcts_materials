from synthesis_planner.cli import build_parser, load_config


def test_load_config_missing_file_returns_empty_dict(tmp_path):
    assert load_config(str(tmp_path / "missing.json")) == {}


def test_plan_subcommand_parses_target():
    parser = build_parser({})
    args = parser.parse_args(["plan", "--target", "BaTiO3"])
    assert args.command == "plan"
    assert args.target == "BaTiO3"
    assert args.modality == "solid_state"


def test_download_data_subcommand_parses():
    parser = build_parser({})
    args = parser.parse_args(["download-data"])
    assert args.command == "download-data"
