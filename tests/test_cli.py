"""Tests for mcts_crystal.cli: config loading and argument-parser construction."""

import json

import pytest

from mcts_crystal.cli import build_parser, load_config


class TestLoadConfig:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert load_config(str(tmp_path / "nope.json")) == {}

    def test_valid_file_is_parsed(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"mp_api_key": "abc123", "iterations": 42}))
        config = load_config(str(config_path))
        assert config == {"mp_api_key": "abc123", "iterations": 42}

    def test_malformed_json_degrades_to_empty_dict(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{not valid json")
        assert load_config(str(config_path)) == {}


class TestBuildParser:
    def test_rollout_method_choices(self):
        parser = build_parser({})
        rollout_action = next(a for a in parser._actions if a.dest == 'rollout_method')
        assert rollout_action.choices == ['ehull', 'ehull_rdos', 'rdos']

    def test_default_rollout_method_is_ehull(self):
        args = build_parser({}).parse_args([])
        assert args.rollout_method == 'ehull'
        assert args.beta == 1.0
        assert args.gamma == 0.0001

    def test_default_selection_mode_is_ucb1(self):
        args = build_parser({}).parse_args([])
        assert args.selection_mode == 'ucb1'
        assert args.epsilon == 0.2
        assert args.temperature == 1.0

    def test_selection_mode_choices(self):
        parser = build_parser({})
        action = next(a for a in parser._actions if a.dest == 'selection_mode')
        assert action.choices == ['ucb1', 'epsilon_greedy', 'boltzmann', 'puct', 'hybrid']

    def test_config_overrides_selection_mode_default(self):
        args = build_parser({'selection_mode': 'boltzmann', 'temperature': 0.5}).parse_args([])
        assert args.selection_mode == 'boltzmann'
        assert args.temperature == 0.5

    def test_cli_flag_overrides_config_selection_mode(self):
        args = build_parser({'selection_mode': 'boltzmann'}).parse_args(['--selection-mode', 'puct'])
        assert args.selection_mode == 'puct'

    def test_invalid_rollout_method_rejected(self):
        with pytest.raises(SystemExit):
            build_parser({}).parse_args(['--rollout-method', 'not_a_real_method'])

    def test_config_supplies_default_without_cli_flag(self):
        args = build_parser({"mp_api_key": "from-config"}).parse_args([])
        assert args.mp_api_key == "from-config"

    def test_cli_flag_overrides_config_default(self):
        args = build_parser({"mp_api_key": "from-config"}).parse_args(
            ["--mp-api-key", "from-cli"]
        )
        assert args.mp_api_key == "from-cli"

    def test_unknown_config_keys_are_ignored(self):
        # Should not raise even if config.json has a key that isn't a CLI flag
        parser = build_parser({"not_a_real_option": 123})
        args = parser.parse_args([])
        assert not hasattr(args, "not_a_real_option")
