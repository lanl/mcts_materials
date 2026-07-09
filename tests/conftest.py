import json
from pathlib import Path

import pytest

from synthesis_planner.datasets import prepare_processed_data


@pytest.fixture
def sample_raw_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    solid_payload = {
        "release_date": "2020-07-13",
        "reactions": [
            {
                "doi": "10.1000/example1",
                "paragraph_string": "BaCO3 and TiO2 were mixed, calcined, reground, and annealed in air.",
                "synthesis_type": "solid-state",
                "reaction_string": "BaCO3 + TiO2 -> BaTiO3",
                "target": {"material_formula": "BaTiO3"},
                "precursors": [
                    {"material_formula": "BaCO3", "composition": [{"elements": {"Ba": "1", "C": "1", "O": "3"}}]},
                    {"material_formula": "TiO2", "composition": [{"elements": {"Ti": "1", "O": "2"}}]},
                ],
                "operations": [
                    {"type": "StartingSynthesis", "token": "prepared", "conditions": {}},
                    {"type": "MixingOperation", "token": "ground", "conditions": {}},
                    {
                        "type": "HeatingOperation",
                        "token": "calcined",
                        "conditions": {
                            "heating_temperature": [{"values": [900.0], "units": "C"}],
                            "heating_time": [{"values": [6.0], "units": "h"}],
                            "heating_atmosphere": "air",
                        },
                    },
                    {"type": "MixingOperation", "token": "reground", "conditions": {}},
                    {
                        "type": "HeatingOperation",
                        "token": "annealed",
                        "conditions": {
                            "heating_temperature": [{"values": [1100.0], "units": "C"}],
                            "heating_time": [{"values": [10.0], "units": "h"}],
                            "heating_atmosphere": "air",
                        },
                    },
                ],
                "targets_string": ["BaTiO3"],
            },
            {
                "doi": "10.1000/example2",
                "paragraph_string": "SrCO3 and TiO2 were mixed and fired in air.",
                "synthesis_type": "solid-state",
                "reaction_string": "SrCO3 + TiO2 -> SrTiO3",
                "target": {"material_formula": "SrTiO3"},
                "precursors": [
                    {"material_formula": "SrCO3", "composition": [{"elements": {"Sr": "1", "C": "1", "O": "3"}}]},
                    {"material_formula": "TiO2", "composition": [{"elements": {"Ti": "1", "O": "2"}}]},
                ],
                "operations": [
                    {"type": "MixingOperation", "token": "mixed", "conditions": {}},
                    {
                        "type": "HeatingOperation",
                        "token": "fired",
                        "conditions": {
                            "heating_temperature": [{"values": [1000.0], "units": "C"}],
                            "heating_time": [{"values": [8.0], "units": "h"}],
                            "heating_atmosphere": "air",
                        },
                    },
                ],
                "targets_string": ["SrTiO3"],
            },
        ],
    }

    solution_payload = [
        {
            "doi": "10.1000/solution1",
            "paragraph_string": "A hydrothermal BaTiO3 route.",
            "reaction": {"left_side": ["BaCl2", "TiCl4"], "right_side": ["BaTiO3"]},
            "reaction_string": "BaCl2 + TiCl4 -> BaTiO3",
            "target": {"material_formula": "BaTiO3"},
            "targets_string": ["BaTiO3"],
            "precursors": [
                {"material_formula": "BaCl2", "composition": [{"elements": {"Ba": "1", "Cl": "2"}}]},
                {"material_formula": "TiCl4", "composition": [{"elements": {"Ti": "1", "Cl": "4"}}]},
            ],
            "solvents_string": ["water"],
            "operations": [
                {"type": "MixingOperation", "string": "mixed", "conditions": {}},
                {
                    "type": "HeatingOperation",
                    "string": "heated",
                    "conditions": {
                        "temperature": [{"values": [180.0], "units": "C"}],
                        "time": [{"values": [24.0], "units": "h"}],
                        "atmosphere": ["sealed"],
                    },
                },
            ],
            "quantities": [],
            "type": "hydrothermal",
        }
    ]

    import lzma
    import zipfile

    with lzma.open(data_dir / "solid-state_dataset_20200713.json.xz", "wt", encoding="utf-8") as handle:
        json.dump(solid_payload, handle)
    with zipfile.ZipFile(data_dir / "solution-synthesis_dataset_2021-8-5.json.zip", "w") as archive:
        archive.writestr("solution.json", json.dumps(solution_payload))

    return data_dir


@pytest.fixture
def processed_data(sample_raw_data: Path, tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    prepare_processed_data(sample_raw_data, processed)
    return processed
