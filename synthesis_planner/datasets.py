"""Dataset download, loading, and normalization helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
import lzma
from pathlib import Path
import re
from typing import Iterable
from urllib.request import urlretrieve
import zipfile

from .formula import infer_target_class, parse_formula
from .schema import NumericRange, OperationRecord, PrecursorRecord, RouteRecord

SOLID_STATE_URL = (
    "https://raw.githubusercontent.com/CederGroupHub/text-mined-synthesis_public/"
    "master/solid-state_dataset_20200713.json.xz"
)
SOLUTION_URL = (
    "https://raw.githubusercontent.com/CederGroupHub/text-mined-solution-synthesis_public/"
    "main/solution-synthesis_dataset_2021-8-5.json.zip"
)


def download_public_datasets(data_dir: str | Path) -> dict[str, Path]:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    destinations = {
        "solid_state": path / "solid-state_dataset_20200713.json.xz",
        "solution": path / "solution-synthesis_dataset_2021-8-5.json.zip",
    }
    if not destinations["solid_state"].exists():
        urlretrieve(SOLID_STATE_URL, destinations["solid_state"])
    if not destinations["solution"].exists():
        urlretrieve(SOLUTION_URL, destinations["solution"])
    return destinations


def load_raw_solid_state(data_dir: str | Path) -> list[dict]:
    data_path = Path(data_dir) / "solid-state_dataset_20200713.json.xz"
    with lzma.open(data_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["reactions"]


def load_raw_solution(data_dir: str | Path) -> list[dict]:
    data_path = Path(data_dir) / "solution-synthesis_dataset_2021-8-5.json.zip"
    with zipfile.ZipFile(data_path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            return json.load(handle)


def prepare_processed_data(data_dir: str | Path, processed_dir: str | Path) -> dict[str, Path]:
    processed_root = Path(processed_dir)
    processed_root.mkdir(parents=True, exist_ok=True)

    solid_routes = [normalize_solid_state_record(entry, i) for i, entry in enumerate(load_raw_solid_state(data_dir))]
    solution_routes = [normalize_solution_record(entry, i) for i, entry in enumerate(load_raw_solution(data_dir))]

    solid_path = processed_root / "solid_state_routes.jsonl"
    solution_path = processed_root / "solution_routes.jsonl"
    _write_jsonl(solid_path, solid_routes)
    _write_jsonl(solution_path, solution_routes)
    return {"solid_state": solid_path, "solution": solution_path}


def load_processed_routes(processed_dir: str | Path, modality: str) -> list[RouteRecord]:
    filename = {
        "solid_state": "solid_state_routes.jsonl",
        "hydrothermal": "solution_routes.jsonl",
        "precipitation": "solution_routes.jsonl",
    }[modality]
    routes = []
    with Path(processed_dir, filename).open() as handle:
        for line in handle:
            routes.append(_route_from_dict(json.loads(line)))
    if modality == "solid_state":
        return routes
    return [route for route in routes if route.modality == modality]


def normalize_solid_state_record(entry: dict, index: int) -> RouteRecord:
    target_formula = entry["target"]["material_formula"]
    target_elements = tuple(sorted(_extract_material_elements(entry["target"])))
    return RouteRecord(
        route_id=f"solid-state-{index}",
        source_doi=entry.get("doi", ""),
        modality="solid_state",
        target_formula=target_formula,
        target_elements=target_elements,
        target_class=_infer_target_class(target_formula, target_elements),
        precursors=tuple(_normalize_precursor(precursor) for precursor in entry.get("precursors", [])),
        operations=tuple(_normalize_operation(operation) for operation in entry.get("operations", [])),
        reaction_string=entry.get("reaction_string", ""),
        paragraph_excerpt=entry.get("paragraph_string", ""),
        source_dataset="text-mined-synthesis_public",
    )


def normalize_solution_record(entry: dict, index: int) -> RouteRecord:
    modality = entry.get("type", "solution")
    target_formula = entry["target"]["material_formula"]
    target_elements = tuple(sorted(_extract_material_elements(entry["target"])))
    return RouteRecord(
        route_id=f"solution-{index}",
        source_doi=entry.get("doi", ""),
        modality=modality,
        target_formula=target_formula,
        target_elements=target_elements,
        target_class=_infer_target_class(target_formula, target_elements),
        precursors=tuple(_normalize_precursor(precursor) for precursor in entry.get("precursors", [])),
        operations=tuple(_normalize_operation(operation) for operation in entry.get("operations", [])),
        reaction_string=entry.get("reaction_string", ""),
        paragraph_excerpt=entry.get("paragraph_string", ""),
        source_dataset="text-mined-solution-synthesis_public",
    )


def _normalize_precursor(precursor: dict) -> PrecursorRecord:
    formula = precursor.get("material_formula") or precursor.get("material_string") or "UNKNOWN"
    elements = _extract_material_elements(precursor)
    return PrecursorRecord(
        formula=formula,
        class_name=classify_precursor(formula),
        elements=tuple(sorted(elements)),
    )


def classify_precursor(formula: str) -> str:
    if not formula or formula == "UNKNOWN":
        return "elemental_or_other"
    lowered = formula.lower()
    parsed = _safe_parse_formula(formula)
    if "co3" in lowered:
        return "carbonate"
    if "no3" in lowered:
        return "nitrate"
    if "coo" in lowered or "ch3coo" in lowered or "c2h3o2" in lowered:
        return "acetate"
    if "oh" in lowered:
        return "hydroxide"
    if any(token in formula for token in ("Cl", "Br", "I", "F")):
        return "halide"
    if "so4" in lowered:
        return "sulfate"
    if "S" in parsed and "O" not in parsed:
        return "sulfide"
    if "O" in parsed:
        return "oxide"
    return "elemental_or_other"


def _normalize_operation(operation: dict) -> OperationRecord:
    conditions = operation.get("conditions", {})
    return OperationRecord(
        verb=_normalize_operation_type(operation.get("type", ""), operation.get("string") or operation.get("token")),
        temperature_c=_extract_range(conditions.get("temperature") or conditions.get("heating_temperature")),
        time_h=_extract_range(conditions.get("time") or conditions.get("heating_time")),
        atmosphere=_extract_atmosphere(conditions.get("atmosphere") or conditions.get("heating_atmosphere")),
        source_label=operation.get("string") or operation.get("token"),
    )


def _normalize_operation_type(raw_type: str, raw_label: str | None) -> str:
    mapping = {
        "StartingSynthesis": "start",
        "MixingOperation": "mix",
        "HeatingOperation": "heat",
        "ShapingOperation": "shape",
        "DryingOperation": "dry",
        "QuenchingOperation": "quench",
        "PurificationOperation": "wash",
        "CoolingOperation": "cool",
    }
    if raw_type in mapping:
        return mapping[raw_type]
    if raw_label:
        return raw_label.lower().replace(" ", "_")
    return raw_type.lower() or "other"


def _extract_range(value: list[dict] | dict | None) -> NumericRange | None:
    if not value:
        return None
    record = value[0] if isinstance(value, list) else value
    values = record.get("values") or []
    minimum = record.get("min_value")
    maximum = record.get("max_value")
    if minimum is None and values:
        minimum = min(values)
    if maximum is None and values:
        maximum = max(values)
    units = record.get("units")
    if units in {"K", "kelvin"}:
        minimum = minimum - 273.15 if minimum is not None else None
        maximum = maximum - 273.15 if maximum is not None else None
        units = "C"
    elif units and units.lower() in {"min", "minute", "minutes"}:
        minimum = minimum / 60.0 if minimum is not None else None
        maximum = maximum / 60.0 if maximum is not None else None
        units = "h"
    return NumericRange(minimum=minimum, maximum=maximum, units=units)


def _extract_atmosphere(value: list[str] | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(value) if value else None
    return str(value)


def _extract_material_elements(material: dict) -> set[str]:
    elements = set()
    for composition in material.get("composition", []):
        elements.update(composition.get("elements", {}).keys())
    if elements:
        return elements
    formula = material.get("material_formula") or material.get("material_string") or ""
    if not formula:
        return set()
    return set(_safe_parse_formula(formula).keys())


def _safe_parse_formula(formula: str) -> dict[str, float]:
    if not formula:
        return {}
    try:
        return parse_formula(formula)
    except Exception:
        element_tokens = re.findall(r"[A-Z][a-z]?", formula)
        if not element_tokens:
            return {}
        return {element: 1.0 for element in element_tokens}


def _infer_target_class(target_formula: str, target_elements: tuple[str, ...]) -> str:
    try:
        return infer_target_class(target_formula)
    except Exception:
        elements = set(target_elements)
        if "O" in elements and "P" in elements:
            return "phosphate"
        if "S" in elements and "O" not in elements:
            return "sulfide"
        if "N" in elements and "O" not in elements:
            return "nitride"
        if elements & {"F", "Cl", "Br", "I"}:
            return "halide"
        if "O" in elements:
            return "oxide"
        return "other"


def _write_jsonl(path: Path, routes: Iterable[RouteRecord]) -> None:
    with path.open("w") as handle:
        for route in routes:
            handle.write(json.dumps(route.to_dict()) + "\n")


def _route_from_dict(payload: dict) -> RouteRecord:
    return RouteRecord(
        route_id=payload["route_id"],
        source_doi=payload["source_doi"],
        modality=payload["modality"],
        target_formula=payload["target_formula"],
        target_elements=tuple(payload["target_elements"]),
        target_class=payload["target_class"],
        precursors=tuple(PrecursorRecord(**precursor) for precursor in payload["precursors"]),
        operations=tuple(
            OperationRecord(
                verb=operation["verb"],
                temperature_c=NumericRange(**operation["temperature_c"]) if operation["temperature_c"] else None,
                time_h=NumericRange(**operation["time_h"]) if operation["time_h"] else None,
                atmosphere=operation["atmosphere"],
                source_label=operation["source_label"],
            )
            for operation in payload["operations"]
        ),
        reaction_string=payload["reaction_string"],
        paragraph_excerpt=payload["paragraph_excerpt"],
        source_dataset=payload["source_dataset"],
    )
