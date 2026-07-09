# Datasets

## Public sources

The planner uses the public datasets cited in the proposal.

### Solid-state

- Source repo: `CederGroupHub/text-mined-synthesis_public`
- File used: `solid-state_dataset_20200713.json.xz`
- Content: text-mined solid-state synthesis reactions with targets, precursors, operations, and reaction strings

### Solution-based

- Source repo: `CederGroupHub/text-mined-solution-synthesis_public`
- File used: `solution-synthesis_dataset_2021-8-5.json.zip`
- Content: text-mined hydrothermal and precipitation procedures with targets, precursors, solvents, operations, and quantities

## Local layout

Downloaded files live under `data/raw/`.

- `data/raw/solid-state_dataset_20200713.json.xz`
- `data/raw/solution-synthesis_dataset_2021-8-5.json.zip`

Normalized outputs are generated under `data/processed/`.

- `data/processed/solid_state_routes.jsonl`
- `data/processed/solution_routes.jsonl`

These processed files are intentionally gitignored because they are regenerated locally from the public inputs.

## Normalization

Each mined record is transformed into a shared `RouteRecord` with:

- route id
- source DOI
- modality
- target formula and target family
- precursor list with coarse precursor classes
- normalized operation list
- reaction string
- short paragraph excerpt
- source dataset label

The current normalization step is conservative:

- it uses mined composition metadata when available
- it falls back to lightweight formula parsing when necessary
- malformed or underspecified formulas degrade to element-level handling instead of aborting the full dataset build

## Current use

- `solid_state_routes.jsonl` is used directly by the current planner.
- `solution_routes.jsonl` is prepared now so hydrothermal and precipitation planning can be added without revisiting the ingestion layer.
