"""
Command-line interface for the MCTS materials framework.

Usage:
    mcts-run run --config my_run.yaml
    mcts-run validate --config my_run.yaml

`run` loads and validates a config, builds the appropriate MCTS (intermetallic
or molecule), executes the search, and writes results to the output directory.
`validate` just loads/validates the config and prints a summary - useful for
checking a config before launching a long run.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import asyncio
import logging
from pathlib import Path

import typer

from ..core.config import Config

app = typer.Typer(
    add_completion=False,
    help="Monte Carlo Tree Search for materials discovery.",
)


def _load_config(config_path: str) -> Config:
    """Load a Config from a .yaml/.yml or .json file, dispatching on suffix."""
    suffix = Path(config_path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return Config.from_yaml(config_path)
    if suffix == ".json":
        return Config.from_json(config_path)
    raise typer.BadParameter(
        f"Config must be .yaml, .yml, or .json (got {suffix!r})"
    )


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML/JSON config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run an MCTS search from a config file and save results."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate config first so bad configs fail fast with a clear message.
    try:
        cfg = _load_config(config)
    except Exception as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Build components lazily (heavy imports happen here). Surface missing
    # optional dependencies as a friendly message rather than a raw traceback.
    from .builders import build_mcts
    from .results import save_results

    try:
        mcts = build_mcts(cfg)
    except ImportError as exc:
        typer.secho(
            f"Missing dependency for material_type={cfg.material_type!r}: {exc}\n"
            f"Install the matching extra, e.g. "
            f"pip install -e '.[{cfg.material_type}]'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.secho(
        f"Running {cfg.material_type} search: {cfg.mcts.iterations} iterations",
        fg=typer.colors.GREEN,
    )
    asyncio.run(mcts.run(cfg.mcts.iterations))

    paths = save_results(mcts, cfg.mcts.output_dir, config=cfg)

    summary = mcts.summary()
    typer.secho("\nSearch complete.", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Best material : {summary['best_material']}")
    typer.echo(f"  Best reward   : {summary['best_reward']:.4f}")
    typer.echo(f"  Unique found  : {summary['unique_materials']}")
    typer.echo(f"  Results       : {cfg.mcts.output_dir}/")
    for name, path in paths.items():
        typer.echo(f"    - {name}: {path}")


@app.command()
def validate(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML/JSON config"),
) -> None:
    """Load and validate a config file without running a search."""
    try:
        cfg = _load_config(config)
    except Exception as exc:
        typer.secho(f"Invalid config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("Config is valid.", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  material_type : {cfg.material_type}")
    typer.echo(f"  iterations    : {cfg.mcts.iterations}")
    typer.echo(f"  selection     : {cfg.mcts.selection_mode}")
    if cfg.material_type == "intermetallic":
        typer.echo(f"  rollout       : {cfg.intermetallic.rollout_method}")
        typer.echo(f"  f_block_mode  : {cfg.intermetallic.f_block_mode}")
    else:
        typer.echo(f"  objective     : {cfg.molecule.objective}")
        typer.echo(f"  start SMILES  : {cfg.molecule.starting_smiles}")


if __name__ == "__main__":
    app()
