"""
Type-safe configuration models (Pydantic v2).

Configuration is split into:
    - MCTSConfig          : material-agnostic search hyperparameters
    - IntermetallicConfig : crystal-structure-specific settings
    - MoleculeConfig      : molecule-specific settings
    - Config              : top-level wrapper selecting the material type

Configs load from YAML or JSON and validate on construction, so a bad run
fails fast with a clear message instead of deep in the search.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import json
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# --- F-block substitution modes ------------------------------------------
#
# All lanthanide/U modes share the SAME element set - lanthanides Ce(58)-Lu(71)
# plus U(92). They differ only in which MOVES (edges) are allowed:
#
#   u_only                  : f-block frozen at U; no lanthanide moves.
#   lanthanides_u           : +/-move_step neighbors, WITH wrap-around (Ce<->Lu),
#                             plus the U bridge (see u_bridge below).
#   lanthanides_u_no_wrap   : +/-move_step neighbors, NO wrap-around (Ce/Lu are
#                             chain ends), plus the U bridge.
#   full_f_block            : lanthanides Ce-Lu + actinides Th(90)-Pu(94),
#                             +/-1 neighbors plus vertical Ln<->An analog moves.
#
# The lanthanide jump range is set by move_step (not the mode), and the U
# bridge width by u_bridge - both are orthogonal to the mode. (The former
# "lanthanides_u_extended" mode conflated these; it is gone - use
# lanthanides_u with move_step and/or u_bridge='wide' instead.)

FBlockMode = Literal[
    "u_only",
    "lanthanides_u",
    "lanthanides_u_no_wrap",
    "full_f_block",
]

# U-bridge widths: which lanthanides U(92) connects to. 'narrow' = Nd only;
# 'wide' = Nd/Gd/Er. Orthogonal to f_block_mode and move_step.
UBridge = Literal["narrow", "wide"]


class MCTSConfig(BaseModel):
    """Material-agnostic MCTS search hyperparameters."""

    iterations: int = Field(1000, ge=1, description="Number of MCTS iterations")
    exploration_constant: float = Field(
        0.1, ge=0.0, description="UCB/PUCT exploration weight (c)"
    )
    termination_limit: int = Field(
        60, ge=1, description="Visits without improvement before node termination"
    )

    selection_mode: Literal["ucb1", "puct", "epsilon_greedy", "boltzmann"] = "ucb1"
    epsilon: float = Field(
        0.2, ge=0.0, le=1.0, description="Exploration rate (epsilon_greedy)"
    )
    temperature: float = Field(
        1.0, gt=0.0, description="Softmax temperature (boltzmann)"
    )

    rollout_depth: int = Field(1, ge=0, description="Random steps per rollout sample")
    n_rollout: int = Field(5, ge=1, description="Rollout samples per expansion")
    rollout_aggregation: Literal["max", "mean"] = Field(
        "max",
        description="Combine a node's n_rollout samples: 'max' (optimistic; "
        "extra samples discounted by 0.9**rollout_depth) or 'mean' (unbiased "
        "average of undiscounted samples)",
    )
    search_mode: Literal["fast", "thorough"] = Field(
        "fast",
        description="When to stop: 'fast' (stop once the root converges - "
        "fewest evaluations, finds the optimum quickly) or 'thorough' (run the "
        "full iteration budget unless the reachable space is exhausted - "
        "explores more compounds for a better top-N list)",
    )

    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    output_dir: str = Field("mcts_results", description="Output directory")

    model_config = {"extra": "forbid"}


class IntermetallicConfig(BaseModel):
    """Intermetallic crystal-structure search settings."""

    structure_path: str = Field(..., description="Path to starting CIF file")

    # See the module-level FBlockMode notes for exact per-mode move rules.
    f_block_mode: FBlockMode = Field(
        "u_only",
        description=(
            "F-block move rules. All lanthanide/U modes share the Ce-Lu + U "
            "element set and differ only in allowed moves: "
            "u_only (frozen at U); "
            "lanthanides_u (+/-move_step, wrap-around, plus U bridge); "
            "lanthanides_u_no_wrap (+/-move_step, no wrap, plus U bridge); "
            "full_f_block (Ce-Lu + Th-Pu, +/-1 plus vertical Ln<->An analogs). "
            "Jump range is set by move_step and the U bridge by u_bridge."
        ),
    )

    move_step: int = Field(
        1,
        ge=1,
        description="Max positions a substitution may jump along the "
        "transition-metal / Group IV / lanthanide axes (1 = adjacent only; "
        "3 = extended-range exploration)",
    )

    u_bridge: UBridge = Field(
        "narrow",
        description="Which lanthanides U(92) connects to in the lanthanide/U "
        "modes: 'narrow' (Nd only) or 'wide' (Nd/Gd/Er). Orthogonal to "
        "f_block_mode and move_step; ignored by u_only / full_f_block.",
    )

    rollout_method: Literal[
        "ehull", "ehull_rdos", "ehull_rdos_product", "rdos"
    ] = "ehull"
    beta: float = Field(1.0, description="E_hull weight (ehull_rdos)")
    gamma: float = Field(
        0.0001, description="rDOS weight (ehull_rdos / ehull_rdos_product)"
    )

    mp_api_key: Optional[str] = Field(None, description="Materials Project API key")
    doscar_data_path: Optional[str] = Field(
        None, description="Path to DOSCAR peaks CSV (required for rdos/ehull_rdos)"
    )
    cache_path: Optional[str] = Field(
        None, description="Path to MACE energy cache CSV"
    )

    # Optional composition overrides applied to the starting structure.
    transition_metal: Optional[str] = Field(None, description="Override transition metal")
    group_iv: Optional[str] = Field(None, description="Override Group IV element")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_reward_requirements(self) -> "IntermetallicConfig":
        needs_key = self.rollout_method in (
            "ehull", "ehull_rdos", "ehull_rdos_product"
        )
        if needs_key and not self.mp_api_key:
            raise ValueError(
                f"rollout_method={self.rollout_method!r} requires mp_api_key"
            )
        needs_doscar = self.rollout_method in (
            "rdos", "ehull_rdos", "ehull_rdos_product"
        )
        if needs_doscar and not self.doscar_data_path:
            raise ValueError(
                f"rollout_method={self.rollout_method!r} requires doscar_data_path"
            )
        return self


class MoleculeConfig(BaseModel):
    """Molecule search settings."""

    starting_smiles: str = Field(..., description="Starting molecule SMILES")
    functional_groups: List[str] = Field(
        default_factory=lambda: ["C", "CC", "O", "N"],
        description="SMILES strings for functional groups to substitute",
    )

    objective: Literal[
        "melting_point", "h2_capacity", "synthesizability", "multi_objective"
    ] = "melting_point"
    objective_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Weights for multi_objective mode, e.g. "
        "{'h2_capacity': 1.0, 'melting_point': -0.5}",
    )

    # Optional external model directories (else auto-detected by molecule-modifier).
    chemprop_model_dir: Optional[str] = None
    xgboost_model_dir: Optional[str] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_multi_objective(self) -> "MoleculeConfig":
        if self.objective == "multi_objective" and not self.objective_weights:
            raise ValueError(
                "objective='multi_objective' requires objective_weights"
            )
        return self


class Config(BaseModel):
    """Top-level configuration selecting a material type and its settings."""

    material_type: Literal["intermetallic", "molecule"]
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    intermetallic: Optional[IntermetallicConfig] = None
    molecule: Optional[MoleculeConfig] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_material_section_present(self) -> "Config":
        if self.material_type == "intermetallic" and self.intermetallic is None:
            raise ValueError(
                "material_type='intermetallic' requires an 'intermetallic' section"
            )
        if self.material_type == "molecule" and self.molecule is None:
            raise ValueError(
                "material_type='molecule' requires a 'molecule' section"
            )
        return self

    # ---------------------------------------------------------------- #
    # Loaders
    # ---------------------------------------------------------------- #

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a Config from a plain dict."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        import yaml  # local import so PyYAML is only needed if used

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    # ---------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------- #

    #: Placeholder written in place of a real secret when redacting.
    REDACTED_PLACEHOLDER: ClassVar[str] = "<redacted>"

    def dump_yaml(self, path: str, redact_secrets: bool = True) -> None:
        """
        Write this config to a YAML file next to a run's results.

        Persisting the exact config a run used lets post-run analysis read back
        the same parameters (gamma, beta, data paths) instead of re-specifying
        them. With redact_secrets=True (default) the Materials Project API key is
        replaced by a placeholder so the file is safe to keep alongside
        shareable outputs. A placeholder (rather than a blank) is used so the
        redacted file still passes validation on reload via from_yaml - some
        rollout methods require a non-empty key - though it obviously cannot run
        a live energy calculation until a real key is restored.
        """
        import yaml  # local import so PyYAML is only needed if used

        data = self.model_dump()
        if redact_secrets and data.get("intermetallic"):
            if data["intermetallic"].get("mp_api_key"):
                data["intermetallic"]["mp_api_key"] = self.REDACTED_PLACEHOLDER
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
