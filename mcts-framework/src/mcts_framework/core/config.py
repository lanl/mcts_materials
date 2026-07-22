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
from typing import Literal, Optional, Dict, List, Any

from pydantic import BaseModel, Field, field_validator, model_validator


# --- F-block substitution modes ------------------------------------------
#
# All lanthanide/U modes share the SAME element set - lanthanides Ce(58)-Lu(71)
# plus U(92). They differ only in which MOVES (edges) are allowed:
#
#   u_only                  : f-block frozen at U; no lanthanide moves.
#   lanthanides_u           : +/-1 neighbor, WITH wrap-around (Ce<->Lu);
#                             U bridges to/from Nd only.
#   lanthanides_u_extended  : +/-1,2,3 jumps, WITH wrap-around; U bridges
#                             to/from Nd, Gd, Er (light/mid/heavy).
#   lanthanides_u_no_wrap   : +/-1 neighbor, NO wrap-around (Ce/Lu are chain
#                             ends); U bridges to/from Nd only. (Was named
#                             "experimental"; its old code comment mislabeled
#                             the set as actinides - it is lanthanides.)
#   full_f_block            : lanthanides Ce-Lu + actinides Th(90)-Pu(94),
#                             +/-1 neighbors plus vertical Ln<->An analog moves.
#
# Canonical names are the values below. Deprecated aliases are normalized to
# their canonical form by a validator.

FBlockMode = Literal[
    "u_only",
    "lanthanides_u",
    "lanthanides_u_extended",
    "lanthanides_u_no_wrap",
    "full_f_block",
]

# Deprecated -> canonical alias map.
_F_BLOCK_ALIASES: Dict[str, str] = {
    "experimental": "lanthanides_u_no_wrap",
}


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
    # The deprecated alias "experimental" is accepted and normalized to
    # "lanthanides_u_no_wrap".
    f_block_mode: FBlockMode = Field(
        "u_only",
        description=(
            "F-block move rules. All lanthanide/U modes share the Ce-Lu + U "
            "element set and differ only in allowed moves: "
            "u_only (frozen at U); "
            "lanthanides_u (+/-1, wrap-around, U<->Nd); "
            "lanthanides_u_extended (+/-1..3, wrap-around, U<->Nd/Gd/Er); "
            "lanthanides_u_no_wrap (+/-1, no wrap, U<->Nd; formerly "
            "'experimental'); "
            "full_f_block (Ce-Lu + Th-Pu, +/-1 plus vertical Ln<->An analogs)."
        ),
    )

    move_step: int = Field(
        1,
        ge=1,
        description="Max positions a substitution may jump along the "
        "transition-metal / Group IV / lanthanide axes (1 = adjacent only; "
        "3 = extended-range exploration)",
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

    @field_validator("f_block_mode", mode="before")
    @classmethod
    def _normalize_f_block_alias(cls, v: Any) -> Any:
        """Map deprecated f_block_mode aliases to their canonical name."""
        if isinstance(v, str) and v in _F_BLOCK_ALIASES:
            return _F_BLOCK_ALIASES[v]
        return v

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
