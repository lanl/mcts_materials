"""
Type-safe configuration models (Pydantic v2).

Configuration is split into:
    - MCTSConfig          : material-agnostic search hyperparameters
    - IntermetallicConfig : crystal-structure-specific settings
    - MoleculeConfig      : molecule-specific settings
    - SuperhydrideConfig  : ternary-superhydride settings
    - Config              : top-level wrapper selecting the material type

Configs load from YAML or JSON and validate on construction, so a bad run
fails fast with a clear message instead of deep in the search.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import json
import os
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
    n_rollout: int = Field(
        5, ge=0,
        description="Random lookahead walks per expansion, additional to the "
        "node's own evaluation (0 = no lookahead; value is the node's reward)",
    )
    rollout_aggregation: Literal["max", "mean"] = Field(
        "max",
        description="Combine a node's reward samples (its own evaluation plus "
        "n_rollout walks): 'max' (optimistic; best reward reachable within "
        "rollout_depth steps) or 'mean' (unbiased average). Samples are "
        "undiscounted (deterministic evaluations).",
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
        0.0001,
        description="rDOS weight (ehull_rdos only; ehull_rdos_product is "
        "ehull_reward * r_DOS and ignores gamma)",
    )

    mp_api_key: Optional[str] = Field(
        None,
        description="Materials Project API key. If left unset, falls back to "
        "the MP_API_KEY environment variable (keeps the key out of shared "
        "config files).",
    )
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
        import warnings

        # Fall back to the MP_API_KEY environment variable when the config
        # leaves mp_api_key unset, so the key can stay out of the (shareable)
        # YAML. An explicit value in the config still takes precedence.
        if not self.mp_api_key:
            env_key = os.environ.get("MP_API_KEY")
            if env_key:
                self.mp_api_key = env_key

        needs_key = self.rollout_method in (
            "ehull", "ehull_rdos", "ehull_rdos_product"
        )
        if needs_key and not self.mp_api_key:
            raise ValueError(
                f"rollout_method={self.rollout_method!r} requires an MP API key: "
                f"set mp_api_key in the config or export MP_API_KEY"
            )
        needs_doscar = self.rollout_method in (
            "rdos", "ehull_rdos", "ehull_rdos_product"
        )
        if needs_doscar and not self.doscar_data_path:
            raise ValueError(
                f"rollout_method={self.rollout_method!r} requires doscar_data_path"
            )

        # Deprecation warning for composition override fields. The actual
        # substitution (and the f-block/f_block_mode compatibility check on the
        # loaded structure) happens in cli.builders._build_intermetallic, where
        # the CIF is already read - config validation stays pure (no file I/O,
        # no ASE dependency).
        if self.transition_metal or self.group_iv:
            warnings.warn(
                "The transition_metal and group_iv fields are deprecated and will be "
                "removed in a future version. Instead, provide a CIF file with the "
                "desired starting composition via structure_path.",
                DeprecationWarning,
                stacklevel=2
            )

        return self


#: Host-element palettes for the superhydride search. See
#: mcts_framework.superhydride.elements for the element sets behind each.
HostPalette = Literal["electropositive", "covalent", "high_tc", "all"]


class QuantumEspressoConfig(BaseModel):
    """
    Quantum ESPRESSO settings for computing the ELF descriptors on demand.

    The numerical fields are the scientific protocol, not tuning knobs: two
    networking values computed at different cutoffs, meshes or pseudopotential
    families are not a comparison. Recording them in the config is what makes a
    campaign reproducible, so they live here rather than being hard-coded.
    """

    # --- Pseudopotentials ---
    pseudo_dir: Optional[str] = Field(
        None,
        description="Directory of UPF files. Falls back to the ESPRESSO_PSEUDO "
        "environment variable. Never mix pseudopotential families within a set "
        "of numbers meant to be compared.",
    )
    pseudo_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Explicit element -> UPF filename map; unlisted elements "
        "default to '<Element>.upf'",
    )

    # --- Where the binaries are ---
    bin_dir: Optional[str] = Field(
        None,
        description="Directory holding pw.x / pp.x / projwfc.x. Falls back to "
        "the QE_BIN_DIR environment variable, then to PATH.",
    )
    mpi_command: str = Field("mpirun", description="MPI launcher; empty runs serially")
    ranks: int = Field(
        4,
        ge=1,
        description="Maximum MPI ranks. Clamped per structure to the FFT plane "
        "count, since ranks that get no planes abort the run.",
    )
    environment_setup: Optional[str] = Field(
        None,
        description="Shell snippet sourced before each binary, for clusters "
        "where the toolchain lives behind modules (e.g. 'module load "
        "gcc/13.2.0 openmpi/4.1.6'). Falls back to QE_ENV_SETUP.",
    )
    timeout_s: float = Field(7200.0, gt=0, description="Wall-clock limit per QE step")

    # --- Protocol ---
    ecutwfc: float = Field(90.0, gt=0, description="Plane-wave cutoff (Ry)")
    ecutrho: float = Field(
        360.0,
        gt=0,
        description="Charge-density cutoff (Ry). 4x ecutwfc is exact for "
        "norm-conserving pseudopotentials; ultrasoft and PAW need 8-12x.",
    )
    degauss: float = Field(0.02, gt=0, description="Smearing width (Ry)")
    conv_thr: float = Field(1e-10, gt=0, description="SCF convergence threshold (Ry)")
    kspacing_scf: float = Field(
        0.2262, gt=0, description="k-point spacing for SCF (1/A, 2*pi convention)"
    )
    kspacing_nscf: float = Field(
        0.1131, gt=0, description="k-point spacing for NSCF (1/A, 2*pi convention)"
    )

    pressure_gpa: Optional[float] = Field(
        None,
        description="Target pressure for the relaxation. Required when relax is "
        "true: a pressure-stabilised hydride relaxed to 0 GPa is a different "
        "material.",
    )
    relax: bool = Field(True, description="vc-relax each candidate before the SCF")
    relax_passes: int = Field(
        2,
        ge=1,
        description="vc-relax passes. 2 is the minimum that sheds the Pulay "
        "error in the stress; a single pass can be 100+ kbar out.",
    )

    # --- I/O ---
    work_root: str = Field(
        "qe_runs",
        description="Parent directory for per-candidate run directories. Put it "
        "on scratch - the funnel writes wavefunctions and cubes.",
    )
    cache_path: Optional[str] = Field(
        None,
        description="CSV of computed descriptors, in the descriptor-table "
        "schema, so an interrupted campaign resumes and a finished one can be "
        "replayed with descriptor_table_path.",
    )
    keep_scratch: bool = Field(False, description="Keep each candidate's QE scratch dir")
    keep_cube: bool = Field(
        False,
        description="Keep the ELF cubes. Tens of megabytes each; a campaign "
        "that keeps every one fills a filesystem before it finishes.",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _resolve_environment(self) -> "QuantumEspressoConfig":
        # Cluster paths belong in the environment, not in a shareable config.
        if not self.pseudo_dir:
            self.pseudo_dir = os.environ.get("ESPRESSO_PSEUDO")
        if not self.bin_dir:
            self.bin_dir = os.environ.get("QE_BIN_DIR")
        if not self.environment_setup:
            self.environment_setup = os.environ.get("QE_ENV_SETUP")

        if not self.pseudo_dir:
            raise ValueError(
                "Quantum ESPRESSO needs a pseudopotential directory: set "
                "pseudo_dir in the config or export ESPRESSO_PSEUDO"
            )
        if self.relax and self.pressure_gpa is None:
            raise ValueError(
                "relax=true requires pressure_gpa. Relaxing a "
                "pressure-stabilised hydride with no target relaxes it to "
                "0 GPa, which is a different material."
            )
        return self


class SuperhydrideConfig(BaseModel):
    """
    Ternary superhydride search settings.

    The search substitutes the non-hydrogen (host) sublattice of the template
    at structure_path, scoring candidates by the ELF-based Tc fit (Belli et
    al., Ann. Phys. 2025, 537, e00280, Eq. 2). Stability is not scored.
    """

    structure_path: str = Field(
        ..., description="Path to the starting hydride template (CIF)"
    )

    host_palette: HostPalette = Field(
        "high_tc",
        description=(
            "Which elements a host site may take: electropositive (alkali, "
            "alkaline earth, rare earth, early transition metals, Al, Th/U); "
            "covalent (p-block elements forming X-H bonds); high_tc (the union "
            "of the two classes that reach high Tc, default); all (adds late "
            "transition metals, which give low-Tc interstitial hydrides)"
        ),
    )

    preserve_distinct_hosts: bool = Field(
        True,
        description=(
            "Drop substitutions that would make two host species identical, "
            "keeping a ternary ternary. Set False to let the search collapse "
            "onto the binary hydrides a template contains."
        ),
    )

    evaluator: Literal["table", "quantum_espresso"] = Field(
        "table",
        description=(
            "Where the ELF descriptors come from: 'table' reads them from "
            "descriptor_table_path (fast, and the search runs without a DFT "
            "stack); 'quantum_espresso' computes them per candidate by running "
            "the ground-state funnel."
        ),
    )

    descriptor_table_path: Optional[str] = Field(
        None,
        description=(
            "CSV of precomputed ELF descriptors with columns "
            "formula, phi, phi_star, h_dos. Compositions absent from it score "
            "0.0, so a run with no table only enumerates the search space."
        ),
    )

    quantum_espresso: Optional[QuantumEspressoConfig] = Field(
        None, description="Required when evaluator='quantum_espresso'"
    )

    normalize_reward: bool = Field(
        True,
        description=(
            "Divide the Tc estimate by its analytic maximum (427.7 K) so "
            "rewards land in (0, 1]. Ranking is unchanged; set False for raw "
            "kelvin and retune exploration_constant by ~2 orders of magnitude."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_evaluator_requirements(self) -> "SuperhydrideConfig":
        if self.evaluator == "quantum_espresso" and self.quantum_espresso is None:
            raise ValueError(
                "evaluator='quantum_espresso' requires a 'quantum_espresso' section"
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

    material_type: Literal["intermetallic", "molecule", "superhydride"]
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    intermetallic: Optional[IntermetallicConfig] = None
    molecule: Optional[MoleculeConfig] = None
    superhydride: Optional[SuperhydrideConfig] = None

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
        if self.material_type == "superhydride" and self.superhydride is None:
            raise ValueError(
                "material_type='superhydride' requires a 'superhydride' section"
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

    # Selection params that are only consumed under a specific selection_mode,
    # so the persisted config records only what the run actually used.
    _MODE_SPECIFIC_SELECTION_PARAMS: ClassVar[Dict[str, str]] = {
        "epsilon": "epsilon_greedy",
        "temperature": "boltzmann",
    }

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

        Selection knobs that the active selection_mode does not consume (e.g.
        temperature under ucb1, epsilon unless epsilon_greedy) are omitted, so
        the recorded config reflects only the settings the run actually used.
        """
        import yaml  # local import so PyYAML is only needed if used

        data = self.model_dump()
        if redact_secrets and data.get("intermetallic"):
            if data["intermetallic"].get("mp_api_key"):
                data["intermetallic"]["mp_api_key"] = self.REDACTED_PLACEHOLDER

        mcts = data.get("mcts")
        if mcts:
            mode = mcts.get("selection_mode")
            for param, owning_mode in self._MODE_SPECIFIC_SELECTION_PARAMS.items():
                if mode != owning_mode:
                    mcts.pop(param, None)

        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
