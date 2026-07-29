# MCTS Product-Mode Studies

This directory contains the configuration and scripts for running two product-mode MCTS studies that replicate the original `mcts_crystal` analyses with the new `mcts_framework` implementation.

## Studies

### 1. U-Only Study (`u_only/`)

**Design space**: 108 U-containing compounds (no other f-block elements)  
**Starting composition**: Cr6Sn6U  
**Iterations**: 1,000 per seed  
**Seeds**: 0–4 (5 independent runs)

**Key parameters**:
- `f_block_mode`: `u_only`
- `rollout_method`: `ehull_rdos_product` (multiplicative: r_Ehull × r_DOS)
- `gamma`: 1.0 (raw, unnormalized r_DOS)
- `rollout_depth`: 3
- `n_rollout`: 1
- `rollout_aggregation`: max
- `termination_limit`: 25

### 2. Lanthanide+U Study (`lanthanide_u/`)

**Design space**: 1,620 lanthanide+U compounds (La–Lu + U)  
**Starting composition**: Cr6Sn6Tb (Tb instead of U)  
**Iterations**: 500 per seed  
**Seeds**: 0–4 (5 independent runs)

**Key parameters**:
- `f_block_mode`: `lanthanides_u`
- `move_step`: 3 (extended-range moves)
- `rollout_method`: `ehull_rdos_product`
- `gamma`: 0.00039742998860786596 (normalized = 1 / max r_DOS across U-only space)
- `rollout_depth`: 2
- `n_rollout`: 2
- `rollout_aggregation`: max
- `termination_limit`: 25

## Directory Structure

```
study/
├── u_only/
│   ├── configs/          # YAML configs for seeds 0-4
│   ├── results/          # Output directories for each seed
│   ├── figures/          # Generated publication figures
│   ├── run_all_seeds.sh  # Run all 5 seeds sequentially
│   └── generate_figures.py  # Generate product-mode figures
├── lanthanide_u/
│   ├── configs/
│   ├── results/
│   ├── figures/
│   ├── run_all_seeds.sh
│   └── generate_figures.py
└── README.md             # This file
```

## Running the Studies

### Prerequisites

1. Install the framework with required dependencies:
   ```bash
   pip install -e ".[intermetallic,viz]"
   # or with uv:
   uv sync --extra intermetallic --extra viz
   ```

2. Ensure data files are present in repo root:
   - `high_throughput_mace_results.full.csv` (MACE energy cache)
   - `doscar_peaks_data_with_U.csv` (DOSCAR r_DOS data)

3. Materials Project API key (optional, for live MP queries):
   ```bash
   export MP_API_KEY="your-key-here"
   ```

### Running U-Only Study

```bash
cd study/u_only

# Run all 5 seeds (takes ~hours depending on hardware)
bash run_all_seeds.sh

# Or run individual seeds
mcts-run run --config configs/seed_0.yaml
mcts-run run --config configs/seed_1.yaml
# ... etc

# Generate figures after all seeds complete
python generate_figures.py
```

### Running Lanthanide+U Study

```bash
cd study/lanthanide_u

# Run all 5 seeds
bash run_all_seeds.sh

# Generate figures
python generate_figures.py
```

## Output Files

Each seed run produces (in `results/seed_N/`):
- `summary.json` - Best material, reward, tree statistics
- `best_materials.csv` - Top candidates with properties
- `convergence.csv` - Per-iteration best reward history
- `tree.json` - Complete search tree structure
- `config.yaml` - Exact config used (MP key redacted)
- `report.txt` - Human-readable analysis

After running `generate_figures.py`, the `figures/` directory contains:
- U-only: `ehull_vs_rdos_product.png`, `radial_tree_composite_product.png`
- Lanthanide+U: `ehull_vs_rdos_product_with_experimental.png`

## Analysis

### Pooling Results Across Seeds

To pool and deduplicate compounds across all 5 seeds:

```python
import pandas as pd
from pathlib import Path

study_dir = Path("study/u_only/results")
frames = []
for seed in range(5):
    csv = study_dir / f"seed_{seed}" / "best_materials.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        frames.append(df)

pooled = pd.concat(frames, ignore_index=True)
pooled = pooled.sort_values("reward").drop_duplicates(subset=["formula"], keep="first")
print(f"Total unique compounds discovered: {len(pooled)}")
print(pooled.head(15))  # Top 15
```

### Convergence Analysis

Compare convergence across seeds:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for seed in range(5):
    csv = study_dir / f"seed_{seed}" / "convergence.csv"
    df = pd.read_csv(csv)
    ax.plot(df["iteration"], df["best_reward"], label=f"Seed {seed}", alpha=0.7)

ax.set_xlabel("Iteration")
ax.set_ylabel("Best Product Reward")
ax.legend()
plt.savefig("convergence_comparison.png", dpi=300)
```

## Computational Requirements

### U-Only Study
- **Design space**: 108 compounds
- **Per seed**: ~1,000 iterations × ~5 expansions = ~5,000 evaluations
- **Time estimate**: 2-4 hours per seed (depends on MACE/MP cache hits)
- **Total**: 10-20 hours for all 5 seeds

### Lanthanide+U Study
- **Design space**: 1,620 compounds
- **Per seed**: ~500 iterations × ~10 expansions = ~5,000 evaluations
- **Time estimate**: 2-4 hours per seed
- **Total**: 10-20 hours for all 5 seeds

**Parallelization**: Seeds are independent and can run in parallel if you have multiple cores:
```bash
# Run 5 seeds in parallel (requires 5+ cores)
for seed in 0 1 2 3 4; do
    mcts-run run --config configs/seed_${seed}.yaml &
done
wait
```

## Differences from Original `mcts_crystal` Studies

### Preserved
- ✅ Exact reward functions (ehull_reward, rDOS Gaussian σ=0.5)
- ✅ Same design space definitions (U-only, lanthanide+U)
- ✅ Identical search parameters (depths, aggregation, termination)
- ✅ Same starting compositions
- ✅ Product-mode gamma values

### Improved
- ✅ Cleaner config management (YAML, not CLI args)
- ✅ Automatic result serialization (tree.json, not pickle)
- ✅ Reproducible figure generation (from saved configs)
- ✅ No per-study code duplication
- ✅ Type-safe configuration (Pydantic validation)

## Troubleshooting

**"No module named mcts_framework"**
- Install the package: `pip install -e .` or `uv sync`

**"Materials Project API key required"**
- Set environment variable: `export MP_API_KEY="your-key"`
- Or add to configs: `mp_api_key: "your-key"` (not recommended for version control)

**"MACE cache file not found"**
- Ensure `high_throughput_mace_results.full.csv` is in repo root
- Or update `cache_path` in config YAML files

**"DOSCAR data file not found"**
- Ensure `doscar_peaks_data_with_U.csv` is in repo root
- Or update `doscar_data_path` in config YAML files

**Runs are slow**
- First run builds caches; subsequent runs are faster
- MACE cache hits avoid expensive relaxations
- Consider using fewer `n_rollout` or shallower `rollout_depth` for testing

## Citation

If using these studies for publication, cite:
- The `mcts_framework` package
- The original `mcts_crystal` paper (if comparing)
- Materials Project (if using MP energies)
- MACE force field

## Contact

For questions about the study setup or results, see:
- Framework README: `../../README.md`
- Product figures README: `../../src/mcts_framework/postprocessing/README_PRODUCT_FIGURES.md`
- Original study notes: `../../PROGRESS.md`
