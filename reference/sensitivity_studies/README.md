# MCTS Sensitivity Studies

Replicate-run sensitivity studies for the U-only `ehull_rdos` search, used to
"calibrate" MCTS-algorithm hyperparameters (not the reward formula itself -
the ehull_reward sharpness constant (120), the E_hull stability threshold
(0.05), and the rDOS Gaussian width sigma (0.5) are physics-informed and are
**not** swept here).

## Methodology

Every sweep holds the calibrated baseline (matching `config.json` except for
the starting material - see `common.py`'s `BASELINE` docstring: U-only
f-block mode, `ucb1` selection, `exploration_constant=0.5`, `Pd6Ge6U` starting
material (d=6 from the true global best, an unremarkable rank-54/108 point -
chosen instead of config.json's Cr6Sn6U (d=3) so hyperparameter effects
aren't masked by starting too close to the optimum), `ehull_rdos` reward with
`beta=1.0`/`gamma=0.0001`, `termination_limit=25`, `rollout_depth=3`,
`n_rollout=2`) and varies exactly one parameter (or a small closely-related
group of parameters) away from it. For each value tested, 5 replicate seeds
are run for 500 iterations each (all compounds in this search space are
already in `high_throughput_mace_results.full.csv`, so this is fast - no live
MACE/Materials Project calls). The metric is the running-best composite
reward per iteration, averaged across the 5 seeds with a shaded 10th-90th
percentile band (not mean +/- std, which can extend past the best value any
seed actually reached), on a log-scale iteration axis (convergence in this
search space happens within the first ~10-50 iterations, so a linear axis
compresses all the interesting dynamics into an unreadable sliver).

Run `bash scripts/run_all.sh` to reproduce everything (~15-20 min).

## Sweeps and findings

These findings are from the current `Pd6Ge6U` (d=6) baseline, at the
calibrated `n_rollout=2`/`rollout_depth=3` (bumped from `n_rollout=1`, which
made `rollout_depth` a complete no-op - see `common.py`'s `BASELINE`
docstring and `mcts.py`'s `_run_rollout_samples` docstring for why one extra
chained rollout sample requires `n_rollout=2`, not 1). An earlier round of
these sweeps used the calibrated `Cr6Sn6U` (d=3) starting point and found *no*
measurable effect for `c` or `selection_mode` - starting that close to the
optimum, the search finished during the mode-independent expansion-exhaustion
phase described below before either parameter ever got a chance to matter.
Moving the baseline to d=6 surfaces a real `selection_mode` effect and
reverses the `termination_limit` finding; see "Why distance-to-optimum
changes which parameters matter" below.

### `c_sweep` - exploration_constant (0.05, 0.1, 0.2, 0.5 calibrated, 1.0)
**Still no measurable effect** even at d=6: all 5 values converge to the
identical final composite score (1.1664) across all 5 seeds each. `c` only
ever governs the choice *among already-expanded children*, and (per the
explanation below) that choice apparently still doesn't change which
compound is found first in this search space, regardless of starting
distance.

### `starting_material_sweep` - V6Ge6U / Ru6Ge6U / Pd6Ge6U / Cu6Ge6U
(move-graph distance to the true global-best U-only compound of d=2/4/6/8
respectively, out of a max possible d=9 in this search space)

**Clear, monotonic effect**: convergence speed tracks starting-point distance
directly - V6Ge6U (d=2) reaches the plateau fastest, Cu6Ge6U (d=8) slowest.
This is the parameter that actually matters most for "rate of convergence" in
this search space. The sweep script for this one now lives in
`analysis/ehull_rdos_u_only_study/sweep_starting_material.py` (it directly
feeds that study's `convergence_by_starting_material.png`, so it's kept next
to the other figure-generating code rather than here, and it isn't part of
`run_all.sh`'s scope) - it still reuses this directory's `common.py` harness
and writes to the same `results/starting_material_sweep/convergence_data.csv`
below. (Not yet rerun at `n_rollout=2`; still reflects `n_rollout=1`.)

### `selection_mode_sweep` - ucb1 (calibrated) / epsilon_greedy / boltzmann / puct / hybrid
**Now a real effect** (this reverses the old d=3-baseline finding):
`boltzmann` gets stuck at a suboptimal composite score (1.1486-1.1497) in
**5/5 seeds**, never reaching the true optimum (1.1664) that `ucb1`,
`epsilon_greedy`, `puct`, and `hybrid` all reach in 5/5 seeds. `boltzmann`
samples children proportional to `exp(reward/T)` rather than UCB1's
optimism-under-uncertainty bonus, so once it finds a good-but-not-best child
its temperature-scaled sampling can starve the still-better sibling of
further visits - a failure mode that simply never gets triggered when the
optimum is only 1-2 hops away (d=3 baseline), but does at d=6. Unchanged
under the `n_rollout=2` baseline.

### `rollout_params_sweep` - n_rollout, rollout_depth, termination_limit (one-at-a-time)
- **n_rollout** (1, 2 calibrated, 5): real effect, but **not a clean
  "more is better" ordering** - n_rollout=5 is consistently fastest (0.85 by
  iteration 2, 1.01 by iteration 5), but the calibrated n_rollout=2 is *not*
  clearly better than n_rollout=1: at iteration 5, n_rollout=1 (1.02) actually
  beats n_rollout=2 (0.74). With only one extra chained sample at n_rollout=2,
  discounted by `0.9**rollout_depth` before being maxed against the node's own
  real reward, the variance-reduction benefit is small enough that it doesn't
  reliably beat the plain single-evaluation case - you need several extra
  samples (n_rollout=5) before the benefit becomes consistent.
- **rollout_depth** (1, 3 calibrated, 5; n_rollout fixed at the calibrated 2
  here): real effect, **direction flipped again** from the (now superseded)
  n_rollout=3-fixed measurement - at n_rollout=2, rollout_depth=5 is fastest
  (0.73 by iteration 2, 1.06 by iteration 3) and rollout_depth=1 is slowest
  (0.33/0.39/0.40 at iterations 2/3/5, only catching up by iteration 10). This
  direction is evidently sensitive to which n_rollout the sub-sweep holds
  fixed, not just the starting-material distance - worth keeping in mind
  before treating either direction as a stable conclusion.
- **termination_limit** (10, 25 calibrated, 50): **real effect on search
  quality, not just search length** (confirms the earlier d=6/n_rollout=1
  finding, now even more consistent) - limit=10 lets the search halt before
  finding the true optimum in **5/5 seeds** (stuck at 1.1486 instead of
  1.1664), while limit=25 and limit=50 reach the true optimum in 5/5 seeds.
  Mean iterations completed (of a 500-iteration budget): 170.4 (limit=10),
  425.6 (limit=25), 500.0/never-terminates (limit=50) - see
  `iterations_vs_termination_limit.png`.

## Why distance-to-optimum changes which parameters matter

This isn't a bug - it follows directly from how expansion works in
`mcts_crystal/node.py`/`mcts.py`. A node's `expandable` flag stays `True`
until **every** child in its `expansion_list` has been added to the tree via
a uniform-random draw (`expansion_simulation`'s `random.choice` over the
remaining expansion list) - and `selection_mode`/`exploration_constant` only
govern the choice *among already-expanded children* in `select_node()`,
which only runs once a node is no longer `expandable`. With only 8 immediate
moves from any node in this u_only search space (4 transition-metal options x
2 Group IV chain-neighbors x 1 f-block option), a search starting within 1-2
hops of the optimum (the old Cr6Sn6U, d=3 baseline) typically finds it purely
through this mode-independent exhaustion phase, before `selection_mode`/`c`/
`termination_limit` ever get a chance to matter. Starting at d=6 (the current
Pd6Ge6U baseline) means many more hops - and therefore many more calls to
`select_node()` - are needed, which is exactly why `selection_mode` (via
`boltzmann`'s sampling failure mode) and `termination_limit` (more
opportunities to halt before arrival) now show real effects, while `c` still
doesn't (UCB1-family modes apparently still pick the right child regardless
of `c`'s magnitude in this search space). `starting_material` and
`n_rollout`/`rollout_depth` show real effects at both distances because they
act *during* (or before) that same early phase, not after it.

## Files

```
sensitivity_studies/
  scripts/
    common.py                     # shared setup + replicate-running helper
    sweep_c.py
    sweep_selection_mode.py
    sweep_rollout_params.py        # n_rollout, rollout_depth, termination_limit
    plot_sweep.py                  # generic convergence-curve figure (3x3in, 10pt)
    plot_termination_iterations.py # iterations-completed bar chart for termination_limit
    run_all.sh                      # reproduces everything except starting_material_sweep
  results/
    c_sweep/
    starting_material_sweep/        # generated by analysis/ehull_rdos_u_only_study/sweep_starting_material.py
    selection_mode_sweep/
    rollout_params_sweep/
```

Each `results/<sweep>/` directory has a `convergence_data*.csv` (long-format:
`value, seed, iteration, best_reward`) and the corresponding `.png` figure(s).
