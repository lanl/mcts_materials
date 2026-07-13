"""Shared metadata for the U-only ehull_rdos study.

Single source of truth for the four U-compounds our experimentalists have
successfully synthesized in the past, used to flag the "Synth" column in
write_top15_table and the synthesized-compound overlay in plot_ehull_vs_rdos
(both in generate_figures.py). Dash "f_block-groupIV-metal" format - see
DoscarRewardLookup.convert_formula_to_doscar_format.
"""

SYNTHESIZED_COMPOUNDS = ['U-Sn-V', 'U-Sn-Nb', 'U-Ge-Cr', 'U-Ge-Co']
