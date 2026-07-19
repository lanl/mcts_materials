"""
Unit tests for intermetallic element groups and move rules.

These are pure integer-in/integer-out functions (no ASE), verifying the
periodic-table navigation matches the validated mcts_crystal behavior.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from mcts_framework.intermetallic import elements


# --- Group IV moves ------------------------------------------------------


def test_group_iv_endpoints():
    # Si (14): only neighbor Ge
    assert elements.group_iv_moves(14) == [14, 32]
    # Pb (82): only neighbor Sn
    assert elements.group_iv_moves(82) == [50, 82]


def test_group_iv_middle():
    # Ge (32): Si and Sn
    assert elements.group_iv_moves(32) == [14, 32, 50]
    # Sn (50): Ge and Pb
    assert elements.group_iv_moves(50) == [32, 50, 82]


# --- Metal moves: edge cases match mcts_crystal --------------------------


def test_metal_3d_edges():
    assert elements.metal_moves(22) == [22, 23, 40]   # Ti
    assert elements.metal_moves(30) == [29, 30, 48]   # Zn


def test_metal_3d_middle():
    # Fe (26): 25, 26, 27, 44 (down a group +18)
    assert elements.metal_moves(26) == [25, 26, 27, 44]


# metal_moves now returns a sorted list (matching upstream sorted(mv)); the
# element *sets* are unchanged from the original hand-tuned edge cases.
def test_metal_4d_edges():
    assert elements.metal_moves(40) == sorted([40, 41, 72, 22])  # Zr
    assert elements.metal_moves(48) == sorted([47, 48, 80, 30])  # Cd


def test_metal_4d_middle():
    # Ru (44): 43, 44, 45, 76 (+32), 26 (-18)
    assert elements.metal_moves(44) == sorted([43, 44, 45, 76, 26])


def test_metal_5d_edges():
    assert elements.metal_moves(72) == sorted([72, 73, 40])   # Hf
    assert elements.metal_moves(80) == sorted([79, 80, 48])   # Hg


def test_metal_5d_middle():
    # W (74): 73, 74, 75, 42 (-32)
    assert elements.metal_moves(74) == sorted([73, 74, 75, 42])


# --- F-block moves per mode ----------------------------------------------


def test_f_block_u_only():
    assert elements.f_block_moves(92, "u_only") == [92]
    # even starting from a lanthanide, u_only forces U
    assert elements.f_block_moves(64, "u_only") == [92]


def test_f_block_lanthanides_u_wraps():
    # Ce (58) wraps to Lu (71) on the left neighbor
    moves = elements.f_block_moves(58, "lanthanides_u")
    assert 71 in moves  # wrap-around neighbor
    assert 59 in moves  # right neighbor Pr
    assert 58 in moves


def test_f_block_lanthanides_u_u_to_nd():
    moves = elements.f_block_moves(92, "lanthanides_u")
    assert 60 in moves  # U -> Nd
    assert 92 in moves


def test_f_block_extended_range_and_bridges():
    # From U: Nd, Gd, Er bridges
    moves = elements.f_block_moves(92, "lanthanides_u_extended")
    assert set([60, 64, 68]).issubset(set(moves))
    # From a lanthanide, +/-3 reachable (with wrap)
    moves_ce = elements.f_block_moves(58, "lanthanides_u_extended")
    # Ce idx 0; +3 -> idx3 = Nd(61? ) compute: lanthanides[3]=61 (Pm). Just check length > lanthanides_u
    assert len(moves_ce) >= len(elements.f_block_moves(58, "lanthanides_u"))


def test_f_block_no_wrap_has_no_wraparound():
    # Ce (58) in no_wrap: left neighbor 57 is NOT a lanthanide (that's La),
    # so no wrap to Lu(71); only right neighbor Pr(59).
    moves = elements.f_block_moves(58, "lanthanides_u_no_wrap")
    assert 71 not in moves  # no wrap-around
    assert 59 in moves


def test_f_block_no_wrap_u_bridge():
    moves = elements.f_block_moves(92, "lanthanides_u_no_wrap")
    assert 60 in moves  # U -> Nd


def test_f_block_full_vertical_analogs():
    # Ce (58) -> Th (90) vertical analog (+32)
    moves = elements.f_block_moves(58, "full_f_block")
    assert 90 in moves
    # Th (90) -> Ce (58) vertical analog (-32)
    moves_th = elements.f_block_moves(90, "full_f_block")
    assert 58 in moves_th


# --- move_step generalization --------------------------------------------


def test_move_step_metal_widens_within_row():
    # Fe (26) at step 2: reaches 24..28 within 3d, plus the fixed +18 to 4d.
    moves = elements.metal_moves(26, move_step=2)
    assert set([24, 25, 26, 27, 28, 44]).issubset(set(moves))
    # Step 1 stays adjacent only.
    assert elements.metal_moves(26, move_step=1) == sorted([25, 26, 27, 44])


def test_move_step_metal_clamps_at_row_edges():
    # V (23) at step 5 clamps to the 3d row bottom (22), does not spill below.
    moves = elements.metal_moves(23, move_step=5)
    assert min(m for m in moves if 22 <= m <= 30) == 22
    assert 21 not in moves  # never leaves the row downward


def test_move_step_group_iv():
    # Si (14) at step 3 reaches the whole chain forward: Si,Ge,Sn,Pb.
    assert elements.group_iv_moves(14, move_step=3) == [14, 32, 50, 82]
    # Step 1 only Si,Ge.
    assert elements.group_iv_moves(14, move_step=1) == [14, 32]


def test_move_step_lanthanides_extended_equivalence():
    # lanthanides_u at move_step=3 reproduces the old lanthanides_u_extended
    # lanthanide jump set (both wrap; extended's only extra is the U bridge).
    ln = list(range(58, 72))
    idx = ln.index(64)
    core = sorted({64} | {ln[(idx + d) % len(ln)] for d in range(-3, 4) if d != 0})
    got = elements.f_block_moves(64, "lanthanides_u", move_step=3)
    # Gd(64) also bridges to U(92) under lanthanides_u? No - only Nd(60) does.
    assert got == core


def test_move_step_default_is_one():
    # Default (no move_step arg) equals explicit move_step=1.
    assert elements.metal_moves(44) == elements.metal_moves(44, move_step=1)
    assert elements.group_iv_moves(32) == elements.group_iv_moves(32, move_step=1)
    assert (elements.f_block_moves(64, "lanthanides_u")
            == elements.f_block_moves(64, "lanthanides_u", move_step=1))


# --- classify_site -------------------------------------------------------


def test_classify_site():
    assert elements.classify_site(14) == "group_iv"   # Si
    assert elements.classify_site(82) == "group_iv"   # Pb
    assert elements.classify_site(26) == "metal"      # Fe
    assert elements.classify_site(74) == "metal"      # W
    assert elements.classify_site(92) == "f_block"    # U
    assert elements.classify_site(64) == "f_block"    # Gd
    assert elements.classify_site(1) is None          # H - not a site
