"""
Test script to verify f-block element substitution functionality.
"""

import sys
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import MCTSTreeNode, MCTS, MaceEnergyCalculator
from ase.io import read


def main():
    """Test f-block element substitution."""
    print("=" * 60)
    print("TESTING F-BLOCK ELEMENT SUBSTITUTION")
    print("=" * 60)
    
    # Load the test structure (contains U = atomic number 92)
    cif_file = Path(__file__).parent / "mat_Pb6U1W6_sg191.cif"
    atoms = read(str(cif_file))
    
    print(f"1. Initial structure: {atoms.get_chemical_formula()}")
    print(f"   Atomic numbers: {list(set(atoms.get_atomic_numbers()))}")
    
    # Test both modes
    print(f"\n2. Testing U-only mode (default):")
    node_u_only = MCTSTreeNode(atoms, f_block_mode='u_only')
    node_u_only._determine_possible_moves()
    print(f"   F-block moves: {getattr(node_u_only, 'f_block_move', 'None')}")
    
    print(f"\n3. Testing full f-block mode:")
    node_full = MCTSTreeNode(atoms, f_block_mode='full_f_block')
    node_full._determine_possible_moves()
    print(f"   F-block moves: {getattr(node_full, 'f_block_move', 'None')}")
    
    # Use the full mode node for the rest of the tests
    node = node_full
    
    print(f"\n4. F-block element detected: {node.f_block}")
    print(f"   F-block possible moves: {getattr(node, 'f_block_move', 'None')}")
    
    # Show what elements are available
    if hasattr(node, 'f_block_move') and node.f_block_move:
        print(f"\n5. Available f-block substitutions:")
        element_names = {
            57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
            65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
            89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu'
        }
        
        for atomic_num in node.f_block_move:
            symbol = element_names.get(atomic_num, f'Z={atomic_num}')
            print(f"   - {atomic_num}: {symbol}")
            
        # Test expansion with f-block elements
        print(f"\n6. Testing expansion with f-block substitutions...")
        node.expand()
        
        print(f"   Total possible moves: {len(node.expansion_list)}")
        print(f"   Transition metal options: {len(node.metal_move)}")
        print(f"   Group IV options: {len(node.g_iv_move)}")  
        print(f"   F-block options: {len(node.f_block_move)}")
        print(f"   Expected combinations: {len(node.metal_move)} × {len(node.g_iv_move)} × {len(node.f_block_move)} = {len(node.metal_move) * len(node.g_iv_move) * len(node.f_block_move)}")
        
        # Show some example substitutions
        print(f"\n7. Example f-block substitutions:")
        for i, child in enumerate(node.expansion_list[:10]):  # Show first 10
            formula = child.get_chemical_formula()
            f_elements_in_child = [num for num in child.atoms.get_atomic_numbers() 
                                 if 57 <= num <= 71 or 89 <= num <= 94]
            if f_elements_in_child:
                f_symbol = element_names.get(f_elements_in_child[0], f'Z={f_elements_in_child[0]}')
                print(f"   {i+1}. {formula} (contains {f_symbol})")
    else:
        print("   No f-block elements found - this is unexpected!")
        
    print(f"\n8. Verification - elements 95-103 excluded:")
    excluded_elements = list(range(95, 104))  # Am(95) through Lr(103)
    excluded_names = ['Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    
    for i, (num, name) in enumerate(zip(excluded_elements, excluded_names)):
        included = num in getattr(node, 'f_block_move', [])
        status = "❌ EXCLUDED" if not included else "⚠️  INCLUDED (ERROR!)"
        print(f"   {num} ({name}): {status}")
        
    print("\n" + "=" * 60)
    print("✅ F-BLOCK SUBSTITUTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()