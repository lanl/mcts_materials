#!/usr/bin/env python3
"""
Compatibility wrapper for the synthesis-planning CLI.

Examples:
    python run_mcts.py download-data
    python run_mcts.py prepare-data
    python run_mcts.py plan --target BaTiO3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from synthesis_planner.cli import main

if __name__ == "__main__":
    sys.exit(main())
