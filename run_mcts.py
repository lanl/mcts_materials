#!/usr/bin/env python3
"""
Thin compatibility wrapper for `python run_mcts.py [OPTIONS]`.

The implementation lives in mcts_crystal/cli.py so it ships as part of the
installable package (`pip install -e .` also gives you the `mcts-run` console
command). This wrapper exists so `python run_mcts.py ...` keeps working exactly
as before, with or without an editable install. See `python run_mcts.py --help`
or mcts_crystal/cli.py for usage.
"""

import sys
from pathlib import Path

# Allow running directly from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).parent))

from mcts_crystal.cli import main

if __name__ == "__main__":
    sys.exit(main())
