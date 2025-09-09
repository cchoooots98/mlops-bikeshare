# test/conftest.py
# Purpose: Ensure tests can import packages under the 'src' directory.
# This adds the repository root to sys.path so "import src. ..." works.

import os
import sys

# Compute the repository root (parent of the 'test' directory)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Prepend the repo root to sys.path if it is not already there
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
