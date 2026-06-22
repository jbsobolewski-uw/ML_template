"""
src/__init__.py
---------------
Framework root module initialization.
"""

import os
import sys

# Add the executor directory to sys.path to bridge the internal absolute
# imports used within the executor package (e.g., 'import accelerator')
# with the parent-level package architecture.
_base_dir = os.path.dirname(os.path.abspath(__file__))
_executor_dir = os.path.join(_base_dir, "executor")

if _executor_dir not in sys.path:
    sys.path.insert(0, _executor_dir)