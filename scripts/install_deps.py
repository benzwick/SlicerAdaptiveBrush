#!/usr/bin/env python
"""Install optional dependencies in Slicer Python environment."""

import subprocess
import sys

packages = ["scikit-image", "scikit-learn"]

print(f"Installing: {packages}")
sys.stdout.flush()

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        timeout=300,  # 5 minute timeout
    )
    print("Dependencies installed successfully")
except subprocess.TimeoutExpired:
    print("ERROR: Installation timed out")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"ERROR: Installation failed: {e}")
    sys.exit(1)
