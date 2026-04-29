"""Shared fixtures for 4_env_test unit tests."""

import sys
import os

# Add src/ to path so imports work without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
