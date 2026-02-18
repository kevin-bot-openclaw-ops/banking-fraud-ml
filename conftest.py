"""
Pytest configuration for banking-fraud-ml test suite.
"""
import pytest
import sys
import os

# Ensure src/ is importable from all test files
sys.path.insert(0, os.path.dirname(__file__))
