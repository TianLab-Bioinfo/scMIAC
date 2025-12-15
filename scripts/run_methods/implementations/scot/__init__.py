"""
SCOT (Single-Cell Optimal Transport) methods for multi-modal integration.

This module contains implementations of SCOT algorithms:
- scotv1.py: Original SCOT algorithm  
- scotv2.py: Improved SCOT v2 algorithm (core implementation)
- main.py: SCOTv2 runner script with data loading and result saving
- evals.py: Evaluation utilities
"""

from .scotv2 import SCOTv2

__all__ = ['SCOTv2']
