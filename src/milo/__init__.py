# milo/__init__.py

"""
MILO: detecting Microsatellite Instability using LOng-deletin signatutre

This package provides tools to predict microsatellite instability (MSI) status
from 83-channel indel mutation profiles.
"""

__version__ = "1.8.0"
__author__ = "Qingli Guo"

# Expose key classes and functions for easier library usage
from .core import MILO
from .plots import plot_id83_profile, plot_id83_comparison
