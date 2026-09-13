"""Reproduction code for "Benefits of marriage as a search strategy"."""

from .affinity import GAUSSIAN, UNIFORM, DenseAffinity, LazyAffinity, make_affinity
from .model import SimulationResult, simulate, simulate_pair

__all__ = [
    "GAUSSIAN", "UNIFORM", "DenseAffinity", "LazyAffinity", "make_affinity",
    "SimulationResult", "simulate", "simulate_pair",
]
__version__ = "0.1.0"
