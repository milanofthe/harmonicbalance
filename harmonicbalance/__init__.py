"""HarmonicBalance - Object-based harmonic balance framework.

A minimalistic framework for calculating nonlinear periodic steady state
responses using an object-based harmonic balance approach.
"""

from __future__ import annotations

from harmonicbalance.fourier import Fourier
from harmonicbalance.predictorcorrector import PredictorCorrectorSolver
from harmonicbalance.solvers import (
    auto_jacobian,
    fouriersolve,
    fouriersolve_arclength,
    fouriersolve_autonomous,
    fouriersolve_autonomous_trajectory,
    fouriersolve_multi_autonomous_trajectory,
    fouriersolve_ode,
    numerical_jacobian,
    timer,
)

__version__ = "0.2.0"

__all__ = [
    "Fourier",
    "PredictorCorrectorSolver",
    "auto_jacobian",
    "fouriersolve",
    "fouriersolve_arclength",
    "fouriersolve_autonomous",
    "fouriersolve_autonomous_trajectory",
    "fouriersolve_multi_autonomous_trajectory",
    "fouriersolve_ode",
    "numerical_jacobian",
    "timer",
]
