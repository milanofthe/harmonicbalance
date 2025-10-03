"""Solver wrappers to work with Fourier objects.

This module provides various solver wrappers that interface scipy optimization
routines with the Fourier class for solving nonlinear periodic problems.

Author: Milan Rother 2024
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import root

from .fourier import Fourier

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.optimize import OptimizeResult


# helper ============================================================================================

def timer(func: Callable) -> Callable:
    """Decorator for performance evaluation.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints runtime
    """
    def wrap(*args: Any, **kwargs: Any) -> Any:
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        print(f"runtime of '{func.__name__}' : {(t2-t1)*1e3} ms")
        return result
    return wrap


def numerical_jacobian(
    func: Callable,
    x: float | NDArray[np.floating],
    h: float = 1e-8,
) -> float | NDArray[np.floating]:
    """Numerically compute the Jacobian using central differences.

    The default stepsize h=1e-8 is chosen where the truncation error
    of central differences balances with the machine accuracy of 64-bit
    floating point numbers.

    Args:
        func: Function to compute Jacobian for
        x: Value at which the Jacobian is evaluated
        h: Step size for central differences

    Returns:
        Jacobian matrix or gradient for scalar input
    """
    # Catch scalar case (gradient)
    if np.isscalar(x):
        return 0.5 * (func(x + h) - func(x - h)) / h

    # Perturbation matrix and jacobian
    e = np.eye(len(x)) * h
    return 0.5 * np.array([func(x_p) - func(x_m) for x_p, x_m in zip(x + e, x - e)]).T / h


def auto_jacobian(func: Callable) -> Callable:
    """Wrap function to compute Jacobian with respect to first argument.

    This is intended to compute the Jacobian 'jac(x, u, t)' of
    the right hand side function 'func(x, u, t)' of numerical
    integrators with respect to 'x'.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that computes the Jacobian
    """
    def wrap_func(*args: Any) -> NDArray[np.floating]:
        _x, *_args = args
        return numerical_jacobian(lambda x: func(x, *_args), _x)
    return wrap_func



# solver wrappers ===================================================================================

@timer
def fouriersolve(
    residual_func: Callable[[Fourier], Fourier],
    X0: Fourier,
    use_jac: bool = True,
    **solverargs: Any,
) -> tuple[Fourier, OptimizeResult]:
    """Solve nonlinear system using Fourier objects.

    Wrapper for scipy root solver to handle Fourier objects directly.

    Args:
        residual_func: Function that defines the residual
        X0: Initial guess for Fourier variable/coefficients
        use_jac: Use numerical central differences Jacobian for solver
        **solverargs: Additional arguments passed to scipy.optimize.root

    Returns:
        Tuple of (solution Fourier object, optimization result)
    """
    # Wrapper for fourier coefficients
    def residual_func_num(X_coeffs: NDArray[np.floating]) -> NDArray[np.floating]:
        X = Fourier.from_coeffs(X_coeffs, X0.omega)
        R = residual_func(X)
        return R.coeffs()

    # Numerical jacobian using central differences
    residual_func_num_jac = auto_jacobian(residual_func_num) if use_jac else None

    # Actual solver call
    _sol = root(residual_func_num, X0.coeffs(), jac=residual_func_num_jac, **solverargs)

    return Fourier.from_coeffs(_sol.x, X0.omega), _sol


@timer
def fouriersolve_autonomous(residual_func, X0, use_jac=False, **solverargs):
    """
    wrapper for scipy root solver to handle 'Fourier' objects directly
    but with the fundamental frequency as a free parameter for the solver to optimize

    INPUTS:
        residual_func : (callable) function that defines residual
        X0            : (Fourier) initial guess for fourier variable / coefficients
        use_jac       : (bool) use numerical central differences jacobian for solver
    """

    #wrapper for fourier coefficients
    def residual_func_num(params):
        X = Fourier.from_params(params)
        R = residual_func(X)
        return np.append(R.coeffs(), 0.0)

    #numerical jacobian using central differences
    residual_func_num_jac = auto_jacobian(residual_func_num) if use_jac else None

    #actual solver call
    _sol = root(residual_func_num, X0.params(), jac=residual_func_num_jac, **solverargs)

    #return fourier object from solution
    return Fourier.from_params(_sol.x), _sol


@timer
def fouriersolve_autonomous_trajectory(residual_func, X0, use_jac=False, **solverargs):
    """
    wrapper for scipy root solver to handle 'Fourier' objects directly
    but with the fundamental frequency as a free parameter for the solver to optimize
    and the trajectory passing through the initial condition as part of the residual

    INPUTS:
        residual_func : (callable) function that defines residual
        X0            : (Fourier) initial guess for fourier variable / coefficients
        use_jac       : (bool) use numerical central differences jacobian for solver
    """

    #wrapper for fourier coefficients
    def residual_func_num(params):
        X = Fourier.from_params(params)
        R = residual_func(X)
        dX = X - X0
        tr = dX(0.0)**2 # trajectory
        return np.append(R.coeffs(), tr)

    #numerical jacobian using central differences
    residual_func_num_jac = auto_jacobian(residual_func_num) if use_jac else None

    #actual solver call
    _sol = root(residual_func_num, X0.params(), jac=residual_func_num_jac, **solverargs)

    #return fourier object from solution
    return Fourier.from_params(_sol.x), _sol


@timer
def fouriersolve_arclength(residual_func, X0, Xref, ds, use_jac=True, **solverargs):

    """
    wrapper for scipy root solver to handle 'Fourier' objects directly
    but with the fundamental frequency as a free parameter for the
    solver to optimize and an additional constraint that enforces
    a distance (arclength) from a reference solution

    INPUTS:
        residual_func : (callable) function that defines residual
        X0            : (Fourier) initial guess for fourier variable / coefficients
        Xref          : (Fourier) reference solution for arclength enforcement
        ds            : (float) arclength that should be enforced
        use_jac       : (bool) use numerical central differences jacobian for solver
    """

    #wrapper for fourier coefficients
    def residual_func_num(params):
        X = Fourier.from_params(params)
        R = residual_func(X)

        #use all params for arclength
        p = np.linalg.norm(X.params() - Xref.params())**2 - ds**2

        return np.append(R.coeffs(), p)

    #numerical jacobian using central differences
    residual_func_num_jac = auto_jacobian(residual_func_num) if use_jac else None

    #actual solver call
    _sol = root(residual_func_num, X0.params(), jac=residual_func_num_jac, **solverargs)

    #return fourier object from solution
    return Fourier.from_params(_sol.x), _sol


@timer
def fouriersolve_ode(ode_func, X0s, use_jac=False, **solverargs):
    """
    Wrapper for scipy root solver to handle 'Fourier' objects directly.
    Instead of the residual, this function wraps the right hand side of
    an ODE and constructs the residual function from it.

    extended to be applicable to coupled equations

    INPUTS:
        ode_func : (callable) right hand side function of ode
        X0s      : (list[Fourier]) initial guesses for each variable
        use_jac  : (bool) use numerical central differences jacobian for solver
    """

    #number of fourier variables
    nx = len(X0s)
    omega = X0s[0].omega

    #define residual func from ode_func
    def residual_func(Xs):

        #evaluate ode_func
        dXs = ode_func(Xs)

        #build residuals  dx/dt - func(x) = 0
        return [x.dt() - dx for x, dx in zip(Xs, dXs)]

    #wrapper for fourier coefficients
    def residual_func_num(coeffs):

        #set fourier objects from paramsters
        Xs = [Fourier.from_coeffs(cs, omega) for cs in np.split(coeffs, nx)]

        #residuals as fourier objects
        Rs = residual_func(Xs)

        #conversion to coefficients
        return np.hstack([r.coeffs() for r in Rs])

    #numerical jacobian using central differences
    residual_func_num_jac = auto_jacobian(residual_func_num) if use_jac else None

    #initial parameters
    initial_coeffs = np.hstack([x.coeffs() for x in X0s])

    #actual solver call
    _sol = root(residual_func_num, initial_coeffs, jac=residual_func_num_jac, **solverargs)

    #set coefficients and frequency for each variable
    Xs = [Fourier.from_coeffs(cs, omega) for cs in np.split(_sol.x, nx)]

    return Xs, _sol


@timer
def fouriersolve_multi_autonomous_trajectory(residual_func, X0s, Xrefs, use_jac=False, **solverargs):
    """
    wrapper for scipy root solver to handle 'Fourier' objects directly
    but with the fundamental frequency as a free parameter for the solver to optimize
    and the trajectory passing through the initial condition as part of the residual

    extended to be applicable to coupled equations

    INPUTS:
        residual_func : (callable) function that defines residual
        X0s           : (list[Fourier]) initial guesses for each variable
        Xrefs         : (list[Fourier]) fourier objects for trajectory enforcement
        use_jac       : (bool) use numerical central differences jacobian for solver
    """

    #number of fourier variables
    nx = len(X0s)

    #wrapper for fourier coefficients
    def residual_func_num(params):

        #unpack parameters
        coeffs, omega = params[:-1], params[-1]

        #set fourier objects from paramsters
        Xs = [Fourier.from_coeffs(cs, omega) for cs in np.split(coeffs, nx)]

        #residual component of trajectory enforcement
        dXs = [x - xr for x, xr in zip(Xs, Xrefs)]
        tr = sum([dx(0.0)**2 for dx in dXs])

        #residuals as fourier objects
        Rs = residual_func(Xs)

        #conversion to coefficients
        rs = np.hstack([r.coeffs() for r in Rs])

        #adding trajectory enforcement
        return np.append(rs, tr)

    #numerical jacobian using central differences
    residual_func_num_jac = auto_jacobian(residual_func_num) if use_jac else None

    #initial parameters
    initial_params = np.hstack([x.coeffs() for x in X0s] + [X0s[0].omega])

    #actual solver call
    _sol = root(residual_func_num, initial_params, jac=residual_func_num_jac, **solverargs)

    #unpack solution
    coeffs, omega = _sol.x[:-1], _sol.x[-1]

    #set coefficients and frequency for each variable
    Xs = [Fourier.from_coeffs(cs, omega) for cs in np.split(coeffs, nx)]

    return Xs, _sol
