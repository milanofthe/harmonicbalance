# Harmonic Balance implementation

This is a minimalistic framework for solving periodic steady state and limit cycles of nonlinear dynamic systems using the harmonic balance method.
At the core is a `Fourier` class that represents a fourier series and implements basic operators such as addition, and multiplication as well as general nonlinearities and time domain differentiation.
This enables an easy formulation of the residual function of the nonlinear dynamics problem.

There are a lot of common examples for nonlinear dynamical systems in the `examples` directory.

## Example Duffing Oscillator
For example, lets solve the steady state of the driven duffing oscillator which is a basic damped oscillator with an additional cubic stiffness ($g$) term.

$$
m \ddot{x} + c \dot{x} + dx + g x^3 = p \cos(\omega_0 t)
$$


```python
import numpy as np
import matplotlib.pyplot as plt

#import the 'Fourier' class and the solver wrapper
from harmonicbalance.fourier import Fourier
from harmonicbalance.solvers import fouriersolve

#duffing parameters 
m, c, d, g, p = 1.0, 2.0, 0.3, 1.4, 5.0

#excitation, this is the cos term
U = Fourier(omega=1, n=5) 
U[1] = 1 #fundamental frequency cos term

#residual for duffing oscillator using the 'Fourier' class
def residual_duffing(X):
    return  m * X.dt().dt() + c * X.dt() + d * X + g * X**3 - p * U

#initial guess (just use the excitation)
X0 = U.copy()

# solve -> minimize residual, returns another 'Fourier' object
X_sol, _sol = fouriersolve(residual_duffing, X0, method="hybr")
```

    runtime of 'fouriersolve' : 8.30149999819696 ms
    


```python
#examine the 'Fourier' object
print(X_sol)
```

    Fourier(coeff_dc=4.66229072966336e-10, coeffs_cos=[ 1.08236692e+00 -1.51914927e-10  4.60358995e-02  3.88449135e-11
     -1.87784775e-02], coeffs_sin=[-1.15139578e+00  2.06593164e-10 -1.95966475e-01  3.00183167e-11
     -1.75997843e-02], omega=1, n=5)
    

The `Fourier` class also implements some methods for easy time domain evaluation, which we can use to plot the time domain response and the trajectory in the phase space.


```python
# Evaluate the solution
t = np.linspace(0, X_sol._T(), 1000)
x = X_sol.evaluate(t) 
v = X_sol.dt().evaluate(t)
```


```python
# Plot the solution (time domain)
fig, ax = plt.subplots(tight_layout=True, figsize=(8,5), dpi=120)

ax.plot(t, x, label="x")
ax.plot(t, v, label="v")

ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Response");
```


    
![png](README_files/README_6_0.png)
    



```python
# Plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True, figsize=(8,5), dpi=120)

ax.plot(x, v)

ax.set_xlabel("x")
ax.set_ylabel("v");
```


    
![png](README_files/README_7_0.png)
    


## Example Predictor-Corrector Solver

The duffing oscillator is known for its bifurcation, which means that there exist multiple solutions for the same set of parameters. Typically these solutions are hard to find numerically by conventional methods (for example numerical integration). The harmonic balance method can be used to obtain these solutions by continuing the solution curve for a given parameter variation (in this case the excitation frequency $\omega_0$). This results in the well known backbone curve.

This package also implements a simple `PredictorCorrector` solver that uses the secant method for the predictor step and corrects the solution using the harmonic balance method with an additional constraint (arclength / hypersphere).



```python
from harmonicbalance.predictorcorrector import PredictorCorrectorSolver

#duffing parameters
m, c, d, g, p = 1, 0.2, 1, 2, 3

#excitation, this is the cos term
U = Fourier(omega=1, n=5) 
U[1] = 1 #fundamental frequency cos term

#residual for duffing oscillator
def residual_duffing(X):
    return m * X.dt().dt() + c * X.dt() + d * X + g * X**3 - p * U

#initial guess (just use the excitation)
X0 = U.copy()

#initialize the predictor-corrector solver
PCS = PredictorCorrectorSolver(residual_duffing, 
                               X0, 
                               alpha_start=X0.omega, 
                               alpha_end=5, 
                               alpha_step=0.1, 
                               method="hybr")

#find solutions in specified range
solutions = PCS.solve()
```

    runtime of 'fouriersolve' : 3.5116998478770256 ms
    runtime of 'fouriersolve_arclength' : 7.075299974530935 ms
    runtime of 'fouriersolve_arclength' : 2.8400998562574387 ms
    runtime of 'fouriersolve_arclength' : 3.4962999634444714 ms
    runtime of 'fouriersolve_arclength' : 2.8699999675154686 ms
    runtime of 'fouriersolve_arclength' : 4.9443000461906195 ms
    runtime of 'fouriersolve_arclength' : 2.8006997890770435 ms
    runtime of 'fouriersolve_arclength' : 5.312999943271279 ms
    ...
    runtime of 'fouriersolve_arclength' : 4.789300030097365 ms
    runtime of 'fouriersolve_arclength' : 4.773000022396445 ms
    runtime of 'fouriersolve_arclength' : 6.96209981106
    runtime of 'fouriersolve_arclength' : 2.503999974578619 ms
    runtime of 'fouriersolve_arclength' : 2.495999913662672 ms
    runtime of 'solve' : 648.698000004515 ms
    





```python
#find specific solutions at a given frequency from backbone curve
specific_omega = 3
specific_solutions = PCS.solve_specific(specific_omega)
```

    runtime of 'fouriersolve' : 4.860799992457032 ms
    runtime of 'fouriersolve' : 3.131199860945344 ms
    runtime of 'fouriersolve' : 2.359899925068021 ms
    runtime of 'solve_specific' : 10.667799971997738 ms
    


```python
#plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True, figsize=(8,5), dpi=120)

#solution curve
ax.plot([s.omega for s in PCS.solutions], [s.amplitude() for s in solutions], ".-")

#specific solutions
ax.axvline(specific_omega, color="k")
for s in specific_solutions:
    ax.plot(s.omega, s.amplitude(), "o", color="tab:red")

ax.set_xlabel("Omega")
ax.set_ylabel("Amplitude");
```


    
![png](README_files/README_11_0.png)
    

