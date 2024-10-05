import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from harmonicbalance.fourier import Fourier
from harmonicbalance.solvers import fouriersolve_autonomous

#vanderpol parameter
mu = 5.0

# Residual function for Van der Pol oscillator
def residual_vanderpol(X):
    return X.dt().dt() - mu * (1 - X**2) * X.dt() + X

# Initial guess (can be a sine wave)
omega = 1
X0 = Fourier(omega=omega, n=100)
X0[1] = 5  # Fundamental frequency cos term


# solve -> minimize residual
X_sol, _sol = fouriersolve_autonomous(residual_vanderpol, X0, method="lm", use_jac=False)

print(_sol)

print(X_sol.omega)


# Evaluate the solution
t_values = np.linspace(0, X_sol._T(), 5000)
x_values = X_sol.evaluate(t_values) 
v_values = X_sol.dt().evaluate(t_values)


#initial condition for ODE
x0 = [x_values[0], v_values[0]]

# Reference solution
def ode_vanderpol(x, t):
    return np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]])

solution = odeint(ode_vanderpol, x0, t_values, atol=1e-12)
x_ref, v_ref = solution.T


# Plot the solution (time domain)
fig, ax = plt.subplots(tight_layout=True)

ax.plot(t_values, x_values)
ax.plot(t_values, v_values)
ax.plot(t_values, x_ref[::-1], "--")
ax.plot(t_values, v_ref[::-1], "--")
ax.set_xlabel("Time")
ax.set_ylabel("Response")



# Plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True)
ax.plot(x_values, v_values)
ax.plot(x_ref, v_ref, "--")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()