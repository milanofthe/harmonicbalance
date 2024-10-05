import matplotlib.pyplot as plt
import numpy as np

from harmonicbalance.fourier import Fourier
from harmonicbalance.solvers import fouriersolve_autonomous_trajectory

# parameters
g, L = 9.81, 1.0
omega = np.sqrt(g / L)

# Residual function
def residual_pendulum(Theta):
    return Theta.dt().dt() + (g / L) * Theta.nonlinearity(np.sin)

# Initial guess
X0 = Fourier(omega=omega, n=20)
X0[1] = 0.99*np.pi 


# Solve
X_sol, _sol = fouriersolve_autonomous_trajectory(residual_pendulum, X0, method="lm")

print(_sol)

# Evaluate the solution
t_values = np.linspace(0, X_sol._T(), 1000)
x_values = X_sol.evaluate(t_values) 
v_values = X_sol.dt().evaluate(t_values)


# Plot the solution (time domain)
fig, ax = plt.subplots(tight_layout=True)

ax.plot(t_values, x_values)
ax.plot(t_values, v_values)
ax.set_xlabel("Time")
ax.set_ylabel("Response")


# Plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True)
ax.plot(x_values, v_values)
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()