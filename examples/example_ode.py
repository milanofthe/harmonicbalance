import matplotlib.pyplot as plt
import numpy as np

from harmonicbalance.fourier import Fourier
from harmonicbalance.solvers import fouriersolve_ode


#duffing parameters 
m, c, d, g, p = 1.0, 2.0, 0.3, 1.4, 5.0


#excitation
U = Fourier(omega=1, n=7) 
U[1] = 1 #fundamental frequency cos term


#right hand side of duffing ode
def ode_duffing(X):
    return [X[1], -c/m * X[1] - d/m * X[0] - g/m * X[0]**3 + p * U]


#initial guess (just use the excitation)
X0s = [U.copy().dt(), U.copy()]


# solve -> minimize residual
[X_sol, V_sol], _sol = fouriersolve_ode(ode_duffing, X0s, method="lm", use_jac=False)


print(_sol)

# Evaluate the solution
t_values = np.linspace(0, X_sol._T(), 1000)
x_values = X_sol.evaluate(t_values) 
v_values = V_sol.evaluate(t_values)


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