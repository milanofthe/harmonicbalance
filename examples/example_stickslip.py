import matplotlib.pyplot as plt
import numpy as np

from harmonicbalance.fourier import Fourier
from harmonicbalance.solvers import fouriersolve

#simulation and model parameters
m = 20.0    # mass
k = 70.0    # spring constant
d = 10.0    # spring damping
mu = 1.5    # friction coefficient
g = 9.81    # gravity
v = 3.0     # belt velocity magnitude

#fundamental frequency of mechanical oscillator
omega_0 = np.sqrt(k/m)


#excitation
U = Fourier(omega=omega_0/10, n=20)
U[1] = v #fundamental frequency cos term


def residual_stickslip(X):
    F_c = m * mu * g * (X.dt() - U).nonlinearity(lambda x: np.tanh(100*x))
    return m * X.dt().dt() + d * X.dt() + k * X + F_c


#initial guess (just use the excitation)
X0 = U.copy()

# solve -> minimize residual
# X_sol, _sol = fouriersolve(residual_stickslip, X0, method="lm", use_jac=True)
X_sol, _sol = fouriersolve(residual_stickslip, X0, method="krylov", use_jac=False)

print(_sol)


# Evaluate the solution
t_values = np.linspace(0, 2*X_sol._T(), 2000)
x_values = X_sol.evaluate(t_values)[::-1]
v_values = X_sol.dt().evaluate(t_values)[::-1]
u_values = U.evaluate(t_values)[::-1]


# Plot the solution (time domain)
fig, ax = plt.subplots(tight_layout=True)

ax.plot(t_values, x_values)
ax.plot(t_values, v_values)
ax.plot(t_values, u_values)
ax.set_xlabel("Time")
ax.set_ylabel("Response")


# Plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True)
ax.plot(x_values, v_values)
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()