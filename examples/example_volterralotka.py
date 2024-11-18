import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


from harmonicbalance.fourier import Fourier
from harmonicbalance.solvers import fouriersolve_multi_autonomous_trajectory

#parameters 
alpha = 2.65  # growth rate of prey
beta  = 2.1   # predator sucess rate
delta = 1.5   # predator efficiency
gamma = 1.4   # death rate of predators


# Residual function
def residual_volterralotka(X):
    R1 = X[0].dt() - alpha * X[0] + beta * X[0] * X[1]
    R2 = X[1].dt() + gamma * X[1] - delta * X[0] * X[1] 
    return [R1, R2]

# equilibrium
x1_dc = gamma/delta
x2_dc = alpha/beta

#number of harmonics
n = 13

# Initial guess
X01 = Fourier(omega=2, n=n)
X02 = X01.copy()

X01[0] = x1_dc 
X01[1] = 2 #cos component

X02[0] = x2_dc
X02[1+n] = 2 #sin component

# Solve
[X1, X2], _sol = fouriersolve_multi_autonomous_trajectory(
    residual_volterralotka, 
    [X01, X02], 
    [X01, X02], 
    method="lm", 
    use_jac=True
    )

print(_sol)

print(X1.omega)


# Evaluate the solution
t_values = np.linspace(0, X1._T(), 1000)
x1_values = X1(t_values) 
x2_values = X2(t_values)


# Reference solution
def ode_volterralotka(x, t):
    return np.array([ alpha*x[0] - beta*x[0]*x[1], - gamma*x[1] + delta*x[0]*x[1]])

solution = odeint(ode_volterralotka, [X01.evaluate(0.0), X02.evaluate(0.0)], t_values, atol=1e-12)
x1_ref, x2_ref = solution.T



# Plot the solution (time domain)
fig, ax = plt.subplots(tight_layout=True)

ax.plot(t_values, x1_values, label="harmonic balance X1")
ax.plot(t_values, x2_values, label="harmonic balance X2")
ax.plot(t_values, x1_ref[::-1], "--", label="reference X1")
ax.plot(t_values, x2_ref[::-1], "--", label="reference X2")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Response")


# Plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True)
ax.plot(x1_dc, x2_dc, ".", label="equilibrium")
ax.plot(x1_values, x2_values, "-", label="harmonic balance")
ax.plot(X01.evaluate(t_values), X02.evaluate(t_values), ":", label="linear")
ax.plot(x1_ref, x2_ref, "--", label="reference")
ax.axvline(0.0, color="k")
ax.axhline(0.0, color="k")
ax.legend()
ax.set_xlabel("predators")
ax.set_ylabel("prey")



# # Plot spectrum
omegas1, amplitudes1 = X1.spectrum()
omegas2, amplitudes2 = X2.spectrum()

fig, ax = plt.subplots(tight_layout=True)
ax.plot(omegas1, abs(amplitudes1), "o", label="|X1|")
ax.plot(omegas2, abs(amplitudes2), "o", label="|X2|")
ax.legend()
ax.set_xlabel("omega")


plt.show()