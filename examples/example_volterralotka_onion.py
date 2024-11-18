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

n = 30

# Initial guess
X01 = Fourier(omega=2, n=n)
X02 = X01.copy()

X01[0] = x1_dc 
X01[1] = 0.5 #cos component

X02[0] = x2_dc
X02[1+n] = 0.2 #sin component

#trajectory enforcement
Xr1 = X01.copy()
Xr2 = X02.copy()


solutions = []

#initial solution
Xs, _sol = fouriersolve_multi_autonomous_trajectory(
    residual_volterralotka, 
    [X01, X02], 
    [Xr1, Xr2], 
    method="hybr"
    )

solutions.append(Xs)

for _ in range(10):

    #increment trajectory constraint
    Xr1[1] += 0.5

    #onion solutions
    Xs, _sol = fouriersolve_multi_autonomous_trajectory(
        residual_volterralotka, 
        Xs, 
        [Xr1, Xr2], 
        method="hybr", 
        use_jac=True
        )

    solutions.append(Xs)


# Plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True)
ax.plot(x1_dc, x2_dc, ".", label="equilibrium")


for X1, X2 in solutions:

    # Evaluate the solution
    t = np.linspace(0, X1._T(), 1000)    
    ax.plot(X1(t), X2(t), "-")


ax.axvline(0.0, color="k")
ax.axhline(0.0, color="k")
ax.legend()
ax.set_xlabel("predators")
ax.set_ylabel("prey")



plt.show()