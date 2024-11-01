import numpy as np
import matplotlib.pyplot as plt

from harmonicbalance.predictorcorrector import PredictorCorrectorSolver
from harmonicbalance.fourier import Fourier


#duffing parameters (m:mass, c:linear damping, d:linear stiffness, g:nonlienar stiffness, p:forcing amplitude)
m, c, d, g, p = 1, 0.2, 1, 2, 3


#excitation
U = Fourier(omega=0.01, n=3)
U[1] = 1 #fundamental frequency cos term


#residual for duffing oscillator
def residual_duffing(X):
    return m * X.dt().dt() + c * X.dt() + d * X + g * X**3 + p * U


#initial guess (just use the excitation)
X0 = U.copy()

#initialize the predictor-corrector solver
PCS = PredictorCorrectorSolver(residual_duffing, 
                               X0, 
                               alpha_start=X0.omega, 
                               alpha_end=5, 
                               alpha_step=0.1, 
                               use_jac=False, 
                               method="hybr")

#find solutions in range
PCS.solve()

#find specific solutions at a given frequency from backbone curve
specific_omega = 3
specific_solutions = PCS.solve_specific(specific_omega)


#plot the solution (phase diagram)
fig, ax = plt.subplots(tight_layout=True)

#solution curve
ax.plot([s.omega for s in PCS.solutions], [abs(s) for s in PCS.solutions], ".-")

#specific solution multiples
ax.axvline(specific_omega, color="k")
for s in specific_solutions:
    ax.plot(s.omega, abs(s), "o", color="tab:red")

ax.set_xlabel("Omega")
ax.set_ylabel("Amplitude")

plt.show()
