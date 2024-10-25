#####################################################################################################
##
##            Implementation of Predictor-Corrector for solution path continuation
##
##                                      Milan Rother 2024
##
#####################################################################################################

# IMPORTS ===========================================================================================

import numpy as np

import scipy

from .fourier import Fourier
from .solvers import fouriersolve, fouriersolve_arclength, numerical_jacobian, timer


# PREDICTOR-CORRECTOR CLASS =========================================================================

class PredictorCorrectorSolver:
    
    def __init__(self, residual_func, X0, alpha_start, alpha_end, alpha_step, use_jac=False, **solverargs):

        self.residual_func = residual_func
        
        self.X0 = X0

        self.direction = 1.0 if alpha_start < alpha_end else -1.0
        
        self.alpha_start = alpha_start
        self.alpha_end  = alpha_end
        self.alpha_step = alpha_step

        self.use_jac = use_jac
        
        self.solverargs = solverargs

        self.solutions = []
        

    @timer
    def solve_specific(self, alpha):
        
        #find good predictions
        predictions = []
        
        for s1, s2 in zip(self.solutions[:-1], self.solutions[1:]):
            
            a1, a2 = s1.omega, s2.omega
            
            if a1 <= alpha <= a2 or a2 <= alpha <= a1:
            
                if abs(alpha-a1) < abs(alpha-a2):
                    predictions.append(s1.copy())
                else:
                    predictions.append(s2.copy())
        
        #correct predicted solutions
        solutions = []
        
        for X_pred in predictions:
            
            #set omega
            X_pred.omega = alpha
            
            #solve HB with fixed parameter
            X, _ = fouriersolve(self.residual_func, X_pred, use_jac=self.use_jac, **self.solverargs)
                        
            solutions.append(X.copy())
            
        #return corrected solutions
        return solutions
        
    
    def predictor_secant_hypersphere(self):

        #get two previous solutions
        X1, X2 = self.solutions[-1], self.solutions[-2]

        #full parameter vector differentials
        dX = X1.params() - X2.params()

        #linear projection to hypersphere
        params = X1.params() + self.alpha_step * dX / np.linalg.norm(dX)

        #set parameters
        return Fourier.from_params(params)


    @timer
    def solve(self):

        #initial corrector step without additional constraint (with fixed parameter)
        X, _ = fouriersolve(residual_func=self.residual_func, 
                            X0=self.X0, 
                            use_jac=self.use_jac, 
                            **self.solverargs)

        #save initial solution
        self.solutions.append(X.copy())       
                
        #initial predictor step (direction is decided here)
        omega_pred = self.X0.omega + 0.5 * self.alpha_step * self.direction
        X_pred = Fourier.from_coeffs(X.coeffs(), omega_pred)

        #previous solution for arclength enforcement
        X_prev = X.copy()

        # initial corrector step with additional constraint (with fixed parameter)
        X, _ = fouriersolve_arclength(residual_func=self.residual_func, 
                                      X0=X_pred, 
                                      Xref=X_prev, 
                                      ds=self.alpha_step, 
                                      use_jac=self.use_jac,
                                      **self.solverargs)
        
        #update and save solution
        self.solutions.append(X.copy())
        
        #forward solution until max parameter (alpha) is reached
        while True:
            
            #predictor hypersphere arclength
            X_pred = self.predictor_secant_hypersphere()

            #previous solution for arclength enforcement
            X_prev = X.copy()

            X, _ = fouriersolve_arclength(residual_func=self.residual_func, 
                                          X0=X_pred, 
                                          Xref=X_prev, 
                                          ds=self.alpha_step, 
                                          use_jac=self.use_jac,
                                          **self.solverargs)
            
            #update and save solution
            self.solutions.append(X.copy())   
                        
            #check iteration condition
            if (X.omega > max(self.alpha_start, self.alpha_end) or 
                X.omega < min(self.alpha_start, self.alpha_end) ):           
                break

        return self.solutions