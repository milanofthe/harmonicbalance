#####################################################################################################
##
##                             Class that implements a Fourier Series
##
##                                      Milan Rother 2024
##
#####################################################################################################

# IMPORTS ===========================================================================================

import numpy as np
from time import perf_counter
import functools


# NUMPY FUNCTIONS ===================================================================================

FUNCS = [
    # Trigonometric functions
    np.sin,
    np.cos, 
    np.tan, 
    np.arcsin, 
    np.arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.arcsinh, 
    np.arccosh,
    np.arctanh,

    # Exponential and logarithmic functions
    np.exp,
    np.exp2,  
    np.log, 
    np.log2,  
    np.log10,
    np.log1p,
    np.expm1,

    # Power functions
    np.sqrt, 
    np.square, 
    np.power, 
    np.cbrt, 

    # Complex functions
    np.real, 
    np.imag, 
    np.conj, 
    np.abs, 
    np.angle, 

    # Statistical functions
    np.sign,
    ]


# WRAPPER THAT ADDS ADDITIONAL NONLINEARITIES =======================================================

def add_funcs(cls):
    """
    Decorator that adds numpy functions as methods to the 'Fourier' class as 
    additional nonlinearities, utilizes the 'nonlinearity' method.

    Makes the 'Fourier' class compatible with the basic numpy ufuncs.
    """
    def create_method(fnc):
        @functools.wraps(fnc)
        def method(self):
            return self.nonlinearity(fnc)
        return method

    # Add methods to class
    for fnc in FUNCS:
        if not hasattr(cls, fnc.__name__):
            setattr(cls, fnc.__name__, create_method(fnc))
    return cls


# FOURIER CLASS =====================================================================================

@add_funcs
class Fourier:

    """
    Class for Fourier series representation of a signal
    with operators for the time domain representation that
    simplify harmonic balance setup.

    Utilizes NumPy's rfft and irfft for transformations.
    """

    def __init__(self, coeff_dc=0.0, coeffs_cos=None, coeffs_sin=None, omega=1, n=10):
        
        # Number of harmonics
        self.n = n

        # DC component of signal
        self.coeff_dc = coeff_dc

        # Sin and cos components of signal
        self.coeffs_cos = np.zeros(self.n) if coeffs_cos is None else coeffs_cos
        self.coeffs_sin = np.zeros(self.n) if coeffs_sin is None else coeffs_sin * np.sign(omega)

        # Fundamental frequency
        self.omega = abs(omega)


    def __repr__(self):
        return f"Fourier(coeff_dc={self.coeff_dc}, " + \
               f"coeffs_cos={self.coeffs_cos}, " + \
               f"coeffs_sin={self.coeffs_sin}, " + \
               f"omega={self.omega}, n={self.n})"


    def __abs__(self):
        _, amplitudes = self.spectrum()
        return sum(abs(amplitudes))


    def __gt__(self, other):
        if isinstance(other, Fourier):
            return self.amplitude() > other.amplitude()
        else:
            return self.amplitude() > other


    def __lt__(self, other):
        if isinstance(other, Fourier):
            return self.amplitude() < other.amplitude()
        else:
            return self.amplitude() < other


    def __ge__(self, other):
        if isinstance(other, Fourier):
            return self.amplitude() >= other.amplitude()
        else:
            return self.amplitude() >= other


    def __le__(self, other):
        if isinstance(other, Fourier):
            return self.amplitude() <= other.amplitude()
        else:
            return self.amplitude() <= other 


    def __neg__(self):
        return Fourier(
            -self.coeff_dc,
            -self.coeffs_cos,
            -self.coeffs_sin,
            self.omega,
            self.n
        )


    def __add__(self, other):
        if isinstance(other, Fourier):
            return Fourier(
                self.coeff_dc+other.coeff_dc,
                self.coeffs_cos+other.coeffs_cos,
                self.coeffs_sin+other.coeffs_sin,
                self.omega,
                self.n
            )
        else:
            return Fourier(
                self.coeff_dc+other,
                self.coeffs_cos,
                self.coeffs_sin,
                self.omega,
                self.n
            )


    def __radd__(self, other):
        return self.__add__(other)


    def __sub__(self, other):
        return self.__add__(-other)


    def __rsub__(self, other):
        return (-self).__add__(other)


    def __mul__(self, other):
        if isinstance(other, Fourier):
            x1 = self._to_time_domain()
            x2 = other._to_time_domain()
            return self._from_time_domain(x1 * x2)
        else:
            return Fourier(
                self.coeff_dc*other,
                self.coeffs_cos*other,
                self.coeffs_sin*other,
                self.omega,
                self.n
            )


    def __rmul__(self, other):
        return self.__mul__(other)


    def __truediv__(self, other):
        if isinstance(other, Fourier):
            x1 = self._to_time_domain()
            x2 = other._to_time_domain()
            return self._from_time_domain(x1 / x2)
        else:
            return Fourier(
                self.coeff_dc/other,
                self.coeffs_cos/other,
                self.coeffs_sin/other,
                self.omega,
                self.n
            )


    def __rtruediv__(self, other):
        if isinstance(other, Fourier):
            return other.__truediv__(self)
        else:
            return self.nonlinearity(lambda x: other/x)


    def __pow__(self, exponent):
        return self.nonlinearity(lambda x: x**exponent)


    def __getitem__(self, key):
        if key == 0: return self.coeff_dc
        elif key < self.n+1: return self.coeffs_cos[key-1]
        else: return self.coeffs_sin[key-self.n-1]


    def __setitem__(self, key, value):
        if key == 0: self.coeff_dc = value
        elif key < self.n+1: self.coeffs_cos[key-1] = value
        else: self.coeffs_sin[key-self.n-1] = value


    def coeffs(self):
        return np.hstack([self.coeff_dc, self.coeffs_cos, self.coeffs_sin])

    
    def params(self):
        return np.append(self.coeffs(), self.omega)


    def dt(self):
        # Apply differentiation operator on the Fourier series
        omegas = self._omegas()
        return Fourier(
            0.0,
            -omegas*self.coeffs_sin,
            omegas*self.coeffs_cos,
            self.omega,
            self.n
        )


    def _omegas(self):
        return self.omega * np.arange(1, self.n+1)


    def _T(self):
        return 2.0 * np.pi / self.omega


    def _to_time_domain(self, osr=4):

        #number of time domain samples (osr = oversampling ratio)
        N = self.n * osr + 1

        #transform fourier series from sin-cos to complex exponential representation
        coeffs_cexp = np.hstack([self.coeff_dc, 0.5 * (self.coeffs_cos - 1j * self.coeffs_sin)])

        #transform coefficients for numpy fft
        return np.fft.irfft(N*coeffs_cexp, n=N)


    def _from_time_domain(self, x_t):

        N = len(x_t)
        coeffs_cexp = np.fft.rfft(x_t)/N

        #convert to real coefficients (cexp -> cos-sin)
        c_dc = np.real(coeffs_cexp[0])
        c_cos = 2*np.real(coeffs_cexp[1:self.n+1])
        c_sin = -2*np.imag(coeffs_cexp[1:self.n+1])

        return Fourier(
            c_dc,
            c_cos, 
            c_sin,
            self.omega,
            self.n
        )


    def nonlinearity(self, func):
        return self._from_time_domain(func(self._to_time_domain()))


    def copy(self):
        return Fourier(
            self.coeff_dc,
            np.copy(self.coeffs_cos),
            np.copy(self.coeffs_sin),
            self.omega,
            self.n
        )


    # methods for evaluation ------------------------------------------------------------------------

    def amplitude(self):

        #newton parameters
        max_iterations = 200
        tolerance = 1e-4

        #first and second derivative for finding extrema with newton
        dX, ddX = self.dt(), self.dt().dt()

        #evaluation points
        _times = np.linspace(0, self._T(), 2*self.n)

        #vectorized newton iterations until convergence
        for _ in range(max_iterations):
            dx = dX.evaluate(_times)
            
            #check for convergence 
            if max(abs(dx)) < tolerance: 
                break
                
            #newton update
            _times += dx / ddX.evaluate(_times)
                
        #find maximum after dc removal
        return max(abs(self.evaluate(_times)-self.coeff_dc))


    def spectrum(self):
        all_omegas = np.hstack([0.0, self._omegas()])
        cpx_amplitudes = np.hstack([self.coeff_dc, self.coeffs_cos - 1j*self.coeffs_sin])
        return all_omegas, cpx_amplitudes 


    def fundamental_amplitude(self):
        return np.sqrt(self.coeffs_cos[0]**2 + self.coeffs_sin[0]**2)


    def evaluate(self, t):
        result_cos_sin = 0
        for c_cos, c_sin, w in zip(self.coeffs_cos, self.coeffs_sin, self._omegas()):
            result_cos_sin += c_cos * np.cos(w*t) + c_sin * np.sin(w*t)
        return self.coeff_dc + result_cos_sin


    # factory methods -------------------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs, omega):
        n = int((len(coeffs)-1)/2 )
        c_dc, c_cos, c_sin = np.split(coeffs, [1, n+1])
        return cls(c_dc[0], c_cos, c_sin, omega, n)


    @classmethod
    def from_params(cls, params):
        n = int((len(params)-2)/2)
        c_dc, c_cos, c_sin, omega = np.split(params, [1, n+1, 2*n+1])
        return cls(c_dc[0], c_cos, c_sin, omega[0], n)


# LOCAL TESTING =====================================================================================

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    X = Fourier(n=500, omega=1)
    X[1] = 1

    Y = np.sin(X)

    print(X, Y)

    om, am = X.spectrum()
    plt.plot(om, abs(am), "o")

    om, am = Y.spectrum()
    plt.plot(om, abs(am), "o")
    plt.show()
