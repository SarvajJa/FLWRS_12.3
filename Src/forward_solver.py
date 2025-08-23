
import numpy as np
from numpy.fft import fft, ifft, fftfreq

def spectral_fractional_laplace(u, L, alpha):
    N = u.size
    k = fftfreq(N, d=L/N) * 2*np.pi
    u_hat = fft(u)
    u_hat = (np.abs(k)**alpha) * u_hat
    return ifft(u_hat)

def apply_luminissance(u, eps=1e-2, k0=3.0, L=10.0):
    N = u.size
    k = fftfreq(N, d=L/N) * 2*np.pi
    s = np.exp(-0.5*((np.abs(k)-k0)/0.5)**2)
    sigma = 1 - eps*s + 1j*(eps*0.2*s)
    return ifft(sigma * fft(u))
