import numpy as np
from scipy.special import wofz
import matplotlib.pyplot as plt


######################  Gaussian, Lorentzian and Voigt Profiles
def G(x,a):
    # Gaussian Line Shape at x with HWHM a
    return (np.sqrt(np.log(2)/np.pi) / a) * np.exp(-(x/a)**2 * np.log(2))

def L(x,g):
    # Lorentzian Line shape at x with HWHM
    return (g/np.pi) / (x ** 2 + g ** 2)

def V(x,a,g):
    # Voigt Line shape at x with Lorentzian component HWHM g and 
    # Gaussian component HWHM  a
    sigma  = a / np.sqrt(2*np.log(2))
    L1 = 1 / (np.sqrt(2*np.pi ) * sigma)
    z = (x + 1j*g)/(sigma * np.sqrt(2))
    return L1 * np.real(wofz(z))

alpha, gamma = 0.1, 0.1
i  =np.linspace(-2,2,1000)
plt.plot(i,G(i,alpha), ls=':', label = 'Gaussian')
plt.plot(i,L(i,gamma), ls='--', label = 'Lorentzian')
plt.plot(i,V(i, alpha, gamma), label = 'Voigt')
plt.legend()
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(8.5, 8.5)
plt.rcParams.update({'font.size': 9})




