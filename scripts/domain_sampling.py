from bhpwave.trajectory.geodesic import kerr_isco_frequency
from scipy.stats import gaussian_kde
import os

pathname = os.path.dirname(os.path.abspath(__file__))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

OMEGA_MIN = 2.e-3
A_MAX = 0.9999

def alpha_of_a_omega(a, omega):
    oISCO = kerr_isco_frequency(a)
    return alpha_of_omega_ISCO(omega, oISCO)

def alpha_of_omega_ISCO(omega, oISCO):
    return (abs(oISCO**(1./3.) - omega**(1./3.))/(oISCO**(1./3.) - OMEGA_MIN**(1./3.)))**(0.5)

def omega_of_a_alpha(a, alpha):
    oISCO = kerr_isco_frequency(a)
    return omega_of_alpha_ISCO(alpha, oISCO)

def omega_of_alpha_ISCO(alpha, oISCO):
    return pow(pow(oISCO, 1./3.) - pow(alpha, 2.)*(pow(oISCO, 1./3.) - pow(OMEGA_MIN, 1./3.)), 3.)

def chi_of_spin_subfunc(a):
    return pow(1. - a, 1./3.)

def beta_of_spin(a):
    return pow((chi_of_spin_subfunc(a) - chi_of_spin_subfunc(A_MAX))/(chi_of_spin_subfunc(-A_MAX) - chi_of_spin_subfunc(A_MAX)), 0.5)

def spin_of_beta(beta):
    return 1. - pow(chi_of_spin_subfunc(A_MAX) + pow(beta, 2.)*(chi_of_spin_subfunc(-A_MAX) - chi_of_spin_subfunc(A_MAX)), 3.)

def a_omega_to_chi_alpha(a, omega):
    beta = beta_of_spin(a)
    alpha = alpha_of_a_omega(a, omega)
    return (beta, alpha)

Nb = 32
Na = 64
betaData = np.linspace(0, 1, Nb)
alphaData = np.linspace(0, 1, Na)
spin_omega_samples = np.zeros((Nb, Na, 2))
for i in range(Nb):
    for j in range(Na):
        atemp = spin_of_beta(betaData[i])
        otemp = omega_of_a_alpha(atemp, alphaData[j])
        spin_omega_samples[i, j] = [atemp, otemp]
a_isco = np.linspace(-A_MAX, A_MAX, 5*Nb)
omega_isco = kerr_isco_frequency(a_isco)

plot_array = np.reshape(spin_omega_samples, (spin_omega_samples.shape[0]*spin_omega_samples.shape[1], 2))
x = plot_array[:, 0]
y = plot_array[:, 1]
xy = np.vstack([x,np.log10(y)])
z = gaussian_kde(xy)(xy)
plt.plot([-A_MAX, A_MAX], [OMEGA_MIN, OMEGA_MIN], '--', color="gray", zorder=2)
plt.plot(a_isco, omega_isco, '--', color="gray", zorder=2)
plt.plot([-A_MAX, -A_MAX], [OMEGA_MIN, omega_isco[0]], '--', color="gray", zorder=2)
plt.plot([A_MAX, A_MAX], [OMEGA_MIN, omega_isco[-1]], '--', color="gray", zorder=2)
plt.scatter(x, y, c=np.log10(z), s=5, cmap = 'plasma', zorder=1)
plt.yscale('log')
plt.xlabel('$\hat{a}$')
plt.ylabel('$\hat{\Omega}$')
plt.savefig(pathname+"/../figures/domain_sampling.pdf", bbox_inches="tight", dpi=300)