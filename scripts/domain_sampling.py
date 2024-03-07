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

from script_analysis_tools import *

Nb = 32
Na = 64
chiData = np.linspace(0, 1, Nb)
alphaData = np.linspace(0, 1, Na)
spin_omega_samples = np.zeros((Nb, Na, 2))
for i in range(Nb):
    for j in range(Na):
        atemp = spin_of_chi(chiData[i])
        otemp = omega_of_a_alpha(atemp, alphaData[j])
        spin_omega_samples[i, j] = [atemp, otemp]
a_isco = np.linspace(-A_MAX, A_MAX, 5*Nb)
omega_isco = kerr_isco_frequency(a_isco)

plot_array = np.reshape(spin_omega_samples, (Na*Nb, 2))
x = plot_array[:, 0]
y = plot_array[:, 1]
xy = np.vstack([x,np.log10(y)])
z = gaussian_kde(xy)(xy)
plt.plot([-A_MAX, A_MAX], [OMEGA_MIN, OMEGA_MIN], '--', color="gray", zorder=2)
plt.plot(a_isco, omega_isco, '--', color="gray", zorder=2)
plt.plot([-A_MAX, -A_MAX], [OMEGA_MIN, omega_isco[0]], '--', color="gray", zorder=2)
plt.plot([A_MAX, A_MAX], [OMEGA_MIN, omega_isco[-1]], '--', color="gray", zorder=2)
plt.scatter(x, y, c=np.log10(z), s=5, cmap = 'viridis', zorder=1)
plt.yscale('log')
plt.xlabel('$\hat{a}$')
plt.ylabel('$\hat{\Omega}$')

fig_name = "domain_sampling"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname + "/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi=300)