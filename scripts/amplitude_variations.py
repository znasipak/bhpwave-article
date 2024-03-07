from bhpwave.trajectory.geodesic import kerr_isco_radius
from bhpwave.harmonics.amplitudes import HarmonicAmplitudes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
mpl.rc('text', usetex=True)


import os
pathname = os.path.dirname(os.path.abspath(__file__))

amps = HarmonicAmplitudes()

mpl.rc('font', **{'size' : 24})
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8,15))

N = 10
axs[0].set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,N)))
for a in np.linspace(0.9, 0.9999, N):
    axs[0].plot(np.linspace(kerr_isco_radius(a), 3.5, 100), np.abs(amps(2, 2, a, np.linspace(kerr_isco_radius(a), 3.5, 100))).flatten(), label="{:.4f}".format(a), linewidth=2)
#axs[0].set_xlabel('$r_0/M$')
axs[0].set_ylabel('$A_{\ell = 2, m = 2}$')
# plt.title('$(\ell,m)=(2,2)$')
axs[0].legend(ncol=2, fontsize=16)

lmax = 15
N = lmax-1
axs[2].set_prop_cycle("color", plt.cm.viridis(np.flip(np.linspace(0,1,N))))
for l in range(2, lmax+1):
    a = 0.9999
    axs[2].plot(np.linspace(kerr_isco_radius(a), 3.5, 100), np.abs(amps(l, l, a, np.linspace(kerr_isco_radius(a), 3.5, 100))).flatten(), label="$\ell = {}$".format(l), linewidth=3)
axs[2].set_yscale('log')
#axs[1].legend(loc='lower center', ncol = 5, fontsize=18)
axs[2].set_xlabel('$r_0/M$')
axs[2].set_ylabel('$A_{\ell, m= \ell}$')

axs[1].set_prop_cycle("color", plt.cm.viridis(np.flip(np.linspace(0,1,N))))
for l in range(2, lmax+1):
    a = 0.9999
    axs[1].plot(np.linspace(kerr_isco_radius(a), 3.5, 100), np.abs(amps(l, 2, a, np.linspace(kerr_isco_radius(a), 3.5, 100))).flatten(), label="$\ell = {}$".format(l), linewidth=3)
axs[1].set_yscale('log')
#axs[1].set_xlabel('$r_0/M$')
axs[1].set_ylabel('$A_{\ell, m= 2}$')
fig.tight_layout()

fig_name = "amp_combined"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname + "/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi = 300)