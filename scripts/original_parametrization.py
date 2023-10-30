from bhpwave.spline import CubicSpline, BicubicSpline
from bhpwave.trajectory.geodesic import kerr_isco_frequency, kerr_circ_geo_radius
from script_analysis_tools import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

import os
pathname = os.path.dirname(os.path.abspath(__file__))

traj_data_full = load_trajectory_data_file(pathname+"/../data/trajectory.txt")

timeData_temp = traj_data_full[2]
phaseData_temp = traj_data_full[3]

chiData = traj_data_full[7]
alphaData = traj_data_full[8]
fluxData = traj_data_full[9]

downsample_chi = int((phaseData_temp.shape[0] - 1)/(fluxData.shape[0] - 1))
downsample_alpha = int((phaseData_temp.shape[1] - 1)/(fluxData.shape[1] - 1))
phaseData = phaseData_temp[::downsample_chi, ::downsample_alpha]
timeData = timeData_temp[::downsample_chi, ::downsample_alpha]

Edot = BicubicSpline(chiData, alphaData, fluxData)
PhiCheck = BicubicSpline(chiData, alphaData, phaseData)
TCheck = BicubicSpline(chiData, alphaData, timeData)

downsample_rate = 4
Nb = int((chiData.shape[0] - 1)/downsample_rate + 1)
Na = int((alphaData.shape[0] - 1)/downsample_rate + 1)
flux_samples = np.zeros((Nb, Na, 5))
phase_samples = np.zeros((Nb, Na, 5))
time_samples = np.zeros((Nb, Na, 5))
for i in range(Nb):
    for j in range(Na):
        beta = chiData[downsample_rate*i]
        alpha = alphaData[downsample_rate*j]
        atemp = spin_of_chi(beta)
        otemp = omega_of_a_alpha(atemp, alpha)
        EdotData = Edot(beta, alpha)
        PData = PhiCheck(beta, alpha)
        TData = TCheck(beta, alpha)
        flux_samples[i, j] = [atemp, otemp, EdotData, alpha, beta]
        phase_samples[i, j] = [atemp, otemp, PData, alpha, beta]
        time_samples[i, j] = [atemp, otemp, TData, alpha, beta]

# cmap_temp = mpl.colormaps['viridis'].resampled(80)
alist = [0, 14, 20, 25, 29, 32]

N = len(alist)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))

fig, axs = plt.subplots(1, 3,)
fig.set_size_inches(15, 3.5)
# fig.tight_layout()
fig.subplots_adjust(wspace=0.2)
for i in alist: 
    avals, omegas, EdotVals = flux_samples[i,:,:3].T
    EdotVals_reweighted = 32./5.*EdotVals*omegas**(10/3)
    axs[0].plot(omegas, EdotVals_reweighted, label="$\hat{a}" + "= {:.4}$".format(avals[0]), lw=2)
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_xlabel('$\hat{\Omega}$')
axs[0].set_ylabel('$\mathcal{F}_E$')

for i in alist: 
    avals, omegas, PVals = phase_samples[i,:,:3].T
    axs[1].plot(omegas, 1 - PVals, label="$\hat{a}" + "= {:.4}$".format(avals[0]), lw=2)
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_xlabel('$\hat{\Omega}$')
axs[1].set_ylabel('$1 - \check{\Phi}$')
# axs[1].legend()

for i in alist: 
    avals, omegas, TVals = time_samples[i,:,:3].T
    axs[2].plot(omegas, 1 - TVals, label="$\hat{a}" + "= {:.4}$".format(avals[0]), lw=2)
axs[2].set_yscale('log')
axs[2].set_xscale('log')
axs[2].set_xlabel('$\hat{\Omega}$')
axs[2].set_ylabel('$1 - \check{t}$')
axs[2].legend()

plt.savefig(pathname+"/../figures/orginal_parametrization.pdf", bbox_inches="tight", dpi=300)