from bhpwave.spline import CubicSpline, BicubicSpline
from bhpwave.trajectory.geodesic import kerr_isco_frequency, kerr_circ_geo_radius

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

from script_analysis_tools import *

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
        chi = chiData[downsample_rate*i]
        alpha = alphaData[downsample_rate*j]
        atemp = spin_of_chi(chi)
        otemp = omega_of_a_alpha(atemp, alpha)
        EdotData = Edot(chi, alpha)
        PData = PhiCheck(chi, alpha)
        TData = TCheck(chi, alpha)
        flux_samples[i, j] = [atemp, otemp, 5/32*EdotData/pn_flux_noprefactor(otemp), alpha, chi]
        phase_samples[i, j] = [atemp, otemp, 32*PData/pn_phase_noprefactor(atemp, otemp), alpha, chi]
        time_samples[i, j] = [atemp, otemp, 256/5*TData/pn_time_noprefactor(atemp, otemp), alpha, chi]

# cmap_temp = mpl.colormaps['plasma'].resampled(80)
alist = [0, 10, 14, 20, 25, 29, 32]

N = len(alist)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))

fig, axs = plt.subplots(1, 3,)
fig.set_size_inches(15, 3.5)
# fig.tight_layout()
fig.subplots_adjust(wspace=0.2)
for i in alist: 
    avals, omegas, EdotVals = flux_samples[i,:,:3].T
    alphas, _ = flux_samples[i,:,3:].T
    axs[0].plot((alphas), EdotVals, lw=2)
axs[0].set_xlabel('$\\alpha$')
axs[0].set_ylabel('$5/32 \\times \mathcal{F}_N$')

for i in alist: 
    avals, omegas, PVals = phase_samples[i,:,:3].T
    alphas, _ = flux_samples[i,:,3:].T
    axs[1].plot((alphas), PVals, lw=2)
axs[1].set_xlabel('$\\alpha$')
axs[1].set_ylabel('$32 \\times \check{\Phi}_N$')

for i in alist: 
    avals, omegas, TVals = time_samples[i,:,:3].T
    alphas, chis = flux_samples[i,:,3:].T
    axs[2].plot((alphas), TVals, label="$\\chi = {:.4}$".format(chis[0]), lw=2)
axs[2].set_xlabel('$\\alpha$')
axs[2].set_ylabel('$256/5 \\times \check{t}_N$')
axs[2].legend()

print("Saving figure to " + pathname + "/../figures/final_parametrization.pdf")
plt.savefig(pathname+"/../figures/final_parametrization.pdf", bbox_inches="tight", dpi=300)