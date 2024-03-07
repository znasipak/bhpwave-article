from bhpwave.spline import BicubicSpline
from bhpwave.trajectory.geodesic import kerr_isco_frequency, kerr_circ_geo_orbital_frequency

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

from script_analysis_tools import *

import os
pathname = os.path.dirname(os.path.abspath(__file__))

traj_data_full = load_trajectory_data_file(pathname+"/../data/trajectory_downsample.txt")

chiData = traj_data_full[0]
alphaData = traj_data_full[1]
timeData = traj_data_full[2]
phaseData = traj_data_full[3]

chiFluxData = traj_data_full[7]
alphaFluxData = traj_data_full[8]
fluxData = traj_data_full[9]

bhptoolkit_file_location = "https://raw.githubusercontent.com/BlackHolePerturbationToolkit/CircularOrbitSelfForceData/548664620a3f35ce1cd8e55c70b891dede76abee/Kerr/Fluxes/Flux_Edot_a"
flux_spin_list = [0.999, 0.995, 0.9, 0.6, 0.1, -0.5, -0.8, -0.99]
flux_file_list = []
for spin in flux_spin_list:
    aval = str(spin)
    datapath = pathname + "/../data/"
    file_check = datapath + "Flux_Edot_a" + aval + ".dat"
    wget_string = "wget -P " + datapath + " " + bhptoolkit_file_location + aval + ".dat"
    flux_file_list.append(file_check)
    if not os.path.isfile(file_check):
        print("Get " + file_check)
        os.system(wget_string)

alphaa, aa = np.meshgrid(alphaData, spin_of_chi(chiData))
omegaa = omega_of_a_alpha(aa, alphaa)

spl=BicubicSpline(np.linspace(0, 1, 129), np.linspace(0, 1, 257), fluxData/(32./5.*pn_flux_noprefactor(omegaa)), bc="E(3)")
spl2=BicubicSpline(np.linspace(0, 1, 129), np.linspace(0, 1, 257),fluxData/(32./5.*pn_flux_noprefactor(omegaa)), bc="natural")
spl3=BicubicSpline(np.linspace(0, 1, 129), np.linspace(0, 1, 257), fluxData/(32./5.*pn_flux_noprefactor(omegaa)), bc="not-a-knot")
def scaled_energy_flux(a, r0, bc="E(3)"):
    omega = kerr_circ_geo_orbital_frequency(a, r0)
    alpha = alpha_of_a_omega(a, omega)
    chi = chi_of_spin(a)
    if bc == "E(3)":
        return spl(chi, alpha)*32./5.*omega**(10./3.)
    elif bc == "natural":
        return spl2(chi, alpha)*32./5.*omega**(10./3.)
    elif bc == "not-a-knot":
        return spl3(chi, alpha)*32./5.*omega**(10./3.)
    else:
        return spl(chi, alpha)*32./5.*omega**(10./3.)

testData = []
for file, spin in zip(flux_file_list, flux_spin_list):
    tempData = np.loadtxt(file)
    tempData = tempData[tempData[:,0] < 62.]
    testData.append([spin, tempData])

comparisons = []
comparisons2 = []
bc = 'E(3)'
for data in testData:
    a = data[0]
    compData = data[1]
    r_vals = compData[:, 0]
    flux_vals = compData[:, 1] + compData[:, 2]
    flux_comparison = np.zeros((r_vals.shape[0]))
    for i in range(flux_comparison.shape[0]):
        flux_comparison[i] = scaled_energy_flux(a, r_vals[i], bc=bc)
    comparisons.append([r_vals, np.abs(1. - flux_comparison/flux_vals)])
    comparisons2.append([r_vals, np.abs(flux_comparison-flux_vals)])

testData2 = []
for file, spin in zip(flux_file_list[:3], flux_spin_list[:3]):
    tempData = np.loadtxt(file)
    tempData = tempData[tempData[:,0] < 62.]
    testData2.append([spin, tempData])

comparisons3 = []
bc_type= "E(3)"
for data in testData2:
    a = data[0]
    compData = data[1]
    r_vals = compData[:, 0]
    flux_vals = compData[:, 1] + compData[:, 2]
    flux_comparison = np.zeros((r_vals.shape[0]))
    for i in range(flux_comparison.shape[0]):
        flux_comparison[i] = scaled_energy_flux(a, r_vals[i], bc = bc_type)
    comparisons3.append([r_vals, np.abs(1. - flux_comparison/flux_vals)])

mma_values_900=np.loadtxt(pathname + "/../data/mathetamatica_fluxes_a900.dat")
mma_values_999=np.loadtxt(pathname + "/../data/mathetamatica_fluxes_a999.dat")
mma_values_995=np.loadtxt(pathname + "/../data/mathetamatica_fluxes_a995.dat")

comparison_fluxes = []
bc_type = 'E(3)'
for vals in mma_values_900:
    a, r0, flux, flux_error = vals
    flux_comp = scaled_energy_flux(a, r0, bc = bc_type)
    comparison_fluxes.append([r0, np.abs(1 - flux/flux_comp)])
mma_comp_900 = np.array(comparison_fluxes).T
comparison_fluxes = []
for vals in mma_values_995:
    a, r0, flux, flux_error = vals
    flux_comp = scaled_energy_flux(a, r0, bc = bc_type)
    comparison_fluxes.append([r0, np.abs(1 - flux/flux_comp)])
mma_comp_995 = np.array(comparison_fluxes).T
comparison_fluxes = []
for vals in mma_values_999:
    a, r0, flux, flux_error = vals
    flux_comp = scaled_energy_flux(a, r0, bc = bc_type)
    comparison_fluxes.append([r0, np.abs(1 - flux/flux_comp)])
mma_comp_999 = np.array(comparison_fluxes).T

markers = ['.', 'x', '+', 'D', 'o', 'v', 's', '8']
N = len(markers)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0,1,N)))

fig, axs = plt.subplots(1,3, sharey=True)
fig.set_size_inches(13.5, 4.)
for i, comparison in enumerate(comparisons):
    axs[0].plot(comparison[0][::16], comparison[1][::16], markers[i], label="$"+str(flux_spin_list[i])+"$", markersize = 4.5, fillstyle='none')
axs[0].set_yscale('log')
axs[0].legend(loc="upper right", ncol=2)
axs[0].set_xlabel('$r_0/M$')
axs[0].set_ylabel('$|1 - {\mathcal{F}_E^I}/{\mathcal{F}_E^\mathrm{ext}}|$')
axs[0].set_title('Circular Orbit Self-force Data')

for i, comparison in enumerate(comparisons3):
    axs[1].plot(comparison[0][::16], comparison[1][::16], markers[i], label="$"+str(flux_spin_list[i])+"$", markersize = 4.5, fillstyle='none')
axs[1].set_yscale('log')
axs[1].set_xlabel('$r_0/M$')
axs[1].set_xlim(0.95, 3.6)
axs[1].set_title('Circular Orbit Self-force Data')

axs[2].plot(mma_comp_999[0], mma_comp_999[1], markers[0], label="$0.999$", markersize = 4.5, fillstyle='none')
axs[2].plot(mma_comp_995[0], mma_comp_995[1], markers[1], label="$0.995$", markersize = 4.5, fillstyle='none')
axs[2].plot(mma_comp_900[0], mma_comp_900[1], markers[2], label="$0.9$", markersize = 4.5, fillstyle='none')
axs[2].set_xlabel('$r_0/M$')
axs[2].set_xlim(0.95, 3.6)
axs[2].set_title('High-precision Mathematica Values')

fig_name = "flux_comparison"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname + "/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi = 300)
