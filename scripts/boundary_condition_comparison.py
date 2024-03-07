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

traj_data_full = load_trajectory_data_file(pathname+"/../data/trajectory.txt")

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

alphaa, aa = np.meshgrid(alphaFluxData, spin_of_chi(chiFluxData))
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

N = len(testData)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0,1,N)))

bc_array = ["not-a-knot", "natural"]
markers = ['.', 'x', '+', 'D', 'o', 'v', 's', '8']
fig, axs = plt.subplots(len(bc_array), 1, sharex=True)
fig.set_size_inches(4., 6.5)

i = 0
for bc in bc_array:
    comparisons = []
    for data in testData:
        a = data[0]
        compData = data[1]
        r_vals = compData[:, 0]
        flux_vals = compData[:, 1] + compData[:, 2]
        flux_comparison = np.zeros((r_vals.shape[0]))
        for j in range(flux_comparison.shape[0]):
            flux_comparison[j] = scaled_energy_flux(a, r_vals[j], bc=bc)
        comparisons.append([r_vals, np.abs(1. - flux_comparison/flux_vals)])
    
    for j, comparison in enumerate(comparisons):
        axs[i].plot(comparison[0][::32], comparison[1][::32], markers[j], label="$"+str(flux_spin_list[j])+"$", markersize = 4.5, fillstyle='none')
    axs[i].set_yscale('log')
    # if i == 0:
    #     axs[i].legend(loc="upper right", ncol=2)
    axs[i].set_ylabel('$|1 - {\mathcal{F}_E^I}/{\mathcal{F}_E^\mathrm{ext}}|$')
    axs[i].set_title(bc)
    axs[i].set_ylim(1.e-15, 2.e-1)
    i += 1 
axs[i-1].set_xlabel('$r_0/M$')

fig_name = "boundary_condition_comparison"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname+"/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi=300)