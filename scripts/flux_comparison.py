from bhpwave.spline import BicubicSpline
from bhpwave.trajectory.geodesic import kerr_isco_frequency, kerr_circ_geo_orbital_frequency

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

import os
pathname = os.path.dirname(os.path.abspath(__file__))

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

def chi_of_spin(a):
    return pow((chi_of_spin_subfunc(a) - chi_of_spin_subfunc(A_MAX))/(chi_of_spin_subfunc(-A_MAX) - chi_of_spin_subfunc(A_MAX)), 0.5)

def spin_of_chi(chi):
    return 1. - pow(chi_of_spin_subfunc(A_MAX) + pow(chi, 2.)*(chi_of_spin_subfunc(-A_MAX) - chi_of_spin_subfunc(A_MAX)), 3.)

def a_omega_to_chi_alpha(a, omega):
    chi = chi_of_spin(a)
    alpha = alpha_of_a_omega(a, omega)
    return (chi, alpha)

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

EdotDataComp = np.load(pathname + "/../data/bhpwave_edot_redo.npy")
traj = np.loadtxt(pathname + "/../data/trajectory.txt", skiprows=3)
trajHeader = np.loadtxt(pathname + "/../data/trajectory.txt", skiprows=2, max_rows=1, dtype='str')
trajShape = np.loadtxt(pathname + "/../data/trajectory.txt", skiprows=1, max_rows=1, dtype='int')

phaseData = np.ascontiguousarray(traj[:, 4].reshape(trajShape))
timeData = np.ascontiguousarray(traj[:, 3].reshape(trajShape))
fluxData = np.ascontiguousarray(traj[:, 2].reshape(trajShape))
chiData = np.ascontiguousarray(traj[:, 0].reshape(trajShape)[:, 0])
alphaData = np.ascontiguousarray(traj[:, 1].reshape(trajShape)[0])

alphaa, aa = np.meshgrid(alphaData, spin_of_chi(chiData))
omegaa = omega_of_a_alpha(aa, alphaa)

spl=BicubicSpline(np.linspace(0, 1, 129), np.linspace(0, 1, 257), 2.*EdotDataComp/(32./5.*omegaa**(10./3.)), bc="E(3)")
spl2=BicubicSpline(np.linspace(0, 1, 129), np.linspace(0, 1, 257), 2.*EdotDataComp/(32./5.*omegaa**(10./3.)), bc="natural")
spl3=BicubicSpline(np.linspace(0, 1, 129), np.linspace(0, 1, 257), 2.*EdotDataComp/(32./5.*omegaa**(10./3.)), bc="not-a-knot")
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
for file, spin in zip(flux_file_list[:4], flux_spin_list[:4]):
    tempData = np.loadtxt(file)
    tempData = tempData[tempData[:,0] < 5.]
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

mma_values_900=np.loadtxt(pathname + "/../data/mathetamatica_fluxes_a9.dat")
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

markers = ['.', 'x', '+', 'D', 'o', 'v', 's', '8']
fig, axs = plt.subplots(1,3, sharey=True)
fig.set_size_inches(13.5, 4.)
for i, comparison in enumerate(comparisons):
    axs[0].plot(comparison[0][::2], comparison[1][::2], markers[i], label="$"+str(flux_spin_list[i])+"$", markersize = 4.5, fillstyle='none')
axs[0].set_yscale('log')
axs[0].legend(loc="upper right", ncol=2)
axs[0].set_xlabel('$r_0/M$')
axs[0].set_ylabel('$|1 - {\mathcal{F}_E^I}/{\mathcal{F}_E^\mathrm{ext}}|$')

for i, comparison in enumerate(comparisons3):
    axs[1].plot(comparison[0], comparison[1], markers[i], label="$"+str(flux_spin_list[i])+"$", markersize = 4.5, fillstyle='none')
axs[1].set_yscale('log')
axs[1].set_xlabel('$r_0/M$')
axs[1].set_xlim(0.95, 5.1)

axs[2].plot(mma_comp_900[0], mma_comp_900[1], markers[2], label="$0.9$", markersize = 4.5)
axs[2].plot(mma_comp_995[0], mma_comp_995[1], markers[1], label="$0.995$", markersize = 4.5)
axs[2].plot(mma_comp_900[0], mma_comp_900[1], markers[2], label="$0.9$", markersize = 4.5)
axs[2].set_xlabel('$r_0/M$')

print("Saving figure to " + pathname + "/../figures/flux_comparison.pdf")
plt.savefig(pathname+"/../figures/flux_comparison.pdf", bbox_inches="tight", dpi=300)
