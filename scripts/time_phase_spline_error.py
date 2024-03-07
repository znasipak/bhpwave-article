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
betaData = traj_data_full[4]
omegaData = traj_data_full[5]

chiFluxData = traj_data_full[7]
alphaFluxData = traj_data_full[8]
fluxData = traj_data_full[9]

alphaa, aa = np.meshgrid(alphaFluxData, spin_of_chi(chiFluxData))
omegaa = omega_of_a_alpha(aa, alphaa)
fluxData = fluxData/(32./5.*pn_flux_noprefactor(omegaa))

ALPHA, CHI = np.meshgrid(alphaData, chiData)
BETA, _ = np.meshgrid(betaData, chiData)
SPIN = spin_of_chi(CHI)
OMEGA = omega_of_a_alpha(SPIN, ALPHA)
phase_norm_array = (phaseData)/pn_phase_noprefactor(SPIN, OMEGA)
time_norm_array = (timeData)/pn_time_noprefactor(SPIN, OMEGA)
alpha_beta_array = alpha_of_a_omega(SPIN, omegaData)

time_spl = {}
time_spl[0] = BicubicSpline(chiData, alphaData, time_norm_array, bc = 'E(3)')
time_spl[1] = BicubicSpline(chiData[::2], alphaData[::2], time_norm_array[::2, ::2], bc = 'E(3)')
time_spl[2] = BicubicSpline(chiData[::4], alphaData[::4], time_norm_array[::4, ::4], bc = 'E(3)')
time_spl[3] = BicubicSpline(chiData[::8], alphaData[::8], time_norm_array[::8, ::8], bc = 'E(3)')
time_spl[4] = BicubicSpline(chiData[::16], alphaData[::16], time_norm_array[::16, ::16], bc = 'E(3)')

phase_spl = {}
phase_spl[0] = BicubicSpline(chiData, alphaData, phase_norm_array, bc = 'E(3)')
phase_spl[1] = BicubicSpline(chiData[::2], alphaData[::2], phase_norm_array[::2, ::2], bc = 'E(3)')
phase_spl[2] = BicubicSpline(chiData[::4], alphaData[::4], phase_norm_array[::4, ::4], bc = 'E(3)')
phase_spl[3] = BicubicSpline(chiData[::8], alphaData[::8], phase_norm_array[::8, ::8], bc = 'E(3)')
phase_spl[4] = BicubicSpline(chiData[::16], alphaData[::16], phase_norm_array[::16, ::16], bc = 'E(3)')

freq_spl = {}
freq_spl[0] = BicubicSpline(chiData, betaData, alpha_beta_array, bc = 'E(3)')
freq_spl[1] = BicubicSpline(chiData[::2], betaData[::2], alpha_beta_array[::2, ::2], bc = 'E(3)')
freq_spl[2] = BicubicSpline(chiData[::4], betaData[::4], alpha_beta_array[::4, ::4], bc = 'E(3)')
freq_spl[3] = BicubicSpline(chiData[::8], betaData[::8], alpha_beta_array[::8, ::8], bc = 'E(3)')
freq_spl[4] = BicubicSpline(chiData[::16], betaData[::16], alpha_beta_array[::16, ::16], bc = 'E(3)')

a_values = A_MAX*np.cos(np.linspace(0, np.pi, 300))
chi_values = chi_of_spin(a_values)
alpha_values = np.linspace(0, 1, 300)
omega_values = np.array([[omega_of_a_alpha(a, alpha) for alpha in alpha_values] for a in a_values])

def omega_alpha_derivative(omega, oISCO):
    if abs(oISCO - omega) < 1.e-13: 
        return 0.
    return -6.*pow((pow(oISCO, 1./3.) - pow(OMEGA_MIN, 1./3.))*(pow(oISCO, 1./3.) - pow(omega, 1./3.)), 0.5)*pow(omega, 2./3.)

def energy_r_derivative(a, r):
    v = 1./np.sqrt(r)
    return 0.5*(pow(v, 4) - 6.*pow(v, 6) + 8.*a*pow(v, 7) - 3.*a*a*pow(v, 8))/pow(1. + v*v*(2.*a*v - 3.), 1.5)

def r_omega_derivative(a, omega):
    return -2./(3.*pow(omega, 5./3.)*pow(1. - a*omega, 1./3.))

def energy_omega_derivative(a, omega):
    r = kerr_circ_geo_radius(a, omega)
    return energy_r_derivative(a, r)*r_omega_derivative(a, omega)

def energy_omega_alpha_derivative(a, omega):
    r = kerr_circ_geo_radius(a, omega)
    oISCO = kerr_isco_frequency(a)
    domega_dalpha = omega_alpha_derivative(omega, oISCO)
    return energy_r_derivative(a, r)*r_omega_derivative(a, omega)*domega_dalpha

def time_of_a_omega(a, omega, deriv = 0, spl_num = 0):
    if isinstance(omega, np.ndarray):
        return np.asarray([time_of_a_omega(a, o, deriv=deriv) for o in omega])
    chi = chi_of_spin(a)
    oISCO = kerr_isco_frequency(a)
    alpha = alpha_of_omega_ISCO(omega, oISCO)
    if deriv == 0:
        return time_spl[spl_num](chi, alpha)*pn_time_noprefactor(a, omega)
    elif deriv == 1:
        domega_dalpha = omega_alpha_derivative(omega, oISCO)
        return time_spl[spl_num].deriv_y(chi, alpha)*pn_time_noprefactor(a, omega) + domega_dalpha*time_spl[spl_num](chi, alpha)*pn_time_noprefactor_domega(omega)
    else:
        return time_spl[spl_num](chi, alpha)*pn_time_noprefactor(a, omega)

def phase_of_a_omega(a, omega, deriv = 0, spl_num = 0):
    if isinstance(omega, np.ndarray):
        return np.asarray([phase_of_a_omega(a, o, deriv=deriv) for o in omega])
    chi = chi_of_spin(a)
    oISCO = kerr_isco_frequency(a)
    alpha = alpha_of_omega_ISCO(omega, oISCO)
    if deriv == 0:
        return phase_spl[spl_num](chi, alpha)*pn_phase_noprefactor(a, omega)
    elif deriv == 1:
        domega_dalpha = omega_alpha_derivative(omega, oISCO)
        return phase_spl[spl_num].deriv_y(chi, alpha)*pn_phase_noprefactor(a, omega) + domega_dalpha*phase_spl[spl_num](chi, alpha)*pn_phase_noprefactor_domega(omega)
    else:
        return phase_spl[spl_num](chi, alpha)*pn_phase_noprefactor(a, omega)

def phase_difference_frequency(a, alpha, spl_num = 0):
    omega = omega_of_a_alpha(a, alpha)
    return phase_of_a_omega(a, omega, spl_num = spl_num)

def weighted_time_difference_frequency(a, alpha, spl_num = 0):
    omega = omega_of_a_alpha(a, alpha)
    return omega*time_of_a_omega(a, omega, spl_num = spl_num)

def time_difference_frequency(a, alpha, spl_num = 0):
    omega = omega_of_a_alpha(a, alpha)
    return time_of_a_omega(a, omega, spl_num = spl_num)

weighted_time_values = np.array([[weighted_time_difference_frequency(a, alpha) for alpha in alpha_values] for a in a_values])
weighted_time_values_downsample_1 = np.array([[weighted_time_difference_frequency(a, alpha, spl_num = 1) for alpha in alpha_values] for a in a_values])
weighted_time_values_downsample_2 = np.array([[weighted_time_difference_frequency(a, alpha, spl_num = 2) for alpha in alpha_values] for a in a_values])
weighted_time_values_downsample_3 = np.array([[weighted_time_difference_frequency(a, alpha, spl_num = 3) for alpha in alpha_values] for a in a_values])
weighted_time_values_downsample_4 = np.array([[weighted_time_difference_frequency(a, alpha, spl_num = 4) for alpha in alpha_values] for a in a_values])

time_values = np.array([[time_difference_frequency(a, alpha) for alpha in alpha_values] for a in a_values])
time_values_downsample_1 = np.array([[time_difference_frequency(a, alpha, spl_num = 1) for alpha in alpha_values] for a in a_values])
time_values_downsample_2 = np.array([[time_difference_frequency(a, alpha, spl_num = 2) for alpha in alpha_values] for a in a_values])
time_values_downsample_3 = np.array([[time_difference_frequency(a, alpha, spl_num = 3) for alpha in alpha_values] for a in a_values])
time_values_downsample_4 = np.array([[time_difference_frequency(a, alpha, spl_num = 4) for alpha in alpha_values] for a in a_values])

phase_values = np.array([[phase_difference_frequency(a, alpha) for alpha in alpha_values] for a in a_values])
phase_values_downsample_1 = np.array([[phase_difference_frequency(a, alpha, spl_num = 1) for alpha in alpha_values] for a in a_values])
phase_values_downsample_2 = np.array([[phase_difference_frequency(a, alpha, spl_num = 2) for alpha in alpha_values] for a in a_values])
phase_values_downsample_3 = np.array([[phase_difference_frequency(a, alpha, spl_num = 3) for alpha in alpha_values] for a in a_values])
phase_values_downsample_4 = np.array([[phase_difference_frequency(a, alpha, spl_num = 4) for alpha in alpha_values] for a in a_values])

phase_spline_error = np.abs(phase_values - phase_values_downsample_1)
phase_spline_error[phase_spline_error<1.e-16] = 1.e-16
phase_spline_error_2 = np.abs(phase_values_downsample_1 - phase_values_downsample_2)
phase_spline_error_2[phase_spline_error_2<1.e-16] = 1.e-16
phase_spline_error_3 = np.abs(phase_values_downsample_2 - phase_values_downsample_3)
phase_spline_error_3[phase_spline_error_3<1.e-16] = 1.e-16
phase_spline_error_4 = np.abs(phase_values_downsample_3 - phase_values_downsample_4)
phase_spline_error_4[phase_spline_error_4<1.e-16] = 1.e-16

time_spline_error = np.abs(1 - (1-time_values)/(1-time_values_downsample_1))
time_spline_error[time_spline_error<1.e-16] = 1.e-16
time_spline_error_2 = np.abs(1 - (1 - time_values_downsample_1)/(1 - time_values_downsample_2))
time_spline_error_2[time_spline_error_2<1.e-16] = 1.e-16
time_spline_error_3 = np.abs(1 - (1 - time_values_downsample_2)/(1 - time_values_downsample_3))
time_spline_error_3[time_spline_error_3<1.e-16] = 1.e-16
time_spline_error_4 = np.abs(1 - (1-time_values_downsample_3)/(1 - time_values_downsample_4))
time_spline_error_4[time_spline_error_4<1.e-16] = 1.e-16

weighted_time_spline_error = np.abs(weighted_time_values - weighted_time_values_downsample_1)
weighted_time_spline_error[weighted_time_spline_error<1.e-16] = 1.e-16
weighted_time_spline_error_2 = np.abs(weighted_time_values_downsample_1 - weighted_time_values_downsample_2)
weighted_time_spline_error_2[weighted_time_spline_error_2<1.e-16] = 1.e-16
weighted_time_spline_error_3 = np.abs(weighted_time_values_downsample_2 - weighted_time_values_downsample_3)
weighted_time_spline_error_3[weighted_time_spline_error_3<1.e-16] = 1.e-16
weighted_time_spline_error_4 = np.abs(weighted_time_values_downsample_3 - weighted_time_values_downsample_4)
weighted_time_spline_error_4[weighted_time_spline_error_4<1.e-16] = 1.e-16

plt.plot([8, 7, 6, 5], 0.5*(np.max((phase_spline_error_3)) + np.max((time_spline_error_3)))*2.**(4*(4 - np.array([6, 5, 4, 3]))), 'k--', label = "$\propto 2^{-4n}$", linewidth=1)
plt.plot([8, 7, 6, 5], (np.max((time_spline_error_3)))*2.**(4*(4 - np.array([6, 5, 4, 3]))), 'k--', linewidth=1)
plt.plot([8, 7, 6, 5], [np.max((time_spline_error)), np.max((time_spline_error_2)), np.max((time_spline_error_3)), np.max((time_spline_error_4))], 'o', label = "$\Delta^t$", markersize=10)
plt.plot([8, 7, 6, 5], [np.max((weighted_time_spline_error)), np.max((weighted_time_spline_error_2)), np.max((weighted_time_spline_error_3)), np.max((weighted_time_spline_error_4))], 's', label = "$\Delta^{\Omega t}$", markersize=10)
plt.plot([8, 7, 6, 5], [np.max((phase_spline_error)), np.max((phase_spline_error_2)), np.max((phase_spline_error_3)), np.max((phase_spline_error_4))], 'd', label = "$\Delta^\Phi$", markersize=10)
plt.legend()
plt.yscale('log')
plt.ylabel('$||\Delta^{X}_{(n, n)}||_\infty$')
plt.xlabel('$n$')

fig_name = "timeAndPhaseConvergence"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname+"/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi=300)