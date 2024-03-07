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
beta_values = np.linspace(0, 1, 300)

beta_exponent = 6
TMAX = timeData[-1, -1]

def gamma_of_time(a, t):
    return (np.log1p(-t))**(1/beta_exponent)

def time_of_gamma(a, gamma):
    return -np.expm1(gamma**beta_exponent)

def time_of_beta(a, beta, tmax = TMAX):
    gamma = beta*gamma_of_time(a, tmax)
    return time_of_gamma(a, gamma)

def beta_of_time(a, t, tmax = TMAX):
    return gamma_of_time(a, t)/gamma_of_time(a, tmax)

def omega_of_a_time(a, t, alpha_spline):
    chi = chi_of_spin(a)
    beta = beta_of_time(a, t)
    alpha = alpha_spline(chi, beta)
    return omega_of_a_alpha(a, alpha)

def omega_of_a_t(a, t, spl_num = 0):
    return omega_of_a_time(a, t, freq_spl[spl_num])

omega_beta_values = np.array([[omega_of_a_t(a, time_of_beta(a, beta), spl_num = 0) for beta in beta_values] for a in a_values])
omega_beta_values_values_downsample_1 = np.array([[omega_of_a_t(a, time_of_beta(a, beta), spl_num = 1) for beta in beta_values] for a in a_values])
omega_beta_values_values_downsample_2 = np.array([[omega_of_a_t(a, time_of_beta(a, beta), spl_num = 2) for beta in beta_values] for a in a_values])
omega_beta_values_values_downsample_3 = np.array([[omega_of_a_t(a, time_of_beta(a, beta), spl_num = 3) for beta in beta_values] for a in a_values])
omega_beta_values_values_downsample_4 = np.array([[omega_of_a_t(a, time_of_beta(a, beta), spl_num = 4) for beta in beta_values] for a in a_values])

omega_spline_error = np.abs(1. - omega_beta_values_values_downsample_1/omega_beta_values)
omega_spline_error[omega_spline_error<1.e-16] = 1.e-16
omega_spline_error_2 = np.abs(1 - omega_beta_values_values_downsample_2/omega_beta_values_values_downsample_1)
omega_spline_error_2[omega_spline_error_2<1.e-16] = 1.e-16
omega_spline_error_3 = np.abs(1 - omega_beta_values_values_downsample_3/omega_beta_values_values_downsample_2)
omega_spline_error_3[omega_spline_error_3<1.e-16] = 1.e-16
omega_spline_error_4 = np.abs(1 - omega_beta_values_values_downsample_4/omega_beta_values_values_downsample_3)
omega_spline_error_4[omega_spline_error_4<1.e-16] = 1.e-16

plt.plot([8, 7, 6, 5], (np.max((omega_spline_error_3)))*2.**(4*(4 - np.array([6, 5, 4, 3]))), 'k--', label = "$\propto 2^{-4n}$", linewidth=1)
plt.plot([8, 7, 6, 5], [np.max((omega_spline_error)), np.max((omega_spline_error_2)), np.max((omega_spline_error_3)), np.max((omega_spline_error_4))], 'o')
plt.legend()
plt.yscale('log')
plt.ylabel('$||\Delta^{\Omega}_{(n, n)}||_\infty$')
plt.xlabel('$n$')
print(np.max((omega_spline_error))/16)

fig_name = "frequencyConvergence"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname+"/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi=300)