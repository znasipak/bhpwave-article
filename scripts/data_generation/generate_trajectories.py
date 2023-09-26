from bhpwave.spline import CubicSpline, BicubicSpline
from bhpwave.trajectory.geodesic import kerr_isco_frequency, kerr_circ_geo_radius
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import numpy as np

import os
pathname = os.path.dirname(os.path.abspath(__file__))

OMEGA_MIN = 2.e-3
A_MAX = 0.9999

flux_array = np.load(pathname + "/../../data/Edot_circ.npy")
chi_array = np.linspace(0, 1, flux_array.shape[0])
alpha_array = np.linspace(0, 1, flux_array.shape[1])
beta_array = np.linspace(0, 1, alpha_array.shape[0])

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

def omega_alpha_derivative(omega, oISCO):
    if abs(oISCO - omega) < 1.e-13: 
        return 0.
    return -6.*pow((pow(oISCO, 1./3.) - pow(OMEGA_MIN, 1./3.))*(pow(oISCO, 1./3.) - pow(omega, 1./3.)), 0.5)*pow(omega, 2./3.)

def energy_omega_derivative(a, omega):
    r = kerr_circ_geo_radius(a, omega)
    return energy_r_derivative(a, r)*r_omega_derivative(a, omega)

def energy_r_derivative(a, r):
    v = 1./np.sqrt(r)
    return 0.5*(pow(v, 4) - 6.*pow(v, 6) + 8.*a*pow(v, 7) - 3.*a*a*pow(v, 8))/pow(1. + v*v*(2.*a*v - 3.), 1.5)

def r_omega_derivative(a, omega):
    return -2./(3.*pow(omega, 5./3.)*pow(1. - a*omega, 1./3.))

t_data = np.zeros(flux_array.shape)
phi_data = np.zeros(flux_array.shape)
alpha_beta_data = np.zeros(flux_array.shape)

def pn_flux_noprefactor(omega):
    return omega**(10./3.)

ALPHA, CHI = np.meshgrid(alpha_array, chi_array)
SPIN = spin_of_chi(CHI)
OMEGA = omega_of_a_alpha(SPIN, ALPHA)

flux_norm_array = flux_array/pn_flux_noprefactor(OMEGA)

beta_exponent = 6

def time_of_beta(beta, gamma_max_squared):
    gamma_squared = beta**beta_exponent*gamma_max_squared
    return np.expm1(gamma_squared)

print("Generating time and phase evolution")
for i in range(chi_array.shape[0]):
    iterChi = chi_array.shape[0] - 1 - i # we fill things in backwards in order to define \gamma_max early on
    a_val = spin_of_chi(chi_array[iterChi])
    oISCO_val = kerr_isco_frequency(a_val)
    omega_data = omega_of_a_alpha(a_val, alpha_array)
    flux_norm = flux_norm_array[iterChi]
    flux_spline = CubicSpline(alpha_array, flux_norm, bc = 'E(3)')

    def dIdAlphaIntegrate(alpha, t):
        omega = omega_of_a_alpha(a_val, alpha)
        dOmega_dAlpha = omega_alpha_derivative(omega, oISCO_val)
        dE_dOmega = energy_omega_derivative(a_val, omega)
        Edot = flux_spline(alpha)*pn_flux_noprefactor(omega)
        return dE_dOmega*dOmega_dAlpha/Edot*np.array([1., omega])
    
    insp = solve_ivp(dIdAlphaIntegrate, [0., 1.], [0., 0.], method='DOP853', t_eval=alpha_array, rtol=1.e-13, atol=1.e-14)
    t_data[iterChi] = insp.y[0]
    phi_data[iterChi] = insp.y[1]

def pn_timescale_noprefactor(omega):
    return omega**(-8./3.)

def time_of_omega_root(omega, t0, spin_index):
    a_val = spin_of_chi(chi_array[spin_index])
    omega_data = omega_of_a_alpha(a_val, alpha_array)

    t_norm = t_data[spin_index]/pn_timescale_noprefactor(omega_data)
    spl = CubicSpline(alpha_array, t_norm, bc = 'E(3)')

    alpha = alpha_of_a_omega(a_val, omega)
    return spl(alpha)*pn_timescale_noprefactor(omega) - t0

beta_array = np.linspace(0, 1, alpha_array.shape[0])
t_max_values = t_data.T[-1]
# t_max_values_norm = t_max_values/pn_timescale_noprefactor(OMEGA_MIN)
# spl_tmax = CubicSpline(chi_array, t_max_values_norm, bc = "E(3)")
gamma_max_squared = np.log(1. + t_max_values[-1])
print(f'gamma_max_squared: {gamma_max_squared}')

print("Inverting for frequency evolution")
omega_beta_array = np.zeros((chi_array.shape[0], alpha_array.shape[0]))
for i in range(chi_array.shape[0]):
    spin_index = i
    a_val = spin_of_chi(chi_array[spin_index])
    omega_max = omega_of_a_alpha(a_val, 0.)
    print(a_val)
    for j, beta in enumerate(beta_array):
        t0 = time_of_beta(beta, gamma_max_squared)
        if j == 0:
            omega_beta_array[i, 0] = omega_max
        elif j == alpha_array.shape[0] - 1 and i == chi_array.shape[0] - 1:
            omega_beta_array[i, -1] = OMEGA_MIN
        else:
            xtol = np.min([1.e-10, 0.001*t0])
            sol = root_scalar(time_of_omega_root, args=(t0, spin_index), method='brentq', bracket=[OMEGA_MIN, omega_max], xtol = xtol, rtol = 1.e-12)
            omega_beta_array[i, j] = sol.root

BETA, _ = np.meshgrid(beta_array, chi_array)

print("Saving trajectory data to " + pathname + "/../../data/trajectory_v3.txt")

header_txt = "chiN \t alphaN \n129\t257 \nchi\talpha\tflux\ttime\tphase\tbeta\tomega"
full_data_array = np.array([CHI.flatten(), ALPHA.flatten(), flux_array.flatten(), t_data.flatten(), phi_data.flatten(), BETA.flatten(), omega_beta_array.flatten()]).T
np.savetxt(pathname + "/../../data/trajectory_v3.txt", full_data_array, header=header_txt, fmt = "%1.15e", comments='')