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
chi_flux_array = np.linspace(0, 1, flux_array.shape[0])
alpha_flux_array = np.linspace(0, 1, flux_array.shape[1])

chi_array = np.linspace(0, 1, 2**9+1)
alpha_array = np.linspace(0, 1, 2**9+1)
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

ALPHA, CHI = np.meshgrid(alpha_array, chi_array)
BETA, _ = np.meshgrid(beta_array, chi_array)
SPIN = spin_of_chi(CHI)
OMEGA = omega_of_a_alpha(SPIN, ALPHA)

t_data = np.zeros(ALPHA.shape)
phi_data = np.zeros(ALPHA.shape)
alpha_beta_data = np.zeros(ALPHA.shape)
phi_beta_data = np.zeros(ALPHA.shape)

def pn_flux_noprefactor(omega):
    return omega**(10./3.)

ALPHA, CHI = np.meshgrid(alpha_array, chi_array)
BETA, _ = np.meshgrid(beta_array, chi_array)
SPIN = spin_of_chi(CHI)
OMEGA = omega_of_a_alpha(SPIN, ALPHA)

ALPHA_FLUX, CHI_FLUX = np.meshgrid(alpha_flux_array, chi_flux_array)
SPIN_FLUX = spin_of_chi(CHI_FLUX)
OMEGA_FLUX = omega_of_a_alpha(SPIN_FLUX, ALPHA_FLUX)
flux_norm_array = flux_array/pn_flux_noprefactor(OMEGA_FLUX)
flux_spline = BicubicSpline(chi_flux_array, alpha_flux_array, flux_norm_array, bc = 'E(3)')

def time_of_beta(beta, gamma_max_squared):
    gamma_squared = beta**6*gamma_max_squared
    return np.expm1(gamma_squared)

print("Generating time, phase, and frequency evolution")
for i in range(chi_array.shape[0]):
    iterChi = chi_array.shape[0] - 1 - i # we fill things in backwards in order to define \gamma_max early on
    chi_val = chi_array[iterChi]
    a_val = spin_of_chi(chi_val)
    oISCO_val = kerr_isco_frequency(a_val)
    omega_data = omega_of_a_alpha(a_val, alpha_array)
    # flux_norm = flux_norm_array[iterChi]
    # flux_spline = CubicSpline(alpha_array, flux_norm, bc = 'E(3)')

    def dIdAlphaIntegrate(alpha, t):
        omega = omega_of_a_alpha(a_val, alpha)
        dOmega_dAlpha = omega_alpha_derivative(omega, oISCO_val)
        dE_dOmega = energy_omega_derivative(a_val, omega)
        Edot = flux_spline(chi_val, alpha)*pn_flux_noprefactor(omega)
        return dE_dOmega*dOmega_dAlpha/Edot*np.array([1., omega])
    
    insp = solve_ivp(dIdAlphaIntegrate, [0., 1.], [0., 0.], method='DOP853', t_eval=alpha_array, rtol=1.e-13, atol=1.e-14)
    t_data[iterChi] = insp.y[0]
    phi_data[iterChi] = insp.y[1]

    if iterChi == (chi_array.shape[0] - 1):
        t_max_value = insp.y[0][-1]
        gamma_max_squared = np.log(1 + t_max_value)
        t_array = time_of_beta(beta_array, gamma_max_squared)
        max_step_size = np.min(t_array[1:] - t_array[:-1])

    def dtdAlphaIntegrate(alpha, t):
        omega = omega_of_a_alpha(a_val, alpha)
        dOmega_dAlpha = omega_alpha_derivative(omega, oISCO_val)
        dE_dOmega = energy_omega_derivative(a_val, omega)
        Edot = flux_spline(chi_val, alpha)*pn_flux_noprefactor(omega)
        return dE_dOmega*dOmega_dAlpha/Edot*np.array([1.])
    
    def dJdTIntegrate(t, alphaVec):
        alpha = alphaVec[0]
        omega = omega_of_a_alpha(a_val, alpha)
        dOmega_dAlpha = omega_alpha_derivative(omega, oISCO_val)
        dE_dOmega = energy_omega_derivative(a_val, omega)
        Edot = flux_spline(chi_val, alpha)*pn_flux_noprefactor(omega)
        return np.array([1./(dE_dOmega*dOmega_dAlpha/Edot), omega])
    
    # def insp_event(t, y):
    #     sol = 1.
    #     for t0 in (t_array):
    #         sol *= (y[0] - t0)/(y[0] + t0 + 1)
    #     return sol

    # insp = solve_ivp(dtdAlphaIntegrate, [0., 1.], [0.], method='DOP853', t_eval=None, rtol=1.e-13, atol=1.e-14, events = insp_event, max_step=max_step_size)
    # alpha_select = insp.t_events[0][insp.y_events[0].T[0] >= (1. - 1.e-8)*t_array[1]]
    # alpha_beta_data[iterChi] = np.concatenate(([0.], alpha_select))

    def insp_event(t, y):
        return (y[0] - t_array[-1])
    
    def t_of_alpha_root(alpha, t0):
        insp = solve_ivp(dtdAlphaIntegrate, [0., alpha], [0.], method='DOP853', t_eval=None, rtol=1.e-13, atol = 1.e-16)
        t_val = insp.y[0]
        return t_val[-1] - t0

    # insp = solve_ivp(dtdAlphaIntegrate, [0., 1.], [0.], method='DOP853', t_eval=None, rtol=1.e-13, atol=1.e-14, events = insp_event)
    # alpha_select = insp.t_events[0]

    sol = root_scalar(t_of_alpha_root, args=(t_array[1]), method='brentq', bracket=[0, 1], rtol = 1.e-12)
    alpha_initial = sol.root

    insp = solve_ivp(dIdAlphaIntegrate, [0., alpha_initial], [0., 0.], method='DOP853', t_eval=[alpha_initial], rtol=1.e-13, atol = 1.e-16)
    phi_initial = insp.y[1][0]

    insp = solve_ivp(dJdTIntegrate, [t_array[1], t_array[-1]], [alpha_initial, phi_initial], method='DOP853', t_eval=t_array[1:], rtol=1.e-13, atol = 1.e-12)
    # if i < 10:
    print(i)
    print(1 - root_scalar(t_of_alpha_root, args=(t_array[-1]), method='brentq', bracket=[0, 1], rtol = 1.e-12).root/insp.y[0][-1])
    alpha_beta_data[iterChi] = np.concatenate(([0.], insp.y[0]))
    phi_beta_data[iterChi] = np.concatenate(([0.], insp.y[1]))

omega_beta_array = omega_of_a_alpha(SPIN, alpha_beta_data)

# print("Saving trajectory data to " + pathname + "/../../data/trajectory_v2.txt")

# header_txt = "chiN \t alphaN \n129\t257 \nchi\talpha\tflux\ttime\tphase\tbeta\tomega\tPhiT"
# full_data_array = np.array([CHI.flatten(), ALPHA.flatten(), flux_array.flatten(), t_data.flatten(), phi_data.flatten(), BETA.flatten(), omega_beta_array.flatten(), phi_beta_data.flatten()]).T
# np.savetxt(pathname + "/../../data/trajectory_v2.txt", full_data_array, header=header_txt, fmt = "%1.15e", comments='')

print("Saving trajectory data to " + pathname + "/../../data/trajectory_v4.txt")

header_txt = "chiN \t alphaN \n"+str(chi_array.shape[0])+"\t"+str(alpha_array.shape[0])+" \nchi\talpha\ttime\tphase\tbeta\tomega\tPhiT"
full_data_array = np.array([CHI.flatten(), ALPHA.flatten(), t_data.flatten(), phi_data.flatten(), BETA.flatten(), omega_beta_array.flatten(), phi_beta_data.flatten()]).T
np.savetxt(pathname + "/../../data/trajectory_v4.txt", full_data_array, header=header_txt, fmt = "%1.15e", comments='')