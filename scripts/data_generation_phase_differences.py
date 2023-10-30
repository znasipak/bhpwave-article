import pickle
import numpy as np
from script_analysis_tools import *
from bhpwave.spline import BicubicSpline

import os
pathname = os.path.dirname(os.path.abspath(__file__))

traj_data_full = load_trajectory_data_file(pathname+"/../data/trajectory.txt")

chiData = traj_data_full[0]
alphaData = traj_data_full[1]
timeData = traj_data_full[2]
phaseData = traj_data_full[3]
betaData = traj_data_full[4]
omegaData = traj_data_full[5]
phaseBetaData = traj_data_full[6]

chiDataFlux = traj_data_full[7]
alphaDataFlux = traj_data_full[8]
fluxData = traj_data_full[9]

phiNorm2 = np.log1p(-phaseBetaData)

ALPHA_FLUX, CHI_FLUX = np.meshgrid(alphaDataFlux, chiDataFlux)
SPIN_FLUX = spin_of_chi(CHI_FLUX)
OMEGA_FLUX = omega_of_a_alpha(SPIN_FLUX, ALPHA_FLUX)
flux_norm_array = fluxData/pn_flux_noprefactor(OMEGA_FLUX)

ALPHA, CHI = np.meshgrid(alphaData, chiData)
BETA, _ = np.meshgrid(betaData, chiData)
SPIN = spin_of_chi(CHI)
OMEGA = omega_of_a_alpha(SPIN, ALPHA)
phase_norm_array = (phaseData)/pn_phase_noprefactor(SPIN, OMEGA)
time_norm_array = (timeData)/pn_time_noprefactor(SPIN, OMEGA)
alpha_beta_array = alpha_of_a_omega(SPIN, omegaData)

flux_spl = BicubicSpline(chiDataFlux, alphaDataFlux, flux_norm_array, bc = 'E(3)')
phi_spl = BicubicSpline(chiData, betaData, phiNorm2)
time_spl = BicubicSpline(chiData, alphaData, time_norm_array, bc = 'E(3)')
phase_spl = BicubicSpline(chiData, alphaData, phase_norm_array, bc = 'E(3)')
freq_spl = BicubicSpline(chiData, betaData, alpha_beta_array, bc = 'E(3)')

def flux_of_a_omega(a, omega):
    chi = chi_of_spin(a)
    oISCO = kerr_isco_frequency(a)
    alpha = alpha_of_omega_ISCO(omega, oISCO)
    return flux_spl(chi, alpha)*pn_flux_noprefactor(omega)

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

def omega_of_time(a, t, spl_num = 0):
    return omega_of_a_time(a, t, freq_spl[spl_num])

def phase_of_time(a, t, spl_num = 0):
    chi = chi_of_spin(a)
    beta = beta_of_time(a, t)
    
    return -np.expm1(phi_spl[spl_num](chi, beta))

from scipy.integrate import solve_ivp

ISCOWEIGHT = 0.995
def dJdTIntegrate(t, alphaVec, a_val):
    alpha = alphaVec[0]
    oISCO_val = kerr_isco_frequency(a_val)
    omega = omega_of_a_alpha(a_val, alpha)
    if omega >= ISCOWEIGHT*oISCO_val:
        return [0., 0.]
    dOmega_dAlpha = omega_alpha_derivative(omega, oISCO_val)
    dE_dOmega = energy_omega_derivative(a_val, omega)
    chi_val = chi_of_spin(a_val)
    Edot = flux_spl(chi_val, alpha)*pn_flux_noprefactor(omega)
    return np.array([-1./(dE_dOmega*dOmega_dAlpha/Edot), omega])

def phase_test_integration(Tobs, M, mu, a, omega0):
    epsilon = mu/M
    M_yrs = M*Modot_GC1_to_S/yr_MKS
    deltaT = Tobs*epsilon/M_yrs
    alpha0 = alpha_of_a_omega(a, omega0)
    deltaT = np.array(deltaT)
    insp = solve_ivp(dJdTIntegrate, [0, 2*deltaT[-1]], [alpha0, 0.], args=(a,), method='RK45', t_eval=deltaT, rtol=1.e-12, atol = 1.e-10)
    if not insp.success:
        insp = solve_ivp(dJdTIntegrate, [0, 4*deltaT[-1]], [alpha0, 0.], args=(a,), method='RK45', t_eval=deltaT, rtol=1.e-9, atol = 1.e-8)
    phase = np.ones(deltaT.shape[0])*insp.y[1][-1]/epsilon
    if deltaT.shape[0] != insp.y[1].shape[0]:
        print("Requested time samples not the correct length")
        print(insp.y[1].shape[0], deltaT.shape[0])
        print(omega_of_a_alpha(a, insp.y[0][-1]))
    phase[:insp.y[1].shape[0]] = insp.y[1]/epsilon
    return phase

def phase_change_Tobs(Tobs, M, mu, a, omega_i, spl_num = 0):
    epsilon = mu/M
    t_i = time_of_a_omega(a, omega_i, spl_num = spl_num)
    t_ISCO = time_of_a_omega(a, ISCOWEIGHT*kerr_isco_frequency(a), spl_num = spl_num)
    Phi_i = phase_of_time(a, t_i, spl_num = spl_num)
    M_yrs = M*Modot_GC1_to_S/yr_MKS
    deltaT = Tobs*epsilon/M_yrs
    t_f = t_i + deltaT
    t_f[t_f > t_ISCO] = t_ISCO
    Phi_f = np.array([phase_of_time(a, t, spl_num = spl_num) for t in t_f])
    return (Phi_f - Phi_i)/epsilon

def phase_compare_integration(M, mu, a, t_array = np.linspace(0.05, 50, 10)):
    frequencyWindow = np.logspace(np.log10(0.0022), np.log10(0.95*kerr_isco_frequency(a)), 17)
    return np.array([np.abs(phase_change_Tobs(np.array(t_array), M, mu, a, omega, spl_num = 0) - phase_test_integration(np.array(t_array), M, mu, a, omega)) for omega in frequencyWindow])

spinWindow = -0.999*np.cos(np.linspace(0, np.pi/2, 5))**(1/3)
spinWindow = np.concatenate([spinWindow, -np.flip(spinWindow[:-1])])

time_spans = [0.5, 2, 5, 10, 25]
mass_ratio_list_range = [1e2, 1e7]
mass_ratio_list = np.logspace(np.log10(mass_ratio_list_range[0]), np.log10(mass_ratio_list_range[1]), 11)
M_list_range = [1e4, 1e8]
M_list = np.logspace(np.log10(M_list_range[0]), np.log10(M_list_range[1]), 9)
spinTests = {}
for mass_ratio in mass_ratio_list:
    for M in M_list:
        if M/mass_ratio >= 1:
            print(M, M/mass_ratio)
            tmp =np.array([phase_compare_integration(M, M/mass_ratio, a, t_array = time_spans) for a in spinWindow])
            spinTests[(M, M/mass_ratio)] = np.array([phase_compare_integration(M, M/mass_ratio, a, t_array = time_spans) for a in spinWindow])

with open(pathname + '/phase_difference_data.pkl', 'wb') as f:
    pickle.dump(spinTests, f)