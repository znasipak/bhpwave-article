import numpy as np
from bhpwave.trajectory.geodesic import kerr_isco_frequency, kerr_circ_geo_radius

OMEGA_MIN = 2.e-3
A_MAX = 0.9999

Modot_MKS = 1.98841e+30 # kg
GM_MKS = 1.32712440041279419e+20 # m^3/s^2
c_MKS = 299792458. # m/s
pc_MKS = 3.0856775814913674e+16 # m
yr_MKS = 31558149.763545603 # s (sidereal year)

Modot_GC1_to_S = GM_MKS/c_MKS**3
Modot_GC1_to_M = GM_MKS/c_MKS**2
Modot_GC1_to_PC = Modot_GC1_to_M/pc_MKS

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

def pn_flux_noprefactor(omega):
    return omega**(10./3.)

def pn_time_noprefactor(a, omega):
    oISCO = kerr_isco_frequency(a)
    offset = oISCO**(-8/3) + 1.e-6
    return -np.abs(offset - omega**(-8./3.))

def pn_phase_noprefactor(a, omega):
    oISCO = kerr_isco_frequency(a)
    offset = oISCO**(-5/3) - 1.e-6
    return -np.abs(offset - omega**(-5./3.))

def pn_flux_noprefactor_domega(omega):
    return (10./3.)*omega**(7./3.)

def pn_time_noprefactor_domega(omega):
    return -(-8/3)*omega**(-11./3.)

def pn_phase_noprefactor_domega(omega):
    return -(-5/3)*omega**(-8./3.)

def load_trajectory_data_file(filepath):
    traj = np.loadtxt(filepath, skiprows=3)
    #trajHeader = np.loadtxt(filepath, skiprows=2, max_rows=1, dtype='str')
    trajShape = np.loadtxt(filepath, skiprows=1, max_rows=1, dtype='int')

    fluxDataTemp = np.ascontiguousarray(traj[:, 2].reshape(trajShape[:2]))
    alphaDataFluxTemp = np.ascontiguousarray(traj[:, 1].reshape(trajShape[:2])[0])
    chiDataFluxTemp = np.ascontiguousarray(traj[:, 0].reshape(trajShape[:2])[:, 0])

    phaseData = np.ascontiguousarray(traj[:, 4].reshape(trajShape[:2]))
    timeData = np.ascontiguousarray(traj[:, 3].reshape(trajShape[:2]))
    alphaData = np.ascontiguousarray(traj[:, 1].reshape(trajShape[:2])[0])
    chiData = np.ascontiguousarray(traj[:, 0].reshape(trajShape[:2])[:, 0])
    betaData = np.ascontiguousarray(traj[:, 5].reshape(trajShape[:2])[0])
    omegaData = np.ascontiguousarray(traj[:, 6].reshape(trajShape[:2]))
    phaseBetaData = np.ascontiguousarray(traj[:, 7].reshape(trajShape[:2]))

    fluxDownsampleChi = int((trajShape[0] - 1)/(trajShape[2] - 1))
    fluxDownsampleAlpha = int((trajShape[1] - 1)/(trajShape[3] - 1))

    fluxData = fluxDataTemp[::fluxDownsampleChi, ::fluxDownsampleAlpha]
    alphaDataFlux = alphaDataFluxTemp[::fluxDownsampleAlpha]
    chiDataFlux = chiDataFluxTemp[::fluxDownsampleChi]

    return chiData, alphaData, timeData, phaseData, betaData, omegaData, phaseBetaData, chiDataFlux, alphaDataFlux, fluxData
    