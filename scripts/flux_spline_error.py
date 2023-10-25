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

EdotDataComp = np.load(pathname + "/../data/bhpwave_edot_redo.npy")
traj = np.loadtxt(pathname + "/../data/trajectory_downsampled_base.txt", skiprows=3)
trajHeader = np.loadtxt(pathname + "/../data/trajectory_downsampled_base.txt", skiprows=2, max_rows=1, dtype='str')
trajShape = np.loadtxt(pathname + "/../data/trajectory_downsampled_base.txt", skiprows=1, max_rows=1, dtype='int')

phaseData = np.ascontiguousarray(traj[:, 4].reshape(trajShape))
timeData = np.ascontiguousarray(traj[:, 3].reshape(trajShape))
fluxData = np.ascontiguousarray(traj[:, 2].reshape(trajShape))
chiData = np.ascontiguousarray(traj[:, 0].reshape(trajShape)[:, 0])
alphaData = np.ascontiguousarray(traj[:, 1].reshape(trajShape)[0])

alphaa, aa = np.meshgrid(alphaData, spin_of_chi(chiData))
omegaa = omega_of_a_alpha(aa, alphaa)
fluxData = 2.*EdotDataComp/(32./5.*omegaa**(10./3.))

Edot_78 = BicubicSpline(chiData, alphaData, fluxData, bc = "E(3)")
PhiCheck_78 = BicubicSpline(chiData, alphaData, phaseData, bc = "E(3)")
TCheck_78 = BicubicSpline(chiData, alphaData, timeData, bc = "E(3)")

downsample_rate_alpha=2
downsample_rate_chi=2
Edot_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(fluxData[::downsample_rate_chi,::downsample_rate_alpha]))
PhiCheck_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(phaseData[::downsample_rate_chi,::downsample_rate_alpha]))
TCheck_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(timeData[::downsample_rate_chi,::downsample_rate_alpha]))

alphaJs = np.linspace(0., 1., 302)
chiIs = np.linspace(0., 1., 302)
error_test=np.array([[abs(1-Edot_67(chiI, alphaJ)/Edot_78(chiI, alphaJ)) for chiI in chiIs] for alphaJ in alphaJs])
error_test[error_test==0.]=1.e-16

plt.pcolormesh(chiIs, alphaJs, np.log10(error_test), cmap='viridis', shading='gouraud')
plt.ylabel('$\\alpha$')
plt.xlabel('$\\chi$')
plt.title('$\delta_{(7,6)}(\\alpha, \\chi)$')
plt.colorbar()

fig_name = "flux_spline_error"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname+"/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi=100)