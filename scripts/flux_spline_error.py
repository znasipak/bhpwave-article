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

alphaa, aa = np.meshgrid(alphaFluxData, spin_of_chi(chiFluxData))
omegaa = omega_of_a_alpha(aa, alphaa)
fluxData = fluxData/(32./5.*pn_flux_noprefactor(omegaa))

Edot_78 = BicubicSpline(chiFluxData, alphaFluxData, fluxData, bc = "E(3)")
PhiCheck_78 = BicubicSpline(chiData, alphaData, phaseData, bc = "E(3)")
TCheck_78 = BicubicSpline(chiData, alphaData, timeData, bc = "E(3)")

downsample_rate_alpha=2
downsample_rate_chi=2
Edot_67 = BicubicSpline(np.ascontiguousarray(chiFluxData[::downsample_rate_chi]), np.ascontiguousarray(alphaFluxData[::downsample_rate_alpha]), np.ascontiguousarray(fluxData[::downsample_rate_chi,::downsample_rate_alpha]))
PhiCheck_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(phaseData[::downsample_rate_chi,::downsample_rate_alpha]))
TCheck_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(timeData[::downsample_rate_chi,::downsample_rate_alpha]))

alphaJs = np.linspace(0., 1., 302)
chiIs = np.linspace(0., 1., 302)
error_test=np.array([[abs(1-Edot_67(chiI, alphaJ)/Edot_78(chiI, alphaJ)) for chiI in chiIs] for alphaJ in alphaJs])
error_test[error_test==0.]=1.e-16

plt.pcolormesh(chiIs, alphaJs, np.log10(error_test), cmap='viridis', shading='gouraud')
plt.ylabel('$\\alpha$')
plt.xlabel('$\\chi$')
plt.title('$\Delta^\mathcal{F}_{(7,6)}(\\alpha, \\chi)$')
plt.colorbar()

fig_name = "flux_spline_error"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname+"/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi=100)