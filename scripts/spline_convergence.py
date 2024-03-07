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

from script_analysis_tools import *

traj_data_full = load_trajectory_data_file(pathname+"/../data/trajectory.txt")

timeData_temp = traj_data_full[2]
phaseData_temp = traj_data_full[3]

chiData = traj_data_full[7]
alphaData = traj_data_full[8]
fluxData = traj_data_full[9]

downsample_chi = int((phaseData_temp.shape[0] - 1)/(fluxData.shape[0] - 1))
downsample_alpha = int((phaseData_temp.shape[1] - 1)/(fluxData.shape[1] - 1))
phaseData = phaseData_temp[::downsample_chi, ::downsample_alpha]
timeData = timeData_temp[::downsample_chi, ::downsample_alpha]

alphaa, aa = np.meshgrid(alphaData, spin_of_chi(chiData))
omegaa = omega_of_a_alpha(aa, alphaa)
fluxDataNorm = fluxData/(32./5.*omegaa**(10./3.))

Edot_78 = BicubicSpline(chiData, alphaData, fluxDataNorm, bc = "E(3)")
# PhiCheck_78 = BicubicSpline(chiData, alphaData, phaseData, bc = "E(3)")
# TCheck_78 = BicubicSpline(chiData, alphaData, timeData, bc = "E(3)")

downsample_rate_alpha=2
downsample_rate_chi=2
Edot_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(fluxDataNorm[::downsample_rate_chi,::downsample_rate_alpha]))
# PhiCheck_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(phaseData[::downsample_rate_chi,::downsample_rate_alpha]))
# TCheck_67 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(timeData[::downsample_rate_chi,::downsample_rate_alpha]))

downsample_rate_alpha=4
downsample_rate_chi=4
Edot_56 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(fluxDataNorm[::downsample_rate_chi,::downsample_rate_alpha]))
# PhiCheck_56 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(phaseData[::downsample_rate_chi,::downsample_rate_alpha]))
# TCheck_56 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(timeData[::downsample_rate_chi,::downsample_rate_alpha]))

downsample_rate_alpha=8
downsample_rate_chi=8
Edot_45 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(fluxDataNorm[::downsample_rate_chi,::downsample_rate_alpha]))
# PhiCheck_45 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(phaseData[::downsample_rate_chi,::downsample_rate_alpha]))
# TCheck_45 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(timeData[::downsample_rate_chi,::downsample_rate_alpha]))

downsample_rate_alpha=16
downsample_rate_chi=16
Edot_34 = BicubicSpline(np.ascontiguousarray(chiData[::downsample_rate_chi]), np.ascontiguousarray(alphaData[::downsample_rate_alpha]), np.ascontiguousarray(fluxDataNorm[::downsample_rate_chi,::downsample_rate_alpha]))

alphaJs = np.linspace(0., 1., 502)
chiIs = np.linspace(0., 1., 502)
error_test=np.array([[abs(1-Edot_67(chiI, alphaJ)/Edot_78(chiI, alphaJ)) for chiI in chiIs] for alphaJ in alphaJs])
error_test[error_test==0.]=1.e-16
error_test_2=np.array([[abs(1-Edot_56(chiI, alphaJ)/Edot_78(chiI, alphaJ)) for chiI in chiIs] for alphaJ in alphaJs])
error_test_2[error_test_2==0.]=1.e-16
error_test_3=np.array([[abs(1-Edot_45(chiI, alphaJ)/Edot_78(chiI, alphaJ)) for chiI in chiIs] for alphaJ in alphaJs])
error_test_3[error_test_3==0.]=1.e-16
error_test_4=np.array([[abs(1-Edot_34(chiI, alphaJ)/Edot_78(chiI, alphaJ)) for chiI in chiIs] for alphaJ in alphaJs])
error_test_4[error_test_4==0.]=1.e-16

plt.plot([6, 5, 4, 3], [np.max(error_test_2)*16**(5-n) for n in [6, 5, 4, 3]], 'k--', label = "$\propto 2^{-4n}$", linewidth=1)
plt.plot([6, 5, 4, 3], [np.max(error_test), np.max(error_test_2), np.max(error_test_3), np.max(error_test_4)], '.', markersize=10)
plt.yscale('log')
plt.ylabel('$||\Delta^\mathcal{F}_{(n+1,n)}||_\infty$')
plt.xlabel('$n$')
plt.legend()

fig_name = "spline_convergence"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname + "/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi = 300)