import pickle
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 18})

from matplotlib import ticker as mticker
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color="k", lw=1),
                Line2D([0], [0], color="k", lw=1, linestyle="--")]

import os
pathname = os.path.dirname(os.path.abspath(__file__))

with open(pathname + '/../figures/saved_flux_analysis.pkl', 'rb') as f:
    flux_analysis = pickle.load(f)

M_array = [1e6, 5e6]
deltaPhi_dict = {}
Rint_dict = {}
Rext_dict = {}
aval_dict = {}
Mbias_dict = {}
deriv_inds_intrinsic = [0, 1, 2, 3, 11]

for M_select in M_array:
    deltaPhi_dict[M_select] = []
    Rint_dict[M_select] = []
    Rext_dict[M_select] = []
    aval_dict[M_select] = []
    Mbias_dict[M_select] = []
    for num in flux_analysis.keys():
        if flux_analysis[num]["params"][0] == M_select:
            bias = flux_analysis[num]["biases"]
            var = flux_analysis[num]["var"]
            Rfull = np.abs(bias/var)
            Rint = np.max(Rfull[:len(deriv_inds_intrinsic)])
            Rext = np.max(Rfull[len(deriv_inds_intrinsic):])
            Rint_dict[M_select].append(Rint)
            Rext_dict[M_select].append(Rext)
            aval_dict[M_select].append(flux_analysis[num]["params"][2])
            Mbias_dict[M_select].append(flux_analysis[num]["Mbias"])
            deltaPhi_dict[M_select].append(flux_analysis[num]["dephase"])

N = 4
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))

fig, axs = plt.subplots(1,2,sharey=True, figsize=(14, 5))

for jj, Mval in enumerate([1e6, 5e6]):
    axs[jj].loglog(1-np.array(aval_dict[Mval]), np.array(deltaPhi_dict[Mval])*0. + 1., 'k--', linewidth=1)
    axs[jj].loglog(1-np.array(aval_dict[Mval]), Mbias_dict[Mval], 'o-', label = '$\mathcal{M}_\mathrm{bias}$')
    axs[jj].loglog(1-np.array(aval_dict[Mval]), Rext_dict[Mval], 'd-', label = '$\mathcal{R}_\mathrm{ext}$')
    axs[jj].loglog(1-np.array(aval_dict[Mval]), Rint_dict[Mval], 'v-', label = '$\mathcal{R}_\mathrm{int}$')
    axs[jj].loglog(1-np.array(aval_dict[Mval]), deltaPhi_dict[Mval], 's-', label = '$|\Delta\Phi_\mathrm{GW}|$')
    axs[jj].set_xlabel("$1 - \hat{a}$")

axs[0].set_title("$M = 1\\times10^6 M_\odot$", fontsize=18)
axs[1].set_title("$M = 5\\times 10^6 M_\odot$", fontsize=18)
axs[0].set_ylabel("magnitude")
axs[0].legend(fontsize=14)
plt.ylim(1e-6, 1e2)
plt.subplots_adjust(wspace=0.1)
plt.savefig(pathname + "/../figures/flux_error.pdf", bbox_inches='tight')