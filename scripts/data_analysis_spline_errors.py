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

with open(pathname + '/../figures/saved_phase_analysis.pkl', 'rb') as f:
    phase_analysis = pickle.load(f)

M_list = [5e5, 1e6, 5e6]
mu_list = [10, 50]

deltaM_bias = {}
deltaM_var = {}
deltaM_alist = {}

for param_k in [0, 1, 2, 3]:
    for M_test in M_list:
        for mu_test in mu_list:
            params_test = np.array([M_test, mu_test])
            deltaM_bias[(M_test, mu_test, param_k)] = []
            deltaM_var[(M_test, mu_test, param_k)] = []
            deltaM_alist[(M_test, mu_test, param_k)] = []
            for num in phase_analysis.keys():
                if np.all(phase_analysis[num]["params"][:2] == params_test):
                    deltaM_var[(M_test, mu_test, param_k)].append(phase_analysis[num]["var"][param_k])
                    deltaM_bias[(M_test, mu_test, param_k)].append(phase_analysis[num]["biases"][param_k])
                    deltaM_alist[(M_test, mu_test, param_k)].append(phase_analysis[num]["params"][2])
            deltaM_var[(M_test, mu_test, param_k)] = np.array(deltaM_var[(M_test, mu_test, param_k)])
            deltaM_bias[(M_test, mu_test, param_k)] = np.array(deltaM_bias[(M_test, mu_test, param_k)])
            deltaM_alist[(M_test, mu_test, param_k)] = np.array(deltaM_alist[(M_test, mu_test, param_k)])

N = len(M_list)*len(mu_list)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))

line_list = ["o-", "d-", "v-", "s-", "X-", "p-"]
dashed_list = ["o--", "d--", "v--", "s--", "X--", "p--"]
param_list = ["$\Delta \log (M/M_\odot)$", "$\Delta \log (\mu/M_\odot)$", "$\Delta \hat{a}$", "$\Delta p_0$"]

fig, axs = plt.subplots(4,sharex=True, figsize=(7, 20))
# plt.ylim(1e-12, 1) 
for param_k in [0, 1, 2, 3]:
    # axs[param_k].xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    jj = 0
    for M_test in M_list:
        for mu_test in mu_list:
            sorted_args = np.argsort([deltaM_alist[(M_test, mu_test, param_k)]])
            axs[param_k].plot(1-deltaM_alist[(M_test, mu_test, param_k)][sorted_args][0], np.abs(deltaM_var[(M_test, mu_test, param_k)])[sorted_args][0], line_list[jj], label = "$({:1.0f}\\times 10^{:1.0f}, {:.0f})$".format(M_test/(10**int(np.log10(M_test))), int(np.log10(M_test)), mu_test))
            axs[param_k].set_yscale('log')
            axs[param_k].set_xscale('log')
            axs[param_k].set_ylabel(param_list[param_k])
            jj += 1

    jj = 0
    for M_test in M_list:
        for mu_test in mu_list:
            sorted_args = np.argsort([deltaM_alist[(M_test, mu_test, param_k)]])
            axs[param_k].plot(1-deltaM_alist[(M_test, mu_test, param_k)][sorted_args][0], np.abs(deltaM_bias[(M_test, mu_test, param_k)])[sorted_args][0], dashed_list[jj])
            axs[param_k].set_yscale('log')
            axs[param_k].set_xscale('log')
            jj += 1

axs[2].legend(ncol=2, fontsize=14)
axs[param_k].set_xlabel("$ 1- \hat{{a}}$")
plt.subplots_adjust(hspace=0.05)
axs[1].legend(custom_lines, ["$\Delta \\theta^i_\mathrm{stat}$", "$\Delta \\theta^i_\mathrm{bias}$"])

for param_k in [0, 1, 2, 3]:
    # axs[param_k].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    axs[param_k].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

plt.savefig(pathname + "/../figures/phase_error.pdf", bbox_inches='tight')