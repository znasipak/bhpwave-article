import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

import os
pathname = os.path.dirname(os.path.abspath(__file__))

spinWindow = -0.999*np.cos(np.linspace(0, np.pi/2, 5))**(1/3)
spinWindow = np.concatenate([spinWindow, -np.flip(spinWindow[:-1])])

time_spans = [0.5, 2, 5, 10, 25]
mass_ratio_list_range = [1e2, 1e7]
mass_ratio_list = np.logspace(np.log10(mass_ratio_list_range[0]), np.log10(mass_ratio_list_range[1]), 11)
M_list_range = [1e4, 1e8]
M_list = np.logspace(np.log10(M_list_range[0]), np.log10(M_list_range[1]), 9)

with open(pathname + '/../data/phase_difference_data.pkl', 'rb') as f:
    spinTests = pickle.load(f)

phase_diff = {}
for j, time_span in enumerate(time_spans):
    temp_array = np.zeros((len(mass_ratio_list), len(M_list)))
    for ii, mass_ratio in enumerate(mass_ratio_list):
        for jj, M in enumerate(M_list):
            if M/mass_ratio >= 1:
                temp_array[ii, jj] = np.max(spinTests[(M, M/mass_ratio)], axis = (0,1))[j]
    temp_array[temp_array==0] = None
    phase_diff[time_span] = temp_array

vmin_set = []
vmax_set = []
for ii, Tobs in enumerate(time_spans):
    vmin_set.append(np.nanmin(np.log10(phase_diff[Tobs].T)))
    vmax_set.append(np.nanmax(np.log10(phase_diff[Tobs].T)))

vmin = np.min(vmin_set)
vmax = np.max(vmax_set)

_, Z = np.meshgrid(-np.log10(mass_ratio_list), np.log10(M_list))

fig, axs = plt.subplots(1, 5, sharey=True, figsize=(20, 4))
fig.subplots_adjust(wspace=0.08)
# vmax = -1.1922573856978809
# vmin = -5.0125239674462385
# fig.subplots_adjust(hspace=2)
for ii, Tobs in enumerate(time_spans):
    pcm = axs[ii].pcolormesh(-np.log10(mass_ratio_list), np.log10(M_list), np.log10(phase_diff[Tobs].T), shading='auto', vmin=vmin, vmax=vmax)
    # title_string = '$||\delta \Phi^I||_{\hat{{a}},\hat{{\Omega}}}$ for $T_\mathrm{obs}$ = ' + (f"{Tobs} years")
    title_string = '$T_\mathrm{obs}$ = ' + (f"{Tobs} years")
    axs[ii].set_title(title_string)
    axs[ii].set_xlabel("$\log (\mu/M)$")

axs[0].set_ylabel("$\log (M/M_\odot)$")
# plt.plot(-np.log10(mass_ratio_list), Z.T, '.')
fig.subplots_adjust(right=0.8)
colorbar_axes = fig.add_axes([0.81, 0.11, 0.01, 0.77])
cbar = fig.colorbar(pcm, cax=colorbar_axes)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.set_ylabel("$||\\delta\\Phi^I||_{\\hat{a},\\hat{\\Omega}}$", rotation=270)
plt.savefig(pathname + "/../figures/deltaPhiTest.pdf", bbox_inches='tight')