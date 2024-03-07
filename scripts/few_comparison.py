from bhpwave.waveform import KerrWaveform, Modot_GC1_to_S
from bhpwave.harmonics.amplitudes import HarmonicAmplitudes

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *

import os
pathname = os.path.dirname(os.path.abspath(__file__))

# parameters
dt = 15.0  # seconds
T = 4.0  # years
M = 1e6
a = 0.0  # will be ignored in Schwarzschild waveform
mu = 1e1
p0 = 12.05
e0 = 0.0
x0 = 1.0  # will be ignored in Schwarzschild waveform
qK = 0.8  # polar spin angle
phiK = 0.2  # azimuthal viewing angle
qS = 0.3  # polar sky angle
phiS = 0.3  # azimuthal viewing angle
dist = 1.0  # distance
Phi_phi0 = 0.2
Phi_theta0 = 0.0
Phi_r0 = 0.0

few_wave = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux")
bhp_wave = KerrWaveform()

# make sure we are using the same number of modes
specific_modes=bhp_wave.select_modes(M,mu,a,p0,qS,phiS,qK,phiK,Phi_phi0,T=T,dt=dt, eps=1.e-5, pad_nmodes=True)

hF = few_wave(M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T=T,dt=dt,mode_selection=specific_modes)
hB = bhp_wave(M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T=T,dt=dt,num_threads=16,select_modes=specific_modes)

hFComp=hF[:hB.shape[0]]
hBComp=hB[:hF.shape[0]]
t=np.arange(0, hFComp.shape[0])*dt

deltaPhiGW_unscaled = np.unwrap(np.angle(hFComp))-np.unwrap(np.angle(hBComp))

# we have to rescale FEW time sampling because their constants 
# are not consistently defined nor do they agree with ours
miscaling_factor = (Modot_GC1_to_S)/MTSUN_SI

hF_rescale = few_wave(miscaling_factor*M,miscaling_factor*mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T=T/miscaling_factor,dt=dt/miscaling_factor,mode_selection=specific_modes)
hB_rescale = bhp_wave(M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T=T,dt=dt,num_threads=16,select_modes=specific_modes)

hFComp_rescale=hF_rescale[:hB_rescale.shape[0]]
hBComp_rescale=hB_rescale[:hF_rescale.shape[0]]

plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (14.5,3)

delta_index = 300
initial_index = 0

fig, axs = plt.subplots(1, 3, sharey=True)

axs[0].plot(t[initial_index:initial_index+delta_index],-hFComp.imag[initial_index:initial_index+delta_index], label="FEW", linewidth=2)
axs[0].plot(t[initial_index:initial_index+delta_index],-hBComp.imag[initial_index:initial_index+delta_index],'--', label="BHPW", linewidth=2)
axs[0].set_xlabel(r"$t$ (sec)")
axs[0].set_ylabel(r"$h_\times$")
axs[0].legend(fontsize=12)

initial_index = 3*10**6
axs[1].plot(t[initial_index:initial_index+delta_index],hFComp.imag[initial_index:initial_index+delta_index], label="FEW", linewidth=2)
axs[1].plot(t[initial_index:initial_index+delta_index],hBComp.imag[initial_index:initial_index+delta_index],'--', label="BHPW", linewidth=2)
axs[1].set_xlabel(r"$t$ (sec)")

initial_index = -500
axs[2].plot(t[initial_index:initial_index+delta_index],hFComp.imag[initial_index:initial_index+delta_index], label="FEW", linewidth=2)
axs[2].plot(t[initial_index:initial_index+delta_index],hBComp.imag[initial_index:initial_index+delta_index],'--', label="BHPW", linewidth=2)
axs[2].set_xlabel(r"$t$ (sec)")

plt.subplots_adjust(wspace=0.1)

fig_name = "few_bhpwave_overlay"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname + "/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi = 300)

fig = plt.figure()

# heavily downsample the data so that it is easier to see the dashes in the first data set
plt.plot(np.arange(hFComp.shape[0])[::50000]*dt/YRSID_SI, deltaPhiGW_unscaled[::50000], 'c-', label='unscaled FEW', lw=2)
plt.plot(np.arange(hFComp.shape[0])[::50000]*dt/YRSID_SI, (np.unwrap(np.angle(hFComp_rescale))-np.unwrap(np.angle(hBComp_rescale)))[::50000], 'k--', label='scaled FEW', lw=2)
plt.xlabel(r"$t$ (yrs)")
plt.ylabel(r"$\Delta\Phi_\mathrm{GW}$")
plt.legend()

fig_name = "few_phase_comparison"
print("Saving figure to " + pathname + "/../figures/" + fig_name + ".pdf")
plt.savefig(pathname + "/../figures/" + fig_name + ".pdf", bbox_inches="tight", dpi = 300)