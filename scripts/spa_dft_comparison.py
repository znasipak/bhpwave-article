from bhpwave.waveform import KerrWaveform, Modot_GC1_to_S
from bhpwave.harmonics.amplitudes import HarmonicAmplitudes
from lisatools.sensitivity import get_sensitivity

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

import os
pathname = os.path.dirname(os.path.abspath(__file__))

# parameters
dt = 15.0  # seconds
T = 4.0  # years
M = 1e6
a = 0.9  # will be ignored in Schwarzschild waveform
mu = 3e1
p0 = 13.55
e0 = 0.0
x0 = 1.0  # will be ignored in Schwarzschild waveform
qK = 0.8  # polar spin angle
phiK = 0.2  # azimuthal viewing angle
qS = 0.3  # polar sky angle
phiS = 1.3  # azimuthal viewing angle
# phiK = 0.8  # azimuthal viewing angle
dist = 2.0  # distance
Phi_phi0 = 0.2
Phi_theta0 = 0.0
Phi_r0 = 0.0

kerrwave_fd = KerrWaveform(frequency_domain=True)
kerrwave = KerrWaveform(frequency_domain=False)

# use same set of modes
modes = np.array([(2,2), (2,1), (3,3), (3,2)], dtype=np.int32)
ht = kerrwave(
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    T=T,
    dt=dt,
    num_threads=8,
    pad_output=True,
   select_modes=modes
)
hf=kerrwave_fd(
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    T=T,
    dt=dt,
    num_threads=8,
    pad_output=True,
   select_modes=modes
)

freq = np.fft.rfftfreq(2*hf[0].shape[0], d=dt)[1:]
hf1 = np.conj(np.fft.rfft([ht.real, -ht.imag])[:, 1:])*dt


plt.rcParams["figure.figsize"] = (6,6)
plt.plot(freq[::10000], np.abs(hf1[0][::10000]), label='DFT', linewidth=2)
plt.plot(freq[::10000], np.abs(hf[0][::10000]),'--', label='SPA', linewidth=1.2)
plt.plot(freq[::10000], np.sqrt(get_sensitivity(freq[::10000])), 'k-.', label='$\sqrt{S_n(f)}$', linewidth=1.5)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$f$ (Hz)')
plt.ylabel('$|\\tilde{h}_+(f)|$')
plt.legend(loc='upper left')
plt.xlim(1e-4, 3.3e-2)
plt.ylim(1e-21, 1e-16)
plt.savefig(pathname + "/../figures/dft_comparison.pdf", bbox_inches='tight')