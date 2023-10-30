from pybhpt.flux import FluxMode
from pybhpt.geo import KerrGeodesic
from pybhpt.teuk import TeukolskyMode
import numpy as np
from script_analysis_tools import *

# Compute amplitudes

# define filename for storing data
import os
pathname = os.path.dirname(os.path.abspath(__file__))
filename = "../../data/bhpwave_amp.npy"
filepath = pathname + filename
chi_sample_num = 64
alpha_sample_num = 64
lmax = 20

chi = np.linspace(0, 1, chi_sample_num)
alpha = np.linspace(0, 1, alpha_sample_num)
chiA, alphaA = np.meshgrid(chi, alpha)
spinVals = spin_of_chi(chiA)
omegaVals = omega_of_a_alpha(spinVals, alphaA)

# comment out the next two lines if the code crashed or exited early and you want to restart the calculation
# starting with the last saved grid point
mode_lm = np.zeros((chi.shape[0], alpha.shape[0], lmax - 1, lmax + 1), dtype=np.complex128)
np.save(filepath, mode_lm)

for i in range(chi.shape[0]):
    for j in range(alpha.shape[0]):
        mode_lm = np.load(filepath)
        if np.any(np.abs(mode_lm[i, j]) == 0.): # if the grid point has not been calculated
            a0 = spinVals[j, i]
            omega0 = omegaVals[j, i]
            r0 = kerr_circ_geo_radius(a0, omega0)
            x0 = 1
            if a0 < 0: # the code within FluxSummation is only set up to do a >= 0
                a0 *= -1
                x0 = -1
            wave=WaveAmplitude(a0, r0, 0, x0)
            for emm in range(1, ell + 1):
                mode_temp = wave.spherical_amplitudes(lmax, emm, 0)
                if x0 < 0:
                    mode_temp = np.conj(mode_temp)
                for ell in range(2, lmax + 1):
                    mode_lm[i, j, ell, emm] = mode_temp(ell - 2)
            np.save(filepath, mode_lm)