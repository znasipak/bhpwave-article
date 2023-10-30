from pybhpt.flux import FluxMode
from pybhpt.geo import KerrGeodesic
from pybhpt.teuk import TeukolskyMode
import numpy as np
from script_analysis_tools import *

# Compute amplitudes

# define filename for storing data
import os
pathname = os.path.dirname(os.path.abspath(__file__))
filename = "../../data/bhpwave_edot.npy"
filepath = pathname + filename
chi_sample_num = 129
alpha_sample_num = 257


chi = np.linspace(0, 1, chi_sample_num)
alpha = np.linspace(0, 1, alpha_sample_num)
chiA, alphaA = np.meshgrid(chi, alpha)
spinVals = spin_of_chi(chiA)
omegaVals = omega_of_a_alpha(spinVals, alphaA)

# comment out the next two lines if the code crashed or exited early and you want to restart the calculation
# starting with the last saved grid point
EdataGrid = np.zeros((chi.shape[0], alpha.shape[0]))
np.save(filepath, EdataGrid)

for i in range(chi.shape[0]):
    for j in range(alpha.shape[0]):
        EdataGrid = np.load(filepath)
        if EdataGrid[i, j] == 0.: # if the grid point has not been calculated
            a0 = spinVals[j, i]
            omega0 = omegaVals[j, i]
            r0 = kerr_circ_geo_radius(a0, omega0)
            x0 = 1
            if a0 < 0: # the code within FluxSummation is only set up to do a >= 0
                a0 *= -1
                x0 = -1
            sumclass = FluxSummation(a0, r0, 0., x0, 2**2)
            sumclass.sum() # tell the FluxSummation generator to sum over all of the flux modes to compute the total fluxes
            EdataGrid[i, j] = 2.*sumclass.totalfluxes[0] # extract Edot from the tuple of fluxes (Edot, Ldot, Qdot)
            # factor of two accounts for the sum class only summing over positive m-modes
            print(a0*x0, r0, EdataGrid[i, j])
            np.save(filepath, EdataGrid)