from bhpwave.waveform import KerrWaveform
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})

import os
pathname = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner
# from lisatools.utils.utility import AET

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.sensitivity import get_sensitivity
from eryn.utils import TransformContainer
from few.utils.constants import YRSID_SI

import mpmath as mp
def pinv_solve_with_error(mat, U, S, V, count):
    # a function to calculate the pseudoinverse of a function using multiprecision.
    #
    # method: the pseudoinverse is calculated using the singular value decomposition
    # for matrices with large condition numbers, the solver tries to find the optimal
    # number of singular values to include to return the closest approximation to the
    # true pseudoinverse
    temp_list = []
    for val in S[:count+1]:
        temp_list.append(val ** (-1))
    for val in S[count+1:]:
        temp_list.append(0.)
    Sinv = mp.diag(temp_list)  # get S**-1
    temp2 = V.T * Sinv * U.T  # construct pseudo-inverse
    mat_pinv = np.array(temp2.tolist(), dtype=np.float64)
    # check to see if covariances are actually inverses of fisher matrix
    Ip_mat = np.dot(mat_pinv, mat)
    mat_pinv_comp = np.dot(mat, Ip_mat)
    mat_pinv_comp[np.abs(mat_pinv_comp) == 0.] = 1.e-100
    error = np.max(np.abs(1-mat/mat_pinv_comp))
    return error, mat_pinv

def pinv_solve(mat, tol_base = 1e-2):
    # a function to calculate the pseudoinverse of a function using multiprecision.
    #
    # method: the pseudoinverse is calculated using the singular value decomposition
    # for matrices with large condition numbers, the solver tries to find the optimal
    # number of singular values to include to return the closest approximation to the
    # true pseudoinverse
    
    mp.mp.dps = 500
    mat_temp = mat.copy()
    mat_mp = mp.matrix(mat_temp.tolist())
    U, S, V = mp.svd_r(mat_mp)  # singular value decomposition
    max_value = np.max(np.abs(S))
    count = 0
    val = S[count]
    while count < mat.shape[0] and abs(val/max_value) > tol_base:
        count += 1
        val = S[count]
    count -= 1

    error_array = []
    pinv_array = []
    while count < mat.shape[0]:
        data = pinv_solve_with_error(mat, U, S, V, count)
        error_array.append(data[0])
        pinv_array.append(data[1])
        count += 1
    error_array = np.array(error_array)
    pinv_array = np.array(pinv_array)

    min_error = np.min(error_array)
    mat_pinv = pinv_array[error_array == min_error][0]

    return mat_pinv

wave_gen = KerrWaveform()

M = 1e6  # primary mass in solar masses
mu = 1e5 # secondary mass in solar masses
a = 0.9 # dimensionless spin of the primary
p0 = 12.55 # initial semi-latus rectum
e0 = 0.0 # eccentricity is ignored for circular orbits
x0 = 1.0  # inclination is ignored for circular orbits
qS = np.arccos(0.8)  # polar angle of Kerr spin angular momentum
phiS = 0.2  # azimuthal angle of Kerr spin angular momentum
qK = np.arccos(0.3)  # polar sky angle
phiK = 0.3  # azimuthal sky angle
dist = 20.0  # distance to source in Gpc
Phi_phi0 = 0.2 # initial azimuthal position of the secondary
Phi_theta0 = 0.0 # ignored for circular orbits
Phi_r0 = 0.0 # ignored for circular orbits
dt = 0.5  # time steps in seconds
T = 0.00075  # waveform duration in years

injection_paramters = [M, mu, a, p0, e0, x0,
                       dist, qS, phiS, qK, phiK,
                       Phi_phi0, Phi_theta0, Phi_r0,
                       dt, T]

samples = np.load(pathname + "/../data/mcmc_samples_data.npy")

deriv_inds_intrinsic = [0, 1, 2, 3, 11]

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rc('font', **{'size' : 14})
corner_kwargs=dict(
    labels=[
        r"$\log M$",
        r"$\log \mu$",
        r"$\hat{a}$",
        r"$p_0$",
        r"$\Phi_{\phi 0}$",
    ],
    show_titles=True,
    title_fmt='1.3f',
    levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2),
    quiet=True,
    smooth=0.7,
    label_kwargs=dict(fontsize=24),
    title_kwargs=dict(fontsize=18)
)
cmap = plt.cm.get_cmap('plasma', 5)

fish_intrinsic = np.load(pathname + "/../data/fisher_results.npy")

Gamma = fish_intrinsic
inv_Gamma = pinv_solve(Gamma)
injection_params=np.array(injection_paramters).copy()
injection_params[0] = np.log(injection_params[0])
injection_params[1] = np.log(injection_params[1])

truths=injection_params[deriv_inds_intrinsic]
fisher_samples = np.random.multivariate_normal(truths, inv_Gamma, size=samples.shape[0])
fig1 = corner.corner(samples,     
    truths=truths,
    color=cmap(0),
    scale_hist = False,
    **corner_kwargs)
fig2 = corner.corner(fisher_samples,
    fig=fig1,
    color=cmap(3),
    scale_hist = False,
    **corner_kwargs
)
plt.savefig(pathname + "/../figures/mcmc.pdf")