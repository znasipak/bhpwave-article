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

h = wave_gen(*injection_paramters, pad_output=True)

use_gpu = False
Tobs=T
emri_injection_params_temp = [M, mu, a, p0, e0, x0,
                       dist, qS, phiS, qK, phiK,
                       Phi_phi0, Phi_theta0, Phi_r0]
fp = 'paper_mcmc.h5'
ntemps = 3
nwalkers = 10
nsteps = 500
emri_kwargs = {"pad_output": True, "return_list": True, "dt": dt, "T": Tobs, "eps":1e-3}
initialize_state = True

priors_kerr_intrinsic = {
    "emri": ProbDistContainer(
        {
            0: uniform_dist(np.log(1e4), np.log(5e7)),  # logM
            1: uniform_dist(np.log(1e3), np.log(5e7)),  # mu
            2: uniform_dist(0., 0.999),  # a
            3: uniform_dist(9.0, 16.0),  # p0
            4: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
        }
    ) 
}

# N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
# Tobs = (N_obs * dt) / YRSID_SI

# few_gen = GenerateEMRIWaveform(
#     "FastSchwarzschildEccentricFlux", 
#     sum_kwargs=dict(pad_output=True),
#     use_gpu=use_gpu,
#     return_list=False
# )
emri_injection_params = emri_injection_params_temp.copy()

wave_gen = KerrWaveform()

# for transforms
# this is an example of how you would fill parameters 
# if you want to keep them fixed
# (you need to remove them from the other parts of initialization)
fill_dict = {
    "ndim_full": 14,
    #"fill_values": np.array([0.0, 1.0, dist, qK, phiK, qS, phiS, 0.0, 0.0]), # spin and inclination and Phi_theta
    "fill_values": np.array([0.0, 1.0, dist, qS, phiS, qK, phiK, 0.0, 0.0]), # spin and inclination and Phi_theta
    "fill_inds": np.array([4, 5, 6, 7, 8, 9, 10, 12, 13]),
}

# N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
# Tobs = (N_obs * dt) / YRSID_SI

# few_gen = GenerateEMRIWaveform(
#     "FastSchwarzschildEccentricFlux", 
#     sum_kwargs=dict(pad_output=True),
#     use_gpu=use_gpu,
#     return_list=False
# )
emri_injection_params = emri_injection_params_temp.copy()

wave_gen = KerrWaveform()

# for transforms
# this is an example of how you would fill parameters 
# if you want to keep them fixed
# (you need to remove them from the other parts of initialization)
fill_dict = {
    "ndim_full": 14,
    #"fill_values": np.array([0.0, 1.0, dist, qK, phiK, qS, phiS, 0.0, 0.0]), # spin and inclination and Phi_theta
    "fill_values": np.array([0.0, 1.0, dist, qS, phiS, qK, phiK, 0.0, 0.0]), # spin and inclination and Phi_theta
    "fill_inds": np.array([4, 5, 6, 7, 8, 9, 10, 12, 13]),
}

# following code adapted from scripts authored by Lorenzo Speri and Michael Katz
emri_injection_params[0] = np.log(emri_injection_params[0])
emri_injection_params[1] = np.log(emri_injection_params[1])
emri_injection_params[7] = np.cos(emri_injection_params[7]) 
emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
emri_injection_params[9] = np.cos(emri_injection_params[9]) 
emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)

# phases
emri_injection_params[-1] = emri_injection_params[-1] % (2 * np.pi)
emri_injection_params[-2] = emri_injection_params[-2] % (2 * np.pi)
emri_injection_params[-3] = emri_injection_params[-3] % (2 * np.pi)

# remove three we are not sampling from (need to change if you go to adding spin)
emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])

parameter_transforms = {
    0: np.exp,  # M 
    1: np.exp,  # mu 
}

transform_fn = TransformContainer(
    parameter_transforms=parameter_transforms,
    fill_dict=fill_dict,
)

# sampler treats periodic variables by wrapping them properly
periodic = {
    "emri": {4: 2 * np.pi}
}

# get injected parameters after transformation
injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]

# get XYZ
data_channels = wave_gen(*injection_in, **emri_kwargs)

# this is a parent likelihood class that manages the parameter transforms
like = Likelihood(
    wave_gen,
    2,  # of channels
    dt=dt,
    parameter_transforms={"emri": transform_fn},
    use_gpu=use_gpu,
    vectorized=False,
    transpose_params=False,
    subset=1,  # may need this subset
)

nchannels = 2
like.inject_signal(
    data_stream=[data_channels[0], data_channels[1]],
    noise_fn=get_sensitivity,
    noise_kwargs=[{"sens_fn": "cornish_lisa_psd"} for _ in range(nchannels)],
    noise_args=[[] for _ in range(nchannels)],
)

ndim = len(emri_injection_params_in)

# MCMC moves (move, percentage of draws)
moves = [
    StretchMove()
]

# prepare sampler
sampler = EnsembleSampler(
    nwalkers,
    [ndim],  # assumes ndim_max
    like,
    priors_kerr_intrinsic,
    tempering_kwargs={"ntemps": ntemps},
    moves=moves,
    kwargs=emri_kwargs,
    backend=fp,
    vectorize=True,
    periodic=periodic,  # TODO: add periodic to proposals
    branch_names=["emri"],
)

if initialize_state:
    # generate starting points
    factor = 1e-5
    cov = np.ones(ndim) * 1e-3
    cov[0] = 1e-3

    start_like = np.zeros((nwalkers * ntemps))

    iter_check = 0
    max_iter = 1000

    while np.std(start_like) < 30.0:    
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps * nwalkers, ndim))
        fix = np.ones((ntemps * nwalkers), dtype=bool)
        while np.any(fix):
            tmp[fix] = (emri_injection_params_in[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndim)))[fix]

            # phases
            emri_injection_params_in[-1] = emri_injection_params_in[-1] % (2 * np.pi)

            logp = priors_kerr_intrinsic["emri"].logpdf(tmp)

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        # like.injection_channels[:] = 0.0
        start_like = like(tmp, **emri_kwargs)
    
        iter_check += 1
        factor *= 1.5

        print("Standard Deviation of starting likelihood")
        print(np.std(start_like))

        if iter_check > max_iter:
            raise ValueError("Unable to find starting parameters.")

    start_params = tmp.copy()
    print(start_params)

    start_prior = priors_kerr_intrinsic["emri"].logpdf(start_params)

    # start state
    start_state = State(
        {"emri": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
        log_like=start_like.reshape(ntemps, nwalkers), 
        log_prior=start_prior.reshape(ntemps, nwalkers)
    )
else:
    start_state = None

nsteps = 100000
out = sampler.run_mcmc(None, nsteps, progress=True, thin_by=1, burn=0)

samples = sampler.get_chain(discard=500, thin=10)["emri"][:, 0].reshape(-1, ndim)

np.save(pathname + "/../data/mcmc_samples_data.npy")