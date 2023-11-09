from bhpwave.waveform import KerrWaveform, source_angles, polarization
from bhpwave.trajectory.inspiral import TrajectoryData, InspiralGenerator
from bhpwave.trajectory.geodesic import kerr_isco_radius
from lisatools.diagnostic import inner_product
import numpy as np
import pickle
import os

from scipy.optimize import root_scalar
from bhpwave.harmonics.swsh import Yslm
from scipy.signal import savgol_filter
import mpmath as mp

# user specifies which [M, a] combination they want from 1 to 14
# one can also design this as a large for-loop but these computations
# can be expensive and prone to numerical instabilities. Therefore it
# might be easier to run several scripts in parallel to test different
# cases
analysis_number = 1

M_array = [1e6, 1e5]
a_array = [0.9995, 0.99, 0.9, 0.5, 0., -0.5, -0.99]

Ma_dict = {}
i = 1
for M in M_array:
    for a in a_array:
        Ma_dict[i] = [M, a]
        i += 1
imax = i

data_path = "../../data/saved_flux_analysis.pkl"

if os.path.exists(data_path):
    with open(data_path, 'rb') as f:
        flux_analysis = pickle.load(f)
else:
    flux_analysis = {}
YR_CONV=31558149.763545603

traj=TrajectoryData(file_path=os.path.abspath("../tests/trajectory_error.txt"))

def radius_of_time_to_merger_eqn(r, M, mu, a, T):
    return (traj.time_to_merger(M, mu, a, r)/YR_CONV - T)

def radius_of_time_to_merger(M, mu, a, T):
    return root_scalar(radius_of_time_to_merger_eqn, args=(M, mu, a, T), method="brentq", bracket=[1.0001*kerr_isco_radius(a), 50]).root

kerrwave = KerrWaveform(trajectory_data=traj)
kerrwave_GR = KerrWaveform()

# # parameters
dt = 10.0  # seconds
T = 2.0  # years
mu = 1e1
p0 = radius_of_time_to_merger(M, mu, a, T)
e0 = 0.0
x0 = 1.0  # will be ignored in Schwarzschild waveform
qK = 1.8  # polar spin angle
phiK = 1.2  # azimuthal viewing angle
qS = 0.3  # polar sky angle
phiS = 1.3  # azimuthal viewing angle
dist = 1.  # distance
Phi_phi0 = 0.2
Phi_theta0 = 0.0
Phi_r0 = 0.0

M, a = Ma_dict[analysis_number]

injection_params = np.array([M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,dt,T])
waveform_kwargs={
    "num_threads": 16,
    "pad_output": True,
}
waveform_kwargs_full=waveform_kwargs.copy()

modes = kerrwave.select_modes(M, mu, a, p0, qS, phiS, qK, phiK, Phi_phi0, dt=dt, T=T, eps = 1.e-3)
modesN = [(mode[0], mode[1], 0) for mode in modes]
waveform_kwargs_full["select_modes"]=modes

htest = kerrwave(*injection_params, **waveform_kwargs_full)
hGR = kerrwave_GR(*injection_params, **waveform_kwargs_full)

from scipy.signal import windows
window = windows.tukey(htest.shape[0], alpha=0.001)

# rescale distance to get an SNR of 30
snr_scale = 30
snr_test_2 =  np.sqrt(inner_product([window*htest.real, -window*htest.imag], [window*htest.real, -window*htest.imag], dt=dt, PSD="cornish_lisa_psd"))
snr_test =  np.sqrt(inner_product([window*hGR.real, -window*hGR.imag], [window*hGR.real, -window*hGR.imag], dt=dt, PSD="cornish_lisa_psd"))
dist *= snr_test/snr_scale
T = htest[np.abs(hGR)>0.].shape[0]*dt/YR_CONV

injection_params = np.array([M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,dt,T])
waveform_kwargs={
    "num_threads": 16,
    "pad_output": True,
}
waveform_kwargs_full=waveform_kwargs.copy()
modes = kerrwave.select_modes(M, mu, a, p0, qS, phiS, qK, phiK, Phi_phi0, dt=dt, T=T, eps = 1.e-3)
modesN = [(mode[0], mode[1], 0) for mode in modes]

waveform_kwargs_full["select_modes"]=modes

htest=kerrwave(*injection_params, **waveform_kwargs_full)
hGR = kerrwave_GR(*injection_params, **waveform_kwargs_full)

insp_GR = InspiralGenerator()
sol_GR=insp_GR(M, mu, a, p0, dt, T)
insp = InspiralGenerator(trajectory_data=traj)
sol=insp(M, mu, a, p0, dt, T)
dephase = np.max(2*np.abs(sol.phase[:sol_GR.phase.shape[0]]-sol_GR.phase[:sol.phase.shape[0]]))

if analysis_number in flux_analysis.keys():
    flux_analysis[analysis_number]["modes"] = modes
    flux_analysis[analysis_number]["htest"] = htest
    flux_analysis[analysis_number]["hGR"] = hGR
    flux_analysis[analysis_number]["dephase"] = dephase
else:
    flux_analysis[analysis_number] = {"params": injection_params,
        "kwargs": waveform_kwargs,
        "kwargs_full": waveform_kwargs_full,
        "modes" : modes,
        "hm": htest,
        "hGR": hGR,
        "dephase": dephase
        }

window = windows.tukey(hGR.shape[0], alpha=0.001)
freq = np.fft.rfftfreq(hGR.shape[0], d=dt)[1:]
hGR_freq = np.fft.rfft([window*hGR.real, -window*hGR.imag])[:,1:]*dt
h_freq = np.fft.rfft([window*htest.real, -window*htest.imag])[:,1:]*dt

# Functions for calculating the derivatives of the waveform either on an (l,m)-mode basis or on the total waveform
## note that this procedure could be much better optimized (e.g, one could perform the sum over l and just decompose onto m-modes,
## we could restructure bhpwave to just output m-modes)
## at the moment it is clunky because bhpwave returns m + (-m) modes, so we have to disentangle these to get better behaved derivatives

def mode_amplitude_phase(l, m, *args, **kwargs):
    mode_polarization = polarization(*args[7:11])
    mode_angle = source_angles(*args[7:11])[0]
    mode_Yplus = (Yslm(-2, l, m, mode_angle, 0.) + Yslm(-2, l, m, np.pi-mode_angle, 0.)).real
    mode_Ycross = (Yslm(-2, l, m, mode_angle, 0.) - Yslm(-2, l, m, np.pi-mode_angle, 0.)).real
    hlm = kerrwave(*args, **kwargs, select_modes=np.array([(l,m)], dtype=np.int32))
    hlmNorm = hlm/mode_polarization
    hlmNorm = hlmNorm.real/mode_Yplus + 1j*hlmNorm.imag/mode_Ycross
    Alm = np.abs(hlmNorm)
    Philm = np.unwrap(np.angle(hlmNorm))
    # print((mode_polarization*Alm*(mode_Yplus*np.cos(Philm) + 1j*mode_Ycross*np.sin(Philm)))[:10])
    # print(hlm[:10])
    return (Alm, Philm, mode_Yplus, mode_Ycross, mode_polarization)

def mode_amplitude_phase_diff(l, m, *args, **kwargs):
    hlm = kerrwave(*args, **kwargs, select_modes=np.array([(l,m)], dtype=np.int32))
    try:
        hlmGR = kerrwave_GR(*args, **kwargs, select_modes=np.array([(l,m)], dtype=np.int32))
    except:
        hlmGR = kerrwave_GR(*args[:-2], dt = args[-2], T = args[-1], mode_selection=[(l,m,0)], include_minus_m=True)
    mode_polarization = polarization(*args[7:11])
    mode_angle = source_angles(*args[7:11])[0]
    mode_Yplus = (Yslm(-2, l, m, mode_angle, 0.) + Yslm(-2, l, m, np.pi-mode_angle, 0.)).real
    mode_Ycross = (Yslm(-2, l, m, mode_angle, 0.) - Yslm(-2, l, m, np.pi-mode_angle, 0.)).real
    
    hlmNorm = hlm/mode_polarization
    hlmNorm = hlmNorm.real/mode_Yplus + 1j*hlmNorm.imag/mode_Ycross
    Alm = np.abs(hlmNorm)
    Philm = np.unwrap(np.angle(hlmNorm))

    hlmGRNorm = hlmGR/mode_polarization
    hlmGRNorm = hlmGRNorm.real/mode_Yplus + 1j*hlmGRNorm.imag/mode_Ycross
    AlmGR = np.abs(hlmGRNorm)
    PhilmGR = np.unwrap(np.angle(hlmGRNorm))
    return 0.5*mode_polarization*(mode_Yplus + mode_Ycross)*((AlmGR - Alm) + 1j*Alm*(PhilmGR - Philm))*np.exp(1j*Philm) + 0.5*mode_polarization*(mode_Yplus - mode_Ycross)*((AlmGR - Alm) - 1j*Alm*(PhilmGR - Philm))*np.exp(-1j*Philm)
    # return AlmGR*np.exp(1j*PhilmGR) - Alm*np.exp(1j*Philm)

def waveform_diff(*args, **kwargs):
    mode = modes[0]
    hlmDiff = mode_amplitude_phase_diff(mode[0], mode[1], *args, **kwargs)
    for mode in modes[1:]:
        hlmDiff += mode_amplitude_phase_diff(mode[0], mode[1], *args, **kwargs)
    return hlmDiff

def mode_amplitude_phase_deriv(deriv_ind, l, m, eps, *args, **kwargs):
    args=list(args)
    args_copy = args.copy()
    # Alm, Philm = mode_amplitude_phase(l, m, *args_copy, **kwargs)
    if args[deriv_ind] != 0.:
        delta = eps*args[deriv_ind]
    else:
        delta = eps
    
    Alm_eps = {}
    Phi_eps = {}
    Yplus_eps = {}
    Ycross_eps = {}
    P_eps = {}

    if deriv_ind == 11: # dPhi0
        i = 0
        Alm_eps[i], Phi_eps[i], Yplus_eps[i], Ycross_eps[i], P_eps[i] = mode_amplitude_phase(l, m, *args_copy, **kwargs)
        dAlm = 0.*Alm_eps[0]
        mode_dYplus = 0.
        mode_dYcross = 0.
        mode_dP = 0.
        dPhilm = -m*(1. + dAlm)
        return (Alm_eps[0], Phi_eps[0], Yplus_eps[0], Ycross_eps[0], P_eps[0], dAlm, dPhilm, mode_dYplus, mode_dYcross, mode_dP)
    elif deriv_ind == 6:
        i = 0
        Alm_eps[i], Phi_eps[i], Yplus_eps[i], Ycross_eps[i], P_eps[i] = mode_amplitude_phase(l, m, *args_copy, **kwargs)
        dAlm = -Alm_eps[0]/args[deriv_ind]
        mode_dYplus = 0.
        mode_dYcross = 0.
        mode_dP = 0.
        dPhilm = 0.*Phi_eps[0]
    else:
        for i in range(-1, 2):
            args_copy[deriv_ind] = args[deriv_ind] + i*delta
            Alm_eps[i], Phi_eps[i], Yplus_eps[i], Ycross_eps[i], P_eps[i] = mode_amplitude_phase(l, m, *args_copy, **kwargs)
        
        if deriv_ind in [7, 8, 9, 10]:
            dAlm = 0.*Alm_eps[0]
            dPhilm = 0.*Phi_eps[0]
        else:
            dAlm = (Alm_eps[1] - Alm_eps[-1])/(2.*delta)
            dPhilm = (Phi_eps[1] - Phi_eps[-1])/(2.*delta)
            nn = int(len(dAlm)/1000)
            dAlm = np.concatenate((dAlm[:nn], savgol_filter(dAlm[nn:], 50, 3)))
            dPhilm = np.concatenate((dPhilm[:nn], savgol_filter(dPhilm[nn:], 50, 3)))

        if np.all(np.abs(np.array([Ycross_eps[1], Ycross_eps[-1]]) - Ycross_eps[1]) == 0.):
            mode_dYplus = 0.
        else:
            mode_dYplus = (Yplus_eps[1] - Yplus_eps[-1])/(2.*delta)
        if np.all(np.abs(np.array([Ycross_eps[1], Ycross_eps[-1]]) - Ycross_eps[1]) == 0.):
            mode_dYcross = 0.
        else:
            mode_dYcross = (Ycross_eps[1] - Ycross_eps[-1])/(2.*delta)
        if np.all(np.abs(np.array([P_eps[1], P_eps[-1]]) - P_eps[1]) == 0.):
            mode_dP = 0.
        else:
            mode_dP = (P_eps[1] - P_eps[-1])/(2.*delta)

    return (Alm_eps[0], Phi_eps[0], Yplus_eps[0], Ycross_eps[0], P_eps[0], dAlm, dPhilm, mode_dYplus, mode_dYcross, mode_dP)

def mode_deriv(deriv_ind, l, m, eps, *args, **kwargs):
    Alm, Philm, mode_Yplus, mode_Ycross, mode_P, dAlm, dPhilm, mode_dYplus, mode_dYcross, mode_dP = mode_amplitude_phase_deriv(deriv_ind, l, m, eps, *args, **kwargs)
    mode_phase_0 = (mode_Yplus*np.cos(Philm) + 1j*mode_Ycross*np.sin(Philm))
    mode_phase_1 = -(mode_Yplus*np.sin(Philm) - 1j*mode_Ycross*np.cos(Philm))
    dmode = mode_P*(Alm*dPhilm*mode_phase_1 + dAlm*mode_phase_0)
    if np.abs(mode_dYplus) > 1.e-14 or np.abs(mode_dYcross) > 1.e-14:
        mode_phase_2 = (mode_dYplus*np.cos(Philm) + 1j*mode_dYcross*np.sin(Philm))
        dmode += mode_P*Alm*mode_phase_2
    if np.abs(mode_dP) > 1.e-14:
        dmode += mode_dP*Alm*mode_phase_0
    return dmode

def mode_deriv_full(deriv_ind, select_modes, eps, *args, **kwargs):
    args=list(args)
    args_copy = args.copy()
    if args[deriv_ind] != 0.:
        delta = eps*args[deriv_ind]
    else:
        delta = eps
    if deriv_ind == 6: # dist
        dhlm = -kerrwave(*args_copy, **kwargs, select_modes=select_modes)/args_copy[deriv_ind]
    else:
        args_copy[deriv_ind] = args[deriv_ind] + delta
        hlm_plus_eps = kerrwave(*args_copy, **kwargs, select_modes=select_modes)
        args_copy[deriv_ind] = args[deriv_ind]- delta
        hlm_minus_eps = kerrwave(*args_copy, **kwargs, select_modes=select_modes)
        args_copy[deriv_ind] = args[deriv_ind] + 2.*delta
        hlm_plus_2eps = kerrwave(*args_copy, **kwargs, select_modes=select_modes)
        args_copy[deriv_ind] = args[deriv_ind] - 2.*delta
        hlm_minus_2eps = kerrwave(*args_copy, **kwargs, select_modes=select_modes)
        dhlm = (-hlm_plus_2eps + 8.*hlm_plus_eps - 8.*hlm_minus_eps + hlm_minus_2eps)/(12.*delta)
    
    return dhlm

# Functions for taking the (psuedo)inverse of the Fisher matrix

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
    
    mp.mp.dps = 300
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
    # print(error_array)
    # print([np.sqrt(np.diag(pinv)) for pinv in pinv_array[error_array < 10]])

    min_error = np.min(error_array)
    mat_pinv = pinv_array[error_array == min_error][0]

    return mat_pinv

dhs = []
deriv_inds = [0, 1, 2, 3, 11, 6, 7, 8, 9, 10]
deriv_inds_intrinsic = [0, 1, 2, 3, 11]
deriv_inds_extrinsic = [6, 7, 8, 9, 10]
# deriv_inds = [0, 1, 3, 11, 6, 7, 8, 9, 10]
# deriv_inds_intrinsic = [0, 1, 3, 11]
for deriv_ind in deriv_inds:
    print(deriv_ind)
    if deriv_ind in deriv_inds_intrinsic:
        mode = modes[0]
        epsilon = 1.e-6
        hlm = mode_deriv(deriv_ind, mode[0], mode[1], epsilon, *injection_params, **waveform_kwargs)
        for mode in modes[1:]:
            hlm += mode_deriv(deriv_ind, mode[0], mode[1], epsilon, *injection_params, **waveform_kwargs)
    else:
        epsilon = 1.e-6
        hlm = mode_deriv_full(deriv_ind, modes, epsilon, *injection_params, **waveform_kwargs)
    dhs.append(hlm)
dhs = np.array(dhs)


# These are just in case the epsilon values specified above do not lead to sensible numerical derivatives
recalc_intrinsic = False
if recalc_intrinsic:
    for deriv_ind in deriv_inds_intrinsic:
        print(deriv_ind)
        mode = modes[0]
        epsilon = 1.e-4
        hlm = mode_deriv(deriv_ind, mode[0], mode[1], epsilon, *injection_params, **waveform_kwargs)
        for mode in modes[1:]:
            hlm += mode_deriv(deriv_ind, mode[0], mode[1], epsilon, *injection_params, **waveform_kwargs)
        dhs[np.array(deriv_inds)==deriv_ind] = hlm

recalc_extrinsic = False
if recalc_extrinsic:
    for deriv_ind in deriv_inds:
        if deriv_ind not in deriv_inds_intrinsic:
            print(deriv_ind)
            epsilon = 1.e-7
            hlm = mode_deriv_full(deriv_ind, modes, epsilon, *injection_params, **waveform_kwargs)
            dhs[np.array(deriv_inds)==deriv_ind] = hlm

dfs = []
for dh in dhs:
    window = windows.tukey(dh.shape[0], alpha=0.001)
    dfs.append(np.fft.rfft([window*dh.real, -window*dh.imag])*dt)
freq = np.fft.rfftfreq(dhs[0].shape[0], d = dt)

rescale_dfs = True
if rescale_dfs:
    dfs[0] *= injection_params[0]
    dfs[1] *= injection_params[1]

dim = len(dhs)
fish=np.empty((dim,dim))
for i in range(dim):
    print(i)
    for j in range(i, dim):
        if i <= j:
            # throw away some of the high-frequencies due to truncation errors in the DFT
            # we can also downsample the data a bit in the frequency domain to speed-up the calculation without really
            # diminishing the accuracy of the DFT
            fish[i, j] = inner_product(dfs[i][:, 1:-50][:, ::16], dfs[j][:, 1:-50][:, ::16], f_arr=freq[1:-50][::16], PSD="cornish_lisa_psd")
            fish[j, i] = fish[i, j]

# Check that the Fisher matrices and the resulting covariances give self-consistent results
## For EMRIs we expect the intrinsic and extrinsic uncertainties to be fairly independent
## Therefore taking the inverse of the full Fisher matrix should give comparable results 
## to inverting a Fisher matrix for just the intrinsic parameters and just the extrinsic ones
fish_intrinsic = fish[:len(deriv_inds_intrinsic), :len(deriv_inds_intrinsic)]
fish_extrinsic = fish[len(deriv_inds_intrinsic):, len(deriv_inds_intrinsic):]

sigma_full = np.sqrt(np.diag(pinv_solve(fish[:,:])))
sigma_intr = np.sqrt(np.diag(pinv_solve(fish_intrinsic)))
sigma_extr = np.sqrt(np.diag(pinv_solve(fish_extrinsic)))

compare_sigma_intr = np.abs(1 - sigma_full[:len(deriv_inds_intrinsic)]/sigma_intr)
compare_sigma_extr = np.abs(1 - sigma_full[len(deriv_inds_intrinsic):]/sigma_extr)

if compare_sigma_intr > 1.e-1:
    print("Comparison of the intrinsic uncertainties")
    print(sigma_full[:len(deriv_inds_intrinsic)], sigma_intr)

if compare_sigma_extr > 1.e-1:
    print("Comparison of the extrinsic uncertainties")
    print(sigma_full[len(deriv_inds_intrinsic):], sigma_extr)

hdiff = waveform_diff(*injection_params, **waveform_kwargs)
zero_pad_GR = len(hGR[np.abs(hGR) == 0.])
zero_pad_test = len(hGR[np.abs(htest) == 0.])
if zero_pad_GR > zero_pad_test:
    hdiff[np.abs(hGR) == 0.] = -htest[np.abs(hGR) == 0.]
elif zero_pad_GR < zero_pad_test:
    hdiff[np.abs(htest) == 0.] = hGR[np.abs(htest) == 0.]

window = windows.tukey(hdiff.shape[0], alpha=0.001)
hdiff_freq=np.fft.rfft([window*hdiff.real, -window*hdiff.imag])*dt
freq = np.fft.rfftfreq(hdiff.shape[0], d = dt)
bias_ip_vector = np.array([inner_product(dhf[:, 1:], hdiff_freq[:, 1:], f_arr=freq[1:], PSD="cornish_lisa_psd") for dhf in dfs])

inv = pinv_solve(fish[:,:])
biases = np.matmul(inv, bias_ip_vector[:])

biases_full = np.zeros(injection_params.shape[0])
biases_full[deriv_inds[:biases.shape[0]]] = biases.copy()
if rescale_dfs:
    biases_full[0] *= injection_params[0]
    biases_full[1] *= injection_params[1]
injection_params2=injection_params.copy()
injection_params2[deriv_inds] -= biases_full[deriv_inds]
injection_params3=injection_params.copy()
injection_params3[deriv_inds] += biases_full[deriv_inds]

hGR_bias = kerrwave_GR(*injection_params2, **waveform_kwargs_full)
hGR = kerrwave_GR(*injection_params, **waveform_kwargs_full)
htest4 = kerrwave(*injection_params3, **waveform_kwargs_full)

window = windows.tukey(hGR_bias.shape[0], alpha=0.001)
freq = np.fft.rfftfreq(hGR.shape[0], d=dt)[1:]
hGR_bias_freq = np.fft.rfft([window*hGR_bias.real, -window*hGR_bias.imag])[:,1:]*dt

def mismatch_func(h1, h2, dt, freq = None):
    if freq is None:
        freq = np.fft.rfftfreq(h1[0].shape[0], d=dt)[1:]
        h1f = np.fft.rfft(h1)[:,1:]*dt
        h2f = np.fft.rfft(h2)[:,1:]*dt
    else:
        h1f = h1
        h2f = h2
    num = inner_product(h1f, h2f, f_arr=freq, PSD="cornish_lisa_psd")
    snr1 = inner_product(h1f, h1f, f_arr=freq, PSD="cornish_lisa_psd")
    snr0 = inner_product(h2f, h2f, f_arr=freq, PSD="cornish_lisa_psd")
    return 1-num/np.sqrt(snr1*snr0)

Mbias =  mismatch_func(hGR_bias_freq, h_freq, dt=dt, freq=freq)
Mpeak = mismatch_func(hGR_freq, h_freq, dt=dt, freq=freq)

print(f"Mismatch between truth and error model without biases added {Mpeak}")
print(f"Mismatch between truth and error model with biases added {Mbias}")

flux_analysis[analysis_number]["Mbias"] = Mbias
flux_analysis[analysis_number]["Mpeak"] = Mpeak
flux_analysis[analysis_number]["Fisher"] = fish
flux_analysis[analysis_number]["biases"] = biases
flux_analysis[analysis_number]["var"] = np.sqrt(np.diag(inv))
flux_analysis[analysis_number]["R"] = np.max(np.abs(biases)/np.sqrt(np.diag(inv)))

with open('saved_flux_analysis.pkl', 'wb') as f:
    pickle.dump(flux_analysis, f)