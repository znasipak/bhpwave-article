from pybhpt.flux import FluxMode
from pybhpt.geo import KerrGeodesic
from pybhpt.teuk import TeukolskyMode
import numpy as np

OMEGA_MIN = 2.e-3
A_MAX = 0.9999

def convergenceTestModes(fluxMode, fluxTot, fluxComp, fluxComp2, tol = 1e-10, flux_position=[0, 1, 2]):
    if np.all(np.abs(np.array(fluxMode))[flux_position]/np.abs(np.array(fluxTot)[flux_position]) < tol) and np.all(np.abs(np.array(fluxMode)[flux_position]) < np.abs(np.array(fluxComp)[flux_position])) and np.all(np.abs(np.array(fluxMode)[flux_position]) < np.abs(np.array(fluxComp2)[flux_position])):
        return True
    else:
        return False
    
class FluxSummation:
    def __init__(self, a, p, e, x, nsamples0 = 2**7):
        self.a = a
        self.p = p
        self.e = e
        self.x = x
        self.nsamples0 = nsamples0
        self.nsamples = nsamples0
        self.geo = KerrGeodesic(self.a, self.p, self.e, self.x, self.nsamples)
        self.nlimit = 100
        self.mlimit = 100
        self.llimit = 100
        self.fluxref = 2*np.array([32./5.*np.abs(self.geo.frequencies[-1])**(10./3.), 32./5.*np.abs(self.geo.frequencies[-1])**(8./3.), 1.])

        self.totalfluxes = np.zeros(3)
        self.horizonfluxes = np.zeros(3)
        self.infinityfluxes = np.zeros(3)
        self.nfluxes = np.zeros(3)
        self.mfluxes = np.zeros(3)
        self.nmax = 0
        self.nmin = 0
        self.mmaxmodenums = np.zeros((2*self.nlimit + 1, 2), dtype=int)
        self.lmaxmodenums = np.zeros((2*self.nlimit + 1, self.mlimit + 1, 2), dtype=int)
        self.mpeakmodenums = np.zeros((2*self.nlimit + 1), dtype=int)
        self.mpeakmodes = np.zeros((2*self.nlimit + 1, 3))
        self.fluxamplitudes = np.zeros((self.llimit + 1, self.mlimit + 1, 2*self.nlimit + 1), dtype=np.complex128)

        self.n = 0
        self.m = 0
        self.l = 2
        self.flux_positions = [0, 1]

    def check_for_peak_mode(self):
        current_peak_value = self.mpeakmodes[self.n]
        if np.any(np.abs(self.mfluxes) > current_peak_value):
            current_peak_value = np.abs(self.mfluxes)
            self.mpeakmodes[self.n] = current_peak_value
            self.mpeakmodenums[self.n] = self.m

    def sum(self, nmin_include = 3, print_output_n_sum = False, print_output_m_sum = False):
        self.total_mode_summation(nmin_include = nmin_include, print_output_n_sum = print_output_n_sum, print_output_m_sum = print_output_m_sum)

    def total_mode_summation(self, nmin_include = 3, print_output_n_sum = False, print_output_m_sum = False):
        if self.e == 0:
            self.nlimit = 0
            self.n_mode_summation_circ(print_output_n_sum = print_output_n_sum, print_output_m_sum = print_output_m_sum)
        else:
            self.n_mode_summation(0, nmin_include = nmin_include, increment = 1, print_output_n_sum = print_output_n_sum, print_output_m_sum = print_output_m_sum)
            self.mmaxmodenums[self.n][1] = self.n - 1
            self.n_mode_summation(-1, nmin_include = nmin_include, increment = -1, print_output_n_sum = print_output_n_sum, print_output_m_sum = print_output_m_sum)
            self.mmaxmodenums[self.n][0] = self.n + 1

    def n_mode_summation_circ(self, print_output_n_sum = False, print_output_m_sum = False):
        ninit = 0
        include_zero = False
        mpeak = abs(ninit + 1)
        self.n = ninit

        if abs(8*self.n) > self.nsamples:
            self.nsamples *= 2
            self.geo = KerrGeodesic(self.a, self.p, self.e, self.x, nsamples = self.nsamples)

        self.nfluxes = np.zeros(3)
        self.m_mode_summation(mpeak, increment = 1, include_zero = include_zero, print_output = print_output_m_sum)
        mmin = 1
        if include_zero:
            mmin = 0
        self.mmaxmodenums[self.n] = [mmin, self.m - 1] # record the max m-modes used for the sum
        self.m_mode_summation(mpeak - 1, increment = -1, include_zero = include_zero, print_output = print_output_m_sum)

        if print_output_n_sum:
            print(self.mmaxmodenums[self.n][1], self.n, self.nfluxes)
        
        mpeak = self.mpeakmodenums[self.n]

    def n_mode_summation(self, ninit, nmin_include = 3, increment = 1, print_output_n_sum = False, print_output_m_sum = False):
        include_zero = True
        if increment > 0:
            include_zero = False
        
        convergedN = False
        self.n = ninit
        mpeak = abs(ninit + 1)
        previous_nfluxes = np.zeros(3)
        previous_nfluxes_2 = np.zeros(3)

        while (abs(self.n) < self.nlimit and convergedN is False) or (abs(nmin_include) < 3):
            if abs(8*self.n) > self.nsamples:
                self.nsamples *= 2
                self.geo = KerrGeodesic(self.a, self.p, self.e, self.x, nsamples = self.nsamples)

            self.nfluxes = np.zeros(3)
            self.m_mode_summation(mpeak, increment = 1, include_zero = include_zero, print_output = print_output_m_sum)
            mmin = 1
            if include_zero:
                mmin = 0
            self.mmaxmodenums[self.n] = [mmin, self.m - 1] # record the max m-modes used for the sum
            self.m_mode_summation(mpeak - 1, increment = -1, include_zero = include_zero, print_output = print_output_m_sum)

            convergedN = convergenceTestModes(self.nfluxes, self.fluxref, previous_nfluxes, previous_nfluxes_2, flux_position=self.flux_positions)
            previous_nfluxes_2 = previous_nfluxes
            previous_nfluxes = self.nfluxes
            if print_output_n_sum:
                print(self.mmaxmodenums[self.n][1], self.n, self.nfluxes)
            
            mpeak = self.mpeakmodenums[self.n]
            self.n += increment

    def m_mode_summation(self, minit, increment = 1, include_zero = False, print_output = False):
        # increment sets how much to change m. Usually set to +1 or -1
        mmin = 1
        if include_zero:
            mmin = 0
        convergedM = False
        self.m = minit
        previous_mfluxes = np.zeros(3)
        previous_mfluxes_2 = np.zeros(3)
        while (self.m < self.mlimit and self.m >= mmin  and convergedM is False):
            # increase sampling rate for high m-modes
            if 8*self.m > self.nsamples:
                self.nsamples *= 2
                self.geo = KerrGeodesic(self.a, self.p, self.e, self.x, nsamples = self.nsamples)
            # perform sum over l-modes
            if self.m == 0:
                self.flux_positions = 0 # we ignore angular momentum for m = 0 because it vanishes
            else:
                self.flux_positions = [0, 1]
            self.mfluxes = np.zeros(3)
            self.l_mode_summation()
            self.check_for_peak_mode()

            # add this m-mode contribution to n-mode and test for convergence
            self.nfluxes += self.mfluxes
            convergedM = convergenceTestModes(self.mfluxes, self.fluxref, previous_mfluxes, previous_mfluxes_2, flux_position=self.flux_positions)
            previous_mfluxes_2 = previous_mfluxes
            previous_mfluxes = self.mfluxes

            if print_output:
                print(self.l - 1, self.m, self.n, self.mfluxes)
            self.m += increment

    def l_mode_summation(self, precision_tolerance = 1.e-3):
        # initialize convergence criteria to False
        convergedL = False
        previous_lfluxes = np.zeros(3)

        # find minimum l-mode
        lmin = np.max([self.m, 2])
        self.l = lmin

        # sum over l-modes until we hit the l-limit or until the convergence test returns True
        while self.l <= self.llimit and convergedL is False:
            # calculate Teukolsky mode
            teuk = TeukolskyMode(-2, self.l, self.m, 0, self.n, self.geo)
            flux = FluxMode(self.geo, teuk) # extract fluxes from the Teukolsky mode

            if teuk.precision('Up') > precision_tolerance and teuk.precision('In') > precision_tolerance:
                geo_temp = KerrGeodesic(self.a, self.p, self.e, self.x, nsamples = 2*self.nsamples)
                teuk_temp = TeukolskyMode(-2, self.l, self.m, 0, self.n, geo_temp)
                flux_temp = FluxMode(geo_temp, teuk_temp)
                if teuk.precision('Up') < precision_tolerance or teuk.precision('In') < precision_tolerance:
                    self.nsamples *= 2
                    self.geo = geo_temp
                    teuk = teuk_temp
                    flux = flux_temp

            # if the Teukolsky calculation did not lose too much precision, continue with calculation
            # horizon = 0.
            # infinity = 0.
            totalref = 0.*np.array(flux.totalfluxes) 
            if teuk.precision('Up') < precision_tolerance or teuk.precision('In') < precision_tolerance:
                self.mfluxes += flux.totalfluxes # add to m-mode test
                self.totalfluxes += flux.totalfluxes # add to total fluxes
                totalref = flux.totalfluxes 
            
            if teuk.precision('Up') < precision_tolerance:
                self.infinityfluxes += flux.infinityfluxes
                self.fluxamplitudes[self.l, self.m, self.n] = 0.5*teuk.amplitude('Up')/np.abs(teuk.frequency)**2 # store waveform amplitude for later use

            if teuk.precision('In') < precision_tolerance:
                self.horizonfluxes += flux.horizonfluxes
            
            # test for convergence
            convergedL = convergenceTestModes(totalref, self.fluxref, previous_lfluxes, previous_lfluxes, flux_position=self.flux_positions)
            previous_lfluxes = totalref # store this l-mode to compare to the next l-mode
            self.l += 1

        # record the min and max l-modes used for the sum
        self.lmaxmodenums[self.n, self.m] = [lmin, self.l]

class WaveAmplitude:
    def __init__(self, a, p, e, x, nsamples0 = 2**7):
        self.a = a
        self.p = p
        self.e = e
        self.x = x
        self.nsamples0 = nsamples0
        self.nsamples = nsamples0
        self.geo = KerrGeodesic(self.a, self.p, self.e, self.x, self.nsamples)
        self.llimit = 100

        self.n = 0
        self.m = 0
        self.l = 2

    def spheroidal_amplitude(self, j, m, n):
        teuk = TeukolskyMode(-2, j, m, 0, n, self.geo)
        teuk.solve(self.geo)
        return 0.5*teuk.amplitude('Up')/np.abs(teuk.frequency)**2
    
    def spherical_spheroidal_amplitude(self, l, j, m, n):
        teuk = TeukolskyMode(-2, j, m, 0, n, self.geo)
        teuk.solve(self.geo)
        return teuk.couplingcoefficient(l)*0.5*teuk.amplitude('Up')/np.abs(teuk.frequency)**2
    
    def spherical_spheroidal_amplitude(self, j, m, n, lmax = None):
        teuk = TeukolskyMode(-2, j, m, 0, n, self.geo)
        teuk.solve(self.geo)
        if lmax is None:
            lmax = teuk.maxcouplingmode
        return -2.*np.array([teuk.couplingcoefficient(l)*teuk.amplitude('Up')/np.abs(teuk.frequency)**2 for l in range(teuk.mincouplingmode, lmax + 1)])
    
    def spherical_amplitudes(self, lmax, m, n):
        jmin = np.max([2, np.abs(m)])
        amp_tot = 0.j
        j = jmin
        amp_return = np.zeros(lmax - 1, dtype=np.complex128)

        amp = self.spherical_spheroidal_amplitude(j, m, n, lmax = lmax)
        amp += self.spherical_spheroidal_amplitude(j + 1, m, n, lmax = lmax)
        amp_tot += amp
        ref_amp = np.abs(amp_tot)
        ref_amp[ref_amp == 0] = 1.e-16
        j += 2

        while j < self.llimit and np.any(np.abs(amp)/ref_amp > 1e-8):
            amp = self.spherical_spheroidal_amplitude(j, m, n, lmax = lmax)
            amp += self.spherical_spheroidal_amplitude(j + 1, m, n, lmax = lmax)
            amp_tot += amp
            ref_amp = np.abs(amp_tot)
            ref_amp[ref_amp == 0] = 1.e-16
            j += 2

        amp_return[jmin - 2:] = amp_tot
        return amp_return

def kerr_circ_geo_radius(a, omega):
    """
    Calculates the Boyer-Lindquist radius of a circular geodesic with orbital
    frequency `omega` in a Kerr spacetime paramtrized by the Kerr spin `a`

    :param a: Kerr spin parameter
    :type a: double or array
    :param omega: orbital frequency
    :type omega: double or array

    :rtype: double or array
    """
    return (abs(omega)*(1. - a*omega)/(omega**2))**(2./3.)

def kerr_circ_geo_orbital_frequency(a, r):
    """
    Calculates the orbital frequency of a circular geodesic with Boyer-Lindquist radius
    `r` in a Kerr spacetime paramtrized by the Kerr spin `a`

    :param a: Kerr spin parameter
    :type a: double or array
    :param r: orbital radius
    :type r: double or array

    :rtype: double or array
    """
    v = 1./np.sqrt(r)
    return pow(v, 3)/(1 + a*pow(v, 3))

def kerr_isco_radius(a):
    """
    Calculates the Boyer-Lindquist radius of the innermost stable circular orbit (ISCO)
    in a Kerr spacetime paramtrized by the Kerr spin `a`

    :param a: Kerr spin parameter
    :type a: double or array

    :rtype: double or array
    """
    sgnX = np.sign(a)
    z1 = 1 + pow(1 - a*a, 1./3.)*(pow(1 - a, 1./3.) + pow(1 + a, 1./3.))
    z2 = np.sqrt(3*a*a + z1*z1)

    return 3 + z2 - sgnX*np.sqrt((3. - z1)*(3. + z1 + 2.*z2))

def kerr_isco_frequency(a):
    """
    Calculates the orbital frequency of the innermost stable circular orbit (ISCO)
    in a Kerr spacetime paramtrized by the Kerr spin `a`

    :param a: Kerr spin parameter
    :type a: double or array

    :rtype: double or array
    """
    rISCO = kerr_isco_radius(a)
    return kerr_circ_geo_orbital_frequency(a, rISCO)

def alpha_of_a_omega(a, omega):
    oISCO = kerr_isco_frequency(a)
    return alpha_of_omega_ISCO(omega, oISCO)

def alpha_of_omega_ISCO(omega, oISCO):
    return (abs(oISCO**(1./3.) - omega**(1./3.))/(oISCO**(1./3.) - OMEGA_MIN**(1./3.)))**(0.5)

def omega_of_a_alpha(a, alpha):
    oISCO = kerr_isco_frequency(a)
    return omega_of_alpha_ISCO(alpha, oISCO)

def omega_of_alpha_ISCO(alpha, oISCO):
    return pow(pow(oISCO, 1./3.) - pow(alpha, 2.)*(pow(oISCO, 1./3.) - pow(OMEGA_MIN, 1./3.)), 3.)

def chi_of_spin_subfunc(a):
    return pow(1. - a, 1./3.)

def chi_of_spin(a):
    return pow((chi_of_spin_subfunc(a) - chi_of_spin_subfunc(A_MAX))/(chi_of_spin_subfunc(-A_MAX) - chi_of_spin_subfunc(A_MAX)), 0.5)

def spin_of_chi(chi):
    return 1. - pow(chi_of_spin_subfunc(A_MAX) + pow(chi, 2.)*(chi_of_spin_subfunc(-A_MAX) - chi_of_spin_subfunc(A_MAX)), 3.)

def a_omega_to_chi_alpha(a, omega):
    chi = chi_of_spin(a)
    alpha = alpha_of_a_omega(a, omega)
    return (chi, alpha)

def omega_alpha_derivative(omega, oISCO):
    if abs(oISCO - omega) < 1.e-13: 
        return 0.
    return -6.*pow((pow(oISCO, 1./3.) - pow(OMEGA_MIN, 1./3.))*(pow(oISCO, 1./3.) - pow(omega, 1./3.)), 0.5)*pow(omega, 2./3.)

def energy_omega_derivative(a, omega):
    r = kerr_circ_geo_radius(a, omega)
    return energy_r_derivative(a, r)*r_omega_derivative(a, omega)

def energy_r_derivative(a, r):
    v = 1./np.sqrt(r)
    return 0.5*(pow(v, 4) - 6.*pow(v, 6) + 8.*a*pow(v, 7) - 3.*a*a*pow(v, 8))/pow(1. + v*v*(2.*a*v - 3.), 1.5)

def r_omega_derivative(a, omega):
    return -2./(3.*pow(omega, 5./3.)*pow(1. - a*omega, 1./3.))