"""Short code to compute likelihood functions for stochastic gravitational wave background detection using a mid-band experiment.

Neglect neutron stars because in the LISA band they are just a (small) rescaling of the SGWB amplitude.

LISA-specific SGWB from high redshift supermassive BH mergers, EMRIs or IMBHs
"""
import math
import numpy as np
import emcee
import pint
import scipy.interpolate
import scipy.integrate

ureg = pint.UnitRegistry()

def gelman_rubin(chain):
    """Compute the Gelman-Rubin statistic for a chain"""
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    tb = np.mean(chain, axis=1)
    tbb = np.mean(tb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((tbb - tb)**2, axis=0)
    var_t = (n - 1) / n * W + 1 / n * B
    R = np.sqrt(var_t / W)
    return R

def HubbleEz(zzp1):
    """Dimensionless part of the hubble rate. Argument is 1+z"""
    return np.sqrt(0.3 * zzp1**3 + 0.7)

class Sensitivity:
    """Base class for instrument sensitivities.
       For conventions used see 1408.0740"""
    def __init__(self):
        self.hub = ((67.9 * ureg('km/s/Mpc')).to("1/s").magnitude)
        self.length = 1
        #Equal numbers of samples per frequency decade, so that experiments are equally weighted
        #per unit log(frequency)
        self.nsamples_per_dec = 10

    def omegadens(self):
        """This is the omega_gw that would match
        the noise power spectral density of the detector system."""
        #Root PSD
        ff, psd = self.PSD()
        #This is the sd. on Omega_GW in sqrt(sensitivity per hertz per second)
        #This is the total energy in GW with the PSD given by the design curve.
        #This is not quite the thing plotted in various data papers,
        #which use the one-sided PSDs for two detectors
        #and multiply by a tensor overlap function.
        #It seems, comparing with Figure 3 of 1903.02886 that our approximation
        #over-estimates sensitivity at high frequencies where the overlap function
        #is small, and underestimates it at low frequencies where it is large.
        #Since in any case by the time this forecast happens we will have
        #different detectors with a different and unknown overlap function, ignore this.
        omegagw = 4 * math.pi**2 / 3 / self.hub**2 * ff**3 * psd**2
        return ff, omegagw

    def PSD(self):
        """Root power spectral density in Hz^{-1/2}"""
        raise NotImplementedError

    def downsample(self, newfreq, freq, psd):
        """Down-sample a sensitivity curve to a lower desired number of bins."""
        intp = scipy.interpolate.interp1d(np.log(freq), np.log(psd))
        return np.exp(intp(np.log(newfreq)))

class LISASensitivity(Sensitivity):
    """LISA sensitivity curve as a function of frequency
       from http://www.srl.caltech.edu/cgi-bin/cgiCurve"""
    def __init__(self, wd=False):
        super().__init__()
        if wd:
            self.lisa = np.loadtxt("lisa_sensitivity_curve_wd.dat")
        else:
            self.lisa = np.loadtxt("lisa_sensitivity_curve_nowd.dat")
        # Output Curve type is Root Spectral Density, per root Hz
        # f [Hz]    hf [Hz^(-1/2)]
        #self.lisaintp = scipy.interpolate.interp1d(self.lisa[:,0], self.lisa[:,1])
        #Nominal 4 year LISA mission
        self.length = 4 * 3.154e7

    def PSD(self):
        """Get the root power spectral density in Hz^[-1/2]"""
        return self.lisa[:,0], self.lisa[:,1]

class SatelliteSensitivity(Sensitivity):
    """Satellite sensitivity curve as a function of frequency
       using the formula from https://arxiv.org/abs/gr-qc/9909080
       or eq. 2 of 1512.02076 in Hz^(-1/2)."""
    def __init__(self, satellite = "lisa"):
        super().__init__()
        # Output Curve type is Root Spectral Density, per root Hz
        # f [Hz]    hf [Hz^(-1/2)]
        #self.lisaintp = scipy.interpolate.interp1d(self.lisa[:,0], self.lisa[:,1])
        #Nominal 4 year LISA mission x 0.75 efficiency = 3 years.
        self.length = 3 * 3.154e7
        #Speed of light in m/s
        self.light = 299792458
        self.extdata = None
        if satellite == "lisa":
            self.fmax = np.log10(0.5)
            self.fmin = np.log10(5e-7)
            # The satellite arm length in m
            self.L0 = 2.5e9
            #See 1906.09244 eq. 2.1
            #The acceleration noise in m^2 s^-4 /Hz (3 fm)
            self.Sa = 3.e-15**2
            #The position noise in m^2 /Hz (15 pm)
            self.Sx = 1.5e-11**2
        elif satellite == "tiango":
            #From Hang Yu, 6/15/20
            #Arm length: 100 km
            #Laser power: 5 W
            #Laser wavelength: 532 nm
            #Mirror mass: 10 kg
            #Freq-independent squeezing: 10 dB.
            #So the shot noise is about 1.4e-22 [1/rtHz] and the
            #quantum-radiation-pressure noise is ~ 2e-22 * ( f / 0.1 Hz)^{-2}.
            #(Note the factor of L0 to match our units)
            #The excess noise below 0.03 Hz is due to gravity gradient.
            self.extdata = np.loadtxt("TianGOskyAvg.txt")
            self.fmin = -2
            self.fmax = 1
            self.L0 = 1e5
            self.Sx = (2.e-22 * self.L0)**2
            self.Sa = (1.4e-22 * self.L0)**2
        elif satellite == "tianqin":
            self.fmax = 1
            self.fmin = -5
            self.L0 = np.sqrt(3) * 1.e8
            self.Sa = 1.e-15**2
            self.Sx = 1.e-12**2
        elif satellite == "bdecigo":
            self.fmax = 1
            self.fmin = -2
            self.L0 = 1.e5
            self.Sa = (1.e-16/30)**2
            #This number is not in the DECIGO 2017 paper.
            #It is derived by requiring that the strain sensitivity be at 2x10^-23 /Hz^1/2.
            self.Sx = 3.e-18**2
        self.satfreq = np.logspace(self.fmin, self.fmax, int(self.nsamples_per_dec * (self.fmax - self.fmin)))
        if self.extdata is not None:
            self.extdata = self.downsample(self.satfreq, self.extdata[:,0], self.extdata[:,1])

    def PSD(self):
        """Get the root power spectral density in Hz^[-1/2]"""
        return self.satfreq, self.noisepsd(self.satfreq)

    def noisepsd(self, freq):
        """The root power spectral density computed analytically from 1906.09244 .
        See also https://arxiv.org/abs/gr-qc/9909080 or eq. 2 of 1512.02076 in Hz^(-1/2)."""
        if self.extdata is not None:
            return self.extdata
        RR = self.transfer(2 * math.pi * freq / self.light)
        Poms = self.Sx /self.L0**2 * (1 + (2e-3/freq)**4 )
        #8e-3 for LISA
        Pacc = self.Sa /self.L0**2 * (1+ (4e-4/freq)**2) * (1 + (freq/(8e-3 * 2.5e9/self.L0))**4) / (2 * math.pi * freq )**4
        Sn = (Poms + (3 + np.cos(4 * math.pi * freq * self.L0 / self.light)) * Pacc)/RR
        assert np.all(Sn > 0)
        return np.sqrt(Sn)

    def transfer(self, w):
        """Transfer function R(w), see eq. 2.9 of 1906.09244 and cancel the sin^2 factor and the 2 pi f / c."""
        return 0.3 /(1 + 0.6*(w * self.L0)**2)

class LIGOSensitivity(Sensitivity):
    """LIGO sensitivity curve as a function of frequency"""
    def __init__(self, dataset="A+"):
        super().__init__()
        #Power spectral density strain noise amplitude
        #f(Hz)     PSD (Hz^-1/2)
        if dataset == "A+":
            #https://dcc.ligo.org/LIGO-T1800042/public
            self.aligo = np.loadtxt("LIGO_AplusDesign.txt")
            #Say A+ is three years.
            self.length = 3.
        elif dataset == "design":
            #https://dcc.ligo.org/public/0149/T1800044/005/aLIGODesign.txt
            #Design is two years
            self.length = 2.
            self.aligo = np.loadtxt("aLIGODesign.txt")
        elif dataset == "O1":
            #LIGO O1 DCC
            ligoO1ff = np.loadtxt("LIGO_O1_freqVector.txt")
            ligoO1psd = np.loadtxt("LIGO_O1_psd.txt")
            self.aligo = np.vstack([ligoO1ff, ligoO1psd]).T
            #O1 is 4 months long
            self.length = 4./12.
        else:
            raise ValueError("Dataset not supported")
        #self.aligointp = scipy.interpolate.interp1d(self.aligo[:,0], self.aligo[:,1])
        #Convert length to seconds
        self.length *= 3.154e7
        fmin = np.log10(self.aligo[0,0])
        fmax = np.log10(self.aligo[-1,0])
        nsample = int((fmax - fmin) * self.nsamples_per_dec)
        self.ligofreq = np.logspace(fmin, fmax, nsample)
        self.ligopsd = self.downsample(self.ligofreq, self.aligo[:,0], self.aligo[:,1])

    def PSD(self):
        """Get the root power spectral density in Hz^[-1/2]"""
        #The input data is in strain power spectral density noise amplitude.
        #The (dimensionless) characteristic strain is
        #charstrain = np.sqrt(self.aligo[:,0]) * self.aligo[:,1]
        return self.ligofreq, self.ligopsd

class SGWBExperiment:
    """Helper class to hold pre-computed parts of the experimental data,
       presenting a consistent interface for different experiments."""
    def __init__(self, binarybh, sensitivity, trueparams, imribh=None, cstring=None, phase = None):
        self.cstring = cstring
        self.sensitivity = sensitivity
        self.freq, self.omegasens = sensitivity.omegadens()
        self.length = sensitivity.length
        self.bbh_singleamp = binarybh.OmegaGW(self.freq, Norm=1)
        self.mockdata = self.cosmicstringmodel(trueparams[0])
        self.mockdata += self.bhbinarymerger(trueparams[1])
        self.phase = phase

        if self.phase is not None and self.cstring is not None:
            raise ValueError("Must use Cosmic strings or Phase Transition, not both")

        self.imri_singleamp = 0
        #Add bckgrnd from IMRI
        if imribh is not None:
            self.imri_singleamp = imribh.OmegaGW(self.freq, Norm = 1)
            self.mockdata += self.imri_singleamp * trueparams[2]

    def cosmicstringmodel(self, Gmu):
        """The cosmic string SGWB model."""
        if Gmu <= 0:
            return 0
        if self.cstring is None:
            return 0
        return self.cstring.OmegaGW(self.freq, Gmu)

    def phasemodel(self, Ts, alpha):
        """The phase transition SGWB model."""
        if self.phase is None:
            return 0
        return self.phase.OmegaGW(self.freq, Ts, alpha = alpha)

    def imrimodel(self, amp):
        """IMRI model"""
        if amp < 0:
            return 0
        return amp * self.imri_singleamp

    def whitedwarfmodel(self, number):
        """The white dwarf background model.
        This can be neglected: alone of all the signals the WD background is
        localised in the galaxy, and this should allow it to be cleaned.
        It will be anisotropic and thus have a yearly modulation.
        """
        _ = (number)
        return 0

    def bhbinarymerger(self, amp):
        """The unresolved BH binary model. Using the model from
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.131102"""
        return amp * self.bbh_singleamp

    def omegamodel(self, cosmo, bbhamp, imriamp=0, ptalpha = 1):
        """Construct a model with three free parameters,
           combining multiple SGWB sources:
           Strings, BBH (LIGO) and BBH (IMRI)."""
        cos = 0
        if self.cstring is not None:
            cos = self.cosmicstringmodel(cosmo)
        elif self.phase is not None:
            cos = self.phasemodel(cosmo, alpha = ptalpha)
        return cos + self.imrimodel(imriamp) + self.bhbinarymerger(bbhamp)

class Likelihoods:
    """Class to perform likelihood analysis on SGWB.
    Args:
    """
    def __init__(self, nsamples=100, strings=True, phase = False, imri = True, ligo = True, satellites="lisa"):
        self.ureg = ureg
        self.binarybh = BinaryBHGWB()

        self.cstring = None
        if strings:
            self.cstring = CosmicStringGWB()
        self.phase = None
        if phase:
            self.phase = PhaseTransition()

        if self.phase is not None and self.cstring is not None:
            raise ValueError("Must use Cosmic strings or Phase Transition, not both")

        self.imri = None
        if imri:
            self.imribh = IMRIGWB()

        if isinstance(satellites, str):
            satellites = (satellites,)

        self.sensitivities = []
        if ligo:
            self.sensitivities += [LIGOSensitivity(),]
        if satellites[0] != "":
            self.sensitivities += [SatelliteSensitivity(satellite = sat) for sat in satellites]

        if self.cstring is not None:
            #This is the "true" model we are trying to detect: no cosmic strings,
            #LIGO current best fit merger rate.
            self.trueparams = [0, 56., 0.01]
        elif self.phase is not None:
            #Phase transition parameters being zero will lead to divide by zero,
            #so just make them small and the frequency high.
            self.trueparams = [1e10, 56., 0.01, 1e-9]
        self.experiments = [SGWBExperiment(binarybh = self.binarybh, imribh = self.imribh, cstring = self.cstring, phase = self.phase, sensitivity = sens, trueparams = self.trueparams) for sens in self.sensitivities]

        #Expected number of ligo detections at time of LISA launch for the BBH amplitude prior.
        self.nligo = 1000

    def lnlikelihood(self, params):
        """Likelihood parameters:
        0 - cosmic string tension or PT T* (frequency)
        1 - BH binary merger rate amplitude
        2 - IMRI merger rate amplitude
        3 - PT amplitude alpha (if phase is not None)
        """
        #Priors: positive BBH merger rate
        if params[1] < 0:
            return -np.inf
        if params[2] < 0:
            return - np.inf
        ptalpha = 1
        #CS string tension limit from EPTA
        if self.cstring is not None:
            if params[0] > np.log(2.e-11):
                return -np.inf
            #Prevent underflow
            if params[0] < -51:
                return -np.inf
        elif self.phase is not None:
            #Phase transition energy: lower limit so that it lies well inside the LISA band
            #Upper limit so it lies within LIGO band
            if params[0] > np.log(1e9):
                return -np.inf
            if params[0] < np.log(1e2):
                return -np.inf
            #alpha: upper limit set by plausible physical values,
            #lower limit just something slightly above zero.
            if params[3] > 1:
                return -np.inf
            if params[3] < 1e-5:
                return -np.inf
            ptalpha = params[3]
        # LIGO prior: Gaussian on BBH merger rate with central value of the true value.
        # Remove this: it may be that the unresolved high redshift binaries merge
        # at a different rate to the low redshift ones and so we should allow a free amplitude
        # ampprior = -1 * (params[1] - self.trueparams[1])**2 / self.nligo
        llike = 0

        for exp in self.experiments:
            model = exp.omegamodel(cosmo = np.exp(params[0]), bbhamp = params[1], imriamp = params[2], ptalpha=ptalpha)
            llike += - 1 * exp.length * np.trapz(((model - exp.mockdata)/ exp.omegasens)**2, x=exp.freq)
            #like += ampprior * np.size(exp.freq)
        #print(np.exp(params[0]), like)
        return llike

    def do_sampling(self, savefile, nwalkers=100, burnin=300, nsamples = 300, while_loop=True, maxsample=200):
        """Do the sampling with emcee"""
        #Limits
        if self.cstring is not None:
            #Say Gmu ranges from exp(-45) - exp(-14), LIGO merger rate between 0 and 100
            #and IMRI rate from 0 to 1.
            pr = np.array([10, 100, 0.1])
            #Priors are assumed to be in the middle.
            cent = np.array([-40, 55, 0.05])
        elif self.phase is not None:
            pr = np.array([10, 100, 0.1, 0.2])
            cent = np.array([2, 100, 0.1, 0.2])
        p0 = [cent+2*pr/16.*np.random.rand(len(pr))-pr/16. for _ in range(nwalkers)]
        lnk0 = np.array([self.lnlikelihood(pp) for pp in p0])
        assert np.all(np.isfinite(lnk0))
        emcee_sampler = emcee.EnsembleSampler(nwalkers, np.size(pr), self.lnlikelihood)
        pos, _, _ = emcee_sampler.run_mcmc(p0, burnin)
        #Check things are reasonable
        assert np.all(emcee_sampler.acceptance_fraction > 0.01)
        emcee_sampler.reset()
        self.cur_results = emcee_sampler
        gr = 10.
        count = 0
        while np.any(gr > 1.01) and count < maxsample:
            emcee_sampler.run_mcmc(pos, nsamples)
            gr = gelman_rubin(emcee_sampler.chain)
            print("Total samples:",nsamples," Gelman-Rubin: ",gr)
            np.savetxt(savefile, emcee_sampler.flatchain)
            count += 1
            if while_loop is False:
                break
        self.flatchain = emcee_sampler.flatchain
        return emcee_sampler

class SN1AGWB:
    """Model for the supernovae 1a background following 1511.02542."""
    def __init__(self):
        self.ureg = ureg
        self.Normunit = 1*self.ureg("Gpc^-3/year")
        #1308.0137 gives the rate of SN1A at 10^{-4} / yr/ Mpc^{-3}, which is
        #10^{5} / yr/ Gpc^{-3}
        self.cc = (self.ureg("speed_of_light").to("m/s")).magnitude
        self.GG = ((1*self.ureg.newtonian_constant_of_gravitation).to_base_units()).magnitude
        #Solar mass
        self.ms = 1.989e30 #* self.ureg("kg"))

    def rhocrit(self):
        """Critical energy density at z=0 in kg/m^3"""
        rhoc = 3 * (67.9 * self.ureg('km/s/Mpc'))**2
        return rhoc / 8 / math.pi/ self.ureg.newtonian_constant_of_gravitation

    def dEdfstot(self, femit, mu = 0.41):
        """Total energy per unit strain. Fit by eye to figure 5 of 1511.02542. In units of 1e39 erg/Hz"""
        sigma = mu / 2.
        return 4.8*np.exp(-(femit - mu)**2/sigma**2) + 2*np.exp(-(femit - math.pi*mu)**2/(2*sigma)**2)

    def OmegaGW(self, freq, Norm=1e5, mu=0.41):
        """OmegaGW as a function of frequency. Normalization is in units of mergers per Gpc^3 per year.
        Normalisation units are arbitrary right now."""
        Hub = 67.9 * self.ureg("km/s/Mpc")
        #See eq. 2 of 1609.03565
        dE = self.dEdfstot(freq, mu=mu)
        freq = freq * self.ureg("Hz")
        normfac = (Norm * self.Normunit / Hub * freq / self.ureg("speed_of_light")**2 / self.rhocrit())
        normfac = normfac.to_base_units()
        #assert normfac.check("[]")
        return  normfac.magnitude * dE


class BinaryBHGWB:
    """Model for the binary black hole background, fiducial version from LIGO."""
    def __init__(self):
        self.ureg = ureg
        self.Normunit = 1*self.ureg("Gpc^-3/year")
        #We don't need to marginalise these parameters as they do not change
        #the shape at low frequencies, only the normalization.
        self.bb = 1.5
        self.aa = 1.92
        self.zm = 2.6
        # This is the Ajith 2011 (0909.2867) template for the spectral energy density of the black holes *at source* redshift.
        # This needs to be adjusted to the observed frame redshift.
        #Template shape is: frequency, redshift, dEdf
        #self.Ajith_template = np.loadtxt("dEdfs_FULL_fobs_dndM_2p35_mmin_3msun_mmax_100msun_Ajith_spectrum.dat")
        #self.Ajithintp = scipy.interpolate.interp2d(self.Ajith_template[:,0], self.Ajith_template[:,1], self.Ajith_template[:,2], kind='linear')
        self.cc = (self.ureg("speed_of_light").to("m/s")).magnitude
        self.GG = ((1*self.ureg.newtonian_constant_of_gravitation).to_base_units()).magnitude
        #Solar mass
        self.ms = 1.989e30 #* self.ureg("kg"))

    def chi(self, z):
        """Comoving distance to z."""
        Hub = 67.9 * self.ureg("km/s/Mpc")
        integ = lambda z_: 1./np.sqrt(0.3 * (1+z_)**3 + 0.7)
        chi,_ = scipy.integrate.quad(integ, 0, z)
        return (self.ureg("speed_of_light") / Hub  * chi).to_base_units()

    def dLumin(self, z):
        """Luminosity distance"""
        return (1+z) * self.chi(z)

    def rhocrit(self):
        """Critical energy density at z=0 in kg/m^3"""
        rhoc = 3 * (67.9 * self.ureg('km/s/Mpc'))**2
        return rhoc / 8 / math.pi/ self.ureg.newtonian_constant_of_gravitation

    def Rsfrnormless(self, zzp1):
        """Black hole merger rate as a function of the star formation rate. Normalized to unity at =zm."""
        return self.aa * np.exp(self.bb * (zzp1 - 1 - self.zm)) / (self.aa - self.bb + self.bb * np.exp(self.aa*(zzp1 - 1 - self.zm)))

    def fmergerV2(self, msum):
        """Merger phase frequency"""
        fmerg = 0.04*self.cc**3/(self.GG*msum* self.ms)
        return fmerg #.to("Hz")

    def mchirp(self, m1, m2):
        """Power of the chirp mass. Should be m1*m2 / (m1+m2)**(1/3)
        (note this is Mchirp^{5/3}: Mchirp = (m1 m2)^{3/5} / (m1+m2)^{1/5})"""
        return m1*m2 / (m1+m2)**(1/3)

    def dEdfsMergV2(self, m1, m2, femit):
        """Energy per unit strain in merger phase"""
        mchirp = self.mchirp(m1, m2)
        return 1./3.*(math.pi**2*self.GG**2)**(1/3)*(self.ms**(5./3)*mchirp)*femit**(2./3)/self.fmergerV2(m1 + m2)

    def dEdfsInsp(self, mchirp, femit):
        """Energy per unit strain in inspiral phase. mchirp should be m1*m2 / (m1+m2)**(1/3)
        (note this is Mchirp^{5/3}: Mchirp = (m1 m2)^{3/5} / (m1+m2)^{1/5})"""
        return 1./3.*(math.pi**2*self.GG**2/femit)**(1/3)*self.ms**(5./3)*mchirp

    def fqnrV2(self, msum):
        """Ringdown phase frequency."""
        fqnr = 0.915*(self.cc**3*(1. - 0.63*(1. - 0.67)**(3/10.)))/(2*math.pi*self.GG*msum*self.ms)
        return fqnr

    def dEdfs(self, femit, m1, m2):
        """Energy per unit strain for a single merger"""
        fmerg = self.fmergerV2(m1+m2)
        fqnr = self.fqnrV2(m1+m2)
        if femit > fqnr:
            return 0
        if femit > fmerg:
            return self.dEdfsMergV2(m1, m2, femit)
        #femit < fmerg
        mchirp = self.mchirp(m1, m2)
        return self.dEdfsInsp(mchirp, femit)

    def _omegagwz(self, ff, alpha=-2.3, m2min=5, m2max=50):
        """Integrand as a function of redshift, taking care of the redshifting factors.
        For numerical reasons, we split this into two segments, one for mergers, one for inspirals.
        """
        #From z= 0.01 to 20.
        zmax = 10
        #zmin: exclude binaries that are at zero redshift as they are resolved, even with current LIGO.
        zmin = 1.0
        #m1 has a power law weight. m2 is uniformly sampled.
        ominsp = lambda zzp1, m2, m1 : self.Rsfrnormless(zzp1) / HubbleEz(zzp1) * self.dEdfsInsp(self.mchirp(m1, m2), ff*zzp1) * m1**alpha / zzp1
        ommerg = lambda zzp1, m2, m1 : self.Rsfrnormless(zzp1) / HubbleEz(zzp1) * self.dEdfsMergV2(m1, m2, ff*zzp1) * m1**alpha / zzp1
        #If we are always in the inspiral band the integrals become separable.
        if zmax < self.fmergerV2(50+m2max)/ff:
            zzfreq = lambda zzp1: self.Rsfrnormless(zzp1) / HubbleEz(zzp1) * 1./3.*(math.pi**2*self.GG**2/(ff*zzp1))**(1/3)*self.ms**(5./3) / zzp1
            omegagwz, _ = scipy.integrate.quad(zzfreq, zmin, zmax)
            # The m2 integral can be done analytically.
            mm1int = lambda m1: 0.3 * m1**(alpha+1) * ((m1+m2max)**(2./3)*(2*m2max-3*m1)-(m1+m2min)**(2./3) * (2*m2min-3*m1))
            omegagwm1, _ = scipy.integrate.quad(mm1int, 5, 50)
            return omegagwm1 * omegagwz
        #If we are never in the merger phase, do nothing
        if ff * zmin > self.fqnrV2(5 + m2min):
            return 0
        #Constant limits for m2.
        _m2max = lambda m1 : m2max
        _m2min = lambda m1 : m2min
        #Limits for z+1: lower limit is 1, upper depends on f.
        zp1merge = lambda m1, m2 : min([zmax, self.fmergerV2(m1+m2)/ff])
        zp1min = lambda m1, m2 : zmin
        omegagwz, _ = scipy.integrate.tplquad(ominsp, 5, 50, _m2min, _m2max, zp1min, zp1merge)
        zp1ring = lambda m1, m2 : min([zmax, self.fqnrV2(m1+m2)/ff])
        zp1minerge = lambda m1, m2 : min([zmax, max([zmin, self.fmergerV2(m1+m2)/ff])])
        omegamerg, _ = scipy.integrate.tplquad(ommerg, 5, 50, _m2min, _m2max, zp1minerge, zp1ring)
        return omegagwz + omegamerg

    def OmegaGW(self, freq, Norm=56., alpha=-2.3, m2min=5, m2max=50):
        """OmegaGW as a function of frequency. Normalization is in units of mergers per Gpc^3 per year. alpha is the power law of the mass distribution assumed. We are reasonably insensitive to this, so we do not vary it in the chain."""
        Hub = 67.9 * self.ureg("km/s/Mpc")
        omegagw_unnormed = np.array([self._omegagwz(ff, alpha=alpha, m2min=m2min, m2max = m2max) for ff in freq])
        #See eq. 2 of 1609.03565
        freq = freq * self.ureg("Hz")
        normfac = (Norm * self.Normunit / Hub * freq / self.ureg("speed_of_light")**2 / self.rhocrit())/self.Rsfrnormless(1)
        normfac = normfac.to_base_units()
        #assert normfac.check("[]")
        return  normfac.magnitude * omegagw_unnormed

class IMRIGWB(BinaryBHGWB):
    """Subclasses the Binary BH model for IMRIs. Currently assumes that
    the emission frequencies of the phases are as for the normal binaries (which is not totally true)."""
    def _omegagwz(self, ff, alpha=-2.3, m2min=5, m2max=50):
        """Integrand as a function of redshift, taking care of the redshifting factors.
        For numerical reasons, we split this into two segments, one for mergers, one for inspirals.
        We make the approximation that m1 + m2 ~ m2 for IMRIs.
        """
        #From z= 0.01 to 20.
        zmax = 10
        #zmin: exclude binaries that are at zero redshift as they are resolved, even with current LIGO.
        zmin = 1.0
        #m1 has a power law weight. m2 is uniformly sampled.
        #m1+m2 ~ m2 -> mchirp = m2 m1/m2^(1/3) = m2^(2/3) m1
        EInsp = lambda femit: 1./3.*(math.pi**2*self.GG**2/femit)**(1/3)*self.ms**(5./3)
        ominsp = lambda zzp1, m2 : self.Rsfrnormless(zzp1) / HubbleEz(zzp1) * EInsp(ff*zzp1) * m2**(2/3) / zzp1
        Emergapprox = lambda m2: 1./3.*(math.pi**2*self.GG**2)**(1/3)*(self.ms**(5./3)*m2**(2/3))/self.fmergerV2(m2)
        ommerg = lambda zzp1, m2: self.Rsfrnormless(zzp1) / HubbleEz(zzp1) * Emergapprox(m2) *(ff*zzp1)**(2./3)
        #Integrated m1 dependence
        m1integral = (50**(alpha+2)-5**(alpha+2))/(alpha+2)
        #If we are always in the inspiral band the integrals become separable.
        if zmax < self.fmergerV2(50+m2max)/ff:
            zzfreq = lambda zzp1: self.Rsfrnormless(zzp1) / HubbleEz(zzp1) * 1./3.*(math.pi**2*self.GG**2/(ff*zzp1))**(1/3)*self.ms**(5./3)
            omegagwz, _ = scipy.integrate.quad(zzfreq, zmin, zmax)
            # The m2 and m1 integrals can be done analytically as we approximate m1 << m2.
            omegagwm1 = 0.3 * m1integral * ((m2max)**(2./3)*(2*m2max)-(5+m2min)**(2./3) * (2*m2min))
            return omegagwm1 * omegagwz
        #If we are never in the merger phase, do nothing
        if ff * zmin > self.fqnrV2(m2min):
            return 0
        #Limits for z+1: lower limit is 1, upper depends on f.
        zp1merge = lambda m2 : min([zmax, self.fmergerV2(m2)/ff])
        zp1min = lambda m2 : zmin
        omegagwz, _ = scipy.integrate.dblquad(ominsp, m2min, m2max, zp1min, zp1merge)
        zp1ring = lambda m2 : min([zmax, self.fqnrV2(m2)/ff])
        zp1minerge = lambda m2 : min([zmax, max([zmin, self.fmergerV2(m2)/ff])])
        omegamerg, _ = scipy.integrate.dblquad(ommerg, m2min, m2max, zp1minerge, zp1ring)
        return m1integral * (omegagwz + omegamerg)

    def OmegaGW(self, freq, Norm=0.01, alpha=-2.3, m2min=5e2, m2max=1e4):
        """OmegaGW as a function of frequency. Normalization is in units of mergers per Gpc^3 per year."""
        return super().OmegaGW(freq, Norm=Norm, alpha = alpha, m2min=5e2, m2max=1e4)

class EMRIGWB:
    """Uses the EMRI SGWB from Bonetti & Sesana, """
    def __init__(self):
        self.hub = ((67.9 * ureg('km/s/Mpc')).to("1/s").magnitude)
        #Model 1 which is fiducial
        bonetti = np.loadtxt("hc_EMRImodel1nospin_4.0yr_Babak20.txt")
        self.freq = bonetti[:,0]
        #Use the model with detected signals removed
        self.hcc = bonetti[:,2]
        #Omega_GW
        self.omegagw = self.freq**3 * 4 * math.pi**2 / 3 / self.hub**2
        self.omegagwii = scipy.interpolate.interp1d(self.freq, self.omegagw)

    def OmegaGW(self, freq, Norm=1):
        """OmegaGW as a function of frequency. Normalization is relative to the fiducial model."""
        return Norm * self.omegagwii(freq)

def gcorr(x):
    """Corrections to radiation density from freezeout"""
    if x > 753.946:
        return 106.6
    if x > 73.9827:
        return 207.04170721279553 + 9.297437227007177e8/x**4 - 6.762901334043625e7/x**3 + 1.5888320597630164e6/x**2 - \
            17782.040653778313/x - 0.3100369947620232*x + 0.0005219361630722421*x**2 - 4.5089224156018806e-7*x**3 + 1.571478658251381e-10*x**4
    if x > 19.2101:
        return 25.617077215691925 - 85049.61151267929/x**4 + 44821.06301089602/x**3 - 9407.524062731489/x**2 + 1010.3559753329803/x + 1.9046169770731767*x -  \
            0.02182528531424682*x**2 + 0.00013717391195802956*x**3 - 4.656717468928253e-7*x**4
    if x > 0.210363:
        return 96.03236599599052 + 0.1417211377127251/x**4 - 1.9033759380298605/x**3 + 9.91065103843345/x**2 - 26.492041757131332/x - 1.923898527856908*x +  0.18435280458948589*x**2 - \
            0.008350062215001603*x**3 + 0.000158735905184788*x**4
    if x > 0.152771:
        return -2.0899590637819996e8 - 3113.914061747819/x**4 + 140186.13699256277/x**3 - 2.7415777741917237e6/x**2 + 3.0404304467460793e7/x + 9.110822620276868e8*x \
            - 2.4573038140264993e9*x**2 + 3.7443023200934935e9*x**3 - 2.4636941241588144e9*x**4
    if x > 0.0367248:
        return 5426.500494710776 + 0.0019390100970067777/x**4 - 0.23020238075499902/x**3 + 11.587823813540876/x**2 - 322.51882049359654/x - 55993.18120267802*x \
            + 349376.7985871369*x**2 - 1.2042764400287867e6*x**3 + 1.7809006388749087e6*x**4
    if x > 0.00427032:
        return -195.37500200383414 - 3.246844915516518e-8/x**4 + 0.000028390675511486528/x**3 - 0.010121020979848003/x**2 + 1.9052124373133226/x + 13012.873468646329*x \
            - 473484.5788624826*x**2 + 9.244933716566762e6*x**3 - 7.352186782452533e7*x**4
    if x > 0.000163046:
        return 6.995576000906495 - 5.271793610584592e-15/x**4 + 1.1471366111587497e-10/x**3 - 9.017840481639097e-7/x**2 + 0.0022702982380190806/x + 2706.0190947576525*x \
            - 1.1981368143068189e6*x**2 + 2.6230001478632784e8*x**3 - 2.1859643005333782e10*x**4
    if x > 0.0000110255:
        return 1.3910688031037668 - 1.0512592071931046e-19/x**4 + 2.671929070189596e-14/x**3 - 2.5014060168690585e-9/x**2 + 0.00010456662194366741/x + 4706.80818171218*x \
            + 6.752939034535752e6*x**2 + 2.531715178089083e12*x**3 - 1.05933524939722e16*x**4
    return 3.17578

#Here comes the cosmic string model
class CosmicStringGWB:
    """Model for the gravitational wave background from cosmic strings. From Yanou's Mathematica notebook and 1808.08968."""
    def __init__(self):
        self.ureg = ureg
        self.HzoverGeV = 6.58e-25 #(1*self.ureg.planck_constant/2/math.pi).to('s*GeV')
        #String loop formation parameters
        self.Gamma = 50.
        #Large loop size as fraction of Hubble time.
        self.alpha = 0.1
        self.hub = 0.679 #* 100 * ureg("km/s/Mpc").to_base_units()).magnitude
        #Large loop fraction
        self.Fa = 0.1
        #Energy loss to loop formation. These are approximate values from lattice simulations for
        #different background cosmologies
        self.CeffR = 5.7
        self.CeffM = 0.5
        #Newton's constant
        self.GG = 1/1.2211e19**2 #GeV
        #Critical density
        self.rhoc = 1.05375e-5*(1.e-13/5)**3 * self.hub**2 #GeV^4
        #Current time in hubble times.
        self.t0 = 13.8*1.e9 * 365.25 * 24 * 60*60/self.HzoverGeV
        #self.t0 = (13.8e9 * self.ureg('years')).to('s') / HzoverGev
        self.zeq = 3360
        self.teq = self.t0 / (1+self.zeq)**(3./2)
        self.tDelta0 = self.tdelta(1) #T_delta = 5 GeV
        #self.aaeq = 1/(1 + 3360)
        #self.amin = 1e-20
        #self.amax = 1e-5
        #From Yanou's table. TODO: reimplement.
        self.asd = np.loadtxt("a_evolution_in_Standard_cosmology.dat")
        #Input a t get an a out. Goes from a = 1e-30 to a = 1.
        self.aRuntab = scipy.interpolate.interp1d(np.log(self.asd[:,0]), np.log(self.asd[:,1]), kind="quadratic", fill_value="extrapolate")
        self.tF = self.asd[0,0]

    def aRunS(self, logt):
        """Call the a interpolator"""
        return np.exp(self.aRuntab(logt))

    def tdelta(self, T):
        """Time during the inflation era"""
        return 0.30118 /np.sqrt(gcorr(T) * self.GG) / T**2

    def Ceff(self, tik):
        """Effective string decay parameter"""
        return self.CeffM * (tik > self.teq) + self.CeffR * (tik <= self.teq)

    def Gammak(self, k):
        """Mode-dependent Gamma"""
        return self.Gamma * k**(-4./3) / 3.6

    def tik(self, tt, Gmu, freqk, aa):
        """Formation time of loops with mode number k in s.
        Takes as argument freqk, which is freq / k"""
        return (self.Gamma * Gmu * tt + 2 / freqk * aa) / (self.alpha + self.Gamma * Gmu)

    def omegaintegrand(self, logt, Gmu, freqk):
        """Integrand for the Omega contribution from cosmic strings.
        Takes freqk = freq/k"""
        tt = np.exp(logt)
        aa = self.aRunS(logt)
        tik = self.tik(tt, Gmu, freqk, aa)
        #if tik < self.tF:
            #return 0
        #if tt < tik:
            #return 0
        om = self.Ceff(tik) / (tik**4 * self.rhoc) * aa**2 * self.aRunS(np.log(tik))**3 * (tt / self.GG)
        return om

    def OmegaEpochk(self, Gmu, freq, kk, ts, te):
        """The total omega of cosmic strings from a given epoch"""
        #Frequency is in GeV units!
        freq *= self.HzoverGeV
        #Enforce that tt > tik so the strings exist
        tstart2 = lambda logt: np.exp(logt) - self.tik(np.exp(logt), Gmu, freq/kk, self.aRunS(logt))
        if tstart2(np.log(ts)) < 0:
            sol = scipy.optimize.root_scalar(tstart2, x0=np.log(ts), x1=np.log(te))
            #print("old: ", ts, "new:", np.exp(sol.root))
            ts = np.exp(sol.root)
        #Enforce that tik > self.tF so the interpolation table works
        tstart = lambda logt: self.tik(np.exp(logt), Gmu, freq / kk, self.aRunS(logt)) - self.tF
        if tstart(np.log(ts)) < 0:
            sol = scipy.optimize.root_scalar(tstart, x0=np.log(ts), x1=np.log(te))
            #print("tF old: ", ts, "new:", np.exp(sol.root))
            ts = np.exp(sol.root)

        omega , _ = scipy.integrate.quad(self.omegaintegrand, np.log(ts), np.log(te), args=(Gmu, freq/kk), epsabs=1e-10, epsrel=1e-6, limit=150)
        prefac = 2 * kk / freq * self.Fa * self.Gammak(kk) * Gmu**2 / (self.alpha * (self.alpha + self.Gamma * Gmu))
        return omega * prefac

    def OmegaGWMk(self, Gmu, freq, k):
        """Eq. 2.14 of 1808.08968"""
        #Yanou splits this integral into three pieces depending on the expansion time, but we don't need to.
        return self.OmegaEpochk(Gmu, freq, k, self.tF, self.t0)

    def OmegaGW(self, freq, Gmu):
        """SGWB power (omega_gw) for a string forming during radiation domination."""
        #Add extra bins to extend the table to high k, low frequency
        lnewlf = np.log10(freq[0]/31)
        lnewhf = np.log10(freq[-1]*1.02)
        nsamp = int((lnewhf - lnewlf) * 3)
        fextended = np.logspace(lnewlf, lnewhf, nsamp)
        omegagmk1 = np.array([self.OmegaGWMk(Gmu, ff, 1) for ff in fextended])
        omintp = scipy.interpolate.interp1d(np.log(fextended), np.log(omegagmk1))
        P4R = np.array([np.sum([self.Gammak(k) * np.exp(omintp(np.log(ff/k))) for k in range(1,31)]) for ff in freq])/self.Gammak(1)
        #P4R = np.array([np.sum([self.OmegaGWMk(Gmu, ff, k) for k in range(1,30)]) for ff in freq])
        assert np.size(P4R) == np.size(freq)
        return P4R

class PhaseTransition:
    """Model for the gravitational wave contribution from phase transitions. From Yanou's notebook. See for reference:
    1903.09642, 1809.08242. Assume Gamma_dec > H_* There are contributions from sound waves and from turbulence.

    For sound waves: for fixed overall energy scale, if alpha inc then Ts dec  and cRs inc. alpha: strength of the PT.

    GW from turbulence: important if SW period is much shorter than 1 Hubble time, corresponds to plasma entering non-linear regime after oscillation (SW) period. Decreasing cRs->increasing turbulance contribution relative to SW.

    alpha from 0.01 to 30?
    """
    def __init__(self):
        #See 1004.4187 for vw != 1
        self.vw = 1
        #Speed of light = 1
        self.cs = 1/np.sqrt(3)
        hub = 0.679
        self.h2 = hub**2
        #Planck mass in GeV/c^2
        self.Mp = 1.220910e19
        #Fraction of the phase transition energy in bubble walls.
        #Because we are neglecting GW from bubbles, this only appears
        #in defining alphaeff.
        #Usually this is very small and alphaeff ~ alpha.
        self.kcol = 1.e-11

    def Hubble(self, T):
        """Hubble rate at high temperatures"""
        return 1.66 * np.sqrt(gcorr(T)) * T**2 / self.Mp

    def Rs(self, cRs, Ts):
        """Bubble size at t*"""
        return cRs / self.Hubble(Ts)

    def alphaeff(self, alpha):
        """alpha: strength of the PT.
        This is the PT strength minus the energy in bubble walls."""
        return alpha * (1-self.kcol)

    def kkSW(self, alpha):
        """Efficiency coefficient for sound wave GW production, eq. 3.4"""
        aeff = self.alphaeff(alpha)
        return aeff / alpha * aeff / (0.73 + 0.0083 * np.sqrt(aeff) + aeff)

    def Uff(self, alpha):
        """RMS fluid velocity Uf squared. This is approximate, see eq. 3.8 of 1903.09642"""
        aeff = self.alphaeff(alpha)
        return 0.75 * aeff / (1 + aeff) * self.kkSW(alpha)

    def tauSW(self, cRs, Ts, alpha):
        """Sound wave optical depth"""
        return np.min([1/self.Hubble(Ts), self.Rs(cRs, Ts)/self.Uff(alpha)])

    def ffSW(self, cRs, Ts):
        """Sound wave frequency"""
        return 3.4 / ((self.vw - self.cs) * self.Rs(cRs, Ts))

    def OmegaSWs(self, fs, cRs, Ts, alpha):
        """GW spectrum at percolation time t*"""
        tauSW = self.tauSW(cRs, Ts, alpha)
        ffrat = fs / self.ffSW(cRs, Ts)
        #Note cRs = R* H* as in the paper.
        return 0.38 * cRs * self.Hubble(Ts) * tauSW * (self.kkSW(alpha) * alpha / (1 + alpha))**2 * ffrat**3 / ( 1 + 0.75 * ffrat**2)**(7./2)

    def fss(self, f, Ts, Trh):
        """Frequency at t*. Note we set Gamma_dec = H*. Eq. 4.7 of the ref."""
        return f * self.Hubble(Ts) * (100 / gcorr(Trh))**(1./6) * (100 / Trh) / 1.65e-5

    def OmegaSW0(self, f, cRs, Ts, alpha):
        """GW spectrum at present day"""
        #Make instantaneous reheating approximation
        Trh = Ts
        fss = self.fss(f, Ts, Trh)
        return 1.67e-5 * self.h2 * (100 / gcorr(Trh))**(1./3.) * self.OmegaSWs(fss, cRs, Ts, alpha)

    def ffTB(self, cRs, Ts):
        """Frequency of turbulent contributions"""
        return 3.9 / ((self.vw - self.cs) * self.Rs(cRs, Ts))

    def OmegaTBs(self, fs, cRs, Ts, alpha):
        """Omega Turbulent at t*"""
        ffrat = fs / self.ffTB(cRs, Ts)
        ffdep = ffrat**3 * (1 + ffrat)**(-11/3.) / (1 + 8*math.pi*fs/self.Hubble(Ts))
        return 6.8 * cRs * (1 - self.Hubble(Ts) * self.tauSW(cRs, Ts, alpha)) * (self.kkSW(alpha)*alpha / (1 + alpha))**(3/2.) * ffdep

    def OmegaTB0(self, fs, cRs, Ts, alpha):
        """Omega Turbulent at t_*"""
        #Make instantaneous reheating approximation
        Trh = Ts
        fss = self.fss(fs, Ts, Trh)
        omegatbs = self.OmegaTBs(fss, cRs, Ts, alpha)
        return 1.67e-5 * self.h2 * (100 / gcorr(Trh))**(1./3) * omegatbs

    def OmegaGW(self, f, Ts, alpha=1):
        """Total OmegaGW: this is just sound wave dropping the subdominant bubbles and turbulence following 1910.13125.
        We pick alpha = 1 as a fiducial model. It can be from 0.5 to 4 ish.
        beta is beta/H and can be 0.01 < b/H < 1
        Ts is in GeV and can have a variety of values."""
        #Eyeballing the right panel of Figure 17 of 1809.08242 gives alpha = (10 beta/H)^0.8
        #Eq. 6 of 1910.13125
        beta = alpha**(1/0.8) / 10.
        cRs = (8 * math.pi)**(1./3) / beta * np.max([self.vw, self.cs])
        return self.OmegaSW0(f, cRs, Ts, alpha) + self.OmegaTB0(f, cRs, Ts, alpha)

def test_cs():
    """Simple test routine to check the cosmic string model matches the notebook"""
    cs = CosmicStringGWB()
    assert np.abs(cs.teq / 3.39688e36 - 1 ) < 1.e-4
    assert np.abs(cs.tik(cs.teq, 1.e-11, 10 * cs.HzoverGeV/1., cs.aRunS(np.log(cs.teq))) / 1.69834e28 - 1) < 1.e-4
    assert np.abs(cs.Gammak(2)/ 5.51181 - 1 ) < 1.e-4
    assert np.abs(cs.tDelta0 / 4.22024e17 - 1) < 1.e-4
    assert np.abs(gcorr(1) / 75.9416 - 1) < 1.e-4
    assert np.abs(cs.OmegaEpochk(1.e-11, 10, 1, cs.tF, cs.tDelta0) / 2.03699e-14 - 1) < 2.e-3
    assert np.abs(cs.OmegaEpochk(1.e-11, 10, 1, cs.tDelta0, cs.teq) / 3.21958e-11 - 1) < 2.e-2
    assert np.abs(cs.OmegaEpochk(1.e-11, 10, 1, cs.teq, cs.t0) / 1.55956e-17 - 1) < 2.e-3
    tot = cs.OmegaGW([1e-6,20], 1e-11)
    assert np.all(np.abs(tot - np.array([1.05797682e-09, 1.69091713e-10])) < 2.e-3)


if __name__=="__main__":
    #LISA only
    like = Likelihoods(imri=True, strings=True, ligo = True, satellites="lisa")
    like.do_sampling(savefile = "samples_ligo_lisa_string_bbh.txt")
    #LISA + DECIGO
    like = Likelihoods(imri=True, strings=True, ligo = True, satellites=("lisa","bdecigo"))
    like.do_sampling(savefile = "samples_ligo_lisa_decigo_string_bbh.txt")
    #LISA + TianGo
    like = Likelihoods(imri=True, strings=True, ligo = True, satellites=("lisa","tiango"))
    like.do_sampling(savefile = "samples_ligo_lisa_tiango_string_phase_bbh_imri.txt")

    #LISA only with phase transition
    like = Likelihoods(imri=True, phase=True, strings=False, ligo = True, satellites="lisa")
    like.do_sampling(savefile = "samples_ligo_lisa_phase_bbh.txt")
    #LISA+TianGo with phase transition
    like = Likelihoods(imri=True, phase=True, strings=False, ligo = True, satellites=("lisa", "tiango"))
    like.do_sampling(savefile = "samples_ligo_lisa_tiango_phase_bbh.txt")
    #LISA+DECIGO with phase transition
    like = Likelihoods(imri=True, phase=True, strings=False, ligo = True, satellites=("lisa", "bdecigo"))
    like.do_sampling(savefile = "samples_ligo_lisa_decigo_phase_bbh.txt")
