"""Short code to compute likelihood functions for stochastic gravitational wave background detection using a mid-band experiment.

Neglect neutron stars because in the LISA band they are just a (small) rescaling of the SGWB amplitude.

TODO: midband Sn1a background and perhaps a LISA-specific SGWB from high redshift supermassive BH mergers, EMRIs or IMBHs
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
        omegagw = 2 * math.pi**2 / 3 / self.hub**2 * ff**3 * psd**2
        # We assume sampling at 1/32 Hz following 1903.02886
        # and a design run of 2 years.
        omegagw /= np.sqrt(2 * self.length * 1./32)
        return ff, omegagw

    def PSD(self):
        """Root power spectral density in Hz^{-1/2}"""
        raise NotImplementedError

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
        #Nominal 4 year LISA mission
        self.length = 4 * 3.154e7
        #Speed of light in m/s
        self.light = 299792458
        if satellite == "lisa":
            self.satfreq = np.logspace(np.log10(1.95e-7), np.log10(2.08), 400)
            # The satellite arm length
            self.L0 = 2.5e9
            #The acceleration noise
            self.Sa = 3.e-15**2
            #The position noise
            self.Sx = 1.5e-11**2
        elif satellite == "tianqin":
            self.satfreq = np.logspace(-5, 1, 400)
            self.L0 = np.sqrt(3) * 1.e8
            self.Sa = 1.e-15**2
            self.Sx = 1.e-12**2
        elif satellite == "bdecigo":
            self.satfreq = np.logspace(-2, 2, 400)
            self.L0 = 1.e5
            self.Sa = (1.e-16/30)**2
            #This number is not in the DECIGO 2017 paper.
            #It is derived by requiring that the strain sensitivity be at 2x10^-23 /Hz^1/2.
            self.Sx = 3.e-18**2

    def PSD(self):
        """Get the root power spectral density in Hz^[-1/2]"""
        return self.satfreq, self.noisepsd(self.satfreq)

    def noisepsd(self, freq):
        """The root power spectral density computed analytically from
        https://arxiv.org/abs/gr-qc/9909080 or eq. 2 of 1512.02076 in Hz^(-1/2)."""
        RR = np.sqrt(self.transfer(2 * math.pi * freq / self.light))
        hf2 = self.Sx / self.L0**2 + self.Sa/ (2 * math.pi * freq)**4 / self.L0**2 * ( 1 + 1.e-4 / freq)
        return 2 / RR * np.sqrt(hf2)

    def transfer(self, w):
        """The GW transfer function"""
        return 8./15 / ( 1 + (w * self.L0 / 0.41 / math.pi)**2)

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

    def PSD(self):
        """Get the root power spectral density in Hz^[-1/2]"""
        #The input data is in strain power spectral density noise amplitude.
        #The (dimensionless) characteristic strain is
        #charstrain = np.sqrt(self.aligo[:,0]) * self.aligo[:,1]
        return self.aligo[:,0], self.aligo[:,1]

class SGWBExperiment:
    """Helper class to hold pre-computed parts of the experimental data,
       presenting a consistent interface for different experiments."""
    def __init__(self, binarybh, cstring, sensitivity, trueparams, nsamples=400):
        self.cstring = cstring
        self.sensitivity = sensitivity
        self.freq, self.psd = sensitivity.omegadens()
        self.freq, self.psd = self.downsample(nsamples, self.freq, self.psd)
        self.bbh_singleamp = binarybh.OmegaGW(self.freq, Norm=1)
        self.mockdata = self.cosmicstringmodel(trueparams[0])
        self.mockdata += self.bhbinarymerger(trueparams[1])


    def downsample(self, nsamples, freq, psd):
        """Down-sample a sensitivity curve to a lower desired number of bins."""
        intp = scipy.interpolate.interp1d(np.log(freq), np.log(psd))
        jump = int(np.size(freq)/nsamples)
        return freq[::jump], np.exp(intp(np.log(freq[::jump])))

    def cosmicstringmodel(self, Gmu):
        """The cosmic string SGWB model."""
        if Gmu <= 0:
            return 0
        return self.cstring.OmegaGW(self.freq, Gmu)

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

    def omegamodel(self, Gmu, bbhamp):
        """Construct a model with two free parameters,
           combining multiple SGWB sources."""
        #wdgwb = self.whitedwarfmodel(self.freq, wdnumber)
        return self.cosmicstringmodel(Gmu) + self.bhbinarymerger(bbhamp)

class Likelihoods:
    """Class to perform likelihood analysis on SGWB.
    Args:
    """
    def __init__(self, nsamples=400, strings=True, binaries=True, ligo = True, satellites="lisa"):
        self.ureg = ureg
        self.strings = strings
        self.binaries = binaries

        self.cstring = CosmicStringGWB()
        self.binarybh = BinaryBHGWB()

        if isinstance(satellites, str):
            satellites = (satellites,)

        self.sensitivities = []
        if ligo:
            self.sensitivities += [LIGOSensitivity(),]
        if satellites[0] != "":
            self.sensitivities += [SatelliteSensitivity(satellite = sat) for sat in satellites]

        #This is the "true" model we are trying to detect: no cosmic strings, LIGO current best fit merger rate.
        self.trueparams = [0, 56.]
        self.experiments = [SGWBExperiment(self.binarybh, self.cstring, sens, self.trueparams, nsamples=nsamples) for sens in self.sensitivities]

        #Expected number of ligo detections at time of LISA launch for the BBH amplitude prior.
        self.nligo = 1000

    def lnlikelihood(self, params):
        """Likelihood parameters:
        0 - cosmic string tension
        1 - BH binary merger rate amplitude
        """
        #Priors: positive BBH merger rate
        if params[1] < 0:
            return -np.inf
        #CS string tension limit from EPTA
        if params[0] > np.log(2.e-11):
            return -np.inf
        #Prevent underflow
        if params[0] < -80:
            return -np.inf
        #LIGO prior: Gaussian on BBH merger rate with central value of the true value.
        ampprior = -1 * (params[1] - self.trueparams[1])**2 / self.nligo

        like = 0

        for exp in self.experiments:
            model = exp.omegamodel(np.exp(params[0]), params[1])
            like += - 1 * np.trapz(((model - exp.mockdata)/ exp.psd)**2, x=exp.freq)
            like += ampprior * np.size(exp.freq)
        #print(np.exp(params[0]), like)
        return like

    def do_sampling(self, savefile, nwalkers=100, burnin=300, nsamples = 300, while_loop=True, maxsample=200):
        """Do the sampling with emcee"""
        #Limits
        #Say Gmu ranges from exp(-45) - exp(-14) and merger rate between 0 and 100.
        pr = np.array([30, 100])
        #Priors are assumed to be in the middle.
        cent = np.array([-30, 55])
        p0 = [cent+2*pr/16.*np.random.rand(2)-pr/16. for _ in range(nwalkers)]
        assert np.all([np.isfinite(self.lnlikelihood(pp)) for pp in p0])
        emcee_sampler = emcee.EnsembleSampler(nwalkers, 2, self.lnlikelihood)
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
        self.Ajith_template = np.loadtxt("dEdfs_FULL_fobs_dndM_2p35_mmin_3msun_mmax_100msun_Ajith_spectrum.dat")
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

    def dEdfsMergV2(self, m1, m2, femit):
        """Energy per unit strain in merger phase"""
        return 1./3.*(math.pi**2*self.GG**2)**(1/3)*((m1*m2)*self.ms**(5./3)/(m1 + m2)**(1/3))*femit**(2./3)/self.fmergerV2(m1 + m2)

    def dEdfsInsp(self, m1, m2, femit):
        """Energy per unit strain in inspiral phase"""
        return 1./3.*(math.pi**2*self.GG**2/femit)**(1/3)*self.ms**(5./3)*((m1*m2)/((m1 + m2))**(1/3))

    def fqnrV2(self, msum):
        """Ringdown phase frequency."""
        fqnr = 0.915*(self.cc**3*(1. - 0.63*(1. - 0.67)**(3/10.)))/(2*math.pi*self.GG*msum*self.ms)
        return fqnr

    def dEdfstot(self, femit, alpha=-2.3):
        """Total energy per unit strain in all phases.
           This is computed with a weighted sum of a power law, following 1903.02886."""
        nsamp = 100
        m1 = np.linspace(5, 50, nsamp)
        weights = m1**alpha
        weight = np.trapz(y=weights, x=m1)*(50-5)
        #Uniform distribution given first mass.
        m2 = np.linspace(5, 50, nsamp)
        m1rep = np.repeat(m1, nsamp)
        m2rep = np.tile(m2, nsamp)
        fmerg = self.fmergerV2(m1rep + m2rep)
        fqnr = self.fqnrV2(m1rep + m2rep)
        weights = np.repeat(weights, nsamp) / weight
        ii = np.where(femit > fqnr)
        weights[ii] = 0
        ii = np.where((femit < fqnr)*(femit > fmerg))
        weights[ii] *= self.dEdfsMergV2(m1rep[ii], m2rep[ii], femit)
        ii = np.where(femit < fmerg)
        weights[ii] *= self.dEdfsInsp(m1rep[ii], m2rep[ii], femit)
        return np.trapz([np.trapz(weights[i:i+nsamp], m2) for i in range(nsamp)], m1)

    def _omegagwz(self, ff, dEdfstot, fmax):
        """Integrand as a function of redshift, taking care of the redshifting factors."""
        #From z= 0.01 to 10.
        lff = np.log(ff)
        zmax = 20
        if fmax < ff:
            return 0
        if fmax < ff * zmax:
            zmax = fmax / ff * 0.98
        Rsfr = lambda zzp1 : self.Rsfrnormless(zzp1) / HubbleEz(zzp1)
        omz = lambda lzp1 : Rsfr(np.exp(lzp1)) * np.exp(dEdfstot(lff + lzp1))
        omegagwz, _ = scipy.integrate.quad(omz, np.log(1), np.log(zmax), limit=100)
        return omegagwz

    def OmegaGW(self, freq, Norm=56., alpha=-2.3):
        """OmegaGW as a function of frequency. Normalization is in units of mergers per Gpc^3 per year. alpha is the power law of the mass distribution assumed. We are reasonably insensitive to this, so we do not vary it in the chain."""
        Hub = 67.9 * self.ureg("km/s/Mpc")
        lnewlf = np.log10(freq[0]/1.01)
        lnewhf = np.log10(freq[-1]*110)
        nsamp = (lnewhf - lnewlf) * 6
        fextended = np.logspace(lnewlf, lnewhf, nsamp)
        dEdfstot = np.array([self.dEdfstot(ff, alpha=alpha) for ff in fextended])
        ii = np.where(dEdfstot == 0)
        if np.size(ii) > 0:
            fmax = fextended[ii][0]
        else:
            fmax = fextended[-1] * 10000
        #This must be linear or the integration gives the wrong answer!
        dEdfstot_intp = scipy.interpolate.interp1d(np.log(fextended), np.log(dEdfstot+1e-99), kind='linear')
        omegagw_unnormed = np.array([self._omegagwz(ff, dEdfstot_intp, fmax) for ff in freq])
        #See eq. 2 of 1609.03565
        freq = freq * self.ureg("Hz")
        normfac = (Norm * self.Normunit / Hub * freq / self.ureg("speed_of_light")**2 / self.rhocrit())
        normfac = normfac.to_base_units()
        #assert normfac.check("[]")
        return  normfac.magnitude * omegagw_unnormed

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
        nsamp = (lnewhf - lnewlf) * 6
        fextended = np.logspace(lnewlf, lnewhf, nsamp)
        omegagmk1 = np.array([self.OmegaGWMk(Gmu, ff, 1) for ff in fextended])
        omintp = scipy.interpolate.interp1d(np.log(fextended), np.log(omegagmk1))
        P4R = np.array([np.sum([self.Gammak(k) * np.exp(omintp(np.log(ff/k))) for k in range(1,31)]) for ff in freq])/self.Gammak(1)
        #P4R = np.array([np.sum([self.OmegaGWMk(Gmu, ff, k) for k in range(1,30)]) for ff in freq])
        assert np.size(P4R) == np.size(freq)
        return P4R

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
    like = Likelihoods(nsamples=400, strings=True, binaries=True, ligo = True, satellites="lisa")
    like.do_sampling(savefile = "samples_ligo_lisa_string_bbh.txt")
