"""Short code to compute likelihood functions for stochastic gravitational wave background detection using a mid-band experiment.

TODO: fix the CS plot. NS background."""
import math
import numpy as np
import emcee
import pint
import scipy.interpolate
import scipy.integrate

ureg = pint.UnitRegistry()

def hz(zz):
    """Dimensionless part of the hubble rate"""
    return np.sqrt(0.3 * (1+zz)**3 + 0.7)

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

class LIGOO1Sensitivity(Sensitivity):
    """LIGO sensitivity curve as a function of frequency"""
    def __init__(self):
        super().__init__()
        #Power spectral density strain noise amplitude from
        #https://dcc.ligo.org/public/0149/T1800044/005/aLIGODesign.txt
        #f(Hz)     PSD (Hz^-1/2)
        self.ligoO1ff = np.loadtxt("LIGO_O1_freqVector.txt")
        self.ligoO1psd = np.loadtxt("LIGO_O1_psd.txt")

    def PSD(self):
        """Get the root power spectral density in Hz^[-1/2]"""
        #The input data is in strain power spectral density noise amplitude.
        #The (dimensionless) characteristic strain is
        #charstrain = np.sqrt(self.aligo[:,0]) * self.aligo[:,1]
        return self.ligoO1ff, self.ligoO1psd

class SGWB:
    """Class to contain SGWBs.
    Args:
    maxfreq - maximum LIGO frequency, 400 Hz
    minfreq - minimum LISA frequency, 10^-5 Hz
    """
    def __init__(self, obstime=3, maxfreq = 400, minfreq = 1e-5, nbins=1000):
        self.ureg = ureg
        self.freq = np.logspace(np.log10(minfreq), np.log10(maxfreq), nbins) * self.ureg('Hz')
        self.obstime = obstime * self.ureg('year')
        self.lisa = LISASensitivity()
        self.lisafreq, self.lisapsd = self.lisa.omegadens()
        self.ligo = LIGOSensitivity()
        self.ligofreq, self.ligopsd = self.ligo.omegadens()
        self.cstring = CosmicStringGWB()
        self.binarybh = BinaryBHGWB()

    def cosmicstringmodel(self, freq, Gmu):
        """The cosmic string SGWB model."""
        return self.cstring.OmegaGW(freq, Gmu)

    def whitedwarfmodel(self, freq, number):
        """The white dwarf background model.
        This can be neglected: alone of all the signals the WD background is
        localised in the galaxy, and this should allow it to be cleaned.
        It will be anisotropic and thus have a yearly modulation.
        """
        _ = (freq, number)
        return 0

    def bhbinarymerger(self, freq, amp):
        """The unresolved BH binary model. Using the model from
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.131102"""
        return self.binarybh.OmegaGW(freq, amp)

    def omegamodel(self, freq, Gmu, bbhamp):
        """Construct a model with two free parameters,
           combining multiple SGWB sources."""
        csgwb = self.cosmicstringmodel(freq, Gmu)
        #wdgwb = self.whitedwarfmodel(self.freq, wdnumber)
        bbhgb = self.bhbinarymerger(freq, bbhamp)
        return csgwb + bbhgb

    def lnlikelihood(self, params):
        """Likelihood parameters:
        0 - cosmic string tension
        1 - BH binary merger rate amplitude
        """
        lisamodel = self.omegamodel(self.lisafreq, params[0], params[1])
        ligomodel = self.omegamodel(self.ligofreq, params[0], params[1])
        #This is - the signal to noise ratio squared.
        like = - 1 * self.obstime * np.trapz((lisamodel / self.lisapsd)**2, x=self.freq)
        like += - 1 * self.obstime * np.trapz((ligomodel / self.ligo)**2, x=self.freq)
        return like

class BinaryBHGWB:
    """Model for the binary black hole background, fiducial version from LIGO."""
    def __init__(self):
        self.ureg = ureg
        self.Normunit = 1*self.ureg("Gpc^-3/year")
        self.b = 1.5
        self.a = 1.92
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

    def Rsfrnormless(self, z):
        """Black hole merger rate as a function of the star formation rate. Unnormalized."""
        return self.a * np.exp(self.b * (z - self.zm)) / (self.a - self.b + self.b * np.exp(self.a*(z - self.zm)))

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
        nsamp = 30
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

    def _omegagwz(self, ff):
        """Integrand as a function of redshift, taking care of the redshifting factors."""
        #From z= 0.01 to 10.
        zz = np.logspace(-2, 2, 50)
        Rsfr = self.Rsfrnormless(zz)
        omz = np.array([self.dEdfstot(ff * (1 + z)) for z in zz])
        omegagwz = np.trapz(Rsfr * omz / ((1+zz) * hz(zz)), zz)
        return omegagwz

    def OmegaGW(self, freq, Norm=56.):
        """OmegaGW as a function of frequency."""
        Hub = 67.9 * self.ureg("km/s/Mpc")
        omegagw_unnormed = np.array([self._omegagwz(ff) for ff in freq])
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
        return 207.04170721279553 + 9.297437227007177e8/x**4 - 6.762901334043625e7/x**3 + 1.5888320597630164e6/x**2 - 17782.040653778313/x - 0.3100369947620232*x + 0.0005219361630722421*x**2 - 4.5089224156018806e-7*x**3 + 1.571478658251381e-10*x**4
    if x > 19.2101:
        return 25.617077215691925 - 85049.61151267929/x**4 + 44821.06301089602/x**3 - 9407.524062731489/x**2 + 1010.3559753329803/x + 1.9046169770731767*x -  0.02182528531424682*x**2 + 0.00013717391195802956*x**3 - 4.656717468928253e-7*x**4
    if x > 0.210363:
        return 96.03236599599052 + 0.1417211377127251/x**4 - 1.9033759380298605/x**3 + 9.91065103843345/x**2 - 26.492041757131332/x - 1.923898527856908*x +  0.18435280458948589*x**2 - 0.008350062215001603*x**3 + 0.000158735905184788*x**4
    if x > 0.152771:
        return -2.0899590637819996e8 - 3113.914061747819/x**4 + 140186.13699256277/x**3 - 2.7415777741917237e6/x**2 + 3.0404304467460793e7/x + 9.110822620276868e8*x - 2.4573038140264993e9*x**2 + 3.7443023200934935e9*x**3 - 2.4636941241588144e9*x**4
    if x > 0.0367248:
        return 5426.500494710776 + 0.0019390100970067777/x**4 - 0.23020238075499902/x**3 + 11.587823813540876/x**2 - 322.51882049359654/x - 55993.18120267802*x + 349376.7985871369*x**2 - 1.2042764400287867e6*x**3 + 1.7809006388749087e6*x**4
    if x > 0.00427032:
        return -195.37500200383414 - 3.246844915516518e-8/x**4 + 0.000028390675511486528/x**3 - 0.010121020979848003/x**2 + 1.9052124373133226/x + 13012.873468646329*x - 473484.5788624826*x**2 + 9.244933716566762e6*x**3 - 7.352186782452533e7*x**4
    if x > 0.000163046:
        return 6.995576000906495 - 5.271793610584592e-15/x**4 + 1.1471366111587497e-10/x**3 - 9.017840481639097e-7/x**2 + 0.0022702982380190806/x + 2706.0190947576525*x - 1.1981368143068189e6*x**2 + 2.6230001478632784e8*x**3 - 2.1859643005333782e10*x**4
    if x > 0.0000110255:
        return 1.3910688031037668 - 1.0512592071931046e-19/x**4 + 2.671929070189596e-14/x**3 - 2.5014060168690585e-9/x**2 + 0.00010456662194366741/x + 4706.80818171218*x + 6.752939034535752e6*x**2 + 2.531715178089083e12*x**3 - 1.05933524939722e16*x**4
    return 3.17578

#Here comes the cosmic string model
class CosmicStringGWB:
    """Model for the gravitational wave background from cosmic strings. From Yanou's Mathematica notebook and 1808.08968."""
    def __init__(self):
        self.hub = (0.7 * 100 * ureg("km/s/Mpc").to_base_units()).magnitude
        self.ureg = ureg
        self.HzoverGev = (1*self.ureg.planck_constant/2/math.pi).to('s*GeV')
        self.CeffM = 0.5
        self.CeffR = 5.7
        self.alpha = 0.1
        self.Fa = 0.1
        self.Gamma = 50.
        #self.t0 = (13.8e9 * self.ureg('years')).to('s') / HzoverGev
        self.aaeq = 1/(1 + 3360)
        self.amin = 1e-20
        self.amax = 1e-5
        self.aatab = np.logspace(np.log10(self.amin), np.log10(self.amax), 300)
        #Must be first
        self.gcorrtab = [self.gcorr(a) for a in self.aatab]
        self.gcorrintp = scipy.interpolate.interp1d(np.log(self.aatab), np.log(self.gcorrtab))
        self.tttab = [self.ttint(a) for a in self.aatab]
        self.ttintp = scipy.interpolate.interp1d(np.log(self.aatab), self.tttab)
        self.aaintp = scipy.interpolate.interp1d(self.tttab, np.log(self.aatab))

    def OmegaGW(self, freq, Gmu):
        """SGWB power (omega_gw) for a string forming during radiation domination."""
        P3R = [np.sum([self.OmegaGWMk(Gmu, ff, k) for k in range(1,30)]) for ff in freq]
        assert np.size(P3R) == np.size(freq)
        return P3R

    def Ceff(self, aatik):
        """Effective string decay parameter"""
        if aatik > self.aaeq:
            return self.CeffM
        return self.CeffR

    def ttint(self, aa):
        """Time in s between beginning and scale factor a."""
        integ = lambda aa: 1/(aa * self.Hubble(aa)) #.to('s')).magnitude
        return scipy.integrate.romberg(integ, self.amin,aa, divmax=100)

    def time(self, aa):
        """Get the time at scale factor a (pre-computed) in s"""
        return self.ttintp(np.log(aa))

    def tik(self, aa, Gmu, freq, k):
        """Formation time of loops with mode number k in s."""
        tt = self.time(aa)
        return (self.Gamma * Gmu * tt + 2 * k / freq * aa) / (self.alpha + self.Gamma * Gmu)

    def gcorr(self, a):
        """Correction to radiation density before species freeze out"""
        #This should be a root of a T / T0 = (g(T)/g(T0))**(1/3)
        T0 = 2.72556666/(1.16045e13) #CMB temp in GeV
        func = lambda T: a * T * (gcorr(T)/gcorr(T0))**(1./3) - T0
        sol = scipy.optimize.root_scalar(func, x0=T0/a, x1=T0/a*100)
        Ta = sol.root
        return gcorr(Ta)/gcorr(T0)

    def Hubble(self, a):
        """Hubble expansion rate"""
        return self.hub * np.sqrt(0.7 + 0.3/a**3 + np.exp(self.gcorrintp(np.log(a)))**(-1/3) * 8.5744e-5/a**4)

    def OmegaGWMkintegrand(self, aa, Gmu, freq, k):
        """Integrand from eq. 2.14"""
        aat0 = 1
        tik = self.tik(aa/aat0, Gmu, freq, k) # * self.ureg("s")
        aatik = np.exp(self.aaintp(tik))
        #We can neglect the heaviside function for tF because tF = 0.
        #Units of s^-4
        return self.Ceff(aatik) / tik**4 * (aa/aat0)**5 * (aatik/aa)**3 * (aa > aatik)

    def rhocrit(self):
        """Critical energy density at z=0 in kg/m^3 * G"""
        rhoc = 3 * (self.hub)**2 #* self.ureg("1/s"))**2
        return rhoc / 8 / math.pi #/ self.ureg.newtonian_constant_of_gravitation

    def Gammak(self, k):
        """Mode-dependent Gamma"""
        return self.Gamma * k**(-4./3) / 3.6

    def OmegaGWMk(self, Gmu, freq, k):
        """Eq. 2.14 of 1808.08968"""
        #GG = self.ureg.newtonian_constant_of_gravitation
        #Units of s^3
        prefac = (2 * k / freq / self.rhocrit()) * self.Fa * self.Gammak(k) * Gmu**2 / (self.alpha * (self.alpha + self.Gamma * Gmu))
        #Multiply by a jacobian factor dt/da = 1/da/dt = 1/(aH) so we can integrate da
        omint = lambda aa: (prefac * self.OmegaGWMkintegrand(aa, Gmu, freq, k) / (aa * self.Hubble(aa)))
        #Check dimensionless
        #assert omint(0.5).check("[]")
        #omintmag = lambda aa: omint(aa).magnitude
        #We split this integral into three pieces depending on the expansion time
        OmegaGW,_ = scipy.integrate.quad(omint, 1.1*self.amin, 0.9*self.amax)
        return OmegaGW
