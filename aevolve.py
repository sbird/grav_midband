"""Work out the relation between scale factor and time
"""
import numpy as np
import scipy.optimize

def HubbleEz(zzp1):
    """Dimensionless part of the hubble rate. Argument is 1+z"""
    return np.sqrt(0.3 * zzp1**3 + 0.7)

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

class Evolve:
    """Scale factor to time mapping"""
    def __init__(self):
        #Planck mass in GeV/c^2
        self.Mp = 1.220910e19
        #CMB temperature in GeV
        self.T0 = 2.725566 / 1.16045e13
        self.OmegaM = 0.3
        self.OmegaR = 5e-5

    def Hubble(self, T):
        """Hubble rate at high temperatures"""
        return 1.66 * np.sqrt(gcorr(T)) * T**2 / self.Mp

    def Ta(self, a):
        """Find effective time"""
        TTa = lambda T: a * T * ( gcorr(T) / gcorr(self.T0) )**(1./3) - self.T0
        #ie, if the radiation all dropped out already and gcorr(T) = gcorr(T0)
        if self.T0/a < 0.0000110255:
            return self.T0 / a
        x0 = self.T0/a * ( gcorr(self.T0/a) / gcorr(self.T0) )**(-1./3)
        x1 = self.T0/a
        sol = scipy.optimize.root_scalar(TTa, x0=x0, x1=x1)
        return sol.root

    def rhoR(self, a):
        """Rho R"""
        Ta = self.Ta(a)
        return ( gcorr(Ta) / gcorr(self.T0) )**(1./3)

    def tt(self, a):
        """Find t as a function of a"""
        integ = lambda aa: aa /np.sqrt(self.OmegaR * self.RhoR(a) + self.OmegaM * aa + (1 - self.OmegaM)*aa**4)
        return scipy.integrate.quad(integ, 0, a)/scipy.integrate.quad(integ, 0, 1)
