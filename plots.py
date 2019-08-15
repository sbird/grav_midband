"""Make some useful plots"""
import math
import numpy as np
import pint
import matplotlib.pyplot as plt

import gravmidband

def make_sgwb_plot():
    """Plot an example stochastic gravitational wave background"""
    lisa = gravmidband.LISASensitivity()
    ligo = gravmidband.LIGOSensitivity()
    saff, sapo = lisa.omegadens()
    goff, gopo = ligo.omegadens()
    plt.loglog(saff, sapo, "--", color="green", label="LISA")
    plt.loglog(goff, gopo, "-", color="black", label="LIGO")

    freqs = np.logspace(-7, 4, 200)

    csgw = gravmidband.CosmicStringGWB()
    omegacs = csgw.OmegaGW(freqs, 1.e-11)
    plt.loglog(freqs, omegacs, "-", color="blue", label="CS: $G\mu = 10^{-11}$")

    bbh = gravmidband.BinaryBHGWB()
    omegabbh = bbh.OmegaGW(freqs)
    plt.loglog(freqs, omegabbh, ":", color="red", label="Binary Black holes")

    plt.legend(loc="upper left")
    plt.xlabel("f (Hz)")
    plt.ylabel("$\Omega_{GW}$")
