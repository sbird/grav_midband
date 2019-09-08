"""Make some useful plots"""
import math
import numpy as np
import pint
import matplotlib.pyplot as plt
import matplotlib
import gravmidband

matplotlib.use("PDF")

def make_sgwb_plot():
    """Plot an example stochastic gravitational wave background"""
    ligo = gravmidband.LIGOSensitivity()
    #lisa = gravmidband.LISASensitivity()
    #saff, sapo = lisa.omegadens()
    goff, gopo = ligo.omegadens()
    #plt.loglog(saff, sapo, "-", color="green", label="LISA")
    plt.loglog(goff, gopo, "-", color="black", label="LIGO")

    for sat in ("lisa", "tianqin", "bdecigo"):
        ss = gravmidband.SatelliteSensitivity(satellite = sat)
        sff, spo = ss.omegadens()
        plt.loglog(sff, spo, "--", label=sat)

    freqs = np.logspace(-7, 4, 200)

    csgw = gravmidband.CosmicStringGWB()
    omegacs = csgw.OmegaGW(freqs, 1.e-12)
    plt.loglog(freqs, omegacs, "-.", color="blue", label="CS: $G\mu = 10^{-12}$")

    bbh = gravmidband.BinaryBHGWB()
    omegabbh = bbh.OmegaGW(freqs)
    plt.loglog(freqs, omegabbh, ":", color="red", label="Binary Black holes")

    plt.legend(loc="upper left")
    plt.xlabel("f (Hz)")
    plt.ylabel("$\Omega_{GW}$")
    plt.ylim(1e-20, 1)
    plt.tight_layout()
    plt.savefig("sgwb.pdf")

if __name__ == "__main__":
    make_sgwb_plot()
