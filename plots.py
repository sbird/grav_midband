"""Make some useful plots"""
import math
import numpy as np
import pint
import matplotlib.pyplot as plt
import matplotlib
import gravmidband
import json

#matplotlib.use("PDF")

def sensitivity():
    """Make non-modified sens. plots"""
    #These contain characteristic strain, NOT power spectral density.
    for js in ("DECIGO.json", "LISA.json", "TianQin.json"):
        ff = open(js)
        jsdata = json.load(ff)
        data = np.array(jsdata["data"])
        plt.loglog(data[:,0], data[:,1]/np.sqrt(data[:,0]), label=js)
    lisa = gravmidband.LISASensitivity()
    plt.loglog(lisa.lisa[:,0], lisa.lisa[:,1], label="LISA MakeCurve")

    for sat in ("lisa", "tianqin", "bdecigo"):
        ss = gravmidband.SatelliteSensitivity(satellite = sat)
        sff, spo = ss.PSD()
        plt.loglog(sff, spo, "--", label=sat)
    plt.legend()


def make_sgwb_plot():
    """Plot an example stochastic gravitational wave background"""
    ligo = gravmidband.LIGOSensitivity()
    #lisa = gravmidband.LISASensitivity()
    #saff, sapo = lisa.omegadens()
    goff, gopo = ligo.omegadens()
    #plt.loglog(saff, sapo, "-", color="green", label="LISA")
    plt.loglog(goff, gopo, "-", color="black", label="LIGO")

    for sat in ("lisa", "tianqin", "tiango", "bdecigo"):
        ss = gravmidband.SatelliteSensitivity(satellite = sat)
        sff, spo = ss.omegadens()
        plt.loglog(sff, spo, "--", label=sat)

    freqs = np.logspace(-7, 4, 200)

    csgw = gravmidband.CosmicStringGWB()
    omegacs = csgw.OmegaGW(freqs, 1.e-14)
    plt.loglog(freqs, omegacs, "-.", color="blue", label="CS: $G\mu = 10^{-12}$")

    bbh = gravmidband.BinaryBHGWB()
    omegabbh = bbh.OmegaGW(freqs)
    plt.loglog(freqs, omegabbh, ":", color="red", label="Binary Black holes")

    emri = gravmidband.EMRIGWB()
    omegaemri = emri.OmegaGW(freqs)
    plt.loglog(freqs, omegaemri, ":", color="gold", label="EMRI mergers")

    imri = gravmidband.IMRIGWB()
    omegaimri = imri.OmegaGW(freqs)
    plt.loglog(freqs, omegaimri, ":", color="purple", label="IMRI mergers")

    plt.legend(loc="upper left")
    plt.xlabel("f (Hz)")
    plt.ylabel("$\Omega_{GW}$")
    plt.ylim(1e-20, 1)
    plt.tight_layout()
    plt.savefig("sgwb.pdf")

if __name__ == "__main__":
    make_sgwb_plot()
