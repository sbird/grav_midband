"""Make some useful plots"""
import json
import numpy as np
import matplotlib.pyplot as plt
import gravmidband

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

    for sat in ("lisa", "tiango", "bdecigo"):
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

    for sat in ("lisa", "tiango", "bdecigo"):
        ss = gravmidband.SatelliteSensitivity(satellite = sat)
        sff, spo = ss.omegadens()
        plt.loglog(sff, spo, "--", label=sat)

    freqs = np.logspace(-7, 4, 50)

    csgw = gravmidband.CosmicStringGWB()
    omegacs = csgw.OmegaGW(freqs, Gmu=1.e-16)
    plt.loglog(freqs, omegacs, "-.", color="blue", label=r"CS: $G\mu = 10^{-16}$")

    bbh = gravmidband.BinaryBHGWB()
    omegabbh = bbh.OmegaGW(freqs)
    plt.loglog(freqs, omegabbh, ":", color="red", label="Binary Black holes")

    #emri = gravmidband.EMRIGWB()
    #omegaemri = emri.OmegaGW(freqs)
    #plt.loglog(freqs, omegaemri, ":", color="gold", label="EMRI mergers")

    imri = gravmidband.IMRIGWB()
    omegaimri = imri.OmegaGW(freqs)
    plt.loglog(freqs, omegaimri, ":", color="purple", label="IMRI mergers")

    plt.legend(loc="upper left")
    plt.xlabel("f (Hz)")
    plt.ylabel(r"$\Omega_{GW}$")
    plt.ylim(1e-20, 1)
    plt.tight_layout()
    plt.savefig("sgwb.pdf")


def make_string_plot():
    """Plot stochastic gravitational wave backgrounds from cosmic strings."""
    ligo = gravmidband.LIGOSensitivity()
    #lisa = gravmidband.LISASensitivity()
    #saff, sapo = lisa.omegadens()
    goff, gopo = ligo.omegadens()
    #plt.loglog(saff, sapo, "-", color="green", label="LISA")
    plt.loglog(goff, gopo, "-", color="black", label="LIGO")

    for sat in ("lisa", "tiango", "bdecigo"):
        ss = gravmidband.SatelliteSensitivity(satellite = sat)
        sff, spo = ss.omegadens()
        plt.loglog(sff, spo, "--", label=sat)

    freqs = np.logspace(-7, 4, 50)

    csgw = gravmidband.CosmicStringGWB()
    omegacs = csgw.OmegaGW(freqs, Gmu=1.e-16)
    plt.loglog(freqs, omegacs, "-.", color="blue", label=r"CS: $G\mu = 10^{-16}$")

    omegacs = csgw.OmegaGW(freqs, Gmu=1.e-17)
    plt.loglog(freqs, omegacs, "--", color="red", label=r"CS: $G\mu = 10^{-17}$")

    omegacs = csgw.OmegaGW(freqs, Gmu=1.e-18)
    plt.loglog(freqs, omegacs, ":", color="grey", label=r"CS: $G\mu = 10^{-18}$")

    omegacs = csgw.OmegaGW(freqs, Gmu=1.e-19)
    plt.loglog(freqs, omegacs, "-", color="brown", label=r"CS: $G\mu = 10^{-19}$")

    #emri = gravmidband.EMRIGWB()
    #omegaemri = emri.OmegaGW(freqs)
    #plt.loglog(freqs, omegaemri, ":", color="gold", label="EMRI mergers")

    plt.legend(loc="upper left")
    plt.xlabel("f (Hz)")
    plt.ylabel(r"$\Omega_{GW}$")
    plt.ylim(1e-20, 1)
    plt.tight_layout()
    plt.savefig("strings.pdf")


def make_pt_plot():
    """Plot stochastic gravitational wave backgrounds from a phase transition."""
    #ligo = gravmidband.LIGOSensitivity()
    #lisa = gravmidband.LISASensitivity()
    #saff, sapo = lisa.omegadens()
    #goff, gopo = ligo.omegadens()
    #plt.loglog(saff, sapo, "-", color="green", label="LISA")
    #plt.loglog(goff, gopo, "-", color="black", label="LIGO")

    for sat in ("lisa", "tiango"):
        ss = gravmidband.SatelliteSensitivity(satellite = sat)
        sff, spo = ss.omegadens()
        plt.loglog(sff, spo, "--", label=sat)

    freqs = np.logspace(-7, 4, 50)

    csgw = gravmidband.PhaseTransition()
    omegacs = csgw.OmegaGW(freqs, Ts=1, alpha=0.3)
    plt.loglog(freqs, omegacs, "-.", color="blue", label=r"PT: $T_* = 1 \;\mathrm{GeV}$")

    #omegacs = csgw.OmegaGW(freqs, Ts=100)
    #plt.loglog(freqs, omegacs, "--", color="red", label=r"PT: $T_* = 100 \;\mathrm{GeV}$")

    #omegacs = csgw.OmegaGW(freqs, Ts=0.1)
    #plt.loglog(freqs, omegacs, ":", color="grey", label=r"PT: $T_* = 10^{-1} \;\mathrm{GeV}$")

    omegacs = csgw.OmegaGW(freqs, Ts=1e4, alpha=0.3)
    plt.loglog(freqs, omegacs, "-", color="brown", label=r"PT: $T_* = 10^{4} \;\mathrm{GeV}$")

    omegacs = csgw.OmegaGW(freqs, Ts=1e4, alpha=0.1)
    plt.loglog(freqs, omegacs, ":", color="pink", label=r"PT: $\alpha=0.1 T_* = 10^{4} \;\mathrm{GeV}$ ")

    #omegacs = csgw.OmegaGW(freqs, Ts=1e4, alpha=10)
    #plt.loglog(freqs, omegacs, "-", color="orange", label=r"PT: $\alpha=10 T_* = 10^{4} \;\mathrm{GeV}$")

    #omegacs = csgw.OmegaGW(freqs, Ts=1e5, alpha=0.3)
    #plt.loglog(freqs, omegacs, "-", color="green", label=r"PT: $T_* = 10^{5} \;\mathrm{GeV}$")

    #omegacs = csgw.OmegaGW(freqs, Ts=1e6, alpha=1.5)
    #plt.loglog(freqs, omegacs, "-", color="green", label=r"PT: $T_* = 10^{6} \;\mathrm{GeV}$")

    #emri = gravmidband.EMRIGWB()
    #omegaemri = emri.OmegaGW(freqs)
    #plt.loglog(freqs, omegaemri, ":", color="gold", label="EMRI mergers")

    plt.legend(loc="upper left", ncol=2)
    plt.xlabel("f (Hz)")
    plt.ylabel(r"$\Omega_{GW}$")
    plt.ylim(1e-20, 100)
    plt.tight_layout()
    plt.savefig("phasetransition.pdf")

if __name__ == "__main__":
    make_pt_plot()
    plt.clf()
    make_string_plot()
    plt.clf()
    make_sgwb_plot()
    plt.clf()
