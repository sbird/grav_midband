"""Module for plotting generated likelihood chains"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import getdist as gd
import getdist.plots as gdp
matplotlib.use('PDF')

def make_plot(chainfile, savefile, true_parameter_values=None, ranges=None, string=True):
    """Make a getdist plot"""
    samples = np.loadtxt(chainfile)
    ticks = {}
    if string:
        pnames = [ r"G\mu", r"BBH rate", r"IMBH rate"]
    else:
        pnames = [ r"T_s", r"BBH rate", r"IMBH rate", r"\alpha"]
    prange = None
    if ranges is not None:
        prange = {pnames[i] : ranges[i] for i in range(len(pnames))}
    posterior_MCsamples = gd.MCSamples(samples=samples, names=pnames, labels=pnames, label='', ranges=prange)

    print("Sim=",savefile)
    #Get and print the confidence limits
    for i, pn in enumerate(pnames):
        strr = pn+" 1-sigma, 2-sigma: "
        for j in (0.16, 1-0.16, 0.025, 1-0.025):
            strr += str(round(posterior_MCsamples.confidence(i, j),5)) + " "
        print(strr)
    subplot_instance = gdp.getSubplotPlotter()
    subplot_instance.triangle_plot([posterior_MCsamples], filled=True)
#     colour_array = np.array(['black', 'red', 'magenta', 'green', 'green', 'purple', 'turquoise', 'gray', 'red', 'blue'])
    #Ticks we want to show for each parameter
    if string:
        ticks = {pnames[0]: [np.log(1e-20), np.log(1e-17), np.log(1e-15)]}
        ticklabels = {pnames[0] : [r"$10^{-20}$", r"$10^{-17}$", r"$10^{-15}$"]}
    #else:
        #ticks = {pnames[0]: [np.log(1e-20), np.log(1e-17), np.log(1e-15)]}
        #ticklabels = {pnames[0] : [r"$10^{-20}$", r"$10^{-17}$", r"$10^{-15}$"]}
    for pi in range(samples.shape[1]):
        for pi2 in range(pi + 1):
            #Place horizontal and vertical lines for the true point
            ax = subplot_instance.subplots[pi, pi2]
            ax.yaxis.label.set_size(16)
            ax.xaxis.label.set_size(16)
            if pi == samples.shape[1]-1 and pnames[pi2] in ticks:
                ax.set_xticks(ticks[pnames[pi2]])
                ax.set_xticklabels(ticklabels[pnames[pi2]])
            ax.axvline(true_parameter_values[pi2], color='gray', ls='--', lw=2)
            if pi2 < pi:
                ax.axhline(true_parameter_values[pi], color='gray', ls='--', lw=2)
    plt.savefig(savefile)

if __name__ == "__main__":
    #For PT
    true_vals = [0, 56., 0.01, 0]
    #ranges
    ranges = [[-1, 6], [0, 100], [0,1], [1e-6,2]]
    make_plot("samples_ligo_lisa_phase_bbh.txt", "like_ligo_lisa_phase_bbh.pdf", true_parameter_values = true_vals, ranges=ranges, string=False)
    make_plot("samples_ligo_lisa_tiango_phase_bbh.txt", "like_ligo_lisa_tiango_phase_bbh.pdf", true_parameter_values = true_vals, ranges=ranges, string=False)

    #For strings
    true_vals = [0, 56., 0.01]
    #ranges
    ranges = [[-50, np.log(2e-13)], [0, 100], [0,1]]
    make_plot("samples_ligo_lisa_string_bbh.txt", "like_ligo_lisa_string_bbh.pdf", true_parameter_values = true_vals, ranges=ranges)
    make_plot("samples_ligo_lisa_tiango_string_bbh.txt", "like_ligo_lisa_tiango_string_bbh.pdf", true_parameter_values = true_vals, ranges=ranges)
