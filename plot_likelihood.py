"""Module for plotting generated likelihood chains"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import getdist as gd
import getdist.plots as gdp
matplotlib.use('PDF')

def make_plot(chainfile, savefile, true_parameter_values=None, ranges=None):
    """Make a getdist plot"""
    samples = np.loadtxt(chainfile)
    ticks = {}
    pnames = [ r"G\mu", r"BBH rate"]
    #Ticks we want to show for each parameter
    #ticks = {pnames[0]: [1.5, 2.0, 2.5], pnames[1]: [-0.6,-0.3, 0.]}
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

    for pi in range(samples.shape[1]):
        for pi2 in range(pi + 1):
            #Place horizontal and vertical lines for the true point
            ax = subplot_instance.subplots[pi, pi2]
            ax.yaxis.label.set_size(16)
            ax.xaxis.label.set_size(16)
            if pi == samples.shape[1]-1 and pnames[pi2] in ticks:
                ax.set_xticks(ticks[pnames[pi2]])
            if pi2 == 0 and pnames[pi] in ticks:
                ax.set_yticks(ticks[pnames[pi]])
            ax.axvline(true_parameter_values[pi2], color='gray', ls='--', lw=2)
            if pi2 < pi:
                ax.axhline(true_parameter_values[pi], color='gray', ls='--', lw=2)
    plt.savefig(savefile)

if __name__ == "__main__":
    true_vals = [np.log(1e-80), 56]
    #ranges
    ranges = [[np.log(1e-80), np.log(2e-11)], [0, 100]]
    make_plot("samples_ligo_lisa_string_bbh.txt", "like_ligo_lisa_string_bbh.pdf", true_parameter_values = true_vals, ranges=ranges)
