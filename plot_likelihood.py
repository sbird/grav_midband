"""Module for plotting generated likelihood chains"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import getdist as gd
import getdist.plots as gdp
matplotlib.use('PDF')

def make_plot(chainfile, savefile, true_parameter_values=None, ranges=None, string=True, burnin=10000):
    """Make a getdist plot"""
    samples = np.loadtxt(chainfile)
    ticks = {}
    if string:
        pnames = [ r"G\mu", r"\mathrm{SMBBH}", r"\mathrm{IMRI}", r"\mathrm{EMRI}"]
    else:
        pnames = [ r"T_\ast", r"\mathrm{SMBBH}", r"\mathrm{IMRI}", r"\mathrm{EMRI}", r"\alpha"]#, r"\beta"]
    prange = None
    if ranges is not None:
        prange = {pnames[i] : ranges[i] for i in range(len(pnames))}
    posterior_MCsamples = gd.MCSamples(samples=samples[burnin:], names=pnames, labels=pnames, label='', ranges=prange)

    print("Sim=",savefile)
    #Get and print the confidence limits
    for i, pn in enumerate(pnames):
        strr = pn+" 1-sigma, 2-sigma: "
        for j in (0.16, 1-0.16, 0.025, 1-0.025):
            post = posterior_MCsamples.confidence(i, j)
            if pn == r"G\mu":
                post = np.exp(post)
            strr += " %g" % post
        print(strr)
    subplot_instance = gdp.getSubplotPlotter()
    subplot_instance.triangle_plot([posterior_MCsamples], filled=True)
#     colour_array = np.array(['black', 'red', 'magenta', 'green', 'green', 'purple', 'turquoise', 'gray', 'red', 'blue'])
    #Ticks we want to show for each parameter
    if string:
        ticks = {pnames[0]: [np.log(1e-20), np.log(1e-17), np.log(1e-15)]}
        ticklabels = {pnames[0] : [r"$10^{-20}$", r"$10^{-17}$", r"$10^{-15}$"]}
    else:
        ticks = {pnames[0]: [np.log(1e2),np.log(1e4), np.log(1e6)]}#, np.log(1e11)
                 #pnames[4]: [np.log(1e-4), np.log(1e-3), np.log(1e-2), np.log(0.1), 0]}
        ticklabels = {pnames[0] : [r"$10^{2}$", r"$10^{4}$", r"$10^{6}$"]}#, r"$10^{11}$"]},
                      #pnames[4]: [r"$10^{-4}$", r"$10^{-3}$", r"$0.01$", r"$0.1$", r"$1.0$"]}

    ax = subplot_instance.subplots[0, 0]
    ax.set_visible(False)
    for pi in range(samples.shape[1]):
        for pi2 in range(pi + 1):
            #Place horizontal and vertical lines for the true point
            ax = subplot_instance.subplots[pi, pi2]
            ax.yaxis.label.set_size(16)
            ax.xaxis.label.set_size(16)
            if pi == samples.shape[1]-1 and pnames[pi2] in ticks:
                ax.set_xticks(ticks[pnames[pi2]])
                ax.set_xticklabels(ticklabels[pnames[pi2]])
            if pi2 == 0 and pnames[pi] in ticks:
                ax.set_yticks(ticks[pnames[pi]])
                ax.set_yticklabels(ticklabels[pnames[pi]])
            if not np.isnan(true_parameter_values[pi2]):
                ax.axvline(true_parameter_values[pi2], color='gray', ls='--', lw=2)
            if pi2 < pi:
                if not np.isnan(true_parameter_values[pi]):
                    ax.axhline(true_parameter_values[pi], color='gray', ls='--', lw=2)
    plt.savefig(savefile)

if __name__ == "__main__":
    #Models including a cosmo signal
    #For PT
    true_vals = [np.log(1e5), 56., 0.005, 1, 0.2]
    #ranges
    ptranges = [[np.log(100), np.log(1e7)], [0, 100], [0,1], [0.1,10], [0.001,0.8]]#, [1, 1000]]
    make_plot("samples_ligo_lisa_phase_bbh_cosmo.txt", "like_ligo_lisa_phase_bbh_cosmo.pdf", true_parameter_values = true_vals, ranges=ptranges, string=False)
    make_plot("samples_ligo_lisa_tiango_phase_bbh_cosmo.txt", "like_ligo_lisa_tiango_phase_bbh_cosmo.pdf", true_parameter_values = true_vals, ranges=ptranges, string=False)
    make_plot("samples_ligo_lisa_decigo_phase_bbh_cosmo.txt", "like_ligo_lisa_decigo_phase_bbh_cosmo.pdf", true_parameter_values = true_vals, ranges=ptranges, string=False)

    #For strings
    true_vals = [np.log(1e-16), 56., 0.005, 1]
    #ranges
    srranges = [[np.log(1e-20), np.log(2e-13)], [55.5, 56.5], [0.0048,0.0051], [0.985,1.015]]
    make_plot("samples_ligo_lisa_string_bbh_cosmo.txt", "like_ligo_lisa_string_bbh_cosmo.pdf", true_parameter_values = true_vals, ranges=srranges)
    srranges = [[np.log(1e-20), np.log(2e-13)], [55.7, 56.3], [0.00495,0.00505], [0.995,1.005]]
    make_plot("samples_ligo_lisa_tiango_string_bbh_cosmo.txt", "like_ligo_lisa_tiango_string_bbh_cosmo.pdf", true_parameter_values = true_vals, ranges=srranges)
    make_plot("samples_ligo_lisa_decigo_string_bbh_cosmo.txt", "like_ligo_lisa_decigo_string_bbh_cosmo.pdf", true_parameter_values = true_vals, ranges=srranges)

    #For PT
    true_vals = [np.nan, 56., 0.005, 1, np.nan, np.nan]
    #ranges
    ptranges = [[np.log(100), np.log(1e7)], [0, 100], [0,1], [0.1,10], [0.001,0.8]]#, [1, 1000]]
    make_plot("samples_ligo_lisa_phase_bbh.txt", "like_ligo_lisa_phase_bbh.pdf", true_parameter_values = true_vals, ranges=ptranges, string=False)
    make_plot("samples_ligo_lisa_tiango_phase_bbh.txt", "like_ligo_lisa_tiango_phase_bbh.pdf", true_parameter_values = true_vals, ranges=ptranges, string=False)
    make_plot("samples_ligo_lisa_decigo_phase_bbh.txt", "like_ligo_lisa_decigo_phase_bbh.pdf", true_parameter_values = true_vals, ranges=ptranges, string=False)

    #For strings
    true_vals = [np.nan, 56., 0.005, 1]
    #ranges
    srranges = [[np.log(1e-20), np.log(2e-13)], [55.5, 56.5], [0.0048,0.0051], [0.985,1.015]]
    make_plot("samples_ligo_lisa_string_bbh.txt", "like_ligo_lisa_string_bbh.pdf", true_parameter_values = true_vals, ranges=srranges)
    srranges = [[np.log(1e-20), np.log(2e-13)], [55.7, 56.3], [0.00495,0.00505], [0.995,1.005]]
    make_plot("samples_ligo_lisa_tiango_string_bbh.txt", "like_ligo_lisa_tiango_string_bbh.pdf", true_parameter_values = true_vals, ranges=srranges)
    make_plot("samples_ligo_lisa_decigo_string_bbh.txt", "like_ligo_lisa_decigo_string_bbh.pdf", true_parameter_values = true_vals, ranges=srranges)
