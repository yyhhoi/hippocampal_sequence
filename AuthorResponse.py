from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import cm
from pycircstat import vtest
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import binom_test, linregress, spearmanr, chisquare

from common.linear_circular_r import rcc
from common.stattests import p2str, stat_record, my_chi_2way, my_kruskal_2samp, my_kruskal_3samp, \
    fdr_bh_correction, my_ww_2samp, my_fisher_2way, circ_ktest
from common.comput_utils import linear_circular_gauss_density, unfold_binning_2d, midedges, \
    shiftcyc_full2half, repeat_arr, get_numpass_at_angle

from common.visualization import customlegend, plot_marginal_slices

from common.script_wrappers import DirectionalityStatsByThresh, compute_precessangle
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw



def plot_lowspikepass_dist(df, save_dir):
    nspikes_stats = {'CA1': 6, 'CA2': 6, 'CA3': 7}

    adiff_dict = dict()
    for caid, (ca, cadf) in enumerate(df.groupby('ca')):
        cadf = cadf.reset_index(drop=True)
        nsp_thresh = nspikes_stats[ca]
        alladiff_list = []
        for rowi in range(cadf.shape[0]):

            rate_angle, precess_df = df.loc[rowi, ['rate_angle', 'precess_df']]


            lowpass_df = precess_df[precess_df.pass_nspikes < nsp_thresh].reset_index(drop=True)


            adiff = cdiff(rate_angle, lowpass_df.mean_anglesp.to_numpy())
            alladiff_list.append(adiff)


        alladiff = np.concatenate(alladiff_list)
        adiff_dict[ca] = alladiff

    fig, ax = plt.subplots(1, 3, figsize=(10, 3), subplot_kw={'projection': 'polar'}, sharex=True, sharey=True)

    fig2, ax2 = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
        bins, edges = np.histogram(adiff_dict[ca], bins=36)
        normbins = bins/np.sum(bins)
        ax[caid].step(midedges(edges), normbins)
        ax[caid].set_ylim(0, normbins.max()*1.1)
        ax[caid].set_yticks([])
        #     ax[caid].set_xticks([0, np.pi])
        ax[caid].set_title('%s'%(ca))

        close = np.sum(np.abs(adiff_dict[ca]) < (np.pi/4))
        far = np.sum(np.abs(adiff_dict[ca]) > (np.pi*3/4))

        _, p = chisquare([close, far])
        ax2[caid].bar([0, 1], [close, far], width=0.5)

        ax2[caid].set_title('One-way Chisquare test\np=%0.4f'%(p))
        ax2[caid].set_xticks([0, 1])

        ax2[caid].set_xticklabels([r'$<\pm45^\circ$'+ '\nfrom best angle', r'$>\pm135^\circ$' + '\nfrom best angle'])
    ax[0].set_ylabel('Normalized counts\nof low-spike passes', labelpad=30)
    ax2[0].set_ylabel('Low-spike pass counts')
    fig.tight_layout()
    fig2.tight_layout()

    fig.savefig(join(save_dir, 'lowspikepass_alldirect.eps'))
    fig.savefig(join(save_dir, 'lowspikepass_alldirect.png'), dpi=dpi)

    fig2.savefig(join(save_dir, 'lowspikepass_2direct.eps'))
    fig2.savefig(join(save_dir, 'lowspikepass_2direct.png'), dpi=dpi)


if __name__ == '__main__':
    data_pth = 'results/emankin/singlefield_df.pickle'
    df = pd.read_pickle(data_pth)
    save_dir = 'result_plots/author_response'
    os.makedirs(save_dir, exist_ok=True)
    plot_lowspikepass_dist(df, save_dir)