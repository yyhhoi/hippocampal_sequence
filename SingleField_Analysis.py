# This script does directionality and spike phase analysis for single fields.
# Only for Emily's data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d, make_interp_spline
from pycircstat.descriptive import resultant_vector_length, cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import ranksums, chi2_contingency

from common.utils import load_pickle, stat_record
from common.comput_utils import check_border, normalize_distr, TunningAnalyzer, midedges, segment_passes, window_shuffle, \
    compute_straightness, heading, pair_diff, check_border_sim, append_info_from_passes, get_field_directionality, \
    window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, DirectionerBining, DirectionerMLM, \
    window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, circular_density_1d

from common.visualization import color_wheel, directionality_polar_plot, customlegend

from common.script_wrappers import DirectionalityStatsByThresh
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

figext = 'png'
# figext = 'eps'

def omniplot_singlefields(expdf, save_dir=None):

    figl = total_figw/5
    linew = 0.75

    # # Initialization
    stat_fn = 'fig1_single_directionality.txt'
    stat_record(stat_fn, True)
    fig, ax = plt.subplots(2, 3, figsize=(total_figw*(2.5/4), figl*2), sharey='row', sharex='col')
    fig_frac, ax_frac = plt.subplots(2, 1, figsize=(total_figw*(1.5/4), figl*2), sharex=True)
    spike_threshs = np.arange(0, 801, 50)
    stats_getter = DirectionalityStatsByThresh('num_spikes', 'win_pval_mlm', 'shift_pval_mlm', 'fieldR_mlm')

    # # Plot
    data_dict = {'CA%d'%(i):dict() for i in range(1, 4)}
    for caid, (ca, cadf) in enumerate(expdf.groupby('ca')):

        # Get data/stats
        cadf_b = cadf[cadf['border']].reset_index(drop=True)
        cadf_nb = cadf[~cadf['border']].reset_index(drop=True)
        sdict_all = stats_getter.gen_directionality_stats_by_thresh(cadf, spike_threshs)
        sdict_b = stats_getter.gen_directionality_stats_by_thresh(cadf_b, spike_threshs)
        sdict_nb = stats_getter.gen_directionality_stats_by_thresh(cadf_nb, spike_threshs)
        data_dict[ca]['all'] = sdict_all
        data_dict[ca]['border'] = sdict_b
        data_dict[ca]['nonborder'] = sdict_nb

        # Plot
        ax[0, caid].plot(spike_threshs, sdict_all['medianR'], c=ca_c[ca], label='All', linewidth=linew)
        ax[0, caid].plot(spike_threshs, sdict_b['medianR'], linestyle='dotted', c=ca_c[ca], label='B', linewidth=linew)
        ax[0, caid].plot(spike_threshs, sdict_nb['medianR'], linestyle='dashed', c=ca_c[ca], label='N-B', linewidth=linew)
        ax[0, caid].set_title(ca, fontsize=titlesize)
        ax[0, caid].spines['top'].set_visible(False)
        ax[0, caid].spines['right'].set_visible(False)

        ax[1, caid].plot(spike_threshs, sdict_all['sigfrac_shift'], c=ca_c[ca], label='All', linewidth=linew)
        ax[1, caid].plot(spike_threshs, sdict_b['sigfrac_shift'], linestyle='dotted', c=ca_c[ca], label='B', linewidth=linew)
        ax[1, caid].plot(spike_threshs, sdict_nb['sigfrac_shift'], linestyle='dashed', c=ca_c[ca], label='N-B', linewidth=linew)
        ax[1, caid].spines['top'].set_visible(False)
        ax[1, caid].spines['right'].set_visible(False)

        ax_frac[0].plot(spike_threshs, sdict_all['datafrac'], c=ca_c[ca], label=ca, linewidth=linew)
        ax_frac[1].plot(spike_threshs, sdict_b['n']/sdict_all['n'], c=ca_c[ca], label=ca, linewidth=linew)

    # # Statistical test
    for idx, ntresh in enumerate(spike_threshs):

        # Between border cases for each CA
        for ca in ['CA%d'%i for i in range(1, 4)]:
            cad_b = data_dict[ca]['border']
            cad_nb = data_dict[ca]['nonborder']
            rs_bord_stat, rs_bord_pR = ranksums(cad_b['allR'][idx], cad_nb['allR'][idx])

            contin = pd.DataFrame({'border': [cad_b['shift_signum'][idx],
                                              cad_b['shift_nonsignum'][idx]],
                                   'nonborder': [cad_nb['shift_signum'][idx],
                                                 cad_nb['shift_nonsignum'][idx]]})
            try:
                chi_stat, chi_pborder, chi_dof, _ = chi2_contingency(contin)
            except ValueError:
                chi_stat, chi_pborder, chi_dof = None, 1, None

            border_n, nonborder_n = cad_b['n'][idx], cad_nb['n'][idx]
            stat_record(stat_fn, False, '%s BorderEffect Sig. Frac., Thresh=%0.2f, \chi^2_{(df=%d, n=%d)}=%0.2f, p=%0.4f' % \
                        (ca, ntresh, chi_dof, border_n + nonborder_n, chi_stat, chi_pborder))

            mdnR_border, mdnR2_nonrboder = cad_b['medianR'][idx], cad_nb['medianR'][idx]
            stat_record(stat_fn, False,
                        '%s BorderEffect MedianR:%0.2f-%0.2f, Thresh=%0.2f, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (ca, mdnR_border, mdnR2_nonrboder, ntresh, rs_bord_stat, border_n, nonborder_n, rs_bord_pR))

        # Between CAs for each border cases
        for bcase in ['all', 'border', 'nonborder']:
            ca1d, ca2d, ca3d = data_dict['CA1'][bcase], data_dict['CA2'][bcase], data_dict['CA3'][bcase]

            rs_stat13, rs_p13R = ranksums(ca1d['allR'][idx], ca3d['allR'][idx])
            rs_stat23, rs_p23R = ranksums(ca2d['allR'][idx], ca3d['allR'][idx])
            rs_stat12, rs_p12R = ranksums(ca1d['allR'][idx], ca2d['allR'][idx])

            contin13 = pd.DataFrame({'CA1': [ca1d['shift_signum'][idx], ca1d['shift_nonsignum'][idx]],
                                     'CA3': [ca3d['shift_signum'][idx], ca3d['shift_nonsignum'][idx]]})

            contin23 = pd.DataFrame({'CA1': [ca2d['shift_signum'][idx], ca2d['shift_nonsignum'][idx]],
                                     'CA2': [ca3d['shift_signum'][idx], ca3d['shift_nonsignum'][idx]]})

            contin12 = pd.DataFrame({'CA1': [ca1d['shift_signum'][idx], ca1d['shift_nonsignum'][idx]],
                                     'CA2': [ca2d['shift_signum'][idx], ca2d['shift_nonsignum'][idx]]})

            try:
                chi_stat13, chi_p13, chi_dof13, _ = chi2_contingency(contin13)
            except ValueError:
                chi_stat13, chi_p13, chi_dof13 = None, 1, None
            try:
                chi_stat23, chi_p23, chi_dof23, _ = chi2_contingency(contin23)
            except ValueError:
                chi_stat23, chi_p23, chi_dof23 = None, 1, None
            try:
                chi_stat12, chi_p12, chi_dof12, _ = chi2_contingency(contin12)
            except ValueError:
                chi_stat12, chi_p12, chi_dof12 = None, 1, None

            nCA1, nCA2, nCA3 = ca1d['n'][idx], ca2d['n'][idx], ca3d['n'][idx]
            stat_record(stat_fn, False, '%s CA13 Sig. Frac., Thresh=%0.2f, \chi^2_{(df=%d, n=%d)}=%0.2f, p=%0.4f'% \
                        (bcase, ntresh, chi_dof13, nCA1+nCA3, chi_stat13, chi_p13))
            stat_record(stat_fn, False, '%s CA23 Sig. Frac., Thresh=%0.2f, \chi^2_{(df=%d, n=%d)}=%0.2f, p=%0.4f' % \
                        (bcase, ntresh, chi_dof23, nCA2 + nCA3, chi_stat23, chi_p23))
            stat_record(stat_fn, False, '%s CA12 Sig. Frac., Thresh=%0.2f, \chi^2_{(df=%d, n=%d)}=%0.2f, p=%0.4f' % \
                        (bcase, ntresh, chi_dof12, nCA1 + nCA2, chi_stat12, chi_p12))

            mdnR1, mdnR2, mdnR3 = ca1d['medianR'][idx], ca2d['medianR'][idx], ca3d['medianR'][
                idx]

            stat_record(stat_fn, False,
                        '%s CA13 MedianR:%0.2f-%0.2f, Thresh=%0.2f, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (bcase, mdnR1, mdnR3, ntresh, rs_stat13, nCA1, nCA3, rs_p13R))
            stat_record(stat_fn, False,
                        '%s CA23 MedianR:%0.2f-%0.2f, Thresh=%0.2f, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (bcase, mdnR2, mdnR3, ntresh, rs_stat23, nCA2, nCA3, rs_p23R))
            stat_record(stat_fn, False,
                        '%s CA12 MedianR:%0.2f-%0.2f, Thresh=%0.2f, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (bcase, mdnR1, mdnR2, ntresh, rs_stat12, nCA1, nCA2, rs_p12R))


    # # Asthestics of plotting
    customlegend(ax[0, 0], linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')
    customlegend(ax_frac[0], handlelength=0.5, linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')


    ax[0, 0].set_ylabel('Median R', fontsize=fontsize)
    ax[1, 0].set_ylabel('Significant\n Fraction', fontsize=fontsize)
    for i in range(3):
        ax[0, i].set_ylim(0.05, 0.17)  # R, spikes
        ax[0, i].set_yticks([0.06, 0.10, 0.14])
        ax[1, i].set_ylim(0, 0.4)  # Sig Frac., spikes
        ax[1, i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax[1, i].set_yticklabels(['0', '', '', '', '0.4'])
        ax[1, i].set_xticks([0, 250, 500, 750])
        ax[1, i].set_xticklabels(['0', '', '', 750])
    for i in range(2):  # loop for rows
        ax[i, 1].set_ylabel('')
        ax[i, 2].set_ylabel('')
    for ax_each in ax.ravel():
        ax_each.tick_params(labelsize=ticksize)
    fig.text(0.6, 0.01, 'Spike count threshold', ha='center', fontsize=fontsize)

    ax_frac[0].set_title(' ', fontsize=titlesize)
    ax_frac[0].set_ylabel('All fields\nfraction', fontsize=fontsize)
    ax_frac[0].set_yticks([0, 1])
    ax_frac[0].tick_params(labelsize=ticksize)
    ax_frac[0].spines['top'].set_visible(False)
    ax_frac[0].spines['right'].set_visible(False)
    ax_frac[1].set_yticks([0, 0.2, 0.4, 0.6])
    ax_frac[1].set_yticklabels(['0', '', '', '0.6'])
    ax_frac[1].set_xticks([0, 250, 500, 750])
    ax_frac[1].set_xticklabels(['0', '', '', '750'])
    ax_frac[1].tick_params(labelsize=ticksize)
    # ax_frac[1].set_xlabel('Spike count\nthreshold', fontsize=fontsize)
    ax_frac[1].set_ylabel('Border fields\nfraction', fontsize=fontsize)
    ax_frac[1].spines['top'].set_visible(False)
    ax_frac[1].spines['right'].set_visible(False)

    fig_frac.text(0.55, 0.01, 'Spike count threshold', ha='center', fontsize=fontsize)

    fig.tight_layout()
    fig_frac.tight_layout()
    if save_dir:

        fig.savefig(os.path.join(save_dir, 'exp_single_directionality.%s'%(figext)), dpi=dpi)
        fig_frac.savefig(os.path.join(save_dir, 'exp_single_fraction.%s'%(figext)), dpi=dpi)





if __name__ == '__main__':

    # Plot results
    expdf = load_pickle('results/exp/single_field/singlefield_df.pickle')
    omniplot_singlefields(expdf, save_dir='result_plots/single_field/')
