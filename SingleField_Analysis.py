# This script does directionality and spike phase analysis for single fields.
# Only for Emily's data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d, make_interp_spline
from pycircstat.descriptive import resultant_vector_length, cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import chi2_contingency, binom_test

from common.utils import stat_record, p2str
from common.comput_utils import fisherexact, ranksums

from common.visualization import color_wheel, directionality_polar_plot, customlegend

from common.script_wrappers import DirectionalityStatsByThresh
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw


def omniplot_singlefields(expdf, save_dir=None):
    linew = 0.75

    # Initialization of stats
    stat_fn = 'fig1_single_directionality.txt'
    stat_record(stat_fn, True)


    # Initialization of figures
    fig = plt.figure(figsize=(total_figw, total_figw/2))
    xgap_direct_frac = 0.05
    axdirect_w = (1-xgap_direct_frac)/4
    axdirect_h = 1/2
    xsqueeze_all = 0.075
    ysqueeze_all = 0.2


    btw_CAs_xsqueeze = 0.05
    direct_xoffset = 0.05

    ax_directR_y = 1-axdirect_h
    ax_directR = np.array([
        fig.add_axes([0+xsqueeze_all/2+btw_CAs_xsqueeze+direct_xoffset, ax_directR_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
        fig.add_axes([axdirect_w+xsqueeze_all/2+direct_xoffset, ax_directR_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
        fig.add_axes([axdirect_w*2+xsqueeze_all/2-btw_CAs_xsqueeze+direct_xoffset, ax_directR_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
    ])

    axdirectfrac_yoffset = 0.125
    axdirectfrac_y = 1-axdirect_h*2 + axdirectfrac_yoffset
    ax_directFrac = np.array([
        fig.add_axes([0+xsqueeze_all/2+btw_CAs_xsqueeze+direct_xoffset, axdirectfrac_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
        fig.add_axes([axdirect_w+xsqueeze_all/2+direct_xoffset, axdirectfrac_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
        fig.add_axes([axdirect_w*2+xsqueeze_all/2-btw_CAs_xsqueeze+direct_xoffset, axdirectfrac_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
    ])

    axfrac_x = axdirect_w*3 + xgap_direct_frac
    ax_frac =np.array([
        fig.add_axes([axfrac_x+xsqueeze_all/2, ax_directR_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
        fig.add_axes([axfrac_x+xsqueeze_all/2, axdirectfrac_y+ysqueeze_all/2, axdirect_w-xsqueeze_all, axdirect_h-ysqueeze_all]),
    ])

    ax = np.stack([ax_directR, ax_directFrac])

    figl = total_figw / 4
    fig_pval, ax_pval = plt.subplots(2, 1, figsize=(figl*4, figl*4))


    # Initialization of parameters
    spike_threshs = np.arange(0, 401, 50)
    stats_getter = DirectionalityStatsByThresh('num_spikes', 'rate_R_pval', 'rate_R')
    chipval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    rspval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])

    # # Plot data
    data_dict = {'CA%d' % (i): dict() for i in range(1, 4)}
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
        ax[0, caid].plot(spike_threshs, sdict_nb['medianR'], linestyle='dashed', c=ca_c[ca], label='N-B',
                         linewidth=linew)
        ax[0, caid].set_title(ca, fontsize=titlesize)

        ax[1, caid].plot(spike_threshs, sdict_all['sigfrac_shift'], c=ca_c[ca], label='All', linewidth=linew)
        ax[1, caid].plot(spike_threshs, sdict_b['sigfrac_shift'], linestyle='dotted', c=ca_c[ca], label='B',
                         linewidth=linew)
        ax[1, caid].plot(spike_threshs, sdict_nb['sigfrac_shift'], linestyle='dashed', c=ca_c[ca], label='N-B',
                         linewidth=linew)

        ax_frac[0].plot(spike_threshs, sdict_all['datafrac'], c=ca_c[ca], label=ca, linewidth=linew)
        ax_frac[1].plot(spike_threshs, sdict_b['n'] / sdict_all['n'], c=ca_c[ca], label=ca, linewidth=linew)



        # Binomial test for all fields
        signum_all, n_all = sdict_all['shift_signum'][0], sdict_all['n'][0]
        p_binom = binom_test(signum_all, n_all, p=0.05, alternative='greater')
        stat_txt = '%s, Binomial test, greater than p=0.05, %d/%d, p%s'%(ca, signum_all, n_all, p2str(p_binom))
        stat_record(stat_fn, False, stat_txt)
        # ax[1, caid].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize, color=ca_c[ca])



    # # Statistical test
    for idx, ntresh in enumerate(spike_threshs):

        # Between border cases for each CA
        for ca in ['CA%d' % i for i in range(1, 4)]:
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
                chi_stat, chi_pborder, chi_dof = 0, 1, 0

            f_pborder = fisherexact(contin.to_numpy())
            border_n, nonborder_n = cad_b['n'][idx], cad_nb['n'][idx]
            stattxt = "Threshold=%d, %s, border vs non-border: Chi-square  test  for  significant  fraction, \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
            stat_record(stat_fn, False, stattxt % ((ntresh, ca, chi_dof, border_n + nonborder_n, chi_stat,
                                                    p2str(chi_pborder), p2str(f_pborder))))
            chipval_dict[ca + '_be'].append(chi_pborder)

            mdnR_border, mdnR2_nonrboder = cad_b['medianR'][idx], cad_nb['medianR'][idx]
            stattxt = "Threshold=%d, %s, border vs non-border: Mann-Whitney U test for medianR, %0.3f vs %0.3f, U(N_{border}=%d, N_{nonborder}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % (ntresh, ca, mdnR_border, mdnR2_nonrboder, border_n,
                                                   nonborder_n, rs_bord_stat, p2str(rs_bord_pR)))
            rspval_dict[ca + '_be'].append(rs_bord_pR)


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
                chi_stat13, chi_p13, chi_dof13 = 0, 1, 0
            try:
                chi_stat23, chi_p23, chi_dof23, _ = chi2_contingency(contin23)
            except ValueError:
                chi_stat23, chi_p23, chi_dof23 = 0, 1, 0
            try:
                chi_stat12, chi_p12, chi_dof12, _ = chi2_contingency(contin12)
            except ValueError:
                chi_stat12, chi_p12, chi_dof12 = 0, 1, 0

            f_p13 = fisherexact(contin13.to_numpy())
            f_p23 = fisherexact(contin23.to_numpy())
            f_p12 = fisherexact(contin12.to_numpy())
            nCA1, nCA2, nCA3 = ca1d['n'][idx], ca2d['n'][idx], ca3d['n'][idx]
            stattxt = "Threshold=%d, %s, CA1 vs CA3: Chi-square  test  for  significant  fraction, \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, chi_dof13, nCA1 + nCA3, chi_stat13, p2str(chi_p13), p2str(f_p13)))
            stattxt = "Threshold=%d, %s, CA2 vs CA3: Chi-square  test  for  significant  fraction, \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, chi_dof23, nCA2 + nCA3, chi_stat23, p2str(chi_p23), p2str(f_p23)))
            stattxt = "Threshold=%d, %s, CA1 vs CA2: Chi-square  test  for  significant  fraction, \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, chi_dof12, nCA1 + nCA2, chi_stat12, p2str(chi_p12), p2str(f_p12)))

            mdnR1, mdnR2, mdnR3 = ca1d['medianR'][idx], ca2d['medianR'][idx], ca3d['medianR'][idx]
            stattxt = "Threshold=%d, %s, CA1 vs CA3: Mann-Whitney U test for medianR, %0.3f vs %0.3f, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, mdnR1, mdnR3, nCA1, nCA3, rs_stat13, p2str(rs_p13R)))
            stattxt = "Threshold=%d, %s, CA2 vs CA3: Mann-Whitney U test for medianR, %0.3f vs %0.3f, U(N_{CA2}=%d, N_{CA3}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, mdnR2, mdnR3, nCA2, nCA3, rs_stat23, p2str(rs_p23R)))
            stattxt = "Threshold=%d, %s, CA1 vs CA2: Mann-Whitney U test for medianR, %0.3f vs %0.3f, U(N_{CA1}=%d, N_{CA2}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, mdnR1, mdnR2, nCA1, nCA2, rs_stat12, p2str(rs_p12R)))

            if bcase == 'all':
                chipval_dict['all_CA13'].append(chi_p13)
                chipval_dict['all_CA23'].append(chi_p23)
                chipval_dict['all_CA12'].append(chi_p12)
                rspval_dict['all_CA13'].append(rs_p13R)
                rspval_dict['all_CA23'].append(rs_p23R)
                rspval_dict['all_CA12'].append(rs_p12R)

    # Plot pvals for reference
    logalpha = np.log10(0.05)
    for ca in ['CA1', 'CA2', 'CA3']:
        ax_pval[0].plot(spike_threshs, np.log10(chipval_dict[ca + '_be']), c=ca_c[ca], label=ca+'_Chi2', linestyle='-', marker='o')
        ax_pval[0].plot(spike_threshs, np.log10(rspval_dict[ca + '_be']), c=ca_c[ca], label=ca+'_RS', linestyle='--', marker='x')
    ax_pval[0].plot([spike_threshs.min(), spike_threshs.max()], [logalpha, logalpha], c='k', label='p=0.05')
    ax_pval[0].set_xlabel('Spike count threshold')
    ax_pval[0].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_pval[0].set_title('Border effect', fontsize=titlesize)
    customlegend(ax_pval[0], fontsize=legendsize)
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA13']), c='r', label='13_Chi2', linestyle='-', marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA23']), c='g', label='23_Chi2', linestyle='-', marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA12']), c='b', label='12_Chi2', linestyle='-', marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA13']), c='r', label='13_RS', linestyle='--', marker='x')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA23']), c='g', label='23_RS', linestyle='--', marker='x')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA12']), c='b', label='12_RS', linestyle='--', marker='x')
    ax_pval[1].plot([spike_threshs.min(), spike_threshs.max()], [logalpha, logalpha], c='k', label='p=0.05')
    ax_pval[1].set_xlabel('Spike count threshold')
    ax_pval[1].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_pval[1].set_title('Compare CAs', fontsize=titlesize)
    customlegend(ax_pval[1], fontsize=legendsize)

    # # Asthestics of plotting
    customlegend(ax[0, 0], linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')
    customlegend(ax_frac[0], handlelength=0.5, linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8],
                 loc='center')

    ax[0, 0].set_ylabel('Median R', fontsize=fontsize)
    ax[1, 0].set_ylabel('Significant\n Fraction', fontsize=fontsize)
    for i in range(2):  # loop for rows
        ax[i, 1].set_ylabel('')
        ax[i, 2].set_ylabel('')

    ax_frac[0].set_title(' ', fontsize=titlesize)
    ax_frac[0].set_ylabel('All fields\nfraction', fontsize=fontsize)
    ax_frac[1].set_ylabel('Border fields\nfraction', fontsize=fontsize)


    for ax_each in np.concatenate([ax.ravel(), ax_frac]):
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
        ax_each.tick_params(labelsize=ticksize)

    for i in range(3):
        ax[0, i].set_xticks([0, 100, 200, 300, 400])
        ax[0, i].set_xticklabels(['']*5)
        ax[1, i].set_xticks([0, 100, 200, 300, 400])
        ax[1, i].set_xticklabels(['0', '', '', '', '400'])
    ax_frac[0].set_xticks([0, 100, 200, 300, 400])
    ax_frac[0].set_xticklabels(['']*5)
    ax_frac[1].set_xticks([0, 100, 200, 300, 400])
    ax_frac[1].set_xticklabels(['0', '', '', '', '400'])

    # yticks
    for i in range(3):
        yticks = [0.1, 0.2]
        ax_directR[i].set_ylim(0.05, 0.25)
        ax_directR[i].set_yticks(yticks)
        ax_directR[i].set_yticks([0.05, 0.1, 0.15, 0.2, 0.25], minor=True)
        if i != 0:
            ax_directR[i].set_yticklabels(['']*len(yticks))

        yticks = [0, 0.5, 1]
        ax_directFrac[i].set_yticks(yticks)
        ax_directFrac[i].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        if i != 0:
            ax_directFrac[i].set_yticklabels(['']*len(yticks))
    for ax_each in ax_frac:
        ax_each.set_yticks([0, 0.5, 1])
        ax_each.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        ax_each.set_yticklabels(['0', '', '1'])


    ax_directFrac[1].set_xlabel('Spike count\nthreshold', ha='center', fontsize=fontsize)
    ax_frac[1].set_xlabel('Spike count\nthreshold', ha='center', fontsize=fontsize)


    fig_pval.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp_single_directionality.png'), dpi=dpi)
    fig.savefig(os.path.join(save_dir, 'exp_single_directionality.eps'), dpi=dpi)
    fig_pval.savefig(os.path.join(save_dir, 'exp_single_TestSignificance.png'), dpi=dpi)

if __name__ == '__main__':
    # Plot results
    data_pth = 'results/exp/single_field/singlefield_df.pickle'
    # data_pth = 'results/exp/single_field/singlefield_df_minpasst04_NoShuffle.pickle'
    save_dir = 'result_plots/single_field/'

    df = pd.read_pickle(data_pth)

    os.makedirs(save_dir, exist_ok=True)
    omniplot_singlefields(df, save_dir=save_dir)
