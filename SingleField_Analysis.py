# This script does directionality and spike phase analysis for single fields.
# Only for Emily's data
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import cm
from pycircstat import vtest
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import binom_test, linregress, spearmanr

from common.linear_circular_r import rcc
from common.stattests import p2str, stat_record, my_chi_2way, my_kruskal_2samp, my_kruskal_3samp, \
    fdr_bh_correction, my_ww_2samp, my_fisher_2way, circ_ktest
from common.comput_utils import linear_circular_gauss_density, unfold_binning_2d, midedges, \
    shiftcyc_full2half, repeat_arr, get_numpass_at_angle

from common.visualization import customlegend, plot_marginal_slices

from common.script_wrappers import DirectionalityStatsByThresh, compute_precessangle
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

def omniplot_singlefields(expdf, save_dir=None):
    # # Initialization of stats
    stat_fn = 'fig1_single_directionality.txt'
    stat_record(stat_fn, True)

    # # Initialization of figures
    fullcol_figw = 6.3
    main_w = fullcol_figw * 0.7
    frac_w = fullcol_figw * 0.3
    total_figh = fullcol_figw/2
    xgap_direct_frac = 0.05
    axdirect_w = (1 - xgap_direct_frac) / 4
    axdirect_h = 1/3
    xsqueeze_all = 0.075
    ysqueeze_all = 0.15
    btw_CAs_xsqueeze = 0.05
    direct_xoffset = 0.05
    fig = plt.figure(figsize=(fullcol_figw, total_figh), dpi=dpi, facecolor='w')

    # Axes for R population
    xshorten_popR = 0.025
    ax_popR_y = 1 - 1/3
    ax_popR_w = axdirect_w - xsqueeze_all - xshorten_popR
    ax_popR = np.array([
        fig.add_axes(
            [0 + xsqueeze_all / 2 + btw_CAs_xsqueeze + direct_xoffset, ax_popR_y + ysqueeze_all / 2,
             ax_popR_w, axdirect_h - ysqueeze_all]),
        fig.add_axes(
            [axdirect_w + xsqueeze_all / 2 + direct_xoffset, ax_popR_y + ysqueeze_all / 2,
             ax_popR_w, axdirect_h - ysqueeze_all]),
        fig.add_axes(
            [axdirect_w * 2 + xsqueeze_all / 2 - btw_CAs_xsqueeze + direct_xoffset, ax_popR_y + ysqueeze_all / 2,
             ax_popR_w, axdirect_h - ysqueeze_all]),
    ])

    ax_popR_den = np.array([
        fig.add_axes(
            [0 + xsqueeze_all / 2 + btw_CAs_xsqueeze + direct_xoffset + ax_popR_w, ax_popR_y + ysqueeze_all / 2,
             xshorten_popR, axdirect_h - ysqueeze_all]),
        fig.add_axes(
            [axdirect_w + xsqueeze_all / 2 + direct_xoffset + ax_popR_w, ax_popR_y + ysqueeze_all / 2,
             xshorten_popR, axdirect_h - ysqueeze_all]),
        fig.add_axes(
            [axdirect_w * 2 + xsqueeze_all / 2 - btw_CAs_xsqueeze + direct_xoffset + ax_popR_w, ax_popR_y + ysqueeze_all / 2,
             xshorten_popR, axdirect_h - ysqueeze_all]),
    ])

    axdirect_yoffset = 0
    ax_directR_y = 1 - 2/3 + axdirect_yoffset
    ax_directR = np.array([
        fig.add_axes([0 + xsqueeze_all / 2 + btw_CAs_xsqueeze + direct_xoffset, ax_directR_y + ysqueeze_all / 2,
                      axdirect_w - xsqueeze_all, axdirect_h - ysqueeze_all]),
        fig.add_axes(
            [axdirect_w + xsqueeze_all / 2 + direct_xoffset, ax_directR_y + ysqueeze_all / 2, axdirect_w - xsqueeze_all,
             axdirect_h - ysqueeze_all]),
        fig.add_axes(
            [axdirect_w * 2 + xsqueeze_all / 2 - btw_CAs_xsqueeze + direct_xoffset, ax_directR_y + ysqueeze_all / 2,
             axdirect_w - xsqueeze_all, axdirect_h - ysqueeze_all]),
    ])

    axdirectfrac_yoffset = 0.10
    axdirectfrac_y =1 - (3/3) + axdirectfrac_yoffset
    ax_directFrac = np.array([
        fig.add_axes([0 + xsqueeze_all / 2 + btw_CAs_xsqueeze + direct_xoffset, axdirectfrac_y + ysqueeze_all / 2,
                      axdirect_w - xsqueeze_all, axdirect_h - ysqueeze_all]),
        fig.add_axes([axdirect_w + xsqueeze_all / 2 + direct_xoffset, axdirectfrac_y + ysqueeze_all / 2,
                      axdirect_w - xsqueeze_all, axdirect_h - ysqueeze_all]),
        fig.add_axes(
            [axdirect_w * 2 + xsqueeze_all / 2 - btw_CAs_xsqueeze + direct_xoffset, axdirectfrac_y + ysqueeze_all / 2,
             axdirect_w - xsqueeze_all, axdirect_h - ysqueeze_all]),
    ])

    axfrac_h, ysqueeze_axfrac = 1/2, 0.15
    axfrac_yoffset, axfrac_ygap = 0.05, 0.05
    axfrac_x = axdirect_w * 3 + xgap_direct_frac
    axfrac_y1, axfrac_y2 = 1 - axfrac_h - axfrac_ygap + axfrac_yoffset, 1 - axfrac_h*2 + axfrac_ygap + axfrac_yoffset
    ax_frac = np.array([
        fig.add_axes([axfrac_x + xsqueeze_all / 2, axfrac_y1 + ysqueeze_axfrac / 2, axdirect_w - xsqueeze_all,
                      axfrac_h - ysqueeze_axfrac]),
        fig.add_axes([axfrac_x + xsqueeze_all / 2, axfrac_y2 + ysqueeze_axfrac / 2, axdirect_w - xsqueeze_all,
                      axfrac_h - ysqueeze_axfrac]),
    ])

    ax = np.stack([ax_popR, ax_directR, ax_directFrac])

    figl = fullcol_figw / 4
    fig_pval, ax_pval = plt.subplots(2, 1, figsize=(figl * 4, figl * 4))

    # Initialization of parameters
    linew = 0.75
    popR_B_c, popR_B_mark, popR_NB_c, popR_NB_mark = 'teal', '.', 'goldenrod', '.'
    spike_threshs = np.arange(0, 401, 20)  # Original
    stats_getter = DirectionalityStatsByThresh('num_spikes', 'rate_R_pval', 'rate_R')
    redges = np.arange(0, 1.05, 0.1)

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
        ax_popR[caid].scatter(x=cadf_b['num_spikes'], y=cadf_b['rate_R'], c=popR_B_c, s=0.1, marker=popR_B_mark, alpha=0.7)
        ax_popR[caid].scatter(x=cadf_nb['num_spikes'], y=cadf_nb['rate_R'], c=popR_NB_c, s=0.1, marker=popR_NB_mark, alpha=0.7)
        ax_popR[caid].set_title(ca, fontsize=titlesize)
        ax_popR_den[caid].hist(cadf_b['rate_R'], bins=redges, orientation='horizontal', histtype='step', density=True, color=popR_B_c, linewidth=linew)
        ax_popR_den[caid].hist(cadf_nb['rate_R'], bins=redges, orientation='horizontal', histtype='step', density=True, color=popR_NB_c, linewidth=linew)
        ax_popR[caid].axvspan(xmin=0, xmax=40, color='0.9', zorder=0)
        ax_directR[caid].plot(spike_threshs, sdict_all['medianR'], c=ca_c[ca], label='All', linewidth=linew)
        ax_directR[caid].plot(spike_threshs, sdict_b['medianR'], linestyle='dotted', c=ca_c[ca], label='B', linewidth=linew)
        ax_directR[caid].plot(spike_threshs, sdict_nb['medianR'], linestyle='dashed', c=ca_c[ca], label='N-B',
                              linewidth=linew)
        ax_directFrac[caid].plot(spike_threshs, sdict_all['sigfrac_shift'], c=ca_c[ca], label='All', linewidth=linew)
        ax_directFrac[caid].plot(spike_threshs, sdict_b['sigfrac_shift'], linestyle='dotted', c=ca_c[ca], label='B',
                                 linewidth=linew)
        ax_directFrac[caid].plot(spike_threshs, sdict_nb['sigfrac_shift'], linestyle='dashed', c=ca_c[ca], label='N-B',
                                 linewidth=linew)

        ax_frac[0].plot(spike_threshs, sdict_all['datafrac'], c=ca_c[ca], label=ca, linewidth=linew)
        ax_frac[1].plot(spike_threshs, sdict_b['n'] / sdict_all['n'], c=ca_c[ca], label=ca, linewidth=linew)

        # Binomial test for all fields
        for idx, ntresh in enumerate(spike_threshs):
            signum_all, n_all = sdict_all['shift_signum'][idx], sdict_all['n'][idx]
            p_binom = binom_test(signum_all, n_all, p=0.05, alternative='greater')
            stat_txt = '%s, Thresh=%d, Binomial test, greater than p=0.05, %d/%d=%0.4f, $p=%s$' % (ca, ntresh, signum_all, n_all, signum_all/n_all, p2str(p_binom))
            stat_record(stat_fn, False, stat_txt)


    # # Statistical test
    chipval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    fishpval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    rspval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    for idx, ntresh in enumerate(spike_threshs):
        stat_record(stat_fn, False, '======= Threshold=%d ======'%(ntresh))
        # # Border vs non-border for each CA
        for ca in ['CA%d' % i for i in range(1, 4)]:
            cad_b = data_dict[ca]['border']
            cad_nb = data_dict[ca]['nonborder']

            # KW test for border median R
            rs_bord_pR, (border_n, nonborder_n), mdns, rs_txt = my_kruskal_2samp(cad_b['allR'][idx], cad_nb['allR'][idx], 'border', 'nonborder')
            stat_record(stat_fn, False, "%s, Median R, border vs non-border: %s" % (ca, rs_txt))
            rspval_dict[ca + '_be'].append(rs_bord_pR)

            # Chisquared test for border fractions
            contin = pd.DataFrame({'border': [cad_b['shift_signum'][idx],
                                              cad_b['shift_nonsignum'][idx]],
                                   'nonborder': [cad_nb['shift_signum'][idx],
                                                 cad_nb['shift_nonsignum'][idx]]}).to_numpy()
            chi_pborder, _, txt_chiborder = my_chi_2way(contin)
            _, _, fishtxt = my_fisher_2way(contin)
            stat_record(stat_fn, False, "%s, Significant fraction, border vs non-border: %s, %s" % (ca, txt_chiborder, fishtxt))
            chipval_dict[ca + '_be'].append(chi_pborder)


        # # Between CAs for All
        bcase = 'all'
        ca1d, ca2d, ca3d = data_dict['CA1'][bcase], data_dict['CA2'][bcase], data_dict['CA3'][bcase]

        # 3-sample KW test for median R, CA1 vs CA2 vs CA3
        kruskal_ps, dunn_pvals, (n1, n2, n3), _, kw3txt = my_kruskal_3samp(ca1d['allR'][idx], ca2d['allR'][idx],
                                                                           ca3d['allR'][idx], 'CA1', 'CA2', 'CA3')

        stat_record(stat_fn, False, 'Median R between CAs, %s'%(kw3txt))
        rs_p12R, rs_p23R, rs_p13R = dunn_pvals
        rspval_dict['all_CA12'].append(rs_p12R)
        rspval_dict['all_CA23'].append(rs_p23R)
        rspval_dict['all_CA13'].append(rs_p13R)

        # Chisquared test for sig fractions between CAs
        for canum1, canum2 in ((1, 2), (2, 3), (1, 3)):

            cafirst, casecond = 'CA%d'%(canum1), 'CA%d'%(canum2)
            cadictfirst, cadictsecond = data_dict[cafirst][bcase], data_dict[casecond][bcase]
            contin_btwca = pd.DataFrame({cafirst: [cadictfirst['shift_signum'][idx], cadictfirst['shift_nonsignum'][idx]],
                                         casecond: [cadictsecond['shift_signum'][idx], cadictsecond['shift_nonsignum'][idx]]}).to_numpy()
            chi_pbtwca, _, txt_chibtwca = my_chi_2way(contin_btwca)
            fish_pbtwca, _, fishtxt_btwca = my_fisher_2way(contin_btwca)
            stat_record(stat_fn, False, "Between CAs, Significant fraction, %s vs %s: %s, %s" % (cafirst, casecond, txt_chibtwca, fishtxt_btwca))
            chipval_dict['all_CA%d%d'%(canum1, canum2)].append(chi_pbtwca)
            fishpval_dict['all_CA%d%d'%(canum1, canum2)].append(fish_pbtwca)
        chiqs = fdr_bh_correction(np.array([chipval_dict['all_CA12'][-1], chipval_dict['all_CA23'][-1], chipval_dict['all_CA13'][-1]]))
        fishqs = fdr_bh_correction(np.array([fishpval_dict['all_CA12'][-1], fishpval_dict['all_CA23'][-1], fishpval_dict['all_CA13'][-1]]))
        stat_record(stat_fn, False, r"Chisquare Benjamini-Hochberg correction, $p12=%s$, $p23=%s$, $p13=%s$" % (p2str(chiqs[0]), p2str(chiqs[1]), p2str(chiqs[2])))
        stat_record(stat_fn, False, r"Fisher Benjamini-Hochberg correction, $p12=%s$, $p23=%s$, $p13=%s$" % (p2str(fishqs[0]), p2str(fishqs[1]), p2str(fishqs[2])))

    # Plot pvals for reference
    logalpha = np.log10(0.05)
    for ca in ['CA1', 'CA2', 'CA3']:
        ax_pval[0].plot(spike_threshs, np.log10(chipval_dict[ca + '_be']), c=ca_c[ca], label=ca + '_Chi2',
                        linestyle='-', marker='o')
        ax_pval[0].plot(spike_threshs, np.log10(rspval_dict[ca + '_be']), c=ca_c[ca], label=ca + '_RS', linestyle='--',
                        marker='x')
    ax_pval[0].plot([spike_threshs.min(), spike_threshs.max()], [logalpha, logalpha], c='k', label='p=0.05')
    ax_pval[0].set_xlabel('Spike count threshold')
    ax_pval[0].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_pval[0].set_title('Border effect', fontsize=titlesize)
    customlegend(ax_pval[0], fontsize=legendsize)
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA13']), c='r', label='13_Chi2', linestyle='-',
                    marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA23']), c='g', label='23_Chi2', linestyle='-',
                    marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA12']), c='b', label='12_Chi2', linestyle='-',
                    marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA13']), c='r', label='13_RS', linestyle='--', marker='x')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA23']), c='g', label='23_RS', linestyle='--', marker='x')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA12']), c='b', label='12_RS', linestyle='--', marker='x')
    ax_pval[1].plot([spike_threshs.min(), spike_threshs.max()], [logalpha, logalpha], c='k', label='p=0.05')
    ax_pval[1].set_xlabel('Spike count threshold')
    ax_pval[1].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_pval[1].set_title('Compare CAs', fontsize=titlesize)
    customlegend(ax_pval[1], fontsize=legendsize)

    # # Asthestics of plotting
    # ax_popR
    ax_popR[0].scatter(-1, -1, c=popR_B_c, marker=popR_B_mark, label='B', s=8)
    ax_popR[0].scatter(-1, -1, c=popR_NB_c, marker=popR_NB_mark, label='N-B', s=8)
    ax_popR[0].set_ylabel('R', fontsize=fontsize)
    customlegend(ax_popR[0], linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')
    ax_popR[1].set_xlabel('Spike count', fontsize=fontsize)
    for ax_i, ax_popR_each in enumerate(ax_popR):
        ax_popR_each.set_xlim(0, 1000)
        ax_popR_each.set_xticks(np.arange(0, 801, 400))
        ax_popR_each.set_xticks(np.arange(0, 801, 200), minor=True)
        ax_popR_each.set_ylim(0, 1)
        ax_popR_each.set_yticks(np.arange(0, 1.1, 0.5))
        if ax_i == 0:
            ax_popR_each.set_yticklabels(['0', '', '1'])
        else:
            ax_popR_each.set_yticklabels(['', '', ''])
        ax_popR_each.set_yticks(np.arange(0, 1.1, 0.1), minor=True)

    # ax_popR_den
    for i in range(3):
        ax_popR_den[i].set_ylim(0, 1)
        ax_popR_den[i].axis('off')

    # ax_directR
    ax_directR[0].set_ylabel('Median R', fontsize=fontsize)
    customlegend(ax_directR[0], linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')
    for i in range(3):
        ax_directR[i].set_xticks([0, 100, 200, 300, 400])
        ax_directR[i].set_xticklabels([''] * 5)
        ax_directR[i].set_xlim(0, 400)
        ax_directR[i].set_ylim(0.05, 0.25)
        ax_directR[i].set_yticks([0.1, 0.2])
        ax_directR[i].set_yticks([0.05, 0.1, 0.15, 0.2, 0.25], minor=True)
        if i != 0:
            ax_directR[i].set_yticklabels(['']*2)

    # ax_directFrac
    ax_directFrac[0].set_ylabel('Significant\n Fraction', fontsize=fontsize)
    ax_directFrac[1].set_xlabel('Spike count\nthreshold', ha='center', fontsize=fontsize)
    for i in range(3):
        ax_directFrac[i].set_xticks([0, 100, 200, 300, 400])
        ax_directFrac[i].set_xticklabels(['0', '', '', '', '400'])
        ax_directFrac[i].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        if i != 0:
            ax_directFrac[i].set_yticklabels(['']*5)

    # ax_frac
    ax_frac[0].set_xticks([0, 100, 200, 300, 400])
    ax_frac[0].set_xticklabels([''] * 5)
    ax_frac[0].set_ylabel('All fields\nfraction', fontsize=fontsize)
    customlegend(ax_frac[0], handlelength=0.5, linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8],
                 loc='center')
    ax_frac[1].set_xticks([0, 100, 200, 300, 400])
    ax_frac[1].set_xticklabels(['0', '', '', '', '400'])
    ax_frac[1].set_xlabel('Spike count\nthreshold', ha='center', fontsize=fontsize)
    ax_frac[1].set_ylabel('Border fields\nfraction', fontsize=fontsize)
    for i in range(3):
        if i < 2:
            ax_frac[i].set_yticks([0, 0.5, 1])
            ax_frac[i].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
            ax_frac[i].set_yticklabels(['0', '', '1'])

    # All plots
    for i in range(3):
        ax[i, 1].set_ylabel('')
        ax[i, 2].set_ylabel('')

    for ax_each in np.concatenate([ax.ravel(), ax_popR_den, ax_frac]):
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
        ax_each.tick_params(labelsize=ticksize)

    fig.savefig(os.path.join(save_dir, 'exp_single_directionality.png'), dpi=dpi)
    fig.savefig(os.path.join(save_dir, 'exp_single_directionality.pdf'), dpi=dpi)
    fig_pval.tight_layout()
    fig_pval.savefig(os.path.join(save_dir, 'exp_single_TestSignificance.png'), dpi=dpi)


def plot_precession_examples(df, axpeg):
    def plot_precession(ax, d, phase, rcc_m, rcc_c, marker_size, color):
        ax.scatter(np.concatenate([d, d]), np.concatenate([phase, phase + 2 * np.pi]), marker='.',
                   s=marker_size, c=color)
        xdum = np.linspace(0, 1, 10)
        ydum = xdum * (rcc_m * 2 * np.pi) + rcc_c

        ax.plot(xdum, ydum, c='k', linewidth=0.8)
        ax.plot(xdum, ydum + 2 * np.pi, c='k', linewidth=0.8)
        ax.plot(xdum, ydum - 2 * np.pi, c='k', linewidth=0.8)
        ax.set_yticks([0, 2 * np.pi])
        ax.set_yticklabels(['0', '2$\pi$'])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.set_yticks([-np.pi, np.pi, 3 * np.pi], minor=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(-np.pi, 3 * np.pi)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax


    # # Assign field ids within ca
    id_list = []
    animal_list = []
    for caid, (ca, cadf) in enumerate(df.groupby('ca')):
        cadf = cadf.reset_index(drop=True)

        num_fields = cadf.shape[0]

        for nf in range(num_fields):
            animal, nunit = cadf.loc[nf, ['animal', 'nunit']]
            if animal in animal_list:
                continue
            allprecess_df, fieldid_ca = cadf.loc[nf, ['precess_df', 'fieldid_ca']]
            precess_df = allprecess_df[(allprecess_df['precess_exist']) & (allprecess_df['pass_nspikes']>5)].reset_index(drop=True)
            num_precess = precess_df.shape[0]
            if num_precess < 10:
                continue
            dsplist, phasesplist = [], []
            for nprecess in range(num_precess):
                dsp, phasesp = precess_df.loc[nprecess, ['dsp', 'phasesp']]
                dsplist.append(dsp)
                phasesplist.append(phasesp)

            dsp_all = np.concatenate(dsplist)
            phasesp_all = np.concatenate(phasesplist)
            regress = rcc(dsp_all, phasesp_all)
            rcc_m, rcc_c, rcc_rho, rcc_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']
            if rcc_m > -0.4:
                continue
            plot_precession(axpeg[0, caid], dsp_all, phasesp_all, rcc_m, rcc_c, 1, color=ca_c[ca])


            np.random.seed(1)
            precessids = np.random.choice(num_precess, size=3)
            for axid, precessid in enumerate(precessids):
                dsp, phasesp, rcc_m, rcc_c = precess_df.loc[precessid, ['dsp', 'phasesp', 'rcc_m', 'rcc_c']]
                plot_precession(axpeg[axid + 1, caid], dsp, phasesp, rcc_m, rcc_c, 4, color=ca_c[ca])
            id_list.append(fieldid_ca)
            axpeg[0, caid].set_title('%s' % (ca), fontsize=fontsize, pad=11)
            axpeg[0, caid].annotate('rat%s,unit%d'%(animal[3:], nunit), xy=(-0.05, 1.1), xycoords='axes fraction', fontsize=fontsize-3)
            animal_list.append(animal)

            break

    axpeg[3, 1].set_xticks([0, 1])


    for rowi in range(4):
        plt.setp(axpeg[rowi, 1].get_yticklabels(), visible=False)
        plt.setp(axpeg[rowi, 2].get_yticklabels(), visible=False)
    for coli in range(3):
        for rowi in range(3):
            plt.setp(axpeg[rowi, coli].get_xticklabels(), visible=False)


def plot_field_bestprecession(df, ax_pishift, axR):

    linew = 0.75

    nap_thresh = 1

    print('Plot field best precession')
    stat_fn = 'fig2_field_precess.txt'
    stat_record(stat_fn, True)




    allR = []
    allca_ntotal = df.shape[0]
    all_shufp = []

    # Stack data
    for caid, (ca, cadf) in enumerate(df.groupby('ca')):
        cadf = cadf.reset_index(drop=True)
        numpass_mask = cadf['numpass_at_precess'].to_numpy() >= nap_thresh
        numpass_low_mask = cadf['numpass_at_precess_low'].to_numpy() >= nap_thresh

        # Density of Fraction, Spikes and Ratio
        cadf_this = cadf[(~cadf['rate_angle'].isna())].reset_index(drop=True)

        precess_adiff = []
        precess_nspikes = []
        all_adiff = []
        all_slopes = []
        for i in range(cadf_this.shape[0]):

            refangle = cadf_this.loc[i, 'rate_angle']
            allprecess_df = cadf_this.loc[i, 'precess_df']
            if allprecess_df.shape[0] < 1:
                continue
            precess_df = allprecess_df[allprecess_df['precess_exist']]
            precess_counts = precess_df.shape[0]
            if precess_counts > 0:
                precess_adiff.append(cdiff(precess_df['mean_anglesp'].to_numpy(), refangle))
                precess_nspikes.append(precess_df['pass_nspikes'].to_numpy())
            if allprecess_df.shape[0] > 0:
                all_adiff.append(cdiff(allprecess_df['mean_anglesp'].to_numpy(), refangle))
                all_slopes.append(allprecess_df['rcc_m'].to_numpy())

        precess_adiff = np.abs(np.concatenate(precess_adiff))
        precess_nspikes = np.abs(np.concatenate(precess_nspikes))
        adiff_spikes_p = repeat_arr(precess_adiff, precess_nspikes.astype(int))

        adiff_bins = np.linspace(0, np.pi, 45)
        adm = midedges(adiff_bins)
        precess_bins, _ = np.histogram(precess_adiff, bins=adiff_bins)
        spike_bins_p, _ = np.histogram(adiff_spikes_p, bins=adiff_bins)
        spike_bins = spike_bins_p
        norm_bins = (precess_bins / spike_bins)

        precess_allcount = precess_bins.sum()
        rho, pval = spearmanr(adm, norm_bins)
        pm, pc, pr, ppval, _ = linregress(adm, norm_bins / norm_bins.sum())
        xdum = np.linspace(adm.min(), adm.max(), 10)
        linew_ax0 = 0.6
        ax_pishift[0, 0].step(adm, precess_bins / precess_bins.sum(), color=ca_c[ca], linewidth=linew_ax0,
                              label=ca, alpha=0.7)
        ax_pishift[0, 1].step(adm, spike_bins / spike_bins.sum(), color=ca_c[ca], linewidth=linew_ax0, alpha=0.7)
        ax_pishift[0, 2].step(adm, norm_bins / norm_bins.sum(), color=ca_c[ca], linewidth=linew_ax0, alpha=0.7)
        ax_pishift[0, 2].plot(xdum, xdum * pm + pc, linewidth=linew_ax0, color=ca_c[ca], alpha=0.7)
        ax_pishift[0, 2].annotate('p=%s' % (p2str(pval)), xy=(0.1, 0.9 - caid*0.12), xycoords='axes fraction', fontsize=legendsize, color=ca_c[ca])

        stat_record(stat_fn, False, "%s Spearman's correlation: $r_{s(%d)}=%0.2f, p=%s$, Pearson's: m=%0.2f, $r=%0.2f, p=%s$ " % \
                    (ca, precess_allcount, rho, p2str(pval), pm, pr, p2str(ppval)))

        # Plot fraction of neg slopes
        all_adiff = np.abs(np.concatenate(all_adiff))
        all_slopes = np.concatenate(all_slopes)
        frac_N = np.zeros(adiff_bins.shape[0] - 1)
        for edgei in range(adiff_bins.shape[0] - 1):
            idxtmp = np.where((all_adiff > adiff_bins[edgei]) & (all_adiff <= adiff_bins[edgei + 1]))[0]
            frac_N[edgei] = (all_slopes[idxtmp] < 0).mean()
        axR[0].step(adiff_bins[:-1], frac_N, color=ca_c[ca], where='pre', linewidth=0.75, label=ca, alpha=0.9)
        axR[0].set_xlabel(r'$\theta_{pass}$' + ' (rad)', fontsize=fontsize)
        axR[0].set_ylabel('Fraction of\nnegative slopes', fontsize=fontsize, labelpad=5)
        axR[0].set_xticks([0, np.pi / 2, np.pi])
        axR[0].set_xticklabels(['0', '', '$\pi$'])
        axR[0].set_yticks([0, 0.5, 1])
        axR[0].set_yticklabels(['0', '', '1'])
        axR[0].tick_params(labelsize=ticksize)
        axR[0].spines["top"].set_visible(False)
        axR[0].spines["right"].set_visible(False)
        customlegend(axR[0], bbox_to_anchor=[0.6, 0.55], loc='lower left', fontsize=legendsize)

        # Plot precess R
        caR = cadf[(~cadf['precess_R'].isna()) & numpass_mask]['precess_R'].to_numpy()
        allR.append(caR)
        rbins, redges = np.histogram(caR, bins=50)
        rbinsnorm = np.cumsum(rbins) / rbins.sum()
        axR[1].plot(midedges(redges), rbinsnorm, label=ca, c=ca_c[ca], linewidth=linew, alpha=0.9)

        # Plot Rate angles vs Precess angles
        nprecessmask = cadf['precess_df'].apply(lambda x: x['precess_exist'].sum()) > 1
        numpass_mask = numpass_mask[nprecessmask]
        numpass_low_mask = numpass_low_mask[nprecessmask]
        cadf = cadf[nprecessmask].reset_index(drop=True)
        rateangles = cadf['rate_angle'].to_numpy()
        precessangles = cadf['precess_angle'].to_numpy()
        precessangles_low = cadf['precess_angle_low'].to_numpy()

        ax_pishift[1, caid].scatter(rateangles[numpass_mask], precessangles[numpass_mask], marker='.', c=ca_c[ca], s=2)
        ax_pishift[1, caid].plot([0, np.pi], [np.pi, 2 * np.pi], c='k')
        ax_pishift[1, caid].plot([np.pi, 2 * np.pi], [0, np.pi], c='k')
        ax_pishift[1, 1].set_xlabel(r'$\theta_{rate}$' + ' (rad)', fontsize=fontsize)
        ax_pishift[1, caid].set_title(ca, fontsize=titlesize)
        ax_pishift[1, caid].set_xticks([0, np.pi, 2 * np.pi])
        ax_pishift[1, caid].set_xticklabels(['$0$', '$\pi$', '$2\pi$'], fontsize=fontsize)
        ax_pishift[1, caid].set_yticks([0, np.pi, 2 * np.pi])
        ax_pishift[1, caid].set_yticklabels(['$0$', '$\pi$', '$2\pi$'], fontsize=fontsize)
        ax_pishift[1, caid].spines["top"].set_visible(False)
        ax_pishift[1, caid].spines["right"].set_visible(False)

        # Plot Histogram: d(precess, rate)
        mask = (~np.isnan(rateangles)) & (~np.isnan(precessangles) & numpass_mask)
        adiff = cdiff(precessangles[mask], rateangles[mask])
        bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
        bins_norm = bins / np.sum(bins)
        l = bins_norm.max()
        ax_pishift[2, caid].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], zorder=0,
                                color=ca_c[ca])
        linewidth = 1
        mean_angle = shiftcyc_full2half(circmean(adiff))
        ax_pishift[2, caid].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k', zorder=3,
                                     arrowprops=dict(arrowstyle="->"))
        ax_pishift[2, caid].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
        ax_pishift[2, caid].scatter(0, 0, s=16, c='gray')
        ax_pishift[2, caid].annotate(r'$\theta_{rate}$', xy=(0.95, 0.525), xycoords='axes fraction', fontsize=fontsize + 1)
        ax_pishift[2, caid].spines['polar'].set_visible(False)
        ax_pishift[2, caid].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax_pishift[2, caid].set_yticks([0, l / 2])
        ax_pishift[2, caid].set_yticklabels([])
        ax_pishift[2, caid].set_xticklabels([])
        v_pval, v_stat = vtest(adiff, mu=np.pi)
        ax_pishift[2, caid].text(x=0.01, y=0.95, s='p=%s'%p2str(v_pval), fontsize=legendsize,
                                 transform=ax_pishift[2, caid].transAxes)
        stat_record(stat_fn, False, '%s, d(precess, rate), $V_{(%d)}=%0.2f, p=%s$' % (ca, bins.sum(), v_stat,
                                                                                      p2str(v_pval)))

        # Plot Histogram: d(precess_low, rate)
        mask_low = (~np.isnan(rateangles)) & (~np.isnan(precessangles_low) & numpass_low_mask)
        adiff = cdiff(precessangles_low[mask_low], rateangles[mask_low])
        bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
        bins_norm = bins / np.sum(bins)
        l = bins_norm.max()
        ax_pishift[3, caid].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], color=ca_c[ca], zorder=0)
        mean_angle = shiftcyc_full2half(circmean(adiff))
        ax_pishift[3, caid].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k', zorder=3,
                                     arrowprops=dict(arrowstyle="->"))
        ax_pishift[3, caid].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
        ax_pishift[3, caid].scatter(0, 0, s=16, c='gray')
        ax_pishift[3, caid].annotate(r'$\theta_{rate}$', xy=(0.95, 0.525), xycoords='axes fraction', fontsize=fontsize + 1)
        ax_pishift[3, caid].spines['polar'].set_visible(False)
        ax_pishift[3, caid].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax_pishift[3, caid].set_yticks([0, l / 2])
        ax_pishift[3, caid].set_yticklabels([])
        ax_pishift[3, caid].set_xticklabels([])
        v_pval, v_stat = vtest(adiff, mu=np.pi)
        ax_pishift[3, caid].text(x=0.01, y=0.95, s='p=%s'%p2str(v_pval), fontsize=legendsize,
                                 transform=ax_pishift[3, caid].transAxes)
        stat_record(stat_fn, False, '%s, d(precess_low, rate), $V_{(%d)}=%0.2f, p=%s$' % (ca, bins.sum(), v_stat,
                                                                                          p2str(v_pval)))

        # Export stats: Sig fraction
        ntotal_fields = cadf.shape[0]
        n_precessfields = cadf[~cadf['precess_angle'].isna()].shape[0]
        ca_shufp = cadf[(~cadf['precess_R_pval'].isna()) & numpass_mask]['precess_R_pval'].to_numpy()
        sig_num = np.sum(ca_shufp < 0.05)
        frac_amongall = sig_num / ntotal_fields
        frac_amongprecess = sig_num / n_precessfields
        bip_amongall = binom_test(x=sig_num, n=ntotal_fields, p=0.05, alternative='greater')
        bip_amongprecess = binom_test(x=sig_num, n=n_precessfields, p=0.05, alternative='greater')
        stat_record(stat_fn, False,
                    "Sig. Frac., %s, Among all: %d/%d=%0.3f, Binomial test $p=%s$. Among precess: %d/%d=%0.3f, Binomial test $p=%s$" % \
                    (ca, sig_num, ntotal_fields, frac_amongall, p2str(bip_amongall), sig_num, n_precessfields,
                     frac_amongprecess, p2str(bip_amongprecess)))

    # Overal asthestics
    try:
        kruskal_p, dunn_pvals, _, _, btwRtxt = my_kruskal_3samp(allR[0], allR[1], allR[2], 'CA1', 'CA2', 'CA3')
    except:
        kruskal_p = 1
        dunn_pvals = (1, 1, 1)
        btwRtxt = 'Not available'

    stat_record(stat_fn, False, 'R differences between CAs, %s' % (btwRtxt))
    axR[1].text(0.1, 0.02, 'CA1-3 diff.\np=%s' % (p2str(dunn_pvals[2])), fontsize=legendsize)
    axR[1].set_xlabel('R', fontsize=fontsize)
    axR[1].set_ylabel('Cumulative\nfield density', fontsize=fontsize, labelpad=5)
    axR[1].set_yticks([0, 0.5, 1])
    axR[1].set_yticklabels(['0', '', '1'])
    axR[1].set_xticks([0, 0.1, 0.2, 0.3])
    axR[1].tick_params(axis='both', which='major', labelsize=ticksize)
    axR[1].spines["top"].set_visible(False)
    axR[1].spines["right"].set_visible(False)

    ax_pishift[1, 0].set_ylabel(r'$\theta_{Precess}$' + ' (rad)', fontsize=fontsize)

    for ax_each in ax_pishift[0,]:
        ax_each.set_xticks([0, np.pi / 2, np.pi])
        ax_each.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        ax_each.set_ylim([0.01, 0.040])
        ax_each.tick_params(axis='both', which='major', labelsize=fontsize)
        ax_each.set_yticks([0.01, 0.02, 0.03, 0.04])
        ax_each.set_yticks([0.015, 0.025, 0.035, 0.04], minor=True)
        ax_each.spines["top"].set_visible(False)
        ax_each.spines["right"].set_visible(False)
    ax_pishift[0, 1].set_xlabel(r'$|d(\theta_{pass}, \theta_{rate})|$' + ' (rad)', fontsize=fontsize)
    ax_pishift[0, 0].set_ylabel('Density', fontsize=fontsize)


    ax_pishift[0, 0].set_title('Precession\n', fontsize=fontsize)
    ax_pishift[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax_pishift[0, 0].yaxis.get_offset_text().set_x(-0.1)

    ax_pishift[0, 1].set_title('Spike\n', fontsize=fontsize)
    plt.setp(ax_pishift[0, 1].get_yticklabels(), visible=False)

    ax_pishift[0, 2].set_title('Ratio\n', fontsize=fontsize)
    plt.setp(ax_pishift[0, 2].get_yticklabels(), visible=False)

    plt.setp(ax_pishift[1, 1].get_yticklabels(), visible=False)
    plt.setp(ax_pishift[1, 2].get_yticklabels(), visible=False)
    ax_pishift[2, 0].set_ylabel('All\npasses', fontsize=fontsize)
    ax_pishift[2, 0].yaxis.labelpad = 15
    ax_pishift[3, 0].set_ylabel('Low-spike\n passes', fontsize=fontsize)
    ax_pishift[3, 0].yaxis.labelpad = 15

def figure2(df, save_dir, tag=''):

    # Precession examples
    figpeg = plt.figure(figsize=(total_figw * 0.4, total_figw * 0.4 * 1.06667), facecolor=None)
    axpeg_w = 1/3
    axpeg_h = 1/4
    axpeg_xsqueeze, axpeg_ysqueeze = 0.20, 0.175
    axpeg_xbtw = axpeg_xsqueeze/3
    axpeg_btm_ybtw = axpeg_ysqueeze/3
    axpeg_top_yoffset = -0.11
    axpeg_btm_yoffset = -0.025

    axpeg = np.array([
        [figpeg.add_axes([axpeg_w*0 + axpeg_xsqueeze/2 + axpeg_xbtw, axpeg_h*3 + axpeg_ysqueeze/2 + axpeg_top_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*1 + axpeg_xsqueeze/2, axpeg_h*3 + axpeg_ysqueeze/2 + axpeg_top_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*2 + axpeg_xsqueeze/2 - axpeg_xbtw, axpeg_h*3 + axpeg_ysqueeze/2 + axpeg_top_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2])
         ],

        [figpeg.add_axes([axpeg_w*0 + axpeg_xsqueeze/2 + axpeg_xbtw, axpeg_h*2 + axpeg_ysqueeze/2 - axpeg_btm_ybtw + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*1 + axpeg_xsqueeze/2, axpeg_h*2 + axpeg_ysqueeze/2 - axpeg_btm_ybtw + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*2 + axpeg_xsqueeze/2 - axpeg_xbtw, axpeg_h*2 + axpeg_ysqueeze/2 - axpeg_btm_ybtw + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2])
         ],

        [figpeg.add_axes([axpeg_w*0 + axpeg_xsqueeze/2 + axpeg_xbtw, axpeg_h*1 + axpeg_ysqueeze/2 + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*1 + axpeg_xsqueeze/2, axpeg_h*1 + axpeg_ysqueeze/2 + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*2 + axpeg_xsqueeze/2 - axpeg_xbtw, axpeg_h*1 + axpeg_ysqueeze/2 + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2])
         ],

        [figpeg.add_axes([axpeg_w*0 + axpeg_xsqueeze/2 + axpeg_xbtw, axpeg_h*0 + axpeg_ysqueeze/2 + axpeg_btm_ybtw + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*1 + axpeg_xsqueeze/2, axpeg_h*0 + axpeg_ysqueeze/2 + axpeg_btm_ybtw + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2]),
         figpeg.add_axes([axpeg_w*2 + axpeg_xsqueeze/2 - axpeg_xbtw, axpeg_h*0 + axpeg_ysqueeze/2 + axpeg_btm_ybtw + axpeg_btm_yoffset, axpeg_w - axpeg_xsqueeze/2, axpeg_h - axpeg_ysqueeze/2])
         ],

    ])
    axpeg[2, 0].set_ylabel('Phase (rad)', fontsize=legendsize)
    # figpeg.text(0.03, 0.35, 'Phase (rad)', ha='center', rotation=90, fontsize=fontsize)
    figpeg.text(0.55, 0.005, 'Position', ha='center', fontsize=legendsize)
    plot_precession_examples(df, axpeg)




    # Neg-slope, Pi-shift, R
    figpi = plt.figure(figsize=(total_figw * 0.6, total_figw * 0.8), facecolor='white')
    axpi_w = 1/3
    axpi_h = 1/4
    axpi_top_xsqueeze, axpi_top_ysqueeze = 0.15, 0.15
    axpi_bottom_xsqueeze, axpi_bottom_ysqueeze = 0.25, 0.25
    axpi_top_xbtw = 0.025
    axpi_bottom_xbtw = 0.05
    axpi_bottom_leftshift = 0.025
    axpi_yoffset = -0.1
    scatter_yoffset = -0.085
    allpass_yoffset = -0.085
    axpi = np.array([
        [figpi.add_axes([0 + axpi_top_xsqueeze/2 + axpi_top_xbtw, axpi_h*3 + axpi_top_ysqueeze/2 + axpi_yoffset, axpi_w - axpi_top_xsqueeze/2, axpi_h - axpi_top_ysqueeze/2]),
         figpi.add_axes([axpi_w + axpi_top_xsqueeze/2, axpi_h*3 + axpi_top_ysqueeze/2 + axpi_yoffset, axpi_w - axpi_top_xsqueeze/2, axpi_h - axpi_top_ysqueeze/2]),
         figpi.add_axes([axpi_w*2 + axpi_top_xsqueeze/2 - axpi_top_xbtw, axpi_h*3 + axpi_top_ysqueeze/2 + axpi_yoffset, axpi_w - axpi_top_xsqueeze/2, axpi_h - axpi_top_ysqueeze/2])],

        [figpi.add_axes([0 + axpi_bottom_xsqueeze/2 + axpi_bottom_xbtw - axpi_bottom_leftshift, axpi_h*2 + axpi_bottom_ysqueeze/2 + axpi_yoffset + scatter_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2]),
         figpi.add_axes([axpi_w + axpi_bottom_xsqueeze/2 - axpi_bottom_leftshift, axpi_h*2 + axpi_bottom_ysqueeze/2 + axpi_yoffset + scatter_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2]),
         figpi.add_axes([axpi_w*2 + axpi_bottom_xsqueeze/2 - axpi_bottom_xbtw - axpi_bottom_leftshift, axpi_h*2 + axpi_bottom_ysqueeze/2 + axpi_yoffset + scatter_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2])],

        [figpi.add_axes([0 + axpi_bottom_xsqueeze/2 + axpi_bottom_xbtw - axpi_bottom_leftshift, axpi_h*1 + axpi_bottom_ysqueeze/2 + axpi_yoffset + allpass_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2], polar=True),
         figpi.add_axes([axpi_w + axpi_bottom_xsqueeze/2 - axpi_bottom_leftshift, axpi_h*1 + axpi_bottom_ysqueeze/2 + axpi_yoffset + allpass_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2], polar=True),
         figpi.add_axes([axpi_w*2 + axpi_bottom_xsqueeze/2 - axpi_bottom_xbtw - axpi_bottom_leftshift, axpi_h*1 + axpi_bottom_ysqueeze/2 + axpi_yoffset + allpass_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2], polar=True)],

        [figpi.add_axes([0 + axpi_bottom_xsqueeze/2 + axpi_bottom_xbtw - axpi_bottom_leftshift, axpi_h*0 + axpi_bottom_ysqueeze/2 + axpi_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2], polar=True),
         figpi.add_axes([axpi_w + axpi_bottom_xsqueeze/2 - axpi_bottom_leftshift, axpi_h*0 + axpi_bottom_ysqueeze/2 + axpi_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2], polar=True),
         figpi.add_axes([axpi_w*2 + axpi_bottom_xsqueeze/2 - axpi_bottom_xbtw - axpi_bottom_leftshift, axpi_h*0 + axpi_bottom_ysqueeze/2 + axpi_yoffset, axpi_w - axpi_bottom_xsqueeze/2, axpi_h - axpi_bottom_ysqueeze/2], polar=True)],
    ])



    figR = plt.figure(figsize=(total_figw * 0.32, total_figw * 0.4), facecolor='white')
    axR_w = 1
    axR_h = 1/2
    axR_xsqueeze, axR_ysqueeze = 0.6, 0.5
    axR_yoffset = -0.1
    axneg_yoffset = -0.01
    axR = np.array([
        figR.add_axes([0 + axR_xsqueeze/2, axR_h*1 + axR_ysqueeze/2 + axR_yoffset + axneg_yoffset, axR_w - axR_xsqueeze/2, axR_h - axR_ysqueeze/2]),
        figR.add_axes([0 + axR_xsqueeze/2, axR_h*0 + axR_ysqueeze/2 + axR_yoffset, axR_w - axR_xsqueeze/2, axR_h - axR_ysqueeze/2])
    ])

    plot_field_bestprecession(df, axpi, axR)

    figpi.savefig(join(save_dir, 'field_precess_pishift.pdf'), dpi=dpi)
    figpi.savefig(join(save_dir, 'field_precess_pishift.png'), dpi=dpi)
    figR.savefig(join(save_dir, 'field_precess_R.pdf'), dpi=dpi)
    figR.savefig(join(save_dir, 'field_precess_R.png'), dpi=dpi)
    figpeg.savefig(os.path.join(save_dir, 'precess_examples.pdf'), dpi=dpi)
    figpeg.savefig(os.path.join(save_dir, 'precess_examples.png'), dpi=dpi)

def plot_both_slope_offset(df, save_dir):
    import warnings
    warnings.filterwarnings("ignore")
    def norm_div(target_hist, divider_hist):
        target_hist_norm = target_hist / divider_hist.reshape(-1, 1)
        target_hist_norm[np.isnan(target_hist_norm)] = 0
        target_hist_norm[np.isinf(target_hist_norm)] = 0
        target_hist_norm = target_hist_norm / np.sum(target_hist_norm) * np.sum(target_hist)
        return target_hist_norm

    selected_adiff = np.linspace(0, np.pi, 6)  # 20
    adiff_ticks = [0, np.pi / 2, np.pi]
    adiff_ticksl = ['0', '', '$\pi$']
    adiff_label = r'$|d|$'


    offset_slicerange = (0, 2 * np.pi)
    offset_bound = (0, 2 * np.pi)

    offset_slicegap = 0.017  # 0.007

    slope_slicerange = (-2 * np.pi, 0)
    slope_bound = (-2 * np.pi, 0)
    slope_slicegap = 0.01  # 0.007

    adiff_edges = np.linspace(0, np.pi, 100)
    offset_edges = np.linspace(offset_bound[0], offset_bound[1], 100)
    slope_edges = np.linspace(slope_bound[0], slope_bound[1], 100)

    figl = total_figw / 6
    fig1size = (figl * 3 * 0.95, figl * 2.8)  # 0.05 as space for colorbar
    fig2size = (figl * 3 * 0.95, figl * 2.8)
    fig3size = (figl * 3 * 0.95, figl * 2.8)

    stat_fn = 'fig3_slopeoffset.txt'
    stat_record(stat_fn, True, 'Slope, Onset & SpikePhase of phase precession\n')
    fig1, ax1 = plt.subplots(2, 3, figsize=fig1size, sharey='row')  # Density
    fig2, ax2 = plt.subplots(2, 3, figsize=fig2size, sharey='row')  # Marginals
    fig3, ax3 = plt.subplots(2, 3, figsize=fig3size, sharey='row')  # Aver curves + spike phase

    # Construct pass df
    refangle_key = 'rate_angle'
    passdf_dict = {'ca': [], 'anglediff': [], 'slope': [], 'onset': [], 'pass_nspikes': []}
    spikedf_dict = {'ca': [], 'anglediff': [], 'phasesp': []}
    dftmp = df[(~df[refangle_key].isna())].reset_index()

    for i in range(dftmp.shape[0]):
        allprecess_df = dftmp.loc[i, 'precess_df']
        if allprecess_df.shape[0] < 1:
            continue
        ca = dftmp.loc[i, 'ca']
        precessdf = allprecess_df[allprecess_df['precess_exist']].reset_index(drop=True)
        numprecess = precessdf.shape[0]
        if numprecess < 1:
            continue
        ref_angle = dftmp.loc[i, refangle_key]
        anglediff_tmp = cdiff(precessdf['mean_anglesp'].to_numpy(), ref_angle)
        phasesp_tmp = np.concatenate(precessdf['phasesp'].to_list())

        passdf_dict['ca'].extend([ca] * numprecess)
        passdf_dict['anglediff'].extend(anglediff_tmp)
        passdf_dict['slope'].extend(precessdf['rcc_m'])
        passdf_dict['onset'].extend(precessdf['rcc_c'])
        passdf_dict['pass_nspikes'].extend(precessdf['pass_nspikes'])

        spikedf_dict['ca'].extend([ca] * phasesp_tmp.shape[0])
        spikedf_dict['anglediff'].extend(repeat_arr(anglediff_tmp, precessdf['pass_nspikes'].to_numpy().astype(int)))
        spikedf_dict['phasesp'].extend(phasesp_tmp)
    passdf = pd.DataFrame(passdf_dict)
    passdf['slope_piunit'] = passdf['slope'] * 2 * np.pi
    spikedf = pd.DataFrame(spikedf_dict)

    # Plot and analysis for CA1, CA2 and CA3
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
        pass_cadf = passdf[passdf['ca'] == ca].reset_index(drop=True)
        spike_cadf = spikedf[spikedf['ca'] == ca].reset_index(drop=True)

        absadiff_pass = np.abs(pass_cadf['anglediff'].to_numpy())
        offset = pass_cadf['onset'].to_numpy()
        slope = pass_cadf['slope_piunit'].to_numpy()

        absadiff_spike = np.abs(spike_cadf['anglediff'].to_numpy())
        phase_spike = spike_cadf['phasesp'].to_numpy()

        # # Plot density, slices

        # 1D spike hisotgram
        spikes_bins, spikes_edges = np.histogram(absadiff_spike, bins=adiff_edges)

        # 2D slope/offset histogram
        offset_bins, offset_xedges, offset_yedges = np.histogram2d(absadiff_pass, offset,
                                                                   bins=(adiff_edges, offset_edges))
        slope_bins, slope_xedges, slope_yedges = np.histogram2d(absadiff_pass, slope,
                                                                bins=(adiff_edges, slope_edges))
        offset_xedm, offset_yedm = midedges(offset_xedges), midedges(offset_yedges)
        slope_xedm, slope_yedm = midedges(slope_xedges), midedges(slope_yedges)
        offset_normbins = norm_div(offset_bins, spikes_bins)
        slope_normbins = norm_div(slope_bins, spikes_bins)

        # Unbinning
        offset_adiff, offset_norm = unfold_binning_2d(offset_normbins, offset_xedm, offset_yedm)
        slope_adiff, slope_norm = unfold_binning_2d(slope_normbins, slope_xedm, slope_yedm)

        # Linear-circular regression
        regress = rcc(offset_adiff, offset_norm)
        offset_m, offset_c, offset_rho, offset_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']
        regress = rcc(slope_adiff, slope_norm)
        slope_m, slope_c, slope_rho, slope_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']
        slope_c = slope_c - 2 * np.pi
        stat_record(stat_fn, False, '\n'+ ('='*10) + '%s'%(ca) + ('='*10))
        stat_record(stat_fn, False, 'LC_Regression %s Onset-adiff $r_{(%d)}=%0.3f, p=%s$' % (
            ca, offset_bins.sum(), offset_rho, p2str(offset_p)))
        stat_record(stat_fn, False,
                    'LC_Regression %s Slope-adiff $r_{(%d)}=%0.3f, p=%s$' % (ca, slope_bins.sum(), slope_rho, p2str(slope_p)))

        # Density
        offset_xx, offset_yy, offset_zz = linear_circular_gauss_density(offset_adiff, offset_norm,
                                                                        cir_kappa=4 * np.pi, lin_std=0.2, xbins=50,
                                                                        ybins=50, xbound=(0, np.pi),
                                                                        ybound=offset_bound)
        slope_xx, slope_yy, slope_zz = linear_circular_gauss_density(slope_adiff, slope_norm,
                                                                     cir_kappa=1 * np.pi, lin_std=0.3, xbins=50,
                                                                     ybins=50, xbound=(0, np.pi),
                                                                     ybound=slope_bound)

        # Plot offset density
        cmap = 'Blues'
        _, _, edgtmp, _ = ax1[0, caid].hist2d(offset_adiff, offset_norm,
                                              bins=(np.linspace(0, np.pi, 36), np.linspace(0, 2 * np.pi, 36)),
                                              density=True,
                                              cmap=cmap)

        regressed = (offset_c + offset_xedm * offset_m)
        regress_width = np.pi * offset_m
        ax1[0, caid].plot(offset_xedm, regressed, c='r', linewidth=0.85)
        ax1[0, caid].plot([0, np.pi], [offset_c - regress_width * 2.5, offset_c - regress_width * 2.5], c='purple',
                          linewidth=0.5)
        ax1[0, caid].plot([0, np.pi], [offset_c + regress_width * 3, offset_c + regress_width * 3], c='purple',
                          linewidth=0.5)
        pvalstr = 'p=%0.4f' % offset_p if offset_p > 1e-4 else 'p<0.0001'
        ax1[0, caid].text(0.02, 5, pvalstr, fontsize=legendsize)
        ax1[0, caid].set_xticks(adiff_ticks)
        ax1[0, caid].set_xticklabels(adiff_ticksl)
        ax1[0, caid].set_yticks([0, np.pi, 2*np.pi])
        ax1[0, caid].set_yticklabels(['0', '', '$2\pi$'])
        ax1[0, caid].tick_params(labelsize=ticksize)
        ax1[0, caid].set_title(ca, fontsize=legendsize)
        ax1[0, 1].set_xlabel(adiff_label, fontsize=legendsize)

        # Plot slope density
        regressed = (slope_c + slope_xedm * slope_m)
        regress_width = np.pi * slope_m
        _, _, edgtmp, _ = ax1[1, caid].hist2d(slope_adiff, slope_norm,
                                              bins=(np.linspace(0, np.pi, 36), np.linspace(-2 * np.pi, 0, 36)),
                                              density=True,
                                              cmap=cmap)
        ax1[1, caid].plot(slope_xedm, regressed, c='r', linewidth=0.75)
        ax1[1, caid].plot([0, np.pi], [slope_c - regress_width * 0.5, slope_c - regress_width * 0.5], c='purple',
                          linewidth=0.5)
        ax1[1, caid].plot([0, np.pi], [slope_c + regress_width * 1.5, slope_c + regress_width * 1.5], c='purple',
                          linewidth=0.5)
        pvalstr = 'p=%0.4f' % slope_p if slope_p > 1e-4 else 'p<0.0001'
        ax1[1, caid].text(0.02, -3.5 * np.pi / 2, pvalstr, fontsize=legendsize)
        ax1[1, caid].set_xticks(adiff_ticks)
        ax1[1, caid].set_xticklabels(adiff_ticksl)
        ax1[1, caid].set_yticks([-2 * np.pi, -np.pi, 0])
        ax1[1, caid].set_yticklabels(['$-2\pi$', '', '0'])
        ax1[1, caid].tick_params(labelsize=ticksize)
        ax1[1, 1].set_xlabel(adiff_label, fontsize=legendsize)

        # # Marginals
        plot_marginal_slices(ax2[0, caid], offset_xx, offset_yy, offset_zz,
                             selected_adiff,
                             offset_slicerange, offset_slicegap)

        ax2[0, caid].set_xticks([2, 4, 6])
        ax2[0, caid].set_xticklabels(['2', '4', '6'])
        ax2[0, caid].set_xlim(2, 6)
        ax2[0, 1].set_xlabel('Onset phase (rad)', fontsize=legendsize)
        ax2[0, caid].tick_params(labelsize=ticksize)
        ax2[0, caid].set_title(ca, fontsize=legendsize)


        plot_marginal_slices(ax2[1, caid], slope_xx, slope_yy, slope_zz,
                             selected_adiff, slope_slicerange, slope_slicegap)
        ax2[1, caid].set_xticks([-2 * np.pi, -np.pi, 0])
        ax2[1, caid].set_xticklabels(['$-2\pi$', '', '0'])
        ax2[1, 1].set_xlabel('Slope (rad)', fontsize=legendsize)
        ax2[1, caid].tick_params(labelsize=ticksize)
        ax2[1, caid].set_xlim(-2 * np.pi, 0)

        # # Plot average precession curves
        low_mask = absadiff_pass < (np.pi / 6)
        high_mask = absadiff_pass > (np.pi - np.pi / 6)
        slopes_high_all, offsets_high_all = slope[high_mask], offset[high_mask]
        slopes_low_all, offsets_low_all = slope[low_mask], offset[low_mask]

        tag1, tag2 = r'Along-$\theta_{rm rate}$', r'Against-$\theta_{rm rate}$'
        pval_slope, _, slope_descrips, slopetxt = my_kruskal_2samp(slopes_low_all, slopes_high_all, tag1, tag2)
        (mdn_slopel, lqr_slopel, hqr_slopel), (mdn_slopeh, lqr_slopeh, hqr_slopeh) = slope_descrips
        mean_slopel, mean_slopeh = np.mean(slopes_low_all), np.mean(slopes_high_all)
        sem_slopel = np.std(slopes_low_all)/np.sqrt(slopes_low_all.shape[0])
        sem_slopeh = np.std(slopes_high_all)/np.sqrt(slopes_high_all.shape[0])

        pval_offset, _, offset_descrips, offsettxt = my_ww_2samp(offsets_low_all, offsets_high_all, tag1, tag2)
        (cmean_offsetl, sem_offsetl), (cmean_offseth, sem_offseth) = offset_descrips

        xdum = np.linspace(0, 1, 10)
        high_agg_ydum = mdn_slopeh * xdum + cmean_offseth
        low_agg_ydum = mdn_slopel * xdum + cmean_offsetl
        slope_stattext = '%s, mdn=%0.2f, mean+-SEM= %0.2f $\pm$ %0.2f\n'%(tag1, mdn_slopel, mean_slopel, sem_slopel)
        slope_stattext += '%s, mdn=%0.2f, mean+-SEM= %0.2f $\pm$ %0.2f'%(tag2, mdn_slopeh, mean_slopeh, sem_slopeh)
        offset_stattext = '%s, mean+-SEM =%0.2f $\pm$ %0.2f\n'%(tag1, cmean_offsetl, sem_offsetl)
        offset_stattext += '%s, mean+-SEM =%0.2f $\pm$ %0.2f'%(tag2, cmean_offseth, sem_offseth)
        stat_record(stat_fn, False, '===== Average precession curves ====' )
        stat_record(stat_fn, False, 'Slope\n%s\n%s'%(slope_stattext, slopetxt))
        stat_record(stat_fn, False, 'Onset\n%s\n%s'%(offset_stattext, offsettxt))

        ax3[0, caid].plot(xdum, high_agg_ydum, c='lime')
        ax3[0, caid].plot(xdum, low_agg_ydum, c='darkblue')
        ax3[0, caid].annotate('$p_s$'+'=%s'% (p2str(pval_slope)), xy=(0.015, 0.2 + 0.03), xycoords='axes fraction', fontsize=legendsize-1)
        ax3[0, caid].annotate('$p_o$'+'=%s'% (p2str(pval_offset)), xy=(0.015, 0.035 + 0.03), xycoords='axes fraction', fontsize=legendsize-1)
        ax3[0, caid].spines["top"].set_visible(False)
        ax3[0, caid].spines["right"].set_visible(False)
        ax3[0, caid].set_title(ca, fontsize=legendsize)

        # # high- and low-|d| Spike phases
        low_mask_sp = absadiff_spike < (np.pi / 6)
        high_mask_sp = absadiff_spike > (np.pi - np.pi / 6)
        phasesph = phase_spike[high_mask_sp]
        phasespl = phase_spike[low_mask_sp]
        mean_phasesph = circmean(phasesph)
        mean_phasespl = circmean(phasespl)
        fstat, k_pval = circ_ktest(phasesph, phasespl)
        p_ww, _, phasesp_descript, ww_txt = my_ww_2samp(phasespl, phasesph, tag1, tag2)
        ((mean_phasespl, sem_phasespl), (mean_phasesph, sem_phasesph)) = phasesp_descript
        phasesph_bins, phasesp_edges = np.histogram(phasesph, bins=36, range=(-np.pi, np.pi))
        phasespl_bins, _ = np.histogram(phasespl, bins=36, range=(-np.pi, np.pi))
        maxl = max((phasespl_bins/phasespl_bins.sum()).max(), (phasesph_bins/phasesph_bins.sum()).max())
        ax3[1, caid].step(phasesp_edges[:-1], phasesph_bins/phasesph_bins.sum(), where='pre', color='lime', linewidth=0.75)
        ax3[1, caid].step(phasesp_edges[:-1], phasespl_bins/phasespl_bins.sum(), where='pre', color='darkblue', linewidth=0.75)
        ax3[1, caid].axvline(mean_phasesph, ymin=0.9, ymax=1.1, color='lime', linewidth=0.75)
        ax3[1, caid].axvline(mean_phasespl, ymin=0.9, ymax=1.1, color='darkblue', linewidth=0.75)

        ax3[1, caid].annotate(r'$p$=%s' % (p2str(p_ww)), xy=(0.025, 0.045), xycoords='axes fraction', fontsize=legendsize)
        ax3[1, caid].set_xlim(-np.pi, np.pi)
        ax3[1, caid].set_xticks([-np.pi, 0, np.pi])
        ax3[1, caid].set_xticklabels(['$-\pi$', '0', '$\pi$'])
        ax3[1, caid].set_yticks([0, 0.02, 0.04])
        ax3[1, caid].set_yticklabels(['0', '2', '4'])
        ax3[1, caid].set_ylim(0, 0.05)

        ax3[1, caid].tick_params(labelsize=ticksize)
        ax3[1, caid].spines["top"].set_visible(False)
        ax3[1, caid].spines["right"].set_visible(False)
        ax3[1, 1].set_xlabel('Phase (rad)', fontsize=legendsize)
        ax3[1, 0].set_ylabel('Normalized\nspike count', fontsize=legendsize)

        phasesp_stattxt = '============Spike phase mean difference===============\n'
        phasesp_stattxt += '%s: mean+-SEM = %0.2f $\pm$ %0.2f\n'%(tag1, mean_phasespl, sem_phasespl)
        phasesp_stattxt += '%s: mean+-SEM = %0.2f $\pm$ %0.2f\n' %(tag2, mean_phasesph, sem_phasesph)
        phasesp_stattxt += ww_txt
        stat_record(stat_fn, False, phasesp_stattxt)

    # Asthestic for Density (ax1)
    ax1[0, 0].set_ylabel('Onset phase (rad)', fontsize=legendsize)
    ax1[1, 0].set_ylabel('Slope (rad)', fontsize=legendsize)


    # Asthestic for Marginals
    for ax_each in np.append(ax2[0,], ax2[1,]):
        ax_each.set_yticks([])
        ax_each.grid(False, axis='y')
        ax_each.spines["top"].set_visible(False)
        ax_each.spines["right"].set_visible(False)
        ax_each.spines["left"].set_visible(False)
    fig2.text(0.05, 0.35, 'Marginal density\n of precession', rotation=90, fontsize=legendsize)

    # Asthestics for average curves
    for ax_each in ax3[0,]:
        ax_each.set_xlim(0, 1)
        ax_each.set_ylim(-np.pi - 1,  np.pi+0.75)
        ax_each.set_yticks([-np.pi, 0, np.pi])
        ax_each.set_yticklabels(['$-\pi$', '0', '$\pi$'])
        ax_each.tick_params(labelsize=ticksize)
    ax3[0, 1].set_xlabel('Position', fontsize=legendsize)
    ax3[0, 0].set_ylabel('Phase (rad)', fontsize=legendsize)
    ax3[1, 0].annotate(r'$\times 10^{-2}$', xy=(0.02, 0.9), xycoords='axes fraction', fontsize=ticksize)



    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.brg,
                               norm=plt.Normalize(vmin=selected_adiff.min(), vmax=selected_adiff.max()))
    fig_colorbar = plt.figure(figsize=(total_figw * 0.1, figl * 2))

    cbar_ax = fig_colorbar.add_axes([0.1, 0.1, 0.1, 0.8])
    cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
    cb.set_ticks(adiff_ticks)
    cb.set_ticklabels(adiff_ticksl)
    cb.set_label(adiff_label, fontsize=legendsize, labelpad=-5)

    # # Saving
    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.4, hspace=1.2, left=0.2)

    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0.4, hspace=1.2, left=0.2)

    fig3.tight_layout()
    fig3.subplots_adjust(wspace=0.4, hspace=1.2, left=0.2)

    for figext in ['png', 'eps']:
        fig_colorbar.savefig(os.path.join(save_dir, 'adiff_colorbar.%s' % (figext)), dpi=dpi)
        fig1.savefig(os.path.join(save_dir, 'Densities.%s' % (figext)), dpi=dpi)
        fig2.savefig(os.path.join(save_dir, 'Marginals.%s' % (figext)), dpi=dpi)
        fig3.savefig(os.path.join(save_dir, 'AverCurves_SpikePhase.%s' % (figext)), dpi=dpi)
    return None


def plot_precession_pvals(singledf, savedir):
    nap_thresh = 1
    figl = total_figw / 2
    fig, ax = plt.subplots(figsize=(figl, figl))

    cafrac = []
    nall = []
    nsig = []
    for caid, (ca, cadf) in enumerate(singledf.groupby('ca')):
        thisdf = cadf[(cadf['numpass_at_precess'] >= nap_thresh) & (~cadf['precess_R_pval'].isna())].reset_index(
            drop=True)

        nall.append(thisdf['precess_R_pval'].shape[0])
        nsig.append(np.sum(thisdf['precess_R_pval'] < 0.05))
        cafrac.append(np.mean(thisdf['precess_R_pval'] < 0.05))
        logpval = np.log10(thisdf['precess_R_pval'].to_numpy())

        logpval[np.isinf(logpval)] = np.log10(1e-2)
        ax.hist(logpval, bins=30, cumulative=True, density=True, histtype='step', color=ca_c[ca], label=ca)

    ax.vlines(np.log10(0.05), ymin=0, ymax=1, label='p=0.05')
    ax.set_xlabel('log10(pval)', fontsize=fontsize)
    ax.set_ylabel('Cumulative Freq.', fontsize=fontsize)
    ax.set_title('CA1=%d/%d=%0.3f\nCA2=%d/%d=%0.3f\nCA3=%d/%d=%0.3f' % (
        nsig[0], nall[0], cafrac[0], nsig[1], nall[1], cafrac[1], nsig[2], nall[2], cafrac[2]
    ), fontsize=legendsize)

    customlegend(ax, fontsize=legendsize, loc='upper left')
    fig.tight_layout()
    fig.savefig(join(savedir, 'precession_significance.png'), dpi=dpi)


def check_pass_nspikes(df):
    for ca, cadf in df.groupby('ca'):
        allprecessdf = pd.concat(cadf['precess_df'].to_list(), axis=0, ignore_index=True)

        precessdf = allprecessdf[allprecessdf['precess_exist']]
        print('\n%s' % ca)
        print(precessdf['pass_nspikes'].describe())
        print(allprecessdf['pass_nspikes'].describe())


def recompute_precessangle(df, kappa, CA3nsp_thresh):
    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = kappa
    # nspikes_stats = {'CA1': 7, 'CA2': 7, 'CA3': 8}
    nspikes_stats = {'CA1': 5, 'CA2': 4, 'CA3': CA3nsp_thresh}
    precess_angle_low_list = []
    numpass_at_precess_low_list = []
    for i in range(df.shape[0]):
        print('\rReprocessing %d/%d' % (i, df.shape[0]), flush=True, end='')
        ca, precess_df = df.loc[i, ['ca', 'precess_df']]

        # Precession - low-spike passes
        ldf = precess_df[precess_df['pass_nspikes'] < nspikes_stats[ca]]  # 25% quantile
        if (ldf.shape[0] > 0) and (ldf['precess_exist'].sum() > 0):
            precess_angle_low, _, _ = compute_precessangle(pass_angles=ldf['mean_anglesp'].to_numpy(),
                                                           pass_nspikes=ldf['pass_nspikes'].to_numpy(),
                                                           precess_mask=ldf['precess_exist'].to_numpy(),
                                                           kappa=kappa_precess, bins=None)
            _, _, postdoc_dens_low = compute_precessangle(pass_angles=ldf['mean_anglesp'].to_numpy(),
                                                          pass_nspikes=ldf['pass_nspikes'].to_numpy(),
                                                          precess_mask=ldf['precess_exist'].to_numpy(),
                                                          kappa=None, bins=aedges_precess)
            # spikes_angles = np.concatenate(ldf['spikeangle'].to_list())
            # precess_angle_low, _, _ = compute_precessangle_anglesp(pass_angles=ldf['mean_anglesp'].to_numpy(),
            #                                                        anglesp=spikes_angles,
            #                                                        precess_mask=ldf['precess_exist'].to_numpy(),
            #                                                        kappa=kappa_precess, bins=None)
            # _, _, postdoc_dens_low = compute_precessangle_anglesp(pass_angles=ldf['mean_anglesp'].to_numpy(),
            #                                                       anglesp=spikes_angles,
            #                                                       precess_mask=ldf['precess_exist'].to_numpy(),
            #                                                       kappa=None, bins=aedges_precess)



            (_, passbins_p_low, passbins_np_low, _) = postdoc_dens_low
            all_passbins_low = passbins_p_low + passbins_np_low
            numpass_at_precess_low = get_numpass_at_angle(target_angle=precess_angle_low, aedge=aedges_precess,
                                                          all_passbins=all_passbins_low)
        else:
            precess_angle_low = None
            numpass_at_precess_low = None

        precess_angle_low_list.append(precess_angle_low)
        numpass_at_precess_low_list.append(numpass_at_precess_low)
    print()
    df['precess_angle_low'] = precess_angle_low_list
    df['numpass_at_precess_low'] = numpass_at_precess_low_list
    return df


def main():
    # # Setting
    data_pth = 'results/emankin/singlefield_df.pickle'
    save_dir = 'writting/figures'
    figure_dir = 'writting/figures/'
    for i in range(1, 4):
        os.makedirs(join(figure_dir, 'fig%d'%(i)), exist_ok=True)

    # # Loading
    df = pd.read_pickle(data_pth)
    # df = recompute_precessangle(df, kappa=1, CA3nsp_thresh=5)
    # check_pass_nspikes(df)

    # # Analysis
    omniplot_singlefields(df, save_dir=join(save_dir, 'fig1'))
    # figure2(df, save_dir=join(save_dir, 'fig2'))
    # plot_both_slope_offset(df=df, save_dir=join(save_dir, 'fig3'))


    # # Archive
    # plot_precession_pvals(singlefield_df, save_dir)


if __name__ == '__main__':
    main()
