import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import join
from matplotlib import cm

from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import ttest_ind, binom_test

from common.linear_circular_r import rcc
from common.stattests import p2str, stat_record, my_chi_2way, \
    fdr_bh_correction, my_kruskal_3samp, my_kruskal_2samp, my_chi_1way, my_ttest_1samp, my_fisher_2way, my_ww_2samp, \
    angular_dispersion_test, circ_ktest
from common.comput_utils import midedges, circular_density_1d, shiftcyc_full2half, segment_passes
from common.script_wrappers import DirectionalityStatsByThresh
from common.visualization import plot_correlogram, customlegend, plot_overlaid_polormesh
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

figext = 'png'
# figext = 'eps'


def omniplot_pairfields(expdf, save_dir=None):
    # # Initialization of stats
    stat_fn = 'fig4_pair_directionality.txt'
    stat_record(stat_fn, True)

    # # Initialization of figures
    fullcol_figw = 6.3
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
    spike_threshs = np.arange(0, 201, 10)
    stats_getter = DirectionalityStatsByThresh('num_spikes_pair', 'rate_R_pvalp', 'rate_Rp')
    expdf['border'] = expdf['border1'] | expdf['border2']
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
        ax_popR[caid].scatter(x=cadf_b['num_spikes_pair'], y=cadf_b['rate_Rp'], c=popR_B_c, s=0.1, marker=popR_B_mark, alpha=0.7)
        ax_popR[caid].scatter(x=cadf_nb['num_spikes_pair'], y=cadf_nb['rate_Rp'], c=popR_NB_c, s=0.1, marker=popR_NB_mark, alpha=0.7)
        ax_popR[caid].axvspan(xmin=0, xmax=40, color='0.9', zorder=0)
        ax_popR[caid].set_title(ca, fontsize=titlesize)
        ax_popR_den[caid].hist(cadf_b['rate_Rp'], bins=redges, orientation='horizontal', histtype='step', density=True, color=popR_B_c, linewidth=linew)
        ax_popR_den[caid].hist(cadf_nb['rate_Rp'], bins=redges, orientation='horizontal', histtype='step', density=True, color=popR_NB_c, linewidth=linew)
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
            stat_txt = '%s, Thresh=%d, Binomial test, greater than p=0.05, %d/%d=%0.4f, $p%s$' % (
                ca, ntresh, signum_all, n_all, signum_all / n_all, p2str(p_binom))
            stat_record(stat_fn, False, stat_txt)
            # ax[1, caid].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize, color=ca_c[ca])

    # # Statistical test
    chipval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    fishpval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    rspval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    for idx, ntresh in enumerate(spike_threshs):
        stat_record(stat_fn, False, '======= Threshold=%d ======' % (ntresh))
        # # Border vs non-border for each CA
        for ca in ['CA%d' % i for i in range(1, 4)]:
            cad_b = data_dict[ca]['border']
            cad_nb = data_dict[ca]['nonborder']

            # KW test for border median R
            rs_bord_pR, (border_n, nonborder_n), mdns, rs_txt = my_kruskal_2samp(cad_b['allR'][idx],
                                                                                 cad_nb['allR'][idx], 'border',
                                                                                 'nonborder')
            stat_record(stat_fn, False, "%s, Median R, border vs non-border: %s" % (ca, rs_txt))
            rspval_dict[ca + '_be'].append(rs_bord_pR)

            # Chisquared test for border fractions
            contin = pd.DataFrame({'border': [cad_b['shift_signum'][idx],
                                              cad_b['shift_nonsignum'][idx]],
                                   'nonborder': [cad_nb['shift_signum'][idx],
                                                 cad_nb['shift_nonsignum'][idx]]}).to_numpy()
            chi_pborder, _, txt_chiborder = my_chi_2way(contin)
            _, _, fishtxt = my_fisher_2way(contin)
            stat_record(stat_fn, False,
                        "%s, Significant fraction, border vs non-border: %s, %s" % (ca, txt_chiborder, fishtxt))
            chipval_dict[ca + '_be'].append(chi_pborder)

        # # Between CAs for All
        bcase = 'all'
        ca1d, ca2d, ca3d = data_dict['CA1'][bcase], data_dict['CA2'][bcase], data_dict['CA3'][bcase]

        # 3-sample KW test for median R, CA1 vs CA2 vs CA3
        kruskal_ps, dunn_pvals, (n1, n2, n3), _, kw3txt = my_kruskal_3samp(ca1d['allR'][idx], ca2d['allR'][idx],
                                                                           ca3d['allR'][idx], 'CA1', 'CA2', 'CA3')

        stat_record(stat_fn, False, 'Median R between CAs, %s' % (kw3txt))
        rs_p12R, rs_p23R, rs_p13R = dunn_pvals
        rspval_dict['all_CA12'].append(rs_p12R)
        rspval_dict['all_CA23'].append(rs_p23R)
        rspval_dict['all_CA13'].append(rs_p13R)

        # Chisquared test for sig fractions between CAs
        for canum1, canum2 in ((1, 2), (2, 3), (1, 3)):
            cafirst, casecond = 'CA%d' % (canum1), 'CA%d' % (canum2)
            cadictfirst, cadictsecond = data_dict[cafirst][bcase], data_dict[casecond][bcase]
            contin_btwca = pd.DataFrame(
                {cafirst: [cadictfirst['shift_signum'][idx], cadictfirst['shift_nonsignum'][idx]],
                 casecond: [cadictsecond['shift_signum'][idx], cadictsecond['shift_nonsignum'][idx]]}).to_numpy()
            chi_pbtwca, _, txt_chibtwca = my_chi_2way(contin_btwca)

            fish_pbtwca, _, fishtxt_btwca = my_fisher_2way(contin_btwca)
            stat_record(stat_fn, False, "Between CAs, Significant fraction, %s vs %s: %s, %s" % (
                cafirst, casecond, txt_chibtwca, fishtxt_btwca))
            chipval_dict['all_CA%d%d' % (canum1, canum2)].append(chi_pbtwca)
            fishpval_dict['all_CA%d%d' % (canum1, canum2)].append(fish_pbtwca)
        chiqs = fdr_bh_correction(
            np.array([chipval_dict['all_CA12'][-1], chipval_dict['all_CA23'][-1], chipval_dict['all_CA13'][-1]]))
        fishqs = fdr_bh_correction(
            np.array([fishpval_dict['all_CA12'][-1], fishpval_dict['all_CA23'][-1], fishpval_dict['all_CA13'][-1]]))
        stat_record(stat_fn, False, r"Chisquare Benjamini-Hochberg correction, $p12=%s$, $p23=%s$, $p13=%s$" % (
            p2str(chiqs[0]), p2str(chiqs[1]), p2str(chiqs[2])))
        stat_record(stat_fn, False, r"Fisher Benjamini-Hochberg correction, $p12=%s$, $p23=%s$, $p13=%s$" % (
            p2str(fishqs[0]), p2str(fishqs[1]), p2str(fishqs[2])))

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
    ax_pval[1].set_xlabel('Spike-pair threshold')
    ax_pval[1].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_pval[1].set_title('Compare CAs', fontsize=titlesize)
    customlegend(ax_pval[1], fontsize=legendsize)


    # # Asthestics of plotting
    # ax_popR
    ax_popR[0].scatter(-1, -1, c=popR_B_c, marker=popR_B_mark, label='B', s=8)
    ax_popR[0].scatter(-1, -1, c=popR_NB_c, marker=popR_NB_mark, label='N-B', s=8)
    ax_popR[0].set_ylabel('R', fontsize=fontsize)
    customlegend(ax_popR[0], linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')
    ax_popR[1].set_xlabel('Spike-pair count', fontsize=fontsize)
    for ax_i, ax_popR_each in enumerate(ax_popR):
        ax_popR_each.set_xlim(-20, 400)
        ax_popR_each.set_xticks(np.arange(0, 401, 200))
        ax_popR_each.set_xticks(np.arange(0, 401, 100), minor=True)
        ax_popR_each.set_ylim(0, 1.05)
        ax_popR_each.set_yticks(np.arange(0, 1.1, 0.5))
        if ax_i == 0:
            ax_popR_each.set_yticklabels(['0', '', '1'])
        else:
            ax_popR_each.set_yticklabels(['', '', ''])
        ax_popR_each.set_yticks(np.arange(0, 1.1, 0.1), minor=True)

    # ax_popR_den
    for i in range(3):
        ax_popR_den[i].set_ylim(0, 1.05)
        ax_popR_den[i].axis('off')



    # ax_directR
    ax_directR[0].set_ylabel('Median R', fontsize=fontsize)
    customlegend(ax_directR[0], linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')
    for i in range(3):
        ax_directR[i].set_xticks([0, 100, 200])
        ax_directR[i].set_xticklabels([''] * 3)
        ax_directR[i].set_ylim(0, 0.8)
        ax_directR[i].set_yticks(np.arange(0, 0.81, 0.2))
        ax_directR[i].set_yticks(np.arange(0, 0.81, 0.1), minor=True)
        ax_directR[i].set_yticklabels(['0.0', '', '0.4', '', '0.8'])
        if i != 0:
            ax_directR[i].set_yticklabels(['']*5)

    # ax_directFrac
    ax_directFrac[0].set_ylabel('Significant\n Fraction', fontsize=fontsize)
    ax_directFrac[1].set_xlabel('Spike-pair count\nthreshold', ha='center', fontsize=fontsize)
    for i in range(3):
        ax_directFrac[i].set_xticks([0, 100, 200])
        ax_directFrac[i].set_xticklabels(['0', '100', '200'])
        ax_directFrac[i].set_yticks(np.arange(0, 0.41, 0.2))
        ax_directFrac[i].set_yticks(np.arange(0, 0.41, 0.1), minor=True)

        if i != 0:
            ax_directFrac[i].set_yticklabels(['']*3)

    # ax_frac
    ax_frac[0].set_xticks([0, 100, 200])
    ax_frac[0].set_xticklabels([''] * 3)
    ax_frac[0].set_ylabel('All pairs\nfraction', fontsize=fontsize)
    customlegend(ax_frac[0], handlelength=0.5, linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8],
                 loc='center')
    ax_frac[1].set_xticks([0, 100, 200])
    ax_frac[1].set_xticklabels(['0', '100', '200'])
    ax_frac[1].set_xlabel('Spike-pair count\nthreshold', ha='center', fontsize=fontsize)
    ax_frac[1].set_ylabel('Border pairs\nfraction', fontsize=fontsize)
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


    fig_pval.tight_layout()
    fig.savefig(join(save_dir, 'exp_pair_directionality.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'exp_pair_directionality.pdf'), dpi=dpi)
    fig_pval.savefig(join(save_dir, 'exp_pair_TestSignificance.png'), dpi=dpi)


def compare_single_vs_pair_directionality():
    singledata_pth = 'results/emankin/singlefield_df.pickle'
    pairdata_pth = 'results/emankin/pairfield_df.pickle'
    singledf = pd.read_pickle(singledata_pth)
    pairdf = pd.read_pickle(pairdata_pth)
    spike_threshs = np.arange(0, 201, 10)
    stat_fn = 'fig4_SinglePair_R_Comparisons.txt'
    stat_record(stat_fn, True)
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
        stat_record(stat_fn, False, '='*10 + '%s'%(ca) + '='*10)
        for sthresh in spike_threshs:

            single_cadf = singledf[(singledf['ca'] == ca) & (singledf['num_spikes'] > sthresh)]
            pair_cadf = pairdf[(pairdf['ca'] == ca) & (pairdf['num_spikes_pair'] > sthresh)]
            singleR = single_cadf['rate_R'].to_numpy()
            pairR = pair_cadf['rate_Rp'].to_numpy()
            p, _, _, txt = my_kruskal_2samp(singleR, pairR, 'single', 'pair')
            stat_record(stat_fn, False, '%s Spike threshold=%d, Compare Single & Pair R: %s' % (ca, sthresh, txt))


def plot_example_correlograms(expdf, save_dir=None):
    # Data Parameters
    cmap = 'jet'
    alpha_sep=0.6
    alpha_and1=0.5
    alpha_and2=0.25
    plot_axidxs = {467: 0, 971: 1} # Correlogram examples in Fig 5

    # Plotting parameters
    onehalfcol_figw = 4.33  # 1.5 column, =11cm < 11.6cm (required)
    fig_corr, ax_corr = plt.subplots(2, 3, figsize=(onehalfcol_figw, onehalfcol_figw*0.5), facecolor=None)
    fig_field1, ax_field1 = plt.subplots(figsize=(onehalfcol_figw*0.275, onehalfcol_figw*0.275), facecolor=None)  # Plot individually, assemble later
    fig_field2, ax_field2 = plt.subplots(figsize=(onehalfcol_figw*0.275, onehalfcol_figw*0.275), facecolor=None)

    for sid, plotaxidx in plot_axidxs.items():

        # Get data
        overlap, overlap_ratio = expdf.loc[sid, ['overlap', 'overlap_ratio']]
        if np.isnan(overlap_ratio):
            continue
        intrinsicity, extrinsicity, pair_id = expdf.loc[sid, ['overlap_plus', 'overlap_minus', 'pair_id']]
        lagAB, lagBA, corrAB, corrBA = expdf.loc[sid, ['phaselag_AB', 'phaselag_BA', 'corr_info_AB', 'corr_info_BA']]
        xyval1, com1, pf1, mask1 = expdf.loc[sid, ['xyval1', 'com1', 'pf1', 'mask1']]
        xyval2, com2, pf2, mask2 = expdf.loc[sid, ['xyval2', 'com2', 'pf2', 'mask2']]
        X, Y = pf1['X'], pf1['Y']
        mask1 = np.array(mask1, dtype=bool)
        mask2 = np.array(mask2, dtype=bool)
        binsAB, edgesAB, signalAB, alphasAB = corrAB[0], corrAB[1], corrAB[2], corrAB[4]
        binsBA, edgesBA, signalBA, alphasBA = corrBA[0], corrBA[1], corrBA[2], corrBA[4]
        edmAB, edmBA = midedges(edgesAB), midedges(edgesBA)
        binsABnorm, binsBAnorm = binsAB / binsAB.sum(), binsBA / binsBA.sum()

        # Find the time of the lag
        alphasAB = np.mod(alphasAB + 2 * np.pi, 2 * np.pi) - np.pi
        dalphasAB = alphasAB[1:] - alphasAB[0:-1]
        alpha_idxes = np.where(dalphasAB < -np.pi)[0]
        if alpha_idxes.shape[0] > 1:
            absdiffAB = np.abs(alpha_idxes - int(edmAB.shape[0] / 2))
            tmpidx = np.where(absdiffAB == np.min(absdiffAB))[0][0]
            minidxAB = alpha_idxes[tmpidx]
        else:
            minidxAB = alpha_idxes[0]
        peaktimeAB = edmAB[minidxAB]
        maxbinAB = np.max(binsABnorm)

        alphasBA = np.mod(alphasBA + 2 * np.pi, 2 * np.pi) - np.pi
        dalphasBA = alphasBA[1:] - alphasBA[0:-1]
        alpha_idxes = np.where(dalphasBA < -np.pi)[0]
        if alpha_idxes.shape[0] > 1:
            absdiffBA = np.abs(alpha_idxes - int(edmBA.shape[0] / 2))
            tmpidx = np.where(absdiffBA == np.min(absdiffBA))[0][0]
            minidxBA = alpha_idxes[tmpidx]
        else:
            minidxBA = alpha_idxes[0]
        peaktimeBA = edmBA[minidxBA]
        maxbinBA = np.max(binsBAnorm)

        # # Plot correlograms
        ax_corr[plotaxidx, 0].axis('off')
        if plotaxidx == 0:
            plot_overlaid_polormesh(ax_field1, X, Y, pf1['map'], pf2['map'], mask1, mask2, cmap=cmap,
                                    alpha_sep=alpha_sep, alpha_and1=alpha_and1, alpha_and2=alpha_and2)
            ax_field1.annotate(text='Low overlap\n=%0.2f'%(overlap), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=legendsize-2)
            ax_field1.set_xlim(-2, 39)
            ax_field1.set_ylim(-41, 4)
            ax_field1.axis('off')
        else:
            plot_overlaid_polormesh(ax_field2, X, Y, pf1['map'], pf2['map'], mask1, mask2, cmap=cmap,
                                    alpha_sep=alpha_sep, alpha_and1=alpha_and1, alpha_and2=alpha_and2)
            ax_field2.annotate(text='High overlap\n=%0.2f'%(overlap), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=legendsize-2)
            ax_field2.set_xlim(-41-15, 27-15)
            ax_field2.set_ylim(-30, 38)
            ax_field2.axis('off')

        # Plot  AB
        ax_corr[plotaxidx, 1].bar(edmAB, binsABnorm/maxbinAB, width=edmAB[1] - edmAB[0], color='0.7')
        ax_corr[plotaxidx, 1].arrow(x=peaktimeAB, y=1.05, dx=-peaktimeAB, dy=0,
                                    length_includes_head=True, head_width=0.1, head_length=0.005 , width=0.001, zorder=4)
        ax_corr[plotaxidx, 1].axvline(0, color='k', alpha=0.5, linewidth=0.5)
        ax_corr[plotaxidx, 1].annotate(text='$\phi=%0.2f\pi$'%(lagAB/np.pi), xy=(0.25, 1),
                                       xycoords='axes fraction', fontsize=legendsize-2)
        ax_corr[plotaxidx, 1].set_yticks([])
        ax_corr[plotaxidx, 1].set_ylim(0, 1.2)
        ax_corr[plotaxidx, 1].set_xticks([-0.1, 0, 0.1])
        ax_corr[plotaxidx, 1].set_xticks(np.arange(-0.15, 0.15, 0.05), minor=True)
        ax_corr[plotaxidx, 1].set_xticklabels(['-0.1', '0', '0.1'])


        # Plot BA
        ax_corr[plotaxidx, 2].bar(edmBA, binsBAnorm/maxbinBA, width=edmBA[1] - edmBA[0], color='0.7')
        ax_corr[plotaxidx, 2].arrow(x=peaktimeBA, y=1.05, dx=-peaktimeBA, dy=0,
                                    length_includes_head=True, head_width=0.1, head_length=0.005 , width=0.001, zorder=4)
        ax_corr[plotaxidx, 2].axvline(0, color='k', alpha=0.5, linewidth=0.5)
        ax_corr[plotaxidx, 2].annotate(text='$\phi=%0.2f\pi$'%(lagBA/np.pi), xy=(0.25, 1),
                                       xycoords='axes fraction', fontsize=legendsize-2)
        ax_corr[plotaxidx, 2].set_yticks([])
        ax_corr[plotaxidx, 2].set_ylim(0, 1.2)
        ax_corr[plotaxidx, 2].set_xticks([-0.1, 0, 0.1])
        ax_corr[plotaxidx, 2].set_xticks(np.arange(-0.15, 0.15, 0.05), minor=True)
        ax_corr[plotaxidx, 2].set_xticklabels(['-0.1', '0', '0.1'])

    ax_corr[0, 1].set_title(r'$A\rightarrow B$' + '\n', fontsize=legendsize)
    plt.setp(ax_corr[0, 1].get_xticklabels(), visible=False)
    ax_corr[0, 2].set_title(r'$B\rightarrow A$' + '\n', fontsize=legendsize)
    plt.setp(ax_corr[0, 2].get_xticklabels(), visible=False)
    ax_corr[1, 1].set_xlabel('Time lag (s)', fontsize=legendsize)
    ax_corr[1, 2].set_xlabel('Time lag (s)', fontsize=legendsize)

    # Correlogram
    for ax_each in ax_corr.ravel():
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['left'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
        ax_each.tick_params(axis='both', which='major', labelsize=ticksize)

    # fig_corr.text(0.31, 0.3, 'Spike probability', rotation=90, fontsize=legendsize)

    fig_corr.tight_layout()
    fig_corr.subplots_adjust(hspace=0.4)
    fig_corr.subplots_adjust(wspace=0.1)

    fig_corr.savefig(join(save_dir, 'example_pairAB_correlograms.pdf'), dpi=dpi)
    fig_corr.savefig(join(save_dir, 'example_pairAB_correlograms.png'), dpi=dpi)
    fig_field1.savefig(join(save_dir, 'example_pairA_fieldoverlap.pdf'), dpi=dpi)
    fig_field1.savefig(join(save_dir, 'example_pairA_fieldoverlap.png'), dpi=dpi)
    fig_field2.savefig(join(save_dir, 'example_pairB_fieldoverlap.pdf'), dpi=dpi)
    fig_field2.savefig(join(save_dir, 'example_pairB_fieldoverlap.png'), dpi=dpi)


def plot_pair_correlation(expdf, save_dir=None):

    # # Stats
    stat_fn = 'fig5_paircorr.txt'
    stat_record(stat_fn, True)

    # # Plotting parameters
    onehalfcol_figw = 4.33 #  1.5 column, =11cm < 11.6cm (requirement)
    fig, ax = plt.subplots(2, 3, figsize=(onehalfcol_figw, onehalfcol_figw*0.6), sharey='row')
    linew = 0.75
    ms = 1
    for caid, (ca, cadf) in enumerate(expdf.groupby('ca')):
        # A->B
        ax[0, caid], x, y, regress = plot_correlogram(ax=ax[0, caid], df=cadf, tag=ca, direct='A->B', color=ca_c[ca],
                                                      alpha=1,
                                                      markersize=ms, linew=linew)
        ax[0, caid].set_title(ca, fontsize=titlesize)
        nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
        stat_record(stat_fn, False, '%s A->B, y = %0.2fx + %0.2f, $r_{(%d)}=%0.2f, p=%s$' % \
                    (ca, regress['aopt'] * 2 * np.pi, regress['phi0'], nsamples, regress['rho'], p2str(regress['p'])))

        # B->A
        ax[1, caid], x, y, regress = plot_correlogram(ax=ax[1, caid], df=cadf, tag=ca, direct='B->A', color=ca_c[ca],
                                                      alpha=1,
                                                      markersize=ms, linew=linew)

        nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
        stat_record(stat_fn, False, '%s B->A, y = %0.2fx + %0.2f, $r_{(%d)}=%0.2f, p=%s$' % \
                    (ca, regress['aopt'] * 2 * np.pi, regress['phi0'], nsamples, regress['rho'], p2str(regress['p'])))

    for i in range(3):
        for j in range(2):
            ax[j, i].tick_params(labelsize=ticksize)
            ax[j, i].spines["top"].set_visible(False)
            ax[j, i].spines["right"].set_visible(False)

        plt.setp(ax[0, i].get_xticklabels(), visible=False)
        ax[1, i].set_xticks([0, 0.5, 1])
        ax[1, i].set_xticks(np.arange(0, 1, 0.1), minor=True)
        ax[1, i].set_xticklabels(['0', '0.5', '1'])

    ax[1, 1].set_xlabel('Field overlap', fontsize=legendsize)
    fig.text(0.01, 0.30, 'Correlation lag (rad)', rotation=90, fontsize=legendsize)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(join(save_dir, 'PairCorr_vs_FieldOverlap.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'PairCorr_vs_FieldOverlap.eps'), dpi=dpi)


def plot_ALL_examples_EXIN_correlogram(expdf):
    plot_dir = join('result_plots', 'examplesALL', 'exin_correlograms_passes')
    os.makedirs(plot_dir, exist_ok=True)

    for sid in range(expdf.shape[0]):
        print('Correlogram examples %d/%d' % (sid, expdf.shape[0]))
        # Get data
        ca, pair_id, overlap, overlap_ratio = expdf.loc[sid, ['ca', 'pair_id', 'overlap', 'overlap_ratio']]
        intrinsicity, extrinsicity, pair_id = expdf.loc[sid, ['overlap_plus', 'overlap_minus', 'pair_id']]
        lagAB, lagBA, corrAB, corrBA = expdf.loc[sid, ['phaselag_AB', 'phaselag_BA', 'corr_info_AB', 'corr_info_BA']]
        com1, com2 = expdf.loc[sid, ['com1', 'com2']]
        if np.isnan(overlap_ratio):
            continue

        binsAB, edgesAB, signalAB, alphasAB = corrAB[0], corrAB[1], corrAB[2], corrAB[4]
        binsBA, edgesBA, signalBA, alphasBA = corrBA[0], corrBA[1], corrBA[2], corrBA[4]
        edmAB, edmBA = midedges(edgesAB), midedges(edgesBA)
        binsABnorm, binsBAnorm = binsAB / binsAB.sum(), binsBA / binsBA.sum()

        # Find the time of the lag
        alphasAB = np.mod(alphasAB + 2 * np.pi, 2 * np.pi) - np.pi
        dalphasAB = alphasAB[1:] - alphasAB[0:-1]
        alpha_idxes = np.where(dalphasAB < -np.pi)[0]
        if alpha_idxes.shape[0] > 1:
            absdiffAB = np.abs(alpha_idxes - int(edmAB.shape[0] / 2))
            tmpidx = np.where(absdiffAB == np.min(absdiffAB))[0][0]
            minidxAB = alpha_idxes[tmpidx]
        else:
            minidxAB = alpha_idxes[0]

        alphasBA = np.mod(alphasBA + 2 * np.pi, 2 * np.pi) - np.pi
        dalphasBA = alphasBA[1:] - alphasBA[0:-1]
        alpha_idxes = np.where(dalphasBA < -np.pi)[0]
        if alpha_idxes.shape[0] > 1:
            absdiffBA = np.abs(alpha_idxes - int(edmBA.shape[0] / 2))
            tmpidx = np.where(absdiffBA == np.min(absdiffBA))[0][0]
            minidxBA = alpha_idxes[tmpidx]
        else:
            minidxBA = alpha_idxes[0]

        # Get all precessions
        precess_dfp, xyval1, xyval2 = expdf.loc[sid, ['precess_dfp', 'xyval1', 'xyval2']]

        precess_dfpAB = precess_dfp[precess_dfp['direction'] == 'A->B'].reset_index(drop=True)
        precess_dfpBA = precess_dfp[precess_dfp['direction'] == 'B->A'].reset_index(drop=True)

        pooldict = {}

        def getpooledprecess(precess_dfp):
            alldsp1, allphasesp1, alldsp2, allphasesp2 = [], [], [], []
            for nprecess in range(precess_dfp.shape[0]):
                dsp1, phasesp1, tsptheta1 = precess_dfp.loc[nprecess, ['dsp1', 'phasesp1', 'tsp_withtheta1']]
                dsp2, phasesp2, tsptheta2 = precess_dfp.loc[nprecess, ['dsp2', 'phasesp2', 'tsp_withtheta2']]
                alldsp1.append(dsp1)
                alldsp2.append(dsp2)
                allphasesp1.append(phasesp1)
                allphasesp2.append(phasesp2)
            alldsp1, alldsp2 = np.concatenate(alldsp1), np.concatenate(alldsp2)
            allphasesp1, allphasesp2 = np.concatenate(allphasesp1), np.concatenate(allphasesp2)
            regresspool1 = rcc(alldsp1, allphasesp1)
            regresspool2 = rcc(alldsp2, allphasesp2)
            rccpool_m1, rccpool_c1 = regresspool1['aopt'], regresspool1['phi0']
            rccpool_m2, rccpool_c2 = regresspool2['aopt'], regresspool2['phi0']

            maxdpool = max(1, alldsp1.max(), alldsp2.max())
            xdumpool = np.linspace(0, maxdpool, 10)
            ydumpool1 = xdumpool * 2 * np.pi * rccpool_m1 + rccpool_c1
            ydumpool2 = xdumpool * 2 * np.pi * rccpool_m2 + rccpool_c2
            return alldsp1, alldsp2, allphasesp1, allphasesp2, xdumpool, ydumpool1, ydumpool2, maxdpool

        try:
            pooldict['A->B'] = getpooledprecess(precess_dfpAB)
            pooldict['B->A'] = getpooledprecess(precess_dfpBA)
        except ValueError:
            continue

        # Plot pass examples in the ex-in pair

        for nprecess in range(precess_dfp.shape[0]):
            x, y, t, v = precess_dfp.loc[nprecess, ['x', 'y', 't', 'v']]
            straightrank, chunked, direction = precess_dfp.loc[nprecess, ['straightrank', 'chunked', 'direction']]
            precess_exist = precess_dfp.loc[nprecess, 'precess_exist']
            xsp1, ysp1, xsp2, ysp2 = precess_dfp.loc[nprecess, ['spike1x', 'spike1y', 'spike2x', 'spike2y']]
            dsp1, phasesp1, tsptheta1 = precess_dfp.loc[nprecess, ['dsp1', 'phasesp1', 'tsp_withtheta1']]
            dsp2, phasesp2, tsptheta2 = precess_dfp.loc[nprecess, ['dsp2', 'phasesp2', 'tsp_withtheta2']]
            wavet1, phase1, theta1 = precess_dfp.loc[nprecess, ['wave_t1', 'wave_phase1', 'wave_theta1']]
            rcc_m1, rcc_c1, rcc_m2, rcc_c2 = precess_dfp.loc[nprecess, ['rcc_m1', 'rcc_c1', 'rcc_m2', 'rcc_c2']]
            pass_nspikes1, pass_nspikes2 = precess_dfp.loc[nprecess, ['pass_nspikes1', 'pass_nspikes2']]

            fig, ax = plt.subplots(2, 3, figsize=(10, 6))

            # Trajectory
            ax[0, 0].plot(xyval1[:, 0], xyval1[:, 1], c='r', label='Field A')
            ax[0, 0].plot(xyval2[:, 0], xyval2[:, 1], c='b', label='Field B')
            ax[0, 0].scatter(com1[0], com1[1], c='r', marker='^', s=16)
            ax[0, 0].scatter(com2[0], com2[1], c='b', marker='^', s=16)
            ax[0, 0].plot(x, y, c='gray')
            ax[0, 0].scatter(x[0], y[0], c='gray', marker='o')
            ax[0, 0].scatter(xsp1, ysp1, c='r', marker='x')
            ax[0, 0].scatter(xsp2, ysp2, c='b', marker='x')
            ax[0, 0].set_title('Straightrank=%0.2f, Chunk=%d, %s' % (straightrank, chunked, direction))

            # Precession
            ax[0, 1].scatter(dsp1, phasesp1, c='r')
            ax[0, 1].scatter(dsp2, phasesp2, c='b')
            maxd = max(1, dsp1.max(), dsp2.max())
            xdum = np.linspace(0, maxd, 10)
            ydum1 = xdum * 2 * np.pi * rcc_m1 + rcc_c1
            ydum2 = xdum * 2 * np.pi * rcc_m2 + rcc_c2
            ax[0, 1].plot(xdum, ydum1, c='r')
            ax[0, 1].plot(xdum, ydum1 + 2 * np.pi, c='r')
            ax[0, 1].plot(xdum, ydum1 - 2 * np.pi, c='r')
            ax[0, 1].plot(xdum, ydum2, c='b')
            ax[0, 1].plot(xdum, ydum2 + 2 * np.pi, c='b')
            ax[0, 1].plot(xdum, ydum2 - 2 * np.pi, c='b')
            ax[0, 1].set_xlim(0, maxd)
            ax[0, 1].set_ylim(-np.pi, 2 * np.pi)
            ax[0, 1].set_yticks([-np.pi, 0, np.pi, 2 * np.pi])
            ax[0, 1].set_yticklabels(['$-\pi$', '0', '$\pi$', '$2\pi$'])
            ax[0, 1].set_title('Spcount, A=%d, B=%d' % (pass_nspikes1, pass_nspikes2))

            # Theta cycle and phase
            ax[0, 2].plot(wavet1, theta1)
            theta_range = theta1.max() - theta1.min()
            axphase = ax[0, 2].twinx()
            axphase.plot(wavet1, phase1, c='orange', alpha=0.5)
            # axphase.set_ylim(phase1.min()*2, phase1.max()*2)
            ax[0, 2].eventplot(tsptheta1, lineoffsets=theta1.max() * 1.1, linelengths=theta_range * 0.05, colors='r')
            ax[0, 2].eventplot(tsptheta2, lineoffsets=theta1.max() * 1.1 - theta_range * 0.05,
                               linelengths=theta_range * 0.05, colors='b')

            # Correlograms
            ax[1, 1].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A->B')
            ax[1, 1].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='slategray', label='B->A')
            ax[1, 1].annotate('In.=%0.2f' % (intrinsicity), xy=(0.025, 0.7), xycoords='axes fraction', size=titlesize,
                              color='b')

            ax[1, 0].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A->B')
            ax[1, 0].bar(edmBA, np.flip(binsBAnorm), width=edmBA[1] - edmBA[0], color='slategray',
                         label='Flipped B->A')
            ax[1, 0].annotate('Ex.=%0.2f' % (extrinsicity), xy=(0.58, 0.7), xycoords='axes fraction', size=titlesize,
                              color='r')
            for ax_each in [ax[1, 0], ax[1, 1]]:
                ax_each.set_xticks([-0.15, 0, 0.15])
                ax_each.set_xlabel('Time lag (s)', fontsize=titlesize)
                ax_each.set_yticks([])
                ax_each.tick_params(axis='both', which='major', labelsize=ticksize)
                customlegend(ax_each, fontsize=fontsize)
                ax_each.spines['top'].set_visible(False)
                ax_each.spines['left'].set_visible(False)
                ax_each.spines['right'].set_visible(False)
                ax_each.set_ylabel('')

            # All precession
            alldsp1, alldsp2, allphasesp1, allphasesp2, xdumpool, ydumpool1, ydumpool2, maxdpool = pooldict[direction]
            ax[1, 2].scatter(alldsp1, allphasesp1, c='r', alpha=0.3)
            ax[1, 2].scatter(alldsp2, allphasesp2, c='b', alpha=0.3)
            ax[1, 2].plot(xdumpool, ydumpool1, c='r')
            ax[1, 2].plot(xdumpool, ydumpool1 + 2 * np.pi, c='r')
            ax[1, 2].plot(xdumpool, ydumpool1 - 2 * np.pi, c='r')
            ax[1, 2].plot(xdumpool, ydumpool2, c='b')
            ax[1, 2].plot(xdumpool, ydumpool2 + 2 * np.pi, c='b')
            ax[1, 2].plot(xdumpool, ydumpool2 - 2 * np.pi, c='b')
            ax[1, 2].set_xlim(0, maxdpool)
            ax[1, 2].set_ylim(-np.pi, 2 * np.pi)
            ax[1, 2].set_yticks([-np.pi, 0, np.pi, 2 * np.pi])
            ax[1, 2].set_yticklabels(['$-\pi$', '0', '$\pi$', '$2\pi$'])
            ax[1, 2].set_title(direction)

            fig.tight_layout()
            exintag = 'ex' if overlap_ratio > 0 else 'in'
            precesstag = 'precess' if precess_exist else 'nonprecess'
            figname = '%s_%d-%d_%s_%s_%s.png' % (ca, pair_id, nprecess, exintag, precesstag, direction)
            fig.savefig(join(plot_dir, figname), dpi=200, facecolor='white')
            plt.close(fig)




def plot_ALL_example_correlograms(expdf):
    # Plot all correlograms with peak lags and different degree of overlap
    # For finding a good example in Fig 5
    save_dir = 'result_plots/examplesALL/overlap_correlograms'
    os.makedirs(save_dir, exist_ok=True)
    figl = total_figw * 0.5 / 3  # Consistent with pair correlation
    num_pairs = expdf.shape[0]
    for sid in range(num_pairs):
        print('\rPlotting %d/%d'%(sid, num_pairs), flush=True, end='')

        # Get data
        overlap, overlap_ratio = expdf.loc[sid, ['overlap', 'overlap_ratio']]
        if np.isnan(overlap_ratio):
            continue
        intrinsicity, extrinsicity, pair_id = expdf.loc[sid, ['overlap_plus', 'overlap_minus', 'pair_id']]
        lagAB, lagBA, corrAB, corrBA = expdf.loc[sid, ['phaselag_AB', 'phaselag_BA', 'corr_info_AB', 'corr_info_BA']]

        binsAB, edgesAB, signalAB, alphasAB = corrAB[0], corrAB[1], corrAB[2], corrAB[4]
        binsBA, edgesBA, signalBA, alphasBA = corrBA[0], corrBA[1], corrBA[2], corrBA[4]
        edmAB, edmBA = midedges(edgesAB), midedges(edgesBA)
        binsABnorm, binsBAnorm = binsAB / binsAB.sum(), binsBA / binsBA.sum()

        # Find the time of the lag
        alphasAB = np.mod(alphasAB + 2 * np.pi, 2 * np.pi) - np.pi
        dalphasAB = alphasAB[1:] - alphasAB[0:-1]
        alpha_idxes = np.where(dalphasAB < -np.pi)[0]
        if alpha_idxes.shape[0] > 1:
            absdiffAB = np.abs(alpha_idxes - int(edmAB.shape[0] / 2))
            tmpidx = np.where(absdiffAB == np.min(absdiffAB))[0][0]
            minidxAB = alpha_idxes[tmpidx]
        else:
            minidxAB = alpha_idxes[0]
        peaktimeAB = edmAB[minidxAB]
        maxbinAB = np.max(binsABnorm)

        alphasBA = np.mod(alphasBA + 2 * np.pi, 2 * np.pi) - np.pi
        dalphasBA = alphasBA[1:] - alphasBA[0:-1]
        alpha_idxes = np.where(dalphasBA < -np.pi)[0]
        if alpha_idxes.shape[0] > 1:
            absdiffBA = np.abs(alpha_idxes - int(edmBA.shape[0] / 2))
            tmpidx = np.where(absdiffBA == np.min(absdiffBA))[0][0]
            minidxBA = alpha_idxes[tmpidx]
        else:
            minidxBA = alpha_idxes[0]
        peaktimeBA = edmBA[minidxBA]
        maxbinBA = np.max(binsBAnorm)

        # # Plot correlograms
        fig_corr, ax_corr = plt.subplots(2, 1, figsize=(figl * 1.5, figl * 2.35), sharex=True, facecolor='w')

        # Plot  AB
        ax_corr[0].bar(edmAB, binsABnorm/maxbinAB, width=edmAB[1] - edmAB[0], color='0.7')
        ax_corr[0].arrow(x=peaktimeAB, y=1.05, dx=-peaktimeAB, dy=0,
                         length_includes_head=True, head_width=0.1, head_length=0.005 , width=0.001)
        ax_corr[0].annotate(text='$\phi=%0.2f\pi$'%(lagAB/np.pi), xy=(0.25, 1),
                            xycoords='axes fraction', fontsize=legendsize-2)

        ax_corr[0].set_yticks([])
        ax_corr[0].set_title('Field overlap=%0.2f\n' % (overlap) + r'$A\rightarrow B$', fontsize=legendsize)
        ax_corr[0].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_corr[0].set_ylim(0, 1.2)

        # Plot BA
        ax_corr[1].bar(edmBA, binsBAnorm/maxbinBA, width=edmBA[1] - edmBA[0], color='0.7')
        ax_corr[1].arrow(x=peaktimeBA, y=1.05, dx=-peaktimeBA, dy=0,
                         length_includes_head=True, head_width=0.1, head_length=0.005 , width=0.001)
        ax_corr[1].annotate(text='$\phi=%0.2f\pi$'%(lagBA/np.pi), xy=(0.25, 1),
                            xycoords='axes fraction', fontsize=legendsize-2)
        ax_corr[1].set_yticks([])
        ax_corr[1].set_title(r'$B\rightarrow A$', fontsize=legendsize)
        ax_corr[1].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_corr[1].set_ylim(0, 1.2)

        # Shared parameters
        ax_corr[1].set_xlabel('Time lag (s)', fontsize=fontsize)
        ax_corr[1].set_xticks([-0.15, 0, 0.15])

        # Correlogram
        for ax_each in ax_corr.ravel():
            ax_each.spines['top'].set_visible(False)
            ax_each.spines['left'].set_visible(False)
            ax_each.spines['right'].set_visible(False)
        fig_corr.text(0.01, 0.25, 'Spike probability', rotation=90, fontsize=fontsize)

        fig_corr.tight_layout()
        fig_corr.subplots_adjust(hspace=0.6)
        folder_name = 'low_overlap' if overlap < 0.5 else 'high_overlap'
        fig_corr.savefig(join(save_dir, folder_name, 'examples_correlogram_%d.png'%(pair_id)), dpi=dpi)
        plt.close(fig_corr)


def plot_example_exin(expdf, save_dir):

    def plot_exin_pass_eg(ax, expdf, pairid, passid, color_A, color_B, ms, lw, direction):

        xyval1, xyval2, precessdf = expdf.loc[pairid, ['xyval1', 'xyval2', 'precess_dfp']]
        fieldcoor1, fieldcoor2 = expdf.loc[pairid, ['com1', 'com2']]

        x, y, wave_t, wave_theta, wave_phase = precessdf.loc[
            passid, ['x', 'y', 'wave_t1', 'wave_theta1', 'wave_phase1']]
        xsp1, ysp1, xsp2, ysp2 = precessdf.loc[passid, ['spike1x', 'spike1y', 'spike2x', 'spike2y']]

        tsp1, dsp1, phasesp1, rcc_m1, rcc_c1 = precessdf.loc[
            passid, ['tsp_withtheta1', 'dsp1', 'phasesp1', 'rcc_m1', 'rcc_c1']]
        tsp2, dsp2, phasesp2, rcc_m2, rcc_c2 = precessdf.loc[
            passid, ['tsp_withtheta2', 'dsp2', 'phasesp2', 'rcc_m2', 'rcc_c2']]

        min_wavet = wave_t.min()
        wave_t = wave_t - min_wavet
        tsp1 = tsp1 - min_wavet
        tsp2 = tsp2 - min_wavet

        # Trajectory
        allx, ally = np.concatenate([xyval1[:, 0], xyval2[:, 0]]), np.concatenate([xyval1[:, 1], xyval2[:, 1]])
        max_xyl = max(allx.max() - allx.min(), ally.max() - ally.min())
        ax_minx, ax_miny = allx.min(), ally.min()
        ax[0].plot(xyval1[:, 0], xyval1[:, 1], c=color_A, linewidth=lw, label='A')
        ax[0].scatter(fieldcoor1[0], fieldcoor1[1], c=color_A, marker='^', s=16)
        ax[0].plot(xyval2[:, 0], xyval2[:, 1], c=color_B, linewidth=lw, label='B')
        ax[0].scatter(fieldcoor2[0], fieldcoor2[1], c=color_B, marker='^', s=16)
        ax[0].plot(x, y, c='gray', linewidth=1)
        ax[0].scatter(xsp1, ysp1, c=color_A, s=ms, marker='.', zorder=2.5)
        ax[0].scatter(xsp2, ysp2, c=color_B, s=ms, marker='.', zorder=2.5)
        lim_mar = 4
        ax[0].set_xlim(ax_minx - lim_mar, ax_minx + max_xyl + lim_mar)
        ax[0].set_ylim(ax_miny - lim_mar, min(ax_miny + max_xyl, ally.max()) + lim_mar)
        ax[0].axis('off')

        # Theta power
        max_l = wave_theta.max()
        ax[1].plot(wave_t, wave_theta, c='gray', linewidth=lw)
        ax[1].eventplot(tsp1, lineoffsets=max_l + 1 * max_l, linelengths=max_l * 0.75, linewidths=lw, colors=color_A)
        ax[1].eventplot(tsp2, lineoffsets=max_l + 0.25 * max_l, linelengths=max_l * 0.75, linewidths=lw, colors=color_B)
        boundidx_all = np.where(np.diff(wave_phase) < -np.pi)[0]
        #     for i, boundidx in enumerate(boundidx_all):
        #         ax[1].axvline(wave_t[boundidx], linewidth=0.25, color='k')
        for i in range(boundidx_all.shape[0] - 1):
            if ((i % 2) != 0):
                continue
            ax[1].axvspan(wave_t[boundidx_all[i]], wave_t[boundidx_all[i + 1]], color='0.9', zorder=0.1)

        ax[1].set_ylim(wave_theta.min(), max_l + 1.5 * max_l)
        ax[1].set_xticks([])
        ax[1].tick_params(labelsize=ticksize)
        ax[1].set_yticks([])
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)

        # Precession
        precessmask = (precessdf['direction'] == direction)
        dsp1 = np.concatenate(precessdf.loc[precessmask, 'dsp1'].to_list())
        dsp2 = np.concatenate(precessdf.loc[precessmask, 'dsp2'].to_list())
        phasesp1 = np.concatenate(precessdf.loc[precessmask, 'phasesp1'].to_list())
        phasesp2 = np.concatenate(precessdf.loc[precessmask, 'phasesp2'].to_list())
        abound = (-2, 0)
        regress1, regress2 = rcc(dsp1, phasesp1, abound=abound), rcc(dsp2, phasesp2, abound=abound)
        rcc_m1, rcc_c1 = regress1['aopt'], regress1['phi0']
        rcc_m2, rcc_c2 = regress2['aopt'], regress2['phi0']

        maxdsp = max(1, max(dsp1.max(), dsp2.max()))
        xdum = np.linspace(0, maxdsp, 10)
        ydum1 = rcc_m1 * 2 * np.pi * xdum + rcc_c1
        ydum2 = rcc_m2 * 2 * np.pi * xdum + rcc_c2
        ax[2].scatter(dsp1, phasesp1, c=color_A, s=ms, marker='.')
        ax[2].scatter(dsp1, phasesp1 + 2 * np.pi, c=color_A, s=ms, marker='.')
        ax[2].plot(xdum, ydum1, c=color_A, linewidth=lw)
        ax[2].plot(xdum, ydum1 + 2 * np.pi, c=color_A, linewidth=lw)
        ax[2].scatter(dsp2, phasesp2, c=color_B, s=ms, marker='.')
        ax[2].scatter(dsp2, phasesp2 + 2 * np.pi, c=color_B, s=ms, marker='.')
        ax[2].plot(xdum, ydum2, c=color_B, linewidth=lw)
        ax[2].plot(xdum, ydum2 + 2 * np.pi, c=color_B, linewidth=lw)
        ax[2].set_ylim(-np.pi, 3 * np.pi)
        ax[2].set_xticks([0, 1])
        ax[2].set_xlabel('Position', fontsize=legendsize, labelpad=-10)
        ax[2].tick_params(labelsize=ticksize)

    def plot_exin_correlogram_eg(ax, expdf, pairid):
        overlap, overlap_ratio = expdf.loc[pairid, ['overlap', 'overlap_ratio']]
        intrinsicity, extrinsicity, pair_id = expdf.loc[pairid, ['overlap_plus', 'overlap_minus', 'pair_id']]
        lagAB, lagBA, corrAB, corrBA = expdf.loc[pairid, ['phaselag_AB', 'phaselag_BA', 'corr_info_AB', 'corr_info_BA']]
        binsAB, edgesAB, signalAB, alphasAB = corrAB[0], corrAB[1], corrAB[2], corrAB[4]
        binsBA, edgesBA, signalBA, alphasBA = corrBA[0], corrBA[1], corrBA[2], corrBA[4]

        edmAB, edmBA = midedges(edgesAB), midedges(edgesBA)
        binsABnorm, binsBAnorm = binsAB / binsAB.sum(), binsBA / binsBA.sum()
        ax[0].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label=r'$A\rightarrow B$', alpha=0.75)
        ax[0].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='slategray', label=r'$B\rightarrow A$', alpha=0.75)
        ax[0].annotate('In.=%0.2f' % (intrinsicity), xy=(0.025, 1.05), xycoords='axes fraction', size=legendsize,
                       color='b')
        customlegend(ax[0], fontsize=legendsize, bbox_to_anchor=(0.5, 0.5), loc='lower left')

        ax[1].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label=r'$A\rightarrow B$', alpha=0.75)
        ax[1].bar(edmBA, np.flip(binsBAnorm), width=edmBA[1] - edmBA[0], color='slategray', label=r'Flipped $B\rightarrow A$', alpha=0.75)
        ax[1].annotate('Ex.=%0.2f' % (extrinsicity), xy=(0.58, 1.05), xycoords='axes fraction', size=legendsize,
                       color='r')
        customlegend(ax[1], fontsize=legendsize, bbox_to_anchor=(0.01, 0.5), loc='lower left')

        for ax_each in ax.ravel():
            ax_each.set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
            ax_each.set_xticklabels(['', '-0.1', '', '0', '', '0.1', ''])
            ax_each.set_xlabel('Time lag (s)', fontsize=legendsize)
            ax_each.set_yticks([])
            ax_each.spines['left'].set_visible(False)
            ax_each.tick_params(axis='both', which='major', labelsize=ticksize)


    fig_exin = plt.figure(figsize=(total_figw, total_figw / 2))

    axw = 0.9 / 4
    axh1 = 0.9 / 3.5
    axh2 = 0.9 / 2 / 3.5

    btwgp_xgap = -0.05
    biggp_xoffset = -0.025
    biggp_xgap = -0.025

    axtraj_y = 1 - axh1 - 0.1
    axtraj_xoffset, axtraj_yoffset = 0.125, 0.04

    axtraj_wm, axtraj_hm = 0.9, 1
    axes_traj = [fig_exin.add_axes(
        [0 + axtraj_xoffset - biggp_xgap, axtraj_y + axtraj_yoffset, axw * axtraj_wm, axh1 * axtraj_hm]),
        fig_exin.add_axes(
            [0.25 + axtraj_xoffset + btwgp_xgap - biggp_xgap, axtraj_y + axtraj_yoffset, axw * axtraj_wm,
             axh1 * axtraj_hm]),
        fig_exin.add_axes(
            [0.5 + axtraj_xoffset + biggp_xoffset + biggp_xgap, axtraj_y + axtraj_yoffset, axw * axtraj_wm,
             axh1 * axtraj_hm]),
        fig_exin.add_axes(
            [0.75 + axtraj_xoffset + btwgp_xgap + biggp_xoffset + biggp_xgap, axtraj_y + axtraj_yoffset,
             axw * axtraj_wm, axh1 * axtraj_hm])]

    axtheta_y = axtraj_y - axh2
    axtheta_xoffset, axtheta_yoffset = 0.085, 0.05
    axtheta_wm, axtheta_hm, axtheta_hincrease = 0.9, 0.65, 0.045
    axes_theta = [fig_exin.add_axes(
        [0 + axtheta_xoffset - biggp_xgap, axtheta_y + axtheta_yoffset, axw * axtheta_wm, axh2 * axtheta_hm+axtheta_hincrease]),
        fig_exin.add_axes(
            [0.25 + axtheta_xoffset + btwgp_xgap - biggp_xgap, axtheta_y + axtheta_yoffset, axw * axtheta_wm,
             axh2 * axtheta_hm+axtheta_hincrease]),
        fig_exin.add_axes([0.5 + axtheta_xoffset + biggp_xoffset + biggp_xgap, axtheta_y + axtheta_yoffset,
                           axw * axtheta_wm, axh2 * axtheta_hm+axtheta_hincrease]),
        fig_exin.add_axes(
            [0.75 + axtheta_xoffset + btwgp_xgap + biggp_xoffset + biggp_xgap, axtheta_y + axtheta_yoffset,
             axw * axtheta_wm, axh2 * axtheta_hm+axtheta_hincrease])]

    axprecess_y = axtheta_y - axh1
    axprecess_xoffset, axprecess_yoffset = 0.11, 0.15
    axprecess_wm, axprecess_hm = 0.65, 0.6
    axes_precess = [fig_exin.add_axes(
        [0 + axprecess_xoffset - biggp_xgap, axprecess_y + axprecess_yoffset, axw * axprecess_wm, axh1 * axprecess_hm]),
        fig_exin.add_axes(
            [0.25 + axprecess_xoffset + btwgp_xgap - biggp_xgap, axprecess_y + axprecess_yoffset,
             axw * axprecess_wm, axh1 * axprecess_hm]),
        fig_exin.add_axes(
            [0.5 + axprecess_xoffset + biggp_xoffset + biggp_xgap, axprecess_y + axprecess_yoffset,
             axw * axprecess_wm, axh1 * axprecess_hm]),
        fig_exin.add_axes([0.75 + axprecess_xoffset + btwgp_xgap + biggp_xoffset + biggp_xgap,
                           axprecess_y + axprecess_yoffset, axw * axprecess_wm, axh1 * axprecess_hm])]

    axcorr_y = axprecess_y - axh1
    axcorr_xoffset, axcorr_yoffset = 0.10, 0.145
    axcorr_wm, axcorr_hm = 0.75, 0.5
    axes_corr = [fig_exin.add_axes(
        [0 + axcorr_xoffset - biggp_xgap, axcorr_y + axcorr_yoffset, axw * axcorr_wm, axh1 * axcorr_hm]),
        fig_exin.add_axes(
            [0.25 + axcorr_xoffset + btwgp_xgap - biggp_xgap, axcorr_y + axcorr_yoffset, axw * axcorr_wm,
             axh1 * axcorr_hm]),
        fig_exin.add_axes(
            [0.5 + axcorr_xoffset + biggp_xoffset + biggp_xgap, axcorr_y + axcorr_yoffset, axw * axcorr_wm,
             axh1 * axcorr_hm]),
        fig_exin.add_axes(
            [0.75 + axcorr_xoffset + btwgp_xgap + biggp_xoffset + biggp_xgap, axcorr_y + axcorr_yoffset,
             axw * axcorr_wm, axh1 * axcorr_hm])]

    ax_exin = np.array([axes_traj, axes_theta, axes_precess, axes_corr])

    color_A, color_B = 'turquoise', 'purple'
    ms = 0.5
    lw = 0.5

    in_pairid = 162
    in_passABid = 6
    in_passBAid = 1

    ex_pairid = 26
    ex_passABid = 2
    ex_passBAid = 4

    # Pass example 1: extrinsic A->B
    pairid, passid = ex_pairid, ex_passABid
    plot_exin_pass_eg(ax_exin[0:3, 0], expdf, pairid, passid, color_A, color_B, ms, lw, 'A->B')

    # Pass example 2: extrinsic B->A
    pairid, passid = ex_pairid, ex_passBAid
    plot_exin_pass_eg(ax_exin[0:3, 1], expdf, pairid, passid, color_A, color_B, ms, lw, 'B->A')

    # Pass example 3: intrinsic A->B
    pairid, passid = in_pairid, in_passABid
    plot_exin_pass_eg(ax_exin[0:3, 2], expdf, pairid, passid, color_A, color_B, ms, lw, 'A->B')

    # Pass example 4: intrinsic B->A
    pairid, passid = in_pairid, in_passBAid
    plot_exin_pass_eg(ax_exin[0:3, 3], expdf, pairid, passid, color_A, color_B, ms, lw, 'B->A')

    # Extrinsic correlogram
    pairid = ex_pairid
    plot_exin_correlogram_eg(ax_exin[3, 0:2], expdf, pairid)

    # Intrinsic correlogram
    pairid = in_pairid
    plot_exin_correlogram_eg(ax_exin[3, 2:4], expdf, pairid)

    # Hide top and right axis line
    for ax_each in ax_exin.ravel():
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    customlegend(ax_exin[0, 0], fontsize=legendsize, bbox_to_anchor=(0.8, 0), loc='lower left')
    fig_exin.text(0.23, 0.96, 'Pair#%d (Extrinsic)' % (ex_pairid), fontsize=legendsize)
    fig_exin.text(0.655, 0.96, 'Pair#%d (Intrinsic)' % (in_pairid), fontsize=legendsize)

    for i in range(1, 4):
        ax_exin[2, i].set_yticks([0, np.pi, 2 * np.pi])
        ax_exin[2, i].set_yticklabels([''] * 3)
    ax_exin[2, 0].set_yticks([0, np.pi, 2 * np.pi])
    ax_exin[2, 0].set_yticklabels(['0', '', '$2\pi$'])


    fig_exin.text(0.189, 0.925, r'$A\rightarrow B$', fontsize=legendsize)
    fig_exin.text(0.39, 0.925, r'$B\rightarrow A$', fontsize=legendsize)
    fig_exin.text(0.615, 0.925, r'$A\rightarrow B$', fontsize=legendsize)
    fig_exin.text(0.815, 0.925, r'$B\rightarrow A$', fontsize=legendsize)
    fig_exin.text(0.035, 0.44, 'Phase\n(rad)', rotation=90, fontsize=legendsize)
    fig_exin.text(0.035, 0.05, 'Normalized\nspike count', rotation=90, fontsize=legendsize)

    fig_exin.savefig(join(save_dir, 'examples_exintrinsicity.png'), dpi=dpi)
    fig_exin.savefig(join(save_dir, 'examples_exintrinsicity.pdf'), dpi=dpi)


def plot_exintrinsic(expdf, save_dir):
    figl = total_figw / 2 / 3

    stat_fn = 'fig6_exintrinsic.txt'
    stat_record(stat_fn, True)

    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'
    # Filtering
    smallerdf = expdf[(~expdf[ratio_key].isna())]
    # smallerdf['Extrincicity'] = smallerdf[ratio_key].apply(lambda x: 'Extrinsic' if x > 0 else 'Intrinsic')

    fig_exin, ax_exin = plt.subplots(2, 3, figsize=(figl * 3, figl * 2.75), sharey='row', sharex='row')
    ms = 1
    contindf_dict = {}

    for caid, (ca, cadf) in enumerate(smallerdf.groupby('ca')):

        corr_overlap = cadf[oplus_key].to_numpy()
        corr_overlap_flip = cadf[ominus_key].to_numpy()
        corr_overlap_ratio = cadf[ratio_key].to_numpy()

        # 1-sample chisquare test
        n_ex = np.sum(corr_overlap_ratio > 0)
        n_in = np.sum(corr_overlap_ratio <= 0)
        n_total = n_ex + n_in
        contindf_dict[ca] = [n_ex, n_in]
        pchi, _, chitxt = my_chi_1way([n_ex, n_in])
        stat_record(stat_fn, False, 'Oneway Fraction, %s, %d:%d=%0.2f, %s' % (ca, n_ex, n_in, n_ex / n_in, chitxt))

        # 1-sample t test
        mean_ratio = np.mean(corr_overlap_ratio)
        p_1d1samp, _, ttest_txt = my_ttest_1samp(corr_overlap_ratio, 0)
        stat_record(stat_fn, False, 'Ex-In, %s, %s' % (ca, ttest_txt))

        # Plot scatter 2d
        ax_exin[0, caid].scatter(corr_overlap_flip, corr_overlap, s=ms, c=ca_c[ca], marker='.')
        ax_exin[0, caid].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
        ax_exin[0, caid].annotate('%0.2f' % (n_ex / n_in), xy=(0.015, 0.17), xycoords='axes fraction', size=legendsize,
                                  color='r')
        ax_exin[0, caid].annotate('p=%s' % (p2str(pchi)), xy=(0.015, 0.025), xycoords='axes fraction', size=legendsize)
        ax_exin[0, 1].set_xlabel('Extrinsicity', fontsize=fontsize)
        ax_exin[0, caid].set_yticks([0, 1])
        ax_exin[0, caid].set_xlim(0, 1)
        ax_exin[0, caid].set_ylim(0, 1)
        ax_exin[0, caid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_exin[0, caid].set_title(ca, fontsize=titlesize)

        # Plot 1d histogram
        edges = np.linspace(-1, 1, 75)
        width = edges[1] - edges[0]
        (bins, _, _) = ax_exin[1, caid].hist(corr_overlap_ratio, bins=edges, color=ca_c[ca], density=True,
                                             histtype='stepfilled')
        ax_exin[1, caid].plot([mean_ratio, mean_ratio], [0, bins.max()], c='k', linewidth=0.75)
        ax_exin[1, caid].annotate('$\mu=$' + '%0.3f' % (mean_ratio), xy=(0.05, 0.85), xycoords='axes fraction',
                                  size=legendsize)
        ax_exin[1, caid].annotate('p=%s' % (p2str(p_1d1samp)), xy=(0.05, 0.7), xycoords='axes fraction',
                                  size=legendsize)
        ax_exin[1, caid].set_xticks([])
        ax_exin[1, caid].set_yticks([0, 0.1 / width])
        ax_exin[1, caid].set_yticklabels(['0', '0.1'])
        ax_exin[1, caid].tick_params(labelsize=ticksize)
        ax_exin[1, caid].set_xlim(-0.5, 0.5)
        ax_exin[1, caid].set_ylim(0, 0.2 / width)

    ax_exin[0, 0].set_ylabel('Intrinsicity', fontsize=fontsize)
    ax_exin[1, 0].set_ylabel('Normalized count', fontsize=fontsize)
    ax_exin[1, 1].set_xlabel('Extrinsicity - Intrinsicity', fontsize=fontsize)

    for i in [0, 2]:
        ax_exin[0, i].set_xticks([0, 1])
        ax_exin[0, i].set_xticklabels(['0', '1'])
        ax_exin[1, i].set_xticks([-0.4, 0, 0.4])
        ax_exin[1, i].set_xticklabels(['-0.4', '0', '0.4'])
    for ax_each in ax_exin.ravel():
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    # two-sample test, df = n1+n2 - 2
    ratio_1 = smallerdf[smallerdf['ca'] == 'CA1'][ratio_key].to_numpy()
    ratio_2 = smallerdf[smallerdf['ca'] == 'CA2'][ratio_key].to_numpy()
    ratio_3 = smallerdf[smallerdf['ca'] == 'CA3'][ratio_key].to_numpy()
    n1, n2, n3 = ratio_1.shape[0], ratio_2.shape[0], ratio_3.shape[0]

    equal_var = False
    t_13, p_13 = ttest_ind(ratio_1, ratio_3, equal_var=equal_var)
    t_23, p_23 = ttest_ind(ratio_2, ratio_3, equal_var=equal_var)
    t_12, p_12 = ttest_ind(ratio_1, ratio_2, equal_var=equal_var)

    stat_record(stat_fn, False, 'Welch\'s t-test: CA1 vs CA3, t(%d)=%0.2f, p=%s' % (n1 + n3 - 2, t_13, p2str(p_13)))
    stat_record(stat_fn, False, 'Welch\'s t-test: CA2 vs CA3, t(%d)=%0.2f, p=%s' % (n2 + n3 - 2, t_23, p2str(p_23)))
    stat_record(stat_fn, False, 'Welch\'s t-test: CA1 vs CA2, t(%d)=%0.2f, p=%s' % (n1 + n2 - 2, t_12, p2str(p_12)))

    contindf = pd.DataFrame(contindf_dict)
    contindf13, contindf23, contindf12 = contindf[['CA1', 'CA3']], contindf[['CA2', 'CA3']], contindf[['CA1', 'CA2']]
    fisherp12, n12, fishertxt12 = my_fisher_2way(contindf12.to_numpy())
    fisherp23, n23, fishertxt23 = my_fisher_2way(contindf23.to_numpy())
    fisherp13, n13, fishertxt13 = my_fisher_2way(contindf13.to_numpy())

    stat_record(stat_fn, False, "Ex-in Frac. CA1 vs CA2, %s" % (fishertxt12))
    stat_record(stat_fn, False, "Ex-in Frac. CA2 vs CA3, %s" % (fishertxt23))
    stat_record(stat_fn, False, "Ex-in Frac. CA1 vs CA3, %s" % (fishertxt13))
    fig_exin.tight_layout()

    fig_exin.savefig(join(save_dir, 'exp_exintrinsic.png'), dpi=dpi)
    fig_exin.savefig(join(save_dir, 'exp_exintrinsic.eps'), dpi=dpi)


def plot_pairangle_similarity_analysis(expdf, save_dir=None):
    stat_fn = 'fig6_pair_exin_simdisim.txt'
    stat_record(stat_fn, True)
    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'
    # anglekey1, anglekey2 = 'precess_angle1', 'precess_angle2'
    anglekey1, anglekey2 = 'rate_angle1', 'rate_angle2'
    figl = total_figw / 2 / 3

    fig_1d, ax_1d = plt.subplots(1, 3, figsize=(figl * 5, figl * 2), sharex='row', sharey='row')
    fig_1dca, ax_1dca = plt.subplots(1, 2, figsize=(figl * 5, figl * 2.5), sharex=True, sharey=True)
    fig_2d, ax_2d = plt.subplots(2, 3, figsize=(figl * 3, figl * 2.75), sharex='row', sharey='row')

    ratiosimdict = dict()
    ratiodisimdict = dict()
    df_exin_dict = dict(ca=[], sim=[], exnum=[], innum=[])

    ms = 1
    frac = 2
    for caid, (ca, cadf) in enumerate(expdf.groupby(by='ca')):
        cadf = cadf[(~cadf[ratio_key].isna()) & (~cadf[anglekey1].isna()) & (~cadf[anglekey2].isna())].reset_index(
            drop=True)

        s1, s2 = cadf[anglekey1], cadf[anglekey2]
        oratio, ominus, oplus = cadf[ratio_key], cadf[ominus_key], cadf[oplus_key]

        circdiff = np.abs(cdiff(s1, s2))
        simidx = np.where(circdiff < (np.pi / frac))[0]
        disimidx = np.where(circdiff > (np.pi - np.pi / frac))[0]
        orsim, ordisim = oratio[simidx], oratio[disimidx]
        exnumsim, exnumdisim = np.sum(orsim > 0), np.sum(ordisim > 0)
        innumsim, innumdisim = np.sum(orsim <= 0), np.sum(ordisim <= 0)
        exfracsim, exfracdisim = exnumsim / (exnumsim + innumsim), exnumdisim / (exnumdisim + innumdisim)
        ratiosimdict[ca] = orsim
        ratiodisimdict[ca] = ordisim

        # Ex-in for sim & dissim (ax_1d)
        obins_sim, oedges_sim = np.histogram(orsim, bins=30)
        oedges_sim_m = midedges(oedges_sim)
        obins_disim, oedges_disim = np.histogram(ordisim, bins=30)
        oedges_disim_m = midedges(oedges_disim)
        ax_1d[caid].plot(oedges_sim_m, np.cumsum(obins_sim) / np.sum(obins_sim), label='sim')
        ax_1d[caid].plot(oedges_disim_m, np.cumsum(obins_disim) / np.sum(obins_disim), label='dissimilar')

        pval1d_rs, _, _, _ = my_kruskal_2samp(orsim, ordisim, 'SimilarOR', 'DissimilarOR')
        tval1d_t, pval1d_t = ttest_ind(orsim, ordisim, equal_var=False)
        ax_1d[caid].text(0, 0.2, 'sim-dissim\nrs=%0.4f\nt=%0.4f' % (pval1d_rs, pval1d_t), fontsize=legendsize)
        ax_1d[caid].tick_params(labelsize=ticksize)
        customlegend(ax_1d[caid], fontsize=legendsize, loc='upper left')

        # Ex-in for CA1, CA2, CA3 (ax_1dca)
        ax_1dca[0].plot(oedges_sim_m, np.cumsum(obins_sim) / np.sum(obins_sim), label=ca, c=ca_c[ca])
        ax_1dca[1].plot(oedges_disim_m, np.cumsum(obins_disim) / np.sum(obins_disim), label=ca, c=ca_c[ca])

        # 2D scatter of ex-intrinsicitiy for similar/dissimlar (ax_2d)
        p_exfracsim, numsim, exfracsimtxt = my_chi_1way([exnumsim, innumsim])
        p_exfracdisim, numdisim, exfracdisimtxt = my_chi_1way([exnumdisim, innumdisim])
        stat_record(stat_fn, False, 'Similar ex-in ratio, %s, %d:%d=%0.2f, %s' % \
                    (ca, exnumsim, innumsim, exnumsim / innumsim, exfracsimtxt))
        stat_record(stat_fn, False, 'Dissimilar ex-in ratio, %s, %d:%d=%0.2f, %s' % \
                    (ca, exnumdisim, innumdisim, exnumdisim / innumdisim, exfracdisimtxt))

        ax_2d[0, caid].scatter(ominus[simidx], oplus[simidx], s=ms, c=ca_c[ca], marker='.')
        ax_2d[0, caid].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
        ax_2d[0, caid].annotate('%0.2f' % (exnumsim / innumsim), xy=(0.015, 0.17), xycoords='axes fraction',
                                size=legendsize, color='r')
        ax_2d[0, caid].annotate('p=%s' % (p2str(p_exfracsim)), xy=(0.015, 0.025), xycoords='axes fraction',
                                size=legendsize)
        ax_2d[0, caid].set_yticks([0, 1])
        ax_2d[0, caid].set_xlim(0, 1)
        ax_2d[0, caid].set_ylim(0, 1)
        ax_2d[0, caid].tick_params(axis='both', which='major', labelsize=ticksize)

        ax_2d[1, caid].scatter(ominus[disimidx], oplus[disimidx], s=ms, c=ca_c[ca], marker='.')
        ax_2d[1, caid].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
        ax_2d[1, caid].annotate('%0.2f' % (exnumdisim / innumdisim), xy=(0.015, 0.17), xycoords='axes fraction',
                                size=legendsize, color='r')
        ax_2d[1, caid].annotate('p=%s' % (p2str(p_exfracdisim)), xy=(0.015, 0.025), xycoords='axes fraction',
                                size=legendsize)
        ax_2d[1, caid].set_yticks([0, 1])
        ax_2d[1, caid].set_xlim(0, 1)
        ax_2d[1, caid].set_ylim(0, 1)
        ax_2d[1, caid].tick_params(axis='both', which='major', labelsize=ticksize)

        # Append to contingency table
        df_exin_dict['ca'].append(ca)
        df_exin_dict['sim'].append('similar')
        df_exin_dict['exnum'].append(exnumsim)
        df_exin_dict['innum'].append(innumsim)
        df_exin_dict['ca'].append(ca)
        df_exin_dict['sim'].append('dissimilar')
        df_exin_dict['exnum'].append(exnumdisim)
        df_exin_dict['innum'].append(innumdisim)

    ax_2d[0, 0].set_ylabel('Intrinsicity', fontsize=fontsize)
    ax_2d[0, 1].set_xlabel('\n', fontsize=fontsize)
    ax_2d[1, 0].set_ylabel('Intrinsicity', fontsize=fontsize)
    ax_2d[1, 1].set_xlabel('Extrinsicity', fontsize=fontsize)
    for i in range(3):
        ax_2d[1, i].set_xticks([0, 1])

    for ax_each in ax_2d.ravel():
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
    ax_2d[0, 1].set_title('Similar pairs', fontsize=titlesize)
    ax_2d[1, 1].set_title('Dissimilar pairs', fontsize=titlesize)

    # 2-way chisquare for ex-in fraction
    df_exin = pd.DataFrame(df_exin_dict)
    allcadfexin = []
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):  # Compare between sim and dissim
        dftmp = df_exin[df_exin['ca'] == ca]
        dftmp.index = dftmp.sim
        dftmp = dftmp[['exnum', 'innum']]
        sim_exnum, sim_innum = dftmp.loc['similar', ['exnum', 'innum']]
        dissim_exnum, dissim_innum = dftmp.loc['dissimilar', ['exnum', 'innum']]
        allcadfexin.append(dftmp)

        pfisher, nsimdissim, fishertxt = my_fisher_2way(dftmp.to_numpy())
        stattxt = 'Two-way, %s, similar=%d:%d=%0.2f, dissimilar=%d:%d=%0.2f, %s'
        stat_record(stat_fn, False, stattxt % \
                    (ca, sim_exnum, sim_innum, sim_exnum / sim_innum,
                     dissim_exnum, dissim_innum, dissim_exnum / dissim_innum, fishertxt))
    table_allca = allcadfexin[0] + allcadfexin[1] + allcadfexin[2]
    simexn, dissimexn = table_allca.loc['similar', 'exnum'], table_allca.loc['dissimilar', 'exnum']
    siminn, dissiminn = table_allca.loc['similar', 'innum'], table_allca.loc['dissimilar', 'innum']
    pfisher_allca, _, fishertxt_allca = my_fisher_2way(np.array([[simexn, dissimexn], [siminn, dissiminn]]))
    stat_record(stat_fn, False, 'All CAs, similar=%d:%d=%0.2f, dissimilar=%d:%d=%0.2f, %s' % (
        simexn, siminn, simexn / siminn, dissimexn, dissiminn, dissimexn / dissiminn, fishertxt_allca))

    # Compare between sim and dissim across CAs
    for simidtmp, sim_label in enumerate(['similar', 'dissimilar']):
        for capairid, (calabel1, calabel2) in enumerate((('CA1', 'CA2'), ('CA2', 'CA3'), ('CA1', 'CA3'))):
            dftmp = df_exin[(df_exin['sim'] == sim_label) &
                            ((df_exin['ca'] == calabel1) | (df_exin['ca'] == calabel2))][['exnum', 'innum']]
            exnum, innum = dftmp['exnum'].sum(), dftmp['innum'].sum()
            btwca_p, _, btwca_txt = my_fisher_2way(dftmp.to_numpy())
            stattxt = 'Two-way, %s, %s vs %s, %d:%d=%0.2f, %s'
            stat_record(stat_fn, False,
                        stattxt % (sim_label, calabel1, calabel2, exnum, innum, exnum / innum, btwca_txt))

    fig_2d.tight_layout()

    # Saving
    for figext in ['png', 'eps']:
        fig_2d.savefig(join(save_dir, 'exp_simdissim_exintrinsic2d.%s' % (figext)), dpi=dpi)


def plot_intrinsic_precession_property(expdf, plot_example, save_dir):
    import warnings
    warnings.filterwarnings("ignore")
    # # ============================== Organize data =================================
    pdf_dict = dict(pair_id=[], ca=[], overlap_ratio=[], lagAB=[], lagBA=[], onset=[], slope=[], direction=[], dsp=[],
                    phasesp=[], precess_exist=[])

    for i in range(expdf.shape[0]):

        ca, overlap_ratio, lagAB, lagBA, passdf = expdf.loc[
            i, ['ca', 'overlap_ratio', 'phaselag_AB', 'phaselag_BA', 'precess_dfp']]
        pair_id = expdf.loc[i, 'pair_id']

        num_pass = passdf.shape[0]
        for npass in range(num_pass):
            onset1, slope1, onset2, slope2, direction = passdf.loc[
                npass, ['rcc_c1', 'rcc_m1', 'rcc_c2', 'rcc_m2', 'direction']]

            dsp1, dsp2, phasesp1, phasesp2 = passdf.loc[npass, ['dsp1', 'dsp2', 'phasesp1', 'phasesp2']]
            precess_exist1, precess_exist2 = passdf.loc[npass, ['precess_exist1', 'precess_exist2']]

            pdf_dict['pair_id'].extend([pair_id] * 2)
            pdf_dict['ca'].extend([ca] * 2)
            pdf_dict['overlap_ratio'].extend([overlap_ratio] * 2)
            pdf_dict['lagAB'].extend([lagAB] * 2)
            pdf_dict['lagBA'].extend([lagBA] * 2)
            pdf_dict['onset'].extend([onset1, onset2])
            pdf_dict['slope'].extend([slope1, slope2])
            pdf_dict['direction'].extend([direction] * 2)
            pdf_dict['precess_exist'].extend([precess_exist1, precess_exist2])

            pdf_dict['dsp'].append(dsp1)
            pdf_dict['dsp'].append(dsp2)

            pdf_dict['phasesp'].append(phasesp1)
            pdf_dict['phasesp'].append(phasesp2)

    pdf = pd.DataFrame(pdf_dict)
    pdf = pdf[pdf['slope'].abs() < 1.9].reset_index(drop=True)
    preferAB_mask = (pdf['overlap_ratio'] < 0) & (pdf['lagAB'] > 0) & (pdf['lagBA'] > 0) & (pdf['direction'] == 'A->B')
    nonpreferAB_mask = (pdf['overlap_ratio'] < 0) & (pdf['lagAB'] < 0) & (pdf['lagBA'] < 0) & (
                pdf['direction'] == 'A->B')
    preferBA_mask = (pdf['overlap_ratio'] < 0) & (pdf['lagAB'] < 0) & (pdf['lagBA'] < 0) & (pdf['direction'] == 'B->A')
    nonpreferBA_mask = (pdf['overlap_ratio'] < 0) & (pdf['lagAB'] > 0) & (pdf['lagBA'] > 0) & (
                pdf['direction'] == 'B->A')
    prefer_pdf = pdf[preferAB_mask | preferBA_mask].reset_index(drop=True)
    nonprefer_pdf = pdf[nonpreferAB_mask | nonpreferBA_mask].reset_index(drop=True)

    # # ============================== Functions =================================

    def plot_exin_correlograms(ax, binsAB, binsBA, edm):

        edm_width = edm[1] - edm[0]
        binsABnorm, binsBAnorm = binsAB / binsAB.sum(), binsBA / binsBA.sum()

        ax.bar(edm, binsABnorm, width=edm_width, color='tan', label=r'$A\rightarrow B$')
        ax.bar(edm, binsBAnorm, width=edm_width, color='slategray', label=r'$B\rightarrow A$')
        customlegend(ax, fontsize=legendsize, bbox_to_anchor=(-0.5, 0.2), loc='lower left')

        ax.set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
        ax.set_xticklabels(['', '-0.1', '', '0', '', '0.1', ''])
        ax.set_xlabel('Time lag (s)', fontsize=legendsize)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

    def plot_intrinsic_direction_illustration(ax, expdf, pair_id, pre_c, in_direction):
        color_A, color_B = 'turquoise', 'purple'

        i = expdf[expdf['pair_id'] == pair_id].index[0]
        pass_dfp = expdf.loc[i, 'precess_dfp']
        pair_id, xyval1, xyval2 = expdf.loc[i, ['pair_id', 'xyval1', 'xyval2']]
        fieldcoor1, fieldcoor2 = expdf.loc[pair_id, ['com1', 'com2']]
        corrAB, corrBA, inval, exval = expdf.loc[i, ['corr_info_AB', 'corr_info_BA', 'overlap_plus', 'overlap_minus']]
        ABpass_dfp = pass_dfp[pass_dfp['direction'] == 'A->B'].reset_index(drop=True)
        BApass_dfp = pass_dfp[pass_dfp['direction'] == 'B->A'].reset_index(drop=True)

        # Correlation information - ax[0]
        binsAB, edges = corrAB[0], corrAB[1]
        binsBA, _ = corrBA[0], corrBA[1]
        edm = midedges(edges)
        binsABnorm, binsBAnorm = binsAB / binsAB.sum(), binsBA / binsBA.sum()
        plot_exin_correlograms(ax[0], binsAB, binsBA, edm)

        # Plot AB passes - ax[1]
        pass_color = pre_c['prefer'] if in_direction == 'A->B' else pre_c['nonprefer']
        ax[1].plot(xyval1[:, 0], xyval1[:, 1], c=color_A, label='A')
        ax[1].plot(xyval2[:, 0], xyval2[:, 1], c=color_B, label='B')
        ax[1].scatter(fieldcoor1[0], fieldcoor1[1], c=color_A, marker='^', s=8, zorder=3.1)
        ax[1].scatter(fieldcoor2[0], fieldcoor2[1], c=color_B, marker='^', s=8, zorder=3.1)
        for j in range(ABpass_dfp.shape[0]):
            x, y = ABpass_dfp.loc[j, ['x', 'y']]
            ax[1].plot(x, y, c=pass_color)
            ax[1].plot(x[0], y[0], c='k', marker='x')
        customlegend(ax[1], fontsize=legendsize, bbox_to_anchor=(0.9, 0.7), loc='lower left')
        ax[1].axis('off')

        # Plot BA passes - ax[2]
        pass_color = pre_c['prefer'] if in_direction == 'B->A' else pre_c['nonprefer']
        ax[2].plot(xyval1[:, 0], xyval1[:, 1], c=color_A)
        ax[2].plot(xyval2[:, 0], xyval2[:, 1], c=color_B)
        ax[2].scatter(fieldcoor1[0], fieldcoor1[1], c=color_A, marker='^', s=8, zorder=3.1)
        ax[2].scatter(fieldcoor2[0], fieldcoor2[1], c=color_B, marker='^', s=8, zorder=3.1)
        for j in range(BApass_dfp.shape[0]):
            x, y = BApass_dfp.loc[j, ['x', 'y']]
            ax[2].plot(x, y, c=pass_color)
            ax[2].plot(x[0], y[0], c='k', marker='x')
        ax[2].axis('off')

    def plot_prefer_nonprefer(ax, prefer_pdf, nonprefer_pdf, pre_c, stat_fn):
        '''

        Parameters
        ----------
        ax: ndarray
            Numpy array of axes objects with shape (6, ).
            ax[0] - Fraction of negative slope
            ax[1] - Prefer's precession
            ax[2] - Nonprefer's precession
            ax[3] - 2D scatter of onset vs slope for Prefer and Nonprefer
            ax[4] - Difference in phases for Prefer and Nonprefer

        prefer_pdf: dataframe
            Pandas dataframe containing prefered passes (including non-precessing)
        nonprefer_pdf: dataframe
            Pandas dataframe containing nonprefered passes (including non-precessing)

        Returns
        -------
        '''
        precess_ms = 1
        prefer_allslope = prefer_pdf['slope'].to_numpy() * np.pi * 2
        nonprefer_allslope = nonprefer_pdf['slope'].to_numpy() * np.pi * 2
        prefer_slope = prefer_pdf[prefer_pdf['precess_exist']]['slope'].to_numpy() * np.pi * 2
        nonprefer_slope = nonprefer_pdf[nonprefer_pdf['precess_exist']]['slope'].to_numpy() * np.pi * 2
        prefer_onset = prefer_pdf[prefer_pdf['precess_exist']]['onset'].to_numpy()
        nonprefer_onset = nonprefer_pdf[nonprefer_pdf['precess_exist']]['onset'].to_numpy()
        prefer_dsp = np.concatenate(prefer_pdf[prefer_pdf['precess_exist']]['dsp'].to_list())
        nonprefer_dsp = np.concatenate(nonprefer_pdf[nonprefer_pdf['precess_exist']]['dsp'].to_list())
        prefer_phasesp = np.concatenate(prefer_pdf[prefer_pdf['precess_exist']]['phasesp'].to_list())
        nonprefer_phasesp = np.concatenate(nonprefer_pdf[nonprefer_pdf['precess_exist']]['phasesp'].to_list())

        # Precession fraction
        prefer_neg_n, prefer_pos_n = (prefer_allslope < 0).sum(), (prefer_allslope > 0).sum()
        nonprefer_neg_n, nonprefer_pos_n = (nonprefer_allslope < 0).sum(), (nonprefer_allslope > 0).sum()
        table_arr = np.array([[prefer_neg_n, prefer_pos_n], [nonprefer_neg_n, nonprefer_pos_n]])
        table_df = pd.DataFrame(table_arr, columns=['-ve', '+ve'], index=['Prefer', 'Nonprefer'])
        fishfrac_p, ntotalprecess, fishfrac_text = my_fisher_2way(table_arr)
        prefer_frac = prefer_neg_n / (prefer_pos_n + prefer_neg_n)
        nonprefer_frac = nonprefer_neg_n / (nonprefer_pos_n + nonprefer_neg_n)
        bar_w, bar_x, bar_y = 0.5, np.array([0, 1]), np.array([prefer_frac, nonprefer_frac])
        ax[0].bar(bar_x, bar_y, width=bar_w, color=[pre_c['prefer'], pre_c['nonprefer']])
        ax[0].errorbar(x=bar_x.mean(), y=0.85, xerr=0.5, c='k', capsize=2.5)
        ax[0].text(x=0, y=0.92, s='p=%s' % (p2str(fishfrac_p)), fontsize=legendsize)
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(['Same', 'Opp'])
        ax[0].set_yticks([0, 0.5, 1])
        ax[0].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        ax[0].set_ylim(0, 1.1)
        ax[0].set_yticklabels(['0', '', '1'])
        ax[0].set_ylabel('Precession\nfraction', fontsize=legendsize)
        stat_record(stat_fn, False, 'Precess fraction\n' + table_df.to_string() + '\n%s' % (fishfrac_text))

        # Prefer precession
        xdum = np.linspace(0, 1, 10)
        mean_phasesp1 = circmean(prefer_phasesp)
        combined_slope, combined_onset = np.median(prefer_slope), circmean(prefer_onset)
        ydum1 = combined_onset + xdum * combined_slope
        ax[1].scatter(prefer_dsp, prefer_phasesp, marker='.', c=pre_c['prefer'], s=precess_ms)
        ax[1].scatter(prefer_dsp, prefer_phasesp + 2 * np.pi, marker='.', c=pre_c['prefer'], s=precess_ms)
        ax[1].plot(xdum, ydum1, c='k')
        ax[1].plot(xdum, ydum1 + 2 * np.pi, c='k')
        ax[1].plot(xdum, ydum1 + 2 * np.pi, c='k')
        ax[1].axhline(mean_phasesp1, xmin=0, xmax=0.2, color='k')
        ax[1].set_xticks([0, 0.5, 1])
        ax[1].set_xticklabels(['0', '', '1'])
        ax[1].set_xlabel('Position', fontsize=legendsize, labelpad=-5)
        ax[1].set_xlim(0, 1)
        ax[1].set_yticks([0, np.pi])
        ax[1].set_yticklabels(['0', '$\pi$'])
        ax[1].set_ylim(-np.pi + (np.pi / 2), np.pi + (np.pi / 2))
        ax[1].set_ylabel('Phase (rad)', fontsize=legendsize)

        # Non-Prefer precession
        xdum = np.linspace(0, 1, 10)
        mean_phasesp2 = circmean(nonprefer_phasesp)
        combined_slope, combined_onset = np.median(nonprefer_slope), circmean(nonprefer_onset)
        ydum2 = combined_onset + xdum * combined_slope
        ax[2].scatter(nonprefer_dsp, nonprefer_phasesp, marker='.', c=pre_c['nonprefer'], s=precess_ms)
        ax[2].scatter(nonprefer_dsp, nonprefer_phasesp + 2 * np.pi, marker='.', c=pre_c['nonprefer'], s=precess_ms)
        ax[2].plot(xdum, ydum2, c='k')
        ax[2].plot(xdum, ydum2 + 2 * np.pi, c='k')
        ax[2].plot(xdum, ydum2 + 2 * np.pi, c='k')
        ax[2].axhline(mean_phasesp2, xmin=0, xmax=0.2, color='k')
        ax[2].plot(xdum, ydum1, c='k', linestyle='dotted')
        ax[2].plot(xdum, ydum1 + 2 * np.pi, c='k', linestyle='dotted')
        ax[2].plot(xdum, ydum1 + 2 * np.pi, c='k', linestyle='dotted')
        ax[2].axhline(mean_phasesp1, xmin=0, xmax=0.2, color='k', linestyle='dotted')
        ax[2].set_xticks([0, 0.5, 1])
        ax[2].set_xticklabels(['0', '', '1'])
        ax[2].set_xlabel('Position', fontsize=legendsize, labelpad=-10)
        ax[2].set_xlim(0, 1)
        ax[2].set_yticks([0, np.pi])
        ax[2].set_yticklabels(['0', '$\pi$'])
        ax[2].set_ylim(-np.pi + (np.pi / 2), np.pi + (np.pi / 2))
        ax[2].set_ylabel('Phase (rad)', fontsize=legendsize)

        # Onset
        p_onset, _, descrips_onset, ww_onset_txt = my_ww_2samp(prefer_onset, nonprefer_onset, 'Same', 'Opp')
        ((prefer_onset_mean, _), (nonprefer_onset_mean, _)) = descrips_onset
        onset_bins = np.linspace(0, 2 * np.pi, 20)
        prefer_onsetbins, _ = np.histogram(prefer_onset, bins=onset_bins)
        nonprefer_onsetbins, _ = np.histogram(nonprefer_onset, bins=onset_bins)
        ax[3].plot(onset_bins[:-1], np.cumsum(prefer_onsetbins) / prefer_onsetbins.sum(), c=pre_c['prefer'],
                   label='prefer')
        ax[3].plot(onset_bins[:-1], np.cumsum(nonprefer_onsetbins) / nonprefer_onsetbins.sum(), c=pre_c['nonprefer'],
                   label='nonprefer')
        ax[3].annotate('p=%s' % (p2str(p_onset)), xy=(0.3, 0.05), xycoords='axes fraction', fontsize=legendsize)
        ax[3].set_xticks([0, np.pi, 2 * np.pi])
        ax[3].set_xticklabels(['0', '', '$2\pi$'])
        ax[3].set_xlabel('Onset\n(rad)', fontsize=legendsize, labelpad=-10)
        ax[3].set_yticks([0, 0.5, 1])
        ax[3].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        ax[3].set_yticklabels(['0', '', '1'])
        ax[3].set_ylabel('Cumulative\ndensity', fontsize=legendsize)
        stat_record(stat_fn, False,
                    'Onset means: Prefer=%0.2f, Nonprefer=%0.2f' % (prefer_onset_mean, nonprefer_onset_mean))
        stat_record(stat_fn, False, 'Onset difference, %s' % (ww_onset_txt))

        # Slope
        p_slope, _, descrips_slope, slope_rs_text = my_kruskal_2samp(prefer_slope, nonprefer_slope, 'Same', 'Opp')
        ((prefer_slope_mdn, _, _), (nonprefer_slope_mdn, _, _)) = descrips_slope
        slope_bins = np.linspace(-2 * np.pi, 0, 20)
        prefer_slopebins, _ = np.histogram(prefer_slope, bins=slope_bins)
        nonprefer_slopebins, _ = np.histogram(nonprefer_slope, bins=slope_bins)
        ax[4].plot(slope_bins[:-1], np.cumsum(prefer_slopebins) / prefer_slopebins.sum(), c=pre_c['prefer'],
                   label='prefer')
        ax[4].plot(slope_bins[:-1], np.cumsum(nonprefer_slopebins) / nonprefer_slopebins.sum(), c=pre_c['nonprefer'],
                   label='nonprefer')
        ax[4].annotate('p=%s' % (p2str(p_slope)), xy=(0.3, 0.05), xycoords='axes fraction', fontsize=legendsize)
        ax[4].set_xticks([-2 * np.pi, -np.pi, 0])
        ax[4].set_xticklabels(['$-2\pi$', '', '0'])
        ax[4].set_xlabel('Slope\n(rad)', fontsize=legendsize, labelpad=-10)
        ax[4].set_yticks([0, 0.5, 1])
        ax[4].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        ax[4].set_yticklabels(['0', '', '1'])
        ax[4].set_ylabel('Cumulative\ndensity', fontsize=legendsize)
        stat_record(stat_fn, False,
                    'Slope medians: Prefer=%0.2f, Nonprefer=%0.2f' % (prefer_slope_mdn, nonprefer_slope_mdn))
        stat_record(stat_fn, False, 'Slope difference, %s' % (slope_rs_text))

        # Spike phase
        prefer_phasesp_mean, nonprefer_phasesp_mean = circmean(prefer_phasesp), circmean(nonprefer_phasesp)
        p_phasesp, _, _, ww_phasesp_txt = my_ww_2samp(prefer_phasesp, nonprefer_phasesp, 'Same', 'Opp')
        phasesp_bins = np.linspace(-np.pi, np.pi, 20)
        prefer_phasespbins, _ = np.histogram(prefer_phasesp, bins=phasesp_bins)
        nonprefer_phasespbins, _ = np.histogram(nonprefer_phasesp, bins=phasesp_bins)
        norm_preferphasebins = prefer_phasespbins / prefer_phasespbins.sum()
        norm_nonpreferphasebins = nonprefer_phasespbins / nonprefer_phasespbins.sum()
        maxl = max(norm_preferphasebins.max(), norm_nonpreferphasebins.max())
        ax[5].step(midedges(phasesp_bins), norm_preferphasebins, color=pre_c['prefer'], label='prefer')
        ax[5].step(midedges(phasesp_bins), norm_nonpreferphasebins, color=pre_c['nonprefer'], label='nonprefer')
        ax[5].axvline(prefer_phasesp_mean, ymin=0.8, ymax=1, color=pre_c['prefer'], lw=1)
        ax[5].axvline(nonprefer_phasesp_mean, ymin=0.8, ymax=1, color=pre_c['nonprefer'], lw=1)
        ax[5].annotate('p=%s' % (p2str(p_phasesp)), xy=(0.2, 0.05), xycoords='axes fraction', fontsize=legendsize)
        ax[5].set_xticks([-np.pi, 0, np.pi])
        ax[5].set_xticklabels(['$-\pi$', '', '$\pi$'])
        ax[5].set_xlabel('Phase\n(rad)', fontsize=legendsize, labelpad=-10)
        ax[5].set_yticks([0, 0.10])
        ax[5].set_yticklabels(['0', '1'])
        ax[5].set_ylim(0, 0.12)
        ax[5].set_ylabel('Density', fontsize=legendsize)
        stat_record(stat_fn, False, 'Phase difference, %s' % (ww_phasesp_txt))

        # ALL
        for ax_each in ax:
            ax_each.tick_params(labelsize=ticksize)
            ax_each.spines['right'].set_visible(False)
            ax_each.spines['top'].set_visible(False)

        return ax

    def compare_to_extrinsic(expdf, stat_fn):
        from pycircstat.descriptive import std
        stat_record(stat_fn, False, '========= Ex-In comparison==========')
        for caid, (ca, cadf) in enumerate(expdf.groupby('ca')):
            ex_capdf = pd.concat(cadf[cadf['overlap_ratio'] > 0]['precess_dfp'].to_list(), axis=0, ignore_index=True)
            in_capdf = pd.concat(cadf[cadf['overlap_ratio'] < 0]['precess_dfp'].to_list(), axis=0, ignore_index=True)
            ex_phasesp = np.concatenate(
                ex_capdf[ex_capdf.precess_exist]['phasesp1'].to_list() + ex_capdf[ex_capdf.precess_exist][
                    'phasesp2'].to_list())
            in_phasesp = np.concatenate(
                in_capdf[in_capdf.precess_exist]['phasesp1'].to_list() + in_capdf[in_capdf.precess_exist][
                    'phasesp2'].to_list())

            pww, _, exin_descrips, ww_exin_txt = my_ww_2samp(ex_phasesp, in_phasesp, 'Ex', 'In')
            (exmean, exsem), (inmean, insem) = exin_descrips
            stat_record(stat_fn, False,
                        '%s, spike phase difference, Ex ($%0.2f \pm %0.4f$) vs In ($%0.2f \pm %0.4f$), %s' % (
                        ca, exmean, exsem, inmean, insem, ww_exin_txt))

    # # ============================== Plotting =================================
    pre_c = dict(prefer='rosybrown', nonprefer='lightsteelblue')
    fig = plt.figure(figsize=(total_figw, total_figw))

    corr_w, stat_w = 0.4, 0.6

    # Intrinsic A->B Example
    xoffset, yoffset = 0, -0.06
    squeezex_corr, squeezey_corr = 0.025, 0.1
    yoffset_corr = squeezey_corr / 2
    squeezex_pair, squeezey_pair = 0.05, 0.07
    yoffset_pair = 0.025
    xgap_btwpair = 0.01
    corrAB_w, corrAB_h = corr_w / 2, corr_w / 2
    pairAB_y = 1 - corrAB_h + yoffset
    corrAB_y = 1 - corrAB_h * 2 + yoffset
    ax_corrAB = [
        fig.add_axes([0 + xoffset + corrAB_w / 2 + squeezex_corr / 2, pairAB_y + squeezey_corr / 2 + yoffset_corr,
                      corrAB_w - squeezex_corr, corrAB_h - squeezey_corr]),
        fig.add_axes([0 + xoffset + squeezex_pair / 2 + xgap_btwpair, corrAB_y + yoffset_pair + squeezey_pair / 2,
                      corrAB_w - squeezex_pair, corrAB_h - squeezey_pair]),
        fig.add_axes(
            [0 + xoffset + corrAB_w + squeezex_pair / 2 - xgap_btwpair, corrAB_y + yoffset_pair + squeezey_pair / 2,
             corrAB_w - squeezex_pair, corrAB_h - squeezey_pair]),
    ]

    # Intrinsic B->A Example
    xoffset, yoffset = 0, -0.15
    squeezex_corr, squeezey_corr = 0.025, 0.1
    yoffset_corr = squeezey_corr / 2
    squeezex_pair, squeezey_pair = 0.05, 0.07
    yoffset_pair = 0.025
    xgap_btwpair = 0.01
    corrBA_w, corrBA_h = corr_w / 2, corr_w / 2
    pairBA_y = 1 - corrBA_h * 3 + yoffset
    corrBA_y = 1 - corrBA_h * 4 + yoffset
    ax_corrBA = [
        fig.add_axes([0 + xoffset + corrBA_w / 2 + squeezex_corr / 2, pairBA_y + squeezey_corr / 2 + yoffset_corr,
                      corrBA_w - squeezex_corr, corrBA_h - squeezey_corr]),
        fig.add_axes([0 + xoffset + squeezex_pair / 2 + xgap_btwpair, corrBA_y + yoffset_pair + squeezey_pair / 2,
                      corrBA_w - squeezex_pair, corrBA_h - squeezey_pair]),
        fig.add_axes(
            [0 + xoffset + corrBA_w + squeezex_pair / 2 - xgap_btwpair, corrBA_y + yoffset_pair + squeezey_pair / 2,
             corrBA_w - squeezex_pair, corrBA_h - squeezey_pair]),
    ]

    # Params for CA1-CA3 columns
    xoffset, yoffset = 0.025, -0.01
    xgap_btwCAs = 0.015  # squeeze
    ygap_btwPrecess = 0.015  # squeeze
    ysqueeze_OS = 0.025  # for onset and slope, ax[3], ax[4]
    yoffset_each = [0, 0, 0.03, 0.04, 0.06, 0.05]

    # CA1 column
    ca1_w, ca1_h = stat_w / 3, 1 / 6
    ca1_x = corr_w
    squeezex, squeezey = 0.04, 0.05

    ax_ca1 = [fig.add_axes(
        [ca1_x + squeezex / 2 + xoffset + xgap_btwCAs, (1 - ca1_h) + squeezey / 2 + yoffset + yoffset_each[0],
         ca1_w - squeezex, ca1_h - squeezey]),
              fig.add_axes([ca1_x + squeezex / 2 + xoffset + xgap_btwCAs,
                            (1 - ca1_h * 2) + squeezey / 2 + yoffset + yoffset_each[1], ca1_w - squeezex,
                            ca1_h - squeezey]),
              fig.add_axes([ca1_x + squeezex / 2 + xoffset + xgap_btwCAs,
                            (1 - ca1_h * 3) + squeezey / 2 + yoffset + yoffset_each[2], ca1_w - squeezex,
                            ca1_h - squeezey]),
              fig.add_axes([ca1_x + squeezex / 2 + xoffset + xgap_btwCAs,
                            (1 - ca1_h * 4) + squeezey / 2 + yoffset + ysqueeze_OS / 2 + yoffset_each[3],
                            ca1_w - squeezex, ca1_h - squeezey - ysqueeze_OS]),
              fig.add_axes([ca1_x + squeezex / 2 + xoffset + xgap_btwCAs,
                            (1 - ca1_h * 5) + squeezey / 2 + yoffset + ysqueeze_OS / 2 + yoffset_each[4],
                            ca1_w - squeezex, ca1_h - squeezey - ysqueeze_OS]),
              fig.add_axes([ca1_x + squeezex / 2 + xoffset + xgap_btwCAs,
                            (1 - ca1_h * 6) + squeezey / 2 + yoffset + yoffset_each[5], ca1_w - squeezex,
                            ca1_h - squeezey]),
              ]

    # CA2 column
    ca2_w, ca2_h = stat_w / 3, 1 / 6
    ca2_x = corr_w + ca1_w
    squeezex, squeezey = 0.05, 0.05
    ax_ca2 = [fig.add_axes(
        [ca2_x + squeezex / 2 + xoffset, (1 - ca2_h) + squeezey / 2 + yoffset + yoffset_each[0], ca2_w - squeezex,
         ca2_h - squeezey]),
              fig.add_axes([ca2_x + squeezex / 2 + xoffset, (1 - ca2_h * 2) + squeezey / 2 + yoffset + yoffset_each[1],
                            ca2_w - squeezex, ca2_h - squeezey]),
              fig.add_axes([ca2_x + squeezex / 2 + xoffset, (1 - ca2_h * 3) + squeezey / 2 + yoffset + yoffset_each[2],
                            ca2_w - squeezex, ca2_h - squeezey]),
              fig.add_axes([ca2_x + squeezex / 2 + xoffset,
                            (1 - ca2_h * 4) + squeezey / 2 + yoffset + ysqueeze_OS / 2 + yoffset_each[3],
                            ca2_w - squeezex, ca2_h - squeezey - ysqueeze_OS]),
              fig.add_axes([ca2_x + squeezex / 2 + xoffset,
                            (1 - ca2_h * 5) + squeezey / 2 + yoffset + ysqueeze_OS / 2 + yoffset_each[4],
                            ca2_w - squeezex, ca2_h - squeezey - ysqueeze_OS]),
              fig.add_axes([ca2_x + squeezex / 2 + xoffset, (1 - ca2_h * 6) + squeezey / 2 + yoffset + yoffset_each[5],
                            ca2_w - squeezex, ca2_h - squeezey]),
              ]

    # CA2 column
    ca3_w, ca3_h = stat_w / 3, 1 / 6
    ca3_x = corr_w + ca1_w + ca2_w
    squeezex, squeezey = 0.05, 0.05
    ax_ca3 = [fig.add_axes(
        [ca3_x + squeezex / 2 + xoffset - xgap_btwCAs, (1 - ca3_h) + squeezey / 2 + yoffset + yoffset_each[0],
         ca3_w - squeezex, ca3_h - squeezey]),
              fig.add_axes([ca3_x + squeezex / 2 + xoffset - xgap_btwCAs,
                            (1 - ca3_h * 2) + squeezey / 2 + yoffset + yoffset_each[1], ca3_w - squeezex,
                            ca3_h - squeezey]),
              fig.add_axes([ca3_x + squeezex / 2 + xoffset - xgap_btwCAs,
                            (1 - ca3_h * 3) + squeezey / 2 + yoffset + yoffset_each[2], ca3_w - squeezex,
                            ca3_h - squeezey]),
              fig.add_axes([ca3_x + squeezex / 2 + xoffset - xgap_btwCAs,
                            (1 - ca3_h * 4) + squeezey / 2 + yoffset + ysqueeze_OS / 2 + yoffset_each[3],
                            ca3_w - squeezex, ca3_h - squeezey - ysqueeze_OS]),
              fig.add_axes([ca3_x + squeezex / 2 + xoffset - xgap_btwCAs,
                            (1 - ca3_h * 5) + squeezey / 2 + yoffset + ysqueeze_OS / 2 + yoffset_each[4],
                            ca3_w - squeezex, ca3_h - squeezey - ysqueeze_OS]),
              fig.add_axes([ca3_x + squeezex / 2 + xoffset - xgap_btwCAs,
                            (1 - ca3_h * 6) + squeezey / 2 + yoffset + yoffset_each[5], ca3_w - squeezex,
                            ca3_h - squeezey]),
              ]

    ax = [ax_ca1, ax_ca2, ax_ca3]

    pair_id_AB = 978
    pair_id_BA = 916
    stat_fn = 'fig7_Intrinsic_Direction.txt'
    stat_record(stat_fn, True)
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
        caprefer_df = prefer_pdf[prefer_pdf['ca'] == ca]
        canonprefer_df = nonprefer_pdf[nonprefer_pdf['ca'] == ca]
        stat_record(stat_fn, False, ('=' * 10) + ca + ('=' * 10))
        plot_prefer_nonprefer(ax[caid], caprefer_df, canonprefer_df, pre_c, stat_fn)
    if plot_example:
        plot_intrinsic_direction_illustration(np.array(ax_corrAB), expdf, pair_id_AB, pre_c, 'A->B')
        plot_intrinsic_direction_illustration(np.array(ax_corrBA), expdf, pair_id_BA, pre_c, 'B->A')
    compare_to_extrinsic(expdf, stat_fn)

    # Hide yticks and ylabels for column CA1 & CA2
    for i in range(6):
        ax_ca2[i].set_ylabel('')
        plt.setp(ax_ca2[i].get_yticklabels(), visible=False)
        ax_ca3[i].set_ylabel('')
        plt.setp(ax_ca3[i].get_yticklabels(), visible=False)

        if i == 1:  # Hide xticks and xlabels for A>B precession
            ax_ca1[i].set_xlabel('')
            plt.setp(ax_ca1[i].get_xticklabels(), visible=False)
            ax_ca2[i].set_xlabel('')
            plt.setp(ax_ca2[i].get_xticklabels(), visible=False)
            ax_ca3[i].set_xlabel('')
            plt.setp(ax_ca3[i].get_xticklabels(), visible=False)

        # Hide xlabels for column CA1 & CA3
        ax_ca1[i].set_xlabel('')
        ax_ca3[i].set_xlabel('')

    ax_ca1[5].annotate(r'$\times10^{-1}$', xy=(0, 0.9), xycoords='axes fraction', fontsize=legendsize)

    ax_corrAB[0].set_title('Pair#%d\nDirectional bias ' % (pair_id_AB) + r'$A\rightarrow B$', fontsize=legendsize,
                           ha='center')
    fig.text(0.1, 0.55, r'$A\rightarrow B$' + '\nSame', fontsize=legendsize, ha='center')
    fig.text(0.285, 0.55, r'$B\rightarrow A$' + '\nOpposite', fontsize=legendsize, ha='center')
    ax_corrBA[0].set_title('Pair#%d\nDirectional bias ' % (pair_id_BA) + r'$B\rightarrow A$', fontsize=legendsize,
                           ha='center')
    fig.text(0.1, 0.05, r'$A\rightarrow B$' + '\nOpposite', fontsize=legendsize, ha='center')
    fig.text(0.285, 0.05, r'$B\rightarrow A$' + '\nSame', fontsize=legendsize, ha='center')

    ax_ca1[0].set_title('CA1', fontsize=titlesize)
    ax_ca2[0].set_title('CA2', fontsize=titlesize)
    ax_ca3[0].set_title('CA3', fontsize=titlesize)
    fig.savefig(join(save_dir, 'exp_intrinsic_precession.pdf'), dpi=dpi)
    fig.savefig(join(save_dir, 'exp_intrinsic_precession.eps'), dpi=dpi)
    return

def plot_leading_enslaved_analysis(expdf, save_dir):
    """
    The script is intended for:
        1. Plot schematics with artificial data  (Fig 8B and 8D)
        2. Plot Overlap-Nonoverlap distributions (Fig 8E)
    The script starts with defining three functions:
        1. gen_intrinsic_model_data()
        2. gen_extrinsic_model_data()
        3. rcc_wrapper()
    Parameters
    ----------
    expdf
    save_dir

    Returns
    -------

    """
    os.makedirs(save_dir, exist_ok=True)
    stat_fn = 'fig8_LeadingEnslaved.txt'
    stat_record(stat_fn, overwrite=True)

    def gen_intrinsic_model_data(base_onset=3, slope=1.5*np.pi, opp_onsetshift=-0.36*np.pi,
                                 lead_slave_phasediff=0.14*np.pi, lead_slave_posdiff=0.075,
                                 num_spikes=10, pos_bound=(0, 0.925), skip=(0, 3)):
        # Pre-calculation
        if (num_spikes % 2) != 0:
            raise

        overlap_sep = int(num_spikes/2)

        # Same
        highest_phase = base_onset
        same_leadphase = np.linspace(highest_phase-slope,  highest_phase, num_spikes)
        same_leadphase = same_leadphase[::-1]
        same_leadpos = np.linspace(pos_bound[0], pos_bound[1], num_spikes)
        same_slavephase = same_leadphase[overlap_sep:] + lead_slave_phasediff
        same_slavepos = same_leadpos[overlap_sep:] + lead_slave_posdiff

        # Opp
        head_skip, end_skip = skip
        highest_phase = base_onset + opp_onsetshift
        opp_leadphase = np.linspace(highest_phase-slope,  highest_phase, num_spikes)
        opp_leadphase = opp_leadphase[::-1]
        if end_skip == 0:
            opp_leadphase = opp_leadphase[head_skip:]
        else:
            opp_leadphase = opp_leadphase[head_skip:-end_skip]
        opp_leadpos = np.linspace(pos_bound[0], pos_bound[1], num_spikes)
        if end_skip == 0:
            opp_leadpos = opp_leadpos[head_skip:]
        else:
            opp_leadpos = opp_leadpos[head_skip:-end_skip]
        opp_slavephase = opp_leadphase[0:overlap_sep-head_skip] + lead_slave_phasediff
        opp_slavepos = opp_leadpos[0:overlap_sep-head_skip] + lead_slave_posdiff

        # Overlap & Nonoverlap
        overlap_phase = np.concatenate( [same_leadphase[overlap_sep:],
                                         same_slavephase,
                                         opp_leadphase[:overlap_sep-head_skip], opp_slavephase])
        overlap_pos = np.concatenate( [ same_leadpos[overlap_sep:], same_slavepos, opp_leadpos[:overlap_sep-head_skip], opp_slavepos ])
        overlap_leadflag = np.array([1]*same_leadphase[overlap_sep:].shape[0] + \
                                    [0]*same_slavephase.shape[0] + \
                                    [1]*opp_leadphase[:overlap_sep-head_skip].shape[0] + \
                                    [0]*opp_slavephase.shape[0])
        nonoverlap_phase = np.concatenate( [ same_leadphase[:overlap_sep], opp_leadphase[overlap_sep-head_skip:]])
        nonoverlap_pos = np.concatenate( [ same_leadpos[:overlap_sep], opp_leadpos[overlap_sep-head_skip:]])

        # Same & Opp
        same_phase = np.concatenate([same_leadphase, same_slavephase])
        same_pos = np.concatenate([same_leadpos, same_slavepos])
        opp_phase = np.concatenate([opp_leadphase, opp_slavephase])
        opp_pos = np.concatenate([opp_leadpos, opp_slavepos])

        # store
        data = dict(
            same=dict(
                lead=dict(phase=same_leadphase, pos=same_leadpos),
                slave=dict(phase=same_slavephase, pos=same_slavepos),
                all=dict(phase=same_phase, pos=same_pos)
            ),
            opp=dict(
                lead=dict(phase=opp_leadphase, pos=opp_leadpos),
                slave=dict(phase=opp_slavephase, pos=opp_slavepos),
                all=dict(phase=opp_phase, pos=opp_pos)
            ),
            overlap=dict(
                phase=overlap_phase, pos=overlap_pos,
                leadflag=overlap_leadflag
            ),
            nonoverlap=dict(
                phase=nonoverlap_phase, pos=nonoverlap_pos
            )

        )
        return data

    def gen_extrinsic_model_data(base_onset=np.pi * 0.8, slope=1.4*np.pi,
                                 lead_slave_phasediff=0.4*np.pi, lead_slave_posdiff=0.075,
                                 num_spikes=12, pos_bound=(0, 0.925)):

        # # Pre-calculation
        n_onethird = int(num_spikes/3)

        # # Phase
        # Same & Opp share the same set of spikes (phase1, pos1, phase2, pos2)
        highest_phase = base_onset
        phase_all = np.linspace(highest_phase-slope,  highest_phase, num_spikes)
        phase_all = phase_all[::-1]
        pos_all = np.linspace(pos_bound[0], pos_bound[1], num_spikes)
        phase1, pos1 = phase_all[:n_onethird*2], pos_all[:n_onethird*2]
        phase2, pos2 = phase_all[n_onethird:] + lead_slave_phasediff, pos_all[n_onethird:] + lead_slave_posdiff

        # Overlap & Nonoverlap
        overlap_phase = np.concatenate([phase1[n_onethird:], phase2[:n_onethird]])
        overlap_pos = np.concatenate([pos1[n_onethird:], pos2[:n_onethird]])
        overlap_onetag = np.array([1] * pos1[n_onethird:].shape[0] + [0] * pos2[:n_onethird].shape[0])
        nonoverlap_phase = np.concatenate([phase1[:n_onethird], phase2[n_onethird:]])
        nonoverlap_pos = np.concatenate([pos1[:n_onethird], pos2[n_onethird:]])
        nonoverlap_onetag = np.array([1] * pos1[:n_onethird].shape[0] + [0] * pos2[n_onethird:].shape[0])

        # store
        data = dict(
            phase1=phase1,
            pos1=pos1,
            phase2=phase2,
            pos2=pos2,
            normal=dict(phase=np.concatenate([phase1, phase2]), pos=np.concatenate([pos1, pos2])),
            overlap=dict(phase=overlap_phase, pos=overlap_pos, onetag=overlap_onetag),
            nonoverlap=dict(phase=nonoverlap_phase, pos=nonoverlap_pos, onetag=nonoverlap_onetag)
        )
        return data

    def rcc_wrapper(dsp1, phasesp1, dsp2, phasesp2):
        nsp1, nsp2 = dsp1.shape[0], dsp2.shape[0]

        if nsp1 > 1:
            regress1 = rcc(dsp1, phasesp1, abound=(-2, 2))
            rcc_m1, rcc_c1 = regress1['aopt'], regress1['phi0']
        else:
            rcc_m1, rcc_c1 = None, None

        if nsp2 > 1:
            regress2 = rcc(dsp2, phasesp2, abound=(-2, 2))
            rcc_m2, rcc_c2 = regress2['aopt'], regress2['phi0']
        else:
            rcc_m2, rcc_c2 = None, None

        return rcc_m1, rcc_c1, nsp1, rcc_m2, rcc_c2, nsp2

    #  ========================= Plot Intrinsic precession schematics =========================
    print("Plotting Intrinsic precession schematics")
    total_figw_max = 4.16
    fig = plt.figure(figsize=(total_figw_max, 1.75), facecolor='w')
    y_squeeze = 0.3
    x_squeeze = 0.04
    x_squeeze_Den = 0.005
    x_shift_Den = 0
    cul_xshift = 0.0175
    whole_xshift = 0.007
    all_y, all_h = y_squeeze / 2, 1 - y_squeeze
    Den_w = 0.05
    Ras_w = 1/4-0.05

    sameRas_x = 0
    sameRas_ax = fig.add_axes([sameRas_x + x_squeeze / 2 + cul_xshift*4 - whole_xshift,
                               all_y, Ras_w - x_squeeze, all_h])
    sameDen_x = Ras_w
    sameDen_ax = fig.add_axes([sameDen_x + x_squeeze_Den / 2 + cul_xshift*3 + x_shift_Den - whole_xshift,
                               all_y, Den_w - x_squeeze_Den, all_h])

    oppRas_x = sameDen_x + Den_w
    oppRas_ax = fig.add_axes([oppRas_x + x_squeeze / 2 + cul_xshift*3 - whole_xshift,
                              all_y, Ras_w - x_squeeze, all_h])
    oppDen_x = oppRas_x + Ras_w
    oppDen_ax = fig.add_axes([oppDen_x + x_squeeze_Den / 2 + cul_xshift*2 + x_shift_Den - whole_xshift,
                              all_y, Den_w - x_squeeze_Den, all_h])

    overlapRas_x = oppDen_x + Den_w
    overlapRas_ax = fig.add_axes([overlapRas_x + x_squeeze / 2 + cul_xshift*2 - whole_xshift,
                                  all_y, Ras_w - x_squeeze, all_h])
    overlapDen_x = overlapRas_x + Ras_w
    overlapDen_ax = fig.add_axes([overlapDen_x + x_squeeze_Den / 2 + cul_xshift*1 + x_shift_Den - whole_xshift,
                                  all_y, Den_w - x_squeeze_Den, all_h])

    nonoverlapRas_x = overlapDen_x + Den_w
    nonoverlapRas_ax = fig.add_axes([nonoverlapRas_x + x_squeeze / 2 + cul_xshift*1 - whole_xshift,
                                     all_y, Ras_w - x_squeeze, all_h])
    nonoverlapDen_x = nonoverlapRas_x + Ras_w
    nonoverlapDen_ax = fig.add_axes([nonoverlapDen_x + x_squeeze_Den / 2 + cul_xshift*0 + x_shift_Den - whole_xshift,
                                     all_y, Den_w - x_squeeze_Den, all_h])

    lead_c, slave_c, slave_c2 = 'turquoise', 'purple', 'violet'
    same_c, opp_c ='rosybrown', 'lightsteelblue'
    ras_marker = '.'

    shade_c = '0.9'
    lw = 1
    ls2 = 'dotted'
    ms = 4
    mean_xmin = 0

    base_onset = 3
    slope = 1.5*np.pi
    opp_onsetshift = -0.36*np.pi
    lead_slave_phasediff1 = 0.14*np.pi
    lead_slave_phasediff2 = 0.71*np.pi
    skip = (0, 3)
    data = gen_intrinsic_model_data(base_onset=base_onset, slope=slope, opp_onsetshift=opp_onsetshift,
                                    lead_slave_phasediff=lead_slave_phasediff1, lead_slave_posdiff=0.075,
                                    num_spikes=10, pos_bound=(0, 0.925), skip=skip)
    data2 = gen_intrinsic_model_data(base_onset=base_onset, slope=slope, opp_onsetshift=opp_onsetshift,
                                     lead_slave_phasediff=lead_slave_phasediff2, lead_slave_posdiff=0.075,
                                     num_spikes=10, pos_bound=(0, 0.925), skip=skip)

    # # Calculate statistics
    overlap_mean = shiftcyc_full2half(circmean(data['overlap']['phase'] ))
    nonoverlap_mean = shiftcyc_full2half(circmean(data['nonoverlap']['phase']))
    same_mean = shiftcyc_full2half(circmean(data['same']['all']['phase']))
    opp_mean = shiftcyc_full2half(circmean(data['opp']['all']['phase']))
    overlap_mean2 = shiftcyc_full2half(circmean(data2['overlap']['phase'] ))
    same_mean2 = shiftcyc_full2half(circmean(data2['same']['all']['phase']))
    opp_mean2 = shiftcyc_full2half(circmean(data2['opp']['all']['phase']))

    bins, kappa = np.linspace(-3*np.pi, 3*np.pi, 30), 2.5
    circax, overlap_den = circular_density_1d(data['overlap']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, nonoverlap_den = circular_density_1d(data['nonoverlap']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, same_den = circular_density_1d(data['same']['all']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, opp_den = circular_density_1d(data['opp']['all']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, overlap_den2 = circular_density_1d(data2['overlap']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, same_den2 = circular_density_1d(data2['same']['all']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, opp_den2 = circular_density_1d(data2['opp']['all']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))

    sameRas_ax.axvspan(0.5, 1.05, color=shade_c, zorder=0)
    sameRas_ax.scatter(data['same']['lead']['pos'], data['same']['lead']['phase'], c=lead_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['same']['lead']['pos'], data['same']['lead']['phase']-2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['same']['lead']['pos'], data['same']['lead']['phase']+2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['same']['slave']['pos'], data['same']['slave']['phase'], c=slave_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['same']['slave']['pos'], data['same']['slave']['phase']-2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['same']['slave']['pos'], data['same']['slave']['phase']+2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data2['same']['slave']['pos'], data2['same']['slave']['phase'], c=slave_c2, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data2['same']['slave']['pos'], data2['same']['slave']['phase']-2*np.pi, c=slave_c2, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data2['same']['slave']['pos'], data2['same']['slave']['phase']+2*np.pi, c=slave_c2, marker=ras_marker, s=ms)
    sameRas_ax.set_ylim(-np.pi-np.pi, np.pi+np.pi)
    sameRas_ax.set_title('Same', fontsize=legendsize)
    sameDen_ax.plot(same_den/np.max(same_den), circax, color=same_c, linewidth=lw)
    sameDen_ax.plot(same_den2/np.max(same_den2), circax, color=same_c, linewidth=lw, linestyle=ls2)
    sameDen_ax.axhline(same_mean, xmin=mean_xmin, xmax=1.5, color='k', linewidth=lw)
    sameDen_ax.axhline(same_mean2, xmin=mean_xmin, xmax=1.5, color='k', linewidth=lw, linestyle=ls2)

    oppRas_ax.axvspan(-0.05, 0.5, color=shade_c, zorder=0)
    oppRas_ax.scatter(data['opp']['lead']['pos'], data['opp']['lead']['phase'], c=lead_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['opp']['lead']['pos'], data['opp']['lead']['phase']-2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['opp']['lead']['pos'], data['opp']['lead']['phase']+2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['opp']['slave']['pos'], data['opp']['slave']['phase'], c=slave_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['opp']['slave']['pos'], data['opp']['slave']['phase']-2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['opp']['slave']['pos'], data['opp']['slave']['phase']+2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data2['opp']['slave']['pos'], data2['opp']['slave']['phase'], c=slave_c2, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data2['opp']['slave']['pos'], data2['opp']['slave']['phase']-2*np.pi, c=slave_c2, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data2['opp']['slave']['pos'], data2['opp']['slave']['phase']+2*np.pi, c=slave_c2, marker=ras_marker, s=ms)
    oppRas_ax.set_ylim(-np.pi-np.pi, np.pi+np.pi)
    oppRas_ax.set_title('Opp', fontsize=legendsize)
    oppDen_ax.plot(opp_den/np.max(opp_den), circax, color=opp_c, linewidth=lw)
    oppDen_ax.plot(opp_den2/np.max(opp_den2), circax, color=opp_c, linewidth=lw, linestyle=ls2)
    oppDen_ax.axhline(opp_mean, xmin=mean_xmin, color='k', linewidth=lw)
    oppDen_ax.axhline(opp_mean2, xmin=mean_xmin, color='k', linewidth=lw, linestyle=ls2)

    overlap_c = list(map(lambda x : lead_c if x==1 else slave_c, data['overlap']['leadflag']))
    overlap_c2 = list(map(lambda x : lead_c if x==1 else slave_c2, data2['overlap']['leadflag']))
    overlapRas_ax.axvspan(-0.05, 1.05, color=shade_c, zorder=0)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] , c=overlap_c, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] +2*np.pi, c=overlap_c, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] -2*np.pi, c=overlap_c, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data2['overlap']['pos'], data2['overlap']['phase'] , c=overlap_c2, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data2['overlap']['pos'], data2['overlap']['phase'] +2*np.pi, c=overlap_c2, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data2['overlap']['pos'], data2['overlap']['phase'] -2*np.pi, c=overlap_c2, marker=ras_marker, s=ms)
    overlapRas_ax.set_title('Overlap', fontsize=legendsize)
    overlapDen_ax.plot(overlap_den/np.max(overlap_den), circax, color='gray', linewidth=lw)
    overlapDen_ax.plot(overlap_den2/np.max(overlap_den2), circax, color='gray', linewidth=lw, linestyle=ls2)
    overlapDen_ax.axhline(overlap_mean, xmin=mean_xmin, color='k', linewidth=lw)
    overlapDen_ax.axhline(overlap_mean2, xmin=mean_xmin, color='k', linewidth=lw, linestyle=ls2)

    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase'], c=lead_c, marker=ras_marker, label='nonoverlap', s=ms)
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase']+2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase']-2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    nonoverlapRas_ax.set_title('Nonoverlap', fontsize=legendsize)
    nonoverlapDen_ax.plot(nonoverlap_den/np.max(nonoverlap_den), circax, color='gray', linewidth=lw)
    nonoverlapDen_ax.axhline(nonoverlap_mean, xmin=mean_xmin, color='k', linewidth=lw)

    for i, ax_each in enumerate([sameRas_ax, oppRas_ax, overlapRas_ax, nonoverlapRas_ax]):
        ax_each.axhline(-np.pi, color='k', linewidth=lw)
        ax_each.axhline(np.pi, color='k', linewidth=lw)
        ax_each.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        ax_each.set_yticklabels(['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$'])
        ax_each.set_yticks([-2*np.pi, -1.5*np.pi , -np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], minor=True)
        ax_each.set_ylim(-2*np.pi, 2*np.pi)
        ax_each.set_xlim(-0.05, 1.05)
        ax_each.set_xticks([0, 1])
        ax_each.set_xticks([0.5], minor=True)
        ax_each.tick_params(labelsize=ticksize)

        if i > 0:
            plt.setp(ax_each.get_yticklabels(), visible=False)

    for ax_each in [sameDen_ax, oppDen_ax, overlapDen_ax, nonoverlapDen_ax]:
        ax_each.set_ylim(-2*np.pi, 2*np.pi)
        ax_each.set_xlim(-0.05, 1.1)
        ax_each.axis('off')

    fig.savefig(join(save_dir, 'intrisic_model_illu.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'intrisic_model_illu.eps'), dpi=dpi)

    #  ========================= Plot Extrinsic precession schematics =========================
    print("Plotting Extrinsic precession schematics")
    total_figw_max = 4.16
    fig = plt.figure(figsize=(total_figw_max, 1.75), facecolor='w')
    y_squeeze = 0.3
    x_squeeze = 0.04
    x_squeeze_Den = 0.005
    x_shift_Den = 0
    cul_xshift = 0.0175
    whole_xshift = 0.007
    all_y, all_h = y_squeeze / 2, 1 - y_squeeze
    Den_w = 0.05
    Ras_w = 1/4-0.05

    sameRas_x = 0
    sameRas_ax = fig.add_axes([sameRas_x + x_squeeze / 2 + cul_xshift*4 - whole_xshift,
                               all_y, Ras_w - x_squeeze, all_h])
    sameDen_x = Ras_w
    sameDen_ax = fig.add_axes([sameDen_x + x_squeeze_Den / 2 + cul_xshift*3 + x_shift_Den - whole_xshift,
                               all_y, Den_w - x_squeeze_Den, all_h])

    oppRas_x = sameDen_x + Den_w
    oppRas_ax = fig.add_axes([oppRas_x + x_squeeze / 2 + cul_xshift*3 - whole_xshift,
                              all_y, Ras_w - x_squeeze, all_h])
    oppDen_x = oppRas_x + Ras_w
    oppDen_ax = fig.add_axes([oppDen_x + x_squeeze_Den / 2 + cul_xshift*2 + x_shift_Den - whole_xshift,
                              all_y, Den_w - x_squeeze_Den, all_h])

    overlapRas_x = oppDen_x + Den_w
    overlapRas_ax = fig.add_axes([overlapRas_x + x_squeeze / 2 + cul_xshift*2 - whole_xshift,
                                  all_y, Ras_w - x_squeeze, all_h])
    overlapDen_x = overlapRas_x + Ras_w
    overlapDen_ax = fig.add_axes([overlapDen_x + x_squeeze_Den / 2 + cul_xshift*1 + x_shift_Den - whole_xshift,
                                  all_y, Den_w - x_squeeze_Den, all_h])

    nonoverlapRas_x = overlapDen_x + Den_w
    nonoverlapRas_ax = fig.add_axes([nonoverlapRas_x + x_squeeze / 2 + cul_xshift*1 - whole_xshift,
                                     all_y, Ras_w - x_squeeze, all_h])
    nonoverlapDen_x = nonoverlapRas_x + Ras_w
    nonoverlapDen_ax = fig.add_axes([nonoverlapDen_x + x_squeeze_Den / 2 + cul_xshift*0 + x_shift_Den - whole_xshift,
                                     all_y, Den_w - x_squeeze_Den, all_h])

    lead_c, slave_c, slave_c2 = 'turquoise', 'purple', 'violet'
    same_c, opp_c ='rosybrown', 'lightsteelblue'
    ras_marker = '.'
    shade_c = '0.9'
    lw = 1
    ms = 4
    mean_xmin = 0
    base_onset = 3
    slope = 1.4*np.pi
    lead_slave_phasediff1 = 0.4*np.pi
    data = gen_extrinsic_model_data(base_onset=base_onset, slope=slope,
                                    lead_slave_phasediff=lead_slave_phasediff1, lead_slave_posdiff=0.075,
                                    num_spikes=12, pos_bound=(0, 0.925))

    # # Calculate statistics
    overlap_mean = circmean(data['overlap']['phase'] )
    nonoverlap_mean = circmean(data['nonoverlap']['phase'])
    normal_mean = circmean(data['normal']['phase'])

    bins, kappa = np.linspace(-3*np.pi, 3*np.pi, 30), 2.5
    circax, overlap_den = circular_density_1d(data['overlap']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, nonoverlap_den = circular_density_1d(data['nonoverlap']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))
    _, normal_den = circular_density_1d(data['normal']['phase'], kappa, bins=100, bound=(-3*np.pi, 3*np.pi))


    sameRas_ax.axvspan(0.3, 0.7, color=shade_c, zorder=0)
    sameRas_ax.scatter(data['pos1'], data['phase1'], c=lead_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['pos1'], data['phase1']-2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['pos1'], data['phase1']+2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['pos2'], data['phase2'], c=slave_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['pos2'], data['phase2']-2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    sameRas_ax.scatter(data['pos2'], data['phase2']+2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    sameRas_ax.set_ylim(-np.pi-np.pi, np.pi+np.pi)
    sameRas_ax.set_title('Same', fontsize=legendsize)
    sameDen_ax.plot(normal_den/np.max(normal_den), circax, color=same_c, linewidth=lw)
    sameDen_ax.axhline(normal_mean, xmin=mean_xmin, xmax=1.5, color='k', linewidth=lw)

    oppRas_ax.axvspan(0.3, 0.7, color=shade_c, zorder=0)
    oppRas_ax.scatter(data['pos1'], data['phase1'], c=slave_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['pos1'], data['phase1']-2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['pos1'], data['phase1']+2*np.pi, c=slave_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['pos2'], data['phase2'], c=lead_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['pos2'], data['phase2']-2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    oppRas_ax.scatter(data['pos2'], data['phase2']+2*np.pi, c=lead_c, marker=ras_marker, s=ms)
    oppRas_ax.set_ylim(-np.pi-np.pi, np.pi+np.pi)
    oppRas_ax.set_title('Opp', fontsize=legendsize)
    oppDen_ax.plot(normal_den/np.max(normal_den), circax, color=opp_c, linewidth=lw)
    oppDen_ax.axhline(normal_mean, xmin=mean_xmin, color='k', linewidth=lw)

    overlap_c = list(map(lambda x : lead_c if x==1 else slave_c, data['overlap']['onetag']))
    overlap_cr = list(map(lambda x : lead_c if x==0 else slave_c, data['overlap']['onetag']))
    overlapRas_ax.axvspan(0.3, 0.7, color=shade_c, zorder=0)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] , c=overlap_c, marker=ras_marker, label='overlap', s=ms)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] +2*np.pi, c=overlap_c, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] -2*np.pi, c=overlap_c, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] , c=overlap_cr, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] +2*np.pi, c=overlap_cr, marker=ras_marker, s=ms)
    overlapRas_ax.scatter(data['overlap']['pos'], data['overlap']['phase'] -2*np.pi, c=overlap_cr, marker=ras_marker, s=ms)
    overlapRas_ax.set_title('Overlap', fontsize=legendsize)
    overlapDen_ax.plot(overlap_den/np.max(overlap_den), circax, color='gray', linewidth=lw)
    overlapDen_ax.axhline(overlap_mean, xmin=mean_xmin, color='k', linewidth=lw)
    nonoverlap_c = list(map(lambda x : lead_c if x==1 else slave_c, data['overlap']['onetag']))
    nonoverlap_cr = list(map(lambda x : lead_c if x==0 else slave_c, data['overlap']['onetag']))
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase'], c=nonoverlap_c, marker=ras_marker, label='nonoverlap', s=ms)
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase']+2*np.pi, c=nonoverlap_c, marker=ras_marker, s=ms)
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase']-2*np.pi, c=nonoverlap_c, marker=ras_marker, s=ms)
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase'], c=nonoverlap_cr, marker=ras_marker, s=ms)
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase']+2*np.pi, c=nonoverlap_cr, marker=ras_marker, s=ms)
    nonoverlapRas_ax.scatter(data['nonoverlap']['pos'], data['nonoverlap']['phase']-2*np.pi, c=nonoverlap_cr, marker=ras_marker, s=ms)
    nonoverlapRas_ax.set_title('Nonoverlap', fontsize=legendsize)
    nonoverlapDen_ax.plot(nonoverlap_den/np.max(nonoverlap_den), circax, color='gray', linewidth=lw)
    nonoverlapDen_ax.axhline(nonoverlap_mean, xmin=mean_xmin, color='k', linewidth=lw)

    for i, ax_each in enumerate([sameRas_ax, oppRas_ax, overlapRas_ax, nonoverlapRas_ax]):
        ax_each.axhline(-np.pi, color='k', linewidth=lw)
        ax_each.axhline(np.pi, color='k', linewidth=lw)
        ax_each.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
        ax_each.set_yticklabels(['$-2\pi$', '$-\pi$', '0', '$\pi$', '$2\pi$'])
        ax_each.set_yticks([-2*np.pi, -1.5*np.pi , -np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], minor=True)
        ax_each.set_ylim(-2*np.pi, 2*np.pi)
        ax_each.set_xlim(-0.05, 1.05)
        ax_each.set_xticks([0, 1])
        ax_each.set_xticks([0.5], minor=True)
        ax_each.tick_params(labelsize=ticksize)

        if i > 0:
            plt.setp(ax_each.get_yticklabels(), visible=False)

    for ax_each in [sameDen_ax, oppDen_ax, overlapDen_ax, nonoverlapDen_ax]:
        ax_each.axis('off')
        ax_each.set_ylim(-2*np.pi, 2*np.pi)
        ax_each.set_xlim(-0.05, 1.1)
    fig.savefig(join(save_dir, 'extrisic_model_illu.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'extrisic_model_illu.eps'), dpi=dpi)

    #  ========================= Plot Overlap vs Nonoverlap phase distribution =========================
    overlapdf_dict = dict(ca=[], inside=[], segmentid=[], overlap_ratio=[],
                          tsp1=[], tsp2=[], dsp1=[], dsp2=[], phasesp1=[], phasesp2=[],
                          nsp1=[], nsp2=[], rcc_m1=[], rcc_c1=[], rcc_m2=[], rcc_c2=[]
                          )

    for sid in range(expdf.shape[0]):
        print('\rOverlap vs Nonoverlap Correlogram %d/%d' % (sid, expdf.shape[0]), end='', flush=True)

        # Get data
        ca, pair_id, overlap, overlap_ratio = expdf.loc[sid, ['ca', 'pair_id', 'overlap', 'overlap_ratio']]
        precess_dfp = expdf.loc[sid, 'precess_dfp']

        # Plot pass examples in the ex-in pair
        for nprecess in range(precess_dfp.shape[0]):
            x, y, t, v = precess_dfp.loc[nprecess, ['x', 'y', 't', 'v']]
            dsp1, phasesp1, tsptheta1 = precess_dfp.loc[nprecess, ['dsp1', 'phasesp1', 'tsp_withtheta1']]
            dsp2, phasesp2, tsptheta2 = precess_dfp.loc[nprecess, ['dsp2', 'phasesp2', 'tsp_withtheta2']]
            wavet1, phase1, theta1 = precess_dfp.loc[nprecess, ['wave_t1', 'wave_phase1', 'wave_theta1']]
            infield1, infield2 = precess_dfp.loc[nprecess, ['infield1', 'infield2']]

            # Determine lone spikes
            tsp1_alone, tsp2_alone = np.zeros(tsptheta1.shape[0], dtype=bool), np.zeros(tsptheta2.shape[0], dtype=bool)
            cycidx_tmp = np.where(np.diff(phase1) < -6)[0]
            cycidx = np.concatenate([np.array([0]), cycidx_tmp, np.array([phase1.shape[0] - 1])])
            for cyci in range(cycidx.shape[0] - 1):
                tcycstart, tcycend = wavet1[cycidx[cyci]], wavet1[cycidx[cyci + 1]]
                tsp1_incycmask = (tsptheta1 > tcycstart) & (tsptheta1 < tcycend)
                tsp2_incycmask = (tsptheta2 > tcycstart) & (tsptheta2 < tcycend)
                nsp1_incyc, nsp2_incyc = np.sum(tsp1_incycmask), np.sum(tsp2_incycmask)

                if (nsp1_incyc == 0) & (nsp2_incyc == 0):
                    continue
                elif (nsp1_incyc > 0) & (nsp2_incyc == 0):  # tsp1 is alone
                    tsp1_alone[tsp1_incycmask] = True
                elif (nsp1_incyc == 0) & (nsp2_incyc > 0):  # tsp2 is alone
                    tsp2_alone[tsp2_incycmask] = True
                elif (nsp1_incyc > 0) & (nsp2_incyc > 0):  # Both tsp1 and tsp2 are present
                    tsp1_alone[tsp1_incycmask] = False
                    tsp2_alone[tsp2_incycmask] = False
                else:  # Unclassified cases
                    raise RuntimeError

            in_overlap = infield1 & infield2
            out_overlap = ~in_overlap

            all_inoverlap_idx = segment_passes(in_overlap)
            all_outoverlap_idx = segment_passes(out_overlap)
            inout_dict = dict(inside=all_inoverlap_idx, outside=all_outoverlap_idx)

            for inouttype, inoutidx in inout_dict.items():
                for segmentid, (startidx_in, endidx_in) in enumerate(inoutidx):
                    t_start_in, t_end_in = t[startidx_in], t[endidx_in]
                    inmask1 = (tsptheta1 > t_start_in) & (tsptheta1 < t_end_in)
                    inmask2 = (tsptheta2 > t_start_in) & (tsptheta2 < t_end_in)
                    tsp1_in, tsp2_in = tsptheta1[inmask1], tsptheta2[inmask2]
                    phasesp1_in, phasesp2_in = phasesp1[inmask1], phasesp2[inmask2]
                    dsp1_in, dsp2_in = dsp1[inmask1], dsp2[inmask2]

                    if (dsp1_in.shape[0] < 1) and (dsp2_in.shape[0] < 1):
                        continue

                    # Fit regression - all, lone and pair spikes
                    rcc_m1_in, rcc_c1_in, nsp1_in, rcc_m2_in, rcc_c2_in, nsp2_in = rcc_wrapper(dsp1_in, phasesp1_in, dsp2_in, phasesp2_in)

                    if inouttype == 'inside':
                        segmentid = 99
                        inside_flag = True
                    elif inouttype == 'outside':
                        inside_flag = False
                    else:
                        raise RuntimeError

                    overlapdf_dict['ca'].append(ca)
                    overlapdf_dict['inside'].append(inside_flag)
                    overlapdf_dict['segmentid'].append(segmentid)
                    overlapdf_dict['overlap_ratio'].append(overlap_ratio)

                    # all spikes
                    overlapdf_dict['tsp1'].append(tsp1_in)
                    overlapdf_dict['tsp2'].append(tsp2_in)
                    overlapdf_dict['dsp1'].append(dsp1_in)
                    overlapdf_dict['dsp2'].append(dsp2_in)
                    overlapdf_dict['phasesp1'].append(phasesp1_in)
                    overlapdf_dict['phasesp2'].append(phasesp2_in)
                    overlapdf_dict['nsp1'].append(nsp1_in)
                    overlapdf_dict['nsp2'].append(nsp2_in)
                    overlapdf_dict['rcc_m1'].append(rcc_m1_in)
                    overlapdf_dict['rcc_c1'].append(rcc_c1_in)
                    overlapdf_dict['rcc_m2'].append(rcc_m2_in)
                    overlapdf_dict['rcc_c2'].append(rcc_c2_in)

    print()
    overlapdf = pd.DataFrame(overlapdf_dict)

    # # Compare overlap and nonoverlap for extrinsic and intrinsic pairs
    exintags = ['IntrinsicPairs', 'ExtrinsicPairs']
    inside_flags = [False, True]
    margin = 1.8
    lw = 0.75
    lw_mean = 1
    color_dict = dict(CA1=dict(overlap='lightsteelblue', nonoverlap='mediumblue'),
                      CA2=dict(overlap='palegreen', nonoverlap='darkgreen'),
                      CA3=dict(overlap='thistle', nonoverlap='darkmagenta'))


    fig, ax = plt.subplots(2, 3, figsize=(total_figw, 2.8), dpi=200, facecolor='w', sharex=True, sharey=True)

    for exin_id, exintag in enumerate(exintags):
        for caid, (ca, cadf) in enumerate(overlapdf.groupby('ca')):
            if exintag == 'ExtrinsicPairs':
                cadf = cadf[cadf['overlap_ratio'] > 0].reset_index(drop=True)
            elif exintag =='IntrinsicPairs':
                cadf = cadf[cadf['overlap_ratio'] <= 0].reset_index(drop=True)
            data_dict = dict()

            for inside_id, inside_flag in enumerate(inside_flags):
                insidekey = 'overlap' if inside_flag else 'nonoverlap'
                data_dict[insidekey] = dict()

                # Filtering
                dftmp = cadf[cadf.inside == inside_flag].reset_index(drop=True)
                dfmask1 = (dftmp['nsp1'] > 1) & (dftmp['rcc_m1'] < margin) & (dftmp['rcc_m1'] > -margin)
                dfmask2 = (dftmp['nsp2'] > 1) & (dftmp['rcc_m2'] < margin) & (dftmp['rcc_m2'] > -margin)

                dftmp1 = dftmp[dfmask1].reset_index(drop=True)
                dftmp2 = dftmp[dfmask2].reset_index(drop=True)

                # Get quantities
                precess_mask1 = dftmp1['rcc_m1'] < 0
                precess_mask2 = dftmp2['rcc_m2'] < 0
                precessphasesp = np.concatenate(dftmp1[precess_mask1]['phasesp1'].to_list() + dftmp2[precess_mask2]['phasesp2'].to_list())

                # Store for comparison
                data_dict[insidekey]['phasesp'] = precessphasesp

                # Plot (precessing phase)
                label = 'Overlap' if insidekey=='overlap' else 'Nonoverlap'
                bins, edges = np.histogram(data_dict[insidekey]['phasesp'], range=(-np.pi, np.pi), bins=36)
                ax[exin_id, caid].step(edges[:-1], bins/np.sum(bins)*10, where='pre', color=color_dict[ca][insidekey], label='%s'%(label),
                                       linewidth=lw)

            # Test between spikephases overlap and nonoverlap
            tag1, tag2 = 'Overlap', 'Nonoverlap'
            p_phasesp, _, descrips_phasesp, wwtxt = my_ww_2samp(data_dict['overlap']['phasesp'], data_dict['nonoverlap']['phasesp'], tag1, tag2)
            ((phasesp_overlap_mu, phasesp_overlap_sem), (phasesp_nonoverlap_mu, phasesp_nonoverlap_sem)) = descrips_phasesp
            ax[exin_id, caid].annotate(text='p=%s'%(p2str(p_phasesp)), xy=(0.4, 0.05), xycoords='axes fraction')
            ax[exin_id, caid].axvline(phasesp_overlap_mu, ymin=0.75, ymax=0.85, color=color_dict[ca]['overlap'], linewidth=lw_mean)
            ax[exin_id, caid].axvline(phasesp_nonoverlap_mu, ymin=0.75, ymax=0.85, color=color_dict[ca]['nonoverlap'], linewidth=lw_mean)
            customlegend(ax[0, caid], loc='upper left', fontsize=legendsize)
            ax[exin_id, caid].spines['top'].set_visible(False)
            ax[exin_id, caid].spines['right'].set_visible(False)
            ax[0, caid].set_title(ca, fontsize=titlesize)
            ax[1, caid].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax[1, caid].set_xticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
            ax[exin_id, caid].set_ylim(0, 0.7)
            ax[exin_id, caid].set_yticks([0, 0.2, 0.4, 0.6])
            ax[exin_id, caid].set_yticks(np.arange(0, 0.7, 0.1), minor=True)
            ax[exin_id, caid].tick_params(labelsize=ticksize)

            # Export statistics
            stat_txt = '='*10 + '%s %s'%(ca, exintag) + '='*10 + '\n'
            stat_txt += r'%s: mean$\pm$SEM = $%0.2f\pm%0.2f$\n'%(tag1, phasesp_overlap_mu, phasesp_overlap_sem) + '\n'
            stat_txt += r'%s: mean$\pm$SEM = $%0.2f\pm%0.2f$\n'%(tag2, phasesp_nonoverlap_mu, phasesp_nonoverlap_sem) + '\n'
            stat_txt += wwtxt
            stat_record(stat_fn, False, stat_txt)
            print(stat_txt)
    fig.savefig(join(save_dir, 'overlap_nonoverlap_phase.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'overlap_nonoverlap_phase.eps'), dpi=dpi)

def BothParallelOppositeAnalysis(expdf, save_dir):
    import warnings
    warnings.filterwarnings("ignore")
    def organize_4case_dfs(expdf):

        frac_field = 2
        frac_pass = 2

        df = expdf[(~expdf['rate_angle1'].isna()) & (~expdf['rate_angle2'].isna()) & \
                   (~expdf['overlap_ratio'].isna())
                   ].reset_index(drop=True)

        allpdf_dict = dict(ca=[], pair_id=[], field_orient=[], overlap_ratio=[], fanglep=[], fanglediff=[],
                           meanfangle=[],
                           pass_id=[], passangle=[], precess_exist_tmp=[],
                           fangle1=[], fangle2=[], pairlagAB=[], pairlagBA=[],
                           nspikes1=[], nspikes2=[], onset1=[], onset2=[], slope1=[], slope2=[],
                           phasesp1=[], phasesp2=[], dsp1=[], dsp2=[], direction=[]
                           )

        for i in range(df.shape[0]):

            ca, pair_id, overlap_r = df.loc[i, ['ca', 'pair_id', 'overlap_ratio']]
            fangle1, fangle2, fanglep = df.loc[i, ['rate_angle1', 'rate_angle2', 'rate_anglep']]
            lagAB, lagBA = df.loc[i, ['phaselag_AB', 'phaselag_BA']]
            fanglediff = np.abs(cdiff(fangle1, fangle2))
            meanfangle = shiftcyc_full2half(circmean([fangle1, fangle2]))

            # Orientation flipping
            pos1, pos2 = df.loc[i, ['com1', 'com2']]
            posdiff = pos2 - pos1
            field_orient = np.angle(posdiff[0] + 1j * posdiff[1])
            absdiff = np.abs(cdiff(field_orient, meanfangle))
            ABalign = True if absdiff < np.pi / 2 else False

            pass_dfp = df.loc[i, 'precess_dfp']
            num_pass = pass_dfp.shape[0]
            for npass in range(num_pass):
                precess_exist, onset1, onset2, slope1, slope2 = pass_dfp.loc[
                    npass, ['precess_exist', 'rcc_c1', 'rcc_c2', 'rcc_m1', 'rcc_m2']]
                phasesp1, phasesp2, mean_anglesp1, mean_anglesp2 = pass_dfp.loc[
                    npass, ['phasesp1', 'phasesp2', 'mean_anglesp1', 'mean_anglesp2']]
                direction, dsp1, dsp2 = pass_dfp.loc[npass, ['direction', 'dsp1', 'dsp2']]
                nsp1, nsp2 = phasesp1.shape[0], phasesp2.shape[0]
                precess_exist1, precess_exist2 = pass_dfp.loc[npass, ['precess_exist1', 'precess_exist2']]
                passangle = circmean([mean_anglesp1, mean_anglesp2])

                allpdf_dict['ca'].append(ca)
                allpdf_dict['pair_id'].append(pair_id)
                allpdf_dict['field_orient'].append(field_orient)
                allpdf_dict['overlap_ratio'].append(overlap_r)
                allpdf_dict['fanglep'].append(fanglep)
                allpdf_dict['fanglediff'].append(fanglediff)
                allpdf_dict['meanfangle'].append(meanfangle)
                allpdf_dict['pass_id'].append(npass)
                allpdf_dict['passangle'].append(passangle)

                allpdf_dict['precess_exist_tmp'].append((precess_exist1, precess_exist2))
                allpdf_dict['fangle1'].append(fangle1)
                allpdf_dict['fangle2'].append(fangle2)
                allpdf_dict['pairlagAB'].append(lagAB)
                allpdf_dict['pairlagBA'].append(lagBA)
                allpdf_dict['nspikes1'].append(nsp1)
                allpdf_dict['nspikes2'].append(nsp2)
                allpdf_dict['onset1'].append(onset1)
                allpdf_dict['onset2'].append(onset2)
                allpdf_dict['slope1'].append(slope1)
                allpdf_dict['slope2'].append(slope2)
                allpdf_dict['phasesp1'].append(phasesp1)
                allpdf_dict['phasesp2'].append(phasesp2)
                allpdf_dict['dsp1'].append(dsp1)
                allpdf_dict['dsp2'].append(dsp2)
                allpdf_dict['direction'].append(direction)

        allpdf_sep = pd.DataFrame(allpdf_dict)

        # combined 1 & 2
        stay_keys = ['ca', 'pair_id', 'field_orient', 'overlap_ratio', 'fanglep', 'fanglediff', 'meanfangle', 'pass_id',
                     'passangle', 'precess_exist_tmp', 'fangle1', 'fangle2', 'pairlagAB', 'pairlagBA', 'direction']
        change_keys = ['nspikes', 'onset', 'slope', 'phasesp', 'dsp']

        allpdf1 = allpdf_sep[stay_keys + [key + '1' for key in change_keys]]
        allpdf2 = allpdf_sep[stay_keys + [key + '2' for key in change_keys]]
        allpdf1.rename(columns={key + '1': key.replace('1', '') for key in change_keys}, inplace=True)
        allpdf2.rename(columns={key + '2': key.replace('2', '') for key in change_keys}, inplace=True)
        allpdf = pd.concat([allpdf1, allpdf2], axis=0, ignore_index=True)
        allpdf['precess_exist'] = allpdf['precess_exist_tmp'].apply(lambda x: x[0] & x[1])
        allpdf = allpdf[allpdf['precess_exist']].reset_index(drop=True)

        allpdf = allpdf[allpdf['slope'].abs() < 1.9].reset_index(drop=True)
        inAB_mask = (allpdf['overlap_ratio'] < 0) & (allpdf['pairlagAB'] > 0) & (allpdf['pairlagBA'] > 0)
        inBA_mask = (allpdf['overlap_ratio'] < 0) & (allpdf['pairlagAB'] < 0) & (allpdf['pairlagBA'] < 0)
        preferAB_mask = inAB_mask & (allpdf['direction'] == 'A->B')
        nonpreferAB_mask = inBA_mask & (allpdf['direction'] == 'A->B')
        preferBA_mask = inBA_mask & (allpdf['direction'] == 'B->A')
        nonpreferBA_mask = inAB_mask & (allpdf['direction'] == 'B->A')
        allpdf.loc[preferAB_mask | preferBA_mask, 'bias'] = 'same'
        allpdf.loc[nonpreferAB_mask | nonpreferBA_mask, 'bias'] = 'opp'
        allpdf.loc[inAB_mask, 'bias_direction'] = 'A->B'
        allpdf.loc[inBA_mask, 'bias_direction'] = 'B->A'

        # Case 1 & 2
        allpdf['simfield'] = allpdf['fanglediff'] < (np.pi / frac_field)
        allpdf['oppfield'] = allpdf['fanglediff'] > (np.pi - np.pi / frac_field)
        passfield_adiff = np.abs(cdiff(allpdf['passangle'], allpdf['meanfangle']))
        passfield_adiff1 = np.abs(cdiff(allpdf['passangle'], allpdf['fangle1']))
        passfield_adiff2 = np.abs(cdiff(allpdf['passangle'], allpdf['fangle2']))
        passfield_adiffp = np.abs(cdiff(allpdf['passangle'], allpdf['fanglep']))
        allpdf['case1'] = (passfield_adiff1 < (np.pi / frac_pass)) & (passfield_adiff2 < (np.pi / frac_pass)) & allpdf[
            'simfield']
        allpdf['case2'] = (passfield_adiff1 > (np.pi - np.pi / frac_pass)) & (
                    passfield_adiff2 > (np.pi - np.pi / frac_pass)) & allpdf['simfield']
        #     allpdf['case1'] = (passfield_adiff1 < (np.pi/frac_pass)) & (passfield_adiff2 < (np.pi/frac_pass))
        #     allpdf['case2'] = (passfield_adiff1 > (np.pi - np.pi/frac_pass)) & (passfield_adiff2 > (np.pi - np.pi/frac_pass))

        # Case 3 & 4
        diff1tmp = np.abs(cdiff(allpdf['passangle'], allpdf['fangle1']))
        diff2tmp = np.abs(cdiff(allpdf['passangle'], allpdf['fangle2']))
        simto1_AtoB = (diff1tmp < diff2tmp) & (diff1tmp < (np.pi / frac_pass)) & (allpdf['direction'] == 'A->B') & \
                      allpdf['oppfield']
        simto1_BtoA = (diff1tmp >= diff2tmp) & (diff2tmp < (np.pi / frac_pass)) & (allpdf['direction'] == 'B->A') & \
                      allpdf['oppfield']
        simto2_AtoB = (diff1tmp >= diff2tmp) & (diff2tmp < (np.pi / frac_pass)) & (allpdf['direction'] == 'A->B') & \
                      allpdf['oppfield']
        simto2_BtoA = (diff1tmp < diff2tmp) & (diff1tmp < (np.pi / frac_pass)) & (allpdf['direction'] == 'B->A') & \
                      allpdf['oppfield']
        allpdf['case3'] = simto1_AtoB | simto1_BtoA
        allpdf['case4'] = simto2_AtoB | simto2_BtoA

        # exin df
        df_exin_dict = dict(nex=[], nin=[], ca=[], case=[])

        for caid, (ca, capdf) in enumerate(allpdf.groupby('ca')):
            # Onset per CA
            case1df = capdf[capdf['case1']].reset_index(drop=True)
            case2df = capdf[capdf['case2']].reset_index(drop=True)
            case3df = capdf[capdf['case3']].reset_index(drop=True)
            case4df = capdf[capdf['case4']].reset_index(drop=True)

            # Ex-/intrinsic numbers per CA
            case1udf = case1df.drop_duplicates(subset=['pair_id'])
            case2udf = case2df.drop_duplicates(subset=['pair_id'])
            case3udf = case3df.drop_duplicates(subset=['pair_id'])
            case4udf = case4df.drop_duplicates(subset=['pair_id'])

            nex1, nin1 = (case1udf['overlap_ratio'] > 0).sum(), (case1udf['overlap_ratio'] <= 0).sum()
            nex2, nin2 = (case2udf['overlap_ratio'] > 0).sum(), (case2udf['overlap_ratio'] <= 0).sum()
            nex3, nin3 = (case3udf['overlap_ratio'] > 0).sum(), (case3udf['overlap_ratio'] <= 0).sum()
            nex4, nin4 = (case4udf['overlap_ratio'] > 0).sum(), (case4udf['overlap_ratio'] <= 0).sum()
            df_exin_dict['ca'].extend([ca] * 4)
            df_exin_dict['case'].extend([1, 2, 3, 4])
            df_exin_dict['nex'].extend(([nex1, nex2, nex3, nex4]))
            df_exin_dict['nin'].extend(([nin1, nin2, nin3, nin4]))
        df_exin = pd.DataFrame(df_exin_dict)
        df_exin.index = df_exin.case
        getcadf = lambda x, ca: x[x['ca'] == ca][['nex', 'nin']]
        df_total = getcadf(df_exin, 'CA1') + getcadf(df_exin, 'CA2') + getcadf(df_exin, 'CA3')
        df_total['case'] = df_total.index
        df_total['ca'] = 'ALL'
        df_exinall = pd.concat([df_exin, df_total], axis=0)
        df_exinall['ntotal'] = df_exinall['nex'] + df_exinall['nin']
        df_exinall['exfrac'] = df_exinall['nex'] / df_exinall['ntotal']
        df_exinall['exratio'] = df_exinall['nex'] / df_exinall['nin']
        return allpdf, (df_exin, df_total, df_exinall)

    allpdf, (df_exin, df_total, df_exinall) = organize_4case_dfs(expdf)

    # # # ================================ Plotting & Statistical tesing =============================================
    # # Stats
    stat_fn = 'fig9_BothParallelOppositeAnalysis.txt'
    stat_record(stat_fn, True)

    # # Figure condifuration
    onehalfcol_figw = 4.33  # 1.5 column, =11cm < 11.6cm (required)
    fig = plt.figure(figsize=(onehalfcol_figw, onehalfcol_figw*0.9))
    btwCA_xsqu = 0.025
    x_offset = 0.0675

    # Ex-In pair number bar charts (ax_pN)
    ax_pN_y = 1 - 2/3
    ax_pN_w, ax_pN_h = 1/3, 1/3
    xsqueeze_pN, ysqueeze_pN = 0.1, 0.1
    ax_pN = np.array([
        fig.add_axes([ax_pN_w * 0 + xsqueeze_pN / 2 + x_offset + btwCA_xsqu, ax_pN_y + ysqueeze_pN, ax_pN_w - xsqueeze_pN, ax_pN_h - ysqueeze_pN]),
        fig.add_axes([ax_pN_w * 1 + xsqueeze_pN / 2 + x_offset, ax_pN_y + ysqueeze_pN, ax_pN_w - xsqueeze_pN, ax_pN_h - ysqueeze_pN]),
        fig.add_axes([ax_pN_w * 2 + xsqueeze_pN / 2 + x_offset - btwCA_xsqu, ax_pN_y + ysqueeze_pN, ax_pN_w - xsqueeze_pN, ax_pN_h - ysqueeze_pN])
    ])


    # Phasesp
    ax_phase_y = 1 - 3/3
    ax_phase_h, ax_phase_w = 1/3, 1/3
    xsqueeze_phase, ysqueeze_phase = 0.1, 0.125
    ax_phase = np.array([
        fig.add_axes([ax_phase_w * 0 + xsqueeze_phase / 2 + x_offset + btwCA_xsqu, ax_phase_y + ysqueeze_phase, ax_phase_w - xsqueeze_phase, ax_phase_h - ysqueeze_phase]),
        fig.add_axes([ax_phase_w * 1 + xsqueeze_phase / 2 + x_offset, ax_phase_y + ysqueeze_phase, ax_phase_w - xsqueeze_phase, ax_phase_h - ysqueeze_phase]),
        fig.add_axes([ax_phase_w * 2 + xsqueeze_phase / 2 + x_offset - btwCA_xsqu, ax_phase_y + ysqueeze_phase, ax_phase_w - xsqueeze_phase, ax_phase_h - ysqueeze_phase])
    ])

    ax = np.stack([ax_pN, ax_phase])

    case1_c, case2_c = 'darkblue', 'lime'


    # # Ex-in pair numbers
    case_x = np.array([-1, 1])
    exin_box = np.array([-0.4, 0.4])
    boxw = 0.5
    boxlim = (-2, 2)
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
        exin_cadf = df_exinall[df_exinall['ca']==ca]
        exfrac1 = exin_cadf.loc[exin_cadf['case']==1, 'exratio'].iloc[0]
        exfrac2 = exin_cadf.loc[exin_cadf['case']==2, 'exratio'].iloc[0]
        dftmp12 = exin_cadf[(exin_cadf['case'] == 1) | (exin_cadf['case'] == 2)][['nex', 'nin']]
        nexc1, ninc1 = dftmp12.loc[1, ['nex', 'nin']]
        nexc2, ninc2 = dftmp12.loc[2, ['nex', 'nin']]
        max_numpairs = np.max([nexc1, ninc1, nexc2, ninc2])
        pval12, _, pval12txt = my_fisher_2way(dftmp12.to_numpy())
        case1_ex_x, case1_in_x = case_x[0] + exin_box[0], case_x[0] + exin_box[1]
        case2_ex_x, case2_in_x = case_x[1] + exin_box[0], case_x[1] + exin_box[1]

        ax_pN[caid].bar(case1_ex_x, nexc1, width=boxw, color=case1_c)
        ax_pN[caid].bar(case1_in_x, ninc1, width=boxw, color=case1_c)
        ax_pN[caid].bar(case2_ex_x, nexc2, width=boxw, color=case2_c)
        ax_pN[caid].bar(case2_in_x, ninc2, width=boxw, color=case2_c)
        ax_pN[caid].annotate(text='p=%s'%(p2str(pval12)), xy=(0.2, 0.9), xycoords='axes fraction', fontsize=legendsize)
        ax_pN[caid].set_xticks([case1_ex_x, case1_in_x, case2_ex_x, case2_in_x])
        ax_pN[caid].set_xticklabels(['Ex', 'In'] * 2)
        ax_pN[caid].set_xlim(*boxlim)
        ax_pN[caid].set_ylim(0, max_numpairs*1.5)
        ax_pN[caid].set_title(ca, fontsize=legendsize)

        stat_record(stat_fn, False, "Exin-Frac, %s, Both-parallel(Ex:In=%d:%d=%0.2f) vs Both-opposite(Ex:In=%d:%d=%0.2f), %s" % (
            ca, nexc1, ninc1, nexc1 / ninc1, nexc2, ninc2, nexc2 / ninc2, pval12txt))
    nex_ca1_case2, nin_ca1_case2 = df_exinall[(df_exinall['ca']=='CA1')].loc[2, ['nex', 'nin']]
    nex_ca3_case2, nin_ca3_case2 = df_exinall[(df_exinall['ca']=='CA3')].loc[2, ['nex', 'nin']]
    p_ca31_case2, n_ca31_case2, ptxt_ca3_case2 = my_fisher_2way(np.array([[nex_ca3_case2, nin_ca3_case2], [nex_ca1_case2, nin_ca1_case2]]))
    stat_record(stat_fn, False, 'Exin-frac, Case 2, CA1(Ex:In=%d:%d) vs CA3(Ex:In=%d:%d), p=%s'%(
        nex_ca1_case2, nin_ca1_case2, nex_ca3_case2, nin_ca3_case2, p_ca31_case2))

    phasebins = np.linspace(-np.pi, np.pi, 24)
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
        allp_cadf = allpdf[allpdf['ca']==ca].reset_index(drop=True)
        phasesp1 = np.concatenate(allp_cadf.loc[allp_cadf['case1'], 'phasesp'].to_list())
        phasesp2 = np.concatenate(allp_cadf.loc[allp_cadf['case2'], 'phasesp'].to_list())
        bins1, _ = np.histogram(phasesp1, bins=phasebins)
        bins2, _ = np.histogram(phasesp2, bins=phasebins)
        pval12, _, descrips_phasesp12, pval12txt = my_ww_2samp(phasesp1, phasesp2, 'Case1', 'Case2')
        ((cmean1, sem1), (cmean2, sem2)) = descrips_phasesp12

        ax_phase[caid].step(phasebins[:-1], bins1/bins1.sum(), where='pre', color='darkblue', linewidth=0.75)
        ax_phase[caid].axvline(cmean1, ymin=0.825, ymax=0.95, color='darkblue', linewidth=0.75)
        ax_phase[caid].step(phasebins[:-1], bins2/bins2.sum(), where='pre', color='lime', linewidth=0.75)
        ax_phase[caid].axvline(cmean2, ymin=0.825, ymax=0.95, color='lime', linewidth=0.75)
        ax_phase[caid].annotate(text='p=%s'%(p2str(pval12)), xy=(0.2, 0.05), xycoords='axes fraction', fontsize=legendsize)
        ax_phase[caid].set_xticks([-np.pi, 0, np.pi])
        ax_phase[caid].set_xticks([-np.pi/2, np.pi/2], minor=True)
        ax_phase[caid].set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
        ax_phase[caid].set_yticks([0, 0.1])
        ax_phase[caid].set_yticks([0.05], minor=True)
        ax_phase[caid].set_yticklabels(['0', '0.1'])
        ax_phase[caid].set_ylim(0, 0.1)

        stat_record(stat_fn, False, 'Spike phases, %s, both-parallel vs both-opposite, %s' % (ca, pval12txt))
        stat_record(stat_fn, False, r'both-parallel vs both-opposite: $%0.2f\pm%0.4f$ vs $%0.2f\pm%0.4f$' % (
            cmean1, sem1, cmean2, sem2))


    ax_pN[0].set_yticks(np.arange(0, 81, 20))
    ax_pN[0].set_yticks(np.arange(0, 81, 10), minor=True)
    ax_pN[0].set_ylabel('Number\nof pairs', fontsize=legendsize)
    ax_pN[1].set_yticks(np.arange(0, 61, 20))
    ax_pN[1].set_yticks(np.arange(0, 61, 10), minor=True)
    ax_pN[2].set_yticks(np.arange(0, 17, 5))
    ax_pN[2].set_yticks(np.arange(0, 16, 1), minor=True)
    ax_phase[0].set_ylabel('Normalized\nspike count', fontsize=legendsize)
    ax_phase[1].set_xlabel('Phase (rad)')
    plt.setp(ax_phase[1].get_yticklabels(), visible=False)
    plt.setp(ax_phase[2].get_yticklabels(), visible=False)

    for ax_each in ax.ravel():
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
        ax_each.tick_params(labelsize=ticksize)

    fig.savefig(join(save_dir, 'BothParellelOppositeAnalysis.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'BothParellelOppositeAnalysis.eps'), dpi=dpi)
    return None


def main():
    # # Setting
    data_pth = 'results/emankin/pairfield_df.pickle'
    figure_dir = 'writting/figures/'
    for i in range(4, 10):
        os.makedirs(join(figure_dir, 'fig%d'%(i)), exist_ok=True)

    # # Loading
    expdf = pd.read_pickle(data_pth)

    # # For subpopulation analyses as requested by reviewers
    # expdf['border'] = expdf['border1'] | expdf['border2']
    # expdf = expdf[~expdf['border']].reset_index(drop=True)
    # expdf = expdf[expdf['num_spikes_pair']>100].reset_index(drop=True)

    # # Analysis
    omniplot_pairfields(expdf, save_dir=join(figure_dir, 'fig4'))  # Fig 4
    compare_single_vs_pair_directionality()  # Fig 4, without plotting
    plot_example_correlograms(expdf, save_dir=join(figure_dir, 'fig5'))  # Fig 5
    plot_pair_correlation(expdf, save_dir=join(figure_dir, 'fig5'))  # Fig 5
    plot_example_exin(expdf, save_dir=join(figure_dir, 'fig6'))  # Fig 6 exintrinsic example
    plot_exintrinsic(expdf, save_dir=join(figure_dir, 'fig6'))  # Fig 6 ex-in scatter
    plot_pairangle_similarity_analysis(expdf, save_dir=figure_dir)  # Fig 6
    plot_intrinsic_precession_property(expdf, plot_example=True, save_dir=join(figure_dir, 'fig7'))  # Fig 7 Intrinsic direction
    plot_leading_enslaved_analysis(expdf, save_dir='writting/figures/fig8')  # Fig 8 Leading-Enslaved analysis
    BothParallelOppositeAnalysis(expdf, save_dir=join(figure_dir, 'fig9'))  # Fig 9 Four-cases

    # # Archive
    # plot_ALL_example_correlograms(expdf)
    # plot_ALL_examples_EXIN_correlogram(expdf)  # For exploring all ex-in examples in Fig 6

if __name__ == '__main__':
    main()
