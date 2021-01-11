import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from glob import glob
from matplotlib import cm

from pycircstat.tests import watson_williams, rayleigh
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import vonmises, ranksums, pearsonr, chi2_contingency, chi2, ttest_ind, spearmanr, chisquare, \
    ttest_1samp, linregress, ks_2samp
from scipy.interpolate import interp1d

from common.linear_circular_r import rcc
from common.utils import load_pickle, sigtext, stat_record, p2str
from common.comput_utils import dist_overlap, normsig, append_extrinsicity, linear_circular_gauss_density, \
    find_pair_times, directionality, \
    TunningAnalyzer, check_border, window_shuffle, get_field_directionality, \
    compute_straightness, calc_kld, window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, segment_passes, pair_diff, \
    check_border_sim, midedges, linear_gauss_density, circ_ktest, angular_dispersion_test, circular_density_1d, \
    fisherexact
from common.script_wrappers import DirectionalityStatsByThresh
from common.visualization import plot_correlogram, plot_dcf_phi_histograms, plot_marginal_slices, customlegend
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw


figl = 1.75
# figext = 'png'
figext = 'eps'


def plot_pair_correlation(expdf, save_dir=None):
    stat_fn = 'fig6_paircorr.txt'
    stat_record(stat_fn, True)
    fig, ax = plt.subplots(2, 3, figsize=(figl*3, figl*2), sharex='col', sharey='row')

    markersize = 6
    y_dict = {}
    for caid, (ca, cadf) in enumerate(expdf.groupby('ca')):
        # A->B
        ax[0, caid], x, y, regress = plot_correlogram(ax=ax[0, caid], df=cadf, tag=ca, direct='A->B', color=ca_c[ca], alpha=0.3,
                                                      fontsize=fontsize, markersize=markersize, ticksize=ticksize)
        ax[0, caid].set_title(ca, fontsize=titlesize)
        nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
        stat_record(stat_fn, False, '%s A->B, y = %0.2fx + %0.2f, rho=%0.2f, n=%d, p=%0.4f' % \
                    (ca, regress['aopt'] * 2 * np.pi, regress['phi0'], regress['rho'], nsamples, regress['p']))

        # B->A
        ax[1, caid], x, y, regress = plot_correlogram(ax=ax[1, caid], df=cadf, tag=ca, direct='B->A', color=ca_c[ca], alpha=0.3,
                                                      fontsize=fontsize, markersize=markersize, ticksize=ticksize)
        ax[1, caid].set_xlabel('Field overlap', fontsize=fontsize)
        nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
        stat_record(stat_fn, False, '%s B->A, y = %0.2fx + %0.2f, rho=%0.2f, n=%d, p=%0.4f' % \
                    (ca, regress['aopt'] * 2 * np.pi, regress['phi0'], regress['rho'], nsamples, regress['p']))

    ax[0, 0].set_ylabel('Phase shift (rad)', fontsize=fontsize)
    ax[1, 0].set_ylabel('Phase shift (rad)', fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp_paircorr.png'), dpi=dpi)



def plot_exintrinsic(expdf, save_dir, usephase=False):

    stat_fn = 'fig7_exintrinsic.txt'
    stat_record(stat_fn, True)

    phasetag = '_phase' if usephase else ''
    ratio_key, oplus_key, ominus_key = 'overlap_ratio'+phasetag, 'overlap_plus'+phasetag, 'overlap_minus'+phasetag

    # Filtering
    smallerdf = expdf[(~expdf[ratio_key].isna())]
    smallerdf['Extrincicity'] = smallerdf[ratio_key].apply(lambda x: 'Extrinsic' if x > 0 else 'Intrinsic')

    fig_exin, ax_exin = plt.subplots(2, 3, figsize=(figl*3, figl*2), sharey='row', sharex='row')
    contindf_dict = {}
    for caid, (ca, cadf) in enumerate(smallerdf.groupby('ca')):
        corr_overlap = cadf[oplus_key].to_numpy()
        corr_overlap_flip = cadf[ominus_key].to_numpy()
        corr_overlap_ratio = cadf[ratio_key].to_numpy()

        # 1-sample chisquare test
        n_ex = np.sum(corr_overlap_ratio > 0)
        n_in = np.sum(corr_overlap_ratio <= 0)
        n_total = n_ex + n_in
        if ca !='CA2':
            contindf_dict[ca] = [n_ex, n_in]
        chistat, pchi = chisquare([n_ex, n_in])
        stat_record(stat_fn, False, '%s, %d/%d, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f' % \
                    (ca, n_ex, n_total, 1, n_total, chistat, pchi))
        # 1-sample t test
        mean_ratio = np.mean(corr_overlap_ratio)
        ttest_stat, p_1d1samp = ttest_1samp(corr_overlap_ratio, 0)
        stat_record(stat_fn, False, '%s, mean=%0.4f, t(%d)=%0.2f, p=%0.4f' % \
                    (ca, mean_ratio, n_total-1, ttest_stat, p_1d1samp))
        # Plot scatter 2d
        ax_exin[0, caid].scatter(corr_overlap_flip, corr_overlap, alpha=0.5, s=4, c=ca_c[ca])
        ax_exin[0, caid].text(0.25, 0, 'Ex. Frac.=%0.2f\np%s'%(n_ex/n_total, p2str(pchi)), fontsize=legendsize)
        ax_exin[0, caid].plot([0, 1], [0, 1], c='k')
        ax_exin[0, 1].set_xlabel('Extrinsicity', fontsize=fontsize)
        ax_exin[0, caid].set_xticks([0, 1])
        ax_exin[0, caid].set_yticks([0, 1])
        ax_exin[0, caid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_exin[0, caid].set_title(ca, fontsize=titlesize)

        # Plot 1d histogram
        edges = np.linspace(-1, 1, 75)
        width = edges[1]-edges[0]
        (bins, _, _) = ax_exin[1, caid].hist(corr_overlap_ratio, bins=edges, color=ca_c[ca],
                                        density=True, histtype='stepfilled')
        ax_exin[1, caid].plot([mean_ratio, mean_ratio], [0, bins.max()], c='k')
        ax_exin[1, caid].text(-0.25, 0.13/width, 'Mean=%0.3f\np%s'%(mean_ratio, p2str(p_1d1samp)), fontsize=legendsize)
        ax_exin[1, caid].set_xticks([-0.5, 0, 0.5])
        ax_exin[1, caid].set_yticks([0, 0.1/width] )
        ax_exin[1, caid].set_yticklabels(['0', '0.1'])
        ax_exin[1, 1].set_xlabel('Extrinsicity - Intrinsicity', fontsize=fontsize)
        ax_exin[1, caid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_exin[1, caid].set_xlim(-0.5, 0.5)
        ax_exin[1, caid].set_ylim(0, 0.2/width)

    ax_exin[0, 0].set_ylabel('Intrinsicity', fontsize=fontsize)
    ax_exin[1, 0].set_ylabel('Normalized counts', fontsize=fontsize)


    # two-sample test, df = n1+n2 - 2
    ratio_1 = smallerdf[smallerdf['ca'] == 'CA1'][ratio_key].to_numpy()
    ratio_2 = smallerdf[smallerdf['ca'] == 'CA2'][ratio_key].to_numpy()
    ratio_3 = smallerdf[smallerdf['ca'] == 'CA3'][ratio_key].to_numpy()
    n1, n2, n3 = ratio_1.shape[0], ratio_2.shape[0], ratio_3.shape[0]

    equal_var = False
    t_13, p_13 = ttest_ind(ratio_1, ratio_3, equal_var=equal_var)
    t_23, p_23 = ttest_ind(ratio_2, ratio_3, equal_var=equal_var)
    t_12, p_12 = ttest_ind(ratio_1, ratio_2, equal_var=equal_var)

    stat_record(stat_fn, False, 'Welch\'s t-test: CA1 vs CA3, t(%d)=%0.2f, p=%0.4f' % (n1 + n3 - 2, t_13, p_13))
    stat_record(stat_fn, False, 'Welch\'s t-test: CA2 vs CA3, t(%d)=%0.2f, p=%0.4f' % (n2 + n3 - 2, t_23, p_23))
    stat_record(stat_fn, False, 'Welch\'s t-test: CA1 vs CA2, t(%d)=%0.2f, p=%0.4f' % (n1 + n2 - 2, t_12, p_12))

    contindf = pd.DataFrame(contindf_dict)
    fisherp = fisherexact(contindf.to_numpy())
    _, p13_chi2way, _, _ = chi2_contingency(contindf)
    print('2-way Chisquare test (CA1-3) pval=%0.5f. Fisher p = %0.4f' % (p13_chi2way, fisherp))

    if save_dir:
        fig_exin.tight_layout()
        fig_exin.savefig(os.path.join(save_dir, 'exp_exintrinsic.png'), dpi=dpi)


def plot_exintrinsic_pair_correlation(expdf, save_dir=None, usephase=False):
    stat_fn = 'fig7_concentration.txt'
    stat_record(stat_fn, True)
    phasetag = '_phase' if usephase else ''
    ratio_key, oplus_key, ominus_key = 'overlap_ratio'+phasetag, 'overlap_plus'+phasetag, 'overlap_minus'+phasetag

    fig03, ax03 = plt.subplots(2, 3, figsize=(figl * 3, figl * 2))
    fig_hist, ax_hist = plt.subplots(1, 3, figsize=(figl * 3, figl), sharey=True)

    markersize = 6
    alpha = 0.5
    y_dict = {}
    for caid, (ca, cadf) in enumerate(expdf.groupby('ca')):
        exdf = cadf[cadf[ratio_key] > 0]
        indf = cadf[cadf[ratio_key] <= 0]


        # Overlap < 0.3
        ax03[0, caid], _, y03ex, _ = plot_correlogram((ax03[0, caid]), exdf[exdf['overlap'] < 0.3], 'Extrinsic',
                                                      'combined', density=False, regress=False, x_dist=False,
                                                      y_dist=True,
                                                      alpha=alpha, fontsize=fontsize, markersize=markersize, ticksize=ticksize)
        ax03[0, caid].set_xlabel('')
        ax03[1, caid], _, y03in, _ = plot_correlogram((ax03[1, caid]), indf[indf['overlap'] < 0.3], 'Intrinsic',
                                                      'combined', density=False, regress=False, x_dist=False,
                                                      y_dist=True,
                                                      alpha=alpha, fontsize=fontsize, markersize=markersize, ticksize=ticksize)
        y_dict[ca + 'ex'] = y03ex[~np.isnan(y03ex)]
        y_dict[ca + 'in'] = y03in[~np.isnan(y03in)]

        # Phaselag histogram
        yex_ax, yex_den = circular_density_1d(y_dict[ca + 'ex'], 4 * np.pi, 50, (-np.pi, np.pi))
        yin_ax, yin_den = circular_density_1d(y_dict[ca + 'in'], 4 * np.pi, 50, (-np.pi, np.pi))
        F_k, pval_k = circ_ktest(y_dict[ca + 'ex'], y_dict[ca + 'in'])
        stat_record(stat_fn, False, '%s Ex-in concentration difference F_{(n_1=%d, n_2=%d)}=%0.2f, p=%0.4f'%\
                    (ca, y_dict[ca + 'ex'].shape[0], y_dict[ca + 'in'].shape[0], F_k, pval_k))

        ax_hist[caid].plot(yex_ax, yex_den, c='r', label='ex')
        ax_hist[caid].plot(yin_ax, yin_den, c='b', label='in')
        ax_hist[caid].set_xticks([-np.pi, 0, np.pi])
        ax_hist[caid].set_yticks([0, 0.05])
        ax_hist[caid].set_xticklabels(['$-\pi$', 0, '$\pi$'])
        ax_hist[caid].set_xlabel('Phase shift (rad)', fontsize=fontsize)
        ax_hist[caid].tick_params(labelsize=ticksize)
        ax_hist[caid].text(-np.pi/1.8, 0.025, 'Bartlett\'s\np%s'%(p2str(pval_k)), fontsize=legendsize)
        ax_hist[caid].text(np.pi / 4, 0.05, 'Ex.', fontsize=legendsize, color='r')
        ax_hist[caid].text(np.pi / 2 + 0.1, 0.01, 'In.', fontsize=legendsize, color='b')
    ax_hist[0].set_ylabel('Density of pairs', fontsize=fontsize)
    if save_dir:

        fig03.tight_layout()
        fig03.savefig(os.path.join(save_dir, 'exp_exintrinsic_paircorr_03.png'))

        fig_hist.tight_layout()
        fig_hist.savefig(os.path.join(save_dir, 'exp_exintrinsic_concentration.png'), dpi=dpi)


def plot_firing_rate(expdf, save_dir):
    # Filtering
    smallerdf = expdf[(~expdf['corate'].isna()) & (~expdf['rate_AB'].isna()) & (~expdf['rate_BA'].isna())]

    # Figures
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    fig_2d, ax_2d = plt.subplots(1, 3, figsize=(6, 2), sharex=True, sharey=True)
    color_dict = {'CA1': 'k', 'CA2': 'g', 'CA3': 'r'}
    total_dict, diff_dict = dict(), dict()

    totalrate_txt = 'Total rate' + ' (Hz)\n' + '$R_{AB}+R_{BA}$'
    diffrate_txt = '$\dfrac{|R_{AB}-R_{BA}|}{R_{AB}+R_{BA}}$'

    for caid, (ca, cadf) in enumerate(smallerdf.groupby('ca')):
        corate = cadf['corate'].to_numpy()
        rateAB = cadf['rate_AB'].to_numpy()
        rateBA = cadf['rate_BA'].to_numpy()

        rate_diff = np.abs(rateAB - rateBA) / (rateAB + rateBA)

        total_dict[caid] = corate
        diff_dict[caid] = rate_diff

        bins1, edges1 = np.histogram(corate, bins=100)
        bins2, edges2 = np.histogram(rate_diff, bins=100)

        ax[0].plot(midedges(edges1), np.cumsum(bins1) / np.sum(bins1), c=color_dict[ca], label=ca)
        ax[1].plot(midedges(edges2), np.cumsum(bins2) / np.sum(bins2), c=color_dict[ca], label=ca)

        # Co-rate v.s. diff rate
        ax_2d[caid].scatter(corate, rate_diff, marker='.', c=color_dict[ca], alpha=0.5, s=8)
        ax_2d[caid].set_xlabel(totalrate_txt, fontsize=fontsize)
        ax_2d[caid].tick_params(axis='both', which='major', labelsize=fontsize)
        # Regression
        slope, intercept, rho, corrp, _ = linregress(corate, rate_diff)
        x_dum = np.linspace(corate.min(), corate.max(), 10)
        ax_2d[caid].plot(x_dum, x_dum * slope + intercept, c='k')
        ax_2d[caid].set_title('Rho=%0.3f (pval=%0.5f)' % (rho, corrp), fontsize=fontsize)

    ax_2d[0].set_ylabel(diffrate_txt, fontsize=fontsize)
    for ax_each in ax:
        ax_each.legend(fontsize=fontsize - 2)
        ax_each.tick_params(axis='both', which='major', labelsize=fontsize)
    _, p13 = ranksums(total_dict[0], total_dict[2])
    ax[0].text(15, 0.4, 'Ranksums\np13=%0.5f' % (p13), fontsize=fontsize)
    ax[0].set_xlabel(totalrate_txt, fontsize=fontsize)

    _, p13 = ranksums(diff_dict[0], diff_dict[2])
    ax[1].text(0.3, 0.5, 'Ranksums\np13=%0.5f' % (p13), fontsize=fontsize)
    ax[1].set_xlabel(diffrate_txt, fontsize=fontsize)

    if save_dir:
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'exp_firingrates.png'))
        fig_2d.tight_layout()
        fig_2d.savefig(os.path.join(save_dir, 'exp_corate-ratediff.png'))


def omniplot_pairfields(expdf, save_dir=None):

    figl = total_figw/5
    linew = 0.75

    # # Initialization
    stat_fn = 'fig4_pair_directionality.txt'
    stat_record(stat_fn, True)
    fig, ax = plt.subplots(2, 3, figsize=(total_figw*(2.8/4), figl*2), sharey='row', sharex='col')
    fig_frac, ax_frac = plt.subplots(2, 1, figsize=(total_figw*(1.2/4), figl*2), sharex=True)
    spike_threshs = np.arange(0, 1001, 50)
    stats_getter = DirectionalityStatsByThresh('num_spikes_pair', 'win_pval_pair_mlm', 'shift_pval_pair_mlm',
                                               'fieldR_pair_mlm')
    expdf['border'] = expdf['border1'] | expdf['border2']

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
            stat_record(stat_fn, False, '%s BorderEffect Ex/inFrac, Thresh=%0.2f, \chi^2_{(df=%d, n=%d)}=%0.2f, p=%0.4f' % \
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
            stat_record(stat_fn, False, '%s CA13 Ex/inFrac, Thresh=%0.2f, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f'% \
                        (bcase, ntresh, chi_dof13, nCA1+nCA3, chi_stat13, chi_p13))
            stat_record(stat_fn, False, '%s CA23 Ex/inFrac, Thresh=%0.2f, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f' % \
                        (bcase, ntresh, chi_dof23, nCA2 + nCA3, chi_stat23, chi_p23))
            stat_record(stat_fn, False, '%s CA12 Ex/inFrac, Thresh=%0.2f, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f' % \
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

    customlegend(ax[0, 2], linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')
    customlegend(ax_frac[0], handlelength=0.5, linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.75, 0.8], loc='center')


    ax[0, 0].set_ylabel('Median R', fontsize=fontsize)
    ax[1, 0].set_ylabel('Significant\n Fraction', fontsize=fontsize)
    for i in range(3):
        # ax[0, i].set_ylim(0.05, 0.17)  # R, spikes
        ax[0, i].set_yticks([0.1, 0.2, 0.3, 0.4])
        ax[1, i].set_ylim(0, 0.65)
        ax[1, i].set_yticks([0, 0.2, 0.4, 0.6])
        ax[1, i].set_yticklabels(['0', '', '0.4', ''])
        ax[1, i].set_xticks([0, 250, 500, 750, 1000])
        ax[1, i].set_xticklabels(['0', '', '', '', '1000'])
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
    ax_frac[0].set_yticks([0, 1])

    ax_frac[1].set_ylim(0, 0.65)
    ax_frac[1].set_yticks([0, 0.2, 0.4, 0.6])
    ax_frac[1].set_yticklabels(['0', '', '0.4', ''])
    ax_frac[1].set_xticks([0, 250, 500, 750, 1000])
    ax_frac[1].set_xticklabels(['0', '', '', '', '1000'])
    ax_frac[1].tick_params(labelsize=ticksize)
    ax_frac[1].set_ylabel('Border fields\nfraction', fontsize=fontsize)
    ax_frac[1].spines['top'].set_visible(False)
    ax_frac[1].spines['right'].set_visible(False)
    fig_frac.text(0.55, 0.01, 'Spike count threshold', ha='center', fontsize=fontsize)

    fig.tight_layout()
    fig_frac.tight_layout()
    if save_dir:

        fig.savefig(os.path.join(save_dir, 'exp_pair_directionality.%s'%(figext)), dpi=dpi)
        fig_frac.savefig(os.path.join(save_dir, 'exp_pair_fraction.%s'%(figext)), dpi=dpi)


def plot_kld(expdf, save_dir=None):
    stat_fn = 'fig5_kld.txt'
    stat_record(stat_fn, True)

    # Filtering
    expdf['border'] = expdf['border1'] | expdf['border2']

    kld_key = 'kld_mlm'

    figw = total_figw*0.9/2
    figh = total_figw*0.9/4.5

    # # # KLD by threshold
    Nthreshs = np.arange(0, 1001, 100)
    fig_kld_thresh, ax_kld_thresh = plt.subplots(1, 3, figsize=(figw, figh), sharey=True, sharex=True)
    kld_thresh_all_dict = dict(CA1=[], CA2=[], CA3=[])
    kld_thresh_border_dict = dict(CA1=[], CA2=[], CA3=[])
    kld_thresh_nonborder_dict = dict(CA1=[], CA2=[], CA3=[])
    linew = 0.75
    # # KLD by threshold - All
    for ca, cadf in expdf.groupby(by=['ca']):
        all_kld = np.zeros(Nthreshs.shape[0])
        for idx, thresh in enumerate(Nthreshs):
            cadf2 = cadf[cadf['num_spikes_pair'] > thresh]
            kld_arr = cadf2[kld_key].to_numpy()
            kld_thresh_all_dict[ca].append(kld_arr)
            all_kld[idx] = np.median(kld_arr)
        ax_kld_thresh[0].plot(Nthreshs, all_kld, c=ca_c[ca], label='%s' % (ca), linewidth=linew)

    # # KLD by threshold - Border/Non-border
    for (ca, border), gpdf in expdf.groupby(by=['ca', 'border']):
        axid = 1 if border else 2
        all_kld = np.zeros(Nthreshs.shape[0])
        linestyle='dotted' if border else 'dashed'
        for idx, thresh in enumerate(Nthreshs):
            gpdf2 = gpdf[gpdf['num_spikes_pair'] > thresh]
            kld_arr = gpdf2[kld_key].to_numpy()
            if border:
                kld_thresh_border_dict[ca].append(kld_arr)
            else:
                kld_thresh_nonborder_dict[ca].append(kld_arr)
            all_kld[idx] = np.median(kld_arr)
        ax_kld_thresh[axid].plot(Nthreshs, all_kld, c=ca_c[ca], linestyle=linestyle, linewidth=linew)

    # # KLD by threshold - Other plotting
    customlegend(ax_kld_thresh[0], handlelength=0.5, linewidth=1.2, fontsize=legendsize-2, bbox_to_anchor=[0.8, 0.8], loc='center')
    ax_kld_thresh[0].set_ylabel('Median KLD (bit)', fontsize=fontsize)
    ax_kld_thresh[0].set_title('All', fontsize=titlesize)
    ax_kld_thresh[1].set_title('B', fontsize=titlesize)
    ax_kld_thresh[2].set_title('N-B', fontsize=titlesize)

    for ax_each in ax_kld_thresh:

        ax_each.tick_params(labelsize=ticksize)
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
    plt.setp(ax_kld_thresh[0].get_xticklabels(), visible=False)
    ax_kld_thresh[1].set_xticks([0, 500, 1000])
    ax_kld_thresh[1].set_xticklabels(['0', '', '1000'])
    plt.setp(ax_kld_thresh[2].get_xticklabels(), visible=False)

    fig_kld_thresh.text(0.6, 0.01, 'Spike count threshold', ha='center', fontsize=fontsize)

    # # KLD by threshold - Statistical test
    # CA effect by border
    bordertxt = ['ALL', 'Border', 'Nonborder']
    for idx, thresh in enumerate(Nthreshs):
        for borderid, klddict in enumerate([kld_thresh_all_dict, kld_thresh_border_dict, kld_thresh_nonborder_dict]):
            kld1, kld2, kld3 = klddict['CA1'][idx], klddict['CA2'][idx], klddict['CA3'][idx]
            n1, n2, n3 = kld1.shape[0], kld2.shape[0], kld3.shape[0]
            z13, p13 = ranksums(kld1, kld3)
            z23, p23 = ranksums(kld2, kld3)
            z12, p12 = ranksums(kld1, kld2)
            stat_record(stat_fn, False, 'KLD by threshold=%d, CA13, %s, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (thresh, bordertxt[borderid], z13, n1, n3, p13))
            stat_record(stat_fn, False, 'KLD by threshold=%d, CA23, %s, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (thresh, bordertxt[borderid], z23, n2, n3, p23))
            stat_record(stat_fn, False, 'KLD by threshold=%d, CA12, %s, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (thresh, bordertxt[borderid], z12, n1, n2, p12))

    # Border effect by CA
    for ca in ['CA1', 'CA2', 'CA3']:
        for idx, thresh in enumerate(Nthreshs):
            kld_b, kld_nb = kld_thresh_border_dict[ca][idx], kld_thresh_nonborder_dict[ca][idx]
            n_b, n_nd = kld_b.shape[0], kld_nb.shape[0]
            z_nnb, p_nnb = ranksums(kld_b, kld_nb)
            stat_record(stat_fn, False, 'KLD by threshold=%d, %s, Border efffect, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                        (thresh, ca, z_nnb, n_b, n_nd, p_nnb))
    # Save
    fig_kld_thresh.tight_layout()
    fig_kld_thresh.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_kld_thresh.savefig(os.path.join(save_dir, 'exp_kld_thresh.%s'%(figext)), dpi=dpi)


def plot_pairangle_similarity_analysis(expdf, save_dir=None, usephase=False):
    stat_fn = 'fig7_pair_exin_simdisim.txt'
    stat_record(stat_fn, True)
    phasetag = '_phase' if usephase else ''
    ratio_key, oplus_key, ominus_key = 'overlap_ratio'+phasetag, 'overlap_plus'+phasetag, 'overlap_minus'+phasetag

    slice_figsize = (figl*3, figl)
    fig_sim, ax_sim = plt.subplots(2, 3, figsize=(figl*3, figl*2), sharex=True, sharey=True)
    fig_exinbar, ax_exinbar = plt.subplots(figsize=(figl, figl), sharex=True, sharey=True)
    fig_ratio, ax_ratio = plt.subplots(2, 3, figsize=(figl*3, figl*2), sharex='row', sharey='row')
    fig_ratio2, ax_ratio2 = plt.subplots(1, 2, figsize=(figl*2, figl), sharex=True, sharey=True)
    fig_simex, ax_simex = plt.subplots(1, 3, figsize=(figl*3, figl+0.5), sharex=True, sharey=True)
    fig_lagden, ax_lagden = plt.subplots(1, 3, figsize=(figl*3, figl+0.5), sharex=True, sharey=True)
    fig_lagslice, ax_lagslice = plt.subplots(1, 3, figsize=slice_figsize, sharex=True)

    ratiosimdict = dict()
    ratiodisimdict = dict()
    df_exin_dict = dict(ca=[], sim=[], exnum=[], innum=[])
    # extrinsic/intrinsic
    for caid, (ca, cadf) in enumerate(expdf.groupby(by='ca')):

        cadf = cadf[(~cadf[ratio_key].isna()) & (~cadf['precess_angle1'].isna()) & (~cadf['precess_angle2'].isna())].reset_index(drop=True)
        s1, s2 = cadf['precess_angle1'], cadf['precess_angle2']

        ominus, oplus = cadf[ominus_key], cadf[oplus_key]
        oratio = cadf[ratio_key]

        circdiff = np.abs(cdiff(s1, s2))

        simidx = np.where(circdiff < (np.pi / 2))[0]
        disimidx = np.where(circdiff > np.pi - (np.pi / 2))[0]
        
        orsim, ordisim = oratio[simidx], oratio[disimidx]
        exnumsim, exnumdisim = np.sum(orsim>0), np.sum(ordisim>0)
        innumsim, innumdisim = np.sum(orsim<=0), np.sum(ordisim<=0)
        exfracsim, exfracdisim = exnumsim/(exnumsim+innumsim), exnumdisim/(exnumdisim+innumdisim)

        chisq_exsim, p_exfracsim = chisquare([exnumsim, innumsim])
        chisq_exdissim, p_exfracdisim = chisquare([exnumdisim, innumdisim])
        stat_record(stat_fn, False, '%s, similar, %d/%d, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f' % \
                    (ca, exnumsim, orsim.shape[0], 1, orsim.shape[0], chisq_exsim, p_exfracsim))
        stat_record(stat_fn, False, '%s, dissimilar, %d/%d, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f' % \
                    (ca, exnumdisim, ordisim.shape[0], 1, ordisim.shape[0], chisq_exdissim, p_exfracdisim))
        # Append to contingency table
        df_exin_dict['ca'].append(ca)
        df_exin_dict['sim'].append('similar')
        df_exin_dict['exnum'].append(exnumsim)
        df_exin_dict['innum'].append(innumsim)
        df_exin_dict['ca'].append(ca)
        df_exin_dict['sim'].append('dissimilar')
        df_exin_dict['exnum'].append(exnumdisim)
        df_exin_dict['innum'].append(innumdisim)


        # 2D scatter of ex-intrinsicitiy for similar/dissimlar
        ax_sim[0, caid].scatter(ominus[simidx], oplus[simidx], marker='.', alpha=0.5, c=ca_c[ca])
        ax_sim[0, caid].text(0.25, 0, 'Ex. Frac.=%0.2f\np%s'%(exfracsim, p2str(p_exfracsim)), fontsize=legendsize)
        ax_sim[0, caid].plot([0, 1], [0, 1], c='k')
        ax_sim[0, caid].set_title(ca, fontsize=titlesize)
        ax_sim[0, caid].tick_params(labelsize=ticksize)
        ax_sim[1, caid].scatter(ominus[disimidx], oplus[disimidx], marker='.', alpha=0.5, c=ca_c[ca])
        ax_sim[1, caid].text(0.25, 0, 'Ex. Frac.=%0.2f\np%s'%(exfracdisim, p2str(p_exfracdisim)), fontsize=legendsize)
        ax_sim[1, caid].plot([0, 1], [0, 1], c='k')
        ax_sim[1, 1].set_xlabel('Extrinsicity', fontsize=fontsize)
        ax_sim[1, caid].tick_params(labelsize=ticksize)
        ax_sim[0, 0].set_xticks([0, 1])
        ax_sim[0, 0].set_yticks([0, 1])
        ax_sim[0, 0].set_ylabel('Similar\nIntrinsicity', fontsize=fontsize)
        ax_sim[1, 0].set_ylabel('Dissimilar\nIntrinsicity', fontsize=fontsize)
        # ax_sim[0, 0].set_ylabel('Similar\nIntrinsicity', fontsize=titlesize)
        # ax_sim[1, 0].set_ylabel('Dissimilar\nIntrinsicity', fontsize=titlesize)
        
        
        obins_sim, oedges_sim = np.histogram(orsim, bins=30)
        oedges_sim_m = midedges(oedges_sim)
        obins_disim, oedges_disim = np.histogram(ordisim, bins=30)
        oedges_disim_m = midedges(oedges_disim)

        ratiosimdict[ca] = orsim
        ratiodisimdict[ca] = ordisim

        # 1D ex-intrinsic histograms for difference between similar/disimilar
        ax_ratio[0, caid].plot(oedges_sim_m, obins_sim / np.sum(obins_sim), label='sim')
        ax_ratio[0, caid].plot(oedges_disim_m, obins_disim / np.sum(obins_sim), label='dissimilar')
        ax_ratio[1, caid].plot(oedges_sim_m, np.cumsum(obins_sim) / np.sum(obins_sim), label='sim')
        ax_ratio[1, caid].plot(oedges_disim_m, np.cumsum(obins_disim) / np.sum(obins_disim), label='dissimilar')
        ax_ratio[0, caid].legend(fontsize=fontsize - 2, handlelength=1, labelspacing=0.2, handletextpad=0.2, borderpad=0.1)
        ax_ratio[1, caid].legend(fontsize=fontsize - 2, handlelength=1, labelspacing=0.2, handletextpad=0.2, borderpad=0.1)
        ax_ratio[1, caid].set_xlabel('Intrinsic <--> Extrinsic', fontsize=fontsize)
        ax_ratio[0, caid].tick_params(labelsize=fontsize-1)
        ax_ratio[1, caid].tick_params(labelsize=fontsize-1)

        # 1D ex-intrinsic histograms for difference between CA1-3
        ax_ratio2[0].plot(oedges_sim_m, np.cumsum(obins_sim) / np.sum(obins_sim), label=ca, c=ca_c[ca])
        ax_ratio2[1].plot(oedges_disim_m, np.cumsum(obins_disim) / np.sum(obins_disim), label=ca, c=ca_c[ca])

        # Extrinsicity v.s. Cdiff
        ax_simex[caid].scatter(circdiff, oratio, alpha=0.7, s=8)
        slope, intercept, rho, p_regress, _ = linregress(circdiff, oratio)
        dum_x = np.linspace(0, np.pi, 10)
        ax_simex[caid].plot(dum_x, slope * dum_x + intercept, c='k')
        ax_simex[caid].set_title('Rho=%0.4f(%s)' % (rho, sigtext(p_regress)), fontsize=fontsize)
        ax_simex[caid].set_xlabel(r'$|d(\theta_{precess1}, \theta_{precess2})|$', fontsize=fontsize)
        ax_simex[caid].set_xticks([0, np.pi / 2, np.pi])
        ax_simex[caid].set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        ax_simex[caid].tick_params(axis='both', which='major', labelsize=fontsize)
        ax_simex[0].set_ylabel('Intrinsic <-> Extrinsic', fontsize=fontsize)

        # Phaselag v.s. Cdiff
        lagdf = cadf[(~cadf['phaselag_AB'].isna()) & (~cadf['phaselag_BA'].isna())]
        lag4 = np.concatenate([lagdf['phaselag_AB'].to_numpy(), -lagdf['phaselag_BA'].to_numpy()])
        circdiff4 = np.concatenate([circdiff, circdiff])

        # Phaselag v.s. Cdiff (density)
        xx, yy, zz = linear_circular_gauss_density(circdiff4, lag4, cir_kappa=2 * np.pi, lin_std=0.2, xbins=200,
                                                   ybins=200, xbound=(0, np.pi), ybound=(-np.pi, 3 * np.pi))
        
        ax_lagden[caid].pcolormesh(xx, yy, zz)
        cdiff_bins, cdiff_edges = np.histogram(circdiff4, range=(0, np.pi), bins=50)
        phase_bins, phase_edges = np.histogram(lag4, range=(-np.pi, 3 * np.pi), bins=100)
        ax_lagden[caid].plot(phase_bins / np.max(phase_bins), (phase_edges[1:] + phase_edges[:-1]) / 2, c='k',
                             alpha=0.3)
        ax_lagden[caid].plot((cdiff_edges[1:] + cdiff_edges[:-1]) / 2, cdiff_bins / np.max(cdiff_bins) * 2 - np.pi,
                             c='k', alpha=0.3)

        regress = rcc(circdiff4, lag4, abound=[-2, 2])
        r_regress = np.linspace(0, np.pi, 10)
        lag_regress = 2 * np.pi * r_regress * regress['aopt'] + regress['phi0']
        ax_lagden[caid].plot(r_regress, lag_regress, c='k', label='rho=%0.2f(%s)'%(regress['rho'], sigtext(regress['p'])))
        ax_lagden[caid].set_xlabel(r'$|d(\theta_{precess1}, \theta_{precess2})|$', fontsize=fontsize)
        ax_lagden[caid].set_yticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
        ax_lagden[caid].set_yticklabels(['$-\pi$', '0', '$\pi$', '$2\pi$', '$3\pi$'])
        ax_lagden[caid].set_xticks([0, np.pi/2, np.pi])
        ax_lagden[caid].set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        ax_lagden[caid].tick_params(labelsize=fontsize-1)
        ax_lagden[0].set_ylabel('Pair phase lag (rad)', fontsize=fontsize)
        ax_lagden[caid].legend(fontsize=fontsize - 2, handlelength=1, labelspacing=0.2, handletextpad=0.2, borderpad=0.1)

        # Phaselag v.s. Cdiff (Marginal slices)
        x_slices = np.linspace(0, np.pi, 6)
        miny, maxy = 4 - (2 * np.pi), 4
        ax_lagslice[caid], color_list = plot_marginal_slices(ax_lagslice[caid], xx, yy, zz, selected_x=x_slices, 
                                            slice_yrange=(miny, maxy), slicegap=0.01)
        ax_lagslice[caid].set_xlabel('Pair phase lag (rad)')
        ax_lagslice[caid].set_yticks([])
        ax_lagslice[caid].set_xticks([miny, 0, maxy])
        ax_lagslice[caid].set_xticklabels(['$4-2\pi$', 0, '4'])
        ax_lagslice[caid].tick_params(labelsize=fontsize-1)

        ax_lagden[caid].scatter(x_slices, [-np.pi]*x_slices.shape[0], c=color_list)


    # Compare sim/disim
    _, pca1 = ranksums(ratiosimdict['CA1'], ratiodisimdict['CA1'])
    _, pca3 = ranksums(ratiosimdict['CA3'], ratiodisimdict['CA3'])
    ax_ratio[1, 0].set_title('p(sim-disim)=%0.4f' % (pca1), fontsize=fontsize)
    ax_ratio[1, 2].set_title('p(sim-disim)=%0.4f' % (pca3), fontsize=fontsize)

    # Compare CA1-3
    nsim1, nsim2, nsim3 = ratiosimdict['CA1'].shape[0], ratiosimdict['CA2'].shape[0], ratiosimdict['CA3'].shape[0]
    ndisim1, ndisim2, ndisim3 = ratiodisimdict['CA1'].shape[0], ratiodisimdict['CA2'].shape[0], ratiodisimdict['CA3'].shape[0]



    zsim13, psim13 = ranksums(ratiosimdict['CA1'], ratiosimdict['CA3'])
    zdisim13, pdisim13 = ranksums(ratiodisimdict['CA1'], ratiodisimdict['CA3'])
    zsim23, psim23 = ranksums(ratiosimdict['CA2'], ratiosimdict['CA3'])
    zdisim23, pdisim23 = ranksums(ratiodisimdict['CA2'], ratiodisimdict['CA3'])

    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Similar, CA1 vs CA3, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f'% (zsim13, nsim1, nsim3, psim13))
    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Similar, CA2 vs CA3, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f'% (zsim23, nsim2, nsim3, psim23))
    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Dissimilar, CA1 vs CA3, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f'% (zdisim13, ndisim1, ndisim3, pdisim13))
    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Dissimilar, CA2 vs CA3, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f'% (zdisim23, ndisim2, ndisim3, pdisim23))

    _, psim13_ks = ks_2samp(ratiosimdict['CA1'], ratiosimdict['CA3'])
    _, pdisim13_ks = ks_2samp(ratiodisimdict['CA1'], ratiodisimdict['CA3'])
    ax_ratio2[0].text(0.1, 0.2, 'p13rs=\n%0.4f\np13ks=\n%0.4f' % (psim13, psim13_ks), fontsize=fontsize-1)
    ax_ratio2[0].set_title('Similar', fontsize=fontsize)
    ax_ratio2[0].set_xlabel('Intrinsic <-> Extrinsic', fontsize=fontsize)
    ax_ratio2[0].legend(fontsize=fontsize - 2, handlelength=1, labelspacing=0.2, handletextpad=0.2, borderpad=0.1)
    ax_ratio2[1].text(0.1, 0.2, 'p13rs=\n%0.4f\np13ks=\n%0.4f' % (pdisim13, pdisim13_ks), fontsize=fontsize-1)
    ax_ratio2[1].set_title('Dissimilar', fontsize=fontsize)
    ax_ratio2[1].set_xlabel('Intrinsic <-> Extrinsic', fontsize=fontsize)
    ax_ratio2[1].legend(fontsize=fontsize - 2, handlelength=1, labelspacing=0.2, handletextpad=0.2, borderpad=0.1)


    # Compare exin for sim/dissim
    df_exin = pd.DataFrame(df_exin_dict)
    w, xbar = 0.25, np.array([1, 2, 3])
    xbar1 = xbar - w / 2
    xbar2 = xbar + w / 2

    df_exin['exfrac'] = df_exin['exnum'] / (df_exin['exnum'] + df_exin['innum'])
    ax_exinbar.bar(xbar1, df_exin[df_exin['sim'] == 'similar']['exfrac'].to_numpy(), color='darkgreen', label='similar', width=w)
    ax_exinbar.bar(xbar2, df_exin[df_exin['sim'] == 'dissimilar']['exfrac'].to_numpy(), color='darkorange', label='dissimilar',
           width=w)
    ax_exinbar.set_xticks(xbar)
    ax_exinbar.set_xticklabels(['CA1', 'CA2', 'CA3'])
    ax_exinbar.tick_params(labelsize=fontsize-1)
    ax_exinbar.legend(fontsize=fontsize - 2, handlelength=1, labelspacing=0.2, handletextpad=0.2, borderpad=0.1)

    ptext_exinbar = ''
    for ca in ['CA1', 'CA2', 'CA3']:
        df_exin2 = df_exin[df_exin['ca'] == ca][['exnum', 'innum']].reset_index(drop=True)
        chi2_exinbar, p_exinbar, dof_exinbar, _ = chi2_contingency(df_exin2)
        ptext_exinbar += '%s(%s)\n'%(ca, sigtext(p_exinbar))
    ax_exinbar.set_title(ptext_exinbar, fontsize=fontsize)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.brg,
                                norm=plt.Normalize(vmin=x_slices.min(), vmax=x_slices.max()))
    fig_colorbar = plt.figure(figsize=slice_figsize)
    fig_colorbar.subplots_adjust(right=0.8)
    cbar_ax = fig_colorbar.add_axes([0.85, 0.15, 0.03, 0.7])
    cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0, np.pi/2, np.pi])
    cb.set_ticklabels(['0', '$\pi/2$', '$\pi$'])
    cb.set_label(r'$|d(\theta_{precess1}, \theta_{precess2})|$', fontsize=fontsize)


    # Others
    fig_sim.tight_layout()
    fig_simex.tight_layout()
    fig_ratio.tight_layout()
    fig_ratio2.tight_layout()
    fig_lagden.tight_layout()
    fig_lagslice.tight_layout()
    fig_colorbar.tight_layout()
    fig_exinbar.tight_layout()
    if save_dir:
        fig_sim.savefig(os.path.join(save_dir, 'exp_simdissim_exintrinsic2d.png'), dpi=300)
        fig_ratio.savefig(os.path.join(save_dir, 'exp_simdissim_exintrinsic1d.png'), dpi=300)
        fig_ratio2.savefig(os.path.join(save_dir, 'exp_simdissim_exintrinsic1d_ca123.png'), dpi=300)
        fig_simex.savefig(os.path.join(save_dir, 'exp_exintrinsic-cdiff.png'), dpi=300)
        fig_lagden.savefig(os.path.join(save_dir, 'exp_phaselag-abscircdiff_density.png'), dpi=300)
        fig_lagslice.savefig(os.path.join(save_dir, 'exp_phaselag-abscircdiff_sepSlices.png'), dpi=300)
        fig_colorbar.savefig(os.path.join(save_dir, 'sepSlices_colorbar.png'), dpi=300)
        fig_exinbar.savefig(os.path.join(save_dir, 'exp_exinbar.png'), dpi=300)


def plot_correlogram_exin(expdf, save_dir=None):
    num_rows = expdf.shape[0]
    for nrow in range(num_rows):
        if nrow > 300:
            break
        print('\rPlotting %d/%d' % (nrow, num_rows), flush=True, end="")

        # Get data
        overlap, overlap_ratio = expdf.loc[nrow, ['overlap', 'overlap_ratio']]
        intrinsicity, extrinsicity = expdf.loc[nrow, ['overlap_plus', 'overlap_minus']]
        lagAB, lagBA, corrAB, corrBA = expdf.loc[nrow, ['phaselag_AB', 'phaselag_BA', 'corr_info_AB', 'corr_info_BA']]
        if np.isnan(overlap_ratio) or np.isnan(lagAB) or np.isnan(lagBA):
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
        peaktimeAB = edmAB[minidxAB]
        peaksignalAB = signalAB[minidxAB]

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
        peaksignalBA = signalBA[minidxBA]

        # Plot correlograms
        fig_corr, ax_corr = plt.subplots(2, 1, figsize=(3*figl/2, figl*2), sharex=True)
        ax_corr[0].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='gray', alpha=0.5, label='A->B')
        ax_corr[0].plot([0, 0], [0, binsABnorm.max() * 0.85], c='k', alpha=0.7)

        ax_corr[0].set_yticks([])
        ax_corr[0].set_ylabel('Spike density', fontsize=fontsize)
        ax_corr[0].set_title('Field overlap=%0.2f' % (overlap), fontsize=titlesize)
        ax_corr[0].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_corr[0].spines['top'].set_visible(False)
        ax_corr[0].spines['left'].set_visible(False)
        ax_corr[0].spines['right'].set_visible(False)

        ax_corr_filterAB = ax_corr[0].twinx()
        ax_corr_filterAB.plot(edmAB, signalAB, c='k', linewidth=1, alpha=0.7)
        middle = np.sign(peaktimeAB) * 0.0025
        ax_corr_filterAB.plot([peaktimeAB, middle], [peaksignalAB * 1.05, peaksignalAB * 1.05], c='k', alpha=0.7)
        ax_corr_filterAB.scatter(middle, peaksignalAB * 1.05, c='k', s=16, alpha=0.7)
        ax_corr_filterAB.annotate('A->B', xy=(0.05, 0.9), xycoords='axes fraction', size=fontsize,
                                  bbox=dict(boxstyle='square', fc='w'))
        ax_corr_filterAB.axis('off')

        ax_corr[1].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='gray', alpha=0.5, label='B->A')
        ax_corr[1].plot([0, 0], [0, binsBAnorm.max() * 0.85], c='k', alpha=0.7)
        ax_corr[1].set_xticks([-0.15, 0, 0.15])
        ax_corr[1].set_yticks([])
        ax_corr[1].set_xlabel('Time lag (s)', fontsize=fontsize)
        ax_corr[1].set_ylabel('Spike density', fontsize=fontsize)
        ax_corr[1].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_corr[1].spines['top'].set_visible(False)
        ax_corr[1].spines['left'].set_visible(False)
        ax_corr[1].spines['right'].set_visible(False)

        ax_corr_filterBA = ax_corr[1].twinx()
        ax_corr_filterBA.plot(edmBA, signalBA, c='k', linewidth=1, alpha=0.7)
        middle = np.sign(peaktimeBA) * 0.0025
        ax_corr_filterBA.plot([peaktimeBA, middle], [peaksignalBA * 1.05, peaksignalBA * 1.05], c='k', alpha=0.7)
        ax_corr_filterBA.scatter(middle, peaksignalBA * 1.05, c='k', s=16, alpha=0.7)
        ax_corr_filterBA.annotate('B->A', xy=(0.05, 0.9), xycoords='axes fraction', size=fontsize,
                                  bbox=dict(boxstyle='square', fc='w'))
        ax_corr_filterBA.axis('off')

        if save_dir:
            overlap_tag = 'high' if overlap > 0.5 else 'low'
            fig_corr.tight_layout()
            fig_corr.savefig(os.path.join(save_dir, 'examples_correlogram', 'corr_%s_%d.png' % (overlap_tag, nrow)),
                             dpi=300)
            plt.close()

        # Plot exin
        concated = np.concatenate([binsABnorm, binsBAnorm])
        fig_exin, ax_exin = plt.subplots(2, 1, figsize=(3*figl/2, figl*2), sharex=True)

        ax_exin[0].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', alpha=0.5, label='A->B')
        ax_exin[0].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='slategray', alpha=0.5, label='B->A')
        ax_exin[0].plot([0, 0], [0, concated.max()], c='k')
        ax_exin[0].plot([0, 0], [0, concated.min()], c='k')
        ax_exin[0].set_ylabel('Spike density', fontsize=fontsize)
        ax_exin[0].set_title('Intrinsicity=%0.2f' % (intrinsicity), fontsize=titlesize)


        ax_exin[1].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', alpha=0.5, label='A->B')
        ax_exin[1].bar(edmBA, np.flip(binsBAnorm), width=edmBA[1] - edmBA[0], color='slategray', alpha=0.5,
                       label='Flipped B->A')
        ax_exin[1].plot([0, 0], [0, concated.max()], c='k')
        ax_exin[1].plot([0, 0], [0, concated.min()], c='k')
        ax_exin[1].set_xlabel('Time lag (s)', fontsize=fontsize)
        ax_exin[1].set_ylabel('Spike density', fontsize=fontsize)
        ax_exin[1].set_title('Extrinsicity=%0.2f' % (extrinsicity), fontsize=titlesize)
        ax_exin[1].set_xticks([-0.15, 0, 0.15])
        for ax_each in ax_exin:
            ax_each.set_yticks([])
            ax_each.tick_params(axis='both', which='major', labelsize=ticksize)
            ax_each.legend(fontsize=fontsize - 2, handlelength=1, labelspacing=0.2, handletextpad=0.2, borderpad=0.1)
            ax_each.spines['top'].set_visible(False)
            ax_each.spines['left'].set_visible(False)
            ax_each.spines['right'].set_visible(False)

        exin_tag = 'extrinsic' if overlap_ratio > 0 else 'inrinsic'
        if save_dir:
            fig_exin.tight_layout()
            fig_exin.savefig(os.path.join(save_dir, 'examples_exintrinsicity', 'exin_%s_%d.png' % (exin_tag, nrow)),
                             dpi=dpi)
            plt.close()


def plot_pair_single_angles_analysis(expdf, save_dir):
    stat_fn = 'fig5_pairsingle_angles.txt'
    stat_record(stat_fn, True)
    figw = total_figw*0.9/2
    figh = total_figw*0.9/4.5

    figsize = (figw, figh*3)
    fig, ax = plt.subplots(3, 3, figsize=figsize, sharey='row', sharex='row')


    figsize = (figw, figh*1.2)
    fig_test, ax_test = plt.subplots(1, 3, figsize=figsize, sharey='row', sharex='row')


    selected_cmap = cm.cividis
    adiff_dict = dict(CA1=[], CA2=[], CA3=[])
    Nthreshs = np.arange(0, 1001, 100)
    caid = 0
    for ca, cadf in expdf.groupby('ca'):
        cadf = cadf.reset_index(drop=True)

        all_fmeans = []
        all_fpairs = []
        all_nspks = []
        for rowi in range(cadf.shape[0]):
            f1, f2, fp = cadf.loc[rowi, ['fieldangle1_mlm', 'fieldangle2_mlm', 'fieldangle_pair_mlm']]
            nspks = cadf.loc[rowi, 'num_spikes_pair']
            f_mean = circmean(f1, f2)

            all_fmeans.append(f_mean)
            all_fpairs.append(fp)
            all_nspks.append(nspks)

        all_fmeans = np.array(all_fmeans)
        all_fpairs = np.array(all_fpairs)
        all_nspks = np.array(all_nspks)
        ax[0, caid].scatter(all_fmeans, all_fpairs, s=2, c=ca_c[ca], marker='.')
        ax[0, caid].set_title(ca, fontsize=titlesize)
        ax[0, caid].set_yticks([0, 2 * np.pi])
        ax[0, caid].tick_params(labelsize=fontsize)
        ax[0, caid].set_yticklabels(['0', '$2\pi$'])
        ax[0, caid].spines['top'].set_visible(False)
        ax[0, caid].spines['right'].set_visible(False)


        for thresh in Nthreshs:
            mask = (~np.isnan(all_fmeans)) & (~np.isnan(all_fpairs) & (all_nspks > thresh))
            adiff = cdiff(all_fmeans[mask], all_fpairs[mask])
            adiff_dict[ca].append(adiff)
            color = selected_cmap(thresh / Nthreshs.max())

            alpha_ax, den = circular_density_1d(adiff, 8 * np.pi, 100, bound=(-np.pi, np.pi))
            ax[1, caid].plot(alpha_ax, den, c=color, linewidth=0.75)

            abs_adiff = np.abs(adiff)
            absbins, absedges = np.histogram(abs_adiff, bins=np.linspace(0, np.pi, 50))
            cumsum_norm = np.cumsum(absbins) / np.sum(absbins)
            ax[2, caid].plot(midedges(absedges), cumsum_norm, c=color)

        ax[1, caid].set_yticks([])
        ax[1, caid].tick_params(labelsize=ticksize)
        ax[1, caid].spines['top'].set_visible(False)
        ax[1, caid].spines['left'].set_visible(False)
        ax[1, caid].spines['right'].set_visible(False)

        ax[2, caid].set_xticks([0, np.pi])
        ax[2, caid].set_xticklabels(['$0$', '$\pi$'])
        ax[2, caid].set_yticks([0, 1])
        ax[2, caid].tick_params(labelsize=ticksize)
        ax[2, caid].spines['top'].set_visible(False)
        ax[2, caid].spines['right'].set_visible(False)
        caid += 1

    ax[0, 0].set_ylabel(r'$\theta_{pair}$', fontsize=fontsize)
    ax[0, 1].set_xticks([0, 2 * np.pi])
    ax[0, 1].set_xticklabels(['0', '$2\pi$'])
    ax[0, 1].set_xlabel(r'$\theta_{mean}$', fontsize=fontsize)
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    plt.setp(ax[0, 2].get_xticklabels(), visible=False)

    ax[1, 0].set_ylabel('Normalized\ndensity', fontsize=fontsize)
    ax[1, 0].yaxis.labelpad = 15
    ax[1, 1].set_xticks([-np.pi, np.pi])
    ax[1, 1].set_xticklabels(['$-\pi$', '$\pi$'])
    ax[1, 1].set_xlabel(r'$d(\theta_{mean}, \theta_{pair})$', fontsize=fontsize)
    plt.setp(ax[1, 0].get_xticklabels(), visible=False)
    plt.setp(ax[1, 2].get_xticklabels(), visible=False)

    ax[2, 0].set_ylabel('Cumulative\ncounts', fontsize=fontsize)
    ax[2, 1].set_xticks([0, np.pi])
    ax[2, 1].set_xticklabels(['0', '$\pi$'])
    ax[2, 1].set_xlabel(r'$|d(\theta_{mean}, \theta_{pair})|$', fontsize=fontsize)
    plt.setp(ax[2, 0].get_xticklabels(), visible=False)
    plt.setp(ax[2, 2].get_xticklabels(), visible=False)



    p13_all, p23_all, p12_all = [], [], []
    p13_all_2, p23_all_2, p12_all_2 = [], [], []
    for thresh_id, thresh in enumerate(Nthreshs):
        ad1, ad2, ad3 = adiff_dict['CA1'][thresh_id], adiff_dict['CA2'][thresh_id], adiff_dict['CA3'][thresh_id]
        n1, n2, n3 = ad1.shape[0], ad2.shape[0], ad3.shape[0]
        f13, p13 = circ_ktest(ad1, ad3)
        f23, p23 = circ_ktest(ad2, ad3)
        f12, p12 = circ_ktest(ad1, ad2)
        z13, p13_2, _ = angular_dispersion_test(ad1, ad3)
        z23, p23_2, _ = angular_dispersion_test(ad2, ad3)
        z12, p12_2, _ = angular_dispersion_test(ad1, ad2)
        p13_all.append(p13)
        p23_all.append(p23)
        p12_all.append(p12)
        p13_all_2.append(p13_2)
        p23_all_2.append(p23_2)
        p12_all_2.append(p12_2)
        stat_record(stat_fn, False,
                    'Bartlett, CA13, thresh=%d, F_{(%d, %d)}=%0.2f, p=%0.4f' % (thresh, n1 - 1, n3 - 1, f13, p13))
        stat_record(stat_fn, False,
                    'Bartlett, CA23, thresh=%d, F_{(%d, %d)}=%0.2f, p=%0.4f' % (thresh, n2 - 1, n3 - 1, f23, p23))
        stat_record(stat_fn, False,
                    'Bartlett, CA12, thresh=%d, F_{(%d, %d)}=%0.2f, p=%0.4f' % (thresh, n1 - 1, n2 - 1, f12, p12))
        stat_record(stat_fn, False, 'U Mann-Whiney U test, CA13, thresh=%d, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % (
        thresh, z13, n1, n3, p13_2))
        stat_record(stat_fn, False, 'U Mann-Whiney U test, CA23, thresh=%d, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % (
        thresh, z23, n2, n3, p23_2))
        stat_record(stat_fn, False, 'U Mann-Whiney U test, CA12, thresh=%d, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % (
        thresh, z12, n1, n2, p12_2))

    color_bar, marker_bar = 'darkgoldenrod', 'x'
    color_rs, marker_rs = 'purple', '^'
    ms = 6
    ax_test[0].plot(Nthreshs, np.log10(p13_all), c=color_bar, linewidth=0.75, label='BAR')
    ax_test[1].plot(Nthreshs, np.log10(p23_all), c=color_bar, linewidth=0.75, label='BAR')
    ax_test[2].plot(Nthreshs, np.log10(p12_all), c=color_bar, linewidth=0.75, label='BAR')
    # ax_test[0].scatter(Nthreshs, np.log10(p13_all), c=color_bar, marker=marker_bar, s=ms, label='BAR')
    # ax_test[1].scatter(Nthreshs, np.log10(p23_all), c=color_bar, marker=marker_bar, s=ms, label='BAR')
    # ax_test[2].scatter(Nthreshs, np.log10(p12_all), c=color_bar, marker=marker_bar, s=ms, label='BAR')
    ax_test[0].plot(Nthreshs, np.log10(p13_all_2), c=color_rs, linewidth=0.75, label='RS')
    ax_test[1].plot(Nthreshs, np.log10(p23_all_2), c=color_rs, linewidth=0.75, label='RS')
    ax_test[2].plot(Nthreshs, np.log10(p12_all_2), c=color_rs, linewidth=0.75, label='RS')
    # ax_test[0].scatter(Nthreshs, np.log10(p13_all_2), c=color_rs, marker=marker_rs, s=ms, label='RS')
    # ax_test[1].scatter(Nthreshs, np.log10(p23_all_2), c=color_rs, marker=marker_rs, s=ms, label='RS')
    # ax_test[2].scatter(Nthreshs, np.log10(p12_all_2), c=color_rs, marker=marker_rs, s=ms, label='RS')


    for ax_each in ax_test:
        ax_each.plot([Nthreshs.min(), Nthreshs.max()], [np.log10(0.05), np.log10(0.05)], c='k', label='p=0.05', linewidth=0.75)
        ax_each.set_xticks([0, 500, 1000])
        ax_each.tick_params(labelsize=fontsize)
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    customlegend(ax_test[1], handlelength=0.5, linewidth=1.2, fontsize=legendsize-2, bbox_to_anchor=[0.2, 0.5], loc='lower left')

    ax_test[0].set_title('CA1-3', fontsize=titlesize)
    ax_test[1].set_title('CA2-3', fontsize=titlesize)
    ax_test[2].set_title('CA1-2', fontsize=titlesize)
    ax_test[0].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_test[1].set_xticks([0, 500, 1000])
    ax_test[1].set_xticklabels(['0', '', '1000'])
    ax_test[1].set_xlabel('Spike count threshold', fontsize=fontsize)
    plt.setp(ax_test[0].get_xticklabels(), visible=False)
    plt.setp(ax_test[2].get_xticklabels(), visible=False)


    # Color bar

    sm = plt.cm.ScalarMappable(cmap=selected_cmap,
                               norm=plt.Normalize(vmin=Nthreshs.min(), vmax=Nthreshs.max()))
    fig_colorbar = plt.figure(figsize=(total_figw*0.2, figh*2))

    cbar_ax = fig_colorbar.add_axes([0.1, 0.1, 0.05, 0.6])
    cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0, 500, 1000])
    cb.ax.set_yticklabels(['0', '500', '1000'], rotation=90)
    cb.set_label('Spike count thresholds', fontsize=fontsize)


    # Saving

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=1)
    fig.savefig(os.path.join(save_dir, 'pair_single_angles.%s'%(figext)), dpi=dpi)

    fig_test.tight_layout()
    fig_test.subplots_adjust(wspace=0.2, hspace=1)
    fig_test.savefig(os.path.join(save_dir, 'pair_single_angles_test.%s'%(figext)), dpi=dpi)

    fig_colorbar.savefig(os.path.join(save_dir, 'pair_single_angles_colorbar.%s'%(figext)), dpi=dpi)


if __name__ == '__main__':
    usephase = False
    save_dir = 'result_plots/pair_fields/'

    expdf = load_pickle('results/exp/pair_field/pairfield_df_latest.pickle')
    # plot_pair_correlation(expdf, save_dir)
    plot_exintrinsic(expdf, save_dir, usephase)
    # plot_exintrinsic_pair_correlation(expdf, save_dir, usephase)
    # plot_double_circular_fit(expdf, save_dir)
    # plot_firing_rate(expdf, save_dir)

    # omniplot_pairfields(expdf, save_dir=save_dir)
    # plot_kld(expdf, save_dir=save_dir)
    # plot_pair_single_angles_analysis(expdf, save_dir)
    # plot_pairangle_similarity_analysis(expdf, save_dir=os.path.join(save_dir, 'pairangle_similarity'), usephase=usephase)
    # plot_correlogram_exin(expdf, save_dir)

