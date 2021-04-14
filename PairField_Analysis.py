import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from glob import glob
from os.path import join
from matplotlib import cm

from pycircstat.tests import watson_williams, rayleigh
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import vonmises, pearsonr, chi2_contingency, chi2, ttest_ind, spearmanr, chisquare, \
    ttest_1samp, linregress, ks_2samp, binom_test
from scipy.interpolate import interp1d

from common.linear_circular_r import rcc
from common.utils import load_pickle, sigtext, stat_record, p2str, wwtable2text
from common.comput_utils import dist_overlap, normsig, append_extrinsicity, linear_circular_gauss_density, \
    find_pair_times, \
    IndataProcessor, check_border, window_shuffle_gen, \
    compute_straightness, calc_kld, window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, segment_passes, pair_diff, \
    check_border_sim, midedges, linear_gauss_density, circ_ktest, angular_dispersion_test, circular_density_1d, \
    fisherexact, ranksums, shiftcyc_full2half, ci_vonmise
from common.script_wrappers import DirectionalityStatsByThresh, PrecessionFilter
from common.visualization import plot_correlogram, plot_dcf_phi_histograms, plot_marginal_slices, customlegend
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw


figext = 'png'
# figext = 'eps'

maxspcounts = 201
spcount_step = 10
exin_margin = 0.05
# marquan = 0.25
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
        fig.savefig(join(save_dir, 'exp_firingrates.png'))
        fig_2d.tight_layout()
        fig_2d.savefig(join(save_dir, 'exp_corate-ratediff.png'))


def omniplot_pairfields(expdf, save_dir=None):
    linew = 0.75

    # Initialization of stats
    stat_fn = 'fig4_pair_directionality.txt'
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
    spike_threshs = np.arange(0, maxspcounts, spcount_step)
    stats_getter = DirectionalityStatsByThresh('num_spikes_pair', 'rate_R_pvalp', 'rate_Rp')
    chipval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
    rspval_dict = dict(CA1_be=[], CA2_be=[], CA3_be=[], all_CA13=[], all_CA23=[], all_CA12=[])
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

        # Binomial test for all fields
        signum_all, n_all = sdict_all['shift_signum'][0], sdict_all['n'][0]
        p_binom = binom_test(signum_all, n_all, p=0.05, alternative='greater')
        stat_txt = '%s, Binomial test, greater than p=0.05, %d/%d, p%s'%(ca, signum_all, n_all, p2str(p_binom))
        stat_record(stat_fn, False, stat_txt)
        ax[1, caid].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize, color=ca_c[ca])



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
                chi_stat, chi_pborder, chi_dof = 0, 1, 0

            frac_b, signum_b, nonsignum_b = cad_b['sigfrac_shift'][idx], cad_b['shift_signum'][idx], cad_b['shift_nonsignum'][idx]
            frac_nb, signum_nb, nonsignum_nb = cad_nb['sigfrac_shift'][idx], cad_nb['shift_signum'][idx], cad_nb['shift_nonsignum'][idx]
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
        ax[0, i].set_xticks([0, 100, 200])
        ax[0, i].set_xticklabels(['']*3)
        ax[1, i].set_xticks([0, 100, 200])
        ax[1, i].set_xticklabels(['0', '100', '200'])
    ax_frac[0].set_xticks([0, 100, 200])
    ax_frac[0].set_xticklabels(['']*3)
    ax_frac[1].set_xticks([0, 100, 200])
    ax_frac[1].set_xticklabels(['0', '100', '200'])

    # yticks
    for i in range(3):
        yticks = np.arange(0, 0.81, 0.2)
        ax_directR[i].set_ylim(0, 0.8)
        ax_directR[i].set_yticks(yticks)
        ax_directR[i].set_yticks(np.arange(0, 0.81, 0.1), minor=True)
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
    fig.savefig(join(save_dir, 'exp_pair_directionality.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'exp_pair_directionality.eps'), dpi=dpi)
    fig_pval.savefig(join(save_dir, 'exp_pair_TestSignificance.png'), dpi=dpi)

def plot_kld(expdf, save_dir=None):
    stat_fn = 'fig5_kld.txt'
    stat_record(stat_fn, True)

    # Filtering
    expdf['border'] = expdf['border1'] | expdf['border2']

    kld_key = 'kld'

    figw = total_figw*0.9/2
    figh = total_figw*0.9/4.5

    # # # KLD by threshold
    Nthreshs = np.arange(0, maxspcounts, spcount_step)
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
    # plt.setp(ax_kld_thresh[0].get_xticklabels(), visible=False)
    # ax_kld_thresh[1].set_xticks([0, 500, 1000])
    # ax_kld_thresh[1].set_xticklabels(['0', '', '1000'])
    # plt.setp(ax_kld_thresh[2].get_xticklabels(), visible=False)

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

            stattxt = "KLD by threshold=%d, CA13, %s: Mann-Whitney U test, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % (thresh, bordertxt[borderid], n1, n3, z13, p2str(p13)))

            stattxt = "KLD by threshold=%d, CA23, %s: Mann-Whitney U test, U(N_{CA2}=%d, N_{CA3}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % (thresh, bordertxt[borderid], n2, n3, z23, p2str(p23)))

            stattxt = "KLD by threshold=%d, CA12, %s: Mann-Whitney U test, U(N_{CA1}=%d, N_{CA2}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % (thresh, bordertxt[borderid], n1, n2, z12, p2str(p12)))

    # Border effect by CA
    for ca in ['CA1', 'CA2', 'CA3']:
        for idx, thresh in enumerate(Nthreshs):
            kld_b, kld_nb = kld_thresh_border_dict[ca][idx], kld_thresh_nonborder_dict[ca][idx]
            n_b, n_nd = kld_b.shape[0], kld_nb.shape[0]
            z_nnb, p_nnb = ranksums(kld_b, kld_nb)
            stattxt = "KLD by threshold=%d, %s, BorderEffect: Mann-Whitney U test, U(N_{border}=%d, N_{nonborder}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % (thresh, ca, n_b, n_nd, z_nnb, p2str(p_nnb)))
    # Save
    fig_kld_thresh.tight_layout()
    fig_kld_thresh.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_kld_thresh.savefig(join(save_dir, 'exp_kld_thresh.%s'%(figext)), dpi=dpi)

def plot_pair_single_angles_analysis(expdf, save_dir):
    stat_fn = 'fig5_pairsingle_angles.txt'
    stat_record(stat_fn, True)
    figw = total_figw*0.9/2
    figh = total_figw*0.9/4.5

    figsize = (figw, figh*3)
    fig, ax = plt.subplots(3, 3, figsize=figsize, sharey='row', sharex='row')


    figsize = (figw, figh*1.2)
    fig_test, ax_test = plt.subplots(1, 3, figsize=figsize, sharey='row', sharex='row')


    selected_cmap = cm.cool
    adiff_dict = dict(CA1=[], CA2=[], CA3=[])
    Nthreshs = np.arange(0, maxspcounts, spcount_step)
    caid = 0
    for ca, cadf in expdf.groupby('ca'):
        cadf = cadf.reset_index(drop=True)

        all_fmeans = []
        all_fpairs = []
        all_nspks = []
        for rowi in range(cadf.shape[0]):
            f1, f2, fp = cadf.loc[rowi, ['rate_angle1', 'rate_angle2', 'rate_anglep']]
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
            ax[2, caid].plot(midedges(absedges), cumsum_norm, c=color, linewidth=0.75)

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
                    'Bartlett, CA13, thresh=%d, F(%d, %d)=%0.2f, p%s' % (thresh, n1 - 1, n3 - 1, f13, p2str(p13)))
        stat_record(stat_fn, False,
                    'Bartlett, CA23, thresh=%d, F(%d, %d)=%0.2f, p%s' % (thresh, n2 - 1, n3 - 1, f23, p2str(p23)))
        stat_record(stat_fn, False,
                    'Bartlett, CA12, thresh=%d, F(%d, %d)=%0.2f, p%s' % (thresh, n1 - 1, n2 - 1, f12, p2str(p12)))
        stat_record(stat_fn, False, 'U Mann-Whiney U test, CA13, thresh=%d, U(N_{CA1}=%d,N_{CA3}=%d)=%0.2f, p%s )' % (
            thresh, n1, n3, z13, p2str(p13_2)))
        stat_record(stat_fn, False, 'U Mann-Whiney U test, CA23, thresh=%d, U(N_{CA2}=%d,N_{CA3}=%d)=%0.2f, p%s )' % (
            thresh, n2, n3, z23, p2str(p23_2)))
        stat_record(stat_fn, False, 'U Mann-Whiney U test, CA12, thresh=%d, U(N_{CA1}=%d,N_{CA2}=%d)=%0.2f, p%s )' % (
            thresh, n1, n2, z12, p2str(p12_2)))

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
        # ax_each.set_xticks([0, 300])
        ax_each.tick_params(labelsize=fontsize)
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    customlegend(ax_test[1], handlelength=0.5, linewidth=1.2, fontsize=legendsize-2, bbox_to_anchor=[0.2, 0.5], loc='lower left')

    ax_test[0].set_title('CA1-3', fontsize=titlesize)
    ax_test[1].set_title('CA2-3', fontsize=titlesize)
    ax_test[2].set_title('CA1-2', fontsize=titlesize)
    ax_test[0].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_test[1].set_xticks([0, 100, 200])
    ax_test[1].set_xticklabels(['0', '', '200'])
    ax_test[1].set_xlabel('Spike count threshold', fontsize=fontsize)
    plt.setp(ax_test[0].get_xticklabels(), visible=False)
    plt.setp(ax_test[2].get_xticklabels(), visible=False)


    # Color bar

    sm = plt.cm.ScalarMappable(cmap=selected_cmap,
                               norm=plt.Normalize(vmin=Nthreshs.min(), vmax=Nthreshs.max()))
    fig_colorbar = plt.figure(figsize=(total_figw*0.2, figh*2))

    cbar_ax = fig_colorbar.add_axes([0.1, 0.1, 0.05, 0.6])
    cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0, 100, 200])
    cb.ax.set_yticklabels(['0', '', '200'], rotation=90)
    cb.set_label('Spike count thresholds', fontsize=fontsize)


    # Saving

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=1)
    fig.savefig(join(save_dir, 'pair_single_angles.%s'%(figext)), dpi=dpi)

    fig_test.tight_layout()
    fig_test.subplots_adjust(wspace=0.2, hspace=1)
    fig_test.savefig(join(save_dir, 'pair_single_angles_test.%s'%(figext)), dpi=dpi)

    fig_colorbar.savefig(join(save_dir, 'pair_single_angles_colorbar.%s'%(figext)), dpi=dpi)


def plot_pair_correlation(expdf, save_dir=None):


    figl = total_figw*0.5/3
    linew = 0.75
    stat_fn = 'fig5_paircorr.txt'
    stat_record(stat_fn, True)
    fig, ax = plt.subplots(2, 3, figsize=(figl*3, figl*2.35), sharey='row')

    markersize = 1
    for caid, (ca, cadf) in enumerate(expdf.groupby('ca')):
        # A->B
        ax[0, caid], x, y, regress = plot_correlogram(ax=ax[0, caid], df=cadf, tag=ca, direct='A->B', color=ca_c[ca], alpha=1,
                                                      markersize=markersize, linew=linew)
        ax[0, caid].set_title(ca, fontsize=titlesize)
        nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
        stat_record(stat_fn, False, '%s A->B, y = %0.2fx + %0.2f, r(%d)=%0.2f, p%s' % \
                    (ca, regress['aopt'] * 2 * np.pi, regress['phi0'], nsamples, regress['rho'], p2str(regress['p'])))

        # B->A
        ax[1, caid], x, y, regress = plot_correlogram(ax=ax[1, caid], df=cadf, tag=ca, direct='B->A', color=ca_c[ca], alpha=1,
                                                      markersize=markersize, linew=linew)

        nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
        stat_record(stat_fn, False, '%s B->A, y = %0.2fx + %0.2f, r(%d))=%0.2f, p%s' % \
                    (ca, regress['aopt'] * 2 * np.pi, regress['phi0'], nsamples, regress['rho'], p2str(regress['p'])))


    for ax_each in ax.ravel():
        ax_each.tick_params(labelsize=ticksize)
        ax_each.spines["top"].set_visible(False)
        ax_each.spines["right"].set_visible(False)
        ax_each.set_xticks([0, 0.5, 1])
        ax_each.set_xticklabels(['', '', ''])

    ax[1, 1].set_xticks([0, 0.5, 1])
    ax[1, 1].set_xticklabels(['0', '', '1'])
    plt.setp(ax[1, 0].get_xticklabels(), visible=False)
    plt.setp(ax[0, 2].get_xticklabels(), visible=False)
    ax[1, 1].set_xlabel('Field overlap', fontsize=fontsize)

    ax[1, 2].set_xticks([0, 0.3, 0.5 , 1])
    ax[1, 2].set_xticklabels(['', '0.3', '' , ''])

    fig.text(0.01, 0.35, 'Phase shift (rad)', rotation=90, fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(join(save_dir, 'exp_paircorr.%s'%(figext)), dpi=dpi)


def plot_ALL_examples_correlogram(expdf):
    plot_dir = join('result_plots', 'examplesALL', 'exin_correlograms_passes')
    os.makedirs(plot_dir, exist_ok=True)

    for sid in range(expdf.shape[0]):
        print('Correlogram examples %d/%d'%(sid, expdf.shape[0]))
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
            ydumpool1 = xdumpool * 2*np.pi * rccpool_m1 + rccpool_c1
            ydumpool2 = xdumpool * 2*np.pi * rccpool_m2 + rccpool_c2
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
            ax[0, 0].set_title('Straightrank=%0.2f, Chunk=%d, %s'%(straightrank, chunked, direction))


            # Precession
            ax[0, 1].scatter(dsp1, phasesp1, c='r')
            ax[0, 1].scatter(dsp2, phasesp2, c='b')
            maxd = max(1, dsp1.max(), dsp2.max())
            xdum = np.linspace(0, maxd, 10)
            ydum1 = xdum * 2*np.pi * rcc_m1 + rcc_c1
            ydum2 = xdum * 2*np.pi * rcc_m2 + rcc_c2
            ax[0, 1].plot(xdum, ydum1, c='r')
            ax[0, 1].plot(xdum, ydum1+2*np.pi, c='r')
            ax[0, 1].plot(xdum, ydum1-2*np.pi, c='r')
            ax[0, 1].plot(xdum, ydum2, c='b')
            ax[0, 1].plot(xdum, ydum2+2*np.pi, c='b')
            ax[0, 1].plot(xdum, ydum2-2*np.pi, c='b')
            ax[0, 1].set_xlim(0, maxd)
            ax[0, 1].set_ylim(-np.pi, 2*np.pi)
            ax[0, 1].set_yticks([-np.pi, 0, np.pi, 2*np.pi])
            ax[0, 1].set_yticklabels(['$-\pi$', '0', '$\pi$', '$2\pi$'])
            ax[0, 1].set_title('Spcount, A=%d, B=%d'%(pass_nspikes1, pass_nspikes2))

            # Theta cycle and phase
            ax[0, 2].plot(wavet1, theta1)
            theta_range = theta1.max() - theta1.min()
            axphase = ax[0, 2].twinx()
            axphase.plot(wavet1, phase1, c='orange', alpha=0.5)
            # axphase.set_ylim(phase1.min()*2, phase1.max()*2)
            ax[0, 2].eventplot(tsptheta1, lineoffsets=theta1.max()*1.1, linelengths=theta_range*0.05, colors='r')
            ax[0, 2].eventplot(tsptheta2, lineoffsets=theta1.max()*1.1-theta_range*0.05, linelengths=theta_range*0.05, colors='b')

            # Correlograms
            ax[1, 1].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A->B')
            ax[1, 1].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='slategray', label='B->A')
            ax[1, 1].annotate('In.=%0.2f' % (intrinsicity), xy=(0.025, 0.7), xycoords='axes fraction', size=titlesize, color='b')


            ax[1, 0].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A->B')
            ax[1, 0].bar(edmBA, np.flip(binsBAnorm), width=edmBA[1] - edmBA[0], color='slategray',
                         label='Flipped B->A')
            ax[1, 0].annotate('Ex.=%0.2f' % (extrinsicity), xy=(0.58, 0.7), xycoords='axes fraction', size=titlesize, color='r')
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
            ax[1, 2].plot(xdumpool, ydumpool1+2*np.pi, c='r')
            ax[1, 2].plot(xdumpool, ydumpool1-2*np.pi, c='r')
            ax[1, 2].plot(xdumpool, ydumpool2, c='b')
            ax[1, 2].plot(xdumpool, ydumpool2+2*np.pi, c='b')
            ax[1, 2].plot(xdumpool, ydumpool2-2*np.pi, c='b')
            ax[1, 2].set_xlim(0, maxdpool)
            ax[1, 2].set_ylim(-np.pi, 2*np.pi)
            ax[1, 2].set_yticks([-np.pi, 0, np.pi, 2*np.pi])
            ax[1, 2].set_yticklabels(['$-\pi$', '0', '$\pi$', '$2\pi$'])
            ax[1, 2].set_title(direction)

            fig.tight_layout()
            exintag = 'ex' if overlap_ratio>0 else 'in'
            precesstag = 'precess' if precess_exist else 'nonprecess'
            figname = '%s_%d-%d_%s_%s_%s.png'%(ca, pair_id, nprecess, exintag, precesstag, direction)
            fig.savefig(join(plot_dir, figname), dpi=200, facecolor='white')
            plt.close(fig)

def plot_example_correlogram(expdf, save_dir=None):

    selected_ids = [11, 30, 88, 278]
    plot_types = {11:'corr',  30:'corr', 88:'exin', 278:'exin'}
    plot_axidxs = {11:0,  30:1, 88:0, 278:1}

    figl = total_figw*0.5/3  # Consistent with pair correlation
    fig_corr, ax_corr = plt.subplots(2, 2, figsize=(figl*3, figl*2.35), sharex=True, sharey=True)

    figl = total_figw/2/3
    fig_exin, ax_exin = plt.subplots(2, 2, figsize=(figl*3, figl*2.75), sharex=True, sharey=True)


    for sid in selected_ids:
        # Get data
        overlap, overlap_ratio = expdf.loc[sid, ['overlap', 'overlap_ratio']]
        intrinsicity, extrinsicity, pair_id = expdf.loc[sid, ['overlap_plus', 'overlap_minus', 'pair_id']]
        lagAB, lagBA, corrAB, corrBA = expdf.loc[sid, ['phaselag_AB', 'phaselag_BA', 'corr_info_AB', 'corr_info_BA']]

        plottype = plot_types[sid]
        plotaxidx = plot_axidxs[sid]

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
        if plottype == 'corr':

            ax_corr[0, plotaxidx].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='0.7', label='A->B')

            ax_corr[0, plotaxidx].set_yticks([])
            ax_corr[0, plotaxidx].set_title('Field overlap=%0.2f\n'% (overlap) + r'$A\rightarrow B$' , fontsize=legendsize)
            ax_corr[0, plotaxidx].tick_params(axis='both', which='major', labelsize=ticksize)


            marker = '<' if peaktimeAB > 0 else ">"
            ax_corr_filterAB = ax_corr[0, plotaxidx].twinx()
            ax_corr_filterAB.plot(edmAB, signalAB, c='k', linewidth=1)
            middle = np.sign(peaktimeAB) * 0.0025
            ax_corr_filterAB.plot([peaktimeAB, middle], [peaksignalAB * 1.05, peaksignalAB * 1.05], c='k')
            ax_corr_filterAB.scatter(middle, peaksignalAB * 1.05, c='k', s=16, marker=marker)
            ax_corr_filterAB.set_ylim(np.min(signalAB), np.max(signalAB)*1.2)
            ax_corr_filterAB.axis('off')

            ax_corr[1, plotaxidx].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='0.7', label='B->A')
            ax_corr[1, plotaxidx].set_xticks([-0.15, 0, 0.15])
            ax_corr[1, plotaxidx].set_yticks([])
            ax_corr[1, plotaxidx].set_title(r'$B\rightarrow A$', fontsize=titlesize)
            ax_corr[1, plotaxidx].set_xlabel('Time lag (s)', fontsize=fontsize)
            ax_corr[1, plotaxidx].tick_params(axis='both', which='major', labelsize=ticksize)

            marker = '<' if peaktimeBA > 0 else ">"
            ax_corr_filterBA = ax_corr[1, plotaxidx].twinx()
            ax_corr_filterBA.plot(edmBA, signalBA, c='k', linewidth=1)
            middle = np.sign(peaktimeBA) * 0.0025
            ax_corr_filterBA.plot([peaktimeBA, middle], [peaksignalBA * 1.05, peaksignalBA * 1.05], c='k')
            ax_corr_filterBA.scatter(middle, peaksignalBA * 1.05, c='k', s=16, marker=marker)
            ax_corr_filterBA.set_ylim(np.min(signalBA), np.max(signalBA)*1.2)
            ax_corr_filterBA.axis('off')



        # Plot Extrinsic/Intrinsic examples
        if plottype == 'exin':
            concated = np.concatenate([binsABnorm, binsBAnorm])

            ax_exin[0, plotaxidx].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A->B')
            ax_exin[0, plotaxidx].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='slategray', label='B->A')
            # ax_exin[0, plotaxidx].plot([0, 0], [0, concated.max()], c='k')
            # ax_exin[0, plotaxidx].plot([0, 0], [0, concated.min()], c='k')
            ax_exin[0, plotaxidx].set_title('Intrinsicity=%0.2f' % (intrinsicity), fontsize=legendsize)


            ax_exin[1, plotaxidx].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A->B')
            ax_exin[1, plotaxidx].bar(edmBA, np.flip(binsBAnorm), width=edmBA[1] - edmBA[0], color='slategray',
                           label='Flipped B->A')
            # ax_exin[1, plotaxidx].plot([0, 0], [0, concated.max()], c='k')
            # ax_exin[1, plotaxidx].plot([0, 0], [0, concated.min()], c='k')
            ax_exin[1, plotaxidx].set_xlabel('Time lag (s)', fontsize=fontsize)
            ax_exin[1, plotaxidx].set_title('Extrinsicity=%0.2f' % (extrinsicity), fontsize=legendsize)
            ax_exin[1, plotaxidx].set_xticks([-0.15, 0, 0.15])

    # Correlogram
    for ax_each in ax_corr.ravel():
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['left'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
    fig_corr.text(0.01, 0.25, 'Spike probability', rotation=90, fontsize=fontsize)

    fig_corr.tight_layout()
    fig_corr.subplots_adjust(hspace=0.6)
    fig_corr.savefig(join(save_dir, 'examples_correlogram.%s' % (figext)), dpi=dpi)


    # Ex/intrinsic
    for ax_each in ax_exin.ravel():
        ax_each.set_yticks([])
        ax_each.tick_params(axis='both', which='major', labelsize=ticksize)
        customlegend(ax_each, fontsize=legendsize)
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['left'].set_visible(False)
        ax_each.spines['right'].set_visible(False)
    fig_exin.text(0.01, 0.35, 'Spike probability', rotation=90, fontsize=fontsize)
    fig_exin.tight_layout()
    fig_exin.savefig(join(save_dir, 'examples_exintrinsicity.%s'% (figext)), dpi=dpi)

def plot_example_exin(expdf, save_dir):
    def plot_exin_pass_eg(ax, expdf, pairid, passid, color_A, color_B, ms, lw, direction):

        xyval1, xyval2, precessdf = expdf.loc[pairid, ['xyval1', 'xyval2', 'precess_dfp']]
        fieldcoor1, fieldcoor2 = expdf.loc[pairid, ['com1', 'com2']]

        x, y, wave_t, wave_theta, wave_phase = precessdf.loc[passid, ['x', 'y', 'wave_t1', 'wave_theta1', 'wave_phase1']]
        xsp1, ysp1, xsp2, ysp2 = precessdf.loc[passid, ['spike1x', 'spike1y', 'spike2x', 'spike2y']]

        tsp1, dsp1, phasesp1, rcc_m1, rcc_c1 = precessdf.loc[passid, ['tsp_withtheta1', 'dsp1', 'phasesp1', 'rcc_m1', 'rcc_c1']]
        tsp2, dsp2, phasesp2, rcc_m2, rcc_c2 = precessdf.loc[passid, ['tsp_withtheta2', 'dsp2', 'phasesp2', 'rcc_m2', 'rcc_c2']]

        min_wavet = wave_t.min()
        wave_t = wave_t - min_wavet
        tsp1 = tsp1 - min_wavet
        tsp2 = tsp2 - min_wavet

        # Trajectory
        allx, ally = np.concatenate([xyval1[:, 0], xyval2[:, 0]]), np.concatenate([xyval1[:, 1], xyval2[:, 1]])
        max_xyl = max(allx.max()-allx.min(), ally.max()-ally.min())
        ax_minx, ax_miny = allx.min(), ally.min()
        ax[0].plot(xyval1[:, 0], xyval1[:, 1], c=color_A, linewidth=lw, label='A')
        ax[0].scatter(fieldcoor1[0], fieldcoor1[1], c=color_A, marker='^', s=16)

        ax[0].plot(xyval2[:, 0], xyval2[:, 1], c=color_B, linewidth=lw, label='B')
        ax[0].scatter(fieldcoor2[0], fieldcoor2[1], c=color_B, marker='^', s=16)
        ax[0].plot(x, y, c='gray', linewidth=lw)
        ax[0].scatter(x[0], y[0], c='k', s=16, marker='x')
        ax[0].scatter(xsp1, ysp1, c=color_A, s=ms, marker='.', zorder=2.5)
        ax[0].scatter(xsp2, ysp2, c=color_B, s=ms, marker='.', zorder=2.5)
        lim_mar = 4
        ax[0].set_xlim(ax_minx-lim_mar, ax_minx+max_xyl+lim_mar)
        ax[0].set_ylim(ax_miny-lim_mar, min(ax_miny+max_xyl, ally.max())+lim_mar)
        ax[0].plot([ax_minx+max_xyl*0.7, ax_minx+max_xyl*0.7+5], [ax_miny+max_xyl*0.85, ax_miny+max_xyl*0.85], linewidth=lw, c='k', marker='|')
        ax[0].text(ax_minx+max_xyl*0.7+7, ax_miny+max_xyl*0.75, '5cm', fontsize=legendsize)
        ax[0].axis('off')

        # Theta power
        theta_l = wave_theta.max() - wave_theta.min()
        max_l = wave_theta.max()
        ax[1].plot(wave_t, wave_theta, c='gray', linewidth=lw)
        ax[1].eventplot(tsp1, lineoffsets=max_l+1*max_l, linelengths=max_l*0.75, linewidths=lw, colors=color_A)
        ax[1].eventplot(tsp2, lineoffsets=max_l+0.25*max_l, linelengths=max_l*0.75, linewidths=lw, colors=color_B)
        boundidx_all = np.where(np.diff(wave_phase) < -np.pi)[0]
        #     for i, boundidx in enumerate(boundidx_all):
        #         ax[1].axvline(wave_t[boundidx], linewidth=0.25, color='k')
        for i in range(boundidx_all.shape[0]-1):
            if ((i % 2) != 0):
                continue
            ax[1].axvspan(wave_t[boundidx_all[i]], wave_t[boundidx_all[i+1]], color='0.9', zorder=0.1)

        ax[1].set_ylim(wave_theta.min(), max_l+1.5*max_l)
        #     ax[1].set_xlabel('Time (s)', fontsize=fontsize)
        ax[1].set_xticks([])
        ax[1].tick_params(labelsize=ticksize)
        ax[1].set_yticks([])
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)

        # Precession
        precessmask = (precessdf['direction']==direction)
        dsp1 = np.concatenate(precessdf.loc[precessmask, 'dsp1'].to_list())
        dsp2 = np.concatenate(precessdf.loc[precessmask,'dsp2'].to_list())
        phasesp1 = np.concatenate(precessdf.loc[precessmask,'phasesp1'].to_list())
        phasesp2 = np.concatenate(precessdf.loc[precessmask,'phasesp2'].to_list())
        abound = (-2, 0)
        regress1, regress2 = rcc(dsp1, phasesp1, abound=abound), rcc(dsp2, phasesp2, abound=abound)
        rcc_m1, rcc_c1 = regress1['aopt'], regress1['phi0']
        rcc_m2, rcc_c2 = regress2['aopt'], regress2['phi0']

        maxdsp = max(1, max(dsp1.max(), dsp2.max()))
        xdum = np.linspace(0, maxdsp, 10)
        ydum1 = rcc_m1 * 2 * np.pi * xdum + rcc_c1
        ydum2 = rcc_m2 * 2 * np.pi * xdum + rcc_c2
        ax[2].scatter(dsp1, phasesp1, c=color_A, s=ms, marker='.')
        ax[2].scatter(dsp1, phasesp1+2*np.pi, c=color_A, s=ms, marker='.')
        ax[2].plot(xdum, ydum1, c=color_A, linewidth=lw)
        ax[2].plot(xdum, ydum1+2*np.pi, c=color_A, linewidth=lw)
        ax[2].scatter(dsp2, phasesp2, c=color_B, s=ms, marker='.')
        ax[2].scatter(dsp2, phasesp2+2*np.pi, c=color_B, s=ms, marker='.')
        ax[2].plot(xdum, ydum2, c=color_B, linewidth=lw)
        ax[2].plot(xdum, ydum2+2*np.pi, c=color_B, linewidth=lw)
        ax[2].set_ylim(-np.pi, 3*np.pi)
        ax[2].set_xticks([0, 1])
        ax[2].set_xlabel('Position', fontsize=legendsize, labelpad=-10)
        ax[2].tick_params(labelsize=ticksize)



    def plot_exin_correlogram_eg(ax, expdf, pairid):
        overlap, overlap_ratio = expdf.loc[pairid, ['overlap', 'overlap_ratio']]
        intrinsicity, extrinsicity, pair_id = expdf.loc[pairid, ['overlap_plus', 'overlap_minus', 'pair_id']]
        lagAB, lagBA, corrAB, corrBA = expdf.loc[pairid, ['phaselag_AB', 'phaselag_BA', 'corr_info_AB', 'corr_info_BA']]
        binsAB, edgesAB, signalAB, alphasAB = corrAB[0], corrAB[1], corrAB[2], corrAB[4]
        binsBA, edgesBA, signalBA, alphasBA = corrBA[0], corrBA[1], corrBA[2], corrBA[4]

        # Phase
        #     corr_info_AB_phase, corr_info_BA_phase = expdf.loc[pairid, ['corr_info_AB_phase', 'corr_info_BA_phase']]
        #     edgesAB, edgesBA = np.linspace(-np.pi, np.pi, 36), np.linspace(-np.pi, np.pi, 36)
        #     _, binsAB = corr_info_AB_phase
        #     _, binsBA = corr_info_BA_phase

        edmAB, edmBA = midedges(edgesAB), midedges(edgesBA)
        binsABnorm, binsBAnorm = binsAB / binsAB.sum(), binsBA / binsBA.sum()
        ax[0].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A>B')
        ax[0].bar(edmBA, binsBAnorm, width=edmBA[1] - edmBA[0], color='slategray', label='B>A')
        ax[0].annotate('In.=%0.2f' % (intrinsicity), xy=(0.025, 1.05), xycoords='axes fraction', size=legendsize, color='b')
        customlegend(ax[0], fontsize=legendsize, bbox_to_anchor=(0.5, 0.5), loc='lower left')

        ax[1].bar(edmAB, binsABnorm, width=edmAB[1] - edmAB[0], color='tan', label='A>B')
        ax[1].bar(edmBA, np.flip(binsBAnorm), width=edmBA[1] - edmBA[0], color='slategray', label='Flipped B>A')
        ax[1].annotate('Ex.=%0.2f' % (extrinsicity), xy=(0.58, 1.05), xycoords='axes fraction', size=legendsize, color='r')
        customlegend(ax[1], fontsize=legendsize, bbox_to_anchor=(0.01, 0.5), loc='lower left')

        for ax_each in ax.ravel():
            ax_each.set_xticks([-0.15, -0.1, 0.05, 0, 0.05, 0.1, 0.15])
            ax_each.set_xticklabels(['', '-0.1', '', '0', '', '0.1', ''])
            #         ax_each.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
            #         ax_each.set_xticklabels(['', '-0.2', '', '0', '', '0.2', ''])
            ax_each.set_xlabel('Time lag (s)', fontsize=legendsize)
            ax_each.set_yticks([])
            ax_each.spines['left'].set_visible(False)
            ax_each.tick_params(axis='both', which='major', labelsize=ticksize)
    fig_exin = plt.figure(figsize=(total_figw, total_figw/2))

    axw = 0.9/4
    axh1 = 0.9/3.5
    axh2 = 0.9/2/3.5

    btwgp_xgap = -0.05
    biggp_xoffset = -0.025
    biggp_xgap = -0.025

    axtraj_y = 1-axh1-0.1
    axtraj_xoffset, axtraj_yoffset = 0.135, 0.075
    axtraj_wm, axtraj_hm = 0.5, 0.6
    axes_traj = [fig_exin.add_axes([0+axtraj_xoffset-biggp_xgap, axtraj_y+axtraj_yoffset, axw*axtraj_wm, axh1*axtraj_hm]),
                 fig_exin.add_axes([0.25+axtraj_xoffset+btwgp_xgap-biggp_xgap, axtraj_y+axtraj_yoffset, axw*axtraj_wm, axh1*axtraj_hm]),
                 fig_exin.add_axes([0.5+axtraj_xoffset+biggp_xoffset+biggp_xgap, axtraj_y+axtraj_yoffset, axw*axtraj_wm, axh1*axtraj_hm]),
                 fig_exin.add_axes([0.75+axtraj_xoffset+btwgp_xgap+biggp_xoffset+biggp_xgap, axtraj_y+axtraj_yoffset, axw*axtraj_wm, axh1*axtraj_hm])]


    axtheta_y = axtraj_y - axh2
    axtheta_xoffset, axtheta_yoffset = 0.085, 0.09
    axtheta_wm, axtheta_hm = 0.9, 0.65
    axes_theta = [fig_exin.add_axes([0+axtheta_xoffset-biggp_xgap, axtheta_y+axtheta_yoffset, axw*axtheta_wm, axh2*axtheta_hm]),
                  fig_exin.add_axes([0.25+axtheta_xoffset+btwgp_xgap-biggp_xgap, axtheta_y+axtheta_yoffset, axw*axtheta_wm, axh2*axtheta_hm]),
                  fig_exin.add_axes([0.5+axtheta_xoffset+biggp_xoffset+biggp_xgap, axtheta_y+axtheta_yoffset, axw*axtheta_wm, axh2*axtheta_hm]),
                  fig_exin.add_axes([0.75+axtheta_xoffset+btwgp_xgap+biggp_xoffset+biggp_xgap, axtheta_y+axtheta_yoffset, axw*axtheta_wm, axh2*axtheta_hm])]

    axprecess_y = axtheta_y - axh1
    axprecess_xoffset, axprecess_yoffset = 0.11, 0.13
    axprecess_wm, axprecess_hm = 0.65, 0.6
    axes_precess = [fig_exin.add_axes([0+axprecess_xoffset-biggp_xgap, axprecess_y+axprecess_yoffset, axw*axprecess_wm, axh1*axprecess_hm]),
                    fig_exin.add_axes([0.25+axprecess_xoffset+btwgp_xgap-biggp_xgap, axprecess_y+axprecess_yoffset, axw*axprecess_wm, axh1*axprecess_hm]),
                    fig_exin.add_axes([0.5+axprecess_xoffset+biggp_xoffset+biggp_xgap, axprecess_y+axprecess_yoffset, axw*axprecess_wm, axh1*axprecess_hm]),
                    fig_exin.add_axes([0.75+axprecess_xoffset+btwgp_xgap+biggp_xoffset+biggp_xgap, axprecess_y+axprecess_yoffset, axw*axprecess_wm, axh1*axprecess_hm])]

    axcorr_y = axprecess_y - axh1
    axcorr_xoffset, axcorr_yoffset = 0.10, 0.145
    axcorr_wm, axcorr_hm = 0.75, 0.5
    axes_corr = [fig_exin.add_axes([0+axcorr_xoffset-biggp_xgap, axcorr_y+axcorr_yoffset, axw*axcorr_wm, axh1*axcorr_hm]),
                 fig_exin.add_axes([0.25+axcorr_xoffset+btwgp_xgap-biggp_xgap, axcorr_y+axcorr_yoffset, axw*axcorr_wm, axh1*axcorr_hm]),
                 fig_exin.add_axes([0.5+axcorr_xoffset+biggp_xoffset+biggp_xgap, axcorr_y+axcorr_yoffset, axw*axcorr_wm, axh1*axcorr_hm]),
                 fig_exin.add_axes([0.75+axcorr_xoffset+btwgp_xgap+biggp_xoffset+biggp_xgap, axcorr_y+axcorr_yoffset, axw*axcorr_wm, axh1*axcorr_hm])]

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



    ax_exin[0, 0].set_title('A>B', fontsize=legendsize)
    ax_exin[0, 1].set_title('B>A', fontsize=legendsize)
    ax_exin[0, 2].set_title('A>B', fontsize=legendsize)
    ax_exin[0, 3].set_title('B>A', fontsize=legendsize)
    customlegend(ax_exin[0, 0], fontsize=legendsize, bbox_to_anchor=(0.8, 0), loc='lower left')
    fig_exin.text(0.18, 0.96, 'Pair#%d (Extrinsic)'%(ex_pairid), fontsize=fontsize)
    fig_exin.text(0.655, 0.96, 'Pair#%d (Intrinsic)'%(in_pairid), fontsize=fontsize)


    for i in range(1, 4):
        ax_exin[2, i].set_yticks([0, np.pi, 2*np.pi])
        ax_exin[2, i].set_yticklabels(['']*3)
    ax_exin[2, 0].set_yticks([0, np.pi, 2*np.pi])
    ax_exin[2, 0].set_yticklabels(['0', '', '$2\pi$'])

    fig_exin.text(0.035, 0.4, 'Phase\n(rad)', rotation=90, fontsize=fontsize)
    fig_exin.text(0.035, 0.05, 'Normalized\nspike count', rotation=90, fontsize=fontsize)


    fig_exin.savefig(join(save_dir, 'examples_exintrinsicity.%s'%(figext)), dpi=dpi)


def plot_exintrinsic(expdf, save_dir):


    figl = total_figw/2/3

    stat_fn = 'fig6_exintrinsic.txt'
    stat_record(stat_fn, True)

    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'
    # Filtering
    smallerdf = expdf[(~expdf[ratio_key].isna())]
    # smallerdf['Extrincicity'] = smallerdf[ratio_key].apply(lambda x: 'Extrinsic' if x > 0 else 'Intrinsic')

    fig_exin, ax_exin = plt.subplots(2, 3, figsize=(figl*3, figl*2.75), sharey='row', sharex='row')
    ms = 1
    contindf_dict = {}

    for caid, (ca, cadf) in enumerate(smallerdf.groupby('ca')):

        # corr_overlap_ratio = cadf[ratio_key].to_numpy()
        # cadf = cadf[np.abs(corr_overlap_ratio) >= exin_margin].reset_index(drop=True)

        corr_overlap = cadf[oplus_key].to_numpy()
        corr_overlap_flip = cadf[ominus_key].to_numpy()
        corr_overlap_ratio = cadf[ratio_key].to_numpy()

        # 1-sample chisquare test
        n_ex = np.sum(corr_overlap_ratio > 0)
        n_in = np.sum(corr_overlap_ratio <= 0)
        n_total = n_ex + n_in
        contindf_dict[ca] = [n_ex, n_in]
        chistat, pchi = chisquare([n_ex, n_in])
        stattxt = 'Oneway Fraction,  %s, %d/%d, \chi^2(%d, N=%d)=%0.2f, p%s'
        stat_record(stat_fn, False, stattxt % (ca, n_ex, n_total, 1, n_total, chistat, p2str(pchi)))

        # 1-sample t test
        mean_ratio = np.mean(corr_overlap_ratio)
        ttest_stat, p_1d1samp = ttest_1samp(corr_overlap_ratio, 0)
        stattxt = 'Ex-In, %s, mean=%0.4f, t(%d)=%0.2f, p%s'
        stat_record(stat_fn, False, stattxt % (ca, mean_ratio, n_total-1, ttest_stat, p2str(p_1d1samp)))

        # Plot scatter 2d
        ax_exin[0, caid].scatter(corr_overlap_flip, corr_overlap, s=ms, c=ca_c[ca], marker='.')
        ax_exin[0, caid].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
        ax_exin[0, caid].annotate('%0.2f'%(n_ex/n_total), xy=(0.05, 0.17), xycoords='axes fraction', size=legendsize, color='r')
        ax_exin[0, caid].annotate('p%s'%(p2str(pchi)), xy=(0.05, 0.025), xycoords='axes fraction', size=legendsize)
        ax_exin[0, 1].set_xlabel('Extrinsicity', fontsize=fontsize)
        ax_exin[0, caid].set_yticks([0, 1])
        ax_exin[0, caid].set_xlim(0, 1)
        ax_exin[0, caid].set_ylim(0, 1)
        ax_exin[0, caid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_exin[0, caid].set_title(ca, fontsize=titlesize)


        # Plot 1d histogram
        edges = np.linspace(-1, 1, 75)
        width = edges[1]-edges[0]
        (bins, _, _) = ax_exin[1, caid].hist(corr_overlap_ratio, bins=edges, color=ca_c[ca], density=True,
                                             histtype='stepfilled')
        ax_exin[1, caid].plot([mean_ratio, mean_ratio], [0, bins.max()], c='k', linewidth=0.75)
        ax_exin[1, caid].annotate('$\mu=$'+'%0.3f'%(mean_ratio), xy=(0.05, 0.85), xycoords='axes fraction', size=legendsize)
        ax_exin[1, caid].annotate('p%s'%(p2str(p_1d1samp)), xy=(0.05, 0.65), xycoords='axes fraction', size=legendsize)
        ax_exin[1, caid].set_xticks([])
        ax_exin[1, caid].set_yticks([0, 0.1/width] )
        ax_exin[1, caid].set_yticklabels(['0', '0.1'])
        ax_exin[1, caid].tick_params( labelsize=ticksize)
        ax_exin[1, caid].set_xlim(-0.5, 0.5)
        ax_exin[1, caid].set_ylim(0, 0.2/width)




    ax_exin[0, 0].set_ylabel('Intrinsicity', fontsize=fontsize)
    ax_exin[1, 0].set_ylabel('Normalized count', fontsize=fontsize)
    ax_exin[1, 1].set_xlabel('Extrinsicity - Intrinsicity', fontsize=fontsize)
    ax_exin[0, 1].set_xticks([0, 1])
    ax_exin[0, 1].set_xticklabels(['0', '1'])
    ax_exin[1, 1].set_xticks([-0.4, 0, 0.4])
    ax_exin[1, 1].set_xticklabels(['-0.4', '0', '0.4'])
    for i in [0, 2]:
        plt.setp(ax_exin[0, i].get_xticklabels(), visible=False)
        plt.setp(ax_exin[1, i].get_xticklabels(), visible=False)

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

    zratio13, zp13 = ranksums(ratio_1, ratio_3)
    stat_record(stat_fn, False, 'Welch\'s t-test: CA1 vs CA3, t(%d)=%0.2f, p%s' % (n1 + n3 - 2, t_13, p2str(p_13)))
    stat_record(stat_fn, False, 'Welch\'s t-test: CA2 vs CA3, t(%d)=%0.2f, p%s' % (n2 + n3 - 2, t_23, p2str(p_23)))
    stat_record(stat_fn, False, 'Welch\'s t-test: CA1 vs CA2, t(%d)=%0.2f, p%s' % (n1 + n2 - 2, t_12, p2str(p_12)))
    stat_record(stat_fn, False, 'RS-test: CA1 vs CA3, U=%0.2f, p%s' % (zratio13, p2str(zp13)))

    contindf = pd.DataFrame(contindf_dict)
    contindf13, contindf23, contindf12 = contindf[['CA1', 'CA3']], contindf[['CA2', 'CA3']], contindf[['CA1', 'CA2']]
    n13, n23, n12 = np.sum(contindf13.to_numpy()), np.sum(contindf23.to_numpy()), np.sum(contindf12.to_numpy())
    fisherp13 = fisherexact(contindf13.to_numpy())
    fisherp23 = fisherexact(contindf23.to_numpy())
    fisherp12 = fisherexact(contindf12.to_numpy())
    chi13, pchi13, dof13, _ = chi2_contingency(contindf13.to_numpy())
    chi23, pchi23, dof23, _ = chi2_contingency(contindf23.to_numpy())
    chi12, pchi12, dof12, _ = chi2_contingency(contindf12.to_numpy())

    stattxt = "Two-way Ex-in Frac. CA1 vs CA3 \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
    stat_record(stat_fn, False, stattxt % (dof13, n13-2, chi13, p2str(pchi13), p2str(fisherp13)))
    stattxt = "Two-way Ex-in Frac. CA2 vs CA3 \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
    stat_record(stat_fn, False, stattxt % (dof23, n23-2, chi23, p2str(pchi23), p2str(fisherp23)))
    stattxt = "Two-way Ex-in Frac. CA1 vs CA2 \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
    stat_record(stat_fn, False, stattxt % (dof12, n12-2, chi12, p2str(pchi12), p2str(fisherp12)))

    fig_exin.tight_layout()
    fig_exin.savefig(join(save_dir, 'exp_exintrinsic.%s'%(figext)), dpi=dpi)


def plot_exintrinsic_concentration(expdf, save_dir=None):
    figl = total_figw/2/3

    stat_fn = 'fig6_concentration.txt'
    stat_record(stat_fn, True)
    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'

    fig_con, ax_con = plt.subplots(1, 3, figsize=(figl * 3, figl*2.75/2), sharey=True)  # 2.75/2 comes from other panels

    linew = 0.75

    for caid, (ca, cadf) in enumerate(expdf.groupby('ca')):


        exdf = cadf[(cadf[ratio_key] > 0) & (cadf['overlap'] < 0.3)].reset_index(drop=True)
        indf = cadf[(cadf[ratio_key] <= 0) & (cadf['overlap'] < 0.3)].reset_index(drop=True)

        phaselag_ex = np.concatenate([exdf['phaselag_AB'].to_numpy(), -exdf['phaselag_BA'].to_numpy()])
        phaselag_in = np.concatenate([indf['phaselag_AB'].to_numpy(), -indf['phaselag_BA'].to_numpy()])

        phaselag_ex = phaselag_ex[~np.isnan(phaselag_ex)]
        phaselag_in = phaselag_in[~np.isnan(phaselag_in)]


        # Phaselag histogram
        yex_ax, yex_den = circular_density_1d(phaselag_ex, 4 * np.pi, 50, (-np.pi, np.pi))
        yin_ax, yin_den = circular_density_1d(phaselag_in, 4 * np.pi, 50, (-np.pi, np.pi))
        F_k, pval_k = circ_ktest(phaselag_ex, phaselag_in)
        stat_record(stat_fn, False, '%s Ex-in concentration difference F(%d, %d)=%0.2f, p%s'% \
                    (ca, phaselag_ex.shape[0], phaselag_in.shape[0], F_k, p2str(pval_k)))

        ax_con[caid].plot(yex_ax, yex_den, c='r', label='ex', linewidth=linew)
        ax_con[caid].plot(yin_ax, yin_den, c='b', label='in', linewidth=linew)
        ax_con[caid].set_xticks([-np.pi, 0, np.pi])
        ax_con[caid].set_yticks([0, 0.05])
        ax_con[caid].set_xticklabels(['$-\pi$', '0', '$\pi$'])

        ax_con[caid].tick_params(labelsize=ticksize)


        ax_con[caid].annotate('Bartlett\'s\np%s'%(p2str(pval_k)), xy=(0.05, 0.7), xycoords='axes fraction', size=legendsize)

        ax_con[caid].annotate('Ex.', xy=(0.4, 0.5), xycoords='axes fraction', size=legendsize, color='r')
        ax_con[caid].annotate('In.', xy=(0.7, 0.1), xycoords='axes fraction', size=legendsize, color='b')

    # ax_con[0].set_ylabel('Density of pairs', fontsize=fontsize, loc='bottom')
    ax_con[1].set_xlabel('Phase shift (rad)', fontsize=fontsize)
    fig_con.text(0, 0.2, 'Density of pairs', rotation=90, fontsize=fontsize)
    for ax_each in ax_con.ravel():
        ax_each.set_ylim(0, 0.08)
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)


    fig_con.tight_layout()
    fig_con.subplots_adjust(right=0.99)
    fig_con.savefig(join(save_dir, 'exp_exintrinsic_concentration.%s'%(figext)), dpi=dpi)



def plot_pairangle_similarity_analysis(expdf, save_dir=None, nap_thresh=1):
    stat_fn = 'fig6_pair_exin_simdisim.txt'
    stat_record(stat_fn, True)
    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'
    # anglekey1, anglekey2 = 'precess_angle1', 'precess_angle2'
    anglekey1, anglekey2 = 'rate_angle1', 'rate_angle2'
    figl = total_figw/2/3

    fig_1d, ax_1d = plt.subplots(1, 3, figsize=(figl*4, figl*2), sharex='row', sharey='row')
    fig_1dca, ax_1dca = plt.subplots(1, 2, figsize=(figl*4, figl*2.5), sharex=True, sharey=True)
    fig_2d, ax_2d = plt.subplots(2, 3, figsize=(figl*3, figl*2.5), sharex='row', sharey='row')

    ratiosimdict = dict()
    ratiodisimdict = dict()
    df_exin_dict = dict(ca=[], sim=[], exnum=[], innum=[])

    print(expdf[~expdf['overlap_ratio'].isna()]['overlap_ratio'].abs().describe())

    ms = 1
    frac = 2
    for caid, (ca, cadf) in enumerate(expdf.groupby(by='ca')):

        # cadf = cadf[(~cadf[ratio_key].isna()) & (~cadf[anglekey1].isna()) & (~cadf[anglekey2].isna()) & \
        #             (cadf['numpass_at_precess1'] >= nap_thresh) & (cadf['numpass_at_precess2'] >= nap_thresh )
        #             ].reset_index(drop=True)
        cadf = cadf[(~cadf[ratio_key].isna()) & (~cadf[anglekey1].isna()) & (~cadf[anglekey2].isna())].reset_index(drop=True)

        # oratio = cadf[ratio_key]
        # cadf = cadf[np.abs(oratio) >= exin_margin].reset_index(drop=True)

        s1, s2 = cadf[anglekey1], cadf[anglekey2]
        oratio, ominus, oplus = cadf[ratio_key], cadf[ominus_key], cadf[oplus_key]

        circdiff = np.abs(cdiff(s1, s2))
        simidx = np.where(circdiff < (np.pi/frac))[0]
        disimidx = np.where(circdiff > (np.pi - np.pi/frac))[0]
        orsim, ordisim = oratio[simidx], oratio[disimidx]
        exnumsim, exnumdisim = np.sum(orsim>0), np.sum(ordisim>0)
        innumsim, innumdisim = np.sum(orsim<=0), np.sum(ordisim<=0)
        exfracsim, exfracdisim = exnumsim/(exnumsim+innumsim), exnumdisim/(exnumdisim+innumdisim)
        ratiosimdict[ca] = orsim
        ratiodisimdict[ca] = ordisim

        # Ex-in for sim & dissim (ax_1d)
        obins_sim, oedges_sim = np.histogram(orsim, bins=30)
        oedges_sim_m = midedges(oedges_sim)
        obins_disim, oedges_disim = np.histogram(ordisim, bins=30)
        oedges_disim_m = midedges(oedges_disim)
        ax_1d[caid].plot(oedges_sim_m, np.cumsum(obins_sim) / np.sum(obins_sim), label='sim')
        ax_1d[caid].plot(oedges_disim_m, np.cumsum(obins_disim) / np.sum(obins_disim), label='dissimilar')

        _, pval1d_rs = ranksums(orsim, ordisim)
        tval1d_t, pval1d_t = ttest_ind(orsim, ordisim, equal_var=False)
        ax_1d[caid].text(0, 0.2, 'sim-dissim\nrs=%0.4f\nt=%0.4f'%(pval1d_rs, pval1d_t), fontsize=legendsize)
        ax_1d[caid].tick_params(labelsize=ticksize)
        customlegend(ax_1d[caid], fontsize=legendsize, loc='upper left')

        # Ex-in for CA1, CA2, CA3 (ax_1dca)
        ax_1dca[0].plot(oedges_sim_m, np.cumsum(obins_sim) / np.sum(obins_sim), label=ca, c=ca_c[ca])
        ax_1dca[1].plot(oedges_disim_m, np.cumsum(obins_disim) / np.sum(obins_disim), label=ca, c=ca_c[ca])


        # 2D scatter of ex-intrinsicitiy for similar/dissimlar (ax_2d)
        chisq_exsim, p_exfracsim = chisquare([exnumsim, innumsim])
        chisq_exdissim, p_exfracdisim = chisquare([exnumdisim, innumdisim])
        stat_record(stat_fn, False, '1way chi, %s, similar, %d/%d=%0.2f, \chi^2(%d, N=%d)=%0.2f, p%s' % \
                    (ca, exnumsim, orsim.shape[0], exnumsim/orsim.shape[0], 1, orsim.shape[0], chisq_exsim, p2str(p_exfracsim)))
        stat_record(stat_fn, False, '1way shi, %s, dissimilar, %d/%d=%0.2f, \chi^2(%d, N=%d)$=%0.2f, p%s' % \
                    (ca, exnumdisim, ordisim.shape[0], exnumdisim/ordisim.shape[0], 1, ordisim.shape[0], chisq_exdissim, p2str(p_exfracdisim)))

        ax_2d[0, caid].scatter(ominus[simidx], oplus[simidx], s=ms, c=ca_c[ca], marker='.')
        ax_2d[0, caid].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
        ax_2d[0, caid].annotate('%0.2f'%(exfracsim), xy=(0.05, 0.17), xycoords='axes fraction', size=legendsize, color='r')
        ax_2d[0, caid].annotate('p%s'%(p2str(p_exfracsim)), xy=(0.05, 0.025), xycoords='axes fraction', size=legendsize)
        ax_2d[0, caid].set_yticks([0, 1])
        ax_2d[0, caid].set_xlim(0, 1)
        ax_2d[0, caid].set_ylim(0, 1)
        ax_2d[0, caid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax_2d[0, caid].set_title(ca, fontsize=titlesize)


        ax_2d[1, caid].scatter(ominus[disimidx], oplus[disimidx], s=ms, c=ca_c[ca], marker='.')
        ax_2d[1, caid].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
        ax_2d[1, caid].annotate('%0.2f'%(exfracdisim), xy=(0.05, 0.17), xycoords='axes fraction', size=legendsize, color='r')
        ax_2d[1, caid].annotate('p%s'%(p2str(p_exfracdisim)), xy=(0.05, 0.025), xycoords='axes fraction', size=legendsize)
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

    ax_2d[0, 1].set_xticks([0, 1])
    ax_2d[0, 0].set_ylabel('Similar\nIntrinsicity', fontsize=fontsize)
    ax_2d[0, 1].set_xlabel('\n', fontsize=fontsize)
    ax_2d[1, 0].set_ylabel('Dissimilar\nIntrinsicity', fontsize=fontsize)
    ax_2d[1, 1].set_xlabel('Extrinsicity', fontsize=fontsize)
    for i in [0, 2]:
        plt.setp(ax_2d[0, i].get_xticklabels(), visible=False)
        plt.setp(ax_2d[1, i].get_xticklabels(), visible=False)
    for ax_each in ax_2d.ravel():
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    # 2-way chisquare for ex-in fraction
    df_exin = pd.DataFrame(df_exin_dict)
    allcadfexin = []
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):  # Compare between sim and dissim
        dftmp = df_exin[df_exin['ca']==ca]
        dftmp.index = dftmp.sim
        dftmp = dftmp[['exnum', 'innum']]
        allcadfexin.append(dftmp)
        chitmp, pchitmp, doftmp, _ = chi2_contingency(dftmp)
        pfisher = fisherexact(dftmp.to_numpy())
        ntotal = dftmp['exnum'].sum() + dftmp['innum'].sum()
        stattxt = '2-way, %s, sim vs dissim, \chi^2(%d, N=%d)}=%0.2f, p%s. Fisher exact p%s'
        stat_record(stat_fn, False, stattxt % (ca, doftmp, ntotal, chitmp, p2str(pchitmp), p2str(pfisher)))
    from statsmodels.stats.contingency_tables import StratifiedTable
    statmodel_CA13table = StratifiedTable([allcadfexin[0], allcadfexin[2]])
    table_allca = allcadfexin[0] + allcadfexin[1] + allcadfexin[2]
    simexn, dissimexn = table_allca.loc['similar', 'exnum'], table_allca.loc['dissimilar', 'exnum']
    simtotaln, dissimtotaln = table_allca.sum(axis=1)
    pfish_allca = fisherexact(table_allca.to_numpy())
    stat_record(stat_fn, False, 'All brain sim (%d/%d) vs dissim (%d/%d) table Fisher exact p%s'%(simexn, simtotaln, dissimexn, dissimtotaln, p2str(pfish_allca)))

    for simidtmp, sim_label in enumerate(['similar', 'dissimilar']):
        for capairid, (calabel1, calabel2) in enumerate((('CA1', 'CA2'), ('CA2', 'CA3'), ('CA1', 'CA3'))):
            dftmp = df_exin[(df_exin['sim']==sim_label) &
                            ((df_exin['ca']==calabel1) | (df_exin['ca']==calabel2))][['exnum', 'innum']]
            chitmp, pchitmp, doftmp, _ = chi2_contingency(dftmp)
            ntotal = dftmp['exnum'].sum() + dftmp['innum'].sum()
            pfisher = fisherexact(dftmp.to_numpy())
            stattxt = '2-way, %s, %s vs %s, \chi^2(%d, n=%d)=%0.2f, p%s. Fisher exact p%s'
            stat_record(stat_fn, False, stattxt % (sim_label, calabel1, calabel2, doftmp, ntotal, chitmp, p2str(pchitmp), p2str(pfisher)))


    # Compare exin for CA1-3
    nsim1, nsim2, nsim3 = ratiosimdict['CA1'].shape[0], ratiosimdict['CA2'].shape[0], ratiosimdict['CA3'].shape[0]
    ndisim1, ndisim2, ndisim3 = ratiodisimdict['CA1'].shape[0], ratiodisimdict['CA2'].shape[0], ratiodisimdict['CA3'].shape[0]

    zsim13, psim13 = ranksums(ratiosimdict['CA1'], ratiosimdict['CA3'])
    zdisim13, pdisim13 = ranksums(ratiodisimdict['CA1'], ratiodisimdict['CA3'])
    zsim23, psim23 = ranksums(ratiosimdict['CA2'], ratiosimdict['CA3'])
    zdisim23, pdisim23 = ranksums(ratiodisimdict['CA2'], ratiodisimdict['CA3'])

    ax_1dca[0].text(0.1, 0.2, 'p13rs=%0.4f\np23rs=%0.4ff' % (psim13, psim23), fontsize=legendsize)
    ax_1dca[0].set_title('Similar', fontsize=fontsize)
    ax_1dca[0].set_xlabel('Intrinsic <-> Extrinsic', fontsize=fontsize)
    customlegend(ax_1dca[0], fontsize=legendsize)
    ax_1dca[1].text(0.1, 0.2, 'p13rs=%0.4f\np23rs=%0.4f' % (pdisim13, pdisim13), fontsize=legendsize)
    ax_1dca[1].set_title('Dissimilar', fontsize=fontsize)
    ax_1dca[1].set_xlabel('Intrinsic <-> Extrinsic', fontsize=fontsize)
    customlegend(ax_1dca[1], fontsize=legendsize)

    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Similar, CA1 vs CA3, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s'% (nsim1, nsim3, zsim13, p2str(psim13)))
    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Similar, CA2 vs CA3, U(N_{CA2}=%d, N_{CA3}=%d)=%0.2f, p%s'% (nsim2, nsim3, zsim23, p2str(psim23)))
    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Dissimilar, CA1 vs CA3, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s'% (ndisim1, ndisim3, zdisim13, p2str(pdisim13)))
    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Dissimilar, CA2 vs CA3, U(N_{CA2}=%d, N_{CA3}=%d)=%0.2f, p%s'% (ndisim2, ndisim3, zdisim23, p2str(pdisim23)))


    stat_record(stat_fn, False, statmodel_CA13table.summary().as_text())


    # Saving
    fig_1d.tight_layout()
    fig_1dca.tight_layout()
    fig_2d.tight_layout()

    for figext in ['png', 'eps']:
        fig_1d.savefig(join(save_dir, 'exp_simdissim_exintrinsic1d.%s'%(figext)), dpi=dpi)
        fig_1dca.savefig(join(save_dir, 'exp_simdissim_exintrinsic1d_ca123.%s'%(figext)), dpi=dpi)
        fig_2d.savefig(join(save_dir, 'exp_simdissim_exintrinsic2d.%s'%(figext)), dpi=dpi)


def FourCasesAnalysis(expdf, save_dir):
    def flip_direction(direction):
        if direction == "A->B":
            return "B->A"
        elif direction == "B->A":
            return "A->B"
        else:
            return direction
    nap_thresh = 1
    frac_field = 2
    frac_pass = 2
    anglekey1, anglekey2 = 'rate_angle1', 'rate_angle2'




    df = expdf[(~expdf[anglekey1].isna()) & (~expdf[anglekey2].isna()) & \
               #            ((expdf['numpass_at_precess1'] >= nap_thresh) & (expdf['numpass_at_precess2'] >= nap_thresh)) & \
               (~expdf['overlap_ratio'].isna())
               ].reset_index(drop=True)

    df['fanglediff'] = np.abs(cdiff(df[anglekey1].to_numpy(), df[anglekey2].to_numpy()))
    df['meanfangle'] = shiftcyc_full2half(circmean(df[[anglekey1, anglekey2]].to_numpy(), axis=1))


    allprecessdf_dict = dict(ca=[], pair_id=[], foverlap=[], overlap_ratio=[], fanglediff=[], meanfangle=[],
                             pairlagAB=[], pairlagBA=[], fangle1=[], fangle2=[],
                             passangle=[], nspikes=[], onset=[], slope=[], direction=[], precess_exist=[],

                             )


    for i in range(df.shape[0]):

        ca, pair_id, overlap_r, fad, mfa = df.loc[i, ['ca', 'pair_id', 'overlap_ratio', 'fanglediff', 'meanfangle']]
        foverlap = df.loc[i, 'overlap']
        precess_dfp = df.loc[i, 'precess_dfp']

        numprecess = precess_dfp.shape[0]
        if numprecess < 1:
            continue

        nspikes1 = precess_dfp['tsp1'].apply(lambda x : x.shape[0])
        nspikes2 = precess_dfp['tsp2'].apply(lambda x : x.shape[0])

        pos1, pos2 = df.loc[i, ['fieldcoor1', 'fieldcoor2']]
        posdiff = pos2 - pos1
        field_orient = np.angle(posdiff[0] + 1j * posdiff[1])
        absdiff = np.abs(cdiff(field_orient, mfa))
        ABalign = True if absdiff < np.pi/2 else False

        lagABtmp, lagBAtmp = df.loc[i, ['phaselag_AB', 'phaselag_BA']]
        angle1, angle2 = df.loc[i, [anglekey1, anglekey2]]

        allprecessdf_dict['ca'].extend([ca] * numprecess * 2)
        allprecessdf_dict['pair_id'].extend([pair_id] * numprecess * 2)
        allprecessdf_dict['foverlap'].extend([foverlap] * numprecess * 2)
        allprecessdf_dict['overlap_ratio'].extend([overlap_r] * numprecess * 2)
        allprecessdf_dict['fanglediff'].extend([fad] * numprecess * 2)
        allprecessdf_dict['meanfangle'].extend([mfa] * numprecess * 2)
        allprecessdf_dict['fangle1'].extend([angle1] * numprecess * 2)
        allprecessdf_dict['fangle2'].extend([angle2] * numprecess * 2)

        allprecessdf_dict['precess_exist'].extend(precess_dfp['precess_exist'].to_list() * 2)

        # Flipping AB if field A and field
        if ABalign:
            allprecessdf_dict['pairlagAB'].extend([lagABtmp] * numprecess * 2)
            allprecessdf_dict['pairlagBA'].extend([lagBAtmp] * numprecess * 2)
            allprecessdf_dict['direction'].extend(precess_dfp['direction'].to_list() * 2)
            allprecessdf_dict['onset'].extend(precess_dfp['rcc_c1'].to_list() + precess_dfp['rcc_c2'].to_list())
            allprecessdf_dict['slope'].extend(precess_dfp['rcc_m1'].to_list() + precess_dfp['rcc_m2'].to_list())
            allprecessdf_dict['passangle'].extend(precess_dfp['mean_anglesp1'].to_list() + precess_dfp['mean_anglesp2'].to_list())
            allprecessdf_dict['nspikes'].extend(nspikes1.to_list() + nspikes2.to_list())
            # allprecessdf_dict['precess_exist'].extend(precess_dfp['precess_exist1'].to_list() + precess_dfp['precess_exist2'].to_list())
        else:
            allprecessdf_dict['pairlagAB'].extend([-lagBAtmp] * numprecess * 2)
            allprecessdf_dict['pairlagBA'].extend([-lagABtmp] * numprecess * 2)
            allprecessdf_dict['direction'].extend(precess_dfp['direction'].apply(flip_direction).to_list() * 2)
            allprecessdf_dict['onset'].extend(precess_dfp['rcc_c2'].to_list() + precess_dfp['rcc_c1'].to_list())
            allprecessdf_dict['slope'].extend(precess_dfp['rcc_m2'].to_list() + precess_dfp['rcc_m1'].to_list())
            allprecessdf_dict['passangle'].extend(precess_dfp['mean_anglesp2'].to_list() + precess_dfp['mean_anglesp1'].to_list())
            allprecessdf_dict['nspikes'].extend(nspikes2.to_list() + nspikes1.to_list())
            # allprecessdf_dict['precess_exist'].extend(precess_dfp['precess_exist2'].to_list() + precess_dfp['precess_exist1'].to_list())

    allpdf = pd.DataFrame(allprecessdf_dict)
    allpdf = allpdf[allpdf['precess_exist']].reset_index(drop=True)

    # Case 1 & 2
    allpdf['simfield'] = allpdf['fanglediff'] < (np.pi/frac_field)
    allpdf['oppfield'] = allpdf['fanglediff'] > (np.pi - np.pi/frac_field)
    passfield_adiff = np.abs(cdiff(allpdf['passangle'], allpdf['meanfangle']))
    allpdf['case1'] = (passfield_adiff < (np.pi/frac_pass)) & allpdf['simfield']
    allpdf['case2'] = (passfield_adiff > (np.pi - np.pi/frac_pass)) & allpdf['simfield']

    # Case 3 & 4
    diff1tmp = np.abs(cdiff(allpdf['passangle'], allpdf['fangle1']))
    diff2tmp = np.abs(cdiff(allpdf['passangle'], allpdf['fangle2']))
    simto1_AtoB = (diff1tmp < diff2tmp) & (diff1tmp < (np.pi/2)) & (allpdf['direction'] == 'A->B') & allpdf['oppfield']
    simto1_BtoA = (diff1tmp >= diff2tmp) & (diff2tmp < (np.pi/2)) & (allpdf['direction'] == 'B->A') & allpdf['oppfield']
    simto2_AtoB = (diff1tmp >= diff2tmp) & (diff2tmp < (np.pi/2)) & (allpdf['direction'] == 'A->B') & allpdf['oppfield']
    simto2_BtoA = (diff1tmp < diff2tmp) & (diff1tmp < (np.pi/2)) & (allpdf['direction'] == 'B->A') & allpdf['oppfield']
    allpdf['case3'] = simto1_AtoB | simto1_BtoA
    allpdf['case4'] = simto2_AtoB | simto2_BtoA

    # marmask = (allpdf['overlap_ratio'] < -exin_margin) | (allpdf['overlap_ratio'] > exin_margin)
    # allpdf = allpdf[marmask].reset_index(drop=True)


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
    getcadf = lambda x, ca : x[x['ca']==ca][['nex', 'nin']]
    df_total = getcadf(df_exin, 'CA1') + getcadf(df_exin, 'CA2') + getcadf(df_exin, 'CA3')
    df_total['case'] = df_total.index
    df_total['ca'] = 'ALL'
    df_exinall = pd.concat([df_exin, df_total], axis=0)
    df_exinall['ntotal'] = df_exinall['nex'] + df_exinall['nin']
    df_exinall['exfrac'] = df_exinall['nex']/df_exinall['ntotal']

    # # # Plotting & Statistical tesing
    stat_fn = 'fig7_4case.txt'
    stat_record(stat_fn, True)
    fig_4c = plt.figure(figsize=(total_figw, total_figw/1.5))

    main_w = 1

    illu_w, illu_h = main_w/4, 1/3
    illu_y = 1-illu_h
    squeezex, squeezey = 0.075, 0.1
    xoffset = 0
    ax_illu = [fig_4c.add_axes([0+squeezex/2+xoffset, illu_y+squeezey/2, illu_w-squeezex, illu_h-squeezey]),
               fig_4c.add_axes([illu_w+squeezex/2+xoffset, illu_y+squeezey/2, illu_w-squeezex, illu_h-squeezey]),
               fig_4c.add_axes([illu_w*2+squeezex/2+xoffset, illu_y+squeezey/2, illu_w-squeezex, illu_h-squeezey]),
               fig_4c.add_axes([illu_w*3+squeezex/2+xoffset, illu_y+squeezey/2, illu_w-squeezex, illu_h-squeezey]),
               ]


    exin2d_w, exin2d_h = main_w*1.5/4, 2/3
    exin2d_y = 1-illu_h- exin2d_h
    squeezex, squeezey = 0.1, 0.2
    exin2d_xoffset, exin2d_yoffset = 0.025, 0.05
    ax_exin2d = fig_4c.add_axes([0+squeezex/2+exin2d_xoffset, exin2d_y+squeezey/2+exin2d_yoffset, exin2d_w-squeezex, exin2d_h-squeezey])

    exinfrac_w, exinfrac_h = main_w*2.5/4, 1/3
    exinfrac_y = 1-illu_h- exinfrac_h
    squeezex, squeezey = 0.1, 0.1
    exinfrac_xoffset, exinfrac_yoffset = 0.05, 0.05
    ax_exfrac = fig_4c.add_axes([exin2d_w+squeezex/2+exinfrac_xoffset, exinfrac_y+squeezey/2+exinfrac_yoffset, exinfrac_w-squeezex, exinfrac_h-squeezey])

    onset_w, onset_h = main_w*2.5/4, 1/3
    onset_y = 1-illu_h- exinfrac_h- onset_h
    squeezex, squeezey = 0.1, 0.1
    onset_xoffset, onset_yoffset = 0.05, 0.1
    ax_onset = fig_4c.add_axes([exin2d_w+squeezex/2+onset_xoffset, onset_y+squeezey/2+onset_yoffset, onset_w-squeezex, onset_h-squeezey])
    ax_onset.get_shared_x_axes().join(ax_onset, ax_exfrac)


    # # Plot case illustration
    ms=4
    t = np.linspace(0, 2*np.pi, 100)
    r, arrowl = 0.5, 0.25
    arrow_upshift = r+0.1
    text_upshift = r-0.3
    passtext_downshift = 0.4
    passarrowsize = 64
    pass_color = 'mediumblue'
    c1x, c1y, c2x, c2y = -0.4, 0, 0.4, 0
    circ1_x, circ1_y = r * np.sin(t) + c1x, r * np.cos(t) + c1y
    circ2_x, circ2_y = r * np.sin(t) + c2x, r * np.cos(t) + c2y
    for i in range(4):
        ax_illu[i].plot(circ1_x, circ1_y, c='k')
        ax_illu[i].plot(circ2_x, circ2_y, c='k')
        ax_illu[i].axis('off')
        ax_illu[i].set_xlim(-1.1, 1.1)
        ax_illu[i].set_ylim(-1.1, 1.1)
        ax_illu[i].annotate('A', xy=(0.25, 0.6), xycoords='axes fraction', fontsize=legendsize)
        ax_illu[i].annotate('B', xy=(0.65, 0.6), xycoords='axes fraction', fontsize=legendsize)
        ax_illu[i].annotate('Pass', xy=(0.9, 0.3), xycoords='axes fraction', fontsize=legendsize, color=pass_color)
        ax_illu[i].annotate('Case %d'%(i+1), xy=(0.3, 0.85), xycoords='axes fraction', fontsize=titlesize)
        ax_illu[i].plot([c1x-r-0.1, c2x+r+0.1], [c1y, c1y], c=pass_color)
        arrowsign = '>' if ((i==0) or (i==2)) else '<'
        ax_illu[i].scatter(c2x, c2y, marker=arrowsign, color=pass_color, s=passarrowsize)
    ax_illu[0].arrow(c1x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_illu[0].arrow(c2x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_illu[1].arrow(c1x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_illu[1].arrow(c2x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_illu[2].arrow(c1x-arrowl, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_illu[2].arrow(c2x+arrowl, c1y+arrow_upshift, dx=-arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_illu[3].arrow(c1x+arrowl, c1y+arrow_upshift, dx=-arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_illu[3].arrow(c2x-arrowl, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')


    # # Ex-in 2d
    for ca in ['CA1', 'CA2', 'CA3']:
        nex1, nin1 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==1), ['nex', 'nin']].iloc[0]
        nex2, nin2 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==2), ['nex', 'nin']].iloc[0]
        nex3, nin3 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==3), ['nex', 'nin']].iloc[0]
        nex4, nin4 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==4), ['nex', 'nin']].iloc[0]
        #     labeltxt = 'Pooled' if ca=='ALL' else ca
        ax_exin2d.scatter(nex1, nin1, c=ca_c[ca], marker='o', s=ms, label=ca)
        ax_exin2d.scatter(nex2, nin2, c=ca_c[ca], marker='o', s=ms)
        ax_exin2d.scatter(nex3, nin3, c=ca_c[ca], marker='o', s=ms)
        ax_exin2d.scatter(nex4, nin4, c=ca_c[ca], marker='o', s=ms)
        ax_exin2d.annotate('1', xy=(nex1, nin1), color=ca_c[ca], zorder=3.1, fontsize=legendsize)
        ax_exin2d.annotate('2', xy=(nex2, nin2), color=ca_c[ca], zorder=3.1, fontsize=legendsize)
        ax_exin2d.annotate('3', xy=(nex3, nin3), xytext=(0, -3), textcoords='offset points', color=ca_c[ca], zorder=3.1, fontsize=legendsize)
        ax_exin2d.annotate('4', xy=(nex4, nin4), xytext=(2, -3), textcoords='offset points', color=ca_c[ca], zorder=3.1, fontsize=legendsize)
    nexin = df_exin[['nex', 'nin']].to_numpy()
    ax_exin2d.plot([nexin.min()-5, nexin.max()+5], [nexin.min()-5, nexin.max()+5], c='k', linewidth=0.5)
    ax_exin2d.tick_params(labelsize=ticksize)
    ax_exin2d.set_xlim(df_exin['nex'].min()-5, df_exin['nex'].max()+5)
    ax_exin2d.set_ylim(df_exin['nin'].min()-5, df_exin['nin'].max()+5)
    ax_exin2d.set_xlabel('N(extrinsic)', fontsize=fontsize)
    ax_exin2d.set_ylabel('N(intrinsic)', fontsize=fontsize)
    customlegend(ax_exin2d, loc='upper left')

    # CA1 1 vs CA3 1
    nexCA1_1, ninCA1_1 = df_exinall[(df_exinall['ca']=='CA1') & (df_exinall['case']==1)][['nex', 'nin']].iloc[0]
    nexCA3_1, ninCA3_1 = df_exinall[(df_exinall['ca']=='CA3') & (df_exinall['case']==1)][['nex', 'nin']].iloc[0]
    p11_31 = fisherexact(np.array([[nexCA1_1, ninCA1_1], [nexCA3_1, ninCA3_1]]))
    stat_record(stat_fn, False, "Exin-Frac, Case 1, CA1 (Ex:In=%d:%d) vs CA3(Ex:In=%d:%d), Fisher\'s exact test p%s"%(nexCA1_1, ninCA1_1, nexCA3_1, ninCA3_1, p2str(p11_31)))
    ax_exin2d.annotate('p(  vs  )%s'% (p2str(p11_31)), xy=(0.3, 0.2), xycoords='axes fraction', size=legendsize)
    ax_exin2d.annotate('1', xy=(0.365, 0.2), xycoords='axes fraction', size=legendsize, color=ca_c['CA1'])
    ax_exin2d.annotate('1', xy=(0.5, 0.2), xycoords='axes fraction', size=legendsize, color=ca_c['CA3'])


    # CA1 2 vs CA3 2
    nexCA1_2, ninCA1_2 = df_exinall[(df_exinall['ca']=='CA1') & (df_exinall['case']==2)][['nex', 'nin']].iloc[0]
    nexCA3_2, ninCA3_2 = df_exinall[(df_exinall['ca']=='CA3') & (df_exinall['case']==2)][['nex', 'nin']].iloc[0]
    p12_32 = fisherexact(np.array([[nexCA1_2, ninCA1_2], [nexCA3_2, ninCA3_2]]))
    stat_record(stat_fn, False, "Exin-Frac, Case 2, CA1 (Ex:In=%d:%d) vs CA3(Ex:In=%d:%d), Fisher\'s exact test p%s"%(nexCA1_2, ninCA1_2, nexCA3_2, ninCA3_2, p2str(p12_32)))
    ax_exin2d.annotate('p(  vs  )%s'% (p2str(p12_32)), xy=(0.3, 0.125), xycoords='axes fraction', size=legendsize)
    ax_exin2d.annotate('2', xy=(0.365, 0.125), xycoords='axes fraction', size=legendsize, color=ca_c['CA1'])
    ax_exin2d.annotate('2', xy=(0.5, 0.125), xycoords='axes fraction', size=legendsize, color=ca_c['CA3'])

    # CA1 12 vs CA3 12
    nex112, nin112 = df_exinall[(df_exinall['ca']=='CA1') & ((df_exinall['case']==1)| (df_exinall['case']==2))][['nex', 'nin']].sum()
    nex312, nin312 = df_exinall[(df_exinall['ca']=='CA3') & ((df_exinall['case']==1)| (df_exinall['case']==2))][['nex', 'nin']].sum()
    p112_312 = fisherexact(np.array([[nex112, nin112], [nex312, nin312]]))
    stat_record(stat_fn, False, "Exin-Frac, Case 1+2, CA1 (Ex:In=%d:%d) vs CA3(Ex:In=%d:%d), Fisher\'s exact test p%s"%(nex112, nin112, nex312, nin312, p2str(p112_312)))
    ax_exin2d.annotate('p(       vs       )%s'% (p2str(p112_312)), xy=(0.3, 0.05), xycoords='axes fraction', size=legendsize)
    ax_exin2d.annotate('1+2', xy=(0.375, 0.05), xycoords='axes fraction', size=legendsize, color=ca_c['CA1'])
    ax_exin2d.annotate('1+2', xy=(0.625, 0.05), xycoords='axes fraction', size=legendsize, color=ca_c['CA3'])

    # # Ex frac
    ca_c['ALL'] = 'gray'
    box_offsets = np.array([-0.3, -0.1, 0.1, 0.3])
    boxw = np.diff(box_offsets).mean()*0.75
    ca_box_pos = dict(CA1=box_offsets+1, CA2=box_offsets + 2, CA3=box_offsets+3, ALL=box_offsets)
    for ca in ['CA1', 'CA2', 'CA3', 'ALL']:
        exfrac1 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==1), 'exfrac'].iloc[0]
        exfrac2 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==2), 'exfrac'].iloc[0]
        exfrac3 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==3), 'exfrac'].iloc[0]
        exfrac4 = df_exinall.loc[(df_exinall['ca']==ca) & (df_exinall['case']==4), 'exfrac'].iloc[0]
        ax_exfrac.bar(ca_box_pos[ca], [exfrac1, exfrac2, exfrac3, exfrac4], width=boxw, color=ca_c[ca])

        # Test 12
        dftmp12 = df_exinall[(df_exinall['ca']==ca) & ((df_exinall['case']==1) | (df_exinall['case']==2))][['nex', 'nin']]
        nexc1, ninc1 = dftmp12.loc[1, ['nex', 'nin']]
        nexc2, ninc2 = dftmp12.loc[2, ['nex', 'nin']]
        case12_boxmid = np.mean([ca_box_pos[ca][0], ca_box_pos[ca][1]])
        pval12 = fisherexact(dftmp12.to_numpy())
        stat_record(stat_fn, False, "Exin-Frac, %s, Case1(Ex:In=%d:%d) vs Case2(Ex:In=%d:%d), Fisher\'s exact test p%s"%(ca, nexc1, ninc1, nexc2, ninc2, p2str(pval12)))
        ax_exfrac.text(case12_boxmid-boxw*2, 0.8, 'p'+p2str(pval12), fontsize=legendsize)
        ax_exfrac.errorbar(x=case12_boxmid, y=0.75, xerr=boxw/2, c='k', capsize=2.5)


        # Test 12 vs 34
        dftmp34 = df_exinall[(df_exinall['ca']==ca) & ((df_exinall['case']==3) | (df_exinall['case']==4))][['nex', 'nin']]
        nex12, nin12 = dftmp12.sum()
        nex34, nin34 = dftmp34.sum()
        all_boxmid = np.mean(ca_box_pos[ca])
        pval1234 = fisherexact(np.array([[nex12, nin12], [nex34, nin34]]))
        stat_record(stat_fn, False, "Exin-Frac, %s, Case12(Ex:In=%d:%d) vs Case34(Ex:In=%d:%d), Fisher\'s exact test p%s"%(ca, nex12, nin12, nex34, nin34, p2str(pval1234)))
        ax_exfrac.text(all_boxmid-boxw*2, 1.05, 'p'+p2str(pval1234), fontsize=legendsize)
        ax_exfrac.errorbar(x=all_boxmid, y=1, xerr=boxw*1.5, c='k', capsize=2.5)


    ax_exfrac.set_ylim(0, 1.2)
    ax_exfrac.set_xticks(np.concatenate([ca_box_pos['CA1'], ca_box_pos['CA2'], ca_box_pos['CA3'], ca_box_pos['ALL']]))
    ax_exfrac.set_ylabel('Ex. Frac.', fontsize=fontsize)
    ax_exfrac.set_xticklabels(['']*16)

    # # Onset
    onset_dict = dict()
    case34_boxmid = np.mean([ca_box_pos[ca][2], ca_box_pos[ca][3]])
    for caseid in [1, 2, 3, 4]:
        for ca in ['CA1', 'CA2', 'CA3']:
            onset_dict[ca + str(caseid)] = allpdf[(allpdf['ca']==ca) & (allpdf['case%d'%caseid])]['onset'].to_numpy()
        onset_dict['ALL' + str(caseid)] = allpdf[allpdf['case%d'%caseid]]['onset'].to_numpy()

    for ca in ['CA1', 'CA2', 'CA3', 'ALL']:
        x_mean = np.array([circmean(onset_dict[ca + '%d'%i]) for i in range(1, 5)])
        x_tsd = np.array([np.std(onset_dict[ca + '%d'%i])/np.sqrt(onset_dict[ca + '%d'%i].shape[0]) for i in range(1, 5)])
        ax_onset.errorbar(x=ca_box_pos[ca], y=x_mean, yerr=x_tsd*1.96, fmt='_k', capsize=5, ecolor=ca_c[ca])

        # Test 12
        case12_boxmid = np.mean([ca_box_pos[ca][0], ca_box_pos[ca][1]])
        pval12, table12 = watson_williams(onset_dict['%s1'%(ca)], onset_dict['%s2'%(ca)])
        stat_record(stat_fn, False, 'Onset, %s, Case 1 vs Case 2, Watson-Williams test, %s'%(ca, wwtable2text(table12)))
        ax_onset.text(case12_boxmid-boxw*2, 5.1, 'p'+p2str(pval12), fontsize=legendsize)
        ax_onset.errorbar(x=case12_boxmid, y=4.9, xerr=boxw/2, c='k', capsize=2.5)


        # Test 12 vs 34
        all_boxmid = np.mean(ca_box_pos[ca])
        onset12 = np.concatenate([onset_dict['%s1'%(ca)], onset_dict['%s2'%(ca)]])
        onset34 = np.concatenate([onset_dict['%s3'%(ca)], onset_dict['%s4'%(ca)]])
        pval1234, table1234 = watson_williams(onset12, onset34)
        stat_record(stat_fn, False, 'Onset, %s, Case 1+2 vs Case 3+4, Watson-Williams test, %s'%(ca, wwtable2text(table1234)))
        ax_onset.text(all_boxmid-boxw*2, 6, 'p'+p2str(pval1234), fontsize=legendsize)
        ax_onset.errorbar(x=all_boxmid, y=5.8, xerr=boxw*1.5, c='k', capsize=2.5)


    ax_onset.set_ylim(2, 2*np.pi+0.5)
    ax_onset.set_yticks([np.pi, 2*np.pi])
    ax_onset.set_yticklabels(['$\pi$', '$2\pi$'])

    ax_onset.set_ylabel('Onset', fontsize=fontsize)
    ax_onset.set_xticks(np.concatenate([ca_box_pos['CA1'], ca_box_pos['CA2'], ca_box_pos['CA3'], ca_box_pos['ALL']]))
    ax_onset.set_xticklabels(list('1234')*4)

    for ax_each in [ax_exin2d, ax_exfrac, ax_onset]:
        ax_each.spines['right'].set_visible(False)
        ax_each.spines['top'].set_visible(False)


    fig_4c.text(0.425, 0.1025, 'Case', fontsize=fontsize)
    fig_4c.text(0.535, 0.06, 'All', fontsize=fontsize)
    fig_4c.text(0.654, 0.06, 'CA1', fontsize=fontsize)
    fig_4c.text(0.78, 0.06, 'CA2', fontsize=fontsize)
    fig_4c.text(0.908, 0.06, 'CA3', fontsize=fontsize)


    fig_4c.savefig(join(save_dir, 'exp_4case.png'), dpi=dpi)
    fig_4c.savefig(join(save_dir, 'exp_4case.eps'), dpi=dpi)





def RePreprocess(pairdf):
    all_precess_list = []
    precess_filter = PrecessionFilter()
    for i in range(pairdf.shape[0]):
        print('\rRepreprocess %d/%d'%(i, pairdf.shape[0]), end="", flush=True)
        precessdf = pairdf.loc[i, 'precess_dfp']
        minocc1, minocc2 = pairdf.loc[i, ['minocc1', 'minocc2']]
        precess_dfp = precess_filter.filter_pair(precessdf)
        all_precess_list.append(precess_dfp)
    print()

    pairdf['precess_dfp'] = all_precess_list
    return pairdf



if __name__ == '__main__':
    # tag = '_DirectAllChunk_Phase'
    tag = ''
    save_dir = 'result_plots/pair_fields%s'%(tag)
    pairsim_dir = 'pairangle_similarity%s'%(tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(join(save_dir, pairsim_dir), exist_ok=True)
    expdf = pd.read_pickle('results/exp/pair_field/pairfield_df.pickle')

    # expdf['overlap_ratio'] = expdf['overlap_ratio_phase']
    # expdf['overlap_plus'] = expdf['overlap_plus_phase']
    # expdf['overlap_minus'] = expdf['overlap_minus_phase']

    omniplot_pairfields(expdf, save_dir=save_dir)  # Fig 4
    plot_pair_correlation(expdf, save_dir)  # Fig 5
    # plot_example_exin(expdf, save_dir)  # Fig 6 exintrinsic example
    plot_exintrinsic(expdf, save_dir)  # Fig 6 ex-in scatter
    plot_exintrinsic_concentration(expdf, save_dir)  # Fig 6 concentration
    plot_pairangle_similarity_analysis(expdf, save_dir=join(save_dir, pairsim_dir), nap_thresh=1)  # Fig 6
    FourCasesAnalysis(expdf, save_dir=save_dir)  # Fig 7 Four-cases

    # plot_ALL_examples_correlogram(expdf)


    # # Archive
    # plot_kld(expdf, save_dir=save_dir)  # deleted
    # plot_pair_single_angles_analysis(expdf, save_dir)  # deleted