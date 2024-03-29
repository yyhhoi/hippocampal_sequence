# Legacy. Not the latest analysis methods. Only serves as reference!
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import cm
from pycircstat import vtest, watson_williams
from scipy.interpolate import interp1d, make_interp_spline
from pycircstat.descriptive import resultant_vector_length, cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import chi2_contingency, binom_test, spearmanr, linregress, chisquare, ttest_1samp, ttest_ind

from common.linear_circular_r import rcc
from common.stattests import p2str, stat_record, wwtable2text
from common.comput_utils import fisherexact, ranksums, repeat_arr, midedges, shiftcyc_full2half, unfold_binning_2d, \
    linear_circular_gauss_density, circ_ktest, circular_density_1d, angular_dispersion_test

from common.visualization import color_wheel, directionality_polar_plot, customlegend, plot_marginal_slices, \
    plot_correlogram

from common.script_wrappers import DirectionalityStatsByThresh, permutation_test_average_slopeoffset, \
    permutation_test_arithmetic_average_slopeoffset
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw


def omniplot_singlefields_hc3(hc3df, save_dir=None):
    linew = 0.75

    # Initialization of stats
    stat_fn = 'HC3_fig1_single_directionality.txt'
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


    ax_directR[2].axis('off')
    ax_directFrac[2].axis('off')


    figl = total_figw / 4
    fig_pval, ax_pval = plt.subplots(2, 1, figsize=(figl*4, figl*4))


    # Initialization of parameters
    spike_threshs = np.arange(0, 401, 50)
    stats_getter = DirectionalityStatsByThresh('num_spikes', 'rate_R_pval', 'rate_R')
    chipval_dict = dict(CA1_be=[], CA3_be=[], all_CA13=[])
    rspval_dict = dict(CA1_be=[], CA3_be=[], all_CA13=[])

    # # Plot data
    data_dict = {'CA1':{}, 'CA3':{}}
    for caid, (ca, cadf) in enumerate(hc3df.groupby('ca')):

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
        ax[1, caid].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize, color=ca_c[ca])



    # # Statistical test
    for idx, ntresh in enumerate(spike_threshs):

        # Between border cases for each CA
        for ca in ['CA1', 'CA3']:
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
            ca1d, ca3d = data_dict['CA1'][bcase], data_dict['CA3'][bcase]

            rs_stat13, rs_p13R = ranksums(ca1d['allR'][idx], ca3d['allR'][idx])

            contin13 = pd.DataFrame({'CA1': [ca1d['shift_signum'][idx], ca1d['shift_nonsignum'][idx]],
                                     'CA3': [ca3d['shift_signum'][idx], ca3d['shift_nonsignum'][idx]]})

            try:
                chi_stat13, chi_p13, chi_dof13, _ = chi2_contingency(contin13)
            except ValueError:
                chi_stat13, chi_p13, chi_dof13 = 0, 1, 0


            f_p13 = fisherexact(contin13.to_numpy())
            nCA1, nCA3 = ca1d['n'][idx], ca3d['n'][idx]
            stattxt = "Threshold=%d, %s, CA1 vs CA3: Chi-square  test  for  significant  fraction, \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, chi_dof13, nCA1 + nCA3, chi_stat13, p2str(chi_p13), p2str(f_p13)))

            mdnR1, mdnR3 = ca1d['medianR'][idx], ca3d['medianR'][idx]
            stattxt = "Threshold=%d, %s, CA1 vs CA3: Mann-Whitney U test for medianR, %0.3f vs %0.3f, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, mdnR1, mdnR3, nCA1, nCA3, rs_stat13, p2str(rs_p13R)))

            if bcase == 'all':
                chipval_dict['all_CA13'].append(chi_p13)
                rspval_dict['all_CA13'].append(rs_p13R)

    # Plot pvals for reference
    logalpha = np.log10(0.05)
    for ca in ['CA1', 'CA3']:
        ax_pval[0].plot(spike_threshs, np.log10(chipval_dict[ca + '_be']), c=ca_c[ca], label=ca+'_Chi2', linestyle='-', marker='o')
        ax_pval[0].plot(spike_threshs, np.log10(rspval_dict[ca + '_be']), c=ca_c[ca], label=ca+'_RS', linestyle='--', marker='x')
    ax_pval[0].plot([spike_threshs.min(), spike_threshs.max()], [logalpha, logalpha], c='k', label='p=0.05')
    ax_pval[0].set_xlabel('Spike count threshold')
    ax_pval[0].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_pval[0].set_title('Border effect', fontsize=titlesize)
    customlegend(ax_pval[0], fontsize=legendsize)
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA13']), c='r', label='13_Chi2', linestyle='-', marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA13']), c='r', label='13_RS', linestyle='--', marker='x')
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



def plot_field_bestprecession_hc3(df, save_dir):

    figl = total_figw*0.6
    linew = 0.75
    R_figl = total_figw*0.4
    nap_thresh = 1


    print('Plot field best precession')
    stat_fn = 'HC3_fig2_field_precess.txt'
    stat_record(stat_fn, True)
    fig_negslope, ax_negslope = plt.subplots(figsize=(R_figl*0.8, R_figl*0.5))
    fig_R, ax_R = plt.subplots(figsize=(R_figl*0.8, R_figl*0.5))
    fig_pishift = plt.figure(figsize=(figl, figl*4/3))
    ax_pishift = np.array([
        [fig_pishift.add_subplot(4, 3, i ) for i in range(1, 4)],
        [fig_pishift.add_subplot(4, 3, i) for i in range(4, 7)],
        [fig_pishift.add_subplot(4, 3, i, polar=True) for i in range(7, 10)],
        [fig_pishift.add_subplot(4, 3, i, polar=True) for i in range(10, 13)]
    ])

    allR = []
    for i in range(1, 4):
        ax_pishift[i, 2].axis('off')

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
            if allprecess_df.shape[0] <1:
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
        pm, pc, pr, ppval, _ = linregress(adm, norm_bins/ norm_bins.sum())
        xdum = np.linspace(adm.min(), adm.max(), 10)
        linew_ax0 = 0.6
        ax_pishift[0, 0].step(adm, precess_bins / precess_bins.sum(), color=ca_c[ca], linewidth=linew_ax0,
                              label=ca)
        ax_pishift[0, 1].step(adm, spike_bins / spike_bins.sum(), color=ca_c[ca], linewidth=linew_ax0)
        ax_pishift[0, 2].step(adm, norm_bins / norm_bins.sum(), color=ca_c[ca], linewidth=linew_ax0)
        ax_pishift[0, 2].plot(xdum, xdum*pm+pc, linewidth=linew_ax0, color=ca_c[ca])
        ax_pishift[0, 2].text(0, 0.0525 - caid*0.0075, 'p%s'%(p2str(pval)), fontsize=legendsize, color=ca_c[ca])

        stat_record(stat_fn, False, "%s Spearman's correlation: r_s(%d)=%0.2f, p%s, Pearson's: m=%0.2f, r=%0.2f, p%s " % \
                    (ca, precess_allcount, rho, p2str(pval), pm, pr, p2str(ppval)))

        # Plot fraction of neg slopes
        all_adiff = np.abs(np.concatenate(all_adiff))
        all_slopes = np.concatenate(all_slopes)
        frac_N = np.zeros(adiff_bins.shape[0]-1)
        for edgei in range(adiff_bins.shape[0]-1):
            idxtmp = np.where((all_adiff > adiff_bins[edgei]) & (all_adiff <= adiff_bins[edgei+1]))[0]
            frac_N[edgei] = (all_slopes[idxtmp] < 0).mean()
        ax_negslope.plot(adm, frac_N, c=ca_c[ca], linewidth=0.75, label=ca)
        ax_negslope.set_xlabel(r'$\theta_{pass}$', fontsize=fontsize)
        ax_negslope.set_ylabel('Frac. of\n-ve slopes', fontsize=fontsize, labelpad=5)
        ax_negslope.set_xticks([0, np.pi/2, np.pi])
        ax_negslope.set_xticklabels(['0', '', '$\pi$'])
        ax_negslope.set_yticks([0, 0.5, 1])
        ax_negslope.set_yticklabels(['0', '', '1'])
        ax_negslope.tick_params(labelsize=ticksize)
        ax_negslope.spines["top"].set_visible(False)
        ax_negslope.spines["right"].set_visible(False)
        customlegend(ax_negslope, bbox_to_anchor=[0.6, 0.5], loc='lower left', fontsize=legendsize)


        # Plot precess R
        caR = cadf[(~cadf['precess_R'].isna()) & numpass_mask]['precess_R'].to_numpy()
        allR.append(caR)
        rbins, redges = np.histogram(caR, bins=50)
        rbinsnorm = np.cumsum(rbins) / rbins.sum()
        ax_R.plot(midedges(redges), rbinsnorm, label=ca, c=ca_c[ca], linewidth=linew)

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
        ax_pishift[1, 1].set_xlabel(r'$\theta_{rate}$', fontsize=fontsize)
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
        ax_pishift[2, caid].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k',  zorder=3,  arrowprops=dict(arrowstyle="->"))
        ax_pishift[2, caid].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
        ax_pishift[2, caid].scatter(0, 0, s=16, c='gray')
        ax_pishift[2, caid].text(0.4, l * 0.4, r'$\theta_{rate}$', fontsize=fontsize + 1)
        ax_pishift[2, caid].spines['polar'].set_visible(False)
        ax_pishift[2, caid].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax_pishift[2, caid].set_yticks([0, l/2])
        ax_pishift[2, caid].set_yticklabels([])
        ax_pishift[2, caid].set_xticklabels([])
        v_pval, v_stat = vtest(adiff, mu=np.pi)
        pvalstr = 'p=%0.4f'%(v_pval) if v_pval > 0.0001 else 'p<0.0001'
        ax_pishift[2, caid].text(x=0.01, y=0.95, s=pvalstr, fontsize=legendsize,
                                 transform=ax_pishift[2, caid].transAxes)
        stat_record(stat_fn, False, '%s, d(precess, rate), V(%d)=%0.2f, p%s' % (ca, bins.sum(), v_stat,
                                                                                p2str(v_pval)))

        # Plot Histogram: d(precess_low, rate)
        mask_low = (~np.isnan(rateangles)) & (~np.isnan(precessangles_low) & numpass_low_mask)
        adiff = cdiff(precessangles_low[mask_low], rateangles[mask_low])
        bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
        bins_norm = bins / np.sum(bins)
        l = bins_norm.max()
        ax_pishift[3, caid].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], color=ca_c[ca], zorder=0)
        mean_angle = shiftcyc_full2half(circmean(adiff))
        ax_pishift[3, caid].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k',  zorder=3,  arrowprops=dict(arrowstyle="->"))
        ax_pishift[3, caid].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
        ax_pishift[3, caid].scatter(0, 0, s=16, c='gray')
        ax_pishift[3, caid].text(0.4, l * 0.4, r'$\theta_{rate}$', fontsize=fontsize + 1)
        ax_pishift[3, caid].spines['polar'].set_visible(False)
        ax_pishift[3, caid].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax_pishift[3, caid].set_yticks([0, l/2])
        ax_pishift[3, caid].set_yticklabels([])
        ax_pishift[3, caid].set_xticklabels([])
        v_pval, v_stat = vtest(adiff, mu=np.pi)
        pvalstr = 'p=%0.4f'%(v_pval) if v_pval > 0.0001 else 'p<0.0001'
        ax_pishift[3, caid].text(x=0.01, y=0.95, s=pvalstr, fontsize=legendsize,
                                 transform=ax_pishift[3, caid].transAxes)
        stat_record(stat_fn, False, '%s, d(precess_low, rate), V(%d)=%0.2f, p%s' % (ca, bins.sum(), v_stat,
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
        stat_record(stat_fn, False, "Sig. Frac., %s, Among all: %d/%d=%0.3f, Binomial test p%s. Among precess: %d/%d=%0.3f, Binomial test p%s" % \
                    (ca, sig_num, ntotal_fields, frac_amongall, p2str(bip_amongall), sig_num, n_precessfields, frac_amongprecess, p2str(bip_amongprecess)))




    # Overal asthestics
    U13, p13 = ranksums(allR[0], allR[1])
    stat_record(stat_fn, False, 'CA13, R difference, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s' % \
                (allR[0].shape[0], allR[1].shape[0], U13, p2str(p13)))
    ax_R.text(0.1, 0.02, 'CA1-3 diff.\np%s'%(p2str(p13)), fontsize=legendsize)
    ax_R.set_xlabel('R', fontsize=fontsize)
    ax_R.set_ylabel('Cumulative\nfield density', fontsize=fontsize, labelpad=5)
    customlegend(ax_R, linewidth=1.5, fontsize=legendsize, bbox_to_anchor=[0.6, 0.3], loc='lower left')
    ax_R.set_yticks([0, 0.5, 1])
    ax_R.set_yticklabels(['0', '', '1'])
    ax_R.tick_params(axis='both', which='major', labelsize=ticksize)
    ax_R.spines["top"].set_visible(False)
    ax_R.spines["right"].set_visible(False)



    ax_pishift[1, 0].set_ylabel(r'$\theta_{Precess}$', fontsize=fontsize)

    for ax_each in ax_pishift[0, ]:
        ax_each.set_xticks([0, np.pi / 2, np.pi])
        ax_each.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        ax_each.set_ylim([0.01, 0.055])
        ax_each.tick_params(axis='both', which='major', labelsize=fontsize)
        ax_each.spines["top"].set_visible(False)
        ax_each.spines["right"].set_visible(False)
    ax_pishift[0, 1].set_xlabel(r'$|d(\theta_{pass}, \theta_{rate})|$' + ' (rad)', fontsize=fontsize)
    ax_pishift[0, 0].set_ylabel('Density', fontsize=fontsize)
    ax_pishift[0, 0].yaxis.labelpad = 15

    customlegend(ax_pishift[0, 0], handlelength=0.8, linewidth=1.2, fontsize=legendsize, bbox_to_anchor=[0.5, 0.3], loc='lower left')
    ax_pishift[0, 0].set_title('Precession', fontsize=fontsize)
    ax_pishift[0, 0].set_yticks([0.01, 0.03, 0.05])
    plt.setp(ax_pishift[0, 0].get_yticklabels(), visible=False)

    ax_pishift[0, 1].set_title('Spike', fontsize=fontsize)
    ax_pishift[0, 1].set_yticks([0.01, 0.03, 0.05])
    plt.setp(ax_pishift[0, 1].get_yticklabels(), visible=False)

    ax_pishift[0, 2].set_title('Ratio', fontsize=fontsize)
    ax_pishift[0, 2].set_yticks([0.01, 0.03, 0.05])
    plt.setp(ax_pishift[0, 2].get_yticklabels(), visible=False)


    plt.setp(ax_pishift[1, 1].get_yticklabels(), visible=False)
    plt.setp(ax_pishift[1, 2].get_yticklabels(), visible=False)
    ax_pishift[2, 0].set_ylabel('All\npasses', fontsize=fontsize)
    ax_pishift[2, 0].yaxis.labelpad = 15
    ax_pishift[3, 0].set_ylabel('Low-spike\n passes', fontsize=fontsize)
    ax_pishift[3, 0].yaxis.labelpad = 15


    fig_pishift.tight_layout()
    fig_pishift.subplots_adjust(wspace=0.25, hspace=1.25)
    fig_R.tight_layout()
    fig_R.subplots_adjust(bottom=0.3)
    fig_negslope.tight_layout()
    fig_negslope.subplots_adjust(bottom=0.4)

    for figext in ['png', 'eps']:
        fig_pishift.savefig(join(save_dir, 'field_precess_pishift.%s'%(figext)), dpi=dpi)
        fig_R.savefig(os.path.join(save_dir, 'field_precess_R.%s'%(figext)), dpi=dpi)
        fig_negslope.savefig(join(save_dir, 'fraction_neg_slopes.%s'%(figext)), dpi=dpi)



def plot_both_slope_offset_hc3(df, save_dir, NShuffles=200):
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

    offset_label = 'Onset phase (rad)'
    offset_slicerange = (0, 2*np.pi)
    offset_bound = (0, 2 * np.pi)
    offset_xticks = [1, 3, 5]
    offset_xticksl = ['%d' % x for x in offset_xticks]
    offset_slicegap = 0.017  # 0.007

    slope_label = 'Slope (rad)'
    slope_slicerange = (-2 * np.pi, 0)
    slope_bound = (-2 * np.pi, 0)
    slope_xticks = [-2 * np.pi, -np.pi, 0]
    slope_xticksl = ['$-2\pi$', '$-\pi$', '0']
    slope_slicegap = 0.01  # 0.007

    adiff_edges = np.linspace(0, np.pi, 100)
    offset_edges = np.linspace(offset_bound[0], offset_bound[1], 100)
    slope_edges = np.linspace(slope_bound[0], slope_bound[1], 100)



    figl = total_figw/6
    fig1size = (figl*3 * 0.95, figl*2.8)  # 0.05 as space for colorbar
    fig2size = (figl*3 * 0.95, figl*2.8)
    fig3size = (figl*3 * 0.95, figl*2.8)

    stat_fn = 'HC3_fig3_slopeoffset.txt'
    stat_record(stat_fn, True, 'Average Phase precession')
    fig1, ax1 = plt.subplots(2, 3, figsize=fig1size, sharey='row')  # Density
    fig2, ax2 = plt.subplots(2, 3, figsize=fig2size, sharey='row')  # Marginals
    fig3, ax3 = plt.subplots(2, 3, figsize=fig3size, sharey='row')  # Aver curves + spike phase
    fig_s, ax_s = plt.subplots(3, 1, figsize=(figl*6, figl*6), sharey='row')  # Checking onset, slope, phases

    for i in range(2):
        ax1[i, 2].axis('off')
        ax2[i, 2].axis('off')
        ax3[i, 2].axis('off')
    ax_s[2].axis('off')


    # Construct pass df
    refangle_key = 'rate_angle'
    passdf_dict = {'ca':[], 'anglediff':[], 'slope':[], 'onset':[], 'pass_nspikes':[]}
    spikedf_dict = {'ca':[], 'anglediff':[], 'phasesp':[]}
    dftmp = df[(~df[refangle_key].isna())].reset_index()
    # dftmp = self.singlefield_df[(~self.singlefield_df[refangle_key].isna()) & \
    #                             (self.singlefield_df['numpass_at_precess'] >= self.nap_thresh)].reset_index()

    for i in range(dftmp.shape[0]):
        allprecess_df = dftmp.loc[i, 'precess_df']
        if allprecess_df.shape[0]<1:
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

    shift200 = np.deg2rad(200)

    # Plot and analysis for CA1, CA2 and CA3
    for caid, ca in enumerate(['CA1', 'CA3']):


        pass_cadf = passdf[passdf['ca']==ca].reset_index(drop=True)
        spike_cadf = spikedf[spikedf['ca']==ca].reset_index(drop=True)

        # if ca == 'CA3':
        #     spike_cadf['phasesp'] = np.mod(spike_cadf['phasesp'] + np.pi + shift200, 2*np.pi) - np.pi
        #     pass_cadf['onset'] = np.mod(pass_cadf['onset'] + np.pi + shift200, 2*np.pi) - np.pi


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

        stat_record(stat_fn, False, 'LC_Regression %s Onset-adiff r(%d)=%0.3f, p%s'%(ca, offset_bins.sum(), offset_rho, p2str(offset_p)))
        stat_record(stat_fn, False, 'LC_Regression %s Slope-adiff r(%d)=%0.3f, p%s'%(ca, slope_bins.sum(), slope_rho, p2str(slope_p)))

        # Density
        offset_xx, offset_yy, offset_zz = linear_circular_gauss_density(offset_adiff, offset_norm,
                                                                        cir_kappa=4 * np.pi, lin_std=0.2, xbins=50,
                                                                        ybins=50, xbound=(0, np.pi),
                                                                        ybound=offset_bound)
        slope_xx, slope_yy, slope_zz = linear_circular_gauss_density(slope_adiff, slope_norm,
                                                                     cir_kappa=4 * np.pi, lin_std=0.2, xbins=50,
                                                                     ybins=50, xbound=(0, np.pi),
                                                                     ybound=slope_bound)

        # Plot offset density
        cmap = 'Blues'
        _, _, edgtmp, _ = ax1[0, caid].hist2d(offset_adiff, offset_norm,
                                              bins=(np.linspace(0, np.pi, 36), np.linspace(0, 2 * np.pi, 36)), density=True,
                                              cmap=cmap)

        regressed = (offset_c + offset_xedm * offset_m)
        regress_width = np.pi * offset_m
        ax1[0, caid].plot(offset_xedm, regressed, c='r', linewidth=0.85)
        ax1[0, caid].plot([0, np.pi], [offset_c - regress_width * 2.5, offset_c - regress_width * 2.5], c='purple', linewidth=0.5)
        ax1[0, caid].plot([0, np.pi], [offset_c + regress_width * 3, offset_c + regress_width * 3], c='purple', linewidth=0.5)
        pvalstr =  'p=%0.4f'%offset_p if offset_p > 1e-4 else 'p<0.0001'
        ax1[0, caid].text(0.02, 5, pvalstr, fontsize=legendsize)
        ax1[0, caid].set_xticks(adiff_ticks)
        ax1[0, caid].set_xticklabels(adiff_ticksl)
        ax1[0, caid].set_yticks(offset_xticks)
        ax1[0, caid].set_yticklabels(offset_xticksl)
        ax1[0, caid].tick_params(labelsize=ticksize)
        ax1[0, caid].set_title(ca, fontsize=titlesize)
        ax1[0, 1].set_xlabel(adiff_label, fontsize=fontsize)

        # Plot slope density
        regressed = (slope_c + slope_xedm * slope_m)
        regress_width = np.pi * slope_m
        _, _, edgtmp, _ = ax1[1, caid].hist2d(slope_adiff, slope_norm,
                                              bins=(np.linspace(0, np.pi, 36), np.linspace(-2 * np.pi, 0, 36)), density=True,
                                              cmap=cmap)
        ax1[1, caid].plot(slope_xedm, regressed, c='r', linewidth=0.75)
        ax1[1, caid].plot([0, np.pi], [slope_c - regress_width * 0.5, slope_c - regress_width * 0.5], c='purple', linewidth=0.5)
        ax1[1, caid].plot([0, np.pi], [slope_c + regress_width * 1.5, slope_c + regress_width * 1.5], c='purple', linewidth=0.5)
        pvalstr =  'p=%0.4f'%slope_p if slope_p > 1e-4 else 'p<0.0001'
        ax1[1, caid].text(0.02, -3.5*np.pi/2, pvalstr, fontsize=legendsize)
        ax1[1, caid].set_xticks(adiff_ticks)
        ax1[1, caid].set_xticklabels(adiff_ticksl)
        ax1[1, caid].set_yticks(slope_xticks)
        ax1[1, caid].set_yticklabels(slope_xticksl)
        ax1[1, caid].tick_params(labelsize=ticksize)
        ax1[1, 1].set_xlabel(adiff_label, fontsize=fontsize)

        # # Marginals
        plot_marginal_slices(ax2[0, caid], offset_xx, offset_yy, offset_zz,
                             selected_adiff,
                             offset_slicerange, offset_slicegap)


        ax2[0, caid].set_xticks(offset_xticks)
        ax2[0, caid].set_xticklabels(offset_xticksl)
        ax2[0, caid].set_xlim(1, 5)
        ax2[0, 1].set_xlabel(offset_label, fontsize=fontsize)
        ax2[0, caid].tick_params(labelsize=ticksize)
        ax2[0, caid].set_title(ca, fontsize=titlesize, pad=15)

        plot_marginal_slices(ax2[1, caid], slope_xx, slope_yy, slope_zz,
                             selected_adiff, slope_slicerange, slope_slicegap)
        ax2[1, caid].set_xticks(slope_xticks)
        ax2[1, caid].set_xticklabels(slope_xticksl)
        ax2[1, 1].set_xlabel(slope_label, fontsize=fontsize)
        ax2[1, caid].tick_params(labelsize=ticksize)
        ax2[1, caid].set_xlim(-2*np.pi, 0)

        # # Plot average precession curves
        low_mask = absadiff_pass < (np.pi / 6)
        high_mask = absadiff_pass > (np.pi - np.pi / 6)
        print('%s high %d, low %d'%(ca, high_mask.sum(), low_mask.sum()))
        sample_size_high = min(high_mask.sum(), 500)
        sample_size_low = min(low_mask.sum(), 500)
        np.random.seed(1)
        high_ranvec = np.random.choice(high_mask.sum(), size=sample_size_high, replace=False)
        low_ranvec = np.random.choice(low_mask.sum(), size=sample_size_low, replace=False)
        slopes_high, offsets_high = slope[high_mask][high_ranvec], offset[high_mask][high_ranvec]
        slopes_low, offsets_low = slope[low_mask][low_ranvec], offset[low_mask][low_ranvec]
        slopes_high_all, offsets_high_all = slope[high_mask], offset[high_mask]
        slopes_low_all, offsets_low_all = slope[low_mask], offset[low_mask]

        # Annotate marginals
        offsetpval, offsetwwtable = watson_williams(offsets_high_all, offsets_low_all)
        slopeU, slopepval = ranksums(slopes_high_all, slopes_low_all)
        stat_record(stat_fn, False, '%s onset difference p%s\n%s'%(ca, p2str(offsetpval), offsetwwtable.to_string()))
        stat_record(stat_fn, False, '%s Slope difference U(N_{high-|d|}=%d, N_{low-|d|}=%d)=%0.2f, p%s'%(ca, slopes_high_all.shape[0], slopes_low_all.shape[0], slopeU, p2str(slopepval)))

        regress_high, regress_low, pval_slope, pval_offset = permutation_test_average_slopeoffset(
            slopes_high, offsets_high, slopes_low, offsets_low, NShuffles=NShuffles)

        stat_record(stat_fn, False,
                    '%s d=high, slope=%0.2f, offset=%0.2f' % \
                    (ca, regress_high['aopt'] * 2 * np.pi, regress_high['phi0']))
        stat_record(stat_fn, False,
                    '%s d=low, slope=%0.2f, offset=%0.2f' % \
                    (ca, regress_low['aopt'] * 2 * np.pi, regress_low['phi0']))
        stat_record(stat_fn, False,
                    '%s high-low-difference, Slope: p=%0.4f; Offset: p=%0.4f' % \
                    (ca, pval_slope, pval_offset))


        xdum = np.linspace(0, 1, 10)
        high_agg_ydum = 2 * np.pi * regress_high['aopt'] * xdum + regress_high['phi0']
        low_agg_ydum = 2 * np.pi * regress_low['aopt'] * xdum + regress_low['phi0']

        ax3[0, caid].plot(xdum, high_agg_ydum, c='lime', label='$|d|>5\pi/6$')
        ax3[0, caid].plot(xdum, low_agg_ydum, c='darkblue', label='$|d|<\pi/6$')

        pvalstr1 = '$p_s$'+'%s'%p2str(pval_slope)
        pvalstr2 = '$p_o$'+'%s'%p2str(pval_offset)

        ax3[0, caid].annotate('%s'%(pvalstr1), xy=(0, 0.2), xycoords='axes fraction', fontsize=legendsize)
        ax3[0, caid].annotate('%s'%(pvalstr2), xy=(0, 0.035), xycoords='axes fraction', fontsize=legendsize)
        ax3[0, caid].spines["top"].set_visible(False)
        ax3[0, caid].spines["right"].set_visible(False)
        ax3[0, caid].set_title(ca, fontsize=titlesize, pad=15)

        # # high- and low-|d| Spike phases
        low_mask_sp = absadiff_spike < (np.pi / 6)
        high_mask_sp = absadiff_spike > (np.pi - np.pi / 6)
        phasesph = phase_spike[high_mask_sp]
        phasespl = phase_spike[low_mask_sp]
        mean_phasesph = circmean(phasesph)
        mean_phasespl = circmean(phasespl)
        fstat, k_pval = circ_ktest(phasesph, phasespl)
        p_ww, ww_table = watson_williams(phasesph, phasespl)
        nh, _, _ = ax3[1, caid].hist(phasesph, bins=36, density=True, histtype='step', color='lime')
        nl, _, _ = ax3[1, caid].hist(phasespl, bins=36, density=True, histtype='step', color='darkblue')
        ml = max(nh.max(), nl.max())
        # ax3[1, caid].scatter(mean_phasesph, ml*1.2, marker='|', color='lime', linewidth=0.75)
        # ax3[1, caid].scatter(mean_phasespl, ml*1.2, marker='|', color='darkblue', linewidth=0.75)

        ax3[1, caid].axvline(mean_phasesph, color='lime', linewidth=0.75)
        ax3[1, caid].axvline(mean_phasespl, color='darkblue', linewidth=0.75)


        ax3[1, caid].text(-np.pi+0.1, 0.02, '$p$'+'%s'%(p2str(p_ww)), fontsize=legendsize)
        ax3[1, caid].set_xlim(-np.pi, np.pi)
        ax3[1, caid].set_xticks([-np.pi, 0, np.pi])
        ax3[1, caid].set_xticklabels(['$-\pi$', '0', '$\pi$'])
        ax3[1, caid].set_yticks([])
        ax3[1, caid].tick_params(labelsize=ticksize)
        ax3[1, caid].spines["top"].set_visible(False)
        ax3[1, caid].spines["right"].set_visible(False)
        ax3[1, caid].spines["left"].set_visible(False)
        ax3[1, 1].set_xlabel('Phase (rad)', fontsize=fontsize)
        ax3[1, 0].set_ylabel('Relative\nfrequency', fontsize=fontsize)

        stat_record(stat_fn, False, 'SpikePhase-HighLowDiff %s Mean_diff p%s' % (ca, p2str(p_ww)))
        stat_record(stat_fn, False, ww_table.to_string())
        stat_record(stat_fn, False,
                    'SpikePhase-HighLowDiff %s Bartlett\'s test F_{(%d, %d)}=%0.2f, p%s' % \
                    (ca, phasesph.shape[0], phasespl.shape[0], fstat, p2str(k_pval)))


        # # Sanity check: Compare means of onset, slope, phasesp directly
        ps_tmp, _ = watson_williams(slopes_high_all, slopes_low_all)
        po_tmp, _ = watson_williams(offsets_high_all, offsets_low_all)
        ax_s[caid].boxplot([slopes_high_all, slopes_low_all, offsets_high_all, offsets_low_all], positions=[0, 1, 2, 3])
        ax_s[caid].text(0, np.pi, 'p%s'%(p2str(ps_tmp)), fontsize=legendsize)
        ax_s[caid].text(2.5, -np.pi, 'p%s'%(p2str(po_tmp)), fontsize=legendsize)
        ax_s[caid].set_xticklabels(['SH', 'SL', "OH", "OL"], fontsize=legendsize)
        ax_s[caid].set_title(ca, fontsize=legendsize)

    # Asthestic for density and slices
    ax1[0, 0].set_ylabel(offset_label, fontsize=fontsize)
    ax1[0, 0].yaxis.labelpad = 18
    ax1[1, 0].set_ylabel(slope_label, fontsize=fontsize)
    fig2.text(0.05, 0.35, 'Marginal density\n of precession', rotation=90, fontsize=fontsize)

    ax3[1, 0].yaxis.labelpad = 15
    for ax_each in np.append(ax2[0, ], ax2[1, ]):
        ax_each.set_yticks([])
        ax_each.grid(False, axis='y')
        ax_each.spines["top"].set_visible(False)
        ax_each.spines["right"].set_visible(False)
        ax_each.spines["left"].set_visible(False)


    # Asthestics for average curves
    for ax_each in ax3[0, ]:
        ax_each.set_xlim(0, 1)
        ax_each.set_ylim(-np.pi, 2*np.pi)
        ax_each.set_yticks([-np.pi, 0, np.pi])
        ax_each.set_yticklabels(['$-\pi$', '0', '$\pi$'])
        ax_each.tick_params(labelsize=ticksize)
    ax3[0, 1].set_xlabel('Position')
    customlegend(ax3[0, 2], fontsize=legendsize, loc='lower left', handlelength=0.5, bbox_to_anchor=(0.1, 0.7))
    ax3[0, 0].set_ylabel('Phase (rad)', fontsize=fontsize)
    ax3[0, 0].yaxis.labelpad = 5

    # Asthestics for all
    for i in range(2):
        plt.setp(ax1[i, 0].get_xticklabels(), visible=False)
        plt.setp(ax1[i, 2].get_xticklabels(), visible=False)
        plt.setp(ax2[i, 0].get_xticklabels(), visible=False)
        # plt.setp(ax2[i, 2].get_xticklabels(), visible=False)
        plt.setp(ax3[i, 0].get_xticklabels(), visible=False)
        # plt.setp(ax3[i, 2].get_xticklabels(), visible=False)


    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.brg,
                               norm=plt.Normalize(vmin=selected_adiff.min(), vmax=selected_adiff.max()))
    fig_colorbar = plt.figure(figsize=(total_figw*0.1, figl*2))

    cbar_ax = fig_colorbar.add_axes([0.1, 0.1, 0.1, 0.8])
    # plt.gca().set_visible(False)
    cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
    cb.set_ticks(adiff_ticks)
    cb.set_ticklabels(adiff_ticksl)
    cb.set_label(adiff_label, fontsize=fontsize)

    # # Saving
    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.2, hspace=1.2, left=0.2)

    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0.2, hspace=1.2, left=0.2)

    fig3.tight_layout()
    fig3.subplots_adjust(wspace=0.2, hspace=1.2, left=0.2)
    fig_s.tight_layout()

    for figext in ['png', 'eps']:
        fig_colorbar.savefig(os.path.join(save_dir, 'adiff_colorbar.%s'%(figext)), dpi=dpi)
        fig1.savefig(os.path.join(save_dir, 'Densities.%s'%(figext)), dpi=dpi)
        fig2.savefig(os.path.join(save_dir, 'Marginals.%s'%(figext)), dpi=dpi)
        fig3.savefig(os.path.join(save_dir, 'AverCurves_SpikePhase.%s'%(figext)), dpi=dpi)
        fig_s.savefig(os.path.join(save_dir, 'HighLow-d_SanityCheck.%s'%(figext)), dpi=dpi)
    return None


maxspcounts = 250
spcount_step = 20

def omniplot_pairfields_hc3(expdf, save_dir=None):
    linew = 0.75

    # Initialization of stats
    stat_fn = 'HC3_fig4_pair_directionality.txt'
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

    ax_directR[2].axis('off')
    ax_directFrac[2].axis('off')


    figl = total_figw / 4
    fig_pval, ax_pval = plt.subplots(2, 1, figsize=(figl*4, figl*4))

    # Initialization of parameters
    spike_threshs = np.arange(0, maxspcounts, spcount_step)
    stats_getter = DirectionalityStatsByThresh('num_spikes_pair', 'rate_R_pvalp', 'rate_Rp')
    chipval_dict = dict(CA1_be=[], CA3_be=[], all_CA13=[])
    rspval_dict = dict(CA1_be=[], CA3_be=[], all_CA13=[])
    expdf['border'] = expdf['border1'] | expdf['border2']

    # # Plot
    data_dict = {'CA1':{}, 'CA3':{}}
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
        for ca in ['CA1', 'CA3']:
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
            ca1d, ca3d = data_dict['CA1'][bcase], data_dict['CA3'][bcase]

            rs_stat13, rs_p13R = ranksums(ca1d['allR'][idx], ca3d['allR'][idx])

            contin13 = pd.DataFrame({'CA1': [ca1d['shift_signum'][idx], ca1d['shift_nonsignum'][idx]],
                                     'CA3': [ca3d['shift_signum'][idx], ca3d['shift_nonsignum'][idx]]})

            try:
                chi_stat13, chi_p13, chi_dof13, _ = chi2_contingency(contin13)
            except ValueError:
                chi_stat13, chi_p13, chi_dof13 = 0, 1, 0

            f_p13 = fisherexact(contin13.to_numpy())

            nCA1, nCA3 = ca1d['n'][idx], ca3d['n'][idx]
            stattxt = "Threshold=%d, %s, CA1 vs CA3: Chi-square  test  for  significant  fraction, \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, chi_dof13, nCA1 + nCA3, chi_stat13, p2str(chi_p13), p2str(f_p13)))

            mdnR1, mdnR3 = ca1d['medianR'][idx], ca3d['medianR'][idx]
            stattxt = "Threshold=%d, %s, CA1 vs CA3: Mann-Whitney U test for medianR, %0.3f vs %0.3f, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % \
                        (ntresh, bcase, mdnR1, mdnR3, nCA1, nCA3, rs_stat13, p2str(rs_p13R)))
            if bcase == 'all':
                chipval_dict['all_CA13'].append(chi_p13)
                rspval_dict['all_CA13'].append(rs_p13R)


    # Plot pvals for reference
    logalpha = np.log10(0.05)
    for ca in ['CA1', 'CA3']:
        ax_pval[0].plot(spike_threshs, np.log10(chipval_dict[ca + '_be']), c=ca_c[ca], label=ca+'_Chi2', linestyle='-', marker='o')
        ax_pval[0].plot(spike_threshs, np.log10(rspval_dict[ca + '_be']), c=ca_c[ca], label=ca+'_RS', linestyle='--', marker='x')
    ax_pval[0].plot([spike_threshs.min(), spike_threshs.max()], [logalpha, logalpha], c='k', label='p=0.05')
    ax_pval[0].set_xlabel('Spike count threshold')
    ax_pval[0].set_ylabel('$log_{10}(pval)$', fontsize=fontsize)
    ax_pval[0].set_title('Border effect', fontsize=titlesize)
    customlegend(ax_pval[0], fontsize=legendsize)
    ax_pval[1].plot(spike_threshs, np.log10(chipval_dict['all_CA13']), c='r', label='13_Chi2', linestyle='-', marker='o')
    ax_pval[1].plot(spike_threshs, np.log10(rspval_dict['all_CA13']), c='r', label='13_RS', linestyle='--', marker='x')
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

def plot_kld_hc3(expdf, save_dir=None):
    stat_fn = 'HC3_fig5_kld.txt'
    stat_record(stat_fn, True)

    # Filtering
    expdf['border'] = expdf['border1'] | expdf['border2']

    kld_key = 'kld'

    figw = total_figw*0.9/2
    figh = total_figw*0.9/4.5

    # # # KLD by threshold
    Nthreshs = np.arange(0, maxspcounts, spcount_step)
    fig_kld_thresh, ax_kld_thresh = plt.subplots(1, 3, figsize=(figw, figh), sharey=True, sharex=True)

    kld_thresh_all_dict = dict(CA1=[], CA3=[])
    kld_thresh_border_dict = dict(CA1=[], CA3=[])
    kld_thresh_nonborder_dict = dict(CA1=[], CA3=[])
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
            kld1, kld3 = klddict['CA1'][idx], klddict['CA3'][idx]
            n1, n3 = kld1.shape[0], kld3.shape[0]
            z13, p13 = ranksums(kld1, kld3)

            stattxt = "KLD by threshold=%d, CA13, %s: Mann-Whitney U test, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % (thresh, bordertxt[borderid], n1, n3, z13, p2str(p13)))

    # Border effect by CA
    for ca in ['CA1', 'CA3']:
        for idx, thresh in enumerate(Nthreshs):
            kld_b, kld_nb = kld_thresh_border_dict[ca][idx], kld_thresh_nonborder_dict[ca][idx]
            n_b, n_nd = kld_b.shape[0], kld_nb.shape[0]
            z_nnb, p_nnb = ranksums(kld_b, kld_nb)
            stattxt = "KLD by threshold=%d, %s, BorderEffect: Mann-Whitney U test, U(N_{border}=%d, N_{nonborder}=%d)=%0.2f, p%s."
            stat_record(stat_fn, False, stattxt % (thresh, ca, n_b, n_nd, z_nnb, p2str(p_nnb)))
    # Save
    fig_kld_thresh.tight_layout()
    fig_kld_thresh.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_kld_thresh.savefig(join(save_dir, 'exp_kld_thresh.png'), dpi=dpi)

def plot_pair_single_angles_analysis_hc3(expdf, save_dir):
    stat_fn = 'HC3_fig5_pairsingle_angles.txt'
    stat_record(stat_fn, True)
    figw = total_figw*0.9/2
    figh = total_figw*0.9/4.5

    figsize = (figw, figh*3)
    fig, ax = plt.subplots(3, 3, figsize=figsize, sharey='row', sharex='row')
    for i in range(3):
        ax[i, 2].axis('off')

    figsize = (figw, figh*1.2)
    fig_test, ax_test = plt.subplots(1, 3, figsize=figsize, sharey='row', sharex='row')


    selected_cmap = cm.cool
    adiff_dict = dict(CA1=[], CA3=[])
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
        ad1, ad3 = adiff_dict['CA1'][thresh_id], adiff_dict['CA3'][thresh_id]
        n1, n3 = ad1.shape[0], ad3.shape[0]
        f13, p13 = circ_ktest(ad1, ad3)
        z13, p13_2, _ = angular_dispersion_test(ad1, ad3)
        p13_all.append(p13)
        p13_all_2.append(p13_2)
        stat_record(stat_fn, False,
                    'Bartlett, CA13, thresh=%d, F(%d, %d)=%0.2f, p%s' % (thresh, n1 - 1, n3 - 1, f13, p2str(p13)))
        stat_record(stat_fn, False, 'U Mann-Whiney U test, CA13, thresh=%d, U(N_{CA1}=%d,N_{CA3}=%d)=%0.2f, p%s )' % (
            thresh, n1, n3, z13, p2str(p13_2)))

    color_bar, marker_bar = 'darkgoldenrod', 'x'
    color_rs, marker_rs = 'purple', '^'
    ms = 6
    ax_test[0].plot(Nthreshs, np.log10(p13_all), c=color_bar, linewidth=0.75, label='BAR')
    # ax_test[0].scatter(Nthreshs, np.log10(p13_all), c=color_bar, marker=marker_bar, s=ms, label='BAR')
    ax_test[0].plot(Nthreshs, np.log10(p13_all_2), c=color_rs, linewidth=0.75, label='RS')
    # ax_test[0].scatter(Nthreshs, np.log10(p13_all_2), c=color_rs, marker=marker_rs, s=ms, label='RS')


    for ax_each in ax_test:
        ax_each.plot([Nthreshs.min(), Nthreshs.max()], [np.log10(0.05), np.log10(0.05)], c='k', label='p=0.05', linewidth=0.75)
        # ax_each.set_xticks([0, 300])
        ax_each.tick_params(labelsize=fontsize)
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    customlegend(ax_test[1], handlelength=0.5, linewidth=1.2, fontsize=legendsize-2, bbox_to_anchor=[0.2, 0.5], loc='lower left')

    ax_test[0].set_title('CA1-3', fontsize=titlesize)
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
    fig.savefig(join(save_dir, 'pair_single_angles.png'), dpi=dpi)

    fig_test.tight_layout()
    fig_test.subplots_adjust(wspace=0.2, hspace=1)
    fig_test.savefig(join(save_dir, 'pair_single_angles_test.png'), dpi=dpi)

    fig_colorbar.savefig(join(save_dir, 'pair_single_angles_colorbar.png'), dpi=dpi)


def plot_pair_correlation_hc3(expdf, save_dir):


    figl = total_figw*0.5/3
    linew = 0.75
    stat_fn = 'HC3_fig5_paircorr.txt'
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
    fig.savefig(join(save_dir, 'exp_paircorr.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'exp_paircorr.eps'), dpi=dpi)



def plot_exintrinsic_hc3(expdf, save_dir):


    figl = total_figw/2/3

    stat_fn = 'HC3_fig6_exintrinsic.txt'
    stat_record(stat_fn, True)

    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'
    # Filtering
    smallerdf = expdf[(~expdf[ratio_key].isna())]
    # smallerdf['Extrincicity'] = smallerdf[ratio_key].apply(lambda x: 'Extrinsic' if x > 0 else 'Intrinsic')

    fig_exin, ax_exin = plt.subplots(2, 3, figsize=(figl*3, figl*2.75), sharey='row', sharex='row')
    ax_exin[0, 2].axis('off')
    ax_exin[1, 2].axis('off')

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
    ratio_3 = smallerdf[smallerdf['ca'] == 'CA3'][ratio_key].to_numpy()
    n1, n3 = ratio_1.shape[0], ratio_3.shape[0]

    equal_var = False
    t_13, p_13 = ttest_ind(ratio_1, ratio_3, equal_var=equal_var)

    zratio13, zp13 = ranksums(ratio_1, ratio_3)
    stat_record(stat_fn, False, 'Welch\'s t-test: CA1 vs CA3, t(%d)=%0.2f, p%s' % (n1 + n3 - 2, t_13, p2str(p_13)))
    stat_record(stat_fn, False, 'RS-test: CA1 vs CA3, U=%0.2f, p%s' % (zratio13, p2str(zp13)))

    contindf = pd.DataFrame(contindf_dict)
    contindf13 = contindf[['CA1', 'CA3']]
    n13 = np.sum(contindf13.to_numpy())
    fisherp13 = fisherexact(contindf13.to_numpy())
    chi13, pchi13, dof13, _ = chi2_contingency(contindf13.to_numpy())

    stattxt = "Two-way Ex-in Frac. CA1 vs CA3 \chi^2(%d, N=%d)=%0.2f, p%s. Fisher exact test p%s."
    stat_record(stat_fn, False, stattxt % (dof13, n13-2, chi13, p2str(pchi13), p2str(fisherp13)))

    fig_exin.tight_layout()
    fig_exin.savefig(join(save_dir, 'exp_exintrinsic.png'), dpi=dpi)
    fig_exin.savefig(join(save_dir, 'exp_exintrinsic.eps'), dpi=dpi)


def plot_exintrinsic_concentration_hc3(expdf, save_dir):
    figl = total_figw/2/3

    stat_fn = 'HC3_fig6_concentration.txt'
    stat_record(stat_fn, True)
    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'

    fig_con, ax_con = plt.subplots(1, 3, figsize=(figl * 3, figl*2.75/2), sharey=True)  # 2.75/2 comes from other panels
    ax_con[2].axis('off')
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
    fig_con.savefig(join(save_dir, 'exp_exintrinsic_concentration.png'), dpi=dpi)



def plot_pairangle_similarity_analysis_hc3(expdf, save_dir, nap_thresh=1):
    stat_fn = 'HC3_fig6_pair_exin_simdisim.txt'
    stat_record(stat_fn, True)
    ratio_key, oplus_key, ominus_key = 'overlap_ratio', 'overlap_plus', 'overlap_minus'
    # anglekey1, anglekey2 = 'precess_angle1', 'precess_angle2'
    anglekey1, anglekey2 = 'rate_angle1', 'rate_angle2'
    figl = total_figw/2/3

    fig_1d, ax_1d = plt.subplots(1, 3, figsize=(figl*4, figl*2), sharex='row', sharey='row')
    fig_1dca, ax_1dca = plt.subplots(1, 2, figsize=(figl*4, figl*2.5), sharex=True, sharey=True)
    fig_2d, ax_2d = plt.subplots(2, 3, figsize=(figl*3, figl*2.5), sharex='row', sharey='row')

    ax_2d[0, 2].axis('off')
    ax_2d[1, 2].axis('off')

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
    for caid, ca in enumerate(['CA1', 'CA3']):  # Compare between sim and dissim
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
    statmodel_CA13table = StratifiedTable([allcadfexin[0], allcadfexin[1]])
    table_allca = allcadfexin[0] + allcadfexin[1]
    simexn, dissimexn = table_allca.loc['similar', 'exnum'], table_allca.loc['dissimilar', 'exnum']
    simtotaln, dissimtotaln = table_allca.sum(axis=1)
    pfish_allca = fisherexact(table_allca.to_numpy())
    stat_record(stat_fn, False, 'All brain sim (%d/%d) vs dissim (%d/%d) table Fisher exact p%s'%(simexn, simtotaln, dissimexn, dissimtotaln, p2str(pfish_allca)))

    for simidtmp, sim_label in enumerate(['similar', 'dissimilar']):
        calabel1, calabel2 = 'CA1', 'CA3'
        dftmp = df_exin[(df_exin['sim']==sim_label) &
                        ((df_exin['ca']==calabel1) | (df_exin['ca']==calabel2))][['exnum', 'innum']]
        chitmp, pchitmp, doftmp, _ = chi2_contingency(dftmp)
        ntotal = dftmp['exnum'].sum() + dftmp['innum'].sum()
        pfisher = fisherexact(dftmp.to_numpy())
        stattxt = '2-way, %s, %s vs %s, \chi^2(%d, n=%d)=%0.2f, p%s. Fisher exact p%s'
        stat_record(stat_fn, False, stattxt % (sim_label, calabel1, calabel2, doftmp, ntotal, chitmp, p2str(pchitmp), p2str(pfisher)))


    # Compare exin for CA1-3
    nsim1, nsim3 = ratiosimdict['CA1'].shape[0], ratiosimdict['CA3'].shape[0]
    ndisim1, ndisim3 = ratiodisimdict['CA1'].shape[0], ratiodisimdict['CA3'].shape[0]

    zsim13, psim13 = ranksums(ratiosimdict['CA1'], ratiosimdict['CA3'])
    zdisim13, pdisim13 = ranksums(ratiodisimdict['CA1'], ratiodisimdict['CA3'])

    ax_1dca[0].text(0.1, 0.2, 'p13rs=%0.4f' % (psim13), fontsize=legendsize)
    ax_1dca[0].set_title('Similar', fontsize=fontsize)
    ax_1dca[0].set_xlabel('Intrinsic <-> Extrinsic', fontsize=fontsize)
    customlegend(ax_1dca[0], fontsize=legendsize)
    ax_1dca[1].text(0.1, 0.2, 'p13rs=%0.4f' % (pdisim13), fontsize=legendsize)
    ax_1dca[1].set_title('Dissimilar', fontsize=fontsize)
    ax_1dca[1].set_xlabel('Intrinsic <-> Extrinsic', fontsize=fontsize)
    customlegend(ax_1dca[1], fontsize=legendsize)

    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Similar, CA1 vs CA3, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s'% (nsim1, nsim3, zsim13, p2str(psim13)))
    stat_record(stat_fn, False, 'Extrinsicity bias, U Mann-Whiney U test, Dissimilar, CA1 vs CA3, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s'% (ndisim1, ndisim3, zdisim13, p2str(pdisim13)))


    stat_record(stat_fn, False, statmodel_CA13table.summary().as_text())


    # Saving
    fig_1d.tight_layout()
    fig_1dca.tight_layout()
    fig_2d.tight_layout()
    fig_1d.savefig(join(save_dir, 'exp_simdissim_exintrinsic1d.png'), dpi=dpi)
    fig_1dca.savefig(join(save_dir, 'exp_simdissim_exintrinsic1d_ca123.png'), dpi=dpi)
    for figext in ['png', 'eps']:

        fig_2d.savefig(join(save_dir, 'exp_simdissim_exintrinsic2d.%s'%(figext)), dpi=dpi)


def FourCasesAnalysis_hc3(expdf, save_dir):
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
    df_total = getcadf(df_exin, 'CA1') + getcadf(df_exin, 'CA3')
    df_total['case'] = df_total.index
    df_total['ca'] = 'ALL'
    df_exinall = pd.concat([df_exin, df_total], axis=0)
    df_exinall['ntotal'] = df_exinall['nex'] + df_exinall['nin']
    df_exinall['exfrac'] = df_exinall['nex']/df_exinall['ntotal']

    # # # Plotting & Statistical tesing
    stat_fn = 'HC3_fig7_4case.txt'
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
    for ca in ['CA1', 'CA3']:
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
    for ca in ['CA1', 'CA3', 'ALL']:
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
        for ca in ['CA1', 'CA3']:
            onset_dict[ca + str(caseid)] = allpdf[(allpdf['ca']==ca) & (allpdf['case%d'%caseid])]['onset'].to_numpy()
        onset_dict['ALL' + str(caseid)] = allpdf[allpdf['case%d'%caseid]]['onset'].to_numpy()

    for ca in ['CA1', 'CA3', 'ALL']:
        x_mean = np.array([circmean(onset_dict[ca + '%d'%i]) for i in range(1, 5)])
        x_tsd = np.array([np.std(onset_dict[ca + '%d'%i])/np.sqrt(onset_dict[ca + '%d'%i].shape[0]) for i in range(1, 5)])
        ax_onset.errorbar(x=ca_box_pos[ca], y=x_mean, yerr=x_tsd*1.96, fmt='_k', capsize=5, ecolor=ca_c[ca])

        # Test 12
        case12_boxmid = np.mean([ca_box_pos[ca][0], ca_box_pos[ca][1]])
        pval12, table12 = watson_williams(onset_dict['%s1'%(ca)], onset_dict['%s2'%(ca)])
        stat_record(stat_fn, False, 'Onset, %s, Case 1 vs Case 2, Watson-Williams test, %s'%(ca, wwtable2text(table12)))
        ax_onset.text(case12_boxmid-boxw*2, 6, 'p'+p2str(pval12), fontsize=legendsize)
        ax_onset.errorbar(x=case12_boxmid, y=5.8, xerr=boxw/2, c='k', capsize=2.5)


        # Test 12 vs 34
        all_boxmid = np.mean(ca_box_pos[ca])
        onset12 = np.concatenate([onset_dict['%s1'%(ca)], onset_dict['%s2'%(ca)]])
        onset34 = np.concatenate([onset_dict['%s3'%(ca)], onset_dict['%s4'%(ca)]])
        pval1234, table1234 = watson_williams(onset12, onset34)
        stat_record(stat_fn, False, 'Onset, %s, Case 1+2 vs Case 3+4, Watson-Williams test, %s'%(ca, wwtable2text(table1234)))
        ax_onset.text(all_boxmid-boxw*2, 7, 'p'+p2str(pval1234), fontsize=legendsize)
        ax_onset.errorbar(x=all_boxmid, y=6.8, xerr=boxw*1.5, c='k', capsize=2.5)


    ax_onset.set_ylim(0, 3*np.pi)
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






def main():

    # ------------------------------- Single Field -------------------------------
    data_pth = 'results/hc3/single_field_useAllSessions.pickle'
    save_dir = '../result_plots/hc3/single_field_useAllSessions/'
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_pickle(data_pth)
    omniplot_singlefields_hc3(df, save_dir=save_dir)
    # check_pass_nspikes(df=df)
    plot_field_bestprecession_hc3(df=df, save_dir=save_dir)
    plot_both_slope_offset_hc3(df=df, save_dir=save_dir, NShuffles=200)


    # # ------------------------------- Pair Field -------------------------------
    # data_pth = 'results/hc3/pair_field_useAllSessions.pickle'
    # save_dir = 'result_plots/hc3/pair_field_useAllSessions/'
    # os.makedirs(save_dir, exist_ok=True)
    #
    # df = pd.read_pickle(data_pth)
    # omniplot_pairfields_hc3(df, save_dir)
    # plot_kld_hc3(df, save_dir)
    # plot_pair_single_angles_analysis_hc3(df, save_dir)
    # plot_pair_correlation_hc3(df, save_dir=save_dir)
    # plot_exintrinsic_hc3(df, save_dir=save_dir)
    # plot_exintrinsic_concentration_hc3(df, save_dir=save_dir)
    #
    # plot_pairangle_similarity_analysis_hc3(df, save_dir=save_dir)
    #
    # FourCasesAnalysis_hc3(df, save_dir=save_dir)
if __name__ == '__main__':
    main()

