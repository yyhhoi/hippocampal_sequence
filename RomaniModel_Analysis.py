# This script does directionality and spike phase analysis for single fields.
# Only for Simulation's preprocessed data
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from pycircstat import vtest, watson_williams, rayleigh
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import ranksums, chi2_contingency, spearmanr, chisquare, ttest_1samp, linregress, binom_test

from common.linear_circular_r import rcc
from common.utils import load_pickle, stat_record, sigtext, p2str
from common.comput_utils import midedges, circular_density_1d, repeat_arr, unfold_binning_2d, \
    linear_circular_gauss_density, circ_ktest, ranksums, shiftcyc_full2half

from common.visualization import plot_marginal_slices, plot_correlogram, customlegend

from common.script_wrappers import DirectionalityStatsByThresh, permutation_test_average_slopeoffset
from common.shared_vars import total_figw, fontsize, ticksize, legendsize, titlesize, dpi
import warnings
warnings.filterwarnings("ignore")


def omniplot_singlefields_Romani(simdf, ax):


    stat_fn = 'fig8_SIM_single_directionality.txt'
    stat_record(stat_fn, True)

    linew = 0.75

    spike_threshs = np.arange(0, 620, 20)
    stats_getter = DirectionalityStatsByThresh('num_spikes', 'rate_R_pval', 'rate_R')
    linecolor = {'all':'k', 'border':'k', 'nonborder':'k'}
    linestyle = {'all':'solid', 'border':'dotted', 'nonborder':'dashed'}

    # Plot all
    all_dict = stats_getter.gen_directionality_stats_by_thresh(simdf, spike_threshs)
    ax[0].plot(spike_threshs, all_dict['medianR'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[1].plot(spike_threshs, all_dict['sigfrac_shift'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)


    # Plot border
    simdf_b = simdf[simdf['border']].reset_index(drop=True)
    border_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_b, spike_threshs)
    ax[0].plot(spike_threshs, border_dict['medianR'], c=linecolor['border'], linestyle=linestyle['border'], label='B', linewidth=linew)
    ax[1].plot(spike_threshs, border_dict['sigfrac_shift'], c=linecolor['border'], linestyle=linestyle['border'], label='B', linewidth=linew)

    # Plot non-border
    simdf_nb = simdf[~simdf['border']].reset_index(drop=True)
    nonborder_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_nb, spike_threshs)
    ax[0].plot(spike_threshs, nonborder_dict['medianR'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='N-B', linewidth=linew)
    ax[1].plot(spike_threshs, nonborder_dict['sigfrac_shift'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='N-B', linewidth=linew)

    # Plot Fraction
    all_n = simdf.shape[0]
    border_nfrac = border_dict['n']/all_n
    nonborder_nfrac = nonborder_dict['n']/all_n
    ax[2].plot(spike_threshs, all_dict['datafrac'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[2].plot(spike_threshs, border_nfrac, c=linecolor['border'], linestyle=linestyle['border'], label='B', linewidth=linew)
    ax[2].plot(spike_threshs, nonborder_nfrac, c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='N-B', linewidth=linew)


    # Binomial test for all fields
    signum_all, n_all = all_dict['shift_signum'][0], all_dict['n'][0]
    p_binom = binom_test(signum_all, n_all, p=0.05, alternative='greater')
    stat_txt = 'Binomial test, greater than p=0.05, %d/%d, p%s'%(signum_all, n_all, p2str(p_binom))
    stat_record(stat_fn, False, stat_txt)
    # ax[1].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize)

    # Statistical testing
    for idx, ntresh in enumerate(spike_threshs):
        rs_bord_stat, rs_bord_pR = ranksums(border_dict['allR'][idx], nonborder_dict['allR'][idx])

        contin = pd.DataFrame({'border': [border_dict['shift_signum'][idx],
                                          border_dict['shift_nonsignum'][idx]],
                               'nonborder': [nonborder_dict['shift_signum'][idx],
                                             nonborder_dict['shift_nonsignum'][idx]]})
        try:
            chi_stat, chi_pborder, chi_dof, _ = chi2_contingency(contin)
        except ValueError:
            chi_stat, chi_pborder, chi_dof = 0, 1, 1

        border_n, nonborder_n = border_dict['n'][idx], nonborder_dict['n'][idx]
        stat_record(stat_fn, False, 'Sim BorderEffect Frac, Thresh=%d, \chi^2(%d, N=%d)=%0.2f, p%s' % \
                    (ntresh, chi_dof, border_n + nonborder_n, chi_stat, p2str(chi_pborder)))

        mdnR_border, mdnR2_nonrboder = border_dict['medianR'][idx], nonborder_dict['medianR'][idx]
        stat_record(stat_fn, False,
                    'Sim BorderEffect MedianR:%0.2f-%0.2f, Thresh=%d, U(N_{border}=%d, N_{nonborder}=%d)=%0.2f, p%s' % \
                    (mdnR_border, mdnR2_nonrboder, ntresh, border_n, nonborder_n, rs_bord_stat, p2str(rs_bord_pR)))


    # Plotting asthestic
    ax_ylabels = ['Median R', "Sig. Frac.", "Data Frac."]
    for axid in range(3):
        ax[axid].set_xticks([0, 100, 200, 300, 400])
        ax[axid].set_xticklabels(['0', '', '200', '', '400'])

        ax[axid].set_ylabel(ax_ylabels[axid], fontsize=fontsize)
        ax[axid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax[axid].spines['top'].set_visible(False)
        ax[axid].spines['right'].set_visible(False)

    ax[1].set_xlabel('Spike count threshold', fontsize=fontsize)

    customlegend(ax[0], fontsize=legendsize, loc='lower left', bbox_to_anchor=(0.5, 0.5))

    ax[0].set_yticks([0, 0.2, 0.4, 0.6])
    ax[0].set_yticks(np.arange(0, 0.7, 0.1), minor=True)
    ax[1].set_yticks([0, 0.5, 1])
    ax[1].set_yticklabels(['0', '', '1'])
    ax[1].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax[2].set_yticks([0, 0.5, 1])
    ax[2].set_yticklabels(['0', '', '1'])
    ax[2].set_yticks(np.arange(0, 1.1, 0.1), minor=True)


def plot_field_bestprecession_Romani(simdf, ax):
    anglediff_edges = np.linspace(0, np.pi, 50)
    slope_edges = np.linspace(-2, 0, 50)
    offset_edges = np.linspace(0, 2 * np.pi, 50)
    angle_title = r'$d(\theta_{pass}, \theta_{rate})$' + ' (rad)'
    absangle_title  = r'$|d(\theta_{pass}, \theta_{rate})|$' + ' (rad)'
    nap_thresh = 1

    print('Plot field best precession')
    stat_fn = 'fig8_SIM_field_precess.txt'
    stat_record(stat_fn, True)
    allR = []
    allca_ntotal = simdf.shape[0]
    all_shufp = []


    # Stack data
    numpass_mask = simdf['numpass_at_precess'].to_numpy() >= nap_thresh
    numpass_low_mask = simdf['numpass_at_precess_low'].to_numpy() >= nap_thresh
    precess_adiff = []
    precess_nspikes = []
    all_adiff = []
    all_slopes = []

    # Density of Fraction, Spikes and Ratio
    df_this = simdf[numpass_mask & (~simdf['rate_angle'].isna())].reset_index(drop=True)
    for i in range(df_this.shape[0]):
        numpass_at_precess = df_this.loc[i, 'numpass_at_precess']
        if numpass_at_precess < nap_thresh:
            continue
        refangle = df_this.loc[i, 'rate_angle']
        allprecess_df = df_this.loc[i, 'precess_df']
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
    norm_bins[np.isnan(norm_bins)] = 0
    norm_bins[np.isinf(norm_bins)] = 0
    precess_allcount = precess_bins.sum()
    rho, pval = spearmanr(adm, norm_bins)
    pm, pc, pr, ppval, _ = linregress(adm, norm_bins/ norm_bins.sum())
    xdum = np.linspace(adm.min(), adm.max(), 10)
    linew_ax0 = 0.6
    ax[0].step(adm, precess_bins / precess_bins.sum(), color='navy', linewidth=linew_ax0,
               label='Precession')
    ax[0].step(adm, spike_bins / spike_bins.sum(), color='orange', linewidth=linew_ax0, label='Spike')
    ax[0].step(adm, norm_bins / norm_bins.sum(), color='green', linewidth=linew_ax0, label='Ratio')
    ax[0].plot(xdum, xdum*pm+pc, color='green', linewidth=linew_ax0)
    ax[0].set_xticks([0, np.pi / 2, np.pi])
    ax[0].set_xticklabels(['0', '$\pi/2$', '$\pi$'])
    # ax[0].set_ylim([0.01, 0.06])
    ax[0].set_yticks([0, 0.1])
    ax[0].tick_params(labelsize=ticksize)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].set_xlabel(absangle_title, fontsize=fontsize)
    ax[0].set_ylabel('Relative count', fontsize=fontsize, labelpad=5)
    customlegend(ax[0], fontsize=legendsize, bbox_to_anchor=[0, 0.5], loc='lower left')
    ax[0].annotate('p%s'%(p2str(pval)), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=legendsize, color='green')
    stat_record(stat_fn, False, "SIM Spearman's correlation: r_s(%d)=%0.2f, p%s " % (precess_allcount, rho, p2str(pval)))



    # Plot Rate angles vs Precess angles

    nprecessmask = simdf['precess_df'].apply(lambda x: x['precess_exist'].sum()) > 1
    numpass_mask = numpass_mask[nprecessmask]
    numpass_low_mask = numpass_low_mask[nprecessmask]
    simdf2 = simdf[nprecessmask].reset_index(drop=True)
    rateangles = simdf2['rate_angle'].to_numpy()
    precessangles = simdf2['precess_angle'].to_numpy()
    precessangles_low = simdf2['precess_angle_low'].to_numpy()

    ax[1].scatter(rateangles[numpass_mask], precessangles[numpass_mask], marker='.', c='gray', s=2)
    ax[1].plot([0, np.pi], [np.pi, 2 * np.pi], c='k')
    ax[1].plot([np.pi, 2 * np.pi], [0, np.pi], c='k')
    ax[1].set_xlabel(r'$\theta_{rate}$', fontsize=fontsize)
    ax[1].set_xticks([0, np.pi, 2 * np.pi])
    ax[1].set_xticklabels(['$0$', '$\pi$', '$2\pi$'], fontsize=fontsize)
    ax[1].set_yticks([0, np.pi, 2 * np.pi])
    ax[1].set_yticklabels(['$0$', '$\pi$', '$2\pi$'], fontsize=fontsize)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].set_ylabel(r'$\theta_{Precess}$', fontsize=fontsize)


    # Plot Histogram: d(precess, rate)
    mask = (~np.isnan(rateangles)) & (~np.isnan(precessangles) & numpass_mask)
    adiff = cdiff(precessangles[mask], rateangles[mask])
    bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
    bins_norm = bins / np.sum(bins)
    l = bins_norm.max()
    ax[2].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], zorder=0, color='gray')
    linewidth = 1
    mean_angle = shiftcyc_full2half(circmean(adiff))
    ax[2].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k',  zorder=3,  arrowprops=dict(arrowstyle="->"))
    ax[2].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
    ax[2].scatter(0, 0, s=16, c='gray')
    ax[2].spines['polar'].set_visible(False)
    ax[2].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax[2].set_yticks([0, l/2])
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    v_pval, v_stat = vtest(adiff, mu=np.pi)
    ax[2].annotate('p%s'%(p2str(v_pval)), xy=(0.25, 0.95), xycoords='axes fraction', fontsize=legendsize)
    ax[2].annotate(r'$\theta_{rate}$', xy=(0.85, 0.35), xycoords='axes fraction', fontsize=fontsize)
    stat_record(stat_fn, False, 'SIM, d(precess, rate), V(%d)=%0.2f, p%s' % (bins.sum(), v_stat, p2str(v_pval)))
    ax[2].set_ylabel('All\npasses', fontsize=fontsize)

    # Plot Histogram: d(precess_low, rate)
    mask_low = (~np.isnan(rateangles)) & (~np.isnan(precessangles_low) & numpass_low_mask)
    adiff = cdiff(precessangles_low[mask_low], rateangles[mask_low])
    bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
    bins_norm = bins / np.sum(bins)
    l = bins_norm.max()
    ax[3].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], color='gray', zorder=0)
    mean_angle = shiftcyc_full2half(circmean(adiff))
    ax[3].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k',  zorder=3,  arrowprops=dict(arrowstyle="->"))
    ax[3].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
    ax[3].scatter(0, 0, s=16, c='gray')
    ax[3].spines['polar'].set_visible(False)
    ax[3].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax[3].set_yticks([0, l/2])
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])
    v_pval, v_stat = vtest(adiff, mu=np.pi)
    ax[3].annotate('p%s'%(p2str(v_pval)), xy=(0.25, 0.95), xycoords='axes fraction', fontsize=legendsize)
    ax[3].annotate(r'$\theta_{rate}$', xy=(0.85, 0.35), xycoords='axes fraction', fontsize=fontsize)
    stat_record(stat_fn, False, 'SIM, d(precess_low, rate), V(%d)=%0.2f, p%s' % (bins.sum(), v_stat, p2str(v_pval)))
    ax[3].set_ylabel('Low-spike\npasses', fontsize=fontsize)



def plot_both_slope_offset_Romani(simdf, fig, ax, NShuffles=10):

    def norm_div(target_hist, divider_hist):
        target_hist_norm = target_hist / divider_hist.reshape(-1, 1)
        target_hist_norm[np.isnan(target_hist_norm)] = 0
        target_hist_norm[np.isinf(target_hist_norm)] = 0
        target_hist_norm = target_hist_norm / np.sum(target_hist_norm) * np.sum(target_hist)
        return target_hist_norm


    nap_thresh = 1
    selected_adiff = np.linspace(0, np.pi, 6)  # 20
    offset_bound = (0, 2 * np.pi)
    slope_bound = (-2 * np.pi, 0)
    adiff_edges = np.linspace(0, np.pi, 100)
    offset_edges = np.linspace(offset_bound[0], offset_bound[1], 100)
    slope_edges = np.linspace(slope_bound[0], slope_bound[1], 100)


    stat_fn = 'fig8_SIM_slopeoffset.txt'
    stat_record(stat_fn, True, 'Average Phase precession')

    # Construct pass df
    refangle_key = 'rate_angle'
    passdf_dict = {'anglediff':[], 'slope':[], 'onset':[], 'pass_nspikes':[]}
    spikedf_dict = {'anglediff':[], 'phasesp':[]}
    dftmp = simdf[(~simdf[refangle_key].isna()) & (simdf['numpass_at_precess'] >= nap_thresh)].reset_index()

    for i in range(dftmp.shape[0]):
        allprecess_df = dftmp.loc[i, 'precess_df']
        precessdf = allprecess_df[allprecess_df['precess_exist']].reset_index(drop=True)
        numprecess = precessdf.shape[0]
        if numprecess < 1:
            continue
        ref_angle = dftmp.loc[i, refangle_key]
        anglediff_tmp = cdiff(precessdf['mean_anglesp'].to_numpy(), ref_angle)
        phasesp_tmp = np.concatenate(precessdf['phasesp'].to_list())

        passdf_dict['anglediff'].extend(anglediff_tmp)
        passdf_dict['slope'].extend(precessdf['rcc_m'])
        passdf_dict['onset'].extend(precessdf['rcc_c'])
        passdf_dict['pass_nspikes'].extend(precessdf['pass_nspikes'])

        spikedf_dict['anglediff'].extend(repeat_arr(anglediff_tmp, precessdf['pass_nspikes'].to_numpy().astype(int)))
        spikedf_dict['phasesp'].extend(phasesp_tmp)
    passdf = pd.DataFrame(passdf_dict)
    passdf['slope_piunit'] = passdf['slope'] * 2 * np.pi
    spikedf = pd.DataFrame(spikedf_dict)

    absadiff_pass = np.abs(passdf['anglediff'].to_numpy())
    offset = passdf['onset'].to_numpy()
    slope = passdf['slope_piunit'].to_numpy()

    absadiff_spike = np.abs(spikedf['anglediff'].to_numpy())
    phase_spike = spikedf['phasesp'].to_numpy()

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

    stat_record(stat_fn, False, 'LC_Regression SIM Onset-adiff r(%d)=%0.3f, p%s'%(offset_bins.sum(), offset_rho, p2str(offset_p)))
    stat_record(stat_fn, False, 'LC_Regression SIM Slope-adiff r(%d)=%0.3f, p%s'%(slope_bins.sum(), slope_rho, p2str(slope_p)))

    # Density
    offset_xx, offset_yy, offset_zz = linear_circular_gauss_density(offset_adiff, offset_norm,
                                                                    cir_kappa=4 * np.pi, lin_std=0.2, xbins=50,
                                                                    ybins=50, xbound=(0, np.pi),
                                                                    ybound=offset_bound)
    slope_xx, slope_yy, slope_zz = linear_circular_gauss_density(slope_adiff, slope_norm,
                                                                 cir_kappa=4 * np.pi, lin_std=0.2, xbins=50,
                                                                 ybins=50, xbound=(0, np.pi),
                                                                 ybound=slope_bound)


    # # # Marginals
    #
    #
    # # Marginal onset
    # offset_slicerange = (0, 2*np.pi)
    # offset_slicegap = 0.02  # 0.007
    # plot_marginal_slices(ax[0], offset_xx, offset_yy, offset_zz,
    #                      selected_adiff,
    #                      offset_slicerange, offset_slicegap)
    #
    # ax[0].set_xticks([1, 3, 5])
    # ax[0].set_xlim(1, 5)
    # ax[0].set_xlabel('Phase onset (rad)', fontsize=fontsize)
    # ax[0].tick_params(labelsize=ticksize)
    #
    # # Marginal slope
    # slope_slicerange = (-2 * np.pi, 0)
    # slope_slicegap = 0.02  # 0.007
    # plot_marginal_slices(ax[1], slope_xx, slope_yy, slope_zz,
    #                      selected_adiff, slope_slicerange, slope_slicegap)
    # ax[1].set_xticks([-2 * np.pi, -np.pi, 0])
    # ax[1].set_xticklabels(['$-2\pi$', '$-\pi$', '0'])
    # ax[1].set_xlim(-2*np.pi, 0)
    # ax[1].set_xlabel('Slope (rad)', fontsize=fontsize)
    # ax[1].tick_params(labelsize=ticksize)
    # # Colorbar
    # sm = plt.cm.ScalarMappable(cmap=cm.brg,
    #                            norm=plt.Normalize(vmin=selected_adiff.min(), vmax=selected_adiff.max()))
    #
    # # plt.gca().set_visible(False)
    # cb = fig.colorbar(sm, cax=ax[4])
    # cb.set_ticks([0, np.pi / 2, np.pi])
    # cb.set_ticklabels(['0', '', '$\pi$'])
    # cb.set_label(r'$|d(\theta_{pass}, \theta_{precess})|$', fontsize=fontsize, labelpad=-5)

    # # Average precession curves
    low_mask = absadiff_pass < (np.pi / 6)
    high_mask = absadiff_pass > (np.pi - np.pi / 6)
    print('SIM high %d, low %d'%(high_mask.sum(), low_mask.sum()))
    sample_size_high = min(high_mask.sum(), 500)
    sample_size_low = min(low_mask.sum(), 500)
    np.random.seed(1)
    high_ranvec = np.random.choice(high_mask.sum(), size=sample_size_high, replace=False)
    low_ranvec = np.random.choice(low_mask.sum(), size=sample_size_low, replace=False)
    slopes_high, offsets_high = slope[high_mask][high_ranvec], offset[high_mask][high_ranvec]
    slopes_low, offsets_low = slope[low_mask][low_ranvec], offset[low_mask][low_ranvec]

    regress_high, regress_low, pval_slope, pval_offset = permutation_test_average_slopeoffset(
        slopes_high, offsets_high, slopes_low, offsets_low, NShuffles=NShuffles)

    stat_record(stat_fn, False,
                'SIM d=high, rho=%0.2f, p%s, slope=%0.2f, offset=%0.2f' % \
                (regress_high['rho'], p2str(regress_high['p']), regress_high['aopt'] * 2 * np.pi,
                 regress_high['phi0']))
    stat_record(stat_fn, False,
                'SIM d=low, rho=%0.2f, p%s, slope=%0.2f, offset=%0.2f' % \
                (regress_low['rho'], p2str(regress_low['p']), regress_low['aopt'] * 2 * np.pi,
                 regress_low['phi0']))
    stat_record(stat_fn, False,
                'SIM high-low-difference, Slope: p%s; Offset: p%s' % \
                (p2str(pval_slope), p2str(pval_offset)))


    xdum = np.linspace(0, 1, 10)
    high_agg_ydum = 2 * np.pi * regress_high['aopt'] * xdum + regress_high['phi0']
    low_agg_ydum = 2 * np.pi * regress_low['aopt'] * xdum + regress_low['phi0']

    ax[0].plot(xdum, high_agg_ydum, c='lime', label='$|d|>5\pi/6$')
    ax[0].plot(xdum, low_agg_ydum, c='darkblue', label='$|d|<\pi/6$')

    pvalstr1 = '$p_s$'+'%s'%p2str(pval_slope)
    pvalstr2 = '$p_o$'+'%s'%p2str(pval_offset)
    ax[0].text(0.3, np.pi/4+0.2, '%s'%(pvalstr1), fontsize=legendsize)
    ax[0].text(0.3, np.pi/4+0.9, '%s'%(pvalstr2), fontsize=legendsize)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].set_xticks([0, 1])
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(-np.pi/2, np.pi + 0.3)
    ax[0].set_yticks([0, np.pi])
    ax[0].set_yticklabels(['0', '$\pi$'])
    ax[0].tick_params(labelsize=ticksize)
    ax[0].set_xlabel('Position')
    customlegend(ax[0], fontsize=legendsize, loc='lower left', handlelength=0.5, bbox_to_anchor=(0.1, 0.7))
    ax[0].set_ylabel('Phase (rad)', fontsize=fontsize)


    # # Spike phases
    low_mask_sp = absadiff_spike < (np.pi / 6)
    high_mask_sp = absadiff_spike > (np.pi - np.pi / 6)
    phasesph = phase_spike[high_mask_sp]
    phasespl = phase_spike[low_mask_sp]
    fstat, k_pval = circ_ktest(phasesph, phasespl)
    p_ww, ww_table = watson_williams(phasesph, phasespl)
    mean_phasesph = circmean(phasesph)
    mean_phasespl = circmean(phasespl)
    nh, _, _ = ax[1].hist(phasesph, bins=36, density=True, histtype='step', color='lime')
    nl, _, _ = ax[1].hist(phasespl, bins=36, density=True, histtype='step', color='darkblue')
    ml = max(nh.max(), nl.max())
    ax[1].scatter(mean_phasesph, ml*1.1, marker='|', color='lime', linewidth=0.75)
    ax[1].scatter(mean_phasespl, ml*1.1, marker='|', color='darkblue', linewidth=0.75)
    ax[1].annotate('p%s'%(p2str(p_ww)), xy=(0.3, 0.1), xycoords='axes fraction', fontsize=legendsize)
    ax[1].set_xlim(-np.pi, np.pi)
    ax[1].set_xticks([-np.pi, 0, np.pi])
    ax[1].set_xticklabels(['$-\pi$', '0', '$\pi$'])
    ax[1].set_yticks([])
    ax[1].tick_params(labelsize=ticksize)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].set_xlabel('Phase (rad)', fontsize=fontsize)
    ax[1].set_ylabel('Relative\nfrequency', fontsize=fontsize)

    stat_record(stat_fn, False, 'SpikePhase-HighLowDiff Mean_diff p%s' % ( p2str(p_ww)))
    stat_record(stat_fn, False, ww_table.to_string())
    stat_record(stat_fn, False,
                'SpikePhase-HighLowDiff SIM Bartlett\'s test F_{(%d, %d)}=%0.2f, p%s' % \
                (phasesph.shape[0], phasespl.shape[0], fstat, p2str(k_pval)))



def singlefield_analysis_Romani(single_simdf, save_dir):
    fig = plt.figure(figsize=(total_figw, total_figw/1.25))

    ax_baseh = 1/3
    ax_basew = 1/3

    figxshift = 0.032

    x_btw_squeeze = 0.025

    # Directionality
    # 0, 1, 2, 3
    xsqueeze, ysqueeze = 0.15, 0.15

    yshift = 0.05
    ax_direct = [fig.add_axes([0+xsqueeze/2+figxshift+x_btw_squeeze, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew+xsqueeze/2+figxshift, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew*2+xsqueeze/2+figxshift-x_btw_squeeze, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])
                 ]

    # Field's best precession
    # 0, 2
    # 1, 3
    xsqueeze, ysqueeze = 0.15, 0.15
    yshiftall = 0.05
    yshift23 = -0.05
    xshift12, xshift23 = 0, 0
    y_squeeze23 = 0.025
    ax_bestprecess = [fig.add_axes([0+xsqueeze/2+xshift12+figxshift+x_btw_squeeze, 1-ax_baseh*2+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                      fig.add_axes([0+xsqueeze/2+xshift12+figxshift+x_btw_squeeze, 1-ax_baseh*3+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                      fig.add_axes([ax_basew+xsqueeze/2+xshift23+figxshift, 1-ax_baseh*2+ysqueeze/2+yshiftall-y_squeeze23+yshift23, ax_basew-xsqueeze, ax_baseh-ysqueeze], polar=True),
                      fig.add_axes([ax_basew+xsqueeze/2+xshift23+figxshift, 1-ax_baseh*3+ysqueeze/2+yshiftall+y_squeeze23+yshift23, ax_basew-xsqueeze, ax_baseh-ysqueeze], polar=True)]

    # Marginals, aver curve, spike phases
    # 0,
    # 1,
    xsqueeze, ysqueeze = 0.15, 0.15
    xshift01 = 0
    yshiftall = 0.05
    ax_mar = [fig.add_axes([ax_basew*2+xsqueeze/2 + xshift01+figxshift-x_btw_squeeze, 1-ax_baseh*2+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
              fig.add_axes([ax_basew*2+xsqueeze/2 + xshift01+figxshift-x_btw_squeeze, 1-ax_baseh*3+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
              ]


    omniplot_singlefields_Romani(simdf=single_simdf, ax=ax_direct)
    plot_field_bestprecession_Romani(simdf=single_simdf, ax=ax_bestprecess)
    plot_both_slope_offset_Romani(simdf=single_simdf, fig=fig, ax=ax_mar, NShuffles=1000)
    fig.savefig(join(save_dir, 'SIM_single.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'SIM_single.eps'), dpi=dpi)


def omniplot_pairfields_Romani(simdf, ax):
    stat_fn = 'fig9_SIM_pair_directionality.txt'
    stat_record(stat_fn, True)

    linew = 0.75


    spike_threshs = np.arange(0, 420, 20)
    stats_getter = DirectionalityStatsByThresh('num_spikes_pair', 'rate_R_pvalp', 'rate_Rp')
    linecolor = {'all':'k', 'border':'k', 'nonborder':'k'}
    linestyle = {'all':'solid', 'border':'dotted', 'nonborder':'dashed'}

    # Plot all
    all_dict = stats_getter.gen_directionality_stats_by_thresh(simdf, spike_threshs)
    ax[0].plot(spike_threshs, all_dict['medianR'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[1].plot(spike_threshs, all_dict['sigfrac_shift'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)


    # Plot border
    simdf_b = simdf[simdf['border']].reset_index(drop=True)
    border_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_b, spike_threshs)
    ax[0].plot(spike_threshs, border_dict['medianR'], c=linecolor['border'], linestyle=linestyle['border'], label='Border', linewidth=linew)
    ax[1].plot(spike_threshs, border_dict['sigfrac_shift'], c=linecolor['border'], linestyle=linestyle['border'], label='Border', linewidth=linew)

    # Plot non-border
    simdf_nb = simdf[~simdf['border']].reset_index(drop=True)
    nonborder_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_nb, spike_threshs)
    ax[0].plot(spike_threshs, nonborder_dict['medianR'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', linewidth=linew)
    ax[1].plot(spike_threshs, nonborder_dict['sigfrac_shift'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', linewidth=linew)

    # Plot Fraction
    border_nfrac = border_dict['n']/all_dict['n']
    ax[2].plot(spike_threshs, all_dict['datafrac'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[2].plot(spike_threshs, border_nfrac, c=linecolor['border'], linestyle=linestyle['border'], label='Border', linewidth=linew)
    ax[2].plot(spike_threshs, 1-border_nfrac, c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', linewidth=linew)


    # Binomial test for all fields
    signum_all, n_all = all_dict['shift_signum'][0], all_dict['n'][0]
    p_binom = binom_test(signum_all, n_all, p=0.05, alternative='greater')
    stat_txt = 'Binomial test, greater than p=0.05, %d/%d, p%s'%(signum_all, n_all, p2str(p_binom))
    stat_record(stat_fn, False, stat_txt)
    # ax[1].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize)

    # Statistical testing
    for idx, ntresh in enumerate(spike_threshs):
        rs_bord_stat, rs_bord_pR = ranksums(border_dict['allR'][idx], nonborder_dict['allR'][idx])

        contin = pd.DataFrame({'border': [border_dict['shift_signum'][idx],
                                          border_dict['shift_nonsignum'][idx]],
                               'nonborder': [nonborder_dict['shift_signum'][idx],
                                             nonborder_dict['shift_nonsignum'][idx]]})
        try:
            chi_stat, chi_pborder, chi_dof, _ = chi2_contingency(contin)
        except ValueError:
            chi_stat, chi_pborder, chi_dof = 0, 1, 1

        border_n, nonborder_n = border_dict['n'][idx], nonborder_dict['n'][idx]
        stat_record(stat_fn, False, 'Sim BorderEffect Frac, Thresh=%0.2f, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f' % \
                    (ntresh, chi_dof, border_n + nonborder_n, chi_stat, chi_pborder))

        mdnR_border, mdnR2_nonrboder = border_dict['medianR'][idx], nonborder_dict['medianR'][idx]
        stat_record(stat_fn, False,
                    'Sim BorderEffect MedianR:%0.2f-%0.2f, Thresh=%0.2f, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                    (mdnR_border, mdnR2_nonrboder, ntresh, rs_bord_stat, border_n, nonborder_n, rs_bord_pR))




    # Plotting asthestic
    ax_ylabels = ['Median R', "Sig. Frac.", "Data Frac."]
    for axid in range(3):
        ax[axid].set_xticks([0, 100, 200, 300, 400])
        ax[axid].set_xticklabels(['0', '', '200', '', '400'])
        ax[axid].set_ylabel(ax_ylabels[axid], fontsize=fontsize)
        ax[axid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax[axid].spines['top'].set_visible(False)
        ax[axid].spines['right'].set_visible(False)

    ax[1].set_xlabel('Spike count threshold', fontsize=fontsize)

    customlegend(ax[0], fontsize=legendsize, loc='lower left', bbox_to_anchor=(0.2, 0.5))

    ax[0].set_yticks([0, 0.2, 0.4, 0.6])
    ax[0].set_yticks(np.arange(0, 0.7, 0.1), minor=True)
    ax[1].set_yticks([0, 0.5, 1])
    ax[1].set_yticklabels(['0', '', '1'])
    ax[1].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax[2].set_yticks([0, 0.5, 1])
    ax[2].set_yticklabels(['0', '', '1'])
    ax[2].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax[3].axis('off')



def plot_pair_correlation_Romani(simdf, ax):
    stat_fn = 'fig9_SIM_paircorr.txt'
    stat_record(stat_fn, True)

    linew = 0.75
    markersize = 1

    # A->B
    ax[0], x, y, regress = plot_correlogram(ax=ax[0], df=simdf, tag='', direct='A->B', color='gray', alpha=1,
                                            markersize=markersize, linew=linew)

    nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
    stat_record(stat_fn, False, 'SIM A->B, y = %0.2fx + %0.2f, rho=%0.2f, n=%d, p=%0.4f' % \
                (regress['aopt'] * 2 * np.pi, regress['phi0'], regress['rho'], nsamples, regress['p']))

    # B->A
    ax[1], x, y, regress = plot_correlogram(ax=ax[1], df=simdf, tag='', direct='B->A', color='gray', alpha=1,
                                            markersize=markersize, linew=linew)

    nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
    stat_record(stat_fn, False, 'SIM B->A, y = %0.2fx + %0.2f, rho=%0.2f, n=%d, p=%0.4f' % \
                (regress['aopt'] * 2 * np.pi, regress['phi0'], regress['rho'], nsamples, regress['p']))


    ax[0].set_ylabel('Phase shift (rad)', fontsize=fontsize)
    ax[0].get_shared_y_axes().join(ax[0], ax[1])
    ax[1].set_yticklabels(['']*4)

    for ax_each in [ax[0], ax[1]]:
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    ax[0].set_title('A>B', fontsize=fontsize)
    ax[1].set_title('B>A', fontsize=fontsize)


def plot_exintrinsic_Romani(simdf, ax):
    stat_fn = 'fig9_SIM_exintrinsic.txt'
    stat_record(stat_fn, True)

    ms = 0.2
    # Filtering
    smallerdf = simdf[(~simdf['overlap_ratio'].isna())]


    corr_overlap = smallerdf['overlap_plus'].to_numpy()
    corr_overlap_flip = smallerdf['overlap_minus'].to_numpy()
    corr_overlap_ratio = smallerdf['overlap_ratio'].to_numpy()

    # 1-sample chisquare test
    n_ex = np.sum(corr_overlap_ratio > 0)
    n_in = np.sum(corr_overlap_ratio <= 0)
    n_total = n_ex + n_in


    chistat, pchi = chisquare([n_ex, n_in])
    stat_record(stat_fn, False, 'SIM, %d/%d, \chi^2_{(dof=%d, n=%d)}=%0.2f, p=%0.4f' % \
                (n_ex, n_total, 1, n_total, chistat, pchi))
    # 1-sample t test
    mean_ratio = np.mean(corr_overlap_ratio)
    ttest_stat, p_1d1samp = ttest_1samp(corr_overlap_ratio, 0)
    stat_record(stat_fn, False, 'SIM, mean=%0.4f, t(%d)=%0.2f, p=%0.4f' % \
                (mean_ratio, n_total-1, ttest_stat, p_1d1samp))
    # Plot scatter 2d
    ax[0].scatter(corr_overlap_flip, corr_overlap, marker='.', s=ms, c='gray')
    ax[0].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
    ax[0].annotate('%0.2f'%(n_ex/n_total), xy=(0.05, 0.17), xycoords='axes fraction', size=legendsize, color='r')
    ax[0].annotate('p%s'%(p2str(pchi)), xy=(0.05, 0.025), xycoords='axes fraction', size=legendsize)
    ax[0].set_xlabel('Extrinsicity', fontsize=fontsize)
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].tick_params(axis='both', which='major', labelsize=ticksize)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel('Intrinsicity', fontsize=fontsize)

    # Plot 1d histogram
    edges = np.linspace(-1, 1, 75)
    width = edges[1]-edges[0]
    (bins, _, _) = ax[1].hist(corr_overlap_ratio, bins=edges, color='gray',
                              density=True, histtype='stepfilled')
    ax[1].plot([mean_ratio, mean_ratio], [0, bins.max()], c='k')
    ax[1].annotate('$\mu$'+ '=%0.3f\np%s'%(mean_ratio, p2str(p_1d1samp)), xy=(0.2, 0.8), xycoords='axes fraction', fontsize=legendsize)
    ax[1].set_xticks([-0.5, 0, 0.5])
    ax[1].set_yticks([0, 0.1/width] )
    ax[1].set_yticklabels(['0', '0.1'])
    ax[1].set_ylim(0, 6.5)
    ax[1].set_xlabel('Extrinsicity - Intrinsicity', fontsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=ticksize)
    ax[1].set_xlim(-0.5, 0.5)
    ax[1].set_ylabel('Normalized counts', fontsize=fontsize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

def pairfield_analysis_Romani(pair_simdf, save_dir):
    fig = plt.figure(figsize=(total_figw, total_figw/1.75))

    ax_baseh = 1/2
    ax_basew = 1/4

    # Directionality
    # 0, 1, 2, 3
    xsqueeze, ysqueeze = 0.1, 0.23
    xshift, yshift = 0.032, 0.05
    ax_direct = [fig.add_axes([0+xsqueeze/2+xshift, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew+xsqueeze/2+xshift, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew*2+xsqueeze/2+xshift, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew*3+xsqueeze/2+xshift, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])]

    # Pair Correlation
    # 0, 1
    xsqueeze, ysqueeze = 0.1, 0.23
    xbtw_squeeze = 0.05
    xshift = 0.032 - 0.05/2
    yshift = 0.025
    ax_paircorr = [fig.add_axes([0+xsqueeze/2+xbtw_squeeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                   fig.add_axes([ax_basew+xsqueeze/2-xbtw_squeeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])]
    fig.text(0.185, 0.025, 'Field Overlap', fontsize=fontsize)

    # Ex-intrinsicity
    # 0, 1
    xsqueeze, ysqueeze = 0.1, 0.23
    xshift = -0.025
    yshift = 0.025
    ax_exin = [fig.add_axes([ax_basew*2+xsqueeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
               fig.add_axes([ax_basew*3+xsqueeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])]


    omniplot_pairfields_Romani(simdf=pair_simdf, ax=ax_direct)
    plot_pair_correlation_Romani(simdf=pair_simdf, ax=ax_paircorr)
    plot_exintrinsic_Romani(simdf=pair_simdf, ax=ax_exin)
    fig.savefig(join(save_dir, 'SIM_pair.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'SIM_pair.eps'), dpi=dpi)


def main():
    save_dir = 'result_plots/sim'
    single_simdf = load_pickle('results/sim/singlefield_df_square_sub04_AllChunk.pickle')
    singlefield_analysis_Romani(single_simdf, save_dir=save_dir)

    # pair_simdf = load_pickle('results/sim/pairfield_df_square_sub025_AllChunk.pickle')
    # pair_simdf['border'] = (pair_simdf['border1'] & pair_simdf['border2'])
    # pairfield_analysis_Romani(pair_simdf, save_dir=save_dir)

if __name__ == '__main__':
    main()