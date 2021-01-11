# This script does directionality and spike phase analysis for single fields.
# Only for Simulation's preprocessed data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from pycircstat import vtest, watson_williams
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import ranksums, chi2_contingency, spearmanr, chisquare, ttest_1samp

from common.linear_circular_r import rcc
from common.utils import load_pickle, stat_record, sigtext, p2str
from common.comput_utils import midedges, circular_density_1d, repeat_arr, unfold_binning_2d, linear_circular_gauss_density, circ_ktest

from common.visualization import plot_marginal_slices, plot_correlogram, customlegend

from common.script_wrappers import DirectionalityStatsByThresh, permutation_test_average_slopeoffset


fontsize = 9
ticksize = 8
legendsize = 8
titlesize = 9 + 2
figl = 1.75


def omniplot_singlefields_sim(simdf, save_dir=None):
    stat_fn = 'fig9_SIM_single_directionality.txt'
    stat_record(stat_fn, True)

    spike_threshs = np.arange(0, 801, 50)
    stats_getter = DirectionalityStatsByThresh('num_spikes', 'win_pval_mlm', 'shift_pval_mlm', 'fieldR_mlm')
    linecolor = {'all':'k', 'border':'k', 'nonborder':'k'}
    linestyle = {'all':'solid', 'border':'dotted', 'nonborder':'dashed'}
    # Initialize subplots
    fig, ax = plt.subplots(1, 3, figsize=(figl*3, figl), sharex=True)

    # Plot all
    all_dict = stats_getter.gen_directionality_stats_by_thresh(simdf, spike_threshs)
    ax[0].plot(spike_threshs, all_dict['medianR'], c=linecolor['all'], linestyle=linestyle['all'], label='All')
    ax[1].plot(spike_threshs, all_dict['sigfrac_shift'], c=linecolor['all'], linestyle=linestyle['all'], label='All', alpha=0.7)


    # Plot border
    simdf_b = simdf[simdf['border']].reset_index(drop=True)
    border_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_b, spike_threshs)
    ax[0].plot(spike_threshs, border_dict['medianR'], c=linecolor['border'], linestyle=linestyle['border'], label='Border')
    ax[1].plot(spike_threshs, border_dict['sigfrac_shift'], c=linecolor['border'], linestyle=linestyle['border'], label='Border', alpha=0.7)

    # Plot non-border
    simdf_nb = simdf[~simdf['border']].reset_index(drop=True)
    nonborder_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_nb, spike_threshs)
    ax[0].plot(spike_threshs, nonborder_dict['medianR'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border')
    ax[1].plot(spike_threshs, nonborder_dict['sigfrac_shift'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', alpha=0.7)

    # Plot Fraction
    border_nfrac = border_dict['n']/all_dict['n']
    ax[2].plot(spike_threshs, all_dict['datafrac'], c=linecolor['all'], linestyle=linestyle['all'], label='All')
    ax[2].plot(spike_threshs, border_nfrac, c=linecolor['border'], linestyle=linestyle['border'], label='Border')
    ax[2].plot(spike_threshs, 1-border_nfrac, c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border')

    # Plotting asthestic
    ax_ylabels = ['Median R', "Significant Fraction", "Data Fraction"]
    for axid in range(ax.shape[0]):
        ax[axid].set_xticks([0, 400, 800])
        ax[axid].set_ylabel(ax_ylabels[axid], fontsize=fontsize)
        ax[axid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax[axid].set_xlabel('Spike count threshold', fontsize=fontsize)
        ax[axid].spines['top'].set_visible(False)
        ax[axid].spines['right'].set_visible(False)

    customlegend(ax[1], fontsize=legendsize-1, loc='upper right')
    ax[0].set_yticks([0.03, 0.06])
    ax[1].set_yticks([0])
    ax[2].set_yticks([0, 0.5, 1])


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


    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'sim_single_directionality.png'), dpi=300)




class SingleBestAngleAnalyzer_SIM:
    def __init__(self, singlefield_df, plot_dir):
        self.singlefield_df = singlefield_df
        self.plot_dir = plot_dir
        self.anglediff_edges = np.linspace(0, np.pi, 50)
        self.slope_edges = np.linspace(-2, 0, 50)
        self.offset_edges = np.linspace(0, 2 * np.pi, 50)
        self.angle_title = r'$d(\theta_{pass}, \theta_{rate})$' + ' (rad)'


    def plot_overall_bestprecession(self):
        print('Plot overall best precession')
        stat_fn = 'fig9_SIM_overall_precess.txt'
        stat_record(stat_fn, True)
        fig_basic, ax_basic = plt.subplots(figsize=(figl+0.1, figl), sharex=True, sharey=True)


        # Re-organize & filter
        anglediff, pass_nspikes = self.stack_anglediff(self.singlefield_df)
        anglediff = np.abs(anglediff)

        # Plain histogram
        adiff_bins = np.linspace(0, np.pi, 37)
        adm = midedges(adiff_bins)
        spike_bins = np.zeros(adm.shape[0])
        pass_bins = np.zeros(adm.shape[0])
        for i in range(adm.shape[0]):
            pass_mask = (anglediff >= adiff_bins[i]) & (anglediff < adiff_bins[i+1])
            spike_bins[i] = np.sum(pass_nspikes[pass_mask])
            pass_bins[i] = np.sum(pass_mask)
        norm_bins = (pass_bins/spike_bins)

        rho, pval = spearmanr(adm, norm_bins)
        rhotxt = 'p%s' % (p2str(pval))

        ax_basic.step(adm, pass_bins / pass_bins.sum(), color='darkblue', alpha=0.7, linewidth=1, label='Precession')
        ax_basic.step(adm, spike_bins / spike_bins.sum(), color='darkorange', alpha=0.7, linewidth=1, label='Spike')
        ax_basic.step(adm, norm_bins / norm_bins.sum(), color='darkgreen', alpha=0.7, linewidth=1, label='Precession/spike')
        ax_basic.text(1.7, 0.04, rhotxt, fontsize=legendsize-1, color='green')

        customlegend(ax_basic, fontsize=legendsize-1)

        stat_record(stat_fn, False, 'SIM pass_num=%d, spike_num=%d, rho=%0.2f, pval=%0.4f' % \
                    (anglediff.shape[0], pass_nspikes.sum(), rho, pval))


        ax_basic.set_xticks([0, np.pi / 2, np.pi])
        ax_basic.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        ax_basic.set_yticks([0, 0.04])
        ax_basic.tick_params(axis='both', which='major', labelsize=ticksize, direction='in')
        ax_basic.set_ylim(0, 0.045)

        ax_basic.set_ylabel('Normalized counts', fontsize=fontsize)
        ax_basic.set_xlabel(self.angle_title, fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'sim_overall_bestprecession.png'), dpi=300)

    def plot_field_bestprecession(self):
        print('Plot field best precession')
        stat_fn = 'fig9_SIM_field_precess.txt'
        stat_record(stat_fn, True)

        fig_R, ax_R = plt.subplots(figsize=(figl, figl))

        fig_2Dscatter, ax_2Dscatter = plt.subplots(figsize=(figl, figl), sharey=True)

        fig_pishift = plt.figure(figsize=(figl*2, figl))
        ax_pishift = np.array(
            [fig_pishift.add_subplot(1, 2, i+1, polar=True) for i in range(2)]
        )

        fid = 0

        data = {'field': [], 'precess': [], 'precess_low':[], 'R':[], 'shufp':[]}

        # Re-organize & filter
        for i in range(self.singlefield_df.shape[0]):

            precess_info = self.singlefield_df.loc[i, 'precess_info']
            if precess_info is None:
                continue
            fieldangle_mlm, precess_angle = self.singlefield_df.loc[i, ['fieldangle_mlm', 'precess_angle']]
            precess_angle_low = self.singlefield_df.loc[i, 'precess_angle_low']
            data['field'].append(fieldangle_mlm)
            data['precess'].append(precess_angle)
            data['precess_low'].append(precess_angle_low)
            data['R'].append(precess_info['R'])
            data['shufp'].append(precess_info['pval'])


            fid += 1



        # Plot Scatter: field vs precess angles
        ax_2Dscatter.scatter(data['field'], data['precess'], alpha=0.2, s=8, c='gray')
        ax_2Dscatter.plot([0, np.pi], [np.pi, 2 * np.pi], c='k')
        ax_2Dscatter.plot([np.pi, 2 * np.pi], [0, np.pi], c='k')
        ax_2Dscatter.set_xlabel(r'$\theta_{rate}$', fontsize=fontsize)
        ax_2Dscatter.set_xticks([0, np.pi, 2 * np.pi])
        ax_2Dscatter.set_xticklabels(['$0$', '$\pi$', '$2\pi$'])
        ax_2Dscatter.set_yticks([0, np.pi, 2 * np.pi])
        ax_2Dscatter.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
        ax_2Dscatter.tick_params(labelsize=ticksize, direction='inout')
        ax_2Dscatter.set_ylabel(r'$\theta_{Precess}$', fontsize=fontsize)
        fig_2Dscatter.tight_layout()
        fig_2Dscatter.savefig(os.path.join(self.plot_dir, 'sim_field_precess_2Dscatter.png'), dpi=300)


        # Plot Histogram: d(precess, rate)
        fieldangles = np.array(data['field'])
        precessangles = np.array(data['precess'])
        nanmask = (~np.isnan(fieldangles)) & (~np.isnan(precessangles))
        adiff = cdiff(precessangles[nanmask], fieldangles[nanmask])

        bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
        bins_norm = bins / np.sum(bins)
        l = bins_norm.max()
        ax_pishift[0].bar(midedges(edges), bins_norm, width=edges[1] - edges[0],
                                alpha=0.5, color='gray')

        mean_angle = circmean(adiff)
        mean_angle = np.mod(mean_angle + np.pi, 2 * np.pi) - np.pi
        ax_pishift[0].plot([mean_angle, mean_angle], [0, l], c='k', linestyle='dashed')
        ax_pishift[0].plot([0, 0], [0, l], c='k')
        ax_pishift[0].scatter(0, 0, s=16, c='gray')
        ax_pishift[0].text(0.18, l * 0.4, r'$\theta_{rate}$', fontsize=fontsize+2)
        ax_pishift[0].spines['polar'].set_visible(False)
        ax_pishift[0].set_yticklabels([])
        ax_pishift[0].set_xticklabels([])
        ax_pishift[0].grid(False)
        ax_pishift[0].set_ylabel('All passes', fontsize=fontsize)
        # ax_pishift[0].yaxis.labelpad = 5

        v_pval, v_stat = vtest(adiff, mu=np.pi)
        ax_pishift[0].text(x=0.01, y=0.95, s='p%s'%(p2str(v_pval)), fontsize=legendsize,
                           transform=ax_pishift[0].transAxes)
        stat_record(stat_fn, False, 'SIM d(precess, rate), V=%0.2f, n=%d, p=%0.4f' % (v_stat, bins.sum(),
                                                                                      v_pval))

        # Plot Histogram: d(precess_low, rate)
        fieldangles = np.array(data['field'])
        precessangles_low = np.array(data['precess_low'])
        nanmask = (~np.isnan(fieldangles)) & (~np.isnan(precessangles_low))
        adiff = cdiff(precessangles_low[nanmask], fieldangles[nanmask])
        bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
        bins_norm = bins / np.sum(bins)
        l = bins_norm.max()
        ax_pishift[1].bar(midedges(edges), bins_norm, width=edges[1] - edges[0],
                                alpha=0.5, color='gray')
        mean_angle = circmean(adiff)
        mean_angle = np.mod(mean_angle + np.pi, 2 * np.pi) - np.pi
        ax_pishift[1].plot([mean_angle, mean_angle], [0, l], c='k', linestyle='dashed')
        ax_pishift[1].plot([0, 0], [0, l], c='k')
        ax_pishift[1].scatter(0, 0, s=16, c='gray')
        ax_pishift[1].text(0.18, l * 0.4, r'$\theta_{rate}$', fontsize=fontsize+2)

        ax_pishift[1].spines['polar'].set_visible(False)
        ax_pishift[1].set_yticklabels([])
        ax_pishift[1].set_xticklabels([])
        ax_pishift[1].grid(False)
        ax_pishift[1].set_ylabel('Low-spike passes', fontsize=fontsize)


        v_pval, v_stat = vtest(adiff, mu=np.pi)
        ax_pishift[1].text(x=0.01, y=0.95, s='p%s'%(p2str(v_pval)), fontsize=legendsize,
                           transform=ax_pishift[1].transAxes)
        stat_record(stat_fn, False, 'SIM d(precess_low, rate), V=%0.2f, n=%d, p=%0.4f' % (v_stat, bins.sum(),
                                                                                          v_pval))
        fig_pishift.tight_layout()
        fig_pishift.savefig(os.path.join(self.plot_dir, 'sim_field_precess_pishift.png'), dpi=300)


        # Export stats: Sig fraction
        all_shufp = data['shufp']
        ca_shufp = np.array(data['shufp'])
        sig_num = int(np.sum(ca_shufp < 0.05))
        sig_percent = np.mean(ca_shufp < 0.05)

        stat_record(stat_fn, False, 'Shufpval_sigfrac SIM frac=%d/%d, percentage=%0.2f' % \
                    (sig_num, ca_shufp.shape[0], sig_percent))

        # Plot R
        all_R = np.array(data['R'])
        rbins, redges = np.histogram(all_R, bins=50)
        rbinsnorm = np.cumsum(rbins) / rbins.sum()
        ax_R.plot(midedges(redges), rbinsnorm)

        ax_R.set_xlabel('R', fontsize=fontsize)
        ax_R.set_ylabel('Cumulative density', fontsize=fontsize)
        ax_R.tick_params(axis='both', which='major', labelsize=fontsize, direction='inout')
        ax_R.set_yticks([0, 0.5, 1])
        fig_R.tight_layout()
        fig_R.savefig(os.path.join(self.plot_dir, 'sim_field_precess_R.png'), dpi=300)




    def plot_both_slope_offset(self):

        stat_fn = 'fig9_SIM_slopeoffset.txt'
        stat_record(stat_fn, True)
        density_figsize = (figl+0.5, figl*2)
        slice_figsize = (figl+0.5, figl*2)

        selected_adiff = np.linspace(0, np.pi, 6)  # 20
        adiff_ticks = [0, np.pi / 2, np.pi]
        adiff_ticksl = ['0', '$\pi$', '$2\pi$']
        adiff_label = r'$|d(\theta_{pass}, \theta_{precess})|$'

        offset_label = 'Onset phase (rad)'
        offset_slicerange = (0, 2*np.pi)
        offset_bound = (0, 2 * np.pi)
        offset_xticks = [0, np.pi, 2*np.pi]
        offset_xticksl = ['0', '$\pi$', '$2\pi$']
        offset_slicegap = 0.017  # 0.007

        slope_label = 'Slope ' + '$(rad)$'
        slope_slicerange = (-2 * np.pi, 0)
        slope_bound = (-2 * np.pi, 0)
        slope_xticks = [-2 * np.pi, -np.pi, 0]
        slope_xticksl = ['$-2\pi$', '$-\pi$', '0']
        slope_slicegap = 0.01  # 0.007

        adiff_edges = np.linspace(0, np.pi, 500)
        offset_edges = np.linspace(offset_bound[0], offset_bound[1], 500)
        slope_edges = np.linspace(slope_bound[0], slope_bound[1], 500)

        fig_density, ax_density = plt.subplots(2, 1, figsize=density_figsize, sharex='col', sharey='row')
        fig_slices, ax_slices = plt.subplots(2, 1, figsize=slice_figsize, sharex='row', sharey='row')

        anglediff, pass_nspikes = self.stack_anglediff(self.singlefield_df, precess_ref=True)
        anglediff = np.abs(anglediff)
        slope = self.stack_key(self.singlefield_df, 'rcc_m')
        slope = slope * 2 * np.pi
        offset = self.stack_key(self.singlefield_df, 'rcc_c')

        # Expand sample according to spikes
        anglediff_spikes = repeat_arr(anglediff, pass_nspikes)

        # 1D spike hisotgram
        spikes_bins, spikes_edges = np.histogram(anglediff_spikes, bins=adiff_edges)

        # 2D slope/offset histogram
        offset_bins, offset_xedges, offset_yedges = np.histogram2d(anglediff, offset,
                                                                   bins=(adiff_edges, offset_edges))
        slope_bins, slope_xedges, slope_yedges = np.histogram2d(anglediff, slope,
                                                                bins=(adiff_edges, slope_edges))

        offset_normbins = self.norm_div(offset_bins, spikes_bins)
        slope_normbins = self.norm_div(slope_bins, spikes_bins)

        # Unbinning
        offset_xedm, offset_yedm = midedges(offset_xedges), midedges(offset_yedges)
        slope_xedm, slope_yedm = midedges(slope_xedges), midedges(slope_yedges)
        offset_adiff, offset_norm = unfold_binning_2d(offset_normbins, offset_xedm, offset_yedm)
        slope_adiff, slope_norm = unfold_binning_2d(slope_normbins, slope_xedm, slope_yedm)

        # Linear-circular regression
        regress = rcc(offset_adiff, offset_norm)
        offset_m, offset_c, offset_rho, offset_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']
        regress = rcc(slope_adiff, slope_norm)
        slope_m, slope_c, slope_rho, slope_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']
        slope_c = slope_c - 2 * np.pi

        stat_record(stat_fn, False,
                    'Offset v.s. D(pass, precess), y=2pi(%0.2f)+%0.2f, Linear-circular regression, rho=%0.2f, p=%0.4f'% \
                    (offset_m, offset_c, offset_rho, offset_p))
        stat_record(stat_fn, False,
                    'Slope v.s. D(pass, precess), y=2pi(%0.2f)+%0.2f, Linear-circular regression, rho=%0.2f, p=%0.4f' % \
                    (slope_m, slope_c, slope_rho, slope_p))

        # Density

        offset_xx, offset_yy, offset_zz = linear_circular_gauss_density(offset_adiff, offset_norm,
                                                                        cir_kappa=4 * np.pi, lin_std=0.2, xbins=50,
                                                                        ybins=50, xbound=(0, np.pi),
                                                                        ybound=offset_bound)
        slope_xx, slope_yy, slope_zz = linear_circular_gauss_density(slope_adiff, slope_norm,
                                                                     cir_kappa=4 * np.pi, lin_std=0.2, xbins=50,
                                                                     ybins=50, xbound=(0, np.pi),
                                                                     ybound=slope_bound)
        # ax_density[0].pcolormesh(offset_xx, offset_yy, offset_zz)

        # Plot offset density
        cmap = 'Blues'
        ax_density[0].hist2d(offset_adiff, offset_norm,
                                   bins=(np.linspace(0, np.pi, 36), np.linspace(0, 2 * np.pi, 36)), density=True,
                                   cmap=cmap)
        regressed = (offset_c + offset_xedm * offset_m)
        ax_density[0].plot(offset_xedm, regressed, c='k')
        ax_density[0].text(np.pi/3, regressed.max() + 0.5, 'p%s'%p2str(offset_p), fontsize=legendsize)
        ax_density[0].set_xticks(adiff_ticks)
        ax_density[0].set_xticklabels(adiff_ticksl)
        ax_density[0].set_yticks(offset_xticks)
        ax_density[0].set_yticklabels(offset_xticksl)
        ax_density[0].tick_params(labelsize=ticksize, direction='inout')
        ax_density[0].set_ylabel(offset_label, fontsize=fontsize)

        # Plot slope density
        ax_density[1].hist2d(slope_adiff, slope_norm,
                                   bins=(np.linspace(0, np.pi, 36), np.linspace(-2 * np.pi, 0, 36)), density=True,
                                   cmap=cmap)
        regressed = (slope_c + slope_xedm * slope_m)
        # ax_density[1].pcolormesh(slope_xx, slope_yy, slope_zz)
        ax_density[1].plot(slope_xedm, regressed, c='k')
        ax_density[1].text(np.pi/3, regressed.max() + 0.5, 'p%s'%p2str(slope_p), fontsize=legendsize)
        ax_density[1].set_xticks(adiff_ticks)
        ax_density[1].set_xticklabels(adiff_ticksl)
        ax_density[1].set_yticks(slope_xticks)
        ax_density[1].set_yticklabels(slope_xticksl)
        ax_density[1].tick_params(labelsize=ticksize, direction='inout')
        ax_density[1].set_xlabel(adiff_label, fontsize=fontsize)
        ax_density[1].set_ylabel(slope_label, fontsize=fontsize)

        ax_slices[0], _ = plot_marginal_slices(ax_slices[0], offset_xx, offset_yy, offset_zz,
                                                  selected_adiff,
                                                  offset_slicerange, offset_slicegap)
        ax_slices[0].set_xticks(offset_xticks)
        ax_slices[0].set_xticklabels(offset_xticksl)
        ax_slices[0].set_xlabel(offset_label, fontsize=fontsize)
        ax_slices[0].tick_params(labelsize=ticksize, direction='inout')

        ax_slices[1], _ = plot_marginal_slices(ax_slices[1], slope_xx, slope_yy, slope_zz,
                                                  selected_adiff, slope_slicerange, slope_slicegap)

        ax_slices[1].set_xticks(slope_xticks)
        ax_slices[1].set_xticklabels(slope_xticksl)
        ax_slices[1].tick_params(labelsize=ticksize, direction='inout')
        ax_slices[1].set_xlabel(slope_label, fontsize=fontsize)


        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cm.brg,
                                   norm=plt.Normalize(vmin=selected_adiff.min(), vmax=selected_adiff.max()))
        fig_colorbar = plt.figure(figsize=slice_figsize)
        fig_colorbar.subplots_adjust(right=0.8)
        cbar_ax = fig_colorbar.add_axes([0.7, 0.15, 0.03, 0.7])
        cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
        cb.set_ticks(adiff_ticks)
        cb.set_ticklabels(adiff_ticksl)
        cb.set_label(adiff_label, fontsize=fontsize, rotation=90)




        for ax in ax_slices.flatten():
            ax.axes.get_yaxis().set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)


        # Others
        fig_density.tight_layout()
        fig_density.savefig(os.path.join(self.plot_dir, 'sim_density.png'), dpi=300)
        fig_slices.tight_layout()
        fig_slices.savefig(os.path.join(self.plot_dir, 'sim_slices.png'), dpi=300)
        # fig_colorbar.tight_layout()
        fig_colorbar.savefig(os.path.join(self.plot_dir, 'sim_adiff_colorbar.png'), dpi=300)
        return None

    def plot_average_precession(self, NShuffles=1000):

        fig_avercurve, ax_avercurve = plt.subplots(figsize=(figl, figl), sharey=True)
        fig_phasesp, ax_phasesp = plt.subplots(figsize=(figl, figl), sharey=True)

        stat_fn = 'fig9_SIM_average_slopeoffset.txt'
        stat_record(stat_fn, True, 'Average Phase precession')

        anglediff, pass_nspikes = self.stack_anglediff(self.singlefield_df, precess_ref=True)

        anglediff = np.abs(anglediff)
        slopes = self.stack_key(self.singlefield_df, 'rcc_m')
        offsets = self.stack_key(self.singlefield_df, 'rcc_c')

        phasesp = np.concatenate(self.stack_key(self.singlefield_df, 'phasesp'))

        low_mask = anglediff < (np.pi / 6)
        high_mask = anglediff > (np.pi - np.pi / 6)

        sample_size = 500
        np.random.seed(1)
        high_ranvec = np.random.choice(high_mask.sum(), size=sample_size)
        low_ranvec = np.random.choice(low_mask.sum(), size=sample_size)

        anglediff_spikes = repeat_arr(anglediff, pass_nspikes)
        low_mask_sp = anglediff_spikes < (np.pi / 6)
        high_mask_sp = anglediff_spikes > (np.pi - np.pi / 6)
        phasesph = phasesp[high_mask_sp]
        phasespl = phasesp[low_mask_sp]

        slopes_high, offsets_high = slopes[high_mask][high_ranvec], offsets[high_mask][high_ranvec]
        slopes_low, offsets_low = slopes[low_mask][low_ranvec], offsets[low_mask][low_ranvec]

        regress_high, regress_low, pval_slope, pval_offset = permutation_test_average_slopeoffset(
            slopes_high, offsets_high, slopes_low, offsets_low, NShuffles=NShuffles)

        stat_record(stat_fn, False,
                    'SIM d=high, rho=%0.2f, p=%0.4f, slope=%0.2f, offset=%0.2f' % \
                    (regress_high['rho'], regress_high['p'], regress_high['aopt'] * 2 * np.pi,
                     regress_high['phi0']))
        stat_record(stat_fn, False,
                    'SIM d=low, rho=%0.2f, p=%0.4f, slope=%0.2f, offset=%0.2f' % \
                    (regress_low['rho'], regress_low['p'], regress_low['aopt'] * 2 * np.pi,
                     regress_low['phi0']))
        stat_record(stat_fn, False,
                    'SIM high-low-difference, Slope: p=%0.4f; Offset: p=%0.4f' % \
                    (pval_slope, pval_offset))

        xdum = np.linspace(0, 1, 10)
        high_agg_ydum = 2 * np.pi * regress_high['aopt'] * xdum + regress_high['phi0']
        low_agg_ydum = 2 * np.pi * regress_low['aopt'] * xdum + regress_low['phi0']

        # Compare high vs low
        ax_avercurve.plot(xdum, high_agg_ydum, c='lime', label='$|d|>5\pi/6$')
        ax_avercurve.plot(xdum, low_agg_ydum, c='darkblue', label='$|d|<\pi/6$')
        ax_avercurve.text(0.05, -np.pi + 0.5, 'Slope\n  p%s\nOffset\n  p%s'% \
                          (p2str(pval_slope), p2str(pval_offset)), fontsize=legendsize)

        customlegend(ax_avercurve, fontsize=legendsize, loc='upper right')

        #
        aedges = np.linspace(-np.pi, np.pi, 36)

        fstat, k_pval = circ_ktest(phasesph, phasespl)

        p_ww, ww_table = watson_williams(phasesph, phasespl)
        ax_phasesp.hist(phasesph, bins=aedges, density=True, histtype='step', color='lime')
        ax_phasesp.hist(phasespl, bins=aedges, density=True, histtype='step', color='darkblue')
        ax_phasesp.text(0.2, 0.15, 'Bartlett\'s\np%s'%(p2str(k_pval)), transform=ax_phasesp.transAxes, fontsize=fontsize)
        # ax_phasesp.legend(fontsize=legendsize-2, loc='lower center')
        ax_phasesp.set_xticks([-np.pi, 0, np.pi])
        ax_phasesp.set_xticklabels(['$-\pi$', '0', '$\pi$'])
        ax_phasesp.set_yticks([0, 0.2])
        ax_phasesp.tick_params(labelsize=ticksize, direction='inout')
        ax_phasesp.set_xlabel('Phase (rad)', fontsize=fontsize)
        ax_phasesp.set_ylabel('Normalized frequency', fontsize=fontsize)
        ax_phasesp.set_xlim(-np.pi, np.pi)
        stat_record(stat_fn, False, 'SpikePhase-HighLowDiff SIM Mean_diff p=%0.4f' % (p_ww))
        stat_record(stat_fn, False, ww_table.to_string())
        stat_record(stat_fn, False,
                    'SpikePhase-HighLowDiff SIM Bartlett\'s test F_{(%d, %d)}=%0.2f, p=%0.4f' % \
                    (phasesph.shape[0], phasespl.shape[0], fstat, k_pval))

        ax_avercurve.set_xlim(0, 1)
        ax_avercurve.set_ylim(-np.pi, np.pi + 0.3)
        ax_avercurve.set_yticks([-np.pi, 0, np.pi])
        ax_avercurve.set_yticklabels(['$-\pi$', '0', '$\pi$'])
        ax_avercurve.set_xlabel('Position', fontsize=fontsize)
        ax_avercurve.tick_params(labelsize=ticksize, direction='inout')

        ax_avercurve.set_ylabel('Phase (rad)', fontsize=fontsize)

        # fig_avercurve.tight_layout()
        # fig_avercurve.savefig(os.path.join(self.plot_dir, 'sim_aver_precess_curve.png'), dpi=300)

        fig_phasesp.tight_layout()
        fig_phasesp.savefig(os.path.join(self.plot_dir, 'sim_spike_phase_highlow.png'), dpi=300)

        plt.close()

    @staticmethod
    def stack_anglediff(df, precess_ref=False):
        refangle_key = 'precess_angle' if precess_ref else 'fieldangle_mlm'

        df = df.reset_index(drop=True)
        anglediff, pass_nspikes = [], []
        for i in range(df.shape[0]):
            precess_df = df.loc[i, 'precess_df']
            if precess_df.shape[0] < 1:
                continue
            refangle = df.loc[i, refangle_key]
            anglediff.append(cdiff(precess_df['spike_angle'].to_numpy(), refangle))
            pass_nspikes.append(precess_df['pass_nspikes'].to_numpy())

        anglediff = np.concatenate(anglediff)
        pass_nspikes = np.concatenate(pass_nspikes)
        return anglediff, pass_nspikes

    @staticmethod
    def stack_key(df, key, return_list=False):
        df = df.reset_index(drop=True)
        key_list = []
        for i in range(df.shape[0]):
            precess_df = (df.loc[i, 'precess_df'])
            if precess_df.shape[0] < 1:
                continue
            key_list.append(precess_df[key].to_numpy())

        if return_list:
            return key_list
        else:
            key_list = np.concatenate(key_list)
            return key_list

    @staticmethod
    def norm_div(target_hist, divider_hist):
        target_hist_norm = target_hist / divider_hist.reshape(-1, 1)
        target_hist_norm[np.isnan(target_hist_norm)] = 0
        target_hist_norm[np.isinf(target_hist_norm)] = 0
        target_hist_norm = target_hist_norm / np.sum(target_hist_norm) * np.sum(target_hist)
        return target_hist_norm


def plot_pair_correlation_sim(simdf, save_dir=None):
    stat_fn = 'fig10_SIM_paircorr.txt'
    stat_record(stat_fn, True)
    fig, ax = plt.subplots(2, 1, figsize=(figl+0.25, figl*2), sharex=True, sharey=True)

    markersize = 6

    # A->B
    ax[0], x, y, regress = plot_correlogram(ax=ax[0], df=simdf, tag='', direct='A->B', alpha=0.2,
                                                  fontsize=fontsize, markersize=markersize, ticksize=ticksize)
    nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
    stat_record(stat_fn, False, 'SIM A->B, y = %0.2fx + %0.2f, rho=%0.2f, n=%d, p=%0.4f' % \
                (regress['aopt'] * 2 * np.pi, regress['phi0'], regress['rho'], nsamples, regress['p']))

    # B->A
    ax[1], x, y, regress = plot_correlogram(ax=ax[1], df=simdf, tag='', direct='B->A', alpha=0.2,
                                                  fontsize=fontsize, markersize=markersize, ticksize=ticksize)

    nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
    stat_record(stat_fn, False, 'SIM B->A, y = %0.2fx + %0.2f, rho=%0.2f, n=%d, p=%0.4f' % \
                (regress['aopt'] * 2 * np.pi, regress['phi0'], regress['rho'], nsamples, regress['p']))


    ax[0].set_ylabel('Phase shift (rad)', fontsize=fontsize)
    ax[1].set_xlabel('Field overlap', fontsize=fontsize)
    ax[1].set_ylabel('Phase shift (rad)', fontsize=fontsize)
    if save_dir:
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'sim_paircorr.png'), dpi=300)


def plot_exintrinsic_sim(simdf, save_dir):
    stat_fn = 'fig10_SIM_exintrinsic.txt'
    stat_record(stat_fn, True)

    # Filtering
    smallerdf = simdf[(~simdf['overlap_ratio'].isna())]

    fig_exin, ax_exin = plt.subplots(1, 2, figsize=(figl*2, figl))

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
    ax_exin[0].scatter(corr_overlap_flip, corr_overlap, alpha=0.2, s=4, c='gray')
    ax_exin[0].text(0.20, 0, 'Ex. Frac.=%0.2f\np%s'%(n_ex/n_total, p2str(pchi)), fontsize=legendsize)
    ax_exin[0].plot([0, 1], [0, 1], c='k')
    ax_exin[0].set_xlabel('Extrinsicity', fontsize=fontsize)
    ax_exin[0].set_xticks([0, 1])
    ax_exin[0].set_yticks([0, 1])
    ax_exin[0].tick_params(axis='both', which='major', labelsize=ticksize)

    # Plot 1d histogram
    edges = np.linspace(-1, 1, 75)
    width = edges[1]-edges[0]
    (bins, _, _) = ax_exin[1].hist(corr_overlap_ratio, bins=edges, color='gray',
                                    density=True, alpha=0.3, histtype='stepfilled', edgecolor='k')
    ax_exin[1].plot([mean_ratio, mean_ratio], [0, bins.max()], c='k')
    ax_exin[1].text(-0.48, 0.06/width, 'Mean=\n%0.3f\np%s'%(mean_ratio, p2str(p_1d1samp)), fontsize=legendsize)
    ax_exin[1].set_xticks([-0.5, 0, 0.5])
    ax_exin[1].set_yticks([0, 0.1/width] )
    ax_exin[1].set_yticklabels(['0', '0.1'])
    ax_exin[1].set_xlabel('Extrinsicity - Intrinsicity', fontsize=fontsize)
    ax_exin[1].tick_params(axis='both', which='major', labelsize=ticksize)
    ax_exin[1].set_xlim(-0.5, 0.5)

    ax_exin[0].set_ylabel('Intrinsicity', fontsize=fontsize)
    ax_exin[1].set_ylabel('Normalized counts', fontsize=fontsize)


    if save_dir:
        fig_exin.tight_layout()
        fig_exin.savefig(os.path.join(save_dir, 'sim_exintrinsic.png'), dpi=300)


def plot_exintrinsic_pair_correlation_sim(simdf, save_dir=None):
    stat_fn = 'fig10_SIM_concentration.txt'
    stat_record(stat_fn, True)
    fig03, ax03 = plt.subplots(2, figsize=(1.75+0.5, 1.75*2))
    fig_hist, ax_hist = plt.subplots(figsize=(1.75, 1.75), sharey=True)

    markersize = 6
    alpha = 0.5
    y_dict = {}
    exdf = simdf[simdf['overlap_ratio'] > 0]
    indf = simdf[simdf['overlap_ratio'] <= 0]

    # Overlap < 0.3
    ax03[0], _, y03ex, _ = plot_correlogram((ax03[0]), exdf[exdf['overlap'] < 0.3], 'Extrinsic',
                                                  'combined', density=False, regress=False, x_dist=False,
                                                  y_dist=True,
                                                  alpha=alpha, fontsize=fontsize, markersize=markersize, ticksize=ticksize)
    ax03[0].set_xlabel('')
    ax03[1], _, y03in, _ = plot_correlogram((ax03[1]), indf[indf['overlap'] < 0.3], 'Intrinsic',
                                                  'combined', density=False, regress=False, x_dist=False,
                                                  y_dist=True,
                                                  alpha=alpha, fontsize=fontsize, markersize=markersize, ticksize=ticksize)
    y_dict['ex'] = y03ex[~np.isnan(y03ex)]
    y_dict['in'] = y03in[~np.isnan(y03in)]

    # Phaselag histogram
    yex_ax, yex_den = circular_density_1d(y_dict['ex'], 4 * np.pi, 50, (-np.pi, np.pi))
    yin_ax, yin_den = circular_density_1d(y_dict['in'], 4 * np.pi, 50, (-np.pi, np.pi))
    F_k, pval_k = circ_ktest(y_dict['ex'], y_dict['in'])
    stat_record(stat_fn, False, 'SIM Ex-in concentration difference F_{(n_1=%d, n_2=%d)}=%0.2f, p=%0.4f'%\
                (y_dict['ex'].shape[0], y_dict['in'].shape[0], F_k, pval_k))

    ax_hist.plot(yex_ax, yex_den, c='r', label='ex')
    ax_hist.plot(yin_ax, yin_den, c='b', label='in')
    ax_hist.set_xticks([-np.pi, 0, np.pi])
    ax_hist.set_yticks([0, 0.05])
    ax_hist.set_xticklabels(['$-\pi$', 0, '$\pi$'])
    ax_hist.set_xlabel('Phase shift (rad)', fontsize=fontsize)
    ax_hist.tick_params(labelsize=ticksize)
    ax_hist.text(-np.pi / 2 - 1.2, 0.03, 'Bartlett\'s\np%s'%(p2str(pval_k)), fontsize=legendsize)
    ax_hist.text(np.pi / 4, 0.04, 'Ex.', fontsize=legendsize, color='r')
    ax_hist.text(np.pi / 2 + 0.1, 0.02, 'In.', fontsize=legendsize, color='b')
    ax_hist.set_ylabel('Density of pairs', fontsize=fontsize)
    if save_dir:
        fig_hist.tight_layout()
        fig_hist.savefig(os.path.join(save_dir, 'sim_exintrinsic_concentration.png'), dpi=300)


def omniplot_pairfields_sim(simdf, save_dir=None):
    stat_fn = 'fig10_SIM_pair_directionality.txt'
    stat_record(stat_fn, True)

    spike_threshs = np.arange(0, 1001, 50)
    stats_getter = DirectionalityStatsByThresh('num_spikes_pair', 'win_pval_pair_mlm',
                                               'shift_pval_pair_mlm', 'fieldR_pair_mlm')
    linecolor = {'all':'k', 'border':'k', 'nonborder':'k'}
    linestyle = {'all':'solid', 'border':'dotted', 'nonborder':'dashed'}
    # Initialize subplots
    fig, ax = plt.subplots(1, 3, figsize=(figl*3, figl), sharex=True)

    # Plot all
    all_dict = stats_getter.gen_directionality_stats_by_thresh(simdf, spike_threshs)
    ax[0].plot(spike_threshs, all_dict['medianR'], c=linecolor['all'], linestyle=linestyle['all'], label='All')
    ax[1].plot(spike_threshs, all_dict['sigfrac_shift'], c=linecolor['all'], linestyle=linestyle['all'], label='All', alpha=0.7)


    # Plot border
    simdf_b = simdf[simdf['border']].reset_index(drop=True)
    border_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_b, spike_threshs)
    ax[0].plot(spike_threshs, border_dict['medianR'], c=linecolor['border'], linestyle=linestyle['border'], label='Border')
    ax[1].plot(spike_threshs, border_dict['sigfrac_shift'], c=linecolor['border'], linestyle=linestyle['border'], label='Border', alpha=0.7)

    # Plot non-border
    simdf_nb = simdf[~simdf['border']].reset_index(drop=True)
    nonborder_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_nb, spike_threshs)
    ax[0].plot(spike_threshs, nonborder_dict['medianR'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border')
    ax[1].plot(spike_threshs, nonborder_dict['sigfrac_shift'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', alpha=0.7)

    # Plot Fraction
    border_nfrac = border_dict['n']/all_dict['n']
    ax[2].plot(spike_threshs, all_dict['datafrac'], c=linecolor['all'], linestyle=linestyle['all'], label='All')
    ax[2].plot(spike_threshs, border_nfrac, c=linecolor['border'], linestyle=linestyle['border'], label='Border')
    ax[2].plot(spike_threshs, 1-border_nfrac, c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border')

    # Plotting asthestic
    ax_ylabels = ['Median R', "Significant Fraction", "Data Fraction"]
    for axid in range(ax.shape[0]):
        ax[axid].set_xticks([0, 500, 1000])
        ax[axid].set_ylabel(ax_ylabels[axid], fontsize=fontsize)
        ax[axid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax[axid].set_xlabel('Spike count threshold', fontsize=fontsize)
        ax[axid].spines['top'].set_visible(False)
        ax[axid].spines['right'].set_visible(False)

    customlegend(ax[1], fontsize=legendsize-1, loc='upper right')
    ax[0].set_yticks([0.1, 0.2])
    ax[1].set_yticks([0])
    ax[2].set_yticks([0, 0.5, 1])


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


    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'sim_pair_directionality.png'), dpi=300)

def plot_kld_sim(simdf, save_dir=None):
    stat_fn = 'fig9_SIM_kld.txt'
    stat_record(stat_fn, True)

    kld_key = 'kld_mlm'

    colors = dict(all='k', border='darkviolet', nonborder='darkorange')
    kld_dict = dict()



    # # # KLD by threshold
    Nthreshs = np.append(np.arange(0, 1500, 100), 1500)
    fig_kld_thresh, ax_kld_thresh = plt.subplots(figsize=(1.75, 1.75), sharey=True, sharex=True)
    kld_thresh_all_list = []
    kld_thresh_border_list = []
    kld_thresh_nonborder_list = []

    # # KLD by threshold - All

    all_kld = np.zeros(Nthreshs.shape[0])
    for idx, thresh in enumerate(Nthreshs):
        simdf2 = simdf[simdf['num_spikes_pair'] > thresh]
        kld_arr = simdf2[kld_key].to_numpy()
        kld_thresh_all_list.append(kld_arr)
        all_kld[idx] = np.median(kld_arr)
    ax_kld_thresh.plot(Nthreshs, all_kld, c=colors['all'], label='all' )

    # # KLD by threshold - Border/Non-border
    for (border), gpdf in simdf.groupby(by='border'):
        all_kld = np.zeros(Nthreshs.shape[0])
        for idx, thresh in enumerate(Nthreshs):
            gpdf2 = gpdf[gpdf['num_spikes_pair'] > thresh]
            kld_arr = gpdf2[kld_key].to_numpy()
            if border:
                kld_thresh_border_list.append(kld_arr)
            else:
                kld_thresh_nonborder_list.append(kld_arr)
            all_kld[idx] = np.median(kld_arr)
        labeltxt = 'B' if border else 'N-B'
        bordertxt = 'border' if border else 'nonborder'
        ax_kld_thresh.plot(Nthreshs, all_kld, c=colors[bordertxt], label=labeltxt)

    # # KLD by threshold - Other plotting
    ax_kld_thresh.legend(fontsize=fontsize - 2)
    ax_kld_thresh.set_ylabel('Median KLD (bit)', fontsize=fontsize)
    ax_kld_thresh.set_xlabel('Spike count threshold', fontsize=fontsize)

    ax_kld_thresh.set_xticks([0, 500, 1000])
    ax_kld_thresh.tick_params(labelsize=fontsize-2)
    ax_kld_thresh.spines['top'].set_visible(False)
    ax_kld_thresh.spines['right'].set_visible(False)

    # # KLD by threshold - Statistical test

    # Border effect by CA
    for idx, thresh in enumerate(Nthreshs):
        kld_b, kld_nb = kld_thresh_border_list[idx], kld_thresh_nonborder_list[idx]
        n_b, n_nd = kld_b.shape[0], kld_nb.shape[0]
        z_nnb, p_nnb = ranksums(kld_b, kld_nb)
        stat_record(stat_fn, False, 'SIM KLD by threshold=%d, Border efffect, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                    (thresh, z_nnb, n_b, n_nd, p_nnb))
    # Save
    if save_dir:
        fig_kld_thresh.tight_layout()
        fig_kld_thresh.savefig(os.path.join(save_dir, 'sim_kld_thresh.png'), dpi=300)


def plot_pair_single_angles_analysis_sim(simdf, save_dir):
    stat_fn = 'fig9_SIM_pairsingle_angles.txt'
    stat_record(stat_fn, True)
    figsize = (1.75+0.5, 1.75 * 3)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharey='row', sharex='row')


    Nthreshs = np.append(np.arange(0, 1000, 100), 1000)

    all_fmeans = []
    all_fpairs = []
    all_nspks = []
    for rowi in range(simdf.shape[0]):
        f1, f2, fp = simdf.loc[rowi, ['fieldangle1_mlm', 'fieldangle2_mlm', 'fieldangle_pair_mlm']]
        nspks = simdf.loc[rowi, 'num_spikes_pair']
        f_mean = circmean(f1, f2)

        all_fmeans.append(f_mean)
        all_fpairs.append(fp)
        all_nspks.append(nspks)

    all_fmeans = np.array(all_fmeans)
    all_fpairs = np.array(all_fpairs)
    all_nspks = np.array(all_nspks)
    ax[0].scatter(all_fmeans, all_fpairs, alpha=0.2, s=8)
    ax[0].set_xticks([0, 2 * np.pi])
    ax[0].set_xticklabels(['0', '$2\pi$'])
    ax[0].set_yticks([0, 2 * np.pi])
    ax[0].tick_params(labelsize=fontsize)
    ax[0].set_yticklabels(['0', '$2\pi$'])

    for thresh in Nthreshs:
        mask = (~np.isnan(all_fmeans)) & (~np.isnan(all_fpairs) & (all_nspks > thresh))
        adiff = cdiff(all_fmeans[mask], all_fpairs[mask])
        color = cm.coolwarm(thresh / Nthreshs.max())

        alpha_ax, den = circular_density_1d(adiff, 8 * np.pi, 100, bound=(-np.pi, np.pi))
        ax[1].plot(alpha_ax, den, c=color)

        abs_adiff = np.abs(adiff)
        absbins, absedges = np.histogram(abs_adiff, bins=np.linspace(0, np.pi, 50))
        cumsum_norm = np.cumsum(absbins) / np.sum(absbins)
        ax[2].plot(midedges(absedges), cumsum_norm, c=color)

    ax[1].set_xticks([-np.pi, 0, np.pi])
    ax[1].set_xticklabels(['$-\pi$', '0', '$\pi$'])
    ax[1].set_yticks([])
    ax[1].tick_params(labelsize=fontsize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    ax[2].set_xticks([0, np.pi])
    ax[2].set_xticklabels(['$0$', '$\pi$'])
    ax[2].set_yticks([0, 1])
    ax[2].tick_params(labelsize=fontsize)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)


    ax[0].set_xlabel(r'$\theta_{mean}$', fontsize=fontsize)
    ax[0].set_ylabel(r'$\theta_{pair}$', fontsize=fontsize)
    ax[1].set_xlabel(r'$d(\theta_{mean}, \theta_{pair})$', fontsize=fontsize)
    ax[1].set_ylabel(r'Normalized density', fontsize=fontsize)
    ax[2].set_xlabel(r'$|d(\theta_{mean}, \theta_{pair})|$', fontsize=fontsize)
    ax[2].set_ylabel(r'Cumulative counts', fontsize=fontsize)


    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm,
                               norm=plt.Normalize(vmin=Nthreshs.min(), vmax=Nthreshs.max()))
    fig_colorbar = plt.figure(figsize=figsize)
    fig_colorbar.subplots_adjust(right=0.8)
    cbar_ax = fig_colorbar.add_axes([0.7, 0.15, 0.03, 0.3])
    cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0, 500, 1000])
    cb.ax.set_yticklabels(['0', '500', '1000'], rotation=90)
    cb.set_label('Spike count thresholds', fontsize=fontsize)

    if save_dir:
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'sim_pair_single_angles.png'), dpi=300)
        fig_colorbar.savefig(os.path.join(save_dir, 'sim_pair_single_angles_colorbar.png'), dpi=300)



if __name__ == '__main__':


    # # Single field df
    simdf = load_pickle('results/sim/single_field/singlefield_df_square.pickle')
    #
    # # Directionality
    # omniplot_singlefields_sim(simdf, save_dir='result_plots/sim/single_field/')

    # # Pass analysis
    analyzer = SingleBestAngleAnalyzer_SIM(simdf, 'result_plots/sim/passes/')
    # analyzer.plot_overall_bestprecession()
    analyzer.plot_field_bestprecession()
    # analyzer.plot_both_slope_offset()
    # analyzer.plot_average_precession(NShuffles=1000)


    # # Pair field
    # save_dir = 'result_plots/sim/pair_fields/'
    #
    # simdf = load_pickle('results/sim/pair_field/pairfield_df_square.pickle')
    # simdf['border'] = (simdf['border1'] | simdf['border2'])
    # plot_pair_correlation_sim(simdf, save_dir=save_dir)
    # plot_exintrinsic_sim(simdf, save_dir)
    # plot_exintrinsic_pair_correlation_sim(simdf, save_dir)
    # omniplot_pairfields_sim(simdf, save_dir)
    # plot_pair_single_angles_analysis_sim(simdf, save_dir)