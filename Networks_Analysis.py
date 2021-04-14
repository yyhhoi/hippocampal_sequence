import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import torch
from matplotlib import cm
from pycircstat import vtest, watson_williams
from scipy.interpolate import interp1d
from scipy.stats import ranksums, chi2_contingency, spearmanr, linregress
from torch.distributions.von_mises import VonMises
from scipy.integrate import odeint
from pycircstat.descriptive import cdiff, mean as circmean

from Networks_main import get_nidx_np
from common.comput_utils import linear_gauss_density, segment_passes, compute_straightness, midedges, repeat_arr, \
    unfold_binning_2d, linear_circular_gauss_density, circ_ktest
from common.linear_circular_r import rcc
from common.script_wrappers import DirectionalityStatsByThresh, permutation_test_average_slopeoffset
from common.shared_vars import total_figw, ticksize, fontsize, dpi
from common.utils import load_pickle, stat_record, p2str
from common.visualization import customlegend, plot_marginal_slices

legendsize = 8
titlesize = 9 + 2
figl = 1.75


def omniplot_singlefields_network(simdf, plot_dir=None):

    stat_fn = 'NETWORK_single_directionality.txt'
    stat_record(stat_fn, True)
    fontsize = 9
    ticksize = 8
    legendsize = 8
    titlesize = 9 + 2
    figl = 1.75

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

    customlegend(ax[1], fontsize=legendsize-1, loc='center right')
    # ax[0].set_yticks([0.03, 0.06])
    ax[1].set_yticks([0, 1])
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
    fig.savefig(join(plot_dir, 'Networks_single_directionality.png'), dpi=300)


class SingleBestAngleAnalyzer_Networks:
    def __init__(self, singlefield_df, plot_dir):
        self.singlefield_df = singlefield_df
        self.plot_dir = plot_dir
        self.anglediff_edges = np.linspace(0, np.pi, 50)
        self.slope_edges = np.linspace(-2, 0, 50)
        self.offset_edges = np.linspace(0, 2 * np.pi, 50)
        self.angle_title = r'$d(\theta_{pass}, \theta_{rate})$' + ' (rad)'


    def plot_overall_bestprecession(self):
        print('Plot overall best precession')
        fig_basic, ax_basic = plt.subplots(figsize=(figl+0.1, figl), sharex=True, sharey=True)

        precess_adiff = []
        precess_nspikes = []
        nonprecess_adiff = []
        nonprecess_nspikes = []
        for i in range(self.singlefield_df.shape[0]):
            refangle = self.singlefield_df.loc[i, 'fieldangle_mlm']
            allprecess_df = self.singlefield_df.loc[i, 'precess_df']
            precess_df = allprecess_df[allprecess_df['precess_exist']]
            nonprecess_df = allprecess_df[~allprecess_df['precess_exist']]
            precess_counts = precess_df.shape[0]
            nonprecess_counts = nonprecess_df.shape[0]
            if precess_counts > 0:
                precess_adiff.append(cdiff(precess_df['spike_angle'].to_numpy(), refangle))
                precess_nspikes.append(precess_df['pass_nspikes'].to_numpy())
            if nonprecess_counts > 0:
                nonprecess_adiff.append(cdiff(nonprecess_df['spike_angle'].to_numpy(), refangle))
                nonprecess_nspikes.append(nonprecess_df['pass_nspikes'].to_numpy())


        # Density of Fraction, Spikes and Ratio
        precess_adiff = np.abs(np.concatenate(precess_adiff))
        nonprecess_adiff = np.concatenate(nonprecess_adiff)
        precess_nspikes = np.abs(np.concatenate(precess_nspikes))
        nonprecess_nspikes = np.concatenate(nonprecess_nspikes)
        adiff_spikes_p = repeat_arr(precess_adiff, precess_nspikes)
        adiff_spikes_np = repeat_arr(nonprecess_adiff, nonprecess_nspikes)


        adiff_bins = np.linspace(0, np.pi, 41)
        adm = midedges(adiff_bins)
        precess_bins, _ = np.histogram(precess_adiff, bins=adiff_bins)
        nonprecess_bins, _ = np.histogram(nonprecess_adiff, bins=adiff_bins)
        fraction_bins = precess_bins/(precess_bins + nonprecess_bins)
        spike_bins_p, _ = np.histogram(adiff_spikes_p, bins=adiff_bins)
        spike_bins_np, _ = np.histogram(adiff_spikes_np, bins=adiff_bins)
        spike_bins = spike_bins_p + spike_bins_np
        norm_bins = (fraction_bins / spike_bins)

        precess_allcount, nonprecess_allcount = precess_bins.sum(), nonprecess_bins.sum()

        rho, pval = spearmanr(adm, norm_bins)
        pm, pc, pr, ppval, _ = linregress(adm, norm_bins/ norm_bins.sum())
        xdum = np.linspace(adm.min(), adm.max(), 10)
        ax_basic.step(adm, fraction_bins / fraction_bins.sum(), color='darkblue', alpha=0.7, linewidth=1, label='Precession')
        ax_basic.step(adm, spike_bins / spike_bins.sum(), color='darkorange', alpha=0.7, linewidth=1, label='Spike')
        ax_basic.step(adm, norm_bins / norm_bins.sum(), color='darkgreen', alpha=0.7, linewidth=1, label='Ratio')

        customlegend(ax_basic, fontsize=legendsize-1)

        ax_basic.set_xticks([0, np.pi / 2, np.pi])
        ax_basic.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
        # ax_basic.set_yticks([0, 0.04])
        ax_basic.tick_params(axis='both', which='major', labelsize=ticksize, direction='in')
        # ax_basic.set_ylim(0, 0.045)

        ax_basic.set_ylabel('Normalized counts', fontsize=fontsize)
        ax_basic.set_xlabel(self.angle_title, fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(join(self.plot_dir, 'Networks_overall_bestprecession.png'), dpi=300)

    def plot_field_bestprecession(self):

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
        fig_2Dscatter.savefig(join(self.plot_dir, 'Networks_field_precess_2Dscatter.png'), dpi=300)


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
        fig_pishift.tight_layout()
        fig_pishift.savefig(join(self.plot_dir, 'Networks_field_precess_pishift.png'), dpi=300)




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
        fig_R.savefig(join(self.plot_dir, 'Networks_field_precess_R.png'), dpi=300)




    def plot_both_slope_offset(self):

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
        fig_density.savefig(join(self.plot_dir, 'Networks_density.png'), dpi=300)
        fig_slices.tight_layout()
        fig_slices.savefig(join(self.plot_dir, 'Networks_slices.png'), dpi=300)
        # fig_colorbar.tight_layout()
        fig_colorbar.savefig(join(self.plot_dir, 'Networks_adiff_colorbar.png'), dpi=300)
        return None

    def plot_average_precession(self, NShuffles=1000):

        fig_avercurve, ax_avercurve = plt.subplots(figsize=(figl, figl), sharey=True)
        fig_phasesp, ax_phasesp = plt.subplots(figsize=(figl, figl), sharey=True)

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


        xdum = np.linspace(0, 1, 10)
        high_agg_ydum = 2 * np.pi * regress_high['aopt'] * xdum + regress_high['phi0']
        low_agg_ydum = 2 * np.pi * regress_low['aopt'] * xdum + regress_low['phi0']

        # Compare high vs low
        ax_avercurve.plot(xdum, high_agg_ydum, c='lime', label='$|d|>5\pi/6$')
        ax_avercurve.plot(xdum, low_agg_ydum, c='darkblue', label='$|d|<\pi/6$')
        ax_avercurve.text(0.05, -np.pi + 0.5, 'Slope\n  p%s\nOffset\n  p%s'% \
                          (p2str(pval_slope), p2str(pval_offset)), fontsize=legendsize)

        customlegend(ax_avercurve, fontsize=legendsize, loc='center right')

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
        # ax_phasesp.set_yticks([0, 0.2])
        ax_phasesp.tick_params(labelsize=ticksize, direction='inout')
        ax_phasesp.set_xlabel('Phase (rad)', fontsize=fontsize)
        ax_phasesp.set_ylabel('Normalized frequency', fontsize=fontsize)
        ax_phasesp.set_xlim(-np.pi, np.pi)

        ax_avercurve.set_xlim(0, 1)
        ax_avercurve.set_ylim(-np.pi, np.pi + np.pi/2)
        ax_avercurve.set_yticks([-np.pi, 0, np.pi])
        ax_avercurve.set_yticklabels(['$-\pi$', '0', '$\pi$'])
        ax_avercurve.set_xlabel('Position', fontsize=fontsize)
        ax_avercurve.tick_params(labelsize=ticksize, direction='inout')

        ax_avercurve.set_ylabel('Phase (rad)', fontsize=fontsize)

        fig_avercurve.tight_layout()
        fig_avercurve.savefig(join(self.plot_dir, 'Networks_aver_precess_curve.png'), dpi=300)

        fig_phasesp.tight_layout()
        fig_phasesp.savefig(join(self.plot_dir, 'Networks_spike_phase_highlow.png'), dpi=300)

        plt.close()

    @staticmethod
    def stack_anglediff(df, precess_ref=False):
        refangle_key = 'precess_angle' if precess_ref else 'fieldangle_mlm'

        df = df.reset_index(drop=True)
        anglediff, pass_nspikes = [], []
        for i in range(df.shape[0]):
            precess_df = df.loc[i, 'precess_df']
            precess_df = precess_df[precess_df['precess_exist']].reset_index(drop=True)
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
            precess_df = precess_df[precess_df['precess_exist']].reset_index(drop=True)
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





def check_precession(simdf, plot_dir):

    def plot_precession(ax, d, phase, rcc_m, rcc_c, marker_size, color):
        ax.scatter(np.concatenate([d, d]), np.concatenate([phase, phase + 2 * np.pi]), marker='.',
                   s=marker_size, c=color)
        xdum = np.linspace(0, 1, 10)
        ydum = xdum * (rcc_m * 2 * np.pi) + rcc_c

        ax.plot(xdum, ydum, c='k')
        ax.plot(xdum, ydum + 2 * np.pi, c='k')
        ax.plot(xdum, ydum - 2 * np.pi, c='k')
        ax.set_xticks([0, 1])
        ax.set_yticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
        ax.set_yticklabels(['', '0', '', '2$\pi$', ''])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.set_xlim(0, 1)
        ax.set_ylim(-np.pi, 3 * np.pi)
        return ax

    figl = total_figw*0.4




    num_fields = simdf.shape[0]
    id = 0
    all_rcc_m = []
    for nf in range(num_fields):
        print('%d/%d'%(nf, num_fields))
        precess_df = simdf.loc[nf, 'precess_df']
        precess_df = precess_df[precess_df['precess_exist']].reset_index(drop=True)
        num_precess = precess_df.shape[0]

        for precessid in range(num_precess):
            dsp, phasesp, rcc_m, rcc_c = precess_df.loc[precessid, ['dsp', 'phasesp', 'rcc_m', 'rcc_c']]
            if id < 30:
                fig, ax = plt.subplots(figsize= (figl, figl))
                plot_precession(ax, dsp, phasesp, rcc_m, rcc_c, 4, color='gray')
                fig.tight_layout()
                fig.savefig(join(plot_dir, 'precess_examples_%d.png'%(id)), dpi=dpi)
                plt.close()
                id += 1
            all_rcc_m.append(rcc_m)
    fig, ax = plt.subplots()
    ax.hist(all_rcc_m, bins=30)
    fig.savefig(join(plot_dir, 'rcc_m_dist.png'))


def gen_illustration_plots(plot_dir):
    simdata = pd.read_pickle('results/network_proj/sim_result.pickle')
    Indata = simdata['Indata']
    SpikeData = simdata['SpikeData']
    NeuronPos = simdata['NeuronPos']
    xxtun, yytun, aatun = NeuronPos['neuronx'].to_numpy(), NeuronPos['neurony'].to_numpy(), NeuronPos['neurona'].to_numpy()

    dt = 1
    xmin, xmax, nx = -40, 40, 20
    ymin, ymax, ny = -40, 40, 20
    amin, amax, na = -np.pi, np.pi, 8

    n_ex = nx * ny * na
    n_total = n_ex

    xtun = np.linspace(xmin, xmax, nx)
    ytun = np.linspace(ymin, ymax, ny)
    atun = np.linspace(amin, amax, na)


    field_r = 10  # diamter 20 cm
    angle_bound = np.pi /2



    # Plot trajectory
    fig_traj, ax_traj = plt.subplots()
    ax_traj.plot(Indata.x, Indata.y, c='gray')

    # Plot 20 x 20 fields
    xx, yy = np.meshgrid(xtun, ytun)
    xx, yy = np.concatenate([xx[:, 0], xx[0, :]]), np.concatenate([yy[:, 0], yy[0, :]])
    angle_ax = np.linspace(0, 2*np.pi, 100)
    for xcen, ycen in zip(xx, yy):
        ax_traj.scatter(xcen, ycen, marker='x', c='k')
        ax_traj.plot(xcen + field_r * np.cos(angle_ax), ycen + field_r * np.sin(angle_ax), c='r')
    ax_traj.set_xlim(-50, 50)
    ax_traj.set_ylim(-50, 50)
    fig_traj.savefig(join(plot_dir, 'illu_traj.png'), dpi=300)


    # Plot tunning
    fig_atun, ax_atun = plt.subplots(subplot_kw=dict(projection='polar'))
    angle_ax = np.linspace(0, 2*np.pi, 500)
    for atun_each in atun:
        adist = np.exp(-(cdiff(atun_each, angle_ax) ** 2 )/ (angle_bound ** 2))
        ax_atun.plot(angle_ax, adist)
    ax_atun.grid(False)
    fig_atun.savefig(join(plot_dir, 'illu_atun.png'), dpi=300)

    # # Place field Example

    for idx, atun_each in enumerate(atun):
        nidx = get_nidx_np(20, 10, atun_each, xxtun, yytun, aatun)
        spdf = SpikeData[SpikeData['neuronidx']==nidx].reset_index(drop=True)
        neu_x, neu_y, neu_a = NeuronPos.loc[nidx, ['neuronx', 'neurony', 'neurona']]
        tidxsp = spdf['tidxsp']
        xsp, ysp, asp = Indata.loc[tidxsp, 'x'], Indata.loc[tidxsp, 'y'], Indata.loc[tidxsp, 'angle']
        fig_eg, ax_eg = plt.subplots()
        ax_eg.plot(Indata.x, Indata.y, c='gray', alpha=0.5)
        ax_eg.scatter(xsp, ysp, c=asp, marker='o', cmap='hsv', s=16, zorder=2.5, alpha=0.7)
        ax_eg.scatter(neu_x, neu_y, c='k', marker='x', s=16, zorder=2.5)
        ax_eg.set_xlim(-40, 40)
        ax_eg.set_ylim(-40, 40)
        ax_eg.axis('off')
        fig_eg.tight_layout()
        fig_eg.savefig(join(plot_dir, 'illu_colored_spks_%0.2f.png'%(np.rad2deg(neu_a))))


plot_dir = 'result_plots/network_proj/analysis'
simdf = pd.read_pickle('results/network_proj/singlefield_df_networkproj.pickle')
# omniplot_singlefields_network(simdf, plot_dir)
#
# # # Pass analysis
# analyzer = SingleBestAngleAnalyzer_Networks(simdf, plot_dir)
#
# analyzer.plot_overall_bestprecession()
# analyzer.plot_field_bestprecession()
# analyzer.plot_both_slope_offset()
# analyzer.plot_average_precession(NShuffles=10)

check_precession(simdf, 'result_plots/network_proj/precess_examples')

# gen_illustration_plots('result_plots/network_proj')
