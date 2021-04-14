# %% Import libraries
import os
from os.path import join

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
from astropy.stats import kuiper_two
from matplotlib import cm
import matplotlib.colors as mcol
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from pycircstat.tests import vtest, watson_williams, kuiper
from scipy.stats import spearmanr, binom_test, chi2_contingency, chisquare, linregress, pearsonr
import statsmodels.api as smapi
from statsmodels.formula.api import ols
from common.script_wrappers import permutation_test_average_slopeoffset, combined_average_curve, compute_precessangle, \
    permutation_test_arithmetic_average_slopeoffset
from common.comput_utils import midedges, circular_density_1d, linear_circular_gauss_density, \
    unfold_binning_2d, repeat_arr, \
    circ_ktest, shiftcyc_full2half, ci_vonmise, fisherexact, ranksums
from common.linear_circular_r import rcc
from common.utils import load_pickle, sigtext, stat_record, p2str
from common.visualization import plot_marginal_slices, SqueezedNorm, customlegend

from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw


figext = 'png'
# figext = 'eps'


def plot_precession_examples(singledf, save_dir, figext='png'):

    def plot_precession(ax, d, phase, rcc_m, rcc_c, marker_size, color):
        ax.scatter(np.concatenate([d, d]), np.concatenate([phase, phase + 2 * np.pi]), marker='.',
                   s=marker_size, c=color)
        xdum = np.linspace(0, 1, 10)
        ydum = xdum * (rcc_m * 2 * np.pi) + rcc_c

        ax.plot(xdum, ydum, c='k')
        ax.plot(xdum, ydum + 2 * np.pi, c='k')
        ax.plot(xdum, ydum - 2 * np.pi, c='k')
        ax.set_yticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
        ax.set_yticklabels(['', '0', '', '2$\pi$', ''])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.set_xlim(0, 1)
        ax.set_ylim(-np.pi, 3 * np.pi)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    figl = total_figw*0.4

    id_list = []

    fig, ax = plt.subplots(4, 3, figsize=(figl, figl*(4/3)*0.8), sharex=True, sharey=True)

    # # Assign field ids within ca
    for caid, (ca, cadf) in enumerate(singledf.groupby('ca')):
        cadf = cadf.reset_index(drop=True)

        num_fields = cadf.shape[0]


        for nf in range(num_fields):
            allprecess_df, fieldid_ca= cadf.loc[nf, ['precess_df', 'fieldid_ca']]
            precess_df = allprecess_df[allprecess_df['precess_exist']].reset_index(drop=True)
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
            plot_precession(ax[0, caid], dsp_all, phasesp_all, rcc_m, rcc_c, 1, color=ca_c[ca])

            np.random.seed(1)
            precessids = np.random.choice(num_precess, size=3)
            for axid, precessid in enumerate(precessids):
                dsp, phasesp, rcc_m, rcc_c = precess_df.loc[precessid, ['dsp', 'phasesp', 'rcc_m', 'rcc_c']]
                plot_precession(ax[axid+1, caid], dsp, phasesp, rcc_m, rcc_c, 4, color=ca_c[ca])
            id_list.append(fieldid_ca)
            ax[0, caid].set_title('%s#%d'%(ca, fieldid_ca), fontsize=fontsize)

            break


    ax[3, 1].set_xticks([0, 1])
    plt.setp(ax[3, 0].get_xticklabels(), visible=False)
    plt.setp(ax[3, 2].get_xticklabels(), visible=False)
    fig.text(0.03, 0.35, 'Phase (rad)', ha='center', rotation=90, fontsize=fontsize)
    fig.text(0.55, 0.02, 'Position', ha='center', fontsize=fontsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(os.path.join(save_dir, 'precess_examples.%s'%(figext)), dpi=dpi)


def check_pass_nspikes(df):
    for ca, cadf in df.groupby('ca'):

        allprecessdf = pd.concat(cadf['precess_df'].to_list(), axis=0, ignore_index=True)

        precessdf = allprecessdf[allprecessdf['precess_exist']]
        print('\n%s'%ca)
        print(precessdf['pass_nspikes'].describe())

class SingleBestAngleAnalyzer:
    def __init__(self, singlefield_df, plot_dir, nap_thresh=1):
        self.singlefield_df = singlefield_df
        self.plot_dir = plot_dir
        self.anglediff_edges = np.linspace(0, np.pi, 50)
        self.slope_edges = np.linspace(-2, 0, 50)
        self.offset_edges = np.linspace(0, 2 * np.pi, 50)
        self.angle_title = r'$d(\theta_{pass}, \theta_{rate})$' + ' (rad)'
        self.absangle_title  = r'$|d(\theta_{pass}, \theta_{rate})|$' + ' (rad)'
        self.nap_thresh = nap_thresh

    def plot_field_bestprecession(self, savename_tag=''):

        figl = total_figw*0.6
        linew = 0.75
        R_figl = total_figw*0.4

        print('Plot field best precession')
        stat_fn = 'fig2_field_precess.txt'
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
        allca_ntotal = self.singlefield_df.shape[0]
        all_shufp = []

        # Stack data
        for caid, (ca, cadf) in enumerate(self.singlefield_df.groupby('ca')):
            cadf = cadf.reset_index(drop=True)
            numpass_mask = cadf['numpass_at_precess'].to_numpy() >= self.nap_thresh
            numpass_low_mask = cadf['numpass_at_precess_low'].to_numpy() >= self.nap_thresh

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
        U13, p13 = ranksums(allR[0], allR[2])
        stat_record(stat_fn, False, 'CA13, R difference, U(N_{CA1}=%d, N_{CA3}=%d)=%0.2f, p%s' % \
                    (allR[0].shape[0], allR[2].shape[0], U13, p2str(p13)))
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
        ax_pishift[0, 1].set_xlabel(self.absangle_title, fontsize=fontsize)
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
            fig_pishift.savefig(join(self.plot_dir, 'field_precess_pishift.%s'%(figext)), dpi=dpi)
            fig_R.savefig(os.path.join(self.plot_dir, 'field_precess_R.%s'%(figext)), dpi=dpi)
            fig_negslope.savefig(join(self.plot_dir, 'fraction_neg_slopes%s.%s'%(savename_tag, figext)), dpi=dpi)



    def plot_both_slope_offset(self, NShuffles=200, savename_tag=''):

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

        stat_fn = 'fig3_slopeoffset.txt'
        stat_record(stat_fn, True, 'Average Phase precession')
        fig1, ax1 = plt.subplots(2, 3, figsize=fig1size, sharey='row')  # Density
        fig2, ax2 = plt.subplots(2, 3, figsize=fig2size, sharey='row')  # Marginals
        fig3, ax3 = plt.subplots(2, 3, figsize=fig3size, sharey='row')  # Aver curves + spike phase
        fig_s, ax_s = plt.subplots(3, 1, figsize=(figl*6, figl*6), sharey='row')  # Checking onset, slope, phases

        # Construct pass df
        refangle_key = 'rate_angle'
        passdf_dict = {'ca':[], 'anglediff':[], 'slope':[], 'onset':[], 'pass_nspikes':[]}
        spikedf_dict = {'ca':[], 'anglediff':[], 'phasesp':[]}
        dftmp = self.singlefield_df[(~self.singlefield_df[refangle_key].isna())].reset_index()
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


        # Plot and analysis for CA1, CA2 and CA3
        for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
            pass_cadf = passdf[passdf['ca']==ca].reset_index(drop=True)
            spike_cadf = spikedf[spikedf['ca']==ca].reset_index(drop=True)

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
            offset_normbins = self.norm_div(offset_bins, spikes_bins)
            slope_normbins = self.norm_div(slope_bins, spikes_bins)

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
            # ax2[0, caid].annotate('p%s'%(p2str(offsetpval)), xy=(0.2, 1), xycoords='axes fraction', fontsize=legendsize)
            # ax2[0, caid].annotate('', xy=(0.2, 1.1), xytext=(0, 1.1), xycoords='axes fraction', fontsize=legendsize, arrowprops=dict(color='green', arrowstyle='-'))
            # ax2[0, caid].annotate('', xy=(0.2, 1), xytext=(0, 1), xycoords='axes fraction', fontsize=legendsize, arrowprops=dict(color='blue', arrowstyle='-'))
            stat_record(stat_fn, False, '%s onset difference p%s\n%s'%(ca, p2str(offsetpval), offsetwwtable.to_string()))
            # ax2[1, caid].annotate('p%s'%(p2str(slopepval)), xy=(0.2, 1), xycoords='axes fraction', fontsize=legendsize)
            # ax2[1, caid].annotate('', xy=(0.2, 1.1), xytext=(0, 1.1), xycoords='axes fraction', fontsize=legendsize, arrowprops=dict(color='green', arrowstyle='-'))
            # ax2[1, caid].annotate('', xy=(0.2, 1), xytext=(0, 1), xycoords='axes fraction', fontsize=legendsize, arrowprops=dict(color='blue', arrowstyle='-'))
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
            #             ax3[0, caid].text(0.01, -np.pi/2+0.2, '%s'%(pvalstr1), fontsize=legendsize)
            #             ax3[0, caid].text(0.01, -np.pi/2+1.1, '%s'%(pvalstr2), fontsize=legendsize)

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
            ax3[1, caid].scatter(mean_phasesph, 0.3, marker='|', color='lime', linewidth=0.75)
            ax3[1, caid].scatter(mean_phasespl, 0.3, marker='|', color='darkblue', linewidth=0.75)
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
            ps_tmp, _ = watson_williams(slopes_high, slopes_low)
            po_tmp, _ = watson_williams(offsets_high, offsets_low)
            ax_s[caid].boxplot([slopes_high, slopes_low, offsets_high, offsets_low], positions=[0, 1, 2, 3])
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
            ax_each.set_ylim(-np.pi, 1.5*np.pi)
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
            plt.setp(ax2[i, 2].get_xticklabels(), visible=False)
            plt.setp(ax3[i, 0].get_xticklabels(), visible=False)
            plt.setp(ax3[i, 2].get_xticklabels(), visible=False)


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
            fig_colorbar.savefig(os.path.join(self.plot_dir, 'adiff_colorbar.%s'%(figext)), dpi=dpi)
            fig1.savefig(os.path.join(self.plot_dir, 'Densities.%s'%(figext)), dpi=dpi)
            fig2.savefig(os.path.join(self.plot_dir, 'Marginals.%s'%(figext)), dpi=dpi)
            fig3.savefig(os.path.join(self.plot_dir, 'AverCurves_SpikePhase.%s'%(figext)), dpi=dpi)
            fig_s.savefig(os.path.join(self.plot_dir, 'HighLow-d_SanityCheck.%s'%(figext)), dpi=dpi)
        return None


    @staticmethod
    def norm_div(target_hist, divider_hist):
        target_hist_norm = target_hist / divider_hist.reshape(-1, 1)
        target_hist_norm[np.isnan(target_hist_norm)] = 0
        target_hist_norm[np.isinf(target_hist_norm)] = 0
        target_hist_norm = target_hist_norm / np.sum(target_hist_norm) * np.sum(target_hist)
        return target_hist_norm


def plot_precession_pvals(singledf, savedir):

    nap_thresh = 1
    figl = total_figw/2
    fig, ax = plt.subplots(figsize=(figl, figl))

    cafrac = []
    nall = []
    nsig = []
    for caid, (ca, cadf) in enumerate(singledf.groupby('ca')):
        thisdf = cadf[(cadf['numpass_at_precess'] >= nap_thresh) & (~cadf['precess_R_pval'].isna())].reset_index(drop=True)

        nall.append(thisdf['precess_R_pval'].shape[0])
        nsig.append(np.sum(thisdf['precess_R_pval'] < 0.05))
        cafrac.append(np.mean(thisdf['precess_R_pval'] < 0.05))
        logpval = np.log10(thisdf['precess_R_pval'].to_numpy())

        logpval[np.isinf(logpval)] = np.log10(1e-2)
        ax.hist(logpval, bins=30, cumulative=True, density=True, histtype='step', color=ca_c[ca], label=ca)


    ax.vlines(np.log10(0.05), ymin=0, ymax=1, label='p=0.05')
    ax.set_xlabel('log10(pval)', fontsize=fontsize)
    ax.set_ylabel('Cumulative Freq.', fontsize=fontsize)
    ax.set_title('CA1=%d/%d=%0.3f\nCA2=%d/%d=%0.3f\nCA3=%d/%d=%0.3f'%(
        nsig[0], nall[0], cafrac[0], nsig[1], nall[1], cafrac[1], nsig[2], nall[2], cafrac[2]
    ), fontsize=legendsize)

    customlegend(ax, fontsize=legendsize, loc='upper left')
    fig. tight_layout()
    fig.savefig(join(savedir, 'precession_significance.%s'%(figext)), dpi=dpi)




if __name__ == '__main__':

    preprocess_save_dir = 'results/exp/passes/'

    # Single Analysis
    single_plot_dir = 'result_plots/passes/single'
    os.makedirs(single_plot_dir, exist_ok=True)
    # singlefield_df = load_pickle('results/exp/single_field/singlefield_df.pickle')
    singlefield_df = load_pickle('results/exp/single_field/singlefield_df.pickle')
    # check_pass_nspikes(singlefield_df)

    # plot_precession_examples(singlefield_df, single_plot_dir)

    analyzer = SingleBestAngleAnalyzer(singlefield_df, single_plot_dir)
    analyzer.plot_field_bestprecession()
    analyzer.plot_both_slope_offset(NShuffles=1000)
    # plot_precession_pvals(singlefield_df, single_plot_dir)




