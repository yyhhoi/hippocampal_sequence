# %% Import libraries
import os

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
from scipy.stats import spearmanr, ranksums, binom_test, chi2_contingency, chisquare, linregress
import statsmodels.api as smapi
from statsmodels.formula.api import ols
from common.script_wrappers import permutation_test_average_slopeoffset, combined_average_curve
from common.comput_utils import midedges, circular_density_1d, linear_circular_gauss_density, \
    unfold_binning_2d, repeat_arr, \
    circ_ktest, shiftcyc_full2half, ci_vonmise, fisherexact
from common.linear_circular_r import rcc
from common.utils import load_pickle, sigtext, stat_record, p2str
from common.visualization import plot_marginal_slices, SqueezedNorm, customlegend

from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

# figext = 'png'
figext = 'eps'

def cal_sim(x):
    if x < (np.pi / 3):
        return 'sim'
    elif x >= (np.pi / 3) and x < (np.pi - np.pi / 3):
        return 'orth'
    elif x >= (np.pi - np.pi / 3):
        return 'opp'
    else:
        raise

def cal_sim_two(x):
    if x < (np.pi / 2):
        return 'sim'
    elif x >= (np.pi / 2):
        return 'opp'
    else:
        raise

def plot_precession_examples(singledf, save_dir):

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

    id_list = []

    fig, ax = plt.subplots(4, 3, figsize=(figl, figl*(4/3)*0.8), sharex=True, sharey=True)

    # # Assign field ids within ca
    for caid, (ca, cadf) in enumerate(singledf.groupby('ca')):
        cadf = cadf.reset_index(drop=True)

        num_fields = cadf.shape[0]



        for nf in range(num_fields):
            precess_df, fieldid_ca= cadf.loc[nf, ['precess_df', 'fieldid_ca']]
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



    fig.text(0.03, 0.35, 'Phase (rad)', ha='center', rotation=90, fontsize=fontsize)
    fig.text(0.55, 0.02, 'Position', ha='center', fontsize=fontsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(os.path.join(save_dir, 'precess_examples.%s'%(figext)), dpi=dpi)




class SingleBestAngleAnalyzer:
    def __init__(self, singlefield_df, plot_dir):
        self.singlefield_df = singlefield_df
        self.plot_dir = plot_dir
        self.anglediff_edges = np.linspace(0, np.pi, 50)
        self.slope_edges = np.linspace(-2, 0, 50)
        self.offset_edges = np.linspace(0, 2 * np.pi, 50)
        self.angle_title = r'$d(\theta_{pass}, \theta_{rate})$' + ' (rad)'
        self.absangle_title  = r'$|d(\theta_{pass}, \theta_{rate})|$' + ' (rad)'

    def plot_precession_fraction(self):


        figl = total_figw*0.6
        linew = 0.75

        fig, ax = plt.subplots(1, 3, figsize=(figl, figl/3), sharey=True, sharex=True)

        for caid, (ca, cadf) in enumerate(self.singlefield_df.groupby('ca')):

            cadf = cadf.reset_index(drop=True)

            precess_adiff = []
            precess_nspikes = []
            nonprecess_adiff = []
            nonprecess_nspikes = []

            for i in range(cadf.shape[0]):
                refangle = cadf.loc[i, 'fieldangle_mlm']
                precess_df = cadf.loc[i, 'precess_df']
                nonprecess_df = cadf.loc[i, 'nonprecess_df']
                precess_counts = precess_df.shape[0]
                nonprecess_counts = nonprecess_df.shape[0]

                if precess_counts > 0:
                    precess_adiff.append(cdiff(precess_df['spike_angle'].to_numpy(), refangle))
                    precess_nspikes.append(precess_df['pass_nspikes'].to_numpy())

                if nonprecess_counts > 0:
                    nonprecess_adiff.append(cdiff(nonprecess_df['spike_angle'].to_numpy(), refangle))
                    nonprecess_nspikes.append(nonprecess_df['pass_nspikes'].to_numpy())


            precess_adiff = np.abs(np.concatenate(precess_adiff))
            nonprecess_adiff = np.concatenate(nonprecess_adiff)
            precess_nspikes = np.abs(np.concatenate(precess_nspikes))
            nonprecess_nspikes = np.concatenate(nonprecess_nspikes)



            # For all passes (precessions)
            adiff_bins = np.linspace(0, np.pi, 41)
            adm = midedges(adiff_bins)
            spike_bins = np.zeros(adm.shape[0])
            pass_bins = np.zeros(adm.shape[0])
            for i in range(adm.shape[0]):
                adiff_p_mask = (precess_adiff >= adiff_bins[i]) & (precess_adiff < adiff_bins[i + 1])
                adiff_np_mask = (nonprecess_adiff >= adiff_bins[i]) & (nonprecess_adiff < adiff_bins[i + 1])

                count_p = adiff_p_mask.sum()
                count_np = adiff_np_mask.sum()
                pass_bins[i] = np.sum(count_p/(count_p + count_np))

                spike_bins[i] = np.sum(precess_nspikes[adiff_p_mask]) + np.sum(nonprecess_nspikes[adiff_np_mask])


            norm_bins = (pass_bins / spike_bins)
            rho, pval = spearmanr(adm, norm_bins)
            pm, pc, pr, ppval, _ = linregress(adm, norm_bins/ norm_bins.sum())
            xdum = np.linspace(adm.min(), adm.max(), 10)
            ax[0].step(adm, pass_bins / pass_bins.sum(), color=ca_c[ca], linewidth=linew,
                                  label=ca)
            ax[1].step(adm, spike_bins / spike_bins.sum(), color=ca_c[ca], linewidth=linew)
            ax[2].step(adm, norm_bins / norm_bins.sum(), color=ca_c[ca], linewidth=linew)
            ax[2].plot(xdum, xdum*pm+pc, linewidth=linew, color=ca_c[ca])
            ax[2].text(0, 0.0475 - caid*0.007, 'p%s'%(p2str(pval)), fontsize=legendsize, color=ca_c[ca])

        ax[0].set_title('Precession')
        ax[1].set_title('Spike')
        ax[2].set_title('Ratio')



        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, 'precess_fraction.png'), dpi=300)

    def plot_field_bestprecession(self, plot_example=False):

        figl = total_figw*0.6
        linew = 0.75

        R_figl = total_figw*0.4
        print('Plot field best precession')
        stat_fn = 'fig2_field_precess.txt'
        stat_record(stat_fn, True)
        pass_ax = np.linspace(-np.pi, np.pi, num=100)
        fig_R, ax_R = plt.subplots(figsize=(R_figl, R_figl*0.8))
        fig_pishift = plt.figure(figsize=(figl, figl*4/3))
        ax_pishift = np.array([
            [fig_pishift.add_subplot(4, 3, i ) for i in range(1, 4)],
            [fig_pishift.add_subplot(4, 3, i) for i in range(4, 7)],
            [fig_pishift.add_subplot(4, 3, i, polar=True) for i in range(7, 10)],
            [fig_pishift.add_subplot(4, 3, i, polar=True) for i in range(10, 13)]
        ])

        caR = []
        allca_ntotal = self.singlefield_df.shape[0]
        all_shufp = []
        caid = 0
        fid = 0
        for ca, cadf in self.singlefield_df.groupby('ca'):
            cadf = cadf.reset_index(drop=True)

            # For all passes (precessions)
            anglediff, pass_nspikes = self.stack_anglediff(cadf)
            anglediff = np.abs(anglediff)
            adiff_bins = np.linspace(0, np.pi, 41)
            adm = midedges(adiff_bins)
            spike_bins = np.zeros(adm.shape[0])
            pass_bins = np.zeros(adm.shape[0])
            for i in range(adm.shape[0]):
                pass_mask = (anglediff >= adiff_bins[i]) & (anglediff < adiff_bins[i + 1])
                spike_bins[i] = np.sum(pass_nspikes[pass_mask])
                pass_bins[i] = np.sum(pass_mask)
            norm_bins = (pass_bins / spike_bins)
            rho, pval = spearmanr(adm, norm_bins)
            pm, pc, pr, ppval, _ = linregress(adm, norm_bins/ norm_bins.sum())
            xdum = np.linspace(adm.min(), adm.max(), 10)
            ax_pishift[0, 0].step(adm, pass_bins / pass_bins.sum(), color=ca_c[ca], linewidth=linew,
                             label=ca)
            ax_pishift[0, 1].step(adm, spike_bins / spike_bins.sum(), color=ca_c[ca], linewidth=linew)
            ax_pishift[0, 2].step(adm, norm_bins / norm_bins.sum(), color=ca_c[ca], linewidth=linew)
            ax_pishift[0, 2].plot(xdum, xdum*pm+pc, linewidth=linew, color=ca_c[ca])
            ax_pishift[0, 2].text(0, 0.0475 - caid*0.007, 'p%s'%(p2str(pval)), fontsize=legendsize, color=ca_c[ca])

            stat_record(stat_fn, False, "%s pass_num=%d, spike_num=%d, Spearman's: rho=%0.2f, pval=%0.4f, Pearson's: m=%0.2f, rho=%0.2f, pval=%0.4f " % \
                        (ca, anglediff.shape[0], pass_nspikes.sum(), rho, pval, pm, pr, ppval))

            # For each individual fields
            ntotal_fields = cadf.shape[0]
            data = {'field': [], 'precess': [], 'precess_low': [], 'R': [], 'shufp': []}

            for i in range(ntotal_fields):

                precess_info = cadf.loc[i, 'precess_info']
                if precess_info is None:
                    continue
                fieldangle_mlm, precess_angle = cadf.loc[i, ['fieldangle_mlm', 'precess_angle']]
                precess_angle_low = cadf.loc[i, 'precess_angle_low']
                R = precess_info['R']
                data['field'].append(fieldangle_mlm)
                data['precess'].append(precess_angle)
                data['precess_low'].append(precess_angle_low)
                data['R'].append(precess_info['R'])
                data['shufp'].append(precess_info['pval'])

                # Save example
                if plot_example:
                    print('ploting %s %d/%d' % (ca, i, cadf.shape[0]))
                    norm_density = precess_info['norm_prob']
                    fig_example = plt.figure(figsize=(1.75, 1.75))
                    ax_example = fig_example.add_subplot(111, polar=True)
                    ax_example.plot(pass_ax, norm_density, c='k')
                    ax_example.plot([pass_ax[-1], pass_ax[0]], [norm_density[-1], norm_density[0]], c='k')
                    ax_example.plot([precess_angle, precess_angle], [0, norm_density.max()], c='r')

                    l = norm_density.max()
                    vec = 0.3 * l - 1.1 * l * 1j
                    ax_example.text(np.angle(vec), np.abs(vec), '%0.2f' % (R),
                                    fontsize=14, c='k')
                    ax_example.axis('off')
                    ax_example.grid(False)
                    fig_example.tight_layout()
                    fig_example.savefig(os.path.join(self.plot_dir, 'example_field_bestangle', '%s_%d.png' % (ca, i)),
                                        dpi=dpi)
                    plt.close()
                fid += 1

            # Plot R
            all_R = np.array(data['R'])
            caR.append(all_R)
            rbins, redges = np.histogram(all_R, bins=50)
            rbinsnorm = np.cumsum(rbins) / rbins.sum()
            ax_R.plot(midedges(redges), rbinsnorm, label=ca, c=ca_c[ca], linewidth=linew)

            # Plot Scatter: field vs precess angles
            ax_pishift[1, caid].scatter(data['field'], data['precess'], marker='.', c=ca_c[ca], s=2)
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
            fieldangles = np.array(data['field'])
            precessangles = np.array(data['precess'])
            nanmask = (~np.isnan(fieldangles)) & (~np.isnan(precessangles))
            adiff = cdiff(precessangles[nanmask], fieldangles[nanmask])

            bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
            bins_norm = bins / np.sum(bins)
            l = bins_norm.max()
            ax_pishift[2, caid].bar(midedges(edges), bins_norm, width=edges[1] - edges[0],
                                    color=ca_c[ca])

            linewidth = 1
            mean_angle = circmean(adiff)
            mean_angle = np.mod(mean_angle + np.pi, 2 * np.pi) - np.pi
            ax_pishift[2, caid].plot([mean_angle, mean_angle], [0, l], c='k', linestyle='dashed', linewidth=linewidth)
            ax_pishift[2, caid].plot([0, 0], [0, l], c='k', linewidth=linewidth)
            ax_pishift[2, caid].scatter(0, 0, s=16, c='gray')
            ax_pishift[2, caid].text(0.4, l * 0.4, r'$\theta_{rate}$', fontsize=fontsize + 1)
            ax_pishift[2, caid].spines['polar'].set_visible(False)
            ax_pishift[2, caid].set_yticklabels([])
            ax_pishift[2, caid].set_xticklabels([])
            ax_pishift[2, caid].grid(False)
            v_pval, v_stat = vtest(adiff, mu=np.pi)
            pvalstr = 'p=%0.4f'%(v_pval) if v_pval > 0.0001 else 'p<0.0001'
            ax_pishift[2, caid].text(x=0.01, y=0.95, s=pvalstr, fontsize=legendsize,
                                     transform=ax_pishift[2, caid].transAxes)
            stat_record(stat_fn, False, '%s, d(precess, rate), V=%0.2f, n=%d, p=%0.4f' % (ca, v_stat, bins.sum(),
                                                                                          v_pval))

            # Plot Histogram: d(precess_low, rate)
            fieldangles = np.array(data['field'])
            precessangles_low = np.array(data['precess_low'])
            nanmask = (~np.isnan(fieldangles)) & (~np.isnan(precessangles_low))
            adiff = cdiff(precessangles_low[nanmask], fieldangles[nanmask])
            bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
            bins_norm = bins / np.sum(bins)
            l = bins_norm.max()
            ax_pishift[3, caid].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], color=ca_c[ca])
            mean_angle = circmean(adiff)
            mean_angle = np.mod(mean_angle + np.pi, 2 * np.pi) - np.pi
            ax_pishift[3, caid].plot([mean_angle, mean_angle], [0, l], c='k', linestyle='dashed', linewidth=linewidth)
            ax_pishift[3, caid].plot([0, 0], [0, l], c='k', linewidth=linewidth)
            ax_pishift[3, caid].scatter(0, 0, s=16, c='gray')
            ax_pishift[3, caid].text(0.4, l * 0.4, r'$\theta_{rate}$', fontsize=fontsize + 1)
            ax_pishift[3, caid].spines['polar'].set_visible(False)
            ax_pishift[3, caid].set_yticklabels([])
            ax_pishift[3, caid].set_xticklabels([])
            ax_pishift[3, caid].grid(False)
            v_pval, v_stat = vtest(adiff, mu=np.pi)
            pvalstr = 'p=%0.4f'%(v_pval) if v_pval > 0.0001 else 'p<0.0001'
            ax_pishift[3, caid].text(x=0.01, y=0.95, s=pvalstr, fontsize=legendsize,
                                     transform=ax_pishift[3, caid].transAxes)
            stat_record(stat_fn, False, '%s, d(precess_low, rate), V=%0.2f, n=%d, p=%0.4f' % (ca, v_stat, bins.sum(),
                                                                                              v_pval))

            # Export stats: Sig fraction
            all_shufp = all_shufp + data['shufp']
            ca_shufp = np.array(data['shufp'])
            sig_num = int(np.sum(ca_shufp < 0.05))
            sig_percent = sig_num / ntotal_fields
            binom_p = binom_test(x=sig_num, n=ntotal_fields, p=0.05, alternative='greater')

            stat_record(stat_fn, False, "Shufpval-sigfrac, %s, frac=%d/%d, percentage=%0.2f, Binomial test p=%0.4f" % \
                        (ca, sig_num, ntotal_fields, sig_percent * 100, binom_p))

            caid += 1

        all_shufp = np.array(all_shufp)
        all_signum = int(np.sum(all_shufp < 0.05))
        all_sigpercent = all_signum / allca_ntotal
        stat_record(stat_fn, False, 'Shufpval_sigfrac', "ALL", 'frac=%d/%d' % (all_signum, allca_ntotal),
                    'percentage=%0.2f' % (all_sigpercent * 100))

        # R
        z13, p13 = ranksums(caR[0], caR[2])
        stat_record(stat_fn, False, 'CA13, R difference, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f' % \
                    (z13, caR[0].shape[0], caR[2].shape[0], p13))
        ax_R.text(0.285, 0.4, 'CA1-3 diff.\np%s'%(p2str(p13)), fontsize=legendsize)
        ax_R.set_xlabel('R', fontsize=fontsize)
        ax_R.set_ylabel('Cumulative density', fontsize=fontsize)
        customlegend(ax_R, linewidth=1.5, fontsize=legendsize)
        ax_R.set_yticks([0, 0.5, 1])
        ax_R.set_yticklabels(['0', '', '1'])
        ax_R.tick_params(axis='both', which='major', labelsize=ticksize)
        fig_R.tight_layout()
        fig_R.savefig(os.path.join(self.plot_dir, 'field_precess_R.%s'%(figext)), dpi=dpi)

        # 2D scatter
        ax_pishift[1, 0].set_ylabel(r'$\theta_{Precess}$', fontsize=fontsize)

        # All fields
        for ax_each in ax_pishift[0, ]:
            ax_each.set_xticks([0, np.pi / 2, np.pi])
            ax_each.set_xticklabels(['0', '$\pi/2$', '$\pi$'])
            ax_each.set_ylim([0.01, 0.05])
            # ax_each.set_xlabel(self.angle_title, fontsize=fontsize)
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


        # Pi-shift
        plt.setp(ax_pishift[1, 1].get_yticklabels(), visible=False)
        plt.setp(ax_pishift[1, 2].get_yticklabels(), visible=False)
        ax_pishift[2, 0].set_ylabel('All\npasses', fontsize=fontsize)
        ax_pishift[2, 0].yaxis.labelpad = 15
        ax_pishift[3, 0].set_ylabel('Low-spike\n passes', fontsize=fontsize)
        ax_pishift[3, 0].yaxis.labelpad = 15


        fig_pishift.tight_layout()
        fig_pishift.subplots_adjust(wspace=0.25, hspace=1.25)

        fig_pishift.savefig(os.path.join(self.plot_dir, 'field_precess_pishift.%s'%(figext)), dpi=dpi)

    def plot_both_slope_offset(self, NShuffles=200):

        selected_adiff = np.linspace(0, np.pi, 6)  # 20
        adiff_ticks = [0, np.pi / 2, np.pi]
        adiff_ticksl = ['0', '', '$\pi$']
        adiff_label = r'$|d(\theta_{pass}, \theta_{precess})|$'

        offset_label = 'Onset phase (rad)'
        offset_slicerange = (0, 2*np.pi)
        offset_bound = (0, 2 * np.pi)
        offset_xticks = [1, 3, 5]
        offset_xticksl = ['%d' % x for x in offset_xticks]
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


        figl = total_figw/6
        fig1size = (figl*3 * 0.95, figl*4.25)  # 0.95 as a space for colorbar
        fig2size = (figl*3 * 0.95, figl*4.25)

        stat_fn = 'fig3_slopeoffset.txt'
        stat_record(stat_fn, True, 'Average Phase precession')
        fig1, ax1 = plt.subplots(3, 3, figsize=fig1size, sharey='row')  # Density + aver phase
        fig2, ax2 = plt.subplots(3, 3, figsize=fig2size, sharey='row')  # Marginals + concentrations


        caid = 0
        for ca, cadf in self.singlefield_df.groupby('ca'):
            cadf = cadf.reset_index(drop=True)
            anglediff, pass_nspikes = self.stack_anglediff(cadf, precess_ref=True)
            anglediff = np.abs(anglediff)
            slope_piunit = self.stack_key(cadf, 'rcc_m')
            slope = slope_piunit * 2 * np.pi
            offset = self.stack_key(cadf, 'rcc_c')

            # # Plot density, slices
            # Expand sample according to spikes
            anglediff_spikes = repeat_arr(anglediff, pass_nspikes)

            # 1D spike hisotgram
            spikes_bins, spikes_edges = np.histogram(anglediff_spikes, bins=adiff_edges)

            # 2D slope/offset histogram
            offset_bins, offset_xedges, offset_yedges = np.histogram2d(anglediff, offset,
                                                                       bins=(adiff_edges, offset_edges))
            slope_bins, slope_xedges, slope_yedges = np.histogram2d(anglediff, slope,
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

            stat_record(stat_fn, False, 'LC_Regression %s Onset-adiff rho=%0.2f, pval=%0.4f'%(ca, offset_rho, offset_p))
            stat_record(stat_fn, False, 'LC_Regression %s Slope-adiff rho=%0.2f, pval=%0.4f'%(ca, slope_rho, slope_p))

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
            ax2[0, caid].set_title(ca, fontsize=titlesize)

            plot_marginal_slices(ax2[1, caid], slope_xx, slope_yy, slope_zz,
                                                      selected_adiff, slope_slicerange, slope_slicegap)
            ax2[1, caid].set_xticks(slope_xticks)
            ax2[1, caid].set_xticklabels(slope_xticksl)
            ax2[1, 1].set_xlabel(slope_label, fontsize=fontsize)
            ax2[1, caid].tick_params(labelsize=ticksize)
            ax2[1, caid].set_xlim(-2*np.pi, 0)

            # # Plot average precession curves
            phasesp = np.concatenate(self.stack_key(cadf, 'phasesp'))

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

            slopes_high, offsets_high = slope_piunit[high_mask][high_ranvec], offset[high_mask][high_ranvec]
            slopes_low, offsets_low = slope_piunit[low_mask][low_ranvec], offset[low_mask][low_ranvec]

            regress_high, regress_low, pval_slope, pval_offset = permutation_test_average_slopeoffset(
                slopes_high, offsets_high, slopes_low, offsets_low, NShuffles=NShuffles)

            stat_record(stat_fn, False,
                        '%s d=high, rho=%0.2f, p=%0.4f, slope=%0.2f, offset=%0.2f' % \
                        (ca, regress_high['rho'], regress_high['p'], regress_high['aopt'] * 2 * np.pi,
                         regress_high['phi0']))
            stat_record(stat_fn, False,
                        '%s d=low, rho=%0.2f, p=%0.4f, slope=%0.2f, offset=%0.2f' % \
                        (ca, regress_low['rho'], regress_low['p'], regress_low['aopt'] * 2 * np.pi,
                         regress_low['phi0']))
            stat_record(stat_fn, False,
                        '%s high-low-difference, Slope: p=%0.4f; Offset: p=%0.4f' % \
                        (ca, pval_slope, pval_offset))


            xdum = np.linspace(0, 1, 10)
            high_agg_ydum = 2 * np.pi * regress_high['aopt'] * xdum + regress_high['phi0']
            low_agg_ydum = 2 * np.pi * regress_low['aopt'] * xdum + regress_low['phi0']

            # # Compare high vs low
            ax1[2, caid].plot(xdum, high_agg_ydum, c='lime', label='$|d|>5\pi/6$')
            ax1[2, caid].plot(xdum, low_agg_ydum, c='darkblue', label='$|d|<\pi/6$')

            pvalstr1 = '$p_s$'+'=%0.4f'%(pval_slope) if pval_slope > 1e-4 else '$p_s$'+'<0.0001'
            pvalstr2 = '$p_o$'+'=%0.4f'%(pval_offset) if pval_offset > 1e-4 else '$p_o$'+'<0.0001'
            ax1[2, caid].text(0.01, -np.pi+0.3, '%s\n%s'%(pvalstr1, pvalstr2), fontsize=legendsize)
            ax1[2, caid].spines["top"].set_visible(False)
            ax1[2, caid].spines["right"].set_visible(False)

            # # Concentrations
            aedges = np.linspace(-np.pi, np.pi, 36)
            fstat, k_pval = circ_ktest(phasesph, phasespl)
            p_ww, ww_table = watson_williams(phasesph, phasespl)
            ax2[2, caid].hist(phasesph, bins=aedges, density=True, histtype='step', color='lime')
            ax2[2, caid].hist(phasespl, bins=aedges, density=True, histtype='step', color='darkblue')
            pvalstr = "p=%0.4f"%(k_pval) if k_pval > 1e-4 else "p<0.0001"
            ax2[2, caid].text(-np.pi+0.2, 0.02, pvalstr, fontsize=legendsize)
            ax2[2, caid].set_xlim(-np.pi, np.pi)
            ax2[2, caid].set_xticks([-np.pi, 0, np.pi])
            ax2[2, caid].set_xticklabels(['$-\pi$', '0', '$\pi$'])
            ax2[2, caid].set_yticks([])
            ax2[2, caid].tick_params(labelsize=ticksize)
            ax2[2, caid].spines["top"].set_visible(False)
            ax2[2, caid].spines["right"].set_visible(False)
            ax2[2, caid].spines["left"].set_visible(False)
            ax2[2, 1].set_xlabel('Phase (rad)', fontsize=fontsize)
            ax2[2, 0].set_ylabel('Relative\nfrequency', fontsize=fontsize)

            stat_record(stat_fn, False, 'SpikePhase-HighLowDiff %s Mean_diff p=%0.4f' % (ca, p_ww))
            stat_record(stat_fn, False, ww_table.to_string())
            stat_record(stat_fn, False,
                        'SpikePhase-HighLowDiff %s Bartlett\'s test F_{(%d, %d)}=%0.2f, p=%0.4f' % \
                        (ca, phasesph.shape[0], phasespl.shape[0], fstat, k_pval))

            caid += 1



        # Asthestic for density and slices
        ax1[0, 0].set_ylabel(offset_label, fontsize=fontsize)
        ax1[0, 0].yaxis.labelpad = 18
        ax1[1, 0].set_ylabel(slope_label, fontsize=fontsize)
        fig2.text(0.05, 0.55, 'Marginal density\n of precession', rotation=90, fontsize=fontsize)
        # ax2[0, 0].set_ylabel('Marginal density\n of precession ', fontsize=fontsize)
        # ax2[0, 0].yaxis.labelpad = 5
        # ax2[1, 0].set_ylabel('Marginal density\n of precession ', fontsize=fontsize)
        # ax2[1, 0].yaxis.labelpad = 5
        ax2[2, 0].yaxis.labelpad = 15
        for ax_each in np.append(ax2[0, ], ax2[1, ]):
            # ax_each.axes.get_yaxis().set_visible(False)
            ax_each.set_yticks([])
            ax_each.grid(False, axis='y')
            ax_each.spines["top"].set_visible(False)
            ax_each.spines["right"].set_visible(False)
            ax_each.spines["left"].set_visible(False)


        # Asthestics for average curves
        for ax_each in ax1[2, ]:
            ax_each.set_xlim(0, 1)
            ax_each.set_ylim(-np.pi, np.pi + 0.3)
            ax_each.set_yticks([-np.pi, 0, np.pi])
            ax_each.set_yticklabels(['$-\pi$', '0', '$\pi$'])
            ax_each.tick_params(labelsize=ticksize)
        ax1[2, 1].set_xlabel('Position')
        customlegend(ax1[2, 2], fontsize=legendsize, loc='lower left', handlelength=0.5, bbox_to_anchor=(0.05, 0.75))
        ax1[2, 0].set_ylabel('Phase (rad)', fontsize=fontsize)
        ax1[2, 0].yaxis.labelpad = 10

        # Asthestics for all
        for i in range(3):
            plt.setp(ax1[i, 0].get_xticklabels(), visible=False)
            plt.setp(ax1[i, 2].get_xticklabels(), visible=False)
            plt.setp(ax2[i, 0].get_xticklabels(), visible=False)
            plt.setp(ax2[i, 2].get_xticklabels(), visible=False)


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
        fig1.savefig(os.path.join(self.plot_dir, 'Density_Aver.%s'%(figext)), dpi=dpi)

        fig2.tight_layout()
        fig2.subplots_adjust(wspace=0.2, hspace=1.2, left=0.2)

        fig2.savefig(os.path.join(self.plot_dir, 'Marginal_Concentrations.%s'%(figext)), dpi=dpi)

        fig_colorbar.savefig(os.path.join(self.plot_dir, 'adiff_colorbar.%s'%(figext)), dpi=dpi)
        return None


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


def plot_rcc(ax, x, y, xrange=(0, 2 * np.pi), yrange=(-np.pi, 3 * np.pi), tag='',
                    setticks=True, settitle=True, legend=True):


    # Rcc
    xdum = np.linspace(xrange[0], xrange[1], 100)  # rcc_cAB (0, 2pi) as x
    reg = rcc(x, y)
    m, c, rholc, pvallc = reg['aopt'], reg['phi0'], reg['rho'], reg['p']
    y_predlc = 2 * np.pi * m * xdum + c

    # Plot RCC regression
    if legend:
        text = 'rho=%0.2f %s'%(rholc, sigtext(pvallc, hidens=True))
        ax.plot(xdum, y_predlc, c='k', label='rho=%0.2f %s'%(rholc, sigtext(pvallc, hidens=True)))
        ax.annotate(text, xy=(0.95, 0.95), xycoords='axes fraction',
                    size=fontsize+1, ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.8))
    else:
        ax.plot(xdum, y_predlc, c='k')
    ax.plot(xdum, y_predlc + 2 * np.pi, c='k')
    ax.plot(xdum, y_predlc - 2 * np.pi, c='k')

    # Axes title
    if settitle:
        ax.set_title(tag + '\nrho=%0.2f(%s)' % (rholc, sigtext(pvallc)))

    # Axes ticks
    if setticks:
        ax.set_yticks([-3 * np.pi, -2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
        ax.set_yticklabels(['$-3\pi$', '$-2\pi$', '$-\pi$', '$0$', '$\pi$', '$2\pi$', '$3\pi$'], fontsize=fontsize)
        ax.set_xticks([-3 * np.pi, -2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
        ax.set_xticklabels(['$-3\pi$', '$-2\pi$', '$-\pi$', '$0$', '$\pi$', '$2\pi$', '$3\pi$'],
                           fontsize=fontsize)

    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    return ax

def set_xyticks(ax, fontsize):
    ax.set_xticks([0, np.pi, np.pi * 2])
    ax.set_xticklabels(['0', '$\pi$', '$2\pi$'])
    ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
    ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$', '$2\pi$', '$3\pi$'])
    ax.tick_params(labelsize=fontsize)


def pairlag_phase_correlation_similar_passes(pairfield_df, fig_dir, usephase=False):
    phasetag = '_phase' if usephase else ''
    ratio_key, oplus_key, ominus_key = 'overlap_ratio'+phasetag, 'overlap_plus'+phasetag, 'overlap_minus'+phasetag

    def plot_marginal(ax, offset, lag, extrinsic, hist_maxh, hist_maxw, ex_c, in_c, both_c, only_both=False):

        vmarker, hmarker = '|', '_'
        msize = 128
        inoffset, exoffset = offset[extrinsic==0], offset[extrinsic==1]

        if only_both is False:
            in_offset_ax, in_offset_den = circular_density_1d(inoffset, 3*np.pi, 100, (0, 2*np.pi))
            in_offset_den = in_offset_den/in_offset_den.max() * hist_maxh - (np.pi + hist_maxh)
            ax.plot(in_offset_ax, in_offset_den, c=in_c, alpha=0.5)
            ax.scatter(circmean(inoffset), -np.pi - hist_maxh/2, marker=vmarker, s=msize, c=in_c, alpha=0.7)

            ex_offset_ax, ex_offset_den = circular_density_1d(exoffset, 3*np.pi, 100, (0, 2*np.pi))
            ex_offset_den = ex_offset_den/ex_offset_den.max() * hist_maxh - (np.pi + hist_maxh)
            ax.plot(ex_offset_ax, ex_offset_den, c=ex_c, alpha=0.5)
            ax.scatter(circmean(exoffset), -np.pi - hist_maxh/2, marker=vmarker, s=msize, c=ex_c, alpha=0.7)

        both_offset_ax, both_offset_den = circular_density_1d(offset, 3*np.pi, 100, (0, 2*np.pi))
        both_offset_den = both_offset_den/both_offset_den.max() * hist_maxh - (np.pi + hist_maxh)
        ax.plot(both_offset_ax, both_offset_den, c=both_c, alpha=0.5)
        ax.scatter(circmean(offset), -np.pi - hist_maxh/2, marker=vmarker, s=msize, c=both_c, alpha=0.7)


        ax_twin = ax.twinx()
        inlag, exlag = lag[extrinsic==0], lag[extrinsic==1]

        if only_both is False:
            in_lag_ax, in_lag_den = circular_density_1d(inlag, 3*np.pi, 100, (-np.pi, 3 * np.pi))
            in_lag_den = in_lag_den/in_lag_den.max() * hist_maxw - (0 + hist_maxw)
            ax_twin.plot(in_lag_den, in_lag_ax, c=in_c, alpha=0.5 )
            ax_twin.scatter(0-hist_maxw/2, circmean(inlag), marker=hmarker, s=msize, c=in_c, alpha=0.7)

            ex_lag_ax, ex_lag_den = circular_density_1d(exlag, 3*np.pi, 100, (-np.pi, 3 * np.pi))
            ex_lag_den = ex_lag_den/ex_lag_den.max() * hist_maxw - (0 + hist_maxw)
            ax_twin.plot(ex_lag_den, ex_lag_ax, c=ex_c, alpha=0.5 )
            ax_twin.scatter(0-hist_maxw/2, circmean(exlag), marker=hmarker, s=msize, c=ex_c, alpha=0.7)

        both_lag_ax, both_lag_den = circular_density_1d(lag, 3*np.pi, 100, (-np.pi, 3 * np.pi))
        both_lag_den = both_lag_den/both_lag_den.max() * hist_maxw - (0 + hist_maxw)
        ax_twin.plot(both_lag_den, both_lag_ax, c=both_c, alpha=0.5 )
        ax_twin.scatter(0-hist_maxw/2, circmean(lag), marker=hmarker, s=msize, c=both_c, alpha=0.7)

        ax_twin.set_xlim(0 - hist_maxw, 2 * np.pi)
        ax_twin.set_ylim(-np.pi - hist_maxh, 3 * np.pi)
        ax_twin.axis('off')
        return None

    def flip_direction(direction):
        if direction == "A->B":
            return "B->A"
        elif direction == "B->A":
            return "A->B"
        else:
            return direction


    # Two figures - Similar fields, opposite fields
    # Each figure has 3 categories: LagAB, LagBA, LagAB combined with flipped lagBA
    # In similar fields, each category contains two rows (similar pass, opposite pass), 3 cols (CA1-3)
    # In opposite fields, each category contains two rows (similar to 1st field, similar to 2nd field), 3 cols (CA1-3)

    sim_func = cal_sim

    fig_all, ax_all = plt.subplots(3, 4, figsize=(1.75*4, 1.75*3), sharex=True, sharey=True)
    fig_maronset, ax_maronset = plt.subplots(figsize=(1.75*4, 1.75), sharex=True, sharey=True)
    fig_nspikes, ax_nspikes = plt.subplots(figsize=(1.75*4, 1.75), sharex=True, sharey=True)

    stat_fn = 'fig8_Pairlag-Onset.txt'
    stat_record(stat_fn, True)

    ms = 8

    # Filtering

    pairfield_df_nonan = pairfield_df[(~pairfield_df['phaselag_AB'].isna()) &
                           (~pairfield_df['phaselag_BA'].isna()) & (~pairfield_df[ratio_key].isna())].reset_index(drop=True)
    pairfield_df_nonan['extrinsic'] = pairfield_df_nonan[ratio_key] > 0
    df_dict = dict(ca=[], pairlagAB=[], pairlagBA=[], offset1=[], offset2=[], slope1=[], slope2=[],
                   precess_angle1=[], precess_angle2=[], pass_angle1=[], pass_angle2=[], rate1=[], rate2=[], extrinsic=[],
                   direction=[], ABalign=[], exin_bias=[], pairuid=[], precess_exist=[], overlap=[])
    vabsmax = max(pairfield_df_nonan[ratio_key].min(), pairfield_df_nonan[ratio_key].max())
    vmin, vmax = -vabsmax, vabsmax


    # ABallign: True if A->B direction aligns with mean precession angles (no flipping needed). Otherwise False
    pairuid = 0
    simpair_d = dict(half=[], onethird=[])
    for caid, (ca, cadf) in enumerate(pairfield_df_nonan.groupby('ca')):
        # Get data
        cadf = cadf.reset_index(drop=True)
        # 'precess_angle' is in range [0, pi]
        cadf = cadf[(~cadf['precess_angle1'].isna()) & (~cadf['precess_angle2'].isna())].reset_index(drop=True)

        precess_dist = np.abs(cdiff(cadf['precess_angle1'].to_numpy(), cadf['precess_angle2'].to_numpy()))
        simpair_d['onethird'].append([np.sum(precess_dist < (np.pi/3)), np.sum(precess_dist>(np.pi - np.pi/3))])
        simpair_d['half'].append([np.sum(precess_dist < (np.pi/2)), np.sum(precess_dist>(np.pi/2))])

        for i in range(cadf.shape[0]):
            precess_dfp = cadf.loc[i, 'precess_dfp']
            nonprecess_dfp = cadf.loc[i, 'nonprecess_dfp']
            allpass_dfp = pd.concat([precess_dfp, nonprecess_dfp], axis=0, ignore_index=True)

            num_precess = allpass_dfp.shape[0]
            if num_precess < 1:
                continue

            abtmp, batmp, fa1tmp, fa2tmp, extrinsictmp = cadf.loc[
                i, ['phaselag_AB', 'phaselag_BA', 'precess_angle1', 'precess_angle2', 'extrinsic']]
            exin_biastmp, overlap = cadf.loc[i, [ratio_key, 'overlap']]

            # flipping of label A, B
            pos1, pos2 = cadf.loc[i, ['fieldcoor1', 'fieldcoor2']]
            mean_precess_angle = circmean([fa1tmp, fa2tmp])
            posdiff = pos2 - pos1
            field_orient = np.angle(posdiff[0] + 1j * posdiff[1])
            absdiff = np.abs(cdiff(field_orient, mean_precess_angle))
            ABalign = True if absdiff < np.pi/2 else False

            # Rate of each pass
            rate_tmp1 = allpass_dfp['tsp_withtheta1'].apply(lambda x: x.shape[0]).to_list()
            rate_tmp2 = allpass_dfp['tsp_withtheta2'].apply(lambda x: x.shape[0]).to_list()


            # Append
            df_dict['ca'].extend([ca] * num_precess)
            df_dict['pairlagAB'].extend([abtmp] * num_precess)
            df_dict['pairlagBA'].extend([batmp] * num_precess)
            df_dict['precess_angle1'].extend([fa1tmp] * num_precess)
            df_dict['precess_angle2'].extend([fa2tmp] * num_precess)
            df_dict['extrinsic'].extend([extrinsictmp] * num_precess)
            df_dict['ABalign'].extend([ABalign] * num_precess)
            df_dict['exin_bias'].extend([exin_biastmp] * num_precess)
            df_dict['pairuid'].extend([pairuid] * num_precess)
            df_dict['overlap'].extend([overlap] * num_precess)
            df_dict['offset1'].extend(allpass_dfp['rcc_c1'].to_list())
            df_dict['offset2'].extend(allpass_dfp['rcc_c2'].to_list())
            df_dict['slope1'].extend(allpass_dfp['rcc_m1'].to_list())
            df_dict['slope2'].extend(allpass_dfp['rcc_m2'].to_list())
            df_dict['pass_angle1'].extend(allpass_dfp['spike_angle1'].to_list())  #  [-pi, pi]
            df_dict['pass_angle2'].extend(allpass_dfp['spike_angle2'].to_list())  #  [-pi, pi]
            df_dict['rate1'].extend(rate_tmp1)
            df_dict['rate2'].extend(rate_tmp2)
            df_dict['direction'].extend(allpass_dfp['direction'].to_list())
            df_dict['precess_exist'].extend(allpass_dfp['precess_exist'].to_list())

            pairuid += 1
    df_all = pd.DataFrame(df_dict)

    for mode in ['half', 'onethird']:
        simnum1, oppnum1 = simpair_d[mode][0]
        simnum3, oppnum3 = simpair_d[mode][2]
        pval = fisherexact(np.array([[simnum1, oppnum1], [simnum3, oppnum3]]))
        stat_record(stat_fn, False, 'Similar-dissimilar (%s) pair ratio CA1 = %d:%d = %0.2f, CA3 = %d:%d =%0.2f, Fisher pval = %0.4f' % \
                    (mode, simnum1, oppnum1, simnum1/(simnum1+oppnum1), simnum3, oppnum3, simnum3/(simnum3+oppnum3), pval))

    # Field Similarity: "sim", "opp"
    df_all['similarity'] = np.abs(
        cdiff(df_all['precess_angle1'].to_numpy(), df_all['precess_angle2'].to_numpy()))
    df_all['similarity'] = df_all['similarity'].apply(sim_func)

    # Flip AB labels for df_sim: pairlag, offset, slope, pass, precess, direction
    df_all_clone = df_all.copy()  # Pointer alert
    flipmask = (~df_all['ABalign']) & (df_all['similarity'] == 'sim')  # Only flip similar's
    df_all.loc[flipmask, 'pairlagAB'] = -df_all_clone.loc[flipmask, 'pairlagBA']
    df_all.loc[flipmask, 'pairlagBA'] = -df_all_clone.loc[flipmask, 'pairlagAB']
    df_all.loc[flipmask, 'direction'] = df_all_clone.loc[flipmask, 'direction'].apply(flip_direction)
    for label_tmp in ['offset', 'slope', 'precess_angle', 'pass_angle']:
        df_all.loc[flipmask, label_tmp + '1'] = df_all_clone.loc[flipmask, label_tmp + '2']
        df_all.loc[flipmask, label_tmp + '2'] = df_all_clone.loc[flipmask, label_tmp + '1']

    # Combine offset 1,2 and slope 1,2
    def combine_pass12(df):
        df_tmp1 = df.drop(['offset1', 'slope1', 'pass_angle1', 'rate1'], axis=1).rename(columns={'offset2':'offset',
                                                                                                 'slope2':'slope',
                                                                                                 'pass_angle2':'pass_angle',
                                                                                                 'rate2':'rate'})
        df_tmp2 = df.drop(['offset2', 'slope2', 'pass_angle2', 'rate2'], axis=1).rename(columns={'offset1':'offset',
                                                                                                 'slope1':'slope',
                                                                                                 'pass_angle1':'pass_angle',
                                                                                                 'rate1':'rate'})
        df_comrcc = pd.concat([df_tmp1, df_tmp2], axis=0, ignore_index=True)
        return df_comrcc
    df_allcom = combine_pass12(df_all)

    # Separate df_sim and df_opp
    simmask, oppmask = df_allcom['similarity'] == 'sim', df_allcom['similarity'] == 'opp'

    # Pass groups for similar fields (similar/dissimilar to best precession angle)
    df_allcom['mean_precess_angle'] = shiftcyc_full2half(
        circmean(df_allcom[['precess_angle1', 'precess_angle2']].to_numpy(), axis=1))
    df_allcom['d_pass_precess'] = np.abs(cdiff(df_allcom['mean_precess_angle'], df_allcom['pass_angle']))
    df_allcom['passgroup'] = df_allcom['d_pass_precess'].apply(sim_func)

    # Pass groups for opposite fields (similar to first, or second field)
    diff1tmp = np.abs(cdiff(df_allcom['pass_angle'], df_allcom['precess_angle1']))
    diff2tmp = np.abs(cdiff(df_allcom['pass_angle'], df_allcom['precess_angle2']))
    df_allcom.loc[oppmask, 'passgroup'] = 'sim_to_none'
    df_allcom.loc[(diff1tmp < diff2tmp) & (diff1tmp < (np.pi/2)) & (df_allcom['direction'] == 'A->B') & oppmask,
                  'passgroup'] = 'sim_to_1'
    df_allcom.loc[(diff1tmp >= diff2tmp) & (diff2tmp < (np.pi/2)) & (df_allcom['direction'] == 'B->A') & oppmask,
                  'passgroup'] = 'sim_to_1'
    df_allcom.loc[(diff1tmp < diff2tmp) & (diff1tmp < (np.pi/2)) & (df_allcom['direction'] == 'B->A') & oppmask,
                  'passgroup'] = 'sim_to_2'
    df_allcom.loc[(diff1tmp >= diff2tmp) & (diff2tmp < (np.pi/2)) & (df_allcom['direction'] == 'A->B') & oppmask,
                  'passgroup'] = 'sim_to_2'

    hist_maxh, hist_maxw = np.pi, np.pi/2


    # cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",["darkred", 'magenta', "darkblue"])
    cmap = cm.jet
    norm = SqueezedNorm(vmin=vmin, vmax=vmax, mid=0, s1=2.5, s2=2.5)
    ex_c, in_c, both_c = 'darkred', 'darkblue', 'k'
    marker = 'o'
    alpha = 0.5

    df_simprecess = df_allcom[simmask & df_allcom['precess_exist']].reset_index(drop=True)
    df_oppprecess = df_allcom[oppmask & df_allcom['precess_exist']].reset_index(drop=True)

    exin_list = []
    lagdict = {}
    for caid, ca in enumerate(['CA1', 'CA2', 'CA3']):
        df_sim_simpass = df_simprecess[(df_simprecess['ca']==ca) & (df_simprecess['passgroup'] == 'sim')].reset_index(drop=True)
        df_sim_opppass = df_simprecess[(df_simprecess['ca']==ca) & (df_simprecess['passgroup'] == 'opp')].reset_index(drop=True)
        df_opp_simto1 = df_oppprecess[(df_oppprecess['ca']==ca) & (df_oppprecess['passgroup'] == 'sim_to_1')].reset_index(drop=True)
        df_opp_simto2 = df_oppprecess[(df_oppprecess['ca']==ca) & (df_oppprecess['passgroup'] == 'sim_to_2')].reset_index(drop=True)

        # Case 1, Similar fields, PairlagAB, similar passes
        data = df_sim_simpass[['offset', 'pairlagAB', 'extrinsic', 'exin_bias', 'pairuid', 'overlap']].to_numpy().T.astype(float)
        offset, pairlagAB, extrinsic, exin_bias, pairuid, overlap = tuple(data)
        order = np.argsort(np.abs(exin_bias))
        ax_all[caid, 0].scatter(x=offset[order], y=pairlagAB[order], alpha=alpha, s=ms, c=exin_bias[order], cmap=cmap, marker=marker, norm=norm)
        ax_all[caid, 0].scatter(x=offset[order], y=pairlagAB[order] + 2 * np.pi, alpha=alpha, s=ms, c=exin_bias[order], cmap=cmap, marker=marker, norm=norm)
        set_xyticks(ax=ax_all[caid, 0], fontsize=fontsize)
        lagdict[ca + 'case1'] = pairlagAB
        plot_marginal(ax_all[caid, 0], offset, pairlagAB, extrinsic, hist_maxh, hist_maxw, ex_c=ex_c, in_c=in_c, both_c=both_c, only_both=True)
        exnum, innum = df_sim_simpass.loc[extrinsic==1, 'pairuid'].nunique(), df_sim_simpass.loc[extrinsic==0, 'pairuid'].nunique()
        biasdf = pd.DataFrame(dict(id=pairuid, bias=exin_bias, overlap=overlap))
        exin_list.append([ca, 1, exnum, innum, biasdf])
        ax_all[caid, 0].text(-hist_maxw, -np.pi-hist_maxh*0.475, '%d'%(exnum), fontsize=fontsize+2, color=ex_c)
        ax_all[caid, 0].text(-hist_maxw, -np.pi-hist_maxh*0.95, '%d'%(innum), fontsize=fontsize+2, color=in_c)

        # Case 2, Similar fields, PairlagBA, opposite
        data = df_sim_opppass[['offset', 'pairlagBA', 'extrinsic', 'exin_bias', 'pairuid', 'overlap']].to_numpy().T.astype(float)
        offset, pairlagBA, extrinsic, exin_bias, pairuid, overlap = tuple(data)
        pairlagBA = -pairlagBA
        order = np.argsort(np.abs(exin_bias))
        ax_all[caid, 1].scatter(x=offset[order], y=pairlagBA[order], alpha=alpha, s=ms, c=exin_bias[order], cmap=cmap, marker=marker, norm=norm)
        ax_all[caid, 1].scatter(x=offset[order], y=pairlagBA[order] + 2 * np.pi, alpha=alpha, s=ms, c=exin_bias[order], cmap=cmap, marker=marker, norm=norm)
        set_xyticks(ax=ax_all[caid, 1], fontsize=fontsize)
        lagdict[ca + 'case2'] = pairlagBA
        plot_marginal(ax_all[caid, 1], offset, pairlagBA, extrinsic, hist_maxh, hist_maxw, ex_c=ex_c, in_c=in_c, both_c=both_c, only_both=True)
        exnum, innum = df_sim_opppass.loc[extrinsic==1, 'pairuid'].nunique(), df_sim_opppass.loc[extrinsic==0, 'pairuid'].nunique()
        biasdf = pd.DataFrame(dict(id=pairuid, bias=exin_bias, overlap=overlap))
        exin_list.append([ca, 2, exnum, innum, biasdf])
        ax_all[caid, 1].text(-hist_maxw, -np.pi-hist_maxh*0.475, '%d'%(exnum), fontsize=fontsize+2, color=ex_c)
        ax_all[caid, 1].text(-hist_maxw, -np.pi-hist_maxh*0.95, '%d'%(innum), fontsize=fontsize+2, color=in_c)

        # Case 3, Opposite fields, PairlagAB and -PairlagBA, similar to 1st field
        data = df_opp_simto1[['offset', 'pairlagAB' ,'pairlagBA', 'extrinsic', 'exin_bias', 'pairuid', 'overlap']].to_numpy().T.astype(float)
        offset, pairlagAB, pairlagBA, extrinsic, exin_bias, pairuid, overlap = tuple(data)
        pairlagBA = -pairlagBA
        offset_c, pairlag_c = np.concatenate([offset, offset]), np.concatenate([pairlagAB, pairlagBA])
        exin_bias_c, extrinsic_c = np.concatenate([exin_bias, exin_bias]), np.concatenate([extrinsic, extrinsic])
        np.random.seed(1)
        ranvec = np.random.choice(offset.shape[0]*2, offset.shape[0], replace=False)
        offset_r, pairlag_r, extrinsic_r, exin_bias_r = offset_c[ranvec], pairlag_c[ranvec], extrinsic_c[ranvec], exin_bias_c[ranvec]
        order = np.argsort(np.abs(exin_bias_r))
        ax_all[caid, 2].scatter(x=offset_r[order], y=pairlag_r[order], alpha=alpha, s=ms, c=exin_bias_r[order], cmap=cmap, marker=marker, norm=norm)
        ax_all[caid, 2].scatter(x=offset_r[order], y=pairlag_r[order] + 2 * np.pi, alpha=alpha, s=ms, c=exin_bias_r[order], cmap=cmap, marker=marker, norm=norm)
        set_xyticks(ax=ax_all[caid, 2], fontsize=fontsize)
        lagdict[ca + 'case3'] = pairlag_r
        plot_marginal(ax_all[caid, 2], offset_r, pairlag_r, extrinsic_r, hist_maxh, hist_maxw, ex_c=ex_c, in_c=in_c, both_c=both_c, only_both=True)
        exnum, innum = df_opp_simto1.loc[extrinsic==1, 'pairuid'].nunique(), df_opp_simto1.loc[extrinsic==0, 'pairuid'].nunique()
        biasdf = pd.DataFrame(dict(id=pairuid, bias=exin_bias, overlap=overlap))
        exin_list.append([ca, 3, exnum, innum, biasdf])
        ax_all[caid, 2].text(-hist_maxw, -np.pi-hist_maxh*0.475, '%d'%(exnum), fontsize=fontsize+2, color=ex_c)
        ax_all[caid, 2].text(-hist_maxw, -np.pi-hist_maxh*0.95, '%d'%(innum), fontsize=fontsize+2, color=in_c)

        # Case 4, Opposite fields, PairlagAB and -PairlagBA, similar to 2nd field
        data = df_opp_simto2[['offset', 'pairlagAB' ,'pairlagBA', 'extrinsic', 'exin_bias', 'pairuid', 'overlap']].to_numpy().T.astype(float)
        offset, pairlagAB, pairlagBA, extrinsic, exin_bias, pairuid, overlap = tuple(data)
        pairlagBA = -pairlagBA
        offset_c, pairlag_c = np.concatenate([offset, offset]), np.concatenate([pairlagAB, pairlagBA])
        exin_bias_c, extrinsic_c = np.concatenate([exin_bias, exin_bias]), np.concatenate([extrinsic, extrinsic])
        np.random.seed(2)
        ranvec = np.random.choice(offset.shape[0]*2, offset.shape[0], replace=False)
        offset_r, pairlag_r, extrinsic_r, exin_bias_r = offset_c[ranvec], pairlag_c[ranvec], extrinsic_c[ranvec], exin_bias_c[ranvec]
        order = np.argsort(np.abs(exin_bias_r))
        ax_all[caid, 3].scatter(x=offset_r[order], y=pairlag_r[order], alpha=alpha, s=ms, c=exin_bias_r[order], cmap=cmap, marker=marker, norm=norm)
        ax_all[caid, 3].scatter(x=offset_r[order], y=pairlag_r[order] + 2 * np.pi, alpha=alpha, s=ms, c=exin_bias_r[order], cmap=cmap, marker=marker, norm=norm)
        set_xyticks(ax=ax_all[caid, 3], fontsize=fontsize)
        lagdict[ca + 'case4'] = pairlag_r
        plot_marginal(ax_all[caid, 3], offset_r, pairlag_r, extrinsic_r, hist_maxh, hist_maxw, ex_c=ex_c, in_c=in_c, both_c=both_c, only_both=True)
        exnum, innum = df_opp_simto2.loc[extrinsic==1, 'pairuid'].nunique(), df_opp_simto2.loc[extrinsic==0, 'pairuid'].nunique()
        biasdf = pd.DataFrame(dict(id=pairuid, bias=exin_bias, overlap=overlap))
        exin_list.append([ca, 4, exnum, innum, biasdf])
        ax_all[caid, 3].text(-hist_maxw, -np.pi-hist_maxh*0.475, '%d'%(exnum), fontsize=fontsize+2, color=ex_c)
        ax_all[caid, 3].text(-hist_maxw, -np.pi-hist_maxh*0.95, '%d'%(innum), fontsize=fontsize+2, color=in_c)

    # x labels
    for ax in ax_all[2, :]:
        ax.set_xlabel('Onset phase (rad)', fontsize=fontsize)

    # y labels
    ax_all[0, 0].set_ylabel('CA1\nCorrelation lag (rad)', fontsize=fontsize)
    ax_all[1, 0].set_ylabel('CA2\nCorrelation lag (rad)', fontsize=fontsize)
    ax_all[2, 0].set_ylabel('CA3\nCorrelation lag (rad)', fontsize=fontsize)

    # x, y lim
    for ax in ax_all.ravel():
        ax.set_xlim(0 - hist_maxw, 2 * np.pi)
        ax.set_ylim(-np.pi - hist_maxh, 3 * np.pi)

    # # Test chi-square of ex-in proportion
    exindf = pd.DataFrame(exin_list, columns=['ca', 'case', 'exnum', 'innum', 'biasdf']).squeeze()
    exindf['exfrac'] = exindf['exnum']/(exindf['exnum']+exindf['innum'])
    getexindf2ca = lambda x, ca1, ca2, case: x.loc[((x['ca']==ca1) | (x['ca']==ca2)) & (x['case']==case), ['exnum', 'innum']]
    getexindf2case = lambda x, ca, case1, case2: x.loc[((x['case']==case1) | (x['case']==case2)) & (x['ca']==ca), ['exnum', 'innum']]


    # Between ca's per case
    for caseid in range(1, 5):
        df_ca12 = getexindf2ca(exindf, 'CA1', 'CA2', caseid)
        df_ca23 = getexindf2ca(exindf, 'CA2', 'CA3', caseid)
        df_ca13 = getexindf2ca(exindf, 'CA1', 'CA3', caseid)
        pfish12 = fisherexact(df_ca12.to_numpy())
        pfish23 = fisherexact(df_ca23.to_numpy())
        pfish13 = fisherexact(df_ca13.to_numpy())
        stat_record(stat_fn, False, 'Case %d CA1 vs CA2, Fisher exact test p=%0.4f\n%s\n' % (caseid, pfish12, df_ca12.to_string(index=False)))
        stat_record(stat_fn, False, 'Case %d CA2 vs CA3, Fisher exact test p=%0.4f\n%s\n' % (caseid, pfish23, df_ca23.to_string(index=False)))
        stat_record(stat_fn, False, 'Case %d CA1 vs CA3, Fisher exact test p=%0.4f\n%s\n' % (caseid, pfish13, df_ca13.to_string(index=False)))

    # Between cases per ca
    for caid in range(1, 4):
        ca = 'CA%d'%(caid)
        df_case12 = getexindf2case(exindf, ca, 1, 2)
        df_case34 = getexindf2case(exindf, ca, 3, 4)
        pfish12 = fisherexact(df_case12.to_numpy())
        pfish34 = fisherexact(df_case34.to_numpy())
        stat_record(stat_fn, False, '%s Case 1 vs 2, Fisher exact test p=%0.4f\n%s\n' % (ca, pfish12, df_case12.to_string(index=False)))
        stat_record(stat_fn, False, '%s Case 3 vs 4, Fisher exact test p=%0.4f\n%s\n' % (ca, pfish34, df_case34.to_string(index=False)))



    # # Test marginal offsets and rates
    sim_d = {k1+k2:gpdf['offset'] for (k1, k2), gpdf in df_simprecess.groupby(['ca', 'passgroup'])}
    opp_d = {k1+k2:gpdf['offset'] for (k1, k2), gpdf in df_oppprecess.groupby(['ca', 'passgroup'])}
    sim_stat = {k1+k2:ci_vonmise(gpdf['offset'], ci=0.95) for (k1, k2), gpdf in df_simprecess.groupby(['ca', 'passgroup'])}
    opp_stat = {k1+k2:ci_vonmise(gpdf['offset'], ci=0.95) for (k1, k2), gpdf in df_oppprecess.groupby(['ca', 'passgroup'])}

    df_simall = df_allcom[simmask].reset_index(drop=True)
    df_oppall = df_allcom[oppmask].reset_index(drop=True)
    sim_rates_all = {k1+k2:gpdf['rate'] for (k1, k2), gpdf in df_simall.groupby(['ca', 'passgroup'])}
    opp_rates_all = {k1+k2:gpdf['rate'] for (k1, k2), gpdf in df_oppall.groupby(['ca', 'passgroup'])}

    ca_x = np.array([0, 0.5, 1])
    bar_space = 0.1
    capsize = 5
    whisker = lambda x: np.quantile(x, 0.75) + 1.5 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))

    # Case 1
    x = ca_x - (bar_space/2) - bar_space
    ax_maronset.errorbar(x=x, y=[sim_stat['CA%dsim'%(i+1)][0] for i in range(3)],
                         yerr=[(sim_stat['CA%dsim'%(i+1)][1][1]-sim_stat['CA%dsim'%(i+1)][1][0])/2 for i in range(3)],
                         fmt='o', capsize=capsize, label='Case 1')
    ax_nspikes.boxplot(x=[sim_rates_all['CA%dsim'%(i+1)] for i in range(3)], vert=True, widths=bar_space*0.8,
                       positions=x, showfliers=False)


    # Case 2
    x = ca_x - (bar_space/2)
    ax_maronset.errorbar(x=x, y=[sim_stat['CA%dopp'%(i+1)][0]for i in range(3)],
                         yerr=[(sim_stat['CA%dopp'%(i+1)][1][1]-sim_stat['CA%dopp'%(i+1)][1][0])/2 for i in range(3)],
                         fmt='o', capsize=capsize, label='Case 2')
    ax_nspikes.boxplot(x=[sim_rates_all['CA%dopp'%(i+1)] for i in range(3)], vert=True, widths=bar_space*0.8,
                       positions=x, showfliers=False)

    # Test between Case 1 and 2
    for i in range(3):
        # Test marginal onset
        pval12, table12 = watson_williams(sim_d['CA%dsim'%(i+1)], sim_d['CA%dopp'%(i+1)])
        ax_maronset.text(ca_x[i]-bar_space, 4, 'p'+p2str(pval12), fontsize=legendsize)
        ax_maronset.errorbar(x=ca_x[i]-bar_space, y=3.9, xerr=bar_space/2, c='k', capsize=2.5)
        F, dof_col = table12.loc['Columns', ['F', 'df']]
        dof_re = table12.loc['Residual', 'df']
        stat_record(stat_fn, False, 'CA%d, Marginal Onset: Case 1 vs 2, Watson-Williams test, F_{(%d, %d)}=%0.2f, p=%0.4f'% \
                    (i+1, dof_re, dof_col, F, pval12))

        # Test number of spikes per pass
        a, b = sim_rates_all['CA%dsim'%(i+1)], sim_rates_all['CA%dopp'%(i+1)]
        max_y = max(whisker(a), whisker(b))
        z12, pval12 = ranksums(a, b)
        ax_nspikes.text(ca_x[i]-0.15, max_y+20, 'p'+p2str(pval12), fontsize=legendsize)
        ax_nspikes.errorbar(x=ca_x[i]-bar_space, y=max_y+10, xerr=bar_space/2, c='k', capsize=2.5)
        stat_record(stat_fn, False, 'CA%d, Number of spikes per pass: Case 1 vs 2, Median = %d vs %d, Mann-Whiney U test, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f'% \
                    (i+1, np.median(a), np.median(b), z12, a.shape[0], b.shape[0], pval12))



    # Case 3
    x = ca_x + (bar_space/2)
    ax_maronset.errorbar(x=x, y=[opp_stat['CA%dsim_to_1'%(i+1)][0]for i in range(3)],
                         yerr=[(opp_stat['CA%dsim_to_1'%(i+1)][1][1]-opp_stat['CA%dsim_to_1'%(i+1)][1][0])/2 for i in range(3)],
                         fmt='o', capsize=capsize, label='Case 3')
    ax_nspikes.boxplot(x=[opp_rates_all['CA%dsim_to_1'%(i+1)] for i in range(3)], vert=True, widths=bar_space*0.8,
                       positions=x, showfliers=False)
    # Case 4
    x = ca_x + (bar_space/2) + bar_space
    ax_maronset.errorbar(x=x, y=[opp_stat['CA%dsim_to_2'%(i+1)][0]for i in range(3)],
                         yerr=[(opp_stat['CA%dsim_to_2'%(i+1)][1][1]-opp_stat['CA%dsim_to_2'%(i+1)][1][0])/2 for i in range(3)],
                         fmt='o', capsize=capsize, label='Case 4')
    ax_nspikes.boxplot(x=[opp_rates_all['CA%dsim_to_2'%(i+1)] for i in range(3)], vert=True, widths=bar_space*0.8,
                       positions=x, showfliers=False)

    # Test between Case 3 and 4
    for i in range(3):

        # Test marginal onset
        pval34, table34 = watson_williams(opp_d['CA%dsim_to_1'%(i+1)], opp_d['CA%dsim_to_2'%(i+1)])
        # _, pval34 = ranksums(opp_d['CA%dsim_to_1'%(i+1)], opp_d['CA%dsim_to_2'%(i+1)])
        ax_maronset.text(ca_x[i]+bar_space, 4, 'p'+p2str(pval34), fontsize=legendsize)
        ax_maronset.errorbar(x=ca_x[i]+bar_space, y=3.9, xerr=bar_space/2, c='k', capsize=2.5)
        F, dof_col = table34.loc['Columns', ['F', 'df']]
        dof_re = table34.loc['Residual', 'df']
        stat_record(stat_fn, False, 'CA%d, Marginal Onset: Case 3 vs 4, Watson-Williams test, F_{(%d, %d)}=%0.2f, p=%0.4f'% \
                    (i+1, dof_re, dof_col, F, pval34))

        # Test number of spikes per pass
        a, b = opp_rates_all['CA%dsim_to_1'%(i+1)], opp_rates_all['CA%dsim_to_2'%(i+1)]
        max_y = max(whisker(a), whisker(b))
        z34, pval34 = ranksums(a, b)
        ax_nspikes.text(ca_x[i]+0.05, max_y+20, 'p'+p2str(pval34), fontsize=legendsize)
        ax_nspikes.errorbar(x=ca_x[i]+bar_space, y=max_y+10, xerr=bar_space/2, c='k', capsize=2.5)
        stat_record(stat_fn, False, 'CA%d, Number of spikes per pass: Case 3 vs 4, Median = %d vs %d, Mann-Whitney U test, z=%0.2f, n_1=%d, n_2=%d, p=%0.4f'% \
                    (i+1, np.median(a), np.median(b), z34, a.shape[0], b.shape[0], pval34))

    # Test between all 4 cases
    for i in range(3):
        pval1234, table1234 = watson_williams(sim_d['CA%dsim'%(i+1)], sim_d['CA%dopp'%(i+1)], opp_d['CA%dsim_to_1'%(i+1)], opp_d['CA%dsim_to_2'%(i+1)])
        F, dof_col = table1234.loc['Columns', ['F', 'df']]
        dof_re = table1234.loc['Residual', 'df']
        stat_record(stat_fn, False, 'CA%d, Marginal Onset: All Cases, Watson-Williams test, F_{(%d, %d)}=%0.2f, p=%0.4f'% \
                    (i+1, dof_re, dof_col, F, pval1234))

    ax_maronset.legend(fontsize=fontsize, handlelength=1, labelspacing=0.2, handletextpad=0.2,
                       borderpad=0.1, loc='lower right')
    ax_maronset.set_xlim(-0.25, 1.4)
    ax_maronset.set_ylim(2.5, 4.2)
    ax_maronset.set_ylabel('Onset phase (rad)', fontsize=fontsize)
    ax_maronset.set_xticks(ca_x)
    ax_maronset.set_xticklabels(['CA1', 'CA2', 'CA3'])
    ax_maronset.tick_params(labelsize=fontsize)
    ax_nspikes.legend(fontsize=fontsize, handlelength=1, labelspacing=0.2, handletextpad=0.2,
                       borderpad=0.1, loc='lower right')
    ax_nspikes.set_xlim(-0.25, 1.25)
    ax_nspikes.set_ylim(-10, 165)
    casextmp = np.array([-(bar_space/2) - bar_space, -(bar_space/2), (bar_space/2),  (bar_space/2) + bar_space])
    casex = np.concatenate([ca_x, casextmp + ca_x[0], casextmp + ca_x[1], casextmp + ca_x[2]])
    casexlabels = ['\nCA1 case', '\nCA2 case', '\nCA3 case'] + ['%d' % i for i in range(1, 5)] * 3
    ax_nspikes.set_xticks(casex)
    ax_nspikes.set_xticklabels(casexlabels)
    ax_nspikes.set_ylabel('Spike count\nper pass')
    for i, t in enumerate(ax_nspikes.xaxis.get_ticklines()):
        if i < 5:
            t.set_visible(False)
    ax_nspikes.tick_params(labelsize=fontsize)


    # # Pairlag & overlap
    getbiasdf = lambda x, ca, case: x.loc[(x['ca']==ca) & (x['case']==case), 'biasdf'].iloc[0]
    for caid in range(1, 4):
        ca = 'CA%d'%caid
        unique_overlaps = [getbiasdf(exindf, ca, i).drop_duplicates(subset=['id'])['overlap'] for i in range(1, 5)]

        # case 1 vs 2
        pval12, table12 = watson_williams(lagdict[ca + 'case1'], lagdict[ca + 'case2'])
        F, dof_col = table12.loc['Columns', ['F', 'df']]
        dof_re = table12.loc['Residual', 'df']
        stat_record(stat_fn, False, '%s, Marginal Pairlag: Case 1 vs 2, Watson-Williams test, F_{(%d, %d)}=%0.2f, p=%0.4f'% \
                    (ca, dof_re, dof_col, F, pval12))

        rs_z12, rs_p12 = ranksums(unique_overlaps[0], unique_overlaps[1])
        overlap1m, overlap2m = np.median(unique_overlaps[0]), np.median(unique_overlaps[1])
        print('Median overlap %s case 1 vs 2 = %0.2f vs %0.2f (p=%0.4f)'%(ca, overlap1m, overlap2m, rs_p12))

        # case 3 vs 4
        pval34, table34 = watson_williams(lagdict[ca + 'case3'], lagdict[ca + 'case4'])
        F, dof_col = table34.loc['Columns', ['F', 'df']]
        dof_re = table34.loc['Residual', 'df']
        stat_record(stat_fn, False, '%s, Marginal Pairlag: Case 3 vs 4, Watson-Williams test, F_{(%d, %d)}=%0.2f, p=%0.4f'% \
                    (ca, dof_re, dof_col, F, pval34))
        rs_z34, rs_p34 = ranksums(unique_overlaps[2], unique_overlaps[3])
        overlap3m, overlap4m = np.median(unique_overlaps[2]), np.median(unique_overlaps[3])
        print('Median overlap %s case 3 vs 4 = %0.2f vs %0.2f (p=%0.4f)'%(ca, overlap3m, overlap4m, rs_p34))


        # Case 12 vs 34
        lag12 = np.concatenate([lagdict[ca + 'case1'], lagdict[ca + 'case2']])
        lag34 = np.concatenate([lagdict[ca + 'case3'], lagdict[ca + 'case4']])
        pval1234, table1234 = watson_williams(lag12, lag34)
        F, dof_col = table1234.loc['Columns', ['F', 'df']]
        dof_re = table1234.loc['Residual', 'df']
        stat_record(stat_fn, False, '%s, Marginal Pairlag: Case 1+3 vs 3+4, Watson-Williams test, F_{(%d, %d)}=%0.2f, p=%0.4f'% \
                    (ca, dof_re, dof_col, F, pval1234))

        overlap12cat = np.concatenate([unique_overlaps[0], unique_overlaps[1]])
        overlap34cat = np.concatenate([unique_overlaps[2], unique_overlaps[3]])
        rs_z1234, rs_p1234 = ranksums(overlap12cat, overlap34cat)
        overlap12m, overlap34m = np.median(overlap12cat), np.median(overlap34cat)
        print('Median overlap %s case 12 vs 34 = %0.2f vs %0.2f (p=%0.4f)'%(ca, overlap12m, overlap34m, rs_p1234))


    # # Draw case illustrations
    fig_vis, ax_vis = plt.subplots(1, 4, figsize=(1.75*4, 1.75), sharex=True, sharey=True)
    t = np.linspace(0, 2*np.pi, 100)
    r, arrowl = 0.5, 0.25
    arrow_upshift = r+0.1
    text_upshift = r-0.3
    passtext_downshift = 0.3
    passarrowsize = 64
    pass_color = 'mediumblue'
    c1x, c1y, c2x, c2y = -0.4, 0, 0.4, 0
    circ1_x, circ1_y = r * np.sin(t) + c1x, r * np.cos(t) + c1y
    circ2_x, circ2_y = r * np.sin(t) + c2x, r * np.cos(t) + c2y

    for i in range(4):
        ax_vis[i].plot(circ1_x, circ1_y, c='k', alpha=0.7)
        ax_vis[i].plot(circ2_x, circ2_y, c='k', alpha=0.7)
        ax_vis[i].text((c1x+c2x)/2-0.3, c1y+arrow_upshift+0.3, 'Case %d'%(i+1), fontsize=fontsize+2)
        ax_vis[i].axis('off')
        ax_vis[i].set_xlim(-1.1, 1.1)
        ax_vis[i].set_ylim(-1.1, 1.1)

    ax_vis[0].arrow(c1x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[0].arrow(c2x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[0].text(c1x-0.1, c1y+text_upshift, 'A', fontsize=fontsize+4)
    ax_vis[0].text(c2x-0.1, c2y+text_upshift, 'B', fontsize=fontsize+4)
    ax_vis[0].plot([c1x-r-0.1, c2x+r+0.1], [c1y, c1y], c=pass_color)
    ax_vis[0].scatter(c2x, c2y, marker='>', color=pass_color, s=passarrowsize)
    ax_vis[0].text(c2x-r/2, c2y-passtext_downshift, 'Pass', fontsize=fontsize+3, color=pass_color)

    ax_vis[1].arrow(c1x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[1].arrow(c2x, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[1].text(c1x-0.1, c1y+text_upshift, 'A', fontsize=fontsize+4)
    ax_vis[1].text(c2x-0.1, c2y+text_upshift, 'B', fontsize=fontsize+4)
    ax_vis[1].plot([c1x-r-0.1, c2x+r+0.1], [c1y, c1y], c=pass_color)
    ax_vis[1].scatter(c1x, c1y, marker='<', color=pass_color, s=passarrowsize)
    ax_vis[1].text(c1x-r/2, c1y-passtext_downshift, 'Pass', fontsize=fontsize+3, color=pass_color)

    ax_vis[2].arrow(c1x-arrowl, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[2].arrow(c2x+arrowl, c1y+arrow_upshift, dx=-arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[2].text(c1x-0.1, c1y+text_upshift, 'A', fontsize=fontsize+4)
    ax_vis[2].text(c2x-0.1, c2y+text_upshift, 'B', fontsize=fontsize+4)
    ax_vis[2].plot([c1x-r-0.1, c2x+r+0.1], [c1y, c1y], c=pass_color)
    ax_vis[2].scatter(c2x, c2y, marker='>', color=pass_color, s=passarrowsize)
    ax_vis[2].text(c2x-r/2, c2y-passtext_downshift, 'Pass', fontsize=fontsize+3, color=pass_color)

    ax_vis[3].arrow(c1x+arrowl, c1y+arrow_upshift, dx=-arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[3].arrow(c2x-arrowl, c1y+arrow_upshift, dx=arrowl, dy=0, width=0.025, head_width=0.15, color='k')
    ax_vis[3].text(c1x-0.1, c1y+text_upshift, 'A', fontsize=fontsize+4)
    ax_vis[3].text(c2x-0.1, c2y+text_upshift, 'B', fontsize=fontsize+4)
    ax_vis[3].plot([c1x-r-0.1, c2x+r+0.1], [c1y, c1y], c=pass_color)
    ax_vis[3].scatter(c2x, c2y, marker='>', color=pass_color, s=passarrowsize)
    ax_vis[3].text(c2x-r/2, c2y-passtext_downshift, 'Pass', fontsize=fontsize+3, color=pass_color)


    # # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=norm)
    fig_colorbar = plt.figure(figsize=(1.75*4, 1.75*3))
    fig_colorbar.subplots_adjust(right=0.8)
    cbar_ax = fig_colorbar.add_axes([0.85, 0.15, 0.03, 0.7])
    cb = fig_colorbar.colorbar(sm, cax=cbar_ax)
    cb.set_label('Intrinsic <-> Extrinsic', fontsize=fontsize)


    if fig_dir:
        fig_all.tight_layout()
        fig_all.savefig(os.path.join(fig_dir, '4Cases_Pairlag-Onset.png'), dpi=dpi)

        fig_maronset.tight_layout()
        fig_maronset.savefig(os.path.join(fig_dir, '4Cases_MarOnset.png'), dpi=dpi)

        fig_vis.tight_layout()
        fig_vis.savefig(os.path.join(fig_dir, '4Cases_Illustrations.png'), dpi=dpi)

        fig_colorbar.tight_layout()
        fig_colorbar.savefig(os.path.join(fig_dir, '4Cases_Colorbar.png'), dpi=dpi)


        fig_nspikes.tight_layout()
        fig_nspikes.savefig(os.path.join(fig_dir, '4Caes_nspikes.png'), dpi=dpi)


if __name__ == '__main__':
    preprocess_save_dir = 'results/exp/passes/'

    # Single Analysis
    single_plot_dir = 'result_plots/passes/single'
    singlefield_df = load_pickle('results/exp/single_field/singlefield_df.pickle')


    # plot_precession_examples(singlefield_df, single_plot_dir)
    analyzer = SingleBestAngleAnalyzer(singlefield_df, single_plot_dir)
    analyzer.plot_precession_fraction()
    # analyzer.plot_field_bestprecession(plot_example=False)
    # analyzer.plot_both_slope_offset(NShuffles=1000)


    # # # Pair Analysis
    # usephase = False
    # pairfield_df_pth = 'results/exp/pair_field/pairfield_df_latest.pickle'
    # pairfield_df = load_pickle(pairfield_df_pth)
    # pairlag_phase_correlation_similar_passes(pairfield_df, fig_dir='result_plots/passes/pair', usephase=usephase)

