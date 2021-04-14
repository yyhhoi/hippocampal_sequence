import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
import pandas as pd

from PairField_Analysis import plot_pairangle_similarity_analysis
from PassAnalysis import SingleBestAngleAnalyzer, pairlag_phase_correlation_similar_passes
from common.comput_utils import repeat_arr, midedges, normalize_distr, circular_density_1d, get_numpass_at_angle, \
    shiftcyc_full2half
from pycircstat.descriptive import mean as circmean, cdiff
from pycircstat.descriptive import resultant_vector_length

from common.script_wrappers import compute_precessangle, PrecessionFilter



def tmp_compute_new_precess_info(precess_df, kappa, posthoc_aedges):

    if (precess_df.shape[0] > 0) and (precess_df['precess_exist'].sum() > 0):
        # Compute statistics for precession (bestangle, R, norm_distr)
        pass_angles = precess_df['spike_angle'].to_numpy()
        pass_nspikes = precess_df['pass_nspikes'].to_numpy()
        precess_mask = precess_df['precess_exist'].to_numpy()

        bestangle, R, densities = compute_precessangle(pass_angles, pass_nspikes, precess_mask,
                                                       kappa=kappa, bins=None)

        # Post-hoc precession exclusion
        _, _, postdoc_dens = compute_precessangle(pass_angles, pass_nspikes, precess_mask,
                                                  kappa=None, bins=posthoc_aedges)

        (_, passbins_p, passbins_np, _) = postdoc_dens
        all_passbins = passbins_p + passbins_np
        numpass_at_precess = get_numpass_at_angle(target_angle=bestangle, aedge=posthoc_aedges,
                                                  all_passbins=all_passbins)

        return bestangle, R, numpass_at_precess
    else:
        return None, None, None

def tmp_test_pval(single_df, plot_dir):

    aedges = np.linspace(-np.pi, np.pi, 9)

    minpassbin_list = []
    for i in range(single_df.shape[0]):
        precess_df = single_df.loc[i, 'precess_df']

        passangles = precess_df['spike_angle'].to_numpy()

        bins, _ = np.histogram(passangles, bins=aedges)
        minpassbin = np.min(bins)
        minpassbin_list.append(minpassbin)

    single_df['minpassbin'] = minpassbin_list

    ca3df = single_df[(single_df['ca']=='CA3') & (~single_df['precess_angle'].isna())]

    minthreshs = np.arange(0, 6, 1)
    frac_list = []
    nlist = []
    for mt in minthreshs:
        thisdf = ca3df.loc[ca3df['minpassbin']>=mt]
        sigfrac = np.mean(thisdf['precess_R_pval'] < 0.05)
        nlist.append(thisdf.shape[0])
        frac_list.append(sigfrac)

    showdf = pd.DataFrame({'minpassthresh':minthreshs, 'n':nlist, 'frac':frac_list})
    print(showdf)




def tmp_compare_R(single_df):
    precess_filter = PrecessionFilter()
    # Single:
    all_precess_df = []
    aedges = np.linspace(-np.pi, np.pi, 36)
    all_Rk50 = []
    all_Rk1 = []
    all_Rb36 = []
    for i in range(single_df.shape[0]):
        print('Single %d/%d recomputing precession' % (i, single_df.shape[0]))

        precess_df = single_df.loc[i, 'precess_df']
        ca, minocc = single_df.loc[i, ['ca', 'minocc']]
        precess_df = precess_filter.filter_single(precess_df)

        if (precess_df.shape[0] > 0) and (precess_df['precess_exist'].sum() > 0):
            # Compute statistics for precession (bestangle, R, norm_distr)
            pass_angles = precess_df['spike_angle'].to_numpy()
            pass_nspikes = precess_df['pass_nspikes'].to_numpy()
            precess_mask = precess_df['precess_exist'].to_numpy()

            _, Rk50, _ = compute_precessangle(pass_angles, pass_nspikes, precess_mask, kappa=16*np.pi, bins=None)
            _, Rk1, _ = compute_precessangle(pass_angles, pass_nspikes, precess_mask, kappa=1, bins=None)
            _, Rb36, _ = compute_precessangle(pass_angles, pass_nspikes, precess_mask, kappa=None, bins=aedges)

            all_Rk50.append(Rk50)
            all_Rk1.append(Rk1)
            all_Rb36.append(Rb36)

    dftmp = pd.DataFrame({'kappa50': all_Rk50, 'bins36': all_Rb36, 'kappa1':all_Rk1})
    print(dftmp['kappa50'].describe(), '\n')
    print(dftmp['kappa1'].describe(), '\n')
    print(dftmp['bins36'].describe())



def tmp_reconstruct_df(single_df, pair_df, kappa):
    precess_filter = PrecessionFilter()
    # precess_filter.occ_thresh = -1
    posthoc_aedges = np.linspace(-np.pi, np.pi, 6)
    nspikes_stats = {'CA1': 10, 'CA2': 11, 'CA3': 13}  # 25% quantile
    if single_df is not None:
        # Single:
        all_precess_df = []
        for i in range(single_df.shape[0]):
            print('Single %d/%d recomputing precession' % (i, single_df.shape[0]))

            precess_df = single_df.loc[i, 'precess_df']
            ca, minocc = single_df.loc[i, ['ca', 'minocc']]
            precess_df = precess_filter.filter_single(precess_df)

            precessangle, precessR, numpass_at_precess = tmp_compute_new_precess_info(precess_df, kappa, posthoc_aedges)

            ldf = precess_df[precess_df['pass_nspikes'] < nspikes_stats[ca]]  # 25% quantile
            if (ldf.shape[0] > 0) and (ldf['precess_exist'].sum() > 0):
                precess_angle_low, _, numpass_at_precess_low = tmp_compute_new_precess_info(ldf, kappa, posthoc_aedges)
            else:
                precess_angle_low = None
                numpass_at_precess_low = None

            all_precess_df.append(precess_df)
            single_df.loc[i, 'precess_angle'] = precessangle
            single_df.loc[i, 'precess_angle_low'] = precess_angle_low
            single_df.loc[i, 'precess_R'] = precessR
            single_df.loc[i, 'numpass_at_precess'] = numpass_at_precess
            single_df.loc[i, 'numpass_at_precess_low'] = numpass_at_precess_low
        single_df['precess_df'] = all_precess_df
        return single_df
    if pair_df is not None:
        # Pair
        # precess_df1, precess_angle1, precess_R1, precess_df2, precess_angle2, precess_R2,
        # numpass_at_precess1, numpass_at_precess2, precess_dfp
        all_precess_df1, all_precess_df2, all_precess_dfp = [], [], []
        for i in range(pair_df.shape[0]):
            print('Pair %d/%d recomputing precession' % (i, pair_df.shape[0]))
            precess_df1 = pair_df.loc[i, 'precess_df1']
            precess_df2 = pair_df.loc[i, 'precess_df2']
            precess_dfp = pair_df.loc[i, 'precess_dfp']
            ca, minocc1, minocc2 = pair_df.loc[i, ['ca', 'minocc1', 'minocc2']]

            # 1st field
            precess_df1 = precess_filter.filter_single(precess_df1)
            precessangle1, precessR1, numpass_at_precess1 = tmp_compute_new_precess_info(precess_df1, kappa,
                                                                                         posthoc_aedges)
            all_precess_df1.append(precess_df1)

            # 2nd field
            precess_df2 = precess_filter.filter_single(precess_df2)
            precessangle2, precessR2, numpass_at_precess2 = tmp_compute_new_precess_info(precess_df2, kappa,
                                                                                         posthoc_aedges)
            all_precess_df2.append(precess_df2)

            # Paired
            precess_dfp = precess_filter.filter_pair(precess_dfp)
            all_precess_dfp.append(precess_dfp)

            # Change data
            pair_df.loc[i, 'precess_angle1'] = precessangle1
            pair_df.loc[i, 'precess_R1'] = precessR1
            pair_df.loc[i, 'precess_angle2'] = precessangle2
            pair_df.loc[i, 'precess_R2'] = precessR2
            pair_df.loc[i, 'numpass_at_precess1'] = numpass_at_precess1
            pair_df.loc[i, 'numpass_at_precess2'] = numpass_at_precess2

        pair_df['precess_df1'] = all_precess_df1
        pair_df['precess_df2'] = all_precess_df2
        pair_df['precess_dfp'] = all_precess_dfp
        return pair_df


def compare_plot(ax, passax, cadf1_this, i, precess_filter):

    nap_aedges = np.linspace(-np.pi, np.pi, 6)
    nap_adm = midedges(nap_aedges)

    precessdfeg1 = cadf1_this.loc[i, 'precess_df1']
    precessdfeg2 = cadf1_this.loc[i, 'precess_df2']
    pairid, overlap_ratio, minocc1, minocc2 = cadf1_this.loc[i, ['pair_id', 'overlap_ratio', 'minocc1', 'minocc2']]

    precessdfeg1 = precess_filter.filter_single(precessdfeg1, minocc1)
    precessdfeg2 = precess_filter.filter_single(precessdfeg2, minocc2)
    pass_angles1 = precessdfeg1['spike_angle'].to_numpy()
    pass_nspikes1 = precessdfeg1['pass_nspikes'].to_numpy()
    precess_mask1 = precessdfeg1['precess_exist'].to_numpy()
    pass_angles2 = precessdfeg2['spike_angle'].to_numpy()
    pass_nspikes2 = precessdfeg2['pass_nspikes'].to_numpy()
    precess_mask2 = precessdfeg2['precess_exist'].to_numpy()

    bestangle1_k1, R1_k1, den1_k1 = compute_precessangle(pass_angles1, pass_nspikes1, precess_mask1,
                                                         kappa=1, bins=None)
    bestangle2_k1, R2_k1, den2_k1 = compute_precessangle(pass_angles2, pass_nspikes2, precess_mask2,
                                                         kappa=1, bins=None)
    bestangle1_k50, R1_k50, den1_k50 = compute_precessangle(pass_angles1, pass_nspikes1, precess_mask1,
                                                            kappa=16 * np.pi, bins=None)
    bestangle2_k50, R2_k50, den2_k50 = compute_precessangle(pass_angles2, pass_nspikes2, precess_mask2,
                                                            kappa=16 * np.pi, bins=None)

    _, _, den1_bins = compute_precessangle(pass_angles1, pass_nspikes1, precess_mask1,
                                           kappa=None, bins=nap_aedges)
    _, _, den2_bins = compute_precessangle(pass_angles2, pass_nspikes2, precess_mask2,
                                           kappa=None, bins=nap_aedges)

    p1, = ax[0, 0].plot(passax, den1_k50[1], c='r', label='pass')
    p2, = ax[0, 0].plot(passax, den1_k50[3], c='b', label='spike')
    p3, = ax[0, 0].plot(passax, den1_k50[0], c='k', label='ratio')
    ymax = max(den1_k50[0].max(), den1_k50[1].max(), den1_k50[3].max())
    ax[0, 0].vlines(shiftcyc_full2half(bestangle1_k50), ymin=0, ymax=ymax, colors='k')
    ax[0, 0].set_title('kappa=50\n|d|=%0.2fpi, (Ex-In)=%0.2f'%(np.abs(cdiff(bestangle1_k50, bestangle2_k50))/np.pi, overlap_ratio))
    ax[0, 0].set_ylabel('Field 1')
    ax_1k50 = ax[0, 0].twinx()
    p4, = ax_1k50.step(nap_adm, den1_bins[1] + den1_bins[2], c='orange', label='allpass bins', where='mid')

    ax[1, 0].plot(passax, den2_k50[1], c='r')
    ax[1, 0].plot(passax, den2_k50[3], c='b')
    ax[1, 0].plot(passax, den2_k50[0], c='k')
    ymax = max(den2_k50[0].max(), den2_k50[1].max(), den2_k50[3].max())
    ax[1, 0].vlines(shiftcyc_full2half(bestangle2_k50), ymin=0, ymax=ymax, colors='k')
    ax[1, 0].set_ylabel('Field 2')
    ax_2k50 = ax[1, 0].twinx()
    ax_2k50.step(nap_adm, den2_bins[1] + den2_bins[2], c='orange', label='allpass bins', where='mid')

    ax[0, 1].plot(passax, den1_k1[1], c='r')
    ax[0, 1].plot(passax, den1_k1[3], c='b')
    ax[0, 1].plot(passax, den1_k1[0], c='k')
    ymax = max(den1_k1[0].max(), den1_k1[1].max(), den1_k1[3].max())
    ax[0, 1].vlines(shiftcyc_full2half(bestangle1_k1), ymin=0, ymax=ymax, colors='k')
    ax[0, 1].set_title('kappa=1\n|d|=%0.2fpi, (Ex-In)=%0.2f'%(np.abs(cdiff(bestangle1_k1, bestangle2_k1))/np.pi, overlap_ratio))
    ax_1k1 = ax[0, 1].twinx()
    ax_1k1.step(nap_adm, den1_bins[1] + den1_bins[2], c='orange', label='allpass bins', where='mid')

    ax[1, 1].plot(passax, den2_k1[1], c='r')
    ax[1, 1].plot(passax, den2_k1[3], c='b')
    ax[1, 1].plot(passax, den2_k1[0], c='k')
    ymax = max(den2_k1[0].max(), den2_k1[1].max(), den2_k1[3].max())
    ax[1, 1].vlines(shiftcyc_full2half(bestangle2_k1), ymin=0, ymax=ymax, colors='k')
    ax_2k1 = ax[1, 1].twinx()
    ax_2k1.step(nap_adm, den2_bins[1] + den2_bins[2], c='orange', label='allpass bins', where='mid')

    for ax_each in ax.ravel():
        ax_each.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax_each.set_xticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])


    lines = [p1, p2, p3, p4]
    ax[0, 0].legend(lines, [l.get_label() for l in lines])
    return ax

def compare_2df():
    df1 = pd.read_pickle('result_plots/test/nap1/pairfield_df.pickle')
    df2 = pd.read_pickle('result_plots/test/nap1-kappa1/pairfield_df.pickle')

    plot_dir = 'result_plots/test/test_nap1_nap1-kappa1'
    os.makedirs(plot_dir, exist_ok=True)
    nap_thresh = 1

    df1mask = (~df1['overlap_ratio'].isna()) & (~df1['precess_angle1'].isna()) & (~df1['precess_angle2'].isna()) & \
              (df1['numpass_at_precess1'] >= nap_thresh) & (df1['numpass_at_precess2'] >= nap_thresh) & \
              (df1['ca'] == 'CA3')
    df2mask = (~df2['overlap_ratio'].isna()) & (~df2['precess_angle1'].isna()) & (~df2['precess_angle2'].isna()) & \
              (df2['numpass_at_precess1'] >= nap_thresh) & (df2['numpass_at_precess2'] >= nap_thresh) & \
              (df2['ca'] == 'CA3')

    cadf1 = df1[df1mask & df2mask].reset_index(drop=True)
    cadf2 = df2[df1mask & df2mask].reset_index(drop=True)
    cadf1['sim'] = np.abs(cdiff(cadf1['precess_angle1'], cadf1['precess_angle2'])) < (np.pi / 2)
    cadf2['sim'] = np.abs(cdiff(cadf2['precess_angle1'], cadf2['precess_angle2'])) < (np.pi / 2)
    passax = np.linspace(-np.pi, np.pi, 100)
    precess_filter = PrecessionFilter()

    # Similar intrinsic > dissimilar intrinsic
    mask_before = cadf1['sim'] & (cadf1['overlap_ratio'] <= 0)
    mask_after = (~cadf2['sim']) & (cadf2['overlap_ratio'] <= 0)
    cadf1_this = cadf1[mask_before & mask_after].reset_index(drop=True)
    plot_dir1 = join(plot_dir, 'sim2dissim_intrinsic')
    os.makedirs(plot_dir1, exist_ok=True)
    for i in range(cadf1_this.shape[0]):
        print('sim2dissim intrinsic %d/%d'%(i, cadf1_this.shape[0]))
        pairid = cadf1_this.loc[i, 'pair_id']
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        compare_plot(ax, passax, cadf1_this, i, precess_filter)
        fig.tight_layout()
        fig.savefig(join(plot_dir1, '%d.png'%(pairid)), dpi=300)
        plt.close()

    # dissimilar extrinsic > similar extrinsic
    mask_before = (~cadf1['sim']) & (cadf1['overlap_ratio'] > 0)
    mask_after = (cadf2['sim']) & (cadf2['overlap_ratio'] > 0)
    cadf1_this = cadf1[mask_before & mask_after].reset_index(drop=True)
    plot_dir2 = join(plot_dir, 'dissim2sim_extrinsic')
    os.makedirs(plot_dir2, exist_ok=True)
    for i in range(cadf1_this.shape[0]):
        print('dissim2sim extrinsic %d/%d'%(i, cadf1_this.shape[0]))

        pairid = cadf1_this.loc[i, 'pair_id']
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        compare_plot(ax, passax, cadf1_this, i, precess_filter)
        fig.tight_layout()
        fig.savefig(join(plot_dir2, '%d.png'%(pairid)), dpi=300)
        plt.close()
    return


if __name__ == '__main__':
    # parameter & tag
    nap_thresh = 1
    kappa = 1
    tag = ''
    plot_dir = 'result_plots/test/%s' % tag
    os.makedirs(plot_dir, exist_ok=True)

    # Single field analysis
    # single_df = pd.read_pickle('results/exp/single_field/singlefield_df_test.pickle')
    # tmp_compare_R(single_df)
    # single_df = tmp_reconstruct_df(single_df=single_df, pair_df=None, kappa=kappa)
    # analyzer = SingleBestAngleAnalyzer(single_df, plot_dir, nap_thresh=nap_thresh)
    # analyzer.plot_field_bestprecession()
    # analyzer.plot_both_slope_offset(NShuffles=10)
    #
    #
    # # Pair field analysis
    # pair_df = pd.read_pickle('results/exp/pair_field/pairfield_df_test.pickle')
    # pair_df = tmp_reconstruct_df(single_df=None, pair_df=pair_df, kappa=kappa)
    # plot_pairangle_similarity_analysis(pair_df, save_dir=plot_dir, nap_thresh=nap_thresh)
    # pairlag_phase_correlation_similar_passes(pair_df, fig_dir=plot_dir, nap_thresh=nap_thresh)


    # single_df.to_pickle(join(plot_dir, 'singlefield_df.pickle'))
    # pair_df.to_pickle(join(plot_dir, 'pairfield_df.pickle'))

    # compare_2df()


    single_df = pd.read_pickle('results/exp/single_field/singlefield_df_nap1-kappa1_WithShuffle.pickle')
    tmp_test_pval(single_df, plot_dir='result_plots/test/')
    # analyzer = SingleBestAngleAnalyzer(single_df, plot_dir, nap_thresh=nap_thresh)
    # analyzer.plot_field_bestprecession()
    # analyzer.plot_both_slope_offset(NShuffles=1000)