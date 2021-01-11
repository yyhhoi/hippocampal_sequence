# This script does directionality and spike phase analysis for single fields.
# This script combines both Emily's data and simulation results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d, make_interp_spline
from pycircstat.descriptive import resultant_vector_length, cdiff
from pycircstat.descriptive import mean as circmean
import warnings
from common.linear_circular_r import rcc
from common.utils import load_pickle
from common.comput_utils import check_border, normalize_distr, TunningAnalyzer, midedges, segment_passes, window_shuffle, \
    compute_straightness, heading, pair_diff, check_border_sim, append_info_from_passes, get_field_directionality, \
    window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, DirectionerBining, DirectionerMLM, \
    window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, circular_density_1d, passes_time_shift

from common.visualization import color_wheel, directionality_polar_plot

from common.script_wrappers import DirectionalityStatsByThresh, PrecessionProcesser, \
    compute_precession_stats, PrecessionFilter, PrecessionStat, construct_passdf_sim
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw



# figext = 'png'
figext = 'eps'


def single_field_preprocess_exp(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=None):

    shudf_dict = dict(ca=[], num_spikes=[], border=[], aver_rate=[], peak_rate=[],
                      fieldangle=[], fieldR=[], fieldangle_mlm=[], fieldR_mlm=[],
                      spike_bins=[], occ_bins=[], win_pval=[], win_pval_mlm=[], shift_pval=[], shift_pval_mlm=[],
                      precess_df=[], nonprecess_df=[], precess_angle=[], precess_angle_low=[], precess_angle_high=[], precess_info=[])
    num_trials = df.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5
    precess_filter = PrecessionFilter()

    nspikes_stats = {'CA1':(10, 26), 'CA2':(11, 29), 'CA3':(13, 41)}

    for ntrial in range(num_trials):
        wave = df.loc[ntrial, 'wave']
        precesser = PrecessionProcesser(sthresh=sthresh, vthresh=vthresh, wave=wave)
        
        for ca in ['CA%d' % (i + 1) for i in range(3)]:

            # Get data
            field_df = df.loc[ntrial, ca + 'fields']
            indata = df.loc[ntrial, ca + 'indata']
            num_fields = field_df.shape[0]
            if indata.shape[0] < 1:
                continue

            tunner = TunningAnalyzer(indata, vthresh, True)
            interpolater_angle = interp1d(tunner.t, tunner.movedir)
            interpolater_x = interp1d(tunner.t, tunner.x)
            interpolater_y = interp1d(tunner.t, tunner.y)
            interpolater_v = interp1d(tunner.t, tunner.velocity)
            trange = (tunner.t.max(), tunner.t.min())
            dt = tunner.t[1] - tunner.t[0]
            precesser.set_trange(trange)

            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(wave['theta'][1:500])
            ax[1].plot(wave['phase'][1:500])
            plt.show()
            for nf in range(num_fields):
                print('\n%d/%d trial, %s, %d/%d field' % (ntrial, num_trials, ca, nf, num_fields))
                # Check for border
                mask = field_df.loc[nf, 'mask']
                border = check_border(mask)

                # Get info from passes
                passdf = field_df.loc[nf, 'passes']
                beh_info, all_tsp_list = append_info_from_passes(passdf, vthresh, sthresh, trange)
                (all_x_list, all_y_list, all_t_list, all_passangles_list) = beh_info
                if (len(all_tsp_list) < 1) or (len(all_x_list) < 1):
                    continue
                all_x = np.concatenate(all_x_list)
                all_y = np.concatenate(all_y_list)
                all_passangles = np.concatenate(all_passangles_list)
                all_tsp = np.concatenate(all_tsp_list)
                all_anglesp = interpolater_angle(all_tsp)
                xsp, ysp = interpolater_x(all_tsp), interpolater_y(all_tsp)
                pos = np.stack([all_x, all_y]).T
                possp = np.stack([xsp, ysp]).T

                # Average firing rate
                aver_rate = all_tsp.shape[0] / (all_x.shape[0] * dt)
                peak_rate = np.max(field_df.loc[nf, 'pf']['map'] * mask)

                # Field's directionality
                num_spikes = all_tsp.shape[0]
                biner = DirectionerBining(aedges, all_passangles)
                fieldangle, fieldR, (spike_bins, occ_bins, _) = biner.get_directionality(all_anglesp)
                mlmer = DirectionerMLM(pos, all_passangles, dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
                fieldangle_mlm, fieldR_mlm, norm_prob_mlm = mlmer.get_directionality(possp, all_anglesp)

                # Precession per pass
                neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                       spikeangle='spikeangle')
                pass_dict = precesser.gen_precess_dict()
                for precess_dict in precesser.get_single_precession(passdf, neuro_keys_dict):
                    pass_dict = precesser.append_pass_dict(pass_dict, precess_dict)
                precess_df = pd.DataFrame(pass_dict)
                precessdf = precess_filter.filter_single(precess_df, minocc=occ_bins.min())
                precess_stater = PrecessionStat(precessdf)
                filtered_precessdf, precess_info , precess_angle = precess_stater.compute_precession_stats()
                nonprecess_df = precessdf[~precessdf['precess_exist']].reset_index(drop=True)

                # Precession - pass with low/high spike number
                if filtered_precessdf.shape[0] > 0:
                    ldf = filtered_precessdf[filtered_precessdf['pass_nspikes'] < nspikes_stats[ca][0]]  # 25% quantile
                    passangle_key = precess_stater.passangle_key
                    if ldf.shape[0] > 0:
                        passangle_l, pass_nspikes_l = ldf[passangle_key].to_numpy(), ldf['pass_nspikes'].to_numpy()
                        precess_angle_low, _, _ = PrecessionStat.compute_R(passangle_l, pass_nspikes_l)
                    else:
                        precess_angle_low = None
                    hdf = filtered_precessdf[filtered_precessdf['pass_nspikes'] > nspikes_stats[ca][1]]  # 75% quantile
                    if hdf.shape[0] > 0:
                        passangle_h, pass_nspikes_h = hdf[passangle_key].to_numpy(), hdf['pass_nspikes'].to_numpy()
                        precess_angle_high, _, _ = PrecessionStat.compute_R(passangle_h, pass_nspikes_h)
                    else:
                        precess_angle_high = None
                else:
                    precess_angle_low, precess_angle_high = None, None


                # Precession - time shift shuffle
                if precess_info is not None:
                    all_shuffled_R = np.zeros(NShuffles)
                    shufi = 0
                    failure = 0
                    import time
                    tmpt = time.time()
                    while (shufi < NShuffles) & (failure < NShuffles):

                        # Re-construct passdf
                        shifted_tsp_boxes = passes_time_shift(all_t_list, all_tsp_list, return_concat=False, seed=shufi)

                        shifted_spikex_boxes = list()
                        shifted_spikey_boxes = list()
                        shifted_spikev_boxes = list()
                        shifted_spikeangle_boxes = list()

                        for shuffled_tsp_box in shifted_tsp_boxes:
                            shuffled_tsp_box = shuffled_tsp_box[(shuffled_tsp_box < trange[0]) & (shuffled_tsp_box > trange[1])]
                            shifted_spikex_boxes.append(interpolater_x(shuffled_tsp_box))
                            shifted_spikey_boxes.append(interpolater_y(shuffled_tsp_box))
                            shifted_spikev_boxes.append(interpolater_v(shuffled_tsp_box))
                            shifted_spikeangle_boxes.append(interpolater_angle(shuffled_tsp_box))

                        shuffled_passdf = pd.DataFrame({'x':all_x_list, 'y':all_y_list, 't':all_t_list, 'angle':all_passangles_list,
                                                        neuro_keys_dict['tsp']:shifted_tsp_boxes, neuro_keys_dict['spikex']:shifted_spikex_boxes,
                                                        neuro_keys_dict['spikey']:shifted_spikey_boxes, neuro_keys_dict['spikev']:shifted_spikev_boxes,
                                                        neuro_keys_dict['spikeangle']:shifted_spikeangle_boxes})
                        # Re-construct precessdf and compute R
                        shuffled_pass_dict = {x: [] for x in pass_dict_keys}
                        for shuffled_precess_dict in precesser.get_single_precession(shuffled_passdf, neuro_keys_dict):
                            shuffled_pass_dict = precesser.append_pass_dict(shuffled_pass_dict, shuffled_precess_dict)

                        shuffled_precessdf = pd.DataFrame(shuffled_pass_dict)
                        shuffled_precessdf = precess_filter.filter_single(shuffled_precessdf, minocc=occ_bins.min())
                        shuffled_precessdf = shuffled_precessdf[shuffled_precessdf['precess_exist']].reset_index(drop=True)
                        if shuffled_precessdf.shape[0] < 1:
                            failure += 1
                            continue
                        else:
                            pass_angles = shuffled_precessdf['spike_angle'].to_numpy()
                            pass_nspikes = shuffled_precessdf['pass_nspikes' ].to_numpy()
                            _, all_shuffled_R[shufi], _ = PrecessionStat.compute_R_hist(pass_angles, pass_nspikes)
                            shufi += 1
                    precess_info['pval'] = 1 - np.nanmean(precess_info['R'] > all_shuffled_R)
                    precess_shuffletime = time.time()-tmpt

                # Windows shuffling 
                win_pval, win_pval_mlm = window_shuffle_wrapper(all_tsp, fieldR, fieldR_mlm, NShuffles, biner,
                                                                mlmer, interpolater_x, interpolater_y,
                                                                interpolater_angle, trange)

                # Time shift shuffling
                shi_pval, shi_pval_mlm = timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, fieldR, fieldR_mlm,
                                                                       NShuffles, biner, mlmer, interpolater_x,
                                                                       interpolater_y, interpolater_angle, trange)
                # if precess_angle is not None:
                #     print('\nBest precess=%0.2f, pval=%0.5f, time=%0.2f'%(precess_angle, precess_info['pval'], precess_shuffletime))

                shudf_dict['ca'].append(ca)
                shudf_dict['num_spikes'].append(num_spikes)
                shudf_dict['border'].append(border)
                shudf_dict['aver_rate'].append(aver_rate)
                shudf_dict['peak_rate'].append(peak_rate)
                shudf_dict['fieldangle'].append(fieldangle)
                shudf_dict['fieldangle_mlm'].append(fieldangle_mlm)
                shudf_dict['fieldR'].append(fieldR)
                shudf_dict['fieldR_mlm'].append(fieldR_mlm)
                shudf_dict['spike_bins'].append(spike_bins)
                shudf_dict['occ_bins'].append(occ_bins)
                shudf_dict['win_pval'].append(win_pval)
                shudf_dict['win_pval_mlm'].append(win_pval_mlm)
                shudf_dict['shift_pval'].append(shi_pval)
                shudf_dict['shift_pval_mlm'].append(shi_pval_mlm)
                shudf_dict['precess_df'].append(filtered_precessdf)
                shudf_dict['nonprecess_df'].append(nonprecess_df)
                shudf_dict['precess_info'].append(precess_info)
                shudf_dict['precess_angle'].append(precess_angle)
                shudf_dict['precess_angle_low'].append(precess_angle_low)
                shudf_dict['precess_angle_high'].append(precess_angle_high)

    shudf_raw = pd.DataFrame(shudf_dict)

    # Assign field ids within ca
    shudf_raw['fieldid_ca'] = 0
    for ca, cadf in shudf_raw.groupby('ca'):
        fieldidca = np.arange(cadf.shape[0]) + 1
        index_ca = cadf.index
        shudf_raw.loc[index_ca, 'fieldid_ca'] = fieldidca


    if save_pth:
        shudf_raw.to_pickle(save_pth)
    return shudf_raw


def single_field_preprocess_sim(simdata, radius=2, vthresh=2, sthresh=80, NShuffles=200,
                                subsample_fraction=0.4, save_pth=None):
    """

    Parameters
    ----------
    Indata
    SpikeData
    NeuronPos
    radius
    vthresh : float
        Default = 2. Determined by the ratio between avergae speed in Emily's data and the target vthresh 5 there.
    sthresh : float
        Default = 80. Determined by the same percentile (10%) of passes excluded in Emily's data (sthresh = 3 there).
    subsample_fraction : float
        The fraction that the spikes would be subsampled.

    Returns
    -------

    """


    datadict = dict(num_spikes=[], border=[], aver_rate=[], peak_rate=[],
                      fieldangle=[], fieldR=[], fieldangle_mlm=[], fieldR_mlm=[],
                      spike_bins=[], occ_bins=[], win_pval=[], win_pval_mlm=[], shift_pval=[], shift_pval_mlm=[],
                      precess_df=[], precess_angle=[], precess_angle_low=[], precess_angle_high=[], precess_info=[])

    Indata, SpikeData, NeuronPos = simdata['Indata'], simdata['SpikeData'], simdata['NeuronPos']
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 2*np.pi/16

    tunner = TunningAnalyzer(Indata, vthresh, True)
    wave = dict(tax=Indata['t'].to_numpy(), phase=Indata['phase'].to_numpy(),
                theta=np.ones(Indata.shape[0]))


    interpolater_angle = interp1d(tunner.t, tunner.movedir)
    interpolater_x = interp1d(tunner.t, tunner.x)
    interpolater_y = interp1d(tunner.t, tunner.y)
    trange = (tunner.t.max(), tunner.t.min())
    dt = tunner.t[1] - tunner.t[0]
    precesser = PrecessionProcesser(sthresh=sthresh, vthresh=vthresh, wave=wave)
    precesser.set_trange(trange)
    precess_filter = PrecessionFilter()
    nspikes_stats = (13, 40)
    pass_nspikes = []

    num_neurons = NeuronPos.shape[0]



    for nidx in range(num_neurons):
        print('%d/%d Neuron' % (nidx, num_neurons))
        # Get spike indexes + subsample
        spdf = SpikeData[SpikeData['neuronidx'] == nidx].reset_index(drop=True)
        tidxsp = spdf['tidxsp'].to_numpy().astype(int)
        np.random.seed(nidx)
        sampled_tidxsp = np.random.choice(tidxsp.shape[0]-1, int(tidxsp.shape[0] * subsample_fraction), replace=False)
        sampled_tidxsp.sort()
        tidxsp = tidxsp[sampled_tidxsp]

        # Get tok
        neuron_pos = NeuronPos.iloc[nidx].to_numpy()
        dist = np.sqrt((neuron_pos[0] - tunner.x) ** 2 + (neuron_pos[1] - tunner.y) ** 2)
        tok = dist < radius


        # Check border
        border = check_border_sim(neuron_pos[0], neuron_pos[1], radius, (0, 2 * np.pi))

        # Get info from passes
        all_passidx = segment_passes(tok)
        passdf = construct_passdf_sim(tunner, all_passidx, tidxsp)
        beh_info, all_tsp_list = append_info_from_passes(passdf, vthresh, sthresh, trange)
        (all_x_list, all_y_list, all_t_list, all_passangles_list) = beh_info
        if (len(all_tsp_list) < 1) or (len(all_x_list) < 1):
            continue
        all_x = np.concatenate(all_x_list)
        all_y = np.concatenate(all_y_list)
        all_passangles = np.concatenate(all_passangles_list)
        all_tsp = np.concatenate(all_tsp_list)
        all_anglesp = interpolater_angle(all_tsp)
        xsp, ysp = interpolater_x(all_tsp), interpolater_y(all_tsp)
        pos = np.stack([all_x, all_y]).T
        possp = np.stack([xsp, ysp]).T

        # Average firing rate
        aver_rate = all_tsp.shape[0] / (all_x.shape[0] * dt)

        # Field's directionality
        num_spikes = all_tsp.shape[0]
        biner = DirectionerBining(aedges, all_passangles)
        fieldangle, fieldR, (spike_bins, occ_bins, _) = biner.get_directionality(all_anglesp)
        mlmer = DirectionerMLM(pos, all_passangles, dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
        fieldangle_mlm, fieldR_mlm, norm_prob_mlm = mlmer.get_directionality(possp, all_anglesp)

        # Precession per pass
        neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                               spikeangle='spikeangle')
        pass_dict = precesser.gen_precess_dict()
        for precess_dict in precesser.get_single_precession(passdf, neuro_keys_dict):
            pass_dict = precesser.append_pass_dict(pass_dict, precess_dict)
        precess_df = pd.DataFrame(pass_dict)
        precessdf = precess_filter.filter_single(precess_df, minocc=occ_bins.min())
        precess_stater = PrecessionStat(precessdf)
        filtered_precessdf, precess_info, precess_angle = precess_stater.compute_precession_stats()

        pass_nspikes = pass_nspikes + list(filtered_precessdf['pass_nspikes'])
        # Precession - pass with low/high spike number
        if filtered_precessdf.shape[0] > 0:
            ldf = filtered_precessdf[filtered_precessdf['pass_nspikes'] < nspikes_stats[0]]  # 25% quantile
            passangle_key = precess_stater.passangle_key
            if ldf.shape[0] > 0:
                passangle_l, pass_nspikes_l = ldf[passangle_key].to_numpy(), ldf['pass_nspikes'].to_numpy()
                precess_angle_low, _, _ = PrecessionStat.compute_R(passangle_l, pass_nspikes_l)
            else:
                precess_angle_low = None
            hdf = filtered_precessdf[filtered_precessdf['pass_nspikes'] > nspikes_stats[1]]  # 75% quantile
            if hdf.shape[0] > 0:
                passangle_h, pass_nspikes_h = hdf[passangle_key].to_numpy(), hdf['pass_nspikes'].to_numpy()
                precess_angle_high, _, _ = PrecessionStat.compute_R(passangle_h, pass_nspikes_h)
            else:
                precess_angle_high = None
        else:
            precess_angle_low, precess_angle_high = None, None


        # Windows shuffling
        win_pval, win_pval_mlm = window_shuffle_wrapper(all_tsp, fieldR, fieldR_mlm, NShuffles, biner,
                                                        mlmer, interpolater_x, interpolater_y,
                                                        interpolater_angle, trange)

        # Time shift shuffling
        shi_pval, shi_pval_mlm = timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, fieldR, fieldR_mlm,
                                                               NShuffles, biner, mlmer, interpolater_x,
                                                               interpolater_y, interpolater_angle, trange)

        datadict['border'].append(border)
        datadict['num_spikes'].append(num_spikes)
        datadict['aver_rate'].append(aver_rate)
        datadict['peak_rate'].append(None)
        #
        datadict['fieldangle'].append(fieldangle)
        datadict['fieldR'].append(fieldR)
        datadict['fieldangle_mlm'].append(fieldangle_mlm)
        datadict['fieldR_mlm'].append(fieldR_mlm)

        datadict['spike_bins'].append(spike_bins)
        datadict['occ_bins'].append(occ_bins)
        datadict['win_pval'].append(win_pval)
        datadict['win_pval_mlm'].append(win_pval_mlm)
        datadict['shift_pval'].append(shi_pval)
        datadict['shift_pval_mlm'].append(shi_pval_mlm)
        #
        datadict['precess_df'].append(filtered_precessdf)
        datadict['precess_angle'].append(precess_angle)
        datadict['precess_angle_low'].append(precess_angle_low)
        datadict['precess_angle_high'].append(precess_angle_high)
        datadict['precess_info'].append(precess_info)


    print('Num spikes:\n')
    print(pd.DataFrame({'pass_nspikes':pass_nspikes})['pass_nspikes'].describe())
    datadf = pd.DataFrame(datadict)

    datadf.to_pickle(save_pth)



def plot_placefield_examples(rawdf, save_dir=None):

    field_figl = total_figw/8
    field_linew = 0.75
    field_ms = 1

    precess_figl = total_figw/3.5
    warnings.filterwarnings("ignore")


    def plot_precession(ax, d, phase, rcc_m, rcc_c, rcc_p, fontsize, marker_size):
        ax.scatter(np.concatenate([d, d]), np.concatenate([phase, phase + 2 * np.pi]), marker='.',
                   s=marker_size, c='0.7')
        xdum = np.linspace(0, 1, 10)
        ydum = xdum * (rcc_m * 2 * np.pi) + rcc_c

        linestyle = '-'

        ax.plot(xdum, ydum, c='k', linestyle=linestyle)
        ax.plot(xdum, ydum + 2 * np.pi, c='k', linestyle=linestyle)
        ax.plot(xdum, ydum - 2 * np.pi, c='k', linestyle=linestyle)
        ax.set_xlabel('Position', fontsize=fontsize)
        ax.set_ylabel('Phase (rad)', fontsize=fontsize)
        ax.set_xticks([0, 1])
        ax.set_yticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
        ax.set_yticklabels(['$-\pi$', '', '$\pi$', '', '$3\pi$'])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

        ax.set_xlim(0, 1)
        ax.set_ylim(-np.pi, 3 * np.pi)
        return ax

    # Parameters
    vthresh = 5
    sthresh = 3
    num_trials = rawdf.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    aedm = midedges(aedges)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5
    precess_filter = PrecessionFilter()

    # Plot color wheel
    fig_ch, ax_ch = color_wheel('hsv', (field_figl, field_figl))
    fig_ch.savefig(os.path.join(save_dir, 'colorwheel.%s'%(figext)), transparent=True, dpi=dpi)

    # Selected Examples
    example_list = ['CA1-field16', 'CA2-field11', 'CA3-field378',  # Precession
                    'CA1-field151', 'CA1-field222', 'CA1-field389', 'CA1-field400',
                    'CA1-field409', 'CA2-field28', 'CA2-field47', 'CA3-field75']
    fieldid = dict(CA1=0, CA2=0, CA3=0)
    fieldid_skip = dict(CA1=0, CA2=0, CA3=0)

    for ntrial in range(num_trials):

        wave = rawdf.loc[ntrial, 'wave']
        precesser = PrecessionProcesser(sthresh=sthresh, vthresh=vthresh, wave=wave)
        for ca in ['CA%d' % i for i in range(1, 4)]:

            # Get data
            field_df = rawdf.loc[ntrial, ca + 'fields']
            indata = rawdf.loc[ntrial, ca + 'indata']
            if indata.shape[0] < 1:
                continue
            trajx, trajy = indata['x'].to_numpy(), indata['y'].to_numpy()
            num_fields = field_df.shape[0]
            tunner = TunningAnalyzer(indata, vthresh, True)
            interpolater_angle = interp1d(tunner.t, tunner.movedir)
            interpolater_x = interp1d(tunner.t, tunner.x)
            interpolater_y = interp1d(tunner.t, tunner.y)
            all_maxt, all_mint = tunner.t.max(), tunner.t.min()
            trange = (all_maxt, all_mint)
            dt = tunner.t[1] - tunner.t[0]
            precesser.set_trange(trange)
            for nfield in range(num_fields):
                if ('%s-field%d'%(ca, fieldid[ca])) not in example_list:
                    fieldid[ca] += 1
                    continue
                print('Plotting fields: Trial %d/%d %s field %d/%d' % (ntrial, num_trials, ca, nfield, num_fields))

                # Get field spikes
                xytsp, xyval = field_df.loc[nfield, ['xytsp', 'xyval']]
                tsp, xsp, ysp = xytsp['tsp'], xytsp['xsp'], xytsp['ysp']
                pf = field_df.loc[nfield, 'pf']

                # Get spike angles
                insession = np.where((tsp < all_maxt) & (tsp > all_mint))[0]
                tspins, xspins, yspins = tsp[insession], xsp[insession], ysp[insession]
                anglespins = interpolater_angle(tspins)

                # Directionality
                passdf = field_df.loc[nfield, 'passes']
                (xl, yl, tl, hdl), tspl = append_info_from_passes(passdf, vthresh, sthresh, trange)
                if (len(tspl) < 1) or (len(tl) < 1):
                    # fieldid[ca] += 1
                    fieldid_skip[ca] += 1
                    continue

                x = np.concatenate(xl)
                y = np.concatenate(yl)
                pos = np.stack([x, y]).T
                hd = np.concatenate(hdl)
                tsp = np.concatenate(tspl)
                xsp, ysp = interpolater_x(tsp), interpolater_y(tsp)
                possp = np.stack([xsp, ysp]).T
                hdsp = interpolater_angle(tsp)

                # Directionality
                biner = DirectionerBining(aedges, hd)
                fieldangle, fieldR, (spike_bins, occ_bins, _) = biner.get_directionality(hdsp)
                mlmer = DirectionerMLM(pos, hd, dt, sp_binwidth, abind)
                fieldangle, fieldR, norm_prob = mlmer.get_directionality(possp, hdsp)
                norm_prob[np.isnan(norm_prob)] = 0

                # Precession per pass
                neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                       spikeangle='spikeangle')
                pass_dict = precesser.gen_precess_dict()
                for precess_dict in precesser.get_single_precession(passdf, neuro_keys_dict):
                    pass_dict = precesser.append_pass_dict(pass_dict, precess_dict)
                precess_df = pd.DataFrame(pass_dict)
                precessdf = precess_filter.filter_single(precess_df, minocc=occ_bins.min())
                filtered_precessdf = precessdf[precessdf['precess_exist']].reset_index(drop=True)
                precess_info, precess_angle = compute_precession_stats(filtered_precessdf)


                # # (Plot) Precessions in place field
                # Plot all precession
                num_precess = filtered_precessdf.shape[0]
                if num_precess > 1:
                    fig_precessfield, ax_precessfield = plt.subplots(figsize=(precess_figl, precess_figl*0.75))
                    all_dsp = np.concatenate(filtered_precessdf['dsp'].to_list())
                    all_phasesp = np.concatenate(filtered_precessdf['phasesp'].to_list())
                    regress = rcc(all_dsp, all_phasesp)
                    rcc_m, rcc_c, rcc_rho, rcc_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']

                    ax_precessfield = plot_precession(ax_precessfield, all_dsp, all_phasesp,
                                                      rcc_m, rcc_c, rcc_p, fontsize, marker_size=2)
                    fig_precessfield.tight_layout()
                    fig_precessfield.savefig(os.path.join(save_dir, 'example_precessions',
                                                          '%s-field%d.%s'%(ca, fieldid[ca], figext)), dpi=dpi)
                    plt.close()

                # # Plot for precession of each pass
                for i in range(num_precess):
                    fig_precesspass, ax_precesspass = plt.subplots(figsize=(precess_figl, precess_figl*0.8))
                    dsp, phasesp, rcc_m, rcc_c, rcc_p = filtered_precessdf.loc[i, ['dsp', 'phasesp', 'rcc_m',
                                                                                   'rcc_c', 'rcc_p']]
                    ax_precesspass = plot_precession(ax_precesspass, dsp, phasesp,
                                                     rcc_m, rcc_c, rcc_p, fontsize, marker_size=8)

                    fig_precesspass.tight_layout()
                    fig_precesspass.savefig(os.path.join(save_dir, 'example_precessions', '%s-field%d-pass%d.%s' %
                                                         (ca, fieldid[ca], i, figext)), dpi=dpi)
                    plt.close()



                # # (Plot) Place field Example
                fig2 = plt.figure(figsize=(field_figl*2, field_figl))
                peak_rate = pf['map'].max()
                ax_field2 = fig2.add_subplot(1, 2, 1)
                ax_field2.plot(trajx, trajy, c='0.8', linewidth=field_linew)
                ax_field2.plot(xyval[:, 0], xyval[:, 1], c='k', zorder=3, linewidth=field_linew)
                ax_field2.scatter(xspins, yspins, c=anglespins, marker='.', cmap='hsv', s=field_ms, zorder=2.5)
                ax_field2.axis('off')

                x_new, y_new = circular_density_1d(aedm, 30 * np.pi, 100, (-np.pi, np.pi), w=norm_prob)

                ax_polar = fig2.add_axes([0.33, 0.3, 0.6, 0.6], polar=True)
                ax_polar.plot(x_new, y_new, c='gray', linewidth=field_linew)
                ax_polar.plot([x_new[-1], x_new[0]], [y_new[-1], y_new[0]], c='gray', linewidth=field_linew)
                ax_polar.plot([fieldangle, fieldangle], [0, y_new.max()], c='k', linewidth=field_linew)
                ax_polar.axis('off')
                ax_polar.grid(False)
                basey = -0.3
                ax_polar.text(0.5, basey, '%0.2f' % (peak_rate), fontsize=legendsize, transform=ax_polar.transAxes)
                ax_polar.text(0.5, basey+0.25, '%0.2f' % (fieldR), fontsize=legendsize, transform=ax_polar.transAxes)


                fig2.tight_layout()
                fig2.savefig(os.path.join(save_dir,  'example_fields', '%s-field%d.%s' % (ca, fieldid[ca], figext)), dpi=dpi)
                plt.close()

                fieldid[ca] += 1



if __name__ == '__main__':
    rawdf_pth = 'data/emankindata_processed_withwave.pickle'
    # rawdf_pth = 'data/emankindata_processed.pickle'
    plot_dir = 'result_plots/single_field/'

    # Experiment
    df = load_pickle(rawdf_pth)
    single_field_preprocess_exp(df, save_pth='results/exp/single_field/singlefield_df_forAppend.pickle')
    # single_field_preprocess_exp(df, save_pth=None)

    # # Plotting
    # rawdf = load_pickle(rawdf_pth)
    # plot_placefield_examples(rawdf, plot_dir)


    # # Simulation
    # simdata = load_pickle('results/sim/raw/squarematch.pickle')
    # single_field_preprocess_sim(simdata, save_pth='results/sim/singlefield_df_square.pickle')