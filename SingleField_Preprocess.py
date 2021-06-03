# This script preprocess the experimental data of Emily's to single field data
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
from scipy.interpolate import interp1d
from pycircstat.descriptive import resultant_vector_length, cdiff
from pycircstat.descriptive import mean as circmean
import warnings

from common.utils import load_pickle
from common.comput_utils import check_border, IndataProcessor, midedges, segment_passes, \
    check_border_sim, append_info_from_passes, \
    DirectionerBining, DirectionerMLM, timeshift_shuffle_exp_wrapper, circular_density_1d, get_numpass_at_angle, \
    PassShuffler

from common.visualization import color_wheel, directionality_polar_plot, customlegend

from common.script_wrappers import DirectionalityStatsByThresh, PrecessionProcesser, PrecessionFilter, \
    construct_passdf_sim, get_single_precessdf, compute_precessangle
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

figext = 'png'
#
#
# figext = 'eps'

def single_field_preprocess_exp(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=None):
    fielddf_dict = dict(ca=[], num_spikes=[], border=[], aver_rate=[], peak_rate=[],
                      rate_angle=[], rate_R=[], rate_R_pval=[], minocc=[], field_area=[], field_bound=[],
                      precess_df=[], precess_angle=[], precess_angle_low=[], precess_R=[], precess_R_pval=[],
                      numpass_at_precess=[], numpass_at_precess_low=[])

    num_trials = df.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5

    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = 1
    precess_filter = PrecessionFilter()
    nspikes_stats = {'CA1': 6, 'CA2': 6, 'CA3': 7}  # 25% quantile of precessing passes
    for ntrial in range(num_trials):

        wave = df.loc[ntrial, 'wave']
        precesser = PrecessionProcesser(wave=wave)

        for ca in ['CA%d' % (i + 1) for i in range(3)]:

            # Get data
            field_df = df.loc[ntrial, ca + 'fields']
            indata = df.loc[ntrial, ca + 'indata']
            num_fields = field_df.shape[0]
            if indata.shape[0] < 1:
                continue

            tunner = IndataProcessor(indata, vthresh=vthresh, sthresh=sthresh, minpasstime=0.4)
            interpolater_angle = interp1d(tunner.t, tunner.angle)
            interpolater_x = interp1d(tunner.t, tunner.x)
            interpolater_y = interp1d(tunner.t, tunner.y)
            trange = (tunner.t.max(), tunner.t.min())
            dt = tunner.t[1] - tunner.t[0]
            precesser.set_trange(trange)

            for nf in range(num_fields):
                print('%d/%d trial, %s, %d/%d field' % (ntrial, num_trials, ca, nf, num_fields))
                # Get field info
                mask, pf, xyval = field_df.loc[nf, ['mask', 'pf', 'xyval']]
                tsp = field_df.loc[nf, 'xytsp']['tsp']
                xaxis, yaxis = pf['X'][:, 0], pf['Y'][0, :]
                field_area = np.sum(mask)
                field_d = np.sqrt(field_area/np.pi)*2
                border = check_border(mask, margin=2)

                # Construct passes (segment & chunk)
                tok, idin = tunner.get_idin(mask, xaxis, yaxis)
                passdf = tunner.construct_singlefield_passdf(tok, tsp, interpolater_x, interpolater_y, interpolater_angle)
                allchunk_df = passdf[(~passdf['rejected']) & (passdf['chunked']<2)].reset_index(drop=True)
                # allchunk_df = passdf[(~passdf['rejected'])].reset_index(drop=True)

                # Get info from passdf and interpolate
                if allchunk_df.shape[0] < 1:
                    continue
                all_x_list, all_y_list = allchunk_df['x'].to_list(), allchunk_df['y'].to_list()
                all_t_list, all_passangles_list = allchunk_df['t'].to_list(), allchunk_df['angle'].to_list()
                all_tsp_list, all_chunked_list = allchunk_df['tsp'].to_list(), allchunk_df['chunked'].to_list()
                all_x = np.concatenate(all_x_list)
                all_y = np.concatenate(all_y_list)
                all_passangles = np.concatenate(all_passangles_list)
                all_tsp = np.concatenate(all_tsp_list)
                all_anglesp = np.concatenate(allchunk_df['spikeangle'].to_list())
                xsp, ysp = np.concatenate(allchunk_df['spikex'].to_list()), np.concatenate(allchunk_df['spikey'].to_list())
                pos = np.stack([all_x, all_y]).T
                possp = np.stack([xsp, ysp]).T

                # Average firing rate
                aver_rate = all_tsp.shape[0] / (all_x.shape[0] * dt)
                peak_rate = np.max(field_df.loc[nf, 'pf']['map'] * mask)

                # Field's directionality - need angle, anglesp, pos
                num_spikes = all_tsp.shape[0]
                occ_bins, _ = np.histogram(all_passangles, bins=aedges)
                minocc = occ_bins.min()
                mlmer = DirectionerMLM(pos, all_passangles, dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
                rate_angle, rate_R, norm_prob_mlm = mlmer.get_directionality(possp, all_anglesp)


                # Time shift shuffling for rate directionality
                if np.isnan(rate_R):
                    rate_R_pval = np.nan
                else:
                    rate_R_pval = timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, rate_R,
                                                                 NShuffles, mlmer, interpolater_x,
                                                                 interpolater_y, interpolater_angle, trange)



                # Precession per pass
                neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                       spikeangle='spikeangle')
                accept_mask = (~passdf['rejected']) & (passdf['chunked'] < 2)
                passdf['excluded_for_precess'] = ~accept_mask
                precessdf, precess_angle, precess_R, _ = get_single_precessdf(passdf, precesser, precess_filter, neuro_keys_dict,
                                                                              field_d=field_d, kappa=kappa_precess, bins=None)
                fitted_precessdf = precessdf[precessdf['fitted']].reset_index(drop=True)
                # Proceed only if precession exists
                if (precess_angle is not None) and (fitted_precessdf['precess_exist'].sum()>0):

                    # Post-hoc precession exclusion
                    _, binR, postdoc_dens = compute_precessangle(pass_angles=fitted_precessdf['mean_anglesp'].to_numpy(),
                                                              pass_nspikes=fitted_precessdf['pass_nspikes'].to_numpy(),
                                                              precess_mask=fitted_precessdf['precess_exist'].to_numpy(),
                                                              kappa=None, bins=aedges_precess)
                    (_, passbins_p, passbins_np, _) = postdoc_dens
                    all_passbins = passbins_p + passbins_np
                    numpass_at_precess = get_numpass_at_angle(target_angle=precess_angle, aedge=aedges_precess,
                                                              all_passbins=all_passbins)


                    # Precession - low-spike passes
                    ldf = fitted_precessdf[fitted_precessdf['pass_nspikes'] < nspikes_stats[ca]]  # 25% quantile
                    if (ldf.shape[0] > 0) and (ldf['precess_exist'].sum() > 0):
                        precess_angle_low, _, _ = compute_precessangle(pass_angles=ldf['mean_anglesp'].to_numpy(),
                                                                       pass_nspikes=ldf['pass_nspikes'].to_numpy(),
                                                                       precess_mask=ldf['precess_exist'].to_numpy(),
                                                                       kappa=kappa_precess, bins=None)
                        _, _, postdoc_dens_low = compute_precessangle(pass_angles=ldf['mean_anglesp'].to_numpy(),
                                                                       pass_nspikes=ldf['pass_nspikes'].to_numpy(),
                                                                       precess_mask=ldf['precess_exist'].to_numpy(),
                                                                       kappa=None, bins=aedges_precess)
                        (_, passbins_p_low, passbins_np_low, _) = postdoc_dens_low
                        all_passbins_low = passbins_p_low + passbins_np_low
                        numpass_at_precess_low = get_numpass_at_angle(target_angle=precess_angle_low, aedge=aedges_precess,
                                                                      all_passbins=all_passbins_low)
                    else:
                        precess_angle_low = None
                        numpass_at_precess_low = None

                    # # Precession - time shift shuffle
                    # psr = PassShuffler(all_t_list, all_tsp_list)
                    # all_shuffled_R = np.zeros(NShuffles)
                    # shufi = 0
                    # fail_times = 0
                    # repeated_times = 0
                    # tmpt = time.time()
                    # while (shufi < NShuffles) & (fail_times < NShuffles):
                    #
                    #     # Re-construct passdf
                    #     shifted_tsp_boxes = psr.timeshift_shuffle(seed=shufi+fail_times, return_concat=False)
                    #
                    #
                    #     shifted_spikex_boxes, shifted_spikey_boxes = [], []
                    #     shifted_spikeangle_boxes, shifted_rejected_boxes = [], []
                    #
                    #     for boxidx, shuffled_tsp_box in enumerate(shifted_tsp_boxes):
                    #         shuffled_tsp_box = shuffled_tsp_box[(shuffled_tsp_box < trange[0]) & (shuffled_tsp_box > trange[1])]
                    #         shifted_spikex_boxes.append(interpolater_x(shuffled_tsp_box))
                    #         shifted_spikey_boxes.append(interpolater_y(shuffled_tsp_box))
                    #         shifted_spikeangle_boxes.append(interpolater_angle(shuffled_tsp_box))
                    #         rejected, _ = tunner.rejection_singlefield(shuffled_tsp_box, all_t_list[boxidx], all_passangles_list[boxidx])
                    #         shifted_rejected_boxes.append(rejected)
                    #     shuffled_passdf = pd.DataFrame({'x':all_x_list, 'y':all_y_list, 't':all_t_list, 'angle':all_passangles_list,
                    #                                     'chunked':all_chunked_list, 'rejected':shifted_rejected_boxes,
                    #                                     neuro_keys_dict['tsp']:shifted_tsp_boxes, neuro_keys_dict['spikex']:shifted_spikex_boxes,
                    #                                     neuro_keys_dict['spikey']:shifted_spikey_boxes, neuro_keys_dict['spikeangle']:shifted_spikeangle_boxes})
                    #     shuffled_accept_mask = (~shuffled_passdf['rejected']) & (shuffled_passdf['chunked'] < 2)
                    #     shuffled_passdf['excluded_for_precess'] = ~shuffled_accept_mask
                    #     shuf_precessdf, shuf_precessangle, shuf_precessR, _ = get_single_precessdf(shuffled_passdf, precesser, precess_filter, neuro_keys_dict, occ_bins.min(),
                    #                                                                                field_d=field_d, kappa=kappa_precess, bins=None)
                    #
                    #     if (shuf_precessdf.shape[0] > 0) and (shuf_precessR is not None):
                    #         all_shuffled_R[shufi] = shuf_precessR
                    #         shufi += 1
                    #     else:
                    #         fail_times += 1
                    # precess_R_pval = 1 - np.nanmean(precess_R > all_shuffled_R)
                    # precess_shuffletime = time.time()-tmpt
                    # print('Shuf %0.4f - %0.4f - %0.4f, Target %0.4f'%(np.quantile(all_shuffled_R, 0.25), np.quantile(all_shuffled_R, 0.5), np.quantile(all_shuffled_R, 0.75), precess_R))
                    # print('Best precess=%0.2f, pval=%0.5f, time=%0.2f, failed=%d, repeated=%d'%(precess_angle, precess_R_pval, precess_shuffletime, fail_times, repeated_times))

                    precess_R_pval = 1


                else:
                    numpass_at_precess = None
                    precess_angle_low = None
                    numpass_at_precess_low = None
                    precess_R_pval = None

                fielddf_dict['ca'].append(ca)
                fielddf_dict['num_spikes'].append(num_spikes)
                fielddf_dict['field_area'].append(field_area)
                fielddf_dict['field_bound'].append(xyval)
                fielddf_dict['border'].append(border)
                fielddf_dict['aver_rate'].append(aver_rate)
                fielddf_dict['peak_rate'].append(peak_rate)
                fielddf_dict['rate_angle'].append(rate_angle)
                fielddf_dict['rate_R'].append(rate_R)
                fielddf_dict['rate_R_pval'].append(rate_R_pval)
                fielddf_dict['minocc'].append(minocc)
                fielddf_dict['precess_df'].append(fitted_precessdf)
                fielddf_dict['precess_angle'].append(precess_angle)
                fielddf_dict['precess_angle_low'].append(precess_angle_low)
                fielddf_dict['precess_R'].append(precess_R)
                fielddf_dict['precess_R_pval'].append(precess_R_pval)
                fielddf_dict['numpass_at_precess'].append(numpass_at_precess)
                fielddf_dict['numpass_at_precess_low'].append(numpass_at_precess_low)

                # tmpdf = pd.DataFrame(dict(ca=fielddf_dict['ca'], pval=fielddf_dict['precess_R_pval']))
                # for catmp, cadftmp in tmpdf.groupby('ca'):
                #     nonnan_count = cadftmp[~cadftmp['pval'].isna()].shape[0]
                #     if nonnan_count ==0:
                #         nonnan_count = 1
                #     sig_count = cadftmp[cadftmp['pval'] < 0.05].shape[0]
                #     print('%s: ALL %d/%d=%0.2f, Among Precess %d/%d=%0.2f'%(catmp, sig_count, cadftmp.shape[0],
                #                                                             sig_count/cadftmp.shape[0], sig_count,
                #                                                             nonnan_count, sig_count/nonnan_count))


    fielddf_raw = pd.DataFrame(fielddf_dict)

    # Assign field ids within ca
    fielddf_raw['fieldid_ca'] = 0
    for ca, cadf in fielddf_raw.groupby('ca'):
        fieldidca = np.arange(cadf.shape[0]) + 1
        index_ca = cadf.index
        fielddf_raw.loc[index_ca, 'fieldid_ca'] = fieldidca

    fielddf_raw.to_pickle(save_pth)
    return fielddf_raw


def single_field_preprocess_networks(simdata, radius=10, vthresh=2, sthresh=80, NShuffles=200,
                                     subsample_fraction=1, save_pth=None):
    """

    Parameters
    ----------
    Indata
    SpikeData
    NeuronPos
    radius
    vthresh : float
        Default = 5. Same as Emily's data.
    sthresh : float
        Default = 80. Determined by the same percentile (10%) of passes excluded in Emily's data (sthresh = 3 there).
    subsample_fraction : float
        The fraction that the spikes would be subsampled. =1 if no subsampling is needed

    Returns
    -------

    """

    datadict = dict(num_spikes=[], border=[], aver_rate=[], peak_rate=[],
                    fieldangle_mlm=[], fieldR_mlm=[],
                    spike_bins=[], occ_bins=[], shift_pval_mlm=[],
                    precess_df=[], precess_angle=[], precess_angle_low=[], precess_R=[])

    Indata, SpikeData, NeuronPos = simdata['Indata'], simdata['SpikeData'], simdata['NeuronPos']
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5

    tunner = IndataProcessor(Indata, vthresh, True)
    wave = dict(tax=Indata['t'].to_numpy(), phase=Indata['phase'].to_numpy(),
                theta=np.ones(Indata.shape[0]))

    interpolater_angle = interp1d(tunner.t, tunner.angle)
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
        if subsample_fraction < 1:
            np.random.seed(nidx)
            sampled_tidxsp = np.random.choice(tidxsp.shape[0] - 1, int(tidxsp.shape[0] * subsample_fraction),
                                              replace=False)
            sampled_tidxsp.sort()
            tidxsp = tidxsp[sampled_tidxsp]

        # Get tok
        neuronx, neurony = NeuronPos.loc[nidx, ['neuronx', 'neurony']]
        dist = np.sqrt((neuronx - tunner.x) ** 2 + (neurony - tunner.y) ** 2)
        tok = dist < radius

        # Check border
        border = check_border_sim(neuronx, neurony, radius, (-40, 40))

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
        occ_bins, _ = np.histogram(all_passangles, bins=aedges)
        spike_bins, _ = np.histogram(all_anglesp, bins=aedges)
        mlmer = DirectionerMLM(pos, all_passangles, dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
        fieldangle_mlm, fieldR_mlm, norm_prob_mlm = mlmer.get_directionality(possp, all_anglesp)

        # Precession per pass
        neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                               spikeangle='spikeangle')
        precessdf, precess_angle, precess_R, _ = get_single_precessdf(passdf, precesser, precess_filter,
                                                                      neuro_keys_dict)

        filtered_precessdf = precessdf[precessdf['precess_exist']].reset_index(drop=True)

        pass_nspikes = pass_nspikes + list(filtered_precessdf['pass_nspikes'])

        # Precession - pass with low/high spike number
        if precessdf.shape[0] > 0:
            ldf = precessdf[precessdf['pass_nspikes'] < nspikes_stats[0]]  # 25% quantile
            if (ldf.shape[0] > 0) and (ldf['precess_exist'].sum() > 0):
                passangle_l, pass_nspikes_l = ldf['spike_angle'].to_numpy(), ldf['pass_nspikes'].to_numpy()
                precess_mask_l = ldf['precess_exist'].to_numpy()
                precess_angle_low, _, _ = compute_precessangle(passangle_l, pass_nspikes_l, precess_mask_l)
            else:
                precess_angle_low = None
        else:
            precess_angle_low = None

        # Time shift shuffling
        shi_pval_mlm = timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, fieldR_mlm,
                                                     NShuffles, mlmer, interpolater_x,
                                                     interpolater_y, interpolater_angle, trange)

        datadict['border'].append(border)
        datadict['num_spikes'].append(num_spikes)
        datadict['aver_rate'].append(aver_rate)
        datadict['peak_rate'].append(None)
        datadict['fieldangle_mlm'].append(fieldangle_mlm)
        datadict['fieldR_mlm'].append(fieldR_mlm)
        datadict['spike_bins'].append(spike_bins)
        datadict['occ_bins'].append(occ_bins)
        datadict['shift_pval_mlm'].append(shi_pval_mlm)
        datadict['precess_df'].append(precessdf)
        datadict['precess_angle'].append(precess_angle)
        datadict['precess_angle_low'].append(precess_angle_low)
        datadict['precess_R'].append(precess_R)

    print('Num spikes:\n')
    print(pd.DataFrame({'pass_nspikes': pass_nspikes})['pass_nspikes'].describe())
    datadf = pd.DataFrame(datadict)

    datadf.to_pickle(save_pth)


def plot_placefield_examples(rawdf, save_dir=None):

    example_dir = join(save_dir, 'example_fields2')
    os.makedirs(example_dir)

    field_figl = total_figw / 8
    field_linew = 0.75
    field_ms = 1
    warnings.filterwarnings("ignore")

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
    fig_ch.savefig(os.path.join(save_dir, 'colorwheel.%s' % (figext)), transparent=True, dpi=dpi)

    # Selected Examples
    example_list = ['CA1-field16', 'CA2-field11', 'CA3-field378',  # Precession
                    'CA1-field151', 'CA1-field222', 'CA1-field389', 'CA1-field400',
                    'CA1-field409', 'CA2-field28', 'CA2-field47', 'CA3-field75']
    fieldid = dict(CA1=0, CA2=0, CA3=0)
    fieldid_skip = dict(CA1=0, CA2=0, CA3=0)

    for ntrial in range(num_trials):

        wave = rawdf.loc[ntrial, 'wave']
        precesser = PrecessionProcesser(wave=wave)
        for ca in ['CA%d' % i for i in range(1, 4)]:

            # Get data
            field_df = rawdf.loc[ntrial, ca + 'fields']
            indata = rawdf.loc[ntrial, ca + 'indata']
            if indata.shape[0] < 1:
                continue
            trajx, trajy = indata['x'].to_numpy(), indata['y'].to_numpy()
            num_fields = field_df.shape[0]
            tunner = IndataProcessor(indata, vthresh=5, sthresh=3, minpasstime=0.4, smooth=None)
            interpolater_angle = interp1d(tunner.t, tunner.angle)
            interpolater_x = interp1d(tunner.t, tunner.x)
            interpolater_y = interp1d(tunner.t, tunner.y)
            all_maxt, all_mint = tunner.t.max(), tunner.t.min()
            trange = (all_maxt, all_mint)
            dt = tunner.t[1] - tunner.t[0]
            precesser.set_trange(trange)
            for nfield in range(num_fields):
                if ('%s-field%d' % (ca, fieldid[ca])) not in example_list:
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

                # # (Plot) Place field Example
                fig2 = plt.figure(figsize=(field_figl * 2, field_figl))
                peak_rate = pf['map'].max()
                ax_field2 = fig2.add_subplot(1, 2, 1)
                ax_field2.plot(trajx, trajy, c='0.8', linewidth=0.25)
                ax_field2.plot(xyval[:, 0], xyval[:, 1], c='k', zorder=3, linewidth=field_linew)
                ax_field2.scatter(xspins, yspins, c=anglespins, marker='.', cmap='hsv', s=field_ms, zorder=2.5)
                ax_field2.axis('off')

                x_new, y_new = circular_density_1d(aedm, 30 * np.pi, 100, (-np.pi, np.pi), w=norm_prob)
                l = y_new.max()
                ax_polar = fig2.add_axes([0.33, 0.3, 0.6, 0.6], polar=True)
                ax_polar.plot(x_new, y_new, c='0.3', linewidth=field_linew, zorder=2.1)
                ax_polar.plot([x_new[-1], x_new[0]], [y_new[-1], y_new[0]], c='0.3', linewidth=field_linew, zorder=2.1)
                # ax_polar.plot([fieldangle, fieldangle], [0, l], c='k', linewidth=field_linew)
                ax_polar.annotate("", xy=(fieldangle, l), xytext=(0, 0), color='k',  zorder=3,  arrowprops=dict(arrowstyle="->"))
                ax_polar.annotate(r'$\theta_{rate}$', xy=(fieldangle, l), fontsize=legendsize)
                ax_polar.spines['polar'].set_visible(False)
                ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
                ax_polar.set_yticks([])
                ax_polar.set_yticklabels([])
                ax_polar.set_xticklabels([])

                basey = -0.3
                ax_polar.annotate('%0.2f' % (peak_rate), xy=(0.75, 0.05), color='k',  zorder=3, fontsize=legendsize, xycoords='figure fraction')
                ax_polar.annotate('%0.2f' % (fieldR), xy=(0.75, 0.175), color='k',  zorder=3, fontsize=legendsize, xycoords='figure fraction')
                # ax_polar.text(0.5, basey, '%0.2f' % (peak_rate), fontsize=legendsize, transform=ax_polar.transAxes)
                # ax_polar.text(0.5, basey + 0.25, '%0.2f' % (fieldR), fontsize=legendsize, transform=ax_polar.transAxes)

                fig2.tight_layout()
                fig2.savefig(os.path.join(example_dir, '%s-field%d.png' % (ca, fieldid[ca])), dpi=dpi)
                fig2.savefig(os.path.join(example_dir, '%s-field%d.eps' % (ca, fieldid[ca])), dpi=dpi)
                plt.close()

                fieldid[ca] += 1


if __name__ == '__main__':
    rawdf_pth = 'data/emankindata_processed_withwave.pickle'
    save_pth = 'results/emankin/singlefield_df_NoShuffle.pickle'

    # df = load_pickle(rawdf_pth)
    # single_field_preprocess_exp(df, vthresh=5, sthresh=3, save_pth=save_pth)

    # # # Plotting
    plot_dir = 'result_plots/single_field/'
    df = load_pickle(rawdf_pth)
    plot_placefield_examples(df, plot_dir)

    # # Simulation
    # simdata = load_pickle('results/sim/raw/squarematch.pickle')
    # single_field_preprocess_sim(simdata, save_pth='results/sim/singlefield_df_square2.pickle')

    # # Network project
    # simdata = load_pickle('results/network_proj/sim_result_dwdt.pickle')
    # single_field_preprocess_networks(simdata, save_pth='results/network_proj/singlefield_df_networkproj_dwdt.pickle')

    # df = load_pickle(rawdf_pth)
    # precession_dist_inspection(df)
    # test_pass_criteria(df)
    # test_pass_minbins()

    # test_posthoc()
