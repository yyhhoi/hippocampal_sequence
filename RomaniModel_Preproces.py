# This script preprocess the simulated results of Romani's model to single field data
import numpy as np
import pandas as pd
from os.path import join
from scipy.interpolate import interp1d

from common.correlogram import ThetaEstimator
from common.utils import load_pickle
from common.comput_utils import IndataProcessor, check_border_sim, DirectionerMLM, timeshift_shuffle_exp_wrapper, \
    get_numpass_at_angle, append_extrinsicity, pair_diff, calc_kld, find_pair_times


from common.script_wrappers import PrecessionProcesser, PrecessionFilter, get_single_precessdf, compute_precessangle
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw
figext = 'png'




def single_field_preprocess_Romani(simdata, radius=2, vthresh=2, sthresh=80, NShuffles=200, save_pth=None):
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
        It is different since straightrank depends on sampling frequency. It is 1ms in sumulation.
    subsample_fraction : float
        The fraction that the spikes would be subsampled.

    Returns
    -------

    """

    subsample_fraction = 0.4
    datadict = dict(num_spikes=[], border=[], aver_rate=[],
                    rate_angle=[], rate_R=[], rate_R_pval=[], minocc=[],
                    precess_df=[], precess_angle=[], precess_angle_low=[], precess_R=[],
                    numpass_at_precess=[], numpass_at_precess_low=[])

    Indata, SpikeData, NeuronPos = simdata['Indata'], simdata['SpikeData'], simdata['NeuronPos']
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 2 * np.pi / 16
    tunner = IndataProcessor(Indata, vthresh=vthresh, sthresh=sthresh, minpasstime=0.4)
    wave = dict(tax=Indata['t'].to_numpy(), phase=Indata['phase'].to_numpy(),
                theta=np.ones(Indata.shape[0]))

    interpolater_angle = interp1d(tunner.t, tunner.angle)
    interpolater_x = interp1d(tunner.t, tunner.x)
    interpolater_y = interp1d(tunner.t, tunner.y)
    trange = (tunner.t.max(), tunner.t.min())
    dt = tunner.t[1] - tunner.t[0]
    precesser = PrecessionProcesser(wave=wave)
    precesser.set_trange(trange)
    precess_filter = PrecessionFilter()
    lowspike_num = 13

    kappa_precess = 1
    aedges_precess = np.linspace(-np.pi, np.pi, 6)

    num_neurons = NeuronPos.shape[0]

    for nidx in range(num_neurons):
        print('%d/%d Neuron' % (nidx, num_neurons))
        # Get spike indexes + subsample
        spdf = SpikeData[SpikeData['neuronidx'] == nidx].reset_index(drop=True)
        tidxsp = spdf['tidxsp'].to_numpy().astype(int)
        np.random.seed(nidx)
        sampled_tidxsp = np.random.choice(tidxsp.shape[0] - 1, int(tidxsp.shape[0] * subsample_fraction), replace=False)
        sampled_tidxsp.sort()
        tidxsp = tidxsp[sampled_tidxsp]
        tsp = Indata.loc[tidxsp, 't'].to_numpy()
        neuron_pos = NeuronPos.iloc[nidx].to_numpy()


        # Check border
        border = check_border_sim(neuron_pos[0], neuron_pos[1], radius, (0, 2 * np.pi))

        # Construct passdf
        dist = np.sqrt((neuron_pos[0] - tunner.x) ** 2 + (neuron_pos[1] - tunner.y) ** 2)
        tok = dist < radius
        passdf = tunner.construct_singlefield_passdf(tok, tsp, interpolater_x, interpolater_y, interpolater_angle)
        # allchunk_df = passdf[(~passdf['rejected']) & (passdf['chunked']<2)].reset_index(drop=True)
        allchunk_df = passdf[(~passdf['rejected'])].reset_index(drop=True)

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


        # Field's directionality
        num_spikes = all_tsp.shape[0]
        occ_bins, _ = np.histogram(all_passangles, bins=aedges)
        minocc = occ_bins.min()
        mlmer = DirectionerMLM(pos, all_passangles, dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
        rate_angle, rate_R, norm_prob_mlm = mlmer.get_directionality(possp, all_anglesp)

        # Time shift shuffling for rate directionality
        rate_R_pval = timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, rate_R, NShuffles, mlmer,
                                                    interpolater_x, interpolater_y, interpolater_angle, trange)


        # Precession per pass
        neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                               spikeangle='spikeangle')
        accept_mask = (~passdf['rejected']) & (passdf['chunked'] < 2)
        passdf['excluded_for_precess'] = ~accept_mask
        precessdf, precess_angle, precess_R, _ = get_single_precessdf(passdf, precesser, precess_filter, neuro_keys_dict,
                                                                      field_d=radius*2, kappa=kappa_precess, bins=None)
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
            ldf = fitted_precessdf[fitted_precessdf['pass_nspikes'] < lowspike_num]  # 25% quantile
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

        else:
            numpass_at_precess = None
            precess_angle_low = None
            numpass_at_precess_low = None



        datadict['num_spikes'].append(num_spikes)
        datadict['border'].append(border)
        datadict['aver_rate'].append(aver_rate)
        #
        datadict['rate_angle'].append(rate_angle)
        datadict['rate_R'].append(rate_R)
        datadict['rate_R_pval'].append(rate_R_pval)
        datadict['minocc'].append(minocc)

        datadict['precess_df'].append(precessdf)
        datadict['precess_angle'].append(precess_angle)
        datadict['precess_angle_low'].append(precess_angle_low)
        datadict['precess_R'].append(precess_R)

        datadict['numpass_at_precess'].append(numpass_at_precess)
        datadict['numpass_at_precess_low'].append(numpass_at_precess_low)


    datadf = pd.DataFrame(datadict)

    datadf.to_pickle(save_pth)



def pair_field_preprocess_Romani(simdata, save_pth, radius=2, vthresh=2, sthresh=80, NShuffles=200):
    pairdata_dict = dict(neuron1id=[], neuron2id=[],
                         neuron1pos=[], neuron2pos=[], neurondist=[],
                         # overlap is calculated afterward
                         border1=[], border2=[],
                         aver_rate1=[], aver_rate2=[], aver_rate_pair=[],
                         com1=[], com2=[],
                         rate_angle1=[], rate_angle2=[], rate_anglep=[],
                         rate_R1=[], rate_R2=[], rate_Rp=[],
                         num_spikes1=[], num_spikes2=[], num_spikes_pair=[],
                         phaselag_AB=[], phaselag_BA=[], corr_info_AB=[], corr_info_BA=[],
                         rate_AB=[], rate_BA=[], corate=[], pair_rate=[],
                         kld=[], rate_R_pvalp=[],
                         precess_df1=[], precess_angle1=[], precess_R1=[],
                         precess_df2=[], precess_angle2=[], precess_R2=[],
                         numpass_at_precess1=[], numpass_at_precess2=[],
                         precess_dfp=[])


    Indata, SpikeData, NeuronPos = simdata['Indata'], simdata['SpikeData'], simdata['NeuronPos']
    wave = dict(tax=Indata['t'].to_numpy(), phase=Indata['phase'].to_numpy(),
                theta=np.ones(Indata.shape[0]))
    subsample_fraction = 0.25
    # setting
    minpasstime = 0.4
    minspiketresh = 14
    default_T = 1/10
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 2*np.pi/16
    precesser = PrecessionProcesser(wave=wave)
    precess_filter = PrecessionFilter()
    kappa_precess = 1
    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    field_d = radius*2



    # Precomputation
    tunner = IndataProcessor(Indata, vthresh=vthresh, sthresh=sthresh, minpasstime=minpasstime)
    interpolater_angle = interp1d(tunner.t, tunner.angle)
    interpolater_x = interp1d(tunner.t, tunner.x)
    interpolater_y = interp1d(tunner.t, tunner.y)
    all_maxt, all_mint = tunner.t.max(), tunner.t.min()
    trange = (all_maxt, all_mint)
    dt = tunner.t[1] - tunner.t[0]
    precesser.set_trange(trange)

    # Combination of pairs
    np.random.seed(0)
    sampled_idx_all = np.random.choice(NeuronPos.shape[0], 200, replace=False).astype(int)
    samples1 = sampled_idx_all[0:80]
    samples2 = sampled_idx_all[80:160]
    total_num = samples1.shape[0] * samples2.shape[0]


    progress = 0
    for i in samples1:
        for j in samples2:
            print('%d/%d (%d,%d): sample %d' % (progress, total_num, i, j, len(pairdata_dict['neuron1pos'])))

            # Distance
            neu1x, neu1y = NeuronPos.iloc[i]
            neu2x, neu2y = NeuronPos.iloc[j]
            neurondist = np.sqrt((neu1x - neu2x) ** 2 + (neu1y - neu2y) ** 2)

            # Get tok
            tok1 = np.sqrt((tunner.x - neu1x) ** 2 + (tunner.y - neu1y) ** 2) < radius
            tok2 = np.sqrt((tunner.x - neu2x) ** 2 + (tunner.y - neu2y) ** 2) < radius
            tok_union = tok1 | tok2
            tok_intersect = tok1 & tok2
            if np.sum(tok_intersect) < 10:  # anything larger than 0 will do
                progress += 1
                continue

            # Get spike indexes + subsample
            spdf1 = SpikeData[SpikeData['neuronidx'] == i].reset_index(drop=True)
            spdf2 = SpikeData[SpikeData['neuronidx'] == j].reset_index(drop=True)
            tidxsp1, tidxsp2 = spdf1['tidxsp'].to_numpy().astype(int), spdf2['tidxsp'].to_numpy().astype(int)
            np.random.seed(i * j)
            sampled_tidxsp1 = np.random.choice(tidxsp1.shape[0]-1, int(tidxsp1.shape[0] * subsample_fraction),
                                               replace=False)
            sampled_tidxsp1.sort()
            tidxsp1 = tidxsp1[sampled_tidxsp1]
            np.random.seed(i * j + 1)
            sampled_tidxsp2 = np.random.choice(tidxsp2.shape[0]-1, int(tidxsp2.shape[0] * subsample_fraction),
                                               replace=False)
            sampled_tidxsp2.sort()
            tidxsp2 = tidxsp2[sampled_tidxsp2]
            tsp1 = tunner.t[tidxsp1]
            tsp2 = tunner.t[tidxsp2]

            # Condition: Paired spikes exist
            tsp_diff = pair_diff(tsp1, tsp2)
            spok = np.sum(np.abs(tsp_diff) < default_T)
            if spok < minspiketresh:
                progress = progress + 1
                continue

            # Check border
            border1 = check_border_sim(neu1x, neu1y, radius, (0, 2 * np.pi))
            border2 = check_border_sim(neu2x, neu2y, radius, (0, 2 * np.pi))

            # Construct passes (segment & chunk) for field 1 and 2
            passdf1 = tunner.construct_singlefield_passdf(tok1, tsp1, interpolater_x, interpolater_y, interpolater_angle)
            passdf2 = tunner.construct_singlefield_passdf(tok2, tsp2, interpolater_x, interpolater_y, interpolater_angle)
            # allchunk_df1 = passdf1[(~passdf1['rejected']) & (passdf1['chunked']<2)].reset_index(drop=True)
            # allchunk_df2 = passdf2[(~passdf2['rejected']) & (passdf2['chunked']<2)].reset_index(drop=True)
            allchunk_df1 = passdf1[(~passdf1['rejected'])].reset_index(drop=True)
            allchunk_df2 = passdf2[(~passdf2['rejected'])].reset_index(drop=True)
            if (allchunk_df1.shape[0] < 1) or (allchunk_df2.shape[0] < 1):
                continue

            x1list, y1list, angle1list = allchunk_df1['x'].to_list(), allchunk_df1['y'].to_list(), allchunk_df1['angle'].to_list()
            t1list, tsp1list = allchunk_df1['t'].to_list(), allchunk_df1['tsp'].to_list()
            x2list, y2list, angle2list = allchunk_df2['x'].to_list(), allchunk_df2['y'].to_list(), allchunk_df2['angle'].to_list()
            t2list, tsp2list = allchunk_df2['t'].to_list(), allchunk_df2['tsp'].to_list()
            if (len(t1list) < 1) or (len(t2list) < 1) or (len(tsp1list) < 1) or (len(tsp2list) < 1):
                continue
            x1, x2 = np.concatenate(x1list), np.concatenate(x2list)
            y1, y2 = np.concatenate(y1list), np.concatenate(y2list)
            hd1, hd2 = np.concatenate(angle1list), np.concatenate(angle2list)
            pos1, pos2 = np.stack([x1, y1]).T, np.stack([x2, y2]).T

            tsp1, tsp2 = np.concatenate(tsp1list), np.concatenate(tsp2list)
            xsp1, xsp2 = np.concatenate(allchunk_df1['spikex'].to_list()), np.concatenate(allchunk_df2['spikex'].to_list())
            ysp1, ysp2 = np.concatenate(allchunk_df1['spikey'].to_list()), np.concatenate(allchunk_df2['spikey'].to_list())
            possp1, possp2 = np.stack([xsp1, ysp1]).T, np.stack([xsp2, ysp2]).T
            hdsp1 = np.concatenate(allchunk_df1['spikeangle'].to_list())
            hdsp2 = np.concatenate(allchunk_df2['spikeangle'].to_list())
            nspks1, nspks2 = tsp1.shape[0], tsp2.shape[0]


            # Rates
            aver_rate1 = nspks1 / (x1.shape[0] * dt)
            peak_rate1 = None
            aver_rate2 = nspks2 / (x2.shape[0] * dt)
            peak_rate2 = None

            # Directionality
            occbins1, _ = np.histogram(hd1, bins=aedges)
            spbins1, _ = np.histogram(hdsp1, bins=aedges)
            mlmer1 = DirectionerMLM(pos1, hd1, dt=dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
            rate_angle1, rate_R1, normprob1_mlm = mlmer1.get_directionality(possp1, hdsp1)
            normprob1_mlm[np.isnan(normprob1_mlm)] = 0

            occbins2, _ = np.histogram(hd2, bins=aedges)
            spbins2, _ = np.histogram(hdsp2, bins=aedges)
            mlmer2 = DirectionerMLM(pos2, hd2, dt=dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
            rate_angle2, rate_R2, normprob2_mlm = mlmer2.get_directionality(possp2, hdsp2)
            normprob2_mlm[np.isnan(normprob2_mlm)] = 0
            neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                   spikeangle='spikeangle')

            # Precession1 & Post-hoc exclusion for 1st field
            accept_mask1 = (~passdf1['rejected']) & (passdf1['chunked'] < 2)
            passdf1['excluded_for_precess'] = ~accept_mask1
            precessdf1, precessangle1, precessR1, _ = get_single_precessdf(passdf1, precesser, precess_filter, neuro_keys_dict,
                                                                           field_d=field_d, kappa=kappa_precess, bins=None)
            fitted_precessdf1 = precessdf1[precessdf1['fitted']].reset_index(drop=True)
            if (precessangle1 is not None) and (fitted_precessdf1.shape[0] > 0):

                # Post-hoc precession exclusion
                _, _, postdoc_dens1 = compute_precessangle(pass_angles=fitted_precessdf1['mean_anglesp'].to_numpy(),
                                                           pass_nspikes=fitted_precessdf1['pass_nspikes'].to_numpy(),
                                                           precess_mask=fitted_precessdf1['precess_exist'].to_numpy(),
                                                           kappa=None, bins=aedges_precess)
                (_, passbins_p1, passbins_np1, _) = postdoc_dens1
                all_passbins1 = passbins_p1 + passbins_np1
                numpass_at_precess1 = get_numpass_at_angle(target_angle=precessangle1, aedge=aedges_precess,
                                                           all_passbins=all_passbins1)
            else:
                numpass_at_precess1 = None

            # Precession2 & Post-hoc exclusion for 2nd field
            accept_mask2 = (~passdf2['rejected']) & (passdf2['chunked'] < 2)
            passdf2['excluded_for_precess'] = ~accept_mask2
            precessdf2, precessangle2, precessR2, _ = get_single_precessdf(passdf2, precesser, precess_filter, neuro_keys_dict,
                                                                           field_d=field_d, kappa=kappa_precess, bins=None)
            fitted_precessdf2 = precessdf2[precessdf2['fitted']].reset_index(drop=True)
            if (precessangle2 is not None) and (fitted_precessdf2.shape[0] > 0):

                # Post-hoc precession exclusion
                _, _, postdoc_dens2 = compute_precessangle(pass_angles=fitted_precessdf2['mean_anglesp'].to_numpy(),
                                                           pass_nspikes=fitted_precessdf2['pass_nspikes'].to_numpy(),
                                                           precess_mask=fitted_precessdf2['precess_exist'].to_numpy(),
                                                           kappa=None, bins=aedges_precess)
                (_, passbins_p2, passbins_np2, _) = postdoc_dens2
                all_passbins2 = passbins_p2 + passbins_np2
                numpass_at_precess2 = get_numpass_at_angle(target_angle=precessangle2, aedge=aedges_precess,
                                                           all_passbins=all_passbins2)
            else:
                numpass_at_precess2 = None


            # # Paired field processing

            field_d_union = radius*2

            pairedpasses = tunner.construct_pairfield_passdf(tok_union, tok1, tok2, tsp1, tsp2, interpolater_x,
                                                             interpolater_y, interpolater_angle)

            phase_finder = ThetaEstimator(0.005, 0.3, [5, 12])
            AB_tsp1_list, BA_tsp1_list = [], []
            AB_tsp2_list, BA_tsp2_list = [], []
            nspikes_AB_list, nspikes_BA_list = [], []
            duration_AB_list, duration_BA_list = [], []
            t_all = []
            passangles_all, x_all, y_all = [], [], []
            paired_tsp_list = []


            # accepted_df = pairedpasses[(~pairedpasses['rejected']) & (pairedpasses['chunked']<2)].reset_index(drop=True)
            accepted_df = pairedpasses[(~pairedpasses['rejected'])].reset_index(drop=True)
            for npass in range(accepted_df.shape[0]):

                # Get data
                t, tsp1, tsp2 = accepted_df.loc[npass, ['t', 'tsp1', 'tsp2']]
                x, y, pass_angles, v, direction = accepted_df.loc[npass, ['x', 'y', 'angle', 'v', 'direction']]
                duration = t.max() - t.min()


                # Find paired spikes
                pairidx1, pairidx2 = find_pair_times(tsp1, tsp2)
                paired_tsp1, paired_tsp2 = tsp1[pairidx1], tsp2[pairidx2]
                if (paired_tsp1.shape[0] < 1) and (paired_tsp2.shape[0] < 1):
                    continue
                paired_tsp_eachpass = np.concatenate([paired_tsp1, paired_tsp2])
                paired_tsp_list.append(paired_tsp_eachpass)
                passangles_all.append(pass_angles)
                x_all.append(x)
                y_all.append(y)
                t_all.append(t)
                if direction == 'A->B':
                    AB_tsp1_list.append(tsp1)
                    AB_tsp2_list.append(tsp2)
                    nspikes_AB_list.append(tsp1.shape[0] + tsp2.shape[0])
                    duration_AB_list.append(duration)

                elif direction == 'B->A':
                    BA_tsp1_list.append(tsp1)
                    BA_tsp2_list.append(tsp2)
                    nspikes_BA_list.append(tsp1.shape[0] + tsp2.shape[0])
                    duration_BA_list.append(duration)

            # Phase lags
            thetaT_AB, phaselag_AB, corr_info_AB = phase_finder.find_theta_isi_hilbert(AB_tsp1_list, AB_tsp2_list)
            thetaT_BA, phaselag_BA, corr_info_BA = phase_finder.find_theta_isi_hilbert(BA_tsp1_list, BA_tsp2_list)

            # Pair precession
            neuro_keys_dict1 = dict(tsp='tsp1', spikev='spike1v', spikex='spike1x', spikey='spike1y',
                                    spikeangle='spike1angle')
            neuro_keys_dict2 = dict(tsp='tsp2', spikev='spike2v', spikex='spike2x', spikey='spike2y',
                                    spikeangle='spike2angle')



            accept_mask = (~pairedpasses['rejected']) & (pairedpasses['chunked']<2) & ((pairedpasses['direction']=='A->B')| (pairedpasses['direction']=='B->A'))

            pairedpasses['excluded_for_precess'] = ~accept_mask
            precess_dfp = precesser.get_single_precession(pairedpasses, neuro_keys_dict1, field_d_union, tag='1')
            precess_dfp = precesser.get_single_precession(precess_dfp, neuro_keys_dict2, field_d_union, tag='2')
            precess_dfp = precess_filter.filter_pair(precess_dfp)
            fitted_precess_dfp = precess_dfp[precess_dfp['fitted1'] & precess_dfp['fitted2']].reset_index(drop=True)

            # Paired spikes
            if (len(paired_tsp_list) == 0) or (len(passangles_all) == 0):
                continue
            hd_pair = np.concatenate(passangles_all)
            x_pair, y_pair = np.concatenate(x_all), np.concatenate(y_all)
            pos_pair = np.stack([x_pair, y_pair]).T
            paired_tsp = np.concatenate(paired_tsp_list)
            paired_tsp = paired_tsp[(paired_tsp <= all_maxt) & (paired_tsp >= all_mint)]
            if paired_tsp.shape[0] < 1:
                continue
            num_spikes_pair = paired_tsp.shape[0]
            hdsp_pair = interpolater_angle(paired_tsp)
            xsp_pair = interpolater_x(paired_tsp)
            ysp_pair = interpolater_y(paired_tsp)
            possp_pair = np.stack([xsp_pair, ysp_pair]).T
            aver_rate_pair = num_spikes_pair / (x_pair.shape[0] * dt)

            # Pair Directionality
            occbinsp, _ = np.histogram(hd_pair, bins=aedges)
            spbinsp, _ = np.histogram(hdsp_pair, bins=aedges)
            mlmer_pair = DirectionerMLM(pos_pair, hd_pair, dt, sp_binwidth, abind)
            rate_anglep, rate_Rp, normprobp_mlm = mlmer_pair.get_directionality(possp_pair, hdsp_pair)
            normprobp_mlm[np.isnan(normprobp_mlm)] = 0

            # Time shift shuffling
            rate_R_pvalp = timeshift_shuffle_exp_wrapper(paired_tsp_list, t_all, rate_Rp,
                                                         NShuffles, mlmer_pair,
                                                         interpolater_x, interpolater_y,
                                                         interpolater_angle, trange)

            # Rates
            with np.errstate(divide='ignore', invalid='ignore'):  # None means no sample
                rate_AB = np.sum(nspikes_AB_list) / np.sum(duration_AB_list)
                rate_BA = np.sum(nspikes_BA_list) / np.sum(duration_BA_list)
                corate = np.sum(nspikes_AB_list + nspikes_BA_list) / np.sum(duration_AB_list + duration_BA_list)
                pair_rate = num_spikes_pair / np.sum(duration_AB_list + duration_BA_list)

            # KLD
            kld = calc_kld(normprob1_mlm, normprob2_mlm, normprobp_mlm)

            pairdata_dict['neuron1id'].append(i)
            pairdata_dict['neuron2id'].append(j)
            pairdata_dict['neuron1pos'].append(NeuronPos.iloc[i].to_numpy())
            pairdata_dict['neuron2pos'].append(NeuronPos.iloc[j].to_numpy())
            pairdata_dict['neurondist'].append(neurondist)


            pairdata_dict['border1'].append(border1)
            pairdata_dict['border2'].append(border2)

            pairdata_dict['aver_rate1'].append(aver_rate1)
            pairdata_dict['aver_rate2'].append(aver_rate2)
            pairdata_dict['aver_rate_pair'].append(aver_rate_pair)
            pairdata_dict['com1'].append(NeuronPos.iloc[i].to_numpy())
            pairdata_dict['com2'].append(NeuronPos.iloc[j].to_numpy())

            pairdata_dict['rate_angle1'].append(rate_angle1)
            pairdata_dict['rate_angle2'].append(rate_angle2)
            pairdata_dict['rate_anglep'].append(rate_anglep)
            pairdata_dict['rate_R1'].append(rate_R1)
            pairdata_dict['rate_R2'].append(rate_R2)
            pairdata_dict['rate_Rp'].append(rate_Rp)

            pairdata_dict['num_spikes1'].append(nspks1)
            pairdata_dict['num_spikes2'].append(nspks2)
            pairdata_dict['num_spikes_pair'].append(num_spikes_pair)

            pairdata_dict['phaselag_AB'].append(phaselag_AB)
            pairdata_dict['phaselag_BA'].append(phaselag_BA)
            pairdata_dict['corr_info_AB'].append(corr_info_AB)
            pairdata_dict['corr_info_BA'].append(corr_info_BA)

            pairdata_dict['rate_AB'].append(rate_AB)
            pairdata_dict['rate_BA'].append(rate_BA)
            pairdata_dict['corate'].append(corate)
            pairdata_dict['pair_rate'].append(pair_rate)
            pairdata_dict['kld'].append(kld)
            pairdata_dict['rate_R_pvalp'].append(rate_R_pvalp)

            pairdata_dict['precess_df1'].append(fitted_precessdf1)
            pairdata_dict['precess_angle1'].append(precessangle1)
            pairdata_dict['precess_R1'].append(precessR1)
            pairdata_dict['precess_df2'].append(fitted_precessdf2)
            pairdata_dict['precess_angle2'].append(precessangle2)
            pairdata_dict['precess_R2'].append(precessR2)
            pairdata_dict['numpass_at_precess1'].append(numpass_at_precess1)
            pairdata_dict['numpass_at_precess2'].append(numpass_at_precess2)
            pairdata_dict['precess_dfp'].append(fitted_precess_dfp)
            progress = progress + 1
    pairdata = pd.DataFrame(pairdata_dict)

    # Convert distance to overlap
    dist_range = pairdata['neurondist'].max() - pairdata['neurondist'].min()
    pairdata['overlap'] = (pairdata['neurondist'].max() - pairdata['neurondist']) / dist_range
    pairdata.to_pickle(save_pth)
    pairdata = append_extrinsicity(pairdata)
    pairdata.to_pickle(save_pth)
    return pairdata





def main():

    # Single field
    simdata = load_pickle('results/sim/raw/squarematch.pickle')
    single_field_preprocess_Romani(simdata, save_pth='results/sim/singlefield_df.pickle')

    # # Pair field
    simdata = load_pickle('results/sim/raw/squarematch.pickle')
    pair_field_preprocess_Romani(simdata, save_pth='results/sim/pairfield_df.pickle')



if __name__ == '__main__':
    main()