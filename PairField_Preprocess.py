import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pycircstat.tests import watson_williams
from pycircstat.descriptive import mean as circmean
from pycircstat.descriptive import resultant_vector_length
from scipy.stats import vonmises, ranksums
from scipy.interpolate import interp1d

from common.script_wrappers import PrecessionProcesser, PrecessionFilter, construct_passdf_sim, \
    construct_pairedpass_df_sim, get_single_precessdf, compute_precessangle
from common.linear_circular_r import rcc
from common.utils import load_pickle, stat_record
from common.comput_utils import dist_overlap, normsig, append_extrinsicity, linear_circular_gauss_density, \
    find_pair_times, \
    IndataProcessor, check_border, window_shuffle_gen, \
    compute_straightness, calc_kld, window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, segment_passes, \
    pair_diff, \
    check_border_sim, midedges, append_info_from_passes, DirectionerMLM, DirectionerBining, circular_density_1d, \
    get_numpass_at_angle
from common.correlogram import Crosser, ThetaEstimator, Bootstrapper
from common.visualization import plot_correlogram, directionality_polar_plot
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

figext = 'png'
# figext = 'eps'

def pair_field_preprocess_exp(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=None):
    pairdata_dict = dict(pair_id=[], ca=[], overlap=[],
                         border1=[], border2=[], area1=[], area2=[], xyval1=[], xyval2=[],
                         aver_rate1=[], aver_rate2=[], aver_rate_pair=[],
                         peak_rate1=[], peak_rate2=[],
                         minocc1=[], minocc2=[],
                         fieldcoor1=[], fieldcoor2=[], com1=[], com2=[],

                         rate_angle1=[], rate_angle2=[], rate_anglep=[],
                         rate_R1=[], rate_R2=[], rate_Rp=[],

                         num_spikes1=[], num_spikes2=[], num_spikes_pair=[],
                         phaselag_AB=[], phaselag_BA=[], corr_info_AB=[], corr_info_BA=[],
                         thetaT_AB=[], thetaT_BA=[],
                         rate_AB=[], rate_BA=[], corate=[], pair_rate=[],
                         kld=[], rate_R_pvalp=[],
                         precess_df1=[], precess_angle1=[], precess_R1=[],
                         precess_df2=[], precess_angle2=[], precess_R2=[],
                         numpass_at_precess1=[], numpass_at_precess2=[],
                         precess_dfp=[])

    num_trials = df.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5

    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = 1
    precess_filter = PrecessionFilter()

    pair_id = 0
    for ntrial in range(num_trials):
        wave = df.loc[ntrial, 'wave']
        precesser = PrecessionProcesser(wave=wave)


        for ca in ['CA%d' % (i + 1) for i in range(3)]:

            # Get data
            pair_df = df.loc[ntrial, ca + 'pairs']
            field_df = df.loc[ntrial, ca + 'fields']
            indata = df.loc[ntrial, ca + 'indata']
            if (indata.shape[0] < 1) & (pair_df.shape[0] < 1) & (field_df.shape[0] < 1):
                continue

            tunner = IndataProcessor(indata, vthresh=vthresh, sthresh=sthresh, minpasstime=0.4)
            interpolater_angle = interp1d(tunner.t, tunner.angle)
            interpolater_x = interp1d(tunner.t, tunner.x)
            interpolater_y = interp1d(tunner.t, tunner.y)
            all_maxt, all_mint = tunner.t.max(), tunner.t.min()
            trange = (all_maxt, all_mint)
            dt = tunner.t[1] - tunner.t[0]
            precesser.set_trange(trange)

            ##  Loop for pairs
            num_pairs = pair_df.shape[0]
            for npair in range(num_pairs):

                print('%d/%d trial, %s, %d/%d pair id=%d' % (ntrial, num_trials, ca, npair, num_pairs, pair_id))

                # find within-mask indexes
                field_ids = pair_df.loc[npair, 'fi'] - 1  # minus 1 to convert to python index
                mask1, pf1, xyval1 = field_df.loc[field_ids[0], ['mask', 'pf', 'xyval']]
                mask2, pf2, xyval2 = field_df.loc[field_ids[1], ['mask', 'pf', 'xyval']]
                tsp1 = field_df.loc[field_ids[0], 'xytsp']['tsp']
                tsp2 = field_df.loc[field_ids[1], 'xytsp']['tsp']
                xaxis1, yaxis1 = pf1['X'][:, 0], pf1['Y'][0, :]
                xaxis2, yaxis2 = pf2['X'][:, 0], pf2['Y'][0, :]
                assert (np.all(xaxis1==xaxis2) and np.all(yaxis1==yaxis2))
                area1 = mask1.sum()
                area2 = mask2.sum()
                field_d1 = np.sqrt(area1/np.pi)*2
                field_d2 = np.sqrt(area2/np.pi)*2


                # Find overlap
                _, ks_dist, _ = dist_overlap(pf1['map'], pf2['map'], mask1, mask2)

                # Field's center coordinates
                maskedmap1, maskedmap2 = pf1['map'] * mask1, pf2['map'] * mask2
                cooridx1 = np.unravel_index(maskedmap1.argmax(), maskedmap1.shape)
                cooridx2 = np.unravel_index(maskedmap2.argmax(), maskedmap2.shape)
                fcoor1 = np.array([pf1['X'][cooridx1[0], cooridx1[1]], pf1['Y'][cooridx1[0], cooridx1[1]]])
                fcoor2 = np.array([pf2['X'][cooridx2[0], cooridx2[1]], pf2['Y'][cooridx2[0], cooridx2[1]]])
                XY1 = np.stack([pf1['X'].ravel(), pf1['Y'].ravel()])
                XY2 = np.stack([pf2['X'].ravel(), pf2['Y'].ravel()])
                com1 = np.sum(XY1 * (maskedmap1/np.sum(maskedmap1)).ravel().reshape(1, -1), axis=1)
                com2 = np.sum(XY2 * (maskedmap2/np.sum(maskedmap2)).ravel().reshape(1, -1), axis=1)

                # Border
                border1 = check_border(mask1, margin=2)
                border2 = check_border(mask2, margin=2)

                # Construct passes (segment & chunk) for field 1 and 2
                tok1, idin1 = tunner.get_idin(mask1, xaxis1, yaxis1)
                tok2, idin2 = tunner.get_idin(mask2, xaxis2, yaxis2)
                passdf1 = tunner.construct_singlefield_passdf(tok1, tsp1, interpolater_x, interpolater_y, interpolater_angle)
                passdf2 = tunner.construct_singlefield_passdf(tok2, tsp2, interpolater_x, interpolater_y, interpolater_angle)
                allchunk_df1 = passdf1[(~passdf1['rejected']) & (passdf1['chunked']<2)].reset_index(drop=True)
                allchunk_df2 = passdf2[(~passdf2['rejected']) & (passdf2['chunked']<2)].reset_index(drop=True)
                # allchunk_df1 = passdf1[(~passdf1['rejected']) ].reset_index(drop=True)
                # allchunk_df2 = passdf2[(~passdf2['rejected']) ].reset_index(drop=True)
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
                aver_rate1 = nspks1 / (x1.shape[0]*dt)
                peak_rate1 = np.max(pf1['map'] * mask1)
                aver_rate2 = nspks2 / (x2.shape[0] * dt)
                peak_rate2 = np.max(pf2['map'] * mask2)

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
                minocc1, minocc2 = occbins1.min(), occbins2.min()
                neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                        spikeangle='spikeangle')

                # Precession1 & Post-hoc exclusion for 1st field
                accept_mask1 = (~passdf1['rejected']) & (passdf1['chunked'] < 2)
                passdf1['excluded_for_precess'] = ~accept_mask1
                precessdf1, precessangle1, precessR1, _ = get_single_precessdf(passdf1, precesser, precess_filter, neuro_keys_dict, occbins1.min(),
                                                                               field_d=field_d1, kappa=kappa_precess, bins=None)
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
                precessdf2, precessangle2, precessR2, _ = get_single_precessdf(passdf2, precesser, precess_filter, neuro_keys_dict, occbins2.min(),
                                                                               field_d=field_d2, kappa=kappa_precess, bins=None)
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
                # Construct pairedpasses
                mask_union = mask1 | mask2
                field_d_union = np.sqrt(mask_union.sum()/np.pi)*2
                tok_pair, _ = tunner.get_idin(mask_union, xaxis1, yaxis1)
                pairedpasses = tunner.construct_pairfield_passdf(tok_pair, tok1, tok2, tsp1, tsp2, interpolater_x,
                                                                 interpolater_y, interpolater_angle)

                # Phase lags
                phase_finder = ThetaEstimator(0.005, 0.3, [5, 12])
                AB_tsp1_list, BA_tsp1_list = [], []
                AB_tsp2_list, BA_tsp2_list = [], []
                nspikes_AB_list, nspikes_BA_list = [], []
                duration_AB_list, duration_BA_list = [], []
                t_all = []
                passangles_all, x_all, y_all = [], [], []
                paired_tsp_list = []


                accepted_df = pairedpasses[(~pairedpasses['rejected']) & (pairedpasses['chunked']<2)].reset_index(drop=True)
                # accepted_df = pairedpasses[(~pairedpasses['rejected'])].reset_index(drop=True)
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



                pairdata_dict['pair_id'].append(pair_id)
                pairdata_dict['ca'].append(ca)
                pairdata_dict['overlap'].append(ks_dist)
                pairdata_dict['border1'].append(border1)
                pairdata_dict['border2'].append(border2)
                pairdata_dict['area1'].append(area1)
                pairdata_dict['area2'].append(area2)
                pairdata_dict['xyval1'].append(xyval1)
                pairdata_dict['xyval2'].append(xyval2)

                pairdata_dict['aver_rate1'].append(aver_rate1)
                pairdata_dict['aver_rate2'].append(aver_rate2)
                pairdata_dict['aver_rate_pair'].append(aver_rate_pair)
                pairdata_dict['peak_rate1'].append(peak_rate1)
                pairdata_dict['peak_rate2'].append(peak_rate2)
                pairdata_dict['minocc1'].append(minocc1)
                pairdata_dict['minocc2'].append(minocc2)
                pairdata_dict['fieldcoor1'].append(fcoor1)
                pairdata_dict['fieldcoor2'].append(fcoor2)
                pairdata_dict['com1'].append(com1)
                pairdata_dict['com2'].append(com2)

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
                pairdata_dict['thetaT_AB'].append(thetaT_AB)
                pairdata_dict['thetaT_BA'].append(thetaT_BA)

                pairdata_dict['rate_AB'].append(rate_AB)
                pairdata_dict['rate_BA'].append(rate_BA)
                pairdata_dict['corate'].append(corate)
                pairdata_dict['pair_rate'].append(pair_rate)
                pairdata_dict['kld'].append(kld)
                pairdata_dict['rate_R_pvalp'].append(rate_R_pvalp)

                pairdata_dict['precess_df1'].append(fitted_precessdf1)
                pairdata_dict['precess_angle1'].append(precessangle1)
                pairdata_dict['precess_R1'].append(precessR1)
                pairdata_dict['numpass_at_precess1'].append(numpass_at_precess1)
                pairdata_dict['precess_df2'].append(fitted_precessdf2)
                pairdata_dict['precess_angle2'].append(precessangle2)
                pairdata_dict['precess_R2'].append(precessR2)
                pairdata_dict['numpass_at_precess2'].append(numpass_at_precess2)
                pairdata_dict['precess_dfp'].append(fitted_precess_dfp)

                pair_id += 1


    pairdata = pd.DataFrame(pairdata_dict)
    pairdata = append_extrinsicity(pairdata)
    pairdata.to_pickle(save_pth)
    return pairdata



def plot_pair_examples(df, vthresh=5, sthresh=3, plot_dir=None):


    def select_cases(ntrial, ca, npair):

        if (ntrial==0) and (ca== 'CA1') and (npair==9):
            return 'kld', 1
        elif (ntrial==7) and (ca== 'CA1') and (npair==22):
            return 'kld', 0

        elif (ntrial==123) and (ca== 'CA1') and (npair==0):  # 628
            return 'eg', 14
        elif (ntrial==9) and (ca== 'CA1') and (npair==1):  # 140
            return 'eg', 4
        elif (ntrial==56) and (ca== 'CA1') and (npair==1):  # 334
            return 'eg', 8
        elif (ntrial==73) and (ca== 'CA1') and (npair==2):  # 394
            return 'eg', 12
        elif (ntrial==17) and (ca== 'CA2') and (npair==4):  # 256
            return 'eg', 0
        elif (ntrial==18) and (ca== 'CA2') and (npair==4):  # 263
            return 'eg', 6
        elif (ntrial==26) and (ca== 'CA3') and (npair==1):  # 299
            return 'eg', 10
        elif (ntrial==21) and (ca== 'CA3') and (npair==2):  # 283
            return 'eg', 2
        else:
            return None, None

    all_ntrials = [0, 7, 123, 9, 56, 73, 17, 18, 26, 21]


    # Paired spikes
    figw_pairedsp = total_figw*0.8
    figh_pairedsp = figw_pairedsp/5


    # Pair eg
    figw_paireg = total_figw*0.8
    figh_paireg = figw_paireg/4 * 1.1
    fig_paireg = plt.figure(figsize=(figw_paireg, figh_paireg))
    ax_paireg = np.array([
        fig_paireg.add_subplot(2, 8, 1), fig_paireg.add_subplot(2, 8, 2, polar=True),
         fig_paireg.add_subplot(2, 8, 3), fig_paireg.add_subplot(2, 8, 4, polar=True),
         fig_paireg.add_subplot(2, 8, 5), fig_paireg.add_subplot(2, 8, 6, polar=True),
         fig_paireg.add_subplot(2, 8, 7), fig_paireg.add_subplot(2, 8, 8, polar=True),
         fig_paireg.add_subplot(2, 8, 9), fig_paireg.add_subplot(2, 8, 10, polar=True),
         fig_paireg.add_subplot(2, 8, 11), fig_paireg.add_subplot(2, 8, 12, polar=True),
         fig_paireg.add_subplot(2, 8, 13), fig_paireg.add_subplot(2, 8, 14, polar=True),
         fig_paireg.add_subplot(2, 8, 15), fig_paireg.add_subplot(2, 8, 16, polar=True),
    ])

    # KLD
    figw_kld = total_figw*0.9/2  # leave 0.2 for colorbar in fig 5
    figh_kld = total_figw*0.9/4
    fig_kld, ax_kld = plt.subplots(2, 4, figsize=(figw_kld, figh_kld), subplot_kw={'polar':True})


    num_trials = df.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    aedm = midedges(aedges)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5
    precess_filter = PrecessionFilter()
    for ntrial in range(num_trials):
        if ntrial not in all_ntrials:
            continue
        wave = df.loc[ntrial, 'wave']
        precesser = PrecessionProcesser(sthresh=sthresh, vthresh=vthresh, wave=wave)


        for ca in ['CA%d' % (i + 1) for i in range(3)]:

            # Get data
            pair_df = df.loc[ntrial, ca + 'pairs']
            field_df = df.loc[ntrial, ca + 'fields']
            indata = df.loc[ntrial, ca + 'indata']
            if (indata.shape[0] < 1) & (pair_df.shape[0] < 1) & (field_df.shape[0] < 1):
                continue

            tunner = IndataProcessor(indata, vthresh=vthresh, smooth=True)
            interpolater_angle = interp1d(tunner.t, tunner.angle)
            interpolater_x = interp1d(tunner.t, tunner.x)
            interpolater_y = interp1d(tunner.t, tunner.y)
            all_maxt, all_mint = tunner.t.max(), tunner.t.min()
            trange = (all_maxt, all_mint)
            dt = tunner.t[1] - tunner.t[0]
            precesser.set_trange(trange)

            ##  Loop for pairs
            num_pairs = pair_df.shape[0]
            for npair in range(num_pairs):

                case, case_axid = select_cases(ntrial, ca, npair)
                if case is None:
                    continue

                print('trial %d, %s, pair %d, case=%s' % (ntrial, ca, npair, case))

                # find within-mask indexes
                field_ids = pair_df.loc[npair, 'fi'] - 1  # minus 1 to convert to python index
                mask1 = field_df.loc[field_ids[0], 'mask']
                mask2 = field_df.loc[field_ids[1], 'mask']

                # field's boundaries
                xyval1 = field_df.loc[field_ids[0], 'xyval']
                xyval2 = field_df.loc[field_ids[1], 'xyval']

                # Find overlap
                pf1 = field_df.loc[field_ids[0], 'pf']
                pf2 = field_df.loc[field_ids[1], 'pf']
                _, ks_dist, _ = dist_overlap(pf1['map'], pf2['map'], mask1, mask2)

                # Field's center coordinates
                maskedmap1, maskedmap2 = pf1['map'] * mask1, pf2['map'] * mask2
                cooridx1 = np.unravel_index(maskedmap1.argmax(), maskedmap1.shape)
                cooridx2 = np.unravel_index(maskedmap2.argmax(), maskedmap2.shape)
                fcoor1 = np.array([pf1['X'][cooridx1[0], cooridx1[1]], pf1['Y'][cooridx1[0], cooridx1[1]]])
                fcoor2 = np.array([pf2['X'][cooridx2[0], cooridx2[1]], pf2['Y'][cooridx2[0], cooridx2[1]]])


                # get single fields' statistics
                passdf1 = field_df.loc[field_ids[0], 'passes']
                passdf2 = field_df.loc[field_ids[1], 'passes']

                (x1list, y1list, t1list, angle1list), tsp1list = append_info_from_passes(passdf1, vthresh, sthresh,
                                                                                         trange)
                (x2list, y2list, t2list, angle2list), tsp2list = append_info_from_passes(passdf2, vthresh, sthresh,
                                                                                         trange)


                if (len(t1list) < 1) or (len(t2list) < 1) or (len(tsp1list) < 1) or (len(tsp2list) < 1):
                    continue

                x1, x2 = np.concatenate(x1list), np.concatenate(x2list)
                y1, y2 = np.concatenate(y1list), np.concatenate(y2list)
                hd1, hd2 = np.concatenate(angle1list), np.concatenate(angle2list)
                pos1, pos2 = np.stack([x1, y1]).T, np.stack([x2, y2]).T

                tsp1, tsp2 = np.concatenate(tsp1list), np.concatenate(tsp2list)
                xsp1, xsp2 = interpolater_x(tsp1), interpolater_x(tsp2)
                ysp1, ysp2 = interpolater_y(tsp1), interpolater_y(tsp2)
                possp1, possp2 = np.stack([xsp1, ysp1]).T, np.stack([xsp2, ysp2]).T
                hdsp1, hdsp2 = interpolater_angle(tsp1), interpolater_angle(tsp2)
                nspks1, nspks2 = tsp1.shape[0], tsp2.shape[0]


                # Directionality
                biner1 = DirectionerBining(aedges, hd1)
                fangle1, fR1, (spbins1, occbins1, normprob1) = biner1.get_directionality(hdsp1)
                mlmer1 = DirectionerMLM(pos1, hd1, dt=dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
                fangle1_mlm, fR1_mlm, normprob1_mlm = mlmer1.get_directionality(possp1, hdsp1)
                normprob1_mlm[np.isnan(normprob1_mlm)] = 0
                biner2 = DirectionerBining(aedges, hd2)
                fangle2, fR2, (spbins2, occbins2, normprob2) = biner2.get_directionality(hdsp2)
                mlmer2 = DirectionerMLM(pos2, hd2, dt=dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
                fangle2_mlm, fR2_mlm, normprob2_mlm = mlmer2.get_directionality(possp2, hdsp2)
                normprob2_mlm[np.isnan(normprob2_mlm)] = 0

                # Pass-based phaselag, extrinsic/intrinsic
                phase_finder = ThetaEstimator(0.005, 0.3, [5, 12])
                AB_tsp1_list, BA_tsp1_list = [], []
                AB_tsp2_list, BA_tsp2_list = [], []
                nspikes_AB_list, nspikes_BA_list = [], []
                duration_AB_list, duration_BA_list = [], []
                t_all = []
                passangles_all, x_all, y_all = [], [], []
                paired_tsp_list = []

                # Precession per pair
                neuro_keys_dict1 = dict(tsp='tsp1', spikev='spike1v', spikex='spike1x', spikey='spike1y',
                                        spikeangle='spike1angle')
                neuro_keys_dict2 = dict(tsp='tsp2', spikev='spike2v', spikex='spike2x', spikey='spike2y',
                                        spikeangle='spike2angle')

                pass_dict_keys = precesser._gen_precess_infokeys()
                pass_dict1 = precesser.gen_precess_dict(tag='1')
                pass_dict2 = precesser.gen_precess_dict(tag='2')
                pass_dictp = {**pass_dict1, **pass_dict2, **{'direction':[]}}


                pairedpasses = pair_df.loc[npair, 'pairedpasses']
                num_passes = pairedpasses.shape[0]



                for npass in range(num_passes):

                    # Get spikes
                    tsp1, tsp2, vsp1, vsp2 = pairedpasses.loc[npass, ['tsp1', 'tsp2', 'spike1v', 'spike2v']]
                    x, y, pass_angles, v = pairedpasses.loc[npass, ['x', 'y', 'angle', 'v']]

                    # Straightness
                    straightrank = compute_straightness(pass_angles)
                    if straightrank < sthresh:
                        continue

                    # Find direction
                    infield1, infield2, t = pairedpasses.loc[npass, ['infield1', 'infield2', 't']]
                    loc, direction = ThetaEstimator.find_direction(infield1, infield2)
                    duration = t.max() - t.min()

                    # Speed threshold
                    passmask = v > vthresh
                    spmask1, spmask2 = vsp1 > vthresh, vsp2 > vthresh
                    xinv, yinv, pass_angles_inv, tinv = x[passmask], y[passmask], pass_angles[passmask], t[passmask]
                    tsp1_inv, tsp2_inv = tsp1[spmask1], tsp2[spmask2]

                    # Find paired spikes
                    pairidx1, pairidx2 = find_pair_times(tsp1_inv, tsp2_inv)
                    paired_tsp1, paired_tsp2 = tsp1_inv[pairidx1], tsp2_inv[pairidx2]
                    if (paired_tsp1.shape[0] < 1) and (paired_tsp2.shape[0] < 1):
                        continue
                    paired_tsp_eachpass = np.concatenate([paired_tsp1, paired_tsp2])
                    paired_tsp_list.append(paired_tsp_eachpass)

                    # Get pass info
                    passangles_all.append(pass_angles_inv)
                    x_all.append(xinv)
                    y_all.append(yinv)
                    t_all.append(tinv)

                    if direction == 'A->B':
                        AB_tsp1_list.append(tsp1_inv)
                        AB_tsp2_list.append(tsp2_inv)
                        nspikes_AB_list.append(tsp1_inv.shape[0] + tsp2_inv.shape[0])
                        duration_AB_list.append(duration)

                    elif direction == 'B->A':
                        BA_tsp1_list.append(tsp1_inv)
                        BA_tsp2_list.append(tsp2_inv)
                        nspikes_BA_list.append(tsp1_inv.shape[0] + tsp2_inv.shape[0])
                        duration_BA_list.append(duration)

                    if (direction == 'A->B') or (direction == 'B->A'):
                        precess1 = precesser._get_precession(pairedpasses, npass, neuro_keys_dict1)
                        precess2 = precesser._get_precession(pairedpasses, npass, neuro_keys_dict2)
                        if (precess1 is None) or (precess2 is None):
                            continue
                        else:
                            pass_dictp = precesser.append_pass_dict(pass_dictp, precess1, tag='1')
                            pass_dictp = precesser.append_pass_dict(pass_dictp, precess2, tag='2')
                            pass_dictp['direction'].append(direction)



                    ############## Plot paired spikes examples ##############
                    if (ntrial==26) and (ca=='CA3') and (npair==1) and (npass==10):

                        if (tsp1_inv.shape[0] != 0) or (tsp2_inv.shape[0] != 0):
                            mintsp_plt = np.min(np.concatenate([tsp1_inv, tsp2_inv]))
                            tsp1_inv = tsp1_inv - mintsp_plt
                            tsp2_inv = tsp2_inv - mintsp_plt
                            tmp_idx1, tmp_idx2 = find_pair_times(tsp1_inv, tsp2_inv)
                            pairedsp1, pairedsp2 = tsp1_inv[tmp_idx1], tsp2_inv[tmp_idx2]

                            fig_pairsp, ax_pairsp = plt.subplots(figsize=(figw_pairedsp, figh_pairedsp))
                            ax_pairsp.eventplot(tsp1_inv, color='k', lineoffsets=0, linelengths=1, linewidths=0.75)
                            ax_pairsp.eventplot(tsp2_inv, color='k', lineoffsets=1, linelengths=1, linewidths=0.75)
                            ax_pairsp.eventplot(pairedsp1, color='darkorange', lineoffsets=0, linelengths=1, linewidths=0.75)
                            ax_pairsp.eventplot(pairedsp2, color='darkorange', lineoffsets=1, linelengths=1, linewidths=0.75)
                            ax_pairsp.set_yticks([0, 1])
                            ax_pairsp.set_yticklabels(['Field A', 'Field B'])
                            ax_pairsp.set_ylim(-0.7, 1.7)
                            ax_pairsp.tick_params(labelsize=ticksize)
                            ax_pairsp.set_xlabel('t (s)', fontsize=fontsize)
                            ax_pairsp.xaxis.set_label_coords(1, -0.075)
                            ax_pairsp.spines['left'].set_visible(False)
                            ax_pairsp.spines['right'].set_visible(False)
                            ax_pairsp.spines['top'].set_visible(False)

                            fig_pairsp.tight_layout()
                            fig_pairsp.savefig(os.path.join(plot_dir, 'example_pairedspikes.%s' % (figext)), dpi=dpi)
                            # fig_pairsp.savefig(os.path.join(plot_dir, 'examples_pairspikes', 'trial-%d_%s_pair-%d_pass-%d.%s' % (ntrial, ca, npair, npass, figext)), dpi=dpi)


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

                # Pair Directionality
                biner_pair = DirectionerBining(aedges, hd_pair)
                fanglep, fRp, (spbinsp, occbinsp, normprobp) = biner_pair.get_directionality(hdsp_pair)
                mlmer_pair = DirectionerMLM(pos_pair, hd_pair, dt, sp_binwidth, abind)
                fanglep_mlm, fRp_mlm, normprobp_mlm = mlmer_pair.get_directionality(possp_pair, hdsp_pair)
                normprobp_mlm[np.isnan(normprobp_mlm)] = 0

                # KLD
                kld_mlm = calc_kld(normprob1_mlm, normprob2_mlm, normprobp_mlm)

                ############## Plot pair examples ##############
                if case == 'eg':
                    ax_paireg[case_axid].plot(tunner.x, tunner.y, c='0.8', linewidth=0.75)
                    ax_paireg[case_axid].plot(xyval1[:, 0], xyval1[:, 1], c='k', linewidth=1)
                    ax_paireg[case_axid].plot(xyval2[:, 0], xyval2[:, 1], c='k', linewidth=1)

                    ax_paireg[case_axid].axis('off')
                    xp, yp = circular_density_1d(aedm, 20 * np.pi, 60, (-np.pi, np.pi), w=normprobp_mlm)
                    ax_paireg[case_axid+1] = directionality_polar_plot(ax_paireg[case_axid+1], xp, yp, fanglep_mlm, linewidth=0.75)

                    ax_paireg[case_axid].text(0.6, -0.05, '%0.2f' % (ks_dist), fontsize=legendsize, c='k', transform=ax_paireg[case_axid].transAxes)
                    ax_paireg[case_axid+1].text(0.6, -0.05, '%0.2f' % (fRp_mlm), fontsize=legendsize, c='k', transform=ax_paireg[case_axid+1].transAxes)

                ############## Plot KLD examples ##############
                if case == 'kld':
                    x1, y1 = circular_density_1d(aedm, 20 * np.pi, 60, (-np.pi, np.pi), w=normprob1_mlm)
                    x2, y2 = circular_density_1d(aedm, 20 * np.pi, 60, (-np.pi, np.pi), w=normprob2_mlm)
                    xp, yp = circular_density_1d(aedm, 20 * np.pi, 60, (-np.pi, np.pi), w=normprobp_mlm)

                    indep_prob = normprob1_mlm * normprob2_mlm
                    indep_prob = indep_prob / np.sum(indep_prob)
                    indep_angle = circmean(aedm, w=indep_prob, d=abind)
                    xindep, yindep = circular_density_1d(aedm, 20 * np.pi, 60, (-np.pi, np.pi), w=indep_prob)

                    kld_linewidth = 0.75
                    ax_kld[case_axid, 0] = directionality_polar_plot(ax_kld[case_axid, 0], x1, y1, fangle1_mlm, linewidth=kld_linewidth)
                    ax_kld[case_axid, 1] = directionality_polar_plot(ax_kld[case_axid, 1], x2, y2, fangle2_mlm, linewidth=kld_linewidth)
                    ax_kld[case_axid, 2] = directionality_polar_plot(ax_kld[case_axid, 2], xindep, yindep, indep_angle, linewidth=kld_linewidth)
                    ax_kld[case_axid, 3] = directionality_polar_plot(ax_kld[case_axid, 3], xp, yp, fanglep_mlm, linewidth=kld_linewidth)


                    kldtext_x, kldtext_y = 0, -0.2
                    ax_kld[case_axid, 3].text(kldtext_x, kldtext_y, 'KLD=%0.2f' % (kld_mlm), fontsize=legendsize, transform=ax_kld[case_axid, 3].transAxes)
    kldtext_y = 1
    ax_kld[0, 0].text(0.2, kldtext_y, '$P(A)$', fontsize=legendsize, transform=ax_kld[0, 0].transAxes)
    ax_kld[0, 1].text(0.2, kldtext_y, '$P(B)$', fontsize=legendsize, transform=ax_kld[0, 1].transAxes)
    ax_kld[0, 2].text(-0.2, kldtext_y, '$P(A)$'+r'$\times$'+'$P(B)$', fontsize=legendsize, transform=ax_kld[0, 2].transAxes)
    ax_kld[0, 3].text(0.15, kldtext_y, r'$P(A \cap B)$', fontsize=legendsize, transform=ax_kld[0, 3].transAxes)



    fig_paireg.tight_layout()
    fig_paireg.subplots_adjust(wspace=0.01, hspace=0.05)
    fig_paireg.savefig(os.path.join(plot_dir, 'examples_pair.%s' % (figext)), dpi=dpi)


    fig_kld.tight_layout()
    fig_kld.subplots_adjust(wspace=0.05, hspace=0.2)
    fig_kld.savefig(os.path.join(plot_dir, 'examples_kld.%s' % (figext)), dpi=dpi)


if __name__ == '__main__':
    # Experiment's data preprocessing
    data_pth = 'data/emankindata_processed_withwave.pickle'
    save_pth = 'results/exp/pair_field/pairfield_df.pickle'
    plot_dir = 'result_plots/pair_fields/'
    expdf = load_pickle(data_pth)

    pair_field_preprocess_exp(expdf, vthresh=5, sthresh=3, NShuffles=200, save_pth=save_pth)
    #
    # # Plotting
    # plot_pair_examples(expdf, vthresh=5, sthresh=3, plot_dir=plot_dir)


    # simdata = load_pickle('results/sim/raw/squarematch.pickle')
    # pair_field_preprocess_sim(simdata, save_pth='results/sim/pair_field/pairfield_df_square2.pickle')

