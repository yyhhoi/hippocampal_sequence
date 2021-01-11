import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pycircstat.tests import watson_williams
from pycircstat.descriptive import mean as circmean
from pycircstat.descriptive import resultant_vector_length
from scipy.stats import vonmises, ranksums
from scipy.interpolate import interp1d

from common.script_wrappers import PrecessionProcesser, PrecessionFilter, PrecessionStat, construct_passdf_sim, \
    construct_pairedpass_df_sim
from common.linear_circular_r import rcc
from common.utils import load_pickle, stat_record
from common.comput_utils import dist_overlap, normsig, append_extrinsicity, linear_circular_gauss_density, find_pair_times, directionality, \
    TunningAnalyzer, check_border, window_shuffle, get_field_directionality, \
    compute_straightness, calc_kld, window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, segment_passes, \
    pair_diff, \
    check_border_sim, midedges, append_info_from_passes, DirectionerMLM, DirectionerBining, circular_density_1d
from common.correlogram import Crosser, ThetaEstimator, Bootstrapper
from common.visualization import plot_correlogram, directionality_polar_plot
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

figext = 'png'
# figext = 'eps'

def pair_field_preprocess_exp(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=None, shuffle=True):
    pairdata_dict = dict(pair_id=[], ca=[], overlap=[],
                         phi0=[], phi1=[], theta0=[],
                         border1=[], border2=[],
                         area1=[], area2=[],
                         aver_rate1=[], aver_rate2=[], aver_rate_pair=[],
                         peak_rate1=[], peak_rate2=[],
                         fieldcoor1=[], fieldcoor2=[],
                         fieldangle1=[], fieldangle2=[], fieldangle_pair=[],
                         fieldangle1_mlm=[], fieldangle2_mlm=[], fieldangle_pair_mlm=[],
                         fieldR1=[], fieldR2=[], fieldR_pair=[], fieldR1_mlm=[], fieldR2_mlm=[], fieldR_pair_mlm=[],
                         num_spikes1=[], num_spikes2=[], num_spikes_pair=[],
                         spike_bins1=[], spike_bins2=[], spike_bins_pair=[],
                         occ_bins1=[], occ_bins2=[], occ_bins_pair=[],
                         phaselag_AB=[], phaselag_BA=[], corr_info_AB=[], corr_info_BA=[],
                         thetaT_AB=[], thetaT_BA=[],
                         rate_AB=[], rate_BA=[], corate=[], pair_rate=[],
                         kld=[], kld_mlm=[], win_pval_pair=[], win_pval_pair_mlm=[],
                         shift_pval_pair=[], shift_pval_pair_mlm=[],
                         precess_df1=[], precess_angle1=[], precess_info1=[],
                         precess_df2=[], precess_angle2=[], precess_info2=[],
                         precess_dfp=[], nonprecess_dfp=[])

    num_trials = df.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    aedm = midedges(aedges)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5
    pair_id = 0
    precess_filter = PrecessionFilter()
    for ntrial in range(num_trials):
        wave = df.loc[ntrial, 'wave']
        precesser = PrecessionProcesser(sthresh=sthresh, vthresh=vthresh, wave=wave)


        for ca in ['CA%d' % (i + 1) for i in range(3)]:

            # Get data
            pair_df = df.loc[ntrial, ca + 'pairs']
            field_df = df.loc[ntrial, ca + 'fields']
            indata = df.loc[ntrial, ca + 'indata']
            if (indata.shape[0] < 1) & (pair_df.shape[0] < 1) & (field_df.shape[0] < 1):
                continue

            tunner = TunningAnalyzer(indata, vthresh=vthresh, smooth=True)
            interpolater_angle = interp1d(tunner.t, tunner.movedir)
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
                pairfit = pair_df.loc[npair, 'pairfit']

                # find within-mask indexes
                field_ids = pair_df.loc[npair, 'fi'] - 1  # minus 1 to convert to python index
                mask1 = field_df.loc[field_ids[0], 'mask']
                mask2 = field_df.loc[field_ids[1], 'mask']
                area1 = mask1.sum()
                area2 = mask2.sum()

                # field's boundaries
                xyval1 = field_df.loc[field_ids[0], 'xyval']
                xyval2 = field_df.loc[field_ids[1], 'xyval']

                # Find overlap
                pf1 = field_df.loc[field_ids[0], 'pf']
                pf2 = field_df.loc[field_ids[1], 'pf']
                metrics_masked, boundary = dist_overlap(pf1['map'], pf2['map'], mask1, mask2)
                _, ks_dist, _ = metrics_masked

                # Field's center coordinates
                maskedmap1, maskedmap2 = pf1['map'] * mask1, pf2['map'] * mask2
                cooridx1 = np.unravel_index(maskedmap1.argmax(), maskedmap1.shape)
                cooridx2 = np.unravel_index(maskedmap2.argmax(), maskedmap2.shape)
                fcoor1 = np.array([pf1['X'][cooridx1[0], cooridx1[1]], pf1['Y'][cooridx1[0], cooridx1[1]]])
                fcoor2 = np.array([pf2['X'][cooridx2[0], cooridx2[1]], pf2['Y'][cooridx2[0], cooridx2[1]]])



                # Border
                border1 = check_border(mask1)
                border2 = check_border(mask2)

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

                # Rates
                aver_rate1 = nspks1 / (x1.shape[0]*dt)
                peak_rate1 = np.max(pf1['map'] * mask1)
                aver_rate2 = nspks2 / (x2.shape[0] * dt)
                peak_rate2 = np.max(pf2['map'] * mask2)

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

                # Precession per pass
                neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                        spikeangle='spikeangle')
                pass_dict1 = precesser.gen_precess_dict()
                for precess_dict1 in precesser.get_single_precession(passdf1, neuro_keys_dict):
                    pass_dict1 = precesser.append_pass_dict(pass_dict1, precess_dict1)
                precess_df1 = pd.DataFrame(pass_dict1)
                precess_df1 = precess_filter.filter_single(precess_df1, minocc=occbins1.min())
                precess_stater1 = PrecessionStat(precess_df1)
                filtered_precessdf1, precessinfo1, precessangle1 = precess_stater1.compute_precession_stats()


                pass_dict2 = {x: [] for x in pass_dict_keys}
                for precess_dict2 in precesser.get_single_precession(passdf2, neuro_keys_dict):
                    pass_dict2 = precesser.append_pass_dict(pass_dict2, precess_dict2)
                precess_df2 = pd.DataFrame(pass_dict2)
                precess_df2 = precess_filter.filter_single(precess_df2, minocc=occbins2.min())
                precess_stater2 = PrecessionStat(precess_df2)
                filtered_precessdf2, precessinfo2, precessangle2 = precess_stater2.compute_precession_stats()


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


                # Phase lags
                thetaT_AB, phaselag_AB, corr_info_AB = phase_finder.find_theta_isi_hilbert(AB_tsp1_list, AB_tsp2_list)
                thetaT_BA, phaselag_BA, corr_info_BA = phase_finder.find_theta_isi_hilbert(BA_tsp1_list, BA_tsp2_list)

                # Pair precession
                precess_dfp = pd.DataFrame(pass_dictp)
                precess_dfp = precess_filter.filter_pair(precess_dfp, minocc1=occbins1.min(), minocc2=occbins2.min())
                filtered_precessdfp = precess_dfp[precess_dfp['precess_exist']].reset_index(drop=True)
                nonprecess_dfp = precess_dfp[~precess_dfp['precess_exist']].reset_index(drop=True)

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
                biner_pair = DirectionerBining(aedges, hd_pair)
                fanglep, fRp, (spbinsp, occbinsp, normprobp) = biner_pair.get_directionality(hdsp_pair)
                mlmer_pair = DirectionerMLM(pos_pair, hd_pair, dt, sp_binwidth, abind)
                fanglep_mlm, fRp_mlm, normprobp_mlm = mlmer_pair.get_directionality(possp_pair, hdsp_pair)
                normprobp_mlm[np.isnan(normprobp_mlm)] = 0

                # Rates
                rate_AB = np.sum(nspikes_AB_list) / np.sum(duration_AB_list)
                rate_BA = np.sum(nspikes_BA_list) / np.sum(duration_BA_list)
                corate = np.sum(nspikes_AB_list + nspikes_BA_list) / np.sum(duration_AB_list + duration_BA_list)
                pair_rate = num_spikes_pair / np.sum(duration_AB_list + duration_BA_list)

                # KLD
                kld = calc_kld(normprob1, normprob2, normprobp)
                kld_mlm = calc_kld(normprob1_mlm, normprob2_mlm, normprobp_mlm)


                if shuffle:
                    # Window shuffling
                    win_pval, win_pval_mlm = window_shuffle_wrapper(paired_tsp, fRp, fRp_mlm, NShuffles, biner_pair,
                                                                    mlmer_pair, interpolater_x, interpolater_y,
                                                                    interpolater_angle, trange)

                    # Time shift shuffling
                    shi_pval, shi_pval_mlm = timeshift_shuffle_exp_wrapper(paired_tsp_list, t_all, fRp, fRp_mlm,
                                                                           NShuffles, biner_pair, mlmer_pair,
                                                                           interpolater_x, interpolater_y,
                                                                           interpolater_angle, trange)
                else:
                    win_pval, win_pval_mlm, shi_pval, shi_pval_mlm = None, None, None, None

                pairdata_dict['pair_id'].append(pair_id)
                pairdata_dict['ca'].append(ca)
                pairdata_dict['overlap'].append(ks_dist)
                pairdata_dict['phi0'].append(pairfit['phi0'])
                pairdata_dict['phi1'].append(pairfit['phi1'])
                pairdata_dict['theta0'].append(pairfit['theta0'])
                pairdata_dict['border1'].append(border1)
                pairdata_dict['border2'].append(border2)
                pairdata_dict['area1'].append(area1)
                pairdata_dict['area2'].append(area2)

                pairdata_dict['aver_rate1'].append(aver_rate1)
                pairdata_dict['aver_rate2'].append(aver_rate2)
                pairdata_dict['aver_rate_pair'].append(aver_rate_pair)
                pairdata_dict['peak_rate1'].append(peak_rate1)
                pairdata_dict['peak_rate2'].append(peak_rate2)

                pairdata_dict['fieldcoor1'].append(fcoor1)
                pairdata_dict['fieldcoor2'].append(fcoor2)

                pairdata_dict['fieldangle1'].append(fangle1)
                pairdata_dict['fieldangle1_mlm'].append(fangle1_mlm)
                pairdata_dict['fieldangle2'].append(fangle2)
                pairdata_dict['fieldangle2_mlm'].append(fangle2_mlm)
                pairdata_dict['fieldangle_pair'].append(fanglep)
                pairdata_dict['fieldangle_pair_mlm'].append(fanglep_mlm)
                pairdata_dict['fieldR1'].append(fR1)
                pairdata_dict['fieldR1_mlm'].append(fR1_mlm)
                pairdata_dict['fieldR2'].append(fR2)
                pairdata_dict['fieldR2_mlm'].append(fR2_mlm)
                pairdata_dict['fieldR_pair'].append(fRp)
                pairdata_dict['fieldR_pair_mlm'].append(fRp_mlm)

                pairdata_dict['num_spikes1'].append(nspks1)
                pairdata_dict['num_spikes2'].append(nspks2)
                pairdata_dict['num_spikes_pair'].append(num_spikes_pair)

                pairdata_dict['spike_bins1'].append(spbins1)
                pairdata_dict['spike_bins2'].append(spbins2)
                pairdata_dict['spike_bins_pair'].append(spbinsp)

                pairdata_dict['occ_bins1'].append(occbins1)
                pairdata_dict['occ_bins2'].append(occbins2)
                pairdata_dict['occ_bins_pair'].append(occbinsp)

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
                pairdata_dict['kld_mlm'].append(kld_mlm)
                pairdata_dict['win_pval_pair'].append(win_pval)
                pairdata_dict['win_pval_pair_mlm'].append(win_pval_mlm)
                pairdata_dict['shift_pval_pair'].append(shi_pval)
                pairdata_dict['shift_pval_pair_mlm'].append(shi_pval_mlm)

                pairdata_dict['precess_df1'].append(filtered_precessdf1)
                pairdata_dict['precess_angle1'].append(precessangle1)
                pairdata_dict['precess_info1'].append(precessinfo1)
                pairdata_dict['precess_df2'].append(filtered_precessdf2)
                pairdata_dict['precess_angle2'].append(precessangle2)
                pairdata_dict['precess_info2'].append(precessinfo2)
                pairdata_dict['precess_dfp'].append(filtered_precessdfp)
                pairdata_dict['nonprecess_dfp'].append(nonprecess_dfp)

                pair_id += 1


    if save_pth:
        pairdata = pd.DataFrame(pairdata_dict)
        pairdata = append_extrinsicity(pairdata)
        pairdata.to_pickle(save_pth)
    return pairdata


def pair_field_preprocess_sim(simdata, save_pth, radius=2, vthresh=2, sthresh=80, subsample_fraction=0.4, NShuffles=200):
    edges = np.linspace(-np.pi, np.pi, 36)

    pairdata_dict = dict(neuron1pos=[], neuron2pos=[], neurondist=[],
                         # overlap is calculated afterward
                         border1=[], border2=[],
                         aver_rate1=[], aver_rate2=[], aver_rate_pair=[],
                         peak_rate1=[], peak_rate2=[],
                         fieldangle1=[], fieldangle2=[], fieldangle_pair=[],
                         fieldangle1_mlm=[], fieldangle2_mlm=[], fieldangle_pair_mlm=[],
                         fieldR1=[], fieldR2=[], fieldR_pair=[], fieldR1_mlm=[], fieldR2_mlm=[], fieldR_pair_mlm=[],
                         num_spikes1=[], num_spikes2=[], num_spikes_pair=[],
                         spike_bins1=[], spike_bins2=[], spike_bins_pair=[],
                         occ_bins1=[], occ_bins2=[], occ_bins_pair=[],
                         phaselag_AB=[], phaselag_BA=[], corr_info_AB=[], corr_info_BA=[],
                         rate_AB=[], rate_BA=[], corate=[], pair_rate=[],
                         kld=[], kld_mlm=[], win_pval_pair=[], win_pval_pair_mlm=[],
                         shift_pval_pair=[], shift_pval_pair_mlm=[],
                         precess_df1=[], precess_angle1=[], precess_info1=[],
                         precess_df2=[], precess_angle2=[], precess_info2=[],
                         precess_dfp=[])

    Indata, SpikeData, NeuronPos = simdata['Indata'], simdata['SpikeData'], simdata['NeuronPos']
    wave = dict(tax=Indata['t'].to_numpy(), phase=Indata['phase'].to_numpy(),
                theta=np.ones(Indata.shape[0]))

    # setting
    minpasstime = 0.6
    minspiketresh = 14
    default_T = 1/10
    aedges = np.linspace(-np.pi, np.pi, 36)
    aedm = midedges(aedges)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 2*np.pi/16
    precesser = PrecessionProcesser(sthresh=sthresh, vthresh=vthresh, wave=wave)
    precess_filter = PrecessionFilter()


    # Precomputation
    tunner = TunningAnalyzer(Indata, vthresh=vthresh, smooth=True)
    interpolater_angle = interp1d(tunner.t, tunner.movedir)
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

            # Condition: Paired spikes exist
            tsp_diff = pair_diff(tunner.t[tidxsp1], tunner.t[tidxsp2])

            spok = np.sum(np.abs(tsp_diff) < default_T)
            if spok < minspiketresh:
                progress = progress + 1
                continue

            # Check border
            border1 = check_border_sim(neu1x, neu1y, radius, (0, 2 * np.pi))
            border2 = check_border_sim(neu1x, neu1y, radius, (0, 2 * np.pi))

            # Segment passes
            all_passidx_1 = segment_passes(tok1)
            all_passidx_2 = segment_passes(tok2)
            all_passidx_union = segment_passes(tok_union)
            all_passidx_pair = filter(lambda x : True if (np.sum(tok_intersect[x[0]:x[1]])>1) else False,
                                      all_passidx_union)



            # Create Pass df's
            passdf1 = construct_passdf_sim(tunner, all_passidx_1, tidxsp1, minpasstime=minpasstime)
            passdf2 = construct_passdf_sim(tunner, all_passidx_2, tidxsp2, minpasstime=minpasstime)
            pairedpasses = construct_pairedpass_df_sim(tunner, all_passidx_pair, tok1, tok2, tidxsp1, tidxsp2,
                                                       minpasstime=minpasstime)

            num_passes = pairedpasses.shape[0]

            (x1list, y1list, t1list, angle1list), tsp1list = append_info_from_passes(passdf1, vthresh, sthresh,
                                                                                     trange)
            (x2list, y2list, t2list, angle2list), tsp2list = append_info_from_passes(passdf2, vthresh, sthresh,
                                                                                     trange)

            if (len(t1list) < 1) or (len(t2list) < 1) or (len(tsp1list) < 1) or (len(tsp2list) < 1):
                progress = progress + 1
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

            # Rates
            aver_rate1 = nspks1 / (x1.shape[0] * dt)
            peak_rate1 = None
            aver_rate2 = nspks2 / (x2.shape[0] * dt)
            peak_rate2 = None

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

            # Precession per pass
            neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                   spikeangle='spikeangle')
            pass_dict1 = precesser.gen_precess_dict()
            for precess_dict1 in precesser.get_single_precession(passdf1, neuro_keys_dict):
                pass_dict1 = precesser.append_pass_dict(pass_dict1, precess_dict1)
            precess_df1 = pd.DataFrame(pass_dict1)
            precess_df1 = precess_filter.filter_single(precess_df1, minocc=occbins1.min())
            precess_stater1 = PrecessionStat(precess_df1)
            filtered_precessdf1, precessinfo1, precessangle1 = precess_stater1.compute_precession_stats()

            pass_dict2 = precesser.gen_precess_dict()
            for precess_dict2 in precesser.get_single_precession(passdf2, neuro_keys_dict):
                pass_dict2 = precesser.append_pass_dict(pass_dict2, precess_dict2)
            precess_df2 = pd.DataFrame(pass_dict2)
            precess_df2 = precess_filter.filter_single(precess_df2, minocc=occbins2.min())
            precess_stater2 = PrecessionStat(precess_df2)
            filtered_precessdf2, precessinfo2, precessangle2 = precess_stater2.compute_precession_stats()

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

            pass_dict1 = precesser.gen_precess_dict(tag='1')
            pass_dict2 = precesser.gen_precess_dict(tag='2')
            pass_dictp = {**pass_dict1, **pass_dict2}


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


            # Phase lags
            _, phaselag_AB, corr_info_AB = phase_finder.find_theta_isi_hilbert(AB_tsp1_list, AB_tsp2_list,
                                                                               default_Ttheta=default_T)
            _, phaselag_BA, corr_info_BA = phase_finder.find_theta_isi_hilbert(BA_tsp1_list, BA_tsp2_list,
                                                                               default_Ttheta=default_T)

            # Pair precession
            precess_dfp = pd.DataFrame(pass_dictp)
            precess_dfp = precess_filter.filter_pair(precess_dfp, minocc1=occbins1.min(), minocc2=occbins2.min())
            filtered_precessdfp = precess_dfp[precess_dfp['precess_exist']].reset_index(drop=True)

            # Paired spikes
            if (len(paired_tsp_list) == 0) or (len(passangles_all) == 0):
                progress = progress + 1
                continue
            hd_pair = np.concatenate(passangles_all)
            x_pair, y_pair = np.concatenate(x_all), np.concatenate(y_all)
            pos_pair = np.stack([x_pair, y_pair]).T

            paired_tsp = np.concatenate(paired_tsp_list)
            paired_tsp = paired_tsp[(paired_tsp <= all_maxt) & (paired_tsp >= all_mint)]
            if paired_tsp.shape[0] < 1:
                progress = progress + 1
                continue
            num_spikes_pair = paired_tsp.shape[0]

            hdsp_pair = interpolater_angle(paired_tsp)
            xsp_pair = interpolater_x(paired_tsp)
            ysp_pair = interpolater_y(paired_tsp)
            possp_pair = np.stack([xsp_pair, ysp_pair]).T
            aver_rate_pair = num_spikes_pair / (x_pair.shape[0] * dt)

            # Pair Directionality
            biner_pair = DirectionerBining(aedges, hd_pair)
            fanglep, fRp, (spbinsp, occbinsp, normprobp) = biner_pair.get_directionality(hdsp_pair)
            mlmer_pair = DirectionerMLM(pos_pair, hd_pair, dt, sp_binwidth, abind)
            fanglep_mlm, fRp_mlm, normprobp_mlm = mlmer_pair.get_directionality(possp_pair, hdsp_pair)
            normprobp_mlm[np.isnan(normprobp_mlm)] = 0

            # Rates
            rate_AB = np.sum(nspikes_AB_list) / np.sum(duration_AB_list)
            rate_BA = np.sum(nspikes_BA_list) / np.sum(duration_BA_list)
            corate = np.sum(nspikes_AB_list + nspikes_BA_list) / np.sum(duration_AB_list + duration_BA_list)
            pair_rate = num_spikes_pair / np.sum(duration_AB_list + duration_BA_list)

            # KLD
            kld = calc_kld(normprob1, normprob2, normprobp)
            kld_mlm = calc_kld(normprob1_mlm, normprob2_mlm, normprobp_mlm)


            # Window shuffling
            win_pval, win_pval_mlm = window_shuffle_wrapper(paired_tsp, fRp, fRp_mlm, NShuffles, biner_pair,
                                                            mlmer_pair, interpolater_x, interpolater_y,
                                                            interpolater_angle, trange)

            # Time shift shuffling
            shi_pval, shi_pval_mlm = timeshift_shuffle_exp_wrapper(paired_tsp_list, t_all, fRp, fRp_mlm,
                                                                   NShuffles, biner_pair, mlmer_pair,
                                                                   interpolater_x, interpolater_y,
                                                                   interpolater_angle, trange)


            pairdata_dict['neuron1pos'].append(NeuronPos.iloc[i].to_numpy())
            pairdata_dict['neuron2pos'].append(NeuronPos.iloc[j].to_numpy())
            pairdata_dict['neurondist'].append(neurondist)

            pairdata_dict['border1'].append(border1)
            pairdata_dict['border2'].append(border2)

            pairdata_dict['aver_rate1'].append(aver_rate1)
            pairdata_dict['aver_rate2'].append(aver_rate2)
            pairdata_dict['aver_rate_pair'].append(aver_rate_pair)
            pairdata_dict['peak_rate1'].append(peak_rate1)
            pairdata_dict['peak_rate2'].append(peak_rate2)

            pairdata_dict['fieldangle1'].append(fangle1)
            pairdata_dict['fieldangle1_mlm'].append(fangle1_mlm)
            pairdata_dict['fieldangle2'].append(fangle2)
            pairdata_dict['fieldangle2_mlm'].append(fangle2_mlm)
            pairdata_dict['fieldangle_pair'].append(fanglep)
            pairdata_dict['fieldangle_pair_mlm'].append(fanglep_mlm)
            pairdata_dict['fieldR1'].append(fR1)
            pairdata_dict['fieldR1_mlm'].append(fR1_mlm)
            pairdata_dict['fieldR2'].append(fR2)
            pairdata_dict['fieldR2_mlm'].append(fR2_mlm)
            pairdata_dict['fieldR_pair'].append(fRp)
            pairdata_dict['fieldR_pair_mlm'].append(fRp_mlm)

            pairdata_dict['num_spikes1'].append(nspks1)
            pairdata_dict['num_spikes2'].append(nspks2)
            pairdata_dict['num_spikes_pair'].append(num_spikes_pair)

            pairdata_dict['spike_bins1'].append(spbins1)
            pairdata_dict['spike_bins2'].append(spbins2)
            pairdata_dict['spike_bins_pair'].append(spbinsp)

            pairdata_dict['occ_bins1'].append(occbins1)
            pairdata_dict['occ_bins2'].append(occbins2)
            pairdata_dict['occ_bins_pair'].append(occbinsp)

            pairdata_dict['phaselag_AB'].append(phaselag_AB)
            pairdata_dict['phaselag_BA'].append(phaselag_BA)
            pairdata_dict['corr_info_AB'].append(corr_info_AB)
            pairdata_dict['corr_info_BA'].append(corr_info_BA)

            pairdata_dict['rate_AB'].append(rate_AB)
            pairdata_dict['rate_BA'].append(rate_BA)
            pairdata_dict['corate'].append(corate)
            pairdata_dict['pair_rate'].append(pair_rate)
            pairdata_dict['kld'].append(kld)
            pairdata_dict['kld_mlm'].append(kld_mlm)
            pairdata_dict['win_pval_pair'].append(win_pval)
            pairdata_dict['win_pval_pair_mlm'].append(win_pval_mlm)
            pairdata_dict['shift_pval_pair'].append(shi_pval)
            pairdata_dict['shift_pval_pair_mlm'].append(shi_pval_mlm)

            pairdata_dict['precess_df1'].append(filtered_precessdf1)
            pairdata_dict['precess_angle1'].append(precessangle1)
            pairdata_dict['precess_info1'].append(precessinfo1)
            pairdata_dict['precess_df2'].append(filtered_precessdf2)
            pairdata_dict['precess_angle2'].append(precessangle2)
            pairdata_dict['precess_info2'].append(precessinfo2)
            pairdata_dict['precess_dfp'].append(filtered_precessdfp)
            progress = progress + 1

    pairdata = pd.DataFrame(pairdata_dict)

    # Convert distance to overlap
    dist_range = pairdata['neurondist'].max() - pairdata['neurondist'].min()
    pairdata['overlap'] = (pairdata['neurondist'].max() - pairdata['neurondist']) / dist_range

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

            tunner = TunningAnalyzer(indata, vthresh=vthresh, smooth=True)
            interpolater_angle = interp1d(tunner.t, tunner.movedir)
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
                metrics_masked, boundary = dist_overlap(pf1['map'], pf2['map'], mask1, mask2)
                _, ks_dist, _ = metrics_masked

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
    save_pth = 'results/exp/pair_field/pairfield_df_latest.pickle'
    plot_dir = 'result_plots/pair_fields/'
    expdf = load_pickle(data_pth)

    # pair_field_preprocess_exp(expdf, vthresh=5, sthresh=3, NShuffles=200, save_pth=save_pth, shuffle=True)

    # # Plotting
    plot_pair_examples(expdf, vthresh=5, sthresh=3, plot_dir=plot_dir)


    # simdata = load_pickle('results/sim/raw/squarematch.pickle')
    # pair_field_preprocess_sim(simdata, save_pth='results/sim/pair_field/pairfield_df_square.pickle')

