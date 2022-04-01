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

from common.correlogram import ThetaEstimator
from common.utils import load_pickle
from common.comput_utils import check_border, IndataProcessor, midedges, segment_passes, \
    check_border_sim, append_info_from_passes, \
    DirectionerBining, DirectionerMLM, timeshift_shuffle_exp_wrapper, circular_density_1d, get_numpass_at_angle, \
    PassShuffler, dist_overlap, find_pair_times, calc_kld, append_extrinsicity

from common.visualization import color_wheel, directionality_polar_plot, customlegend

from common.script_wrappers import DirectionalityStatsByThresh, PrecessionProcesser, PrecessionFilter, \
    construct_passdf_sim, get_single_precessdf, compute_precessangle
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw

figext = 'png'
# figext = 'eps'

def determine_sthresh_hc3(df):
    print('Computing straightrank of all passes')
    num_sess = df.shape[0]

    capassdf = {'CA1': [], 'CA3': []}
    for nsess in range(num_sess):
        if (nsess > 20) :
            break
        print('\r%d/%d Session'%(nsess, num_sess), flush=True, end='')
        posdf = df.loc[nsess, 'posdf']
        occ_dict = df.loc[nsess, 'occ_dict']
        xx, yy = occ_dict['xx'], occ_dict['yy']
        x_ax, y_ax = xx[0, :], yy[:, 0]
        tunner = IndataProcessor(posdf, vthresh=5, sthresh=0, minpasstime=0.6)
        trange = (tunner.t.max(), tunner.t.min())
        dt = tunner.t[1] - tunner.t[0]
        interpolater_angle = interp1d(tunner.t, tunner.angle)
        interpolater_x = interp1d(tunner.t, tunner.x)
        interpolater_y = interp1d(tunner.t, tunner.y)

        for ca in ['CA1', 'CA3']:
            field_df, spdf = df.loc[nsess, ['%sfields' % (ca), '%s_spdf' % (ca)]]
            num_fields = field_df.shape[0]

            for nfield in range(num_fields):
                cellid, mask = field_df.loc[nfield, ['cellid', 'mask']]
                tsp, pf = spdf.loc[cellid, ['tsp', 'pf']]
                tok, idin = tunner.get_idin(mask.T, x_ax, y_ax)
                passdf = tunner.construct_singlefield_passdf(tok, tsp, interpolater_x, interpolater_y,
                                                             interpolater_angle)


                duration = passdf['t'].apply(lambda x : x.max()-x.min())
                nspks = passdf['tsp'].apply(lambda x : x.shape[0])
                passdf = passdf[(duration > 0.4) & (nspks > 1)].reset_index(drop=True)
                capassdf[ca].append(passdf)

    passdf_ca1 = pd.concat(capassdf['CA1'], axis=0, ignore_index=True)
    passdf_ca3 = pd.concat(capassdf['CA3'], axis=0, ignore_index=True)

    print('\nCA1\n', passdf_ca1['straightrank'].describe())
    print('0.1 quantile = %0.2f'%(np.quantile(passdf_ca1['straightrank'].to_numpy(), 0.1)))
    print('CA3\n', passdf_ca3['straightrank'].describe())
    print('0.1 quantile = %0.2f'%(np.quantile(passdf_ca3['straightrank'].to_numpy(), 0.1)))
    return



def single_field_preprocess_hc3(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=None):
    # Checklist
    # straightrank 3 is ok
    # low-spike passes
    # Border detection 2 is ok
    # tok, x_ax, y_ax

    fielddf_dict = dict(ca=[], num_spikes=[], border=[], aver_rate=[], peak_rate=[],
                        rate_angle=[], rate_R=[], rate_R_pval=[], field_area=[], field_bound=[],
                        precess_df=[], precess_angle=[], precess_angle_low=[], precess_R=[], precess_R_pval=[],
                        numpass_at_precess=[], numpass_at_precess_low=[],
                        ratemap=[], spinfo=[], spinfo_pval=[])

    num_sess = df.shape[0]

    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5

    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = 1
    precess_filter = PrecessionFilter()
    nspikes_stats = {'CA1': 6, 'CA3': 7}  # 25% quantile for 0.4t
    for nsess in range(num_sess):

        posdf = df.loc[nsess, 'posdf']
        occ_dict = df.loc[nsess, 'occ_dict']
        xx, yy = occ_dict['xx'], occ_dict['yy']
        x_ax, y_ax = xx[0, :], yy[:, 0]
        tunner = IndataProcessor(posdf, vthresh=vthresh, sthresh=sthresh, minpasstime=0.8)
        trange = (tunner.t.max(), tunner.t.min())
        dt = tunner.t[1] - tunner.t[0]
        interpolater_angle = interp1d(tunner.t, tunner.angle)
        interpolater_x = interp1d(tunner.t, tunner.x)
        interpolater_y = interp1d(tunner.t, tunner.y)



        for ca in ['CA1', 'CA3']:

            # Get and shift wave
            wave = df.loc[nsess, 'wave']
            if ca == 'CA3':
                wave['phase'] = np.mod(wave['phase'] + np.pi + np.deg2rad(200), 2*np.pi) - np.pi
            wave = {col:np.array(wave[col]) for col in wave.columns}
            precesser = PrecessionProcesser(wave=wave)
            precesser.set_trange(trange)

            # Get data
            field_df, spdf = df.loc[nsess, ['%sfields' % (ca), '%s_spdf' % (ca)]]
            num_fields = field_df.shape[0]

            for nfield in range(num_fields):
                print('%d/%d trial, %s, %d/%d field' % (nsess, num_sess, ca, nfield, num_fields))
                cellid, mask, xyval = field_df.loc[nfield, ['cellid', 'mask', 'xyval']]
                tsp, pf, spinfo, spinfo_p = spdf.loc[cellid, ['tsp', 'pf', 'spatialinfo_perspike', 'spatialinfo_pval']]

                if spinfo < 0.5:
                    continue

                # Get field info
                field_area = np.sum(mask)
                field_d = np.sqrt(field_area / np.pi) * 2
                border = check_border(mask, margin=2)

                # Construct passes (segment & chunk)
                tok, idin = tunner.get_idin(mask.T, x_ax, y_ax)
                passdf = tunner.construct_singlefield_passdf(tok, tsp, interpolater_x, interpolater_y,
                                                             interpolater_angle)

                allchunk_df = passdf[(~passdf['rejected']) & (passdf['chunked'] < 2)].reset_index(drop=True)

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
                xsp, ysp = np.concatenate(allchunk_df['spikex'].to_list()), np.concatenate(
                    allchunk_df['spikey'].to_list())
                pos = np.stack([all_x, all_y]).T
                possp = np.stack([xsp, ysp]).T

                # Average firing rate
                aver_rate = all_tsp.shape[0] / (all_x.shape[0] * dt)
                peak_rate = np.max(pf['rate'] * mask)

                # Field's directionality - need angle, anglesp, pos
                num_spikes = all_tsp.shape[0]
                occ_bins, _ = np.histogram(all_passangles, bins=aedges)
                mlmer = DirectionerMLM(pos, all_passangles, dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
                rate_angle, rate_R, norm_prob_mlm = mlmer.get_directionality(possp, all_anglesp)

                # Time shift shuffling for rate directionality
                rate_R_pval = timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, rate_R,
                                                            NShuffles, mlmer, interpolater_x,
                                                            interpolater_y, interpolater_angle, trange)

                # Precession per pass
                neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                                       spikeangle='spikeangle')
                accept_mask = (~passdf['rejected']) & (passdf['chunked'] < 2)
                passdf['excluded_for_precess'] = ~accept_mask
                precessdf, precess_angle, precess_R, _ = get_single_precessdf(passdf, precesser, precess_filter,
                                                                              neuro_keys_dict,
                                                                              field_d=field_d, kappa=kappa_precess,
                                                                              bins=None)
                fitted_precessdf = precessdf[precessdf['fitted']].reset_index(drop=True)
                # Proceed only if precession exists
                if (precess_angle is not None) and (fitted_precessdf['precess_exist'].sum() > 0):

                    # Post-hoc precession exclusion
                    _, binR, postdoc_dens = compute_precessangle(
                        pass_angles=fitted_precessdf['mean_anglesp'].to_numpy(),
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
                        numpass_at_precess_low = get_numpass_at_angle(target_angle=precess_angle_low,
                                                                      aedge=aedges_precess,
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
                fielddf_dict['precess_df'].append(fitted_precessdf)
                fielddf_dict['precess_angle'].append(precess_angle)
                fielddf_dict['precess_angle_low'].append(precess_angle_low)
                fielddf_dict['precess_R'].append(precess_R)
                fielddf_dict['precess_R_pval'].append(precess_R_pval)
                fielddf_dict['numpass_at_precess'].append(numpass_at_precess)
                fielddf_dict['numpass_at_precess_low'].append(numpass_at_precess_low)
                fielddf_dict['ratemap'].append(pf['rate'])
                fielddf_dict['spinfo'].append(spinfo)
                fielddf_dict['spinfo_pval'].append(spinfo_p)

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



def pair_field_preprocess_hc3(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=None):
    pairdata_dict = dict(pair_id=[], ca=[], overlap=[],
                         border1=[], border2=[], area1=[], area2=[], xyval1=[], xyval2=[],
                         aver_rate1=[], aver_rate2=[], aver_rate_pair=[],
                         peak_rate1=[], peak_rate2=[],
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



    num_sess = df.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5

    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = 1
    precess_filter = PrecessionFilter()

    pair_id = 0
    for nsess in range(num_sess):
        posdf = df.loc[nsess, 'posdf']
        occ_dict = df.loc[nsess, 'occ_dict']
        xx, yy = occ_dict['xx'], occ_dict['yy']
        x_ax, y_ax = xx[0, :], yy[:, 0]
        tunner = IndataProcessor(posdf, vthresh=vthresh, sthresh=sthresh, minpasstime=0.6)
        all_maxt, all_mint = tunner.t.max(), tunner.t.min()
        trange = (all_maxt, all_mint)
        dt = tunner.t[1] - tunner.t[0]
        interpolater_angle = interp1d(tunner.t, tunner.angle)
        interpolater_x = interp1d(tunner.t, tunner.x)
        interpolater_y = interp1d(tunner.t, tunner.y)

        wave = df.loc[nsess, 'wave']
        wave = {col:np.array(wave[col]) for col in wave.columns}
        precesser = PrecessionProcesser(wave=wave)
        precesser.set_trange(trange)



        for ca in ['CA1', 'CA3']:

            # Get and shift wave
            wave = df.loc[nsess, 'wave']
            if ca == 'CA3':
                wave['phase'] = np.mod(wave['phase'] + np.pi + np.deg2rad(200), 2*np.pi) - np.pi
            wave = {col:np.array(wave[col]) for col in wave.columns}
            precesser = PrecessionProcesser(wave=wave)
            precesser.set_trange(trange)

            # Get data
            pair_df = df.loc[nsess, ca + 'pairs']
            field_df = df.loc[nsess, ca + 'fields']
            cell_df = df.loc[nsess, ca + '_spdf']

            ##  Loop for pairs
            num_pairs = pair_df.shape[0]
            for npair in range(num_pairs):

                print('%d/%d trial, %s, %d/%d pair id=%d' % (nsess, num_sess, ca, npair, num_pairs, pair_id))

                # find within-mask indexes
                field1_id, field2_id = pair_df.loc[npair, ['field1_id', 'field2_id']]

                mask1, cellid1, xyval1 = field_df.loc[field1_id, ['mask', 'cellid', 'xyval']]
                mask2, cellid2, xyval2 = field_df.loc[field2_id, ['mask', 'cellid', 'xyval']]

                pf1, tsp1, spinfo1 = cell_df.loc[cellid1, ['pf', 'tsp', 'spatialinfo_perspike']]
                pf2, tsp2, spinfo2 = cell_df.loc[cellid2, ['pf', 'tsp', 'spatialinfo_perspike']]
                if (spinfo1 < 0.5) or (spinfo2 < 0.5):
                    continue
                ratemap1, ratemap2 = pf1['rate'], pf2['rate']
                area1 = mask1.sum()
                area2 = mask2.sum()
                field_d1 = np.sqrt(area1/np.pi)*2
                field_d2 = np.sqrt(area2/np.pi)*2


                # Find overlap
                _, ks_dist, _ = dist_overlap(ratemap1, ratemap2, mask1, mask2)

                # Field's center coordinates
                maskedmap1, maskedmap2 = ratemap1 * mask1, ratemap2 * mask2
                cooridx1 = np.unravel_index(maskedmap1.argmax(), maskedmap1.shape)
                cooridx2 = np.unravel_index(maskedmap2.argmax(), maskedmap2.shape)
                fcoor1 = np.array([xx[cooridx1[0], cooridx1[1]], yy[cooridx1[0], cooridx1[1]]])
                fcoor2 = np.array([xx[cooridx2[0], cooridx2[1]], yy[cooridx2[0], cooridx2[1]]])
                XY1 = np.stack([xx.ravel(), yy.ravel()])
                XY2 = np.stack([xx.ravel(), yy.ravel()])
                com1 = np.sum(XY1 * (maskedmap1/np.sum(maskedmap1)).ravel().reshape(1, -1), axis=1)
                com2 = np.sum(XY2 * (maskedmap2/np.sum(maskedmap2)).ravel().reshape(1, -1), axis=1)

                # Border
                border1 = check_border(mask1, margin=2)
                border2 = check_border(mask2, margin=2)

                # Construct passes (segment & chunk) for field 1 and 2
                tok1, idin1 = tunner.get_idin(mask1.T, x_ax, y_ax)
                tok2, idin2 = tunner.get_idin(mask2.T, x_ax, y_ax)
                passdf1 = tunner.construct_singlefield_passdf(tok1, tsp1, interpolater_x, interpolater_y, interpolater_angle)
                passdf2 = tunner.construct_singlefield_passdf(tok2, tsp2, interpolater_x, interpolater_y, interpolater_angle)
                allchunk_df1 = passdf1[(~passdf1['rejected']) & (passdf1['chunked']<2)].reset_index(drop=True)
                allchunk_df2 = passdf2[(~passdf2['rejected']) & (passdf2['chunked']<2)].reset_index(drop=True)
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
                peak_rate1 = np.max(ratemap1 * mask1)
                aver_rate2 = nspks2 / (x2.shape[0] * dt)
                peak_rate2 = np.max(ratemap2 * mask2)

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
                precessdf2, precessangle2, precessR2, _ = get_single_precessdf(passdf2, precesser, precess_filter, neuro_keys_dict,
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
                tok_pair, _ = tunner.get_idin(mask_union.T, x_ax, y_ax)
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





def main():
    # # -------------------------- Single field ---------------------
    save_pth = 'results/hc3/single_field_useAllSessions.pickle'
    df = pd.read_pickle('data/crcns-hc-3/hc3processed_useAllSessions.pickle')
    single_field_preprocess_hc3(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=save_pth)

    # # -------------------------- Pair field ------------------------
    save_pth = 'results/hc3/pair_field_useAllSessions.pickle'
    df = pd.read_pickle('data/crcns-hc-3/hc3processed_useAllSessions.pickle')
    pair_field_preprocess_hc3(df, vthresh=5, sthresh=3, NShuffles=200, save_pth=save_pth)

    # -------------------------- Others ----------------------------
    # determine_sthresh_hc3(df)

if __name__=='__main__':

    main()