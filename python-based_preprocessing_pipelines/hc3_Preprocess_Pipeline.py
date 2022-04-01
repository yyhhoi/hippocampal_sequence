import numpy as np
import pandas as pd
import os
import shutil
import re
import scipy.io
import matplotlib.pyplot as plt
from os.path import join
from glob import glob
from common.hc3_preprocess import crns_get, get_sess_dict, read_whl, get_sr_from_xml, process_position_df
from common.preprocess import get_occupancy, placefield, get_occupancy_bins, placefield_bins, spatial_information, \
    complete_contourline, segment_fields, identify_pairs
from common.utils import load_pickle
from common.comput_utils import pair_diff, find_pair_times
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, butter, hilbert

from scipy.io import loadmat
from skimage.measure import find_contours
from skimage.draw import polygon2mask


def get_metadata():
    meta_dir = 'data/crcns-hc-3/crcns-hc3-metadata-tables'
    session_df = pd.read_csv(os.path.join(meta_dir, 'hc3-session.csv'),
                             names=['id', 'topdir', 'session', 'behavior', 'fam', 'duration']
                             )
    cell_df = pd.read_csv(os.path.join(meta_dir, 'hc3-cell.csv'),
                          names=['matlab_id', 'topdir', 'animal', 'ele', 'clu', 'region', 'ne', 'ni', 'eg', 'ig', 'ed',
                                 'id', 'fireRate', 'totalFireRate', 'type', 'eDist', 'RefracRatio', 'RefracViol']
                          )
    return session_df, cell_df


def organize_raw_dataframe():
    data_dir = 'data/crcns-hc-3/'
    session_df, cell_df = get_metadata()
    position_fs = 39.06  # See dataset description from crcns
    excluded_dirs = ['data_zips-tables', 'crcns-hc3-metadata-tables']
    sess2topdir, sess2animal, sess2behavior, sess2duration = get_sess_dict(cell_df, session_df)

    df_dict = dict(
        animal=[],
        topdir=[],
        session=[],
        duration=[],
        behavior=[],
        position=[],
        CA1_spdf=[],
        CA3_spdf=[],
        wave=[],
        spike_samplingrate=[],
        lfp_samplingrate=[],
        position_samplingrate=[]
    )
    progress_idx = 0

    # Loop topdir
    for topdir_each in os.listdir(data_dir):
        if topdir_each in excluded_dirs:
            continue
        nextdir = os.path.join(data_dir, topdir_each)
        if os.path.isdir(nextdir) is False:
            print('Skip', nextdir)
            continue

        # Loop session
        for sess_folder in os.listdir(os.path.join(data_dir, topdir_each)):
            progress_idx += 1
            print('\r%d)' % progress_idx, end='', flush=True)
            sess_dir = os.path.join(data_dir, topdir_each, sess_folder)

            # Rat's position and LFP
            position_file_path = os.path.join(os.path.join(sess_dir, '%s.whl' % sess_folder))
            lfp_path = os.path.join(os.path.join(sess_dir, '%s.mat' % sess_folder))
            if (os.path.isfile(position_file_path)) and (os.path.isfile(lfp_path)):
                position_df = read_whl(position_file_path)
                lfp_allch = loadmat(lfp_path)['sesslfp']
            else:
                print('%s/%s .whl or . mat not found' % (topdir_each, sess_folder))
                continue  # spike data without position is not needed
            topdir, animal, behavior, duration = sess2topdir[sess_folder], sess2animal[sess_folder], sess2behavior[
                sess_folder], sess2duration[sess_folder]

            # Get sampling rate
            xml_path = os.path.join(sess_dir, '%s.xml' % (sess_folder))
            spike_sr, lfp_sr = get_sr_from_xml(xml_path)

            # Process theta wave
            dt = 1 / lfp_sr
            thetaband = np.array([5, 12]) * dt * 2
            deltaband = np.array([1, 4]) * dt * 2
            B, A = butter(4, thetaband, btype='band')
            Bd, Ad = butter(2, deltaband, btype='band')
            num_chs = lfp_allch.shape[0]
            thetadeltaratio = np.zeros(num_chs)
            for nch in range(num_chs):
                lfp_eachch = lfp_allch[nch, :]
                theta_filt = filtfilt(B, A, lfp_eachch)
                delta_filt = filtfilt(Bd, Ad, lfp_eachch)
                thetadeltaratio[nch] = np.sum(theta_filt ** 2) / np.sum(delta_filt ** 2)
            maxchid = np.argmax(thetadeltaratio)
            lfp_selected = lfp_allch[maxchid, :]
            tax = np.arange(lfp_selected.shape[0]) / lfp_sr
            theta_selected = filtfilt(B, A, lfp_selected)
            phase = np.angle(hilbert(theta_selected))

            wave_dict = dict()
            wave_dict['theta'] = theta_selected
            wave_dict['lfp'] = lfp_selected
            wave_dict['tax'] = tax
            wave_dict['phase'] = phase
            wavedf = pd.DataFrame(wave_dict)

            # Spike informations
            tsp_dict = {'CA1': {'ele': [], 'clu': [], 'tsp': []},
                        'CA3': {'ele': [], 'clu': [], 'tsp': []}}

            for region in ['CA1', 'CA3']:
                filtered_cell_df = cell_df[
                    (cell_df.topdir == topdir) & (cell_df.animal == animal) & (cell_df.region == region) & (
                            cell_df.type == 'p')]
                region_ele_list = filtered_cell_df.ele.unique()

                for ele in region_ele_list:
                    all_clu_list = filtered_cell_df[(filtered_cell_df.ele == ele) & (filtered_cell_df.type == 'p') & (
                            filtered_cell_df.fireRate < 7)].clu.unique()
                    res_path = os.path.join(sess_dir, '%s.res.%d' % (sess_folder, ele))
                    clu_path = os.path.join(sess_dir, '%s.clu.%d' % (sess_folder, ele))
                    res = pd.read_csv(res_path, names=['tidx'])
                    clu = pd.read_csv(clu_path, header=0, names=['clu_class'])
                    res['time'] = res.tidx / spike_sr
                    for clu_num_each in all_clu_list:
                        tsp = res.loc[clu.clu_class == clu_num_each, 'time'].to_numpy()
                        if tsp.shape[0] > 0:
                            tsp_dict[region]['ele'].append(ele)
                            tsp_dict[region]['clu'].append(clu_num_each)
                            tsp_dict[region]['tsp'].append(tsp)

            CA1_spdf = pd.DataFrame(tsp_dict['CA1'])
            CA3_spdf = pd.DataFrame(tsp_dict['CA3'])

            df_dict['animal'].append(animal)
            df_dict['topdir'].append(topdir)
            df_dict['session'].append(sess_folder)
            df_dict['duration'].append(duration)
            df_dict['behavior'].append(behavior)
            df_dict['position'].append(position_df)
            df_dict['CA1_spdf'].append(CA1_spdf)
            df_dict['CA3_spdf'].append(CA3_spdf)
            df_dict['wave'].append(wavedf)
            df_dict['spike_samplingrate'].append(spike_sr)
            df_dict['lfp_samplingrate'].append(lfp_sr)
            df_dict['position_samplingrate'].append(position_fs)

    hc3data_df = pd.DataFrame(df_dict)
    return hc3data_df


def check_raw_dataframe(hc3data_df):
    # 1. Check sampling rates by duration, positions by pixel
    check_dict = dict(beh=[], xmax=[], ymax=[], xrange=[], yrange=[], duration=[], tmaxRun=[], tspmaxCA1=[],
                      tspmaxCA3=[], tmaxWave=[])
    count_dict = dict(topdir=[], session=[], CA1_ncells=[], CA3_ncells=[])
    clu_all = []
    for i in range(hc3data_df.shape[0]):

        posdf = hc3data_df.loc[i, 'position']
        enclosure, duration = hc3data_df.loc[i, ['behavior', 'duration']]
        topdir, session = hc3data_df.loc[i, ['topdir', 'session']]

        wave = hc3data_df.loc[i, 'wave']

        wave_tmax, wave_tmin = wave['tax'].max(), wave['tax'].min()

        xmax, ymax, tmax = posdf['x'].max(), posdf['y'].max(), posdf['t'].max()

        nonnanmask = (posdf['x'] > -1) & (posdf['y'] > -1)

        xmin, ymin = posdf.loc[nonnanmask, 'x'].min(), posdf.loc[nonnanmask, 'y'].min()

        CA1spdf, CA3spdf = hc3data_df.loc[i, ['CA1_spdf', 'CA3_spdf']]

        if CA1spdf.shape[0] > 0:
            tmaxCA1 = np.concatenate(CA1spdf['tsp'].to_list()).max()

            clu_all.extend(CA1spdf['clu'].to_list())

        else:
            tmaxCA1 = np.nan
        if CA3spdf.shape[0] > 0:
            tmaxCA3 = np.concatenate(CA3spdf['tsp'].to_list()).max()
            clu_all.extend(CA1spdf['clu'].to_list())
        else:
            tmaxCA3 = np.nan

        check_dict['beh'].append(enclosure)
        check_dict['xmax'].append((xmin, xmax))
        check_dict['ymax'].append((ymin, ymax))
        check_dict['xrange'].append(xmax - xmin)
        check_dict['yrange'].append(ymax - ymin)
        check_dict['duration'].append(duration)
        check_dict['tmaxRun'].append(tmax)
        check_dict['tspmaxCA1'].append(tmaxCA1)
        check_dict['tspmaxCA3'].append(tmaxCA3)
        check_dict['tmaxWave'].append(wave_tmax)
        count_dict['topdir'].append(topdir)
        count_dict['session'].append(session)
        count_dict['CA1_ncells'].append(CA1spdf.shape[0])
        count_dict['CA3_ncells'].append(CA3spdf.shape[0])

    check_df = pd.DataFrame(check_dict)
    count_df = pd.DataFrame(count_dict)

    unique_topdirs = count_df.topdir.unique()
    selected_idx = []
    for topdir in unique_topdirs:
        tmpdf = count_df[count_df['topdir'] == topdir]
        maxidx = tmpdf[tmpdf['CA3_ncells'] == tmpdf['CA3_ncells'].max()].index
        if maxidx.shape[0] > 1:
            maxidx = [maxidx[0]]

        selected_idx.extend(maxidx)

    num_allCA1 = count_df.loc[selected_idx, 'CA1_ncells'].sum()
    num_allCA3 = count_df.loc[selected_idx, 'CA3_ncells'].sum()

    print('With unique topdir, num CA1 = %d, num CA3 = %d' % (num_allCA1, num_allCA3))
    print('Unique clu: ', np.unique(clu_all))
    print(check_df)
    return None


def step_clean_position_data(hc3data_df):
    # 1. Interpolate missing positions
    # 2. rescale x, y coordinates from pixel to cm
    print('Cleaning position data')
    arena_l = {'bigSquare': 180, 'midSquare': 120}
    new_posdf_list = []
    for i in range(hc3data_df.shape[0]):
        posdf = hc3data_df.loc[i, 'position']
        beh = hc3data_df.loc[i, 'behavior']
        new_posdf = process_position_df(posdf, arena_l=arena_l[beh])
        new_posdf_list.append(new_posdf)
    hc3data_df['posdf'] = new_posdf_list
    return hc3data_df


def step_concat_sessions_within_topdir(hc3data_df):
    print('Concatenate sessions')
    combined_df_dict = dict(animal=[],
                            topdir=[],
                            session=[],
                            duration=[],
                            behavior=[],
                            CA1_spdf=[],
                            CA3_spdf=[],
                            wave=[],
                            posdf=[],
                            spike_samplingrate=[],
                            lfp_samplingrate=[],
                            position_samplingrate=[])

    for topdir, topdf in hc3data_df.groupby('topdir'):
        if topdf.shape[0] < 2:  # Nothing to combine
            for col in combined_df_dict.keys():
                combined_df_dict[col].append(topdf.iloc[0][col])
        else:
            topdf = topdf.sort_values('session').reset_index(drop=True)
            num_sess = topdf.shape[0]
            animal, beh, ssr, lsr, psr = topdf.loc[
                0, ['animal', 'behavior', 'spike_samplingrate', 'lfp_samplingrate', 'position_samplingrate']]

            # Shift time
            posdf_list, wave_list, CA1_spdf_list, CA3_spdf_list = [], [], [], []
            shifttime = 0
            for nsess in range(num_sess):
                posdf, CA1_spdf, CA3_spdf, wave = topdf.loc[nsess, ['posdf', 'CA1_spdf', 'CA3_spdf', 'wave']]
                pos_tmin, pos_tmax = posdf['t'].min(), posdf['t'].max()
                wave = wave[(wave['tax'] > pos_tmin) & (wave['tax'] < pos_tmax)].reset_index(drop=True)

                posdf['t'] = posdf['t'] - pos_tmin + shifttime
                wave['tax'] = wave['tax'] - pos_tmin + shifttime

                CA1nums, CA3nums = CA1_spdf.shape[0], CA3_spdf.shape[0]
                tspca1_list, tspca3_list = [], []
                for ica1 in range(CA1nums):
                    tspca1 = CA1_spdf.loc[ica1, 'tsp']
                    tspca1 = tspca1[(tspca1 > pos_tmin) & (tspca1 < pos_tmax)]
                    tspca1 = tspca1 - pos_tmin + shifttime
                    tspca1_list.append(tspca1)

                for ica3 in range(CA3nums):
                    tspca3 = CA3_spdf.loc[ica3, 'tsp']
                    tspca3 = tspca3[(tspca3 > pos_tmin) & (tspca3 < pos_tmax)]
                    tspca3 = tspca3 - pos_tmin + shifttime
                    tspca3_list.append(tspca3)
                CA1_spdf['tsp'] = tspca1_list
                CA3_spdf['tsp'] = tspca3_list

                posdf_list.append(posdf)
                wave_list.append(wave)
                CA1_spdf_list.append(CA1_spdf)
                CA3_spdf_list.append(CA3_spdf)

                shifttime = posdf['t'].max() + (1 / psr)

            # Concatenate
            poscat = pd.concat(posdf_list, ignore_index=True, axis=0)
            wavecat = pd.concat(wave_list, ignore_index=True, axis=0)
            duration = poscat['t'].max() - poscat['t'].min()
            ca1dict = {'ele': [], 'clu': [], 'tsp': []}
            CA1_spdf_cat = pd.concat(CA1_spdf_list, ignore_index=True, axis=0)
            CA1_spdf_cat['cell_code'] = CA1_spdf_cat['ele'] * 1000 + CA1_spdf_cat['clu']
            for cell_code, codedf in CA1_spdf_cat.groupby('cell_code'):
                ele = codedf.iloc[0]['ele']
                clu = codedf.iloc[0]['clu']
                tsp_cat = np.concatenate(codedf['tsp'].to_list())
                ca1dict['ele'].append(ele)
                ca1dict['clu'].append(clu)
                ca1dict['tsp'].append(tsp_cat)

            new_CA1_spdf = pd.DataFrame(ca1dict)

            ca3dict = {'ele': [], 'clu': [], 'tsp': []}
            CA3_spdf_cat = pd.concat(CA3_spdf_list, ignore_index=True, axis=0)
            CA3_spdf_cat['cell_code'] = CA3_spdf_cat['ele'] * 1000 + CA3_spdf_cat['clu']
            for cell_code, codedf in CA3_spdf_cat.groupby('cell_code'):
                ele = codedf.iloc[0]['ele']
                clu = codedf.iloc[0]['clu']
                tsp_cat = np.concatenate(codedf['tsp'].to_list())
                ca3dict['ele'].append(ele)
                ca3dict['clu'].append(clu)
                ca3dict['tsp'].append(tsp_cat)

            new_CA3_spdf = pd.DataFrame(ca1dict)

            combined_df_dict['animal'].append(animal)
            combined_df_dict['topdir'].append(topdir)
            combined_df_dict['session'].append('+'.join(topdf.session.to_list()))
            combined_df_dict['duration'].append(duration)
            combined_df_dict['behavior'].append(beh)
            combined_df_dict['CA1_spdf'].append(new_CA1_spdf)
            combined_df_dict['CA3_spdf'].append(new_CA3_spdf)
            combined_df_dict['wave'].append(wavecat)
            combined_df_dict['posdf'].append(poscat)
            combined_df_dict['spike_samplingrate'].append(ssr)
            combined_df_dict['lfp_samplingrate'].append(lsr)
            combined_df_dict['position_samplingrate'].append(psr)

    combined_df = pd.DataFrame(combined_df_dict)
    return combined_df


def step_compute_occupancy(hc3data_df):
    print('Compute occupancy')
    # # Compute occupancy
    # It takes ~4 hours. Recommend saving the result as checkpoint.
    num_sess = hc3data_df.shape[0]
    occ_list = []
    for nsess in range(num_sess):
        print("%d/%d Session - Computing occupancy" % (nsess, num_sess))

        posdf = hc3data_df.loc[nsess, 'posdf']
        x, y, t = posdf.x.to_numpy(), posdf.y.to_numpy(), posdf.t.to_numpy()
        xx, yy, occupancy = get_occupancy(x, y, t)
        occ_list.append(dict(xx=xx, yy=yy, occ=occupancy))
    hc3data_df['occ_dict'] = occ_list
    return hc3data_df


def step_ratemap_shuffling(hc3data_df):
    print('Compute rate map and shuffle spatial information')

    # Shuffling
    Nshuffle = 1000
    seed = 0
    num_sess = hc3data_df.shape[0]
    for nsess in range(num_sess):

        posdf = hc3data_df.loc[nsess, 'posdf']
        x, y, t = posdf.x.to_numpy(), posdf.y.to_numpy(), posdf.t.to_numpy()
        trange = (t.max(), t.min())
        x_interp = interp1d(t, x)
        y_interp = interp1d(t, y)
        x_ax, y_ax, occupancy = get_occupancy_bins(x, y, t, binstep=10)

        occ_kde_dict = hc3data_df.loc[nsess, 'occ_dict']
        xx_kde, yy_kde, occupancy_kde = occ_kde_dict['xx'], occ_kde_dict['yy'], occ_kde_dict['occ']

        for ca in ['CA1', 'CA3']:
            info_list = []
            info_shuf_list = []
            info_pval_list = []
            pf_list = []

            spdf = hc3data_df.loc[nsess, '%s_spdf' % (ca)]
            if spdf.shape[0] < 1:
                spdf['spatialinfo_perspike'] = info_list
                spdf['spatialinfo_perspike_shuf'] = info_shuf_list
                spdf['spatialinfo_pval'] = info_pval_list
                spdf['pf'] = pf_list
                continue

            num_cells = spdf.shape[0]
            for ncell in range(num_cells):
                print('%d/%d Session, %s %d/%d cell ' % (nsess, num_sess, ca, ncell, num_cells), end='')
                tsp = spdf.loc[ncell, 'tsp']
                tsp_in = tsp[(tsp < trange[0]) & (tsp > trange[1])]
                if tsp_in.shape[0] < 2:
                    print('Skip, tsp.shape[0]=%d' % (tsp_in.shape[0]))
                    info_list.append(np.nan)
                    info_shuf_list.append(np.nan)
                    info_pval_list.append(np.nan)
                    pf_list.append(np.nan)
                    continue

                xsp_in = x_interp(tsp_in)
                ysp_in = y_interp(tsp_in)
                freq, rate = placefield_bins(x_ax, y_ax, occupancy, xsp_in, ysp_in)
                info_persecond, info_perspike = spatial_information(rate, occupancy)

                freq_kde, rate_kde = placefield(xx_kde, yy_kde, occupancy_kde, xsp_in, ysp_in, tsp_in)

                # Shuffling
                info_perspike_shuffleds = np.zeros(Nshuffle)
                for nshuf in range(Nshuffle):
                    tsp_in_min = tsp_in.min()
                    tsp_in_offseted = tsp_in - tsp_in_min
                    maxtsp_offseted = tsp_in_offseted.max()
                    np.random.seed(seed)
                    shiftamount = np.random.uniform(0, maxtsp_offseted)
                    tsp_in_shuffled_offseted = np.mod(tsp_in_offseted + shiftamount, maxtsp_offseted)
                    tsp_in_shuffled = tsp_in_shuffled_offseted + tsp_in_min

                    tsp_in_shuffled = tsp_in_shuffled[(tsp_in_shuffled < trange[0]) & (tsp_in_shuffled > trange[1])]
                    xsp_in_shuffled = x_interp(tsp_in_shuffled)
                    ysp_in_shuffled = y_interp(tsp_in_shuffled)
                    freq_shuffled, rate_shuffled = placefield_bins(x_ax, y_ax, occupancy, xsp_in_shuffled,
                                                                   ysp_in_shuffled)
                    _, info_perspike_shuffleds[nshuf] = spatial_information(rate_shuffled, occupancy)

                    seed += 1

                try:
                    pval = 1 - (info_perspike > info_perspike_shuffleds).mean()
                except:
                    pval = 1

                info_list.append(info_perspike)
                info_shuf_list.append(info_perspike_shuffleds)
                info_pval_list.append(pval)
                pf_list.append(dict(freq=freq_kde, rate=rate_kde))
                print('pval=%0.4f' % (pval))
            spdf['spatialinfo_perspike'] = info_list
            spdf['spatialinfo_perspike_shuf'] = info_shuf_list
            spdf['spatialinfo_pval'] = info_pval_list
            spdf['pf'] = pf_list

    return hc3data_df


def step_segment_filter_single_fields(hc3data_df):
    print('Segment and filter single fields')
    # # Segment place fields & Filter place fields
    bounddict = dict(bigSquare=(0, 180), midSquare=(0, 120))
    areathresh_dict = dict(bigSquare=156.25, midSquare=56.25)
    num_sess = hc3data_df.shape[0]

    CA_fielddf_list = dict(CA1=[], CA3=[])
    for nsess in range(num_sess):
        print('%d/%d Session' % (nsess, num_sess))

        behavior = hc3data_df.loc[nsess, 'behavior']
        occ_dict = hc3data_df.loc[nsess, 'occ_dict']
        xx, yy, occ = occ_dict['xx'], occ_dict['yy'], occ_dict['occ']

        for ca in ['CA1', 'CA3']:

            fielddf_dict = dict(cellid=[], mask=[], xyval=[])
            spdf = hc3data_df.loc[nsess, '%s_spdf' % (ca)]
            drop_mask = (spdf['spatialinfo_pval'] > 0.05) | (spdf['pf'].isna())
            spdf.drop(index=spdf[drop_mask].index, axis=0, inplace=True)
            spdf.index = np.arange(spdf.shape[0])

            num_cells = spdf.shape[0]

            for ncell in range(num_cells):
                rate = spdf.loc[ncell, 'pf']['rate']
                freq = spdf.loc[ncell, 'pf']['freq']
                for mask, xyval in segment_fields(xx, yy, freq, rate, areathresh_dict[behavior]):
                    fielddf_dict['cellid'].append(ncell)
                    fielddf_dict['mask'].append(mask)
                    fielddf_dict['xyval'].append(xyval)

            CA_fielddf_list[ca].append(pd.DataFrame(fielddf_dict))

    hc3data_df['CA1fields'] = CA_fielddf_list['CA1']
    hc3data_df['CA3fields'] = CA_fielddf_list['CA3']

    return hc3data_df


def step_choose_one_session(hc3data_df):
    print('Choose one sessions within each topdir')
    # # (Optional) if you didn't concatenate the sub-sessions, then choose one session from each topdir, or do nothing
    # # Choose one session from each topdir, based on max CA3 field count, otherwise max CA1 field count
    selected_dfindex = []
    for topdir_each in hc3data_df.topdir.unique():
        topdirdf = hc3data_df[hc3data_df['topdir'] == topdir_each]
        num_sess = topdirdf.shape[0]
        idx_arr = np.zeros(num_sess)
        count1_arr = np.zeros(num_sess)
        count3_arr = np.zeros(num_sess)
        for arri, dfidx in enumerate(topdirdf.index):
            ca1spdf = topdirdf.loc[dfidx, 'CA1fields']
            ca3spdf = topdirdf.loc[dfidx, 'CA3fields']

            count_ca1 = ca1spdf.shape[0]
            count_ca3 = ca3spdf.shape[0]

            idx_arr[arri] = dfidx
            count1_arr[arri] = count_ca1
            count3_arr[arri] = count_ca3

        maxcount3 = np.max(count3_arr)
        if maxcount3 < 1:
            maxarrid = np.argmax(count1_arr)
        else:
            maxarrid = np.argmax(count3_arr)

        maxdfid = idx_arr[maxarrid]
        selected_dfindex.append(int(maxdfid))

    hc3data_df = hc3data_df.loc[selected_dfindex].reset_index(drop=True)

    # Show field counts
    CA1spdf = pd.concat(hc3data_df['CA1fields'].to_list(), axis=0, ignore_index=True)
    CA3spdf = pd.concat(hc3data_df['CA3fields'].to_list(), axis=0, ignore_index=True)
    print('Field count CA1 = %d, CA3 = %d' % (CA1spdf.shape[0], CA3spdf.shape[0]))
    return hc3data_df


def step_identify_pairs(hc3data_df):
    print('Identify pairs')
    num_sess = hc3data_df.shape[0]

    CA1pairs_list = []
    CA3pairs_list = []

    for nsess in range(num_sess):

        posdf, occ_dict = hc3data_df.loc[nsess, ['posdf', 'occ_dict']]
        x, y, t = posdf['x'].to_numpy(), posdf['y'].to_numpy(), posdf['t'].to_numpy()
        trange = (t.max(), t.min())
        xinterp, yinterp = interp1d(t, x), interp1d(t, y)
        xx, yy = occ_dict['xx'], occ_dict['yy']
        x_ax, y_ax = xx[0, :], yy[:, 0]
        pair_dict = dict(CA1=dict(field1_id=[], field2_id=[]),
                         CA3=dict(field1_id=[], field2_id=[]))
        for ca in ['CA1', 'CA3']:
            fielddf = hc3data_df.loc[nsess, '%sfields' % ca]
            spdf = hc3data_df.loc[nsess, '%s_spdf' % ca]

            # Identify pairs
            num_fields = fielddf.shape[0]
            if num_fields > 1:
                for fieldid1, fieldid2 in identify_pairs(fielddf, spdf, xinterp, yinterp, x_ax, y_ax, trange):
                    pair_dict[ca]['field1_id'].append(fieldid1)
                    pair_dict[ca]['field2_id'].append(fieldid2)


        CA1pairs_list.append(pd.DataFrame(pair_dict['CA1']))
        CA3pairs_list.append(pd.DataFrame(pair_dict['CA3']))

    hc3data_df['CA1pairs'] = CA1pairs_list
    hc3data_df['CA3pairs'] = CA3pairs_list
    print()
    print('Number of pairs in CA1 = %s' % pd.concat(hc3data_df['CA1pairs'].to_list(), axis=0).shape[0])
    print('Number of pairs in CA3 = %s' % pd.concat(hc3data_df['CA3pairs'].to_list(), axis=0).shape[0])
    return hc3data_df


def pipeline_use_all_individual_sessions(hc3data_df):
    print('Pipeline: Use all sessions')
    hc3data_df = step_clean_position_data(hc3data_df)
    hc3data_df = step_compute_occupancy(hc3data_df)
    hc3data_df = pd.read_pickle('data/crcns-hc-3/hc3data_df_raw_StepOccupancy.pickle')
    hc3data_df = step_ratemap_shuffling(hc3data_df)
    hc3data_df = step_segment_filter_single_fields(hc3data_df)
    hc3data_df = step_identify_pairs(hc3data_df)
    return hc3data_df


def main():
    # Most functions have in-place operations.
    # Therefore, don't re-use the variable "hc3data_df" across pipelines, but load the raw dataframe every time instead.

    # # Organize raw data into dataframe
    # hc3data_df = organize_raw_dataframe()
    # check_raw_dataframe(hc3data_df)
    # hc3data_df.to_pickle('data/crcns-hc-3/hc3data_df_raw.pickle')

    hc3data_df = pd.read_pickle('data/crcns-hc-3/hc3data_df_raw.pickle')
    hc3data_df = pipeline_use_all_individual_sessions(hc3data_df)
    hc3data_df.to_pickle('data/crcns-hc-3/hc3processed_useAllSessions.pickle')

    pass


if __name__ == '__main__':
    main()
