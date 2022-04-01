import numpy as np
import pandas as pd
import os
from os.path import join
from glob import glob

from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import filtfilt, butter, hilbert
from skimage.measure import find_contours
from skimage.draw import polygon2mask

import matplotlib.pyplot as plt
from common.preprocess import get_occupancy, placefield, complete_contourline, segment_fields, identify_pairs

def dataframe_conversion():
    # Load all files into a dataframe (per recording session)
    spikes_dir = 'data/emankin/principal_cells_and_path'
    lfp_dir = 'data/emankin/lfp'

    df_dict = dict(
        animal=[],
        date=[],
        ctxid=[],
        region=[],
        indata=[],
        spikedata=[],
        lfp_pth=[],
    )
    for walk in os.walk(spikes_dir):
        if len(walk[2]) > 0:
            dir_path_split = walk[0].split('/')
            rat_num, date = dir_path_split[3], dir_path_split[4]
            if (rat_num == 'rat624') and (date == '2013-09-19'):
                print('Broken LFP file, skip')
                continue
            region_indata = dict()
            region_spikedata = dict()
            region_cellinfo = dict()
            region_ctxid = dict()
            for filename in walk[2]:
                filepath = os.path.join(walk[0], filename)
                name_base, ext = os.path.splitext(filename)
                datakind, day_num, region = name_base.split('_')

                if datakind == 'spikeData':
                    spikedata = loadmat(filepath)
                    region_spikedata[region] = spikedata
                elif datakind == 'indata':
                    indata = loadmat(filepath)
                    region_indata[region] = indata
                elif datakind == 'ctxID':
                    ctxid = loadmat(filepath)
                    region_ctxid[region] = ctxid

                # Corresponding lfp file
                lfp_filedir = os.path.split(filepath)[0].replace('principal_cells_and_path', 'lfp')
                lfp_pth = join(lfp_filedir, 'rawLFP__Full.mat')

            assert sorted(region_spikedata.keys()) == sorted(region_indata.keys())
            all_regions_found = sorted(region_spikedata.keys())
            for region_found in all_regions_found:
                df_dict['animal'].append(rat_num)
                df_dict['date'].append(date)
                df_dict['region'].append(region_found)
                df_dict['indata'].append(region_indata[region_found])
                df_dict['spikedata'].append(region_spikedata[region_found])
                df_dict['ctxid'].append(region_ctxid[region_found])
                df_dict['lfp_pth'].append(lfp_pth)
    df_temp = pd.DataFrame(df_dict)

    #%%

    # Expand sessions to trials
    # Find max lfp
    inref_idx = dict(x=0, y=1, t=2, angle=3, v=4)
    spref_idx = dict(tsp=4, xsp=5, ysp=6, vsp=7, anglesp=10)
    spike_df_cols = ['tsp', 'xsp', 'ysp', 'vsp', 'anglesp']
    indata_df_cols = ['x', 'y', 't', 'angle', 'v']
    lfpref_idx = dict(t=0, lfp=1, fs=2)

    df_dict = dict(
        animal=[],
        date=[],
        behavior=[],
        trial=[],
        region=[],
        position=[],
        spikes=[],
        wave=[]
    )

    for i in range(df_temp.shape[0]):
        print('\r %d/%d' % (i, df_temp.shape[0]), end='', flush=True)
        animal, date, region, ctxid, indata, spikedata = df_temp.iloc[i][['animal', 'date', 'region', 'ctxid', 'indata', 'spikedata']]
        lfp_pth = df_temp.iloc[i]['lfp_pth']
        lfp = loadmat(lfp_pth)['rawLFPData']


        tsp, xsp, ysp, vsp, anglesp = [spikedata['spikeData'][0][0][spref_idx[x]] for x in spike_df_cols]
        ntrial_tsp = tsp.shape[0]
        ntrial_indata = indata['indata'].shape[0]
        assert ntrial_indata == ntrial_tsp

        spike_df_dict = dict()
        indata_df_dict = dict()
        for trial_idx in range(ntrial_tsp):

            # Spike data
            spike_df_dict['tsp'] = [tsp[trial_idx, x].squeeze() for x in range(tsp[trial_idx, ].shape[0]) ]
            spike_df = pd.DataFrame(spike_df_dict)

            # Indata
            x, y, t, angle, v = [indata['indata'][trial_idx][0][inref_idx[x]] for x in indata_df_cols]
            indata_df_dict['t'] = list(t.squeeze())
            indata_df_dict['x'] = list(x.squeeze())
            indata_df_dict['y'] = list(y.squeeze())
            indata_df = pd.DataFrame(indata_df_dict)

            # Wave
            num_chs = lfp.shape[1]
            thetadeltaratio = np.zeros(num_chs)
            for nch in range(num_chs):

                lfp_this = lfp[trial_idx, nch]
                tax = lfp_this[lfpref_idx['t']].squeeze()
                lfp_val = lfp_this[lfpref_idx['lfp']].squeeze()

                dt = np.median(np.diff(tax))
                thetaband = np.array([5, 12]) * dt * 2
                deltaband = np.array([1, 4]) * dt * 2
                B, A = butter(4, thetaband, btype='band')
                Bd, Ad = butter(2, deltaband, btype='band')
                theta_filt = filtfilt(B, A, lfp_val)
                delta_filt = filtfilt(Bd, Ad, lfp_val)
                thetadeltaratio[nch] = np.sum(theta_filt ** 2) / np.sum(delta_filt ** 2)

            maxchid = np.argmax(thetadeltaratio)
            lfp_selected = lfp[trial_idx, maxchid]
            lfp_val = lfp_selected[lfpref_idx['lfp']].squeeze()
            tax = lfp_selected[lfpref_idx['t']].squeeze()
            theta_selected = filtfilt(B, A, lfp_val)
            phase = np.angle(hilbert(theta_selected))
            wave_dict = dict()
            wave_dict['theta'] = theta_selected
            wave_dict['lfp'] = lfp_val
            wave_dict['tax'] = tax
            wave_dict['phase'] = phase
            wavedf = pd.DataFrame(wave_dict)

            df_dict['animal'].append(animal)
            df_dict['date'].append(date)
            df_dict['behavior'].append(ctxid['ctxID'][trial_idx, 0].item())
            df_dict['trial'].append(trial_idx)
            df_dict['region'].append(region)
            df_dict['position'].append(indata_df)
            df_dict['spikes'].append(spike_df)
            df_dict['wave'].append(wavedf)
    dftemp2 = pd.DataFrame(df_dict)

    # Merge rows of CA1, CA2 and CA3

    sepdf_dict = dict(
        animal=[],
        date=[],
        behavior=[],
        trial=[],
        posdf=[],  # indata for CA1, CA2, CA3 are the same. Merge here.
        CA1_spdf=[],
        CA2_spdf=[],
        CA3_spdf=[],
        wave = []  # wave for CA1, CA2, CA3 are the same. Merge here.
    )
    getlen = lambda x : len(x.shape)

    for (animal, date, trial), gpdf in dftemp2.groupby(by=['animal', 'date', 'trial']):
        behavior = gpdf.iloc[0]['behavior']

        sepdf_dict['animal'].append(animal)
        sepdf_dict['date'].append(date)
        sepdf_dict['behavior'].append(behavior)
        sepdf_dict['trial'].append(trial)
        positions_list = []
        waves_list = []
        for ca in ['CA1', 'CA2', 'CA3']:

            cadf = gpdf[gpdf['region'] == ca]
            if cadf.shape[0] <1:
                sepdf_dict['%s_spdf'%ca].append(pd.DataFrame(dict(tsp=[])))
            else:
                position = cadf.iloc[0]['position']
                wave = cadf.iloc[0]['wave']
                spikes = cadf.iloc[0]['spikes']

                # Clean Spike data
                tsplen = spikes.tsp.apply(getlen)
                spikes.drop(index=spikes[tsplen == 2].index, inplace=True)
                spikes.reset_index(drop=True, inplace=True)
                tsplen = spikes.tsp.apply(getlen)
                spikes.loc[tsplen==0, 'tsp'] = spikes.loc[tsplen==0, 'tsp'].apply(np.atleast_1d)

                sepdf_dict['%s_spdf'%ca].append(spikes)
                positions_list.append(position)
                waves_list.append(wave)

        sepdf_dict['posdf'].append(positions_list[0])  # doesn't matter CA1/CA2/CA3
        sepdf_dict['wave'].append(waves_list[0])  # doesn't matter CA1/CA2/CA3.

    df = pd.DataFrame(sepdf_dict)
    df.to_pickle('data/emankin_pythonRaw.pickle')
    return None


def preprocess1():

    df = pd.read_pickle('data/emankin_pythonRaw.pickle')
    area_thresh = 25
    print('Compute occupancy, rate and fields')
    # # Compute occupancy
    # It takes ~4 hours. Recommend saving the result as checkpoint.
    num_sess = df.shape[0]
    occ_list = []

    CA_fielddf_list = dict(CA1=[], CA2=[], CA3=[])
    CA_pairdf_list = dict(CA1=[], CA2=[], CA3=[])
    for nsess in range(num_sess):
        print("%d/%d Session - Computing occupancy and rate" % (nsess, num_sess))

        # Occupancy
        posdf = df.loc[nsess, 'posdf']
        x, y, t = posdf.x.to_numpy(), posdf.y.to_numpy(), posdf.t.to_numpy()
        xx, yy, occupancy = get_occupancy(x, y, t)
        occ_list.append(dict(xx=xx, yy=yy, occ=occupancy))
        xbound = (0, xx.shape[1]-1)
        ybound = (0, yy.shape[0]-1)
        x_ax, y_ax = xx[0, :], yy[:, 0]


        # Rate
        trange = (t.max(), t.min())
        x_interp = interp1d(t, x)
        y_interp = interp1d(t, y)



        for ca in ['CA1', 'CA2', 'CA3']:
            pf_list = []
            pairdf_dict = dict(field1_id=[], field2_id=[])
            fielddf_dict = dict(cellid=[], mask=[], xyval=[])
            spdf = df.loc[nsess, '%s_spdf' % (ca)]
            if spdf.shape[0] < 1:
                CA_fielddf_list[ca].append(pd.DataFrame(fielddf_dict))
                CA_pairdf_list[ca].append(pd.DataFrame(pairdf_dict))
                spdf['pf'] = pf_list
                continue
            num_cells = spdf.shape[0]
            for ncell in range(num_cells):

                # Computation of rate map
                tsp = spdf.loc[ncell, 'tsp']
                tsp_in = tsp[(tsp < trange[0]) & (tsp > trange[1])]
                xsp_in = x_interp(tsp_in)
                ysp_in = y_interp(tsp_in)
                freq, rate = placefield(xx, yy, occupancy, xsp_in, ysp_in, tsp_in)
                pf_list.append(dict(freq=freq, rate=rate))

                # Segment placefields
                for mask, xyval in segment_fields(xx, yy, freq, rate, area_thresh):
                    fielddf_dict['cellid'].append(ncell)
                    fielddf_dict['mask'].append(mask)
                    fielddf_dict['xyval'].append(xyval)
            fielddf = pd.DataFrame(fielddf_dict)
            CA_fielddf_list[ca].append(fielddf)
            spdf['pf'] = pf_list


            # Identify pairs
            num_fields = fielddf.shape[0]
            if num_fields > 1:
                for fieldid1, fieldid2 in identify_pairs(fielddf, spdf, x_interp, y_interp, x_ax, y_ax, trange):
                    pairdf_dict['field1_id'].append(fieldid1)
                    pairdf_dict['field2_id'].append(fieldid2)
            pairdf = pd.DataFrame(pairdf_dict)
            CA_pairdf_list[ca].append(pairdf)


    df['occ_dict'] = occ_list
    df['CA1fields'] = CA_fielddf_list['CA1']
    df['CA2fields'] = CA_fielddf_list['CA2']
    df['CA3fields'] = CA_fielddf_list['CA3']
    df['CA1pairs'] = CA_pairdf_list['CA1']
    df['CA2pairs'] = CA_pairdf_list['CA2']
    df['CA3pairs'] = CA_pairdf_list['CA3']
    df.to_pickle('data/emankin_pythonProcessed.pickle')


def main():
    dataframe_conversion()
    preprocess1()
    return None

if __name__ == '__main__':
    main()