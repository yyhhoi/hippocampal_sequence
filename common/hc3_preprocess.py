# -*- coding: utf-8 -*-
import requests
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from scipy.interpolate import interp1d

URL = 'https://portal.nersc.gov/project/crcns/download/index.php'


def crns_get(datafile, username, password, dest_dir=''):
    login_data = dict(
        username=username,
        password=password,
        fn=datafile,
        submit='Login'
    )

    with requests.Session() as s:
        local_filename = login_data['fn'].split('/')[-1]
        r = s.post(URL, data=login_data, stream=True)
        with open(os.path.join(dest_dir, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    return local_filename


def get_sess_dict(cell_df, session_df):
    topdir2animal = dict()
    sess2topdir = dict()
    sess2animal = dict()
    sess2behavior = dict()
    sess2duration = dict()

    for i in range(cell_df.shape[0]):
        animal_each, topdir_each = cell_df.iloc[i][['animal', 'topdir']]
        topdir2animal[topdir_each] = animal_each

    for i in range(session_df.shape[0]):
        session_each, topdir_each, beh_each, duration = session_df.iloc[i][['session', 'topdir', 'behavior', 'duration']]
        sess2topdir[session_each] = topdir_each
        sess2animal[session_each] = topdir2animal[topdir_each]
        sess2behavior[session_each] = beh_each
        sess2duration[session_each] = duration
    return sess2topdir, sess2animal, sess2behavior, sess2duration


def read_whl(whl_path, position_fs=39.06):
    position_df = pd.read_csv(whl_path, delimiter='\t', names=['x', 'y', 'x2', 'y2'])
    position_df['tidx'] = position_df.index
    position_df['t'] = position_df.tidx/position_fs

    # I just assume we only need the first led position at the head.
    del position_df['x2']
    del position_df['y2']
    return position_df

def process_position_df(position_df, arena_l):
    # Remove starting/trailing time period where position is not available.
    negxidx = np.where(position_df['x']>-1)[0]
    negyidx = np.where(position_df['y']>-1)[0]
    mintidx = min(negxidx.min(), negyidx.min())
    maxtidx = min(negxidx.max(), negyidx.max())
    new_position_df = position_df.loc[mintidx:maxtidx]

    # interpolate missing x and y in the middle.
    x_fit = new_position_df.loc[new_position_df['x']!=-1, 'x'].to_numpy()
    tx_fit = new_position_df.loc[new_position_df['x']!=-1, 't'].to_numpy()
    y_fit = new_position_df.loc[new_position_df['y']!=-1, 'y'].to_numpy()
    ty_fit = new_position_df.loc[new_position_df['y']!=-1, 't'].to_numpy()
    x_interp = interp1d(tx_fit, x_fit)
    y_interp = interp1d(ty_fit, y_fit)
    x_missmask = new_position_df['x']==-1
    y_missmask = new_position_df['y']==-1
    tx_miss = new_position_df.loc[x_missmask, 't'].to_numpy()
    ty_miss = new_position_df.loc[y_missmask, 't'].to_numpy()
    x_fill = x_interp(tx_miss)
    y_fill = y_interp(ty_miss)
    new_position_df_copy = new_position_df.copy()  # avoid chained assignment in pandas
    new_position_df_copy.loc[x_missmask, 'x'] = x_fill
    new_position_df_copy.loc[y_missmask, 'y'] = y_fill
    new_position_df_copy = new_position_df_copy.reset_index(drop=True)

    # Rescale the trajectory to the size of arena
    new_position_df_copy['x'] = new_position_df_copy['x'] - new_position_df_copy['x'].min()
    new_position_df_copy['y'] = new_position_df_copy['y'] - new_position_df_copy['y'].min()
    new_position_df_copy['x'] = new_position_df_copy['x']/new_position_df_copy['x'].max()*arena_l
    new_position_df_copy['y'] = new_position_df_copy['y']/new_position_df_copy['y'].max()*arena_l

    return new_position_df_copy


def get_sr_from_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    spike_sr = float(root.find('acquisitionSystem').find('samplingRate').text)
    lfp_sr = float(root.find('fieldPotentials').find('lfpSamplingRate').text)

    if (spike_sr == 32552) or (spike_sr == 20000):
        return spike_sr, lfp_sr
    else:
        print('Wrong sampling rate encountered')
        raise
