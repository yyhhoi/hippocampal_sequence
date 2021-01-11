# -*- coding: utf-8 -*-
import requests
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

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

    for i in range(cell_df.shape[0]):
        animal_each, topdir_each = cell_df.iloc[i][['animal', 'topdir']]
        topdir2animal[topdir_each] = animal_each

    for i in range(session_df.shape[0]):
        session_each, topdir_each, beh_each = session_df.iloc[i][['session', 'topdir', 'behavior']]
        sess2topdir[session_each] = topdir_each
        sess2animal[session_each] = topdir2animal[topdir_each]
        sess2behavior[session_each] = beh_each
    return sess2topdir, sess2animal, sess2behavior


def read_whl(whl_path, position_fs=39.06):
    position_df = pd.read_csv(whl_path, delimiter='\t', names=['x', 'y', 'x2', 'y2'])
    position_time = np.arange(0, position_df.index.stop) / position_fs
    position_df['t'] = position_time
    del position_df['x2']  # The coordinates of second LED. I just assume we only need the first one.
    del position_df['y2']
    useful_mask = np.invert((position_df.x == -1) | (position_df.y == -1))  # -1 means missing coordinate
    position_df = position_df[useful_mask].reset_index(drop=True)
    return position_df

def get_sr_from_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    sr_element = root.find('acquisitionSystem').find('samplingRate')
    sr = float(sr_element.text)
    if (sr == 32552) or (sr == 20000):
        return sr
    else:
        print('Wrong sampling rate encounter')
        raise

        
def fetch_xysp(position_df, tsp, position_fs=39.06):
    '''
    Args:
        position_df (pandas.DataFrame) : Dataframe that consists position information of the rat (x, y, t) in time series.
        tsp (numpy.darray) : 1d array of the spike times
        
    Returns:
    
    
    '''
    
    
    dist_threshold = position_fs/2

    t = np.array(position_df.t)
    x = np.array(position_df.x)
    y = np.array(position_df.y)


    dist_min_np = np.zeros(tsp.shape[0], dtype=np.int)
    dist_min_idx_np = np.zeros(tsp.shape[0], dtype=np.int)

    for i in range(tsp.shape[0]):  # for-loop has better performance here
        dist = np.abs(t - tsp[i])
        min_idx = np.argmin(dist)
        min_num = dist[min_idx]
        dist_min_np[i] = min_num
        dist_min_idx_np[i] = min_idx



    within_threshold_mask = (dist_min_np < dist_threshold)


    xsp = x[dist_min_idx_np]
    ysp = y[dist_min_idx_np]

    sp_df = pd.DataFrame({'tsp':tsp, 'xsp':xsp, 'ysp':ysp})

    sp_df = sp_df[within_threshold_mask].reset_index(drop=True)
    return sp_df
