import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import polygon2mask
from skimage.measure import find_contours

from common.comput_utils import pair_diff, find_pair_times


def gauss2d(x, y, sig=3):
    exponent = -(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sig, 2))
    denom = 2 * np.pi * np.power(sig, 2)
    return np.exp(exponent) / denom


def get_occupancy(x, y, t):
    # Create meshgrid
    xmax, ymax = np.ceil(np.max(x)), np.ceil(np.max(y))
    xmin, ymin = np.floor(np.min(x)), np.floor(np.min(y))
    x_ax = np.arange(start=xmin, stop=xmax + 1, step=1)
    y_ax = np.arange(start=ymin, stop=ymax + 1, step=1)
    xx, yy = np.meshgrid(x_ax, y_ax)

    # Define chunks' indexes
    dt = t[1:] - t[:-1]
    idx = np.where(dt > 0.3)[0]
    if idx.shape[0] == 0:
        chunk_idxes = np.array([0, t.shape[0]])
    else:
        chunk_idxes = np.append(0, idx)
        chunk_idxes = np.append(chunk_idxes, t.shape[0])

    # Occupancy map
    occupancy = np.zeros(xx.shape)
    for idx_i in range(chunk_idxes.shape[0] - 1):
        start_idx, end_idx = chunk_idxes[idx_i], chunk_idxes[idx_i + 1]
        t_chunk = t[start_idx:end_idx]
        dt_chunk = t_chunk[1:] - t_chunk[0:-1]
        x_chunk = x[start_idx:end_idx - 1]
        y_chunk = y[start_idx:end_idx - 1]

        for i in range(x_chunk.shape[0]):
            occupancy += gauss2d(x_chunk[i] - xx, y_chunk[i] - yy) * dt_chunk[i]

    return xx, yy, occupancy


def placefield(xx, yy, occupancy, xsp, ysp, tsp):
    # Rate map
    freq = np.zeros(xx.shape)
    for i in range(tsp.shape[0]):
        freq = freq + gauss2d(xsp[i] - xx, ysp[i] - yy)

    # Place field map (divided by occupancy)
    rates = freq / (1e-4 + occupancy)

    return freq, rates


def spatial_information(rates, occ):
    occ_prob = occ / np.sum(occ)
    mar_rate = np.mean(rates)
    with np.errstate(divide='ignore', invalid='ignore'):
        logrand = np.log2(rates / mar_rate)
    logrand[np.isinf(logrand)] = 0
    logrand[np.isnan(logrand)] = 0
    integrand = rates * occ_prob * logrand
    info_persecond = np.sum(integrand)
    if mar_rate == 0:
        info_perspike = 0
    else:
        info_perspike = info_persecond / mar_rate

    return info_persecond, info_perspike


def get_occupancy_bins(x, y, t, binstep=10):
    # Create meshgrid
    xmax, ymax = np.ceil(np.max(x)), np.ceil(np.max(y))
    xmin, ymin = np.floor(np.min(x)), np.floor(np.min(y))
    x_ax = np.arange(start=xmin, stop=xmax + binstep, step=binstep)
    y_ax = np.arange(start=ymin, stop=ymax + binstep, step=binstep)

    dt = np.median(t[1:] - t[:-1])
    occ_H, _, _ = np.histogram2d(x, y, bins=(x_ax, y_ax))
    occ_H = occ_H * dt

    return x_ax, y_ax, occ_H


def placefield_bins(x_ax, y_ax, occ_H, xsp, ysp):
    freq_H, _, _ = np.histogram2d(xsp, ysp, bins=(x_ax, y_ax))
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_H = freq_H / occ_H
    rate_H[np.isnan(rate_H)] = 0
    rate_H[np.isinf(rate_H)] = 0
    return freq_H, rate_H


def complete_contourline(c_x, c_y, xbound, ybound):
    """

    Parameters
    ----------
    c_x
    c_y
    xbound: tuple
        In unit of the index of the array (not the actual x corodinates)
    ybound

    Returns
    -------

    """

    xmin, xmax = xbound
    ymin, ymax = ybound

    x0, x1, y0, y1 = c_x[0], c_x[-1], c_y[0], c_y[-1]

    if (x0 == x1) and (y0 == y1):  # Closed loop
        case = 1
    elif (x0 == x1) or (y0 == y1):  # Either x or y doesn't end at the start
        case = 2
        c_x = np.append(c_x, x0)
        c_y = np.append(c_y, y0)  # Complete the loop

    else:  # Both x and y are not ending at the start
        case = 3

        if (x0 == xmin) or (x0 == xmax):  # if x starts at the edge
            c_x = np.append(c_x, x0)
            c_y = np.append(c_y, y1)

        elif (x1 == xmin) or (x1 == xmax):  # if x ends at the edge
            c_x = np.append(c_x, x1)
            c_y = np.append(c_y, y0)

        elif (y0 == ymin) or (y0 == ymax):  # if y starts at the edge
            c_x = np.append(c_x, x1)
            c_y = np.append(c_y, y0)

        elif (y1 == ymin) or (y1 == ymax):  # if y ends at the edge
            c_x = np.append(c_x, x0)
            c_y = np.append(c_y, y1)
        else:
            raise

        if np.all(np.array([c_x[0], c_y[0]]) == np.array([c_x[-1], c_y[-1]])) is False:  # if not close loop
            c_x = np.append(c_x, c_x[0])
            c_y = np.append(c_y, c_y[0])

    return c_x, c_y, case



def segment_fields(xx, yy, freq, rate, area_thresh):
    xbound = (0, xx.shape[1]-1)
    ybound = (0, yy.shape[0]-1)

    normmap = rate / rate.max()
    # Padding to make sure contour is always closed
    padded_normmap = np.zeros((normmap.shape[0]+10, normmap.shape[1]+10))
    padded_normmap[5:-5, 5:-5] = normmap
    contours = find_contours(padded_normmap, level=0.2)

    for c_each in contours:
        c_rowi = c_each[:, 0] - 5  # y
        c_coli = c_each[:, 1] - 5  # x
        c_rowi, c_coli = np.clip(c_rowi, a_min=ybound[0], a_max=ybound[1]), np.clip(c_coli, a_min=xbound[0], a_max=xbound[1])
        c_coli, c_rowi, case = complete_contourline(c_coli, c_rowi, xbound=xbound, ybound=ybound)
        mask = polygon2mask(xx.shape, np.stack([c_rowi, c_coli]).T)  # Remember to transpose!
        c_rowi = np.around(c_rowi).astype(int)
        c_coli = np.around(c_coli).astype(int)
        c_x = xx[c_rowi, c_coli]
        c_y = yy[c_rowi, c_coli]
        xyval = np.stack([c_x, c_y]).T
        area = mask.sum()
        if area < 1:
            continue
        meanrate_in = np.mean(rate[mask])
        meanrate_out = np.mean(rate[np.invert(mask)])
        peak_freq_in = freq[mask].max()

        if (area > area_thresh) and (meanrate_in > meanrate_out) and (peak_freq_in > 1):
            yield mask, xyval



def identify_pairs(fielddf, spdf, xinterp, yinterp, x_ax, y_ax, trange):
    num_fields = fielddf.shape[0]
    for fieldid1 in range(num_fields):
        for fieldid2 in range(fieldid1 + 1, num_fields):
            cellid1, mask1 = fielddf.loc[fieldid1, ['cellid', 'mask']]
            tsp1, pf1 = spdf.loc[cellid1, ['tsp', 'pf']]
            tsp_in1 = tsp1[(tsp1 < trange[0]) & (tsp1 > trange[1])]
            xsp_in1, ysp_in1 = xinterp(tsp_in1), yinterp(tsp_in1)
            ind_x1 = np.argmin(np.square(pair_diff(xsp_in1, x_ax)), axis=1)
            ind_y1 = np.argmin(np.square(pair_diff(ysp_in1, y_ax)), axis=1)
            tok1 = np.zeros(ind_x1.shape[0]).astype(bool)
            for i in range(ind_x1.shape[0]):
                tok1[i] = mask1[ind_y1[i], ind_x1[i]]
            tsp_inmask1 = tsp_in1[tok1]

            cellid2, mask2 = fielddf.loc[fieldid2, ['cellid', 'mask']]
            tsp2, pf2 = spdf.loc[cellid2, ['tsp', 'pf']]
            tsp_in2 = tsp2[(tsp2 < trange[0]) & (tsp2 > trange[1])]
            xsp_in2, ysp_in2 = xinterp(tsp_in2), yinterp(tsp_in2)
            ind_x2 = np.argmin(np.square(pair_diff(xsp_in2, x_ax)), axis=1)
            ind_y2 = np.argmin(np.square(pair_diff(ysp_in2, y_ax)), axis=1)
            tok2 = np.zeros(ind_x2.shape[0]).astype(bool)
            for i in range(ind_x2.shape[0]):
                tok2[i] = mask2[ind_y2[i], ind_x2[i]]
            tsp_inmask2 = tsp_in2[tok2]

            pairtsp_idx1, pairtsp_idx2 = find_pair_times(tsp_inmask1, tsp_inmask2)

            num_pairedspikes = pairtsp_idx1.shape[0] + pairtsp_idx2.shape[0]
            mask_and = mask1 & mask2
            if (num_pairedspikes > 16) & (mask_and.sum() > 0):
                yield (fieldid1, fieldid2)