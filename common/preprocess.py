import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import polygon2mask

def gauss2d(x, y, sig = 3):
    exponent = -(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sig, 2))
    denom = 2 * np.pi * np.power(sig, 2)
    return np.exp(exponent)/denom

def placefield(x, y, t, xsp, ysp, tsp):

    # Create meshgrid
    xmax, ymax = np.ceil(np.max(np.abs(x))), np.ceil(np.max(np.abs(y)))
    xmin, ymin = -xmax, -ymax
    x_ax = np.arange(start=xmin, stop=xmax+1, step=1)
    y_ax = np.arange(start=ymin, stop=ymax+1, step=1)
    xx, yy = np.meshgrid(x_ax, y_ax)

    # Define chunks' indexes
    dt = t[1:] - t[:-1]
    idx = np.where(dt > 0.3)[0] - 1
    if idx.shape[0] == 0:
        chunk_idxes = np.array([0, t.shape[0]])
    else:
        chunk_idxes = np.append(0, idx)
        chunk_idxes = np.append(chunk_idxes, t.shape[0])

    # Occupancy map
    occupancy = np.zeros(xx.shape)
    for idx_i in range(chunk_idxes.shape[0]-1):
        start_idx, end_idx = chunk_idxes[idx_i], chunk_idxes[idx_i + 1]
        t_chunk = t[start_idx:end_idx]
        dt_chunk = t_chunk[1:] - t_chunk[0:-1]
        x_chunk = x[start_idx:end_idx-1]
        y_chunk = y[start_idx:end_idx-1]

        for i in range(x_chunk.shape[0]):
            occupancy += gauss2d(x_chunk[i] - xx, y_chunk[i] - yy)*dt_chunk[i]

    # Rate map
    freq = np.zeros(xx.shape)
    for i in range(tsp.shape[0]):
        freq = freq + gauss2d(xsp[i] - xx, ysp[i]-yy)

    # Place field map (divided by occupancy)
    rates = freq / (1e-4 + occupancy)

    # Normalized occupancy
    occ = occupancy/np.sum(occupancy)

    return xx, yy, occ, freq, rates


def clines_border_check(clines, xx, yy):
    '''
    Args:
        clines (np.darray) : vertices of the contour with shape (N, 2), N = number of samples
        xx (np.darray) : x's corrdinates in 2D meshgrid with shape (Y, X)
        yy (np.darray) : y's corrdinates in 2D meshgrid with shape (Y, X)
        
    Returns:
        new_clines (np.darray) : Vertices of the contour with shape (N + M, 2), W = number of newly appended points
            
    '''
    x_bound_min, x_bound_max = xx[0, 0], xx[0, -1]
    y_bound_min, y_bound_max = yy[0, 0], yy[-1, 0]
    
    x_start, x_end = clines[0, 0], clines[-1, 0]
    y_start, y_end = clines[0, 1], clines[-1, 1]
    
    append_1 = None
    append_2 = [x_start, y_start]
    
    new_clines = clines.copy()
    
    if (x_start != x_end) or (y_start != y_end):
        if (x_start == x_end) or (y_start == y_end):
            append_1 = [x_start, y_start]
        else:
            if (x_start == x_bound_min) or (x_start == x_bound_max):
                append_1 = [x_start, y_end]
            elif (x_end == x_bound_min) or (x_end == x_bound_max):
                append_1 = [x_end, y_start]
            elif (y_start == x_bound_min) or (y_start == x_bound_max):
                append_1 = [x_end, y_start]
            else:
                append_1 = [x_start, y_end]
    
    if append_1:
        if append_1 == append_2:
            new_clines = np.concatenate([new_clines, np.array(append_1).reshape(1, 2)], axis=0)
    
    if np.sum(new_clines[-1, :] == np.array(append_2)) <2:
        new_clines = np.concatenate([new_clines, np.array(append_2).reshape(1, 2)], axis=0)
    
            
    return new_clines
    
    
    
class PlaceFieldHunter:
    def __init__(self):
        self.fields_dict = dict(masks=[], 
                               clines=[], 
                               xx=[], 
                               yy=[], 
                               pf_occ=[], 
                               pf_freq=[], 
                               pf_rates=[], 
                               neuron_unit=[])
    
    def append_pfs(self, x, y, t, xsp, ysp, tsp, neuron_unit, pf_threshold=[0.2]):
        
        xx, yy, pf_occ, pf_freq, pf_rates = placefield(x, y, t, xsp, ysp, tsp)
        norm_map_rates = pf_rates / np.max(pf_rates)
        cs = plt.contour(xx, yy, norm_map_rates, levels=pf_threshold);
        plt.close()
        num_contours = len(cs.allsegs[0])
        
        masks_temp = []
        clines_temp = []

        if num_contours > 0:
            for contour_idx in range(num_contours):
                clines = cs.allsegs[0][contour_idx]
                new_clines = clines_border_check(clines, xx, yy)
                mask = polygon2mask(xx.shape, new_clines[:, ::-1] - np.array([xx[0,0], yy[0, 0]]).reshape(1, 2))

                if (np.max(pf_rates * mask) > 1) and (np.sum(mask) > 25):
                    masks_temp.append(mask)
                    clines_temp.append(new_clines)

        num_masks = len(masks_temp)
        problem_mask_idx = []
        if num_masks > 1:
            for i in range(num_masks):
                mask_1 = masks_temp[i]
                for k in range(num_masks - i -1):
                    j = k + 1 + i
                    mask_2 = masks_temp[j]

                    area_union = np.sum((mask_1 + mask_2) > 0)

                    if area_union == np.sum(mask_1):
                        problem_mask_idx.append(j)
                    if area_union == np.sum(mask_2):
                        problem_mask_idx.append(i)
        if num_masks > 0:
            for i in range(num_masks):
                if i in problem_mask_idx:
                    continue
                else:
                    self.fields_dict['masks'].append(masks_temp[i])
                    self.fields_dict['clines'].append(clines_temp[i])
                    self.fields_dict['xx'].append(xx)
                    self.fields_dict['yy'].append(yy)
                    self.fields_dict['pf_occ'].append(pf_occ)
                    self.fields_dict['pf_freq'].append(pf_freq)
                    self.fields_dict['pf_rates'].append(pf_rates)
                    self.fields_dict['neuron_unit'].append(neuron_unit)

            
    def get_pf_df(self):
        return pd.DataFrame(self.fields_dict)


    
def pf_df_visualizer(pf_df, field_num):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    masks, clines, xx, yy, pf_rates = pf_df.iloc[field_num][['masks', 'clines', 'xx', 'yy', 'pf_rates']]
    ax[0].pcolormesh(xx, yy, masks)
    ax[1].plot(clines[:, 0], clines[:, 1])
    ax[1].set_xlim(np.min(xx), np.max(xx))
    ax[1].set_ylim(np.min(yy), np.max(yy))
    ax[2].pcolormesh(xx, yy, pf_rates)
    return fig, ax