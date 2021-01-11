# %% Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

from scipy.interpolate import interp1d
from pycircstat.descriptive import cdiff
from common.utils import load_pickle
from common.comput_utils import cos_scaled, linear_transform, get_transform_params, cos_scaled_2d
from common.Ensembles import RomaniEnsemble, InputEvaluatorCustom
from common.Environments import Simulation
from common.comput_utils import pair_dist_self, linear_circular_gauss_density
from common.correlogram import ThetaEstimator


def load_exp_trajectory(raw_df_pth):
    # %% Load Experiment dataframe
    df = load_pickle(raw_df_pth)

    # %% Extract Experiment trajectories
    dt = 1e-3
    target_speed = 2 * np.pi / 5  # rad/second
    num_trials = df.shape[0]

    all_t_list, all_pos_list = [], []

    for trial_i in range(num_trials):
        for ca_i in range(3):

            # Retrieve indata
            ca = 'CA%d' % (ca_i + 1)
            indata = df.loc[trial_i, ca + 'indata']
            if indata.shape[0] < 1:
                continue
            rat_t = indata['t'].to_numpy()
            rat_x = indata['x'].to_numpy()
            rat_y = indata['y'].to_numpy()

            # Scale xy to [0, 2pi]
            trans_x, trans_y, scaling_const = get_transform_params(rat_x, rat_y)
            rat_x = linear_transform(rat_x, trans_x, scaling_const)
            rat_y = linear_transform(rat_y, trans_y, scaling_const)

            # Scale time s.t. speed = target_speed
            current_speed = np.mean(np.sqrt(np.diff(rat_x) ** 2 + np.diff(rat_y) ** 2) / np.diff(rat_t))
            rat_t = rat_t * current_speed / target_speed
            new_speed = np.mean(np.sqrt(np.diff(rat_x) ** 2 + np.diff(rat_y) ** 2) / np.diff(rat_t))

            # Interpolate indata
            rat_t = rat_t - rat_t.min()
            t = np.arange(rat_t.min(), rat_t.max(), step=dt)
            xinterer = interp1d(rat_t, rat_x)
            yinterer = interp1d(rat_t, rat_y)
            x = xinterer(t)
            y = yinterer(t)
            pos = np.stack([x, y]).T  # (t, 2)

            all_t_list.append(t)
            all_pos_list.append(pos)
            break
        break

    for i in range(len(all_t_list) - 1):
        all_t_list[i + 1] += np.max(all_t_list[i]) + dt

    t = np.concatenate(all_t_list)  # Used name
    pos = np.concatenate(all_pos_list, axis=0)  # Used name
    current_speed = np.mean(np.sqrt(np.sum(np.square(np.diff(pos, axis=0)), axis=1)) / np.diff(t))

    print('Experiment trajectory')
    print('Speed = ', current_speed)
    print('Time from %0.4f to %0.4f' % (t.min(), t.max()))
    print('time shape = ', t.shape[0])
    return t, pos


def gen_setting(w_func, gt_func):
    ## Parameters
    neuron_theta1 = np.linspace(0, 2 * np.pi, num=32)
    neuron_theta2 = np.linspace(0, 2 * np.pi, num=32)
    thethe1, thethe2 = np.meshgrid(neuron_theta1, neuron_theta2)
    neuron_theta = np.stack([thethe1.reshape(-1), thethe2.reshape(-1)]).T  # (num_neurons, 2)

    J1 = 35
    J0 = 25
    I = -7
    I_L = 15
    I_theta = 5
    f_theta = 10

    # Neurons' positions & Weights
    pair_dist1 = pair_dist_self(neuron_theta[:, 0])
    pair_dist2 = pair_dist_self(neuron_theta[:, 1])

    w = J1 * (w_func(pair_dist1, pair_dist2)) - J0

    I_E = InputEvaluatorCustom(pos, neuron_theta, t, gt_func, f_theta, I_theta, I, I_L)
    I_period = 1 / f_theta
    I_phase = (np.mod(t - (I_period / 2), I_period) / I_period) * 2 * np.pi - np.pi

    # Ensemble
    ensemble_dict = {
        'm_rest': 0,
        'x_rest': 1,
        'tau': 10e-3,
        'tau_R': 0.8,
        'U': 0.8,
        'alpha': 1,
        'w': w
    }
    return neuron_theta, I_E, I_phase, ensemble_dict


def simulate(t, pos, neuron_theta, I_E, I_phase, ensemble_dict, max_duration=None, save_pth=None):
    num_neurons = neuron_theta.shape[0]
    simenv = Simulation(t, **dict(I_E=I_E))
    ensem = RomaniEnsemble(simenv, num_neurons, **ensemble_dict)
    indata_list = []
    spdata_list = []
    t_list = []
    tid_list = []
    m_list = []

    max_t = t.max()

    while ensem.simenv.check_end():

        time_each = ensem.simenv.current_time
        time_idx = ensem.simenv.current_idx
        tarr = np.array([time_idx, time_each])
        if max_duration:
            if time_each > max_duration:
                break

        _, m_each, firing_idx = ensem.state_update()

        # Insert activity, data = (id, tid, t, neuronidx, m), shape = (num_neurons, 4)
        m_list.append(m_each)
        tid_list.append(time_idx)
        t_list.append(time_each)

        # Insert Indata
        inarr = np.concatenate([tarr, pos[time_idx, :], I_phase[[time_idx]]])
        indata_list.append(inarr)

        # Insert SpikeData: id, neuronidx, neuronx, neurony, tidxsp, tsp, xsp, ysp, phasesp
        num_spikes = firing_idx.shape[0]
        if firing_idx.shape[0] > 0:
            neuronxy = neuron_theta[firing_idx, :]
            tidxsp = np.ones(firing_idx.shape[0]).astype(int) * time_idx
            tsp = t[tidxsp]
            xysp = pos[tidxsp, :]
            phasesp = I_phase[tidxsp]
            sparr = np.hstack([firing_idx.reshape(-1, 1), neuronxy, tidxsp.reshape(-1, 1),
                               tsp.reshape(-1, 1), xysp, phasesp.reshape(-1, 1)])
            spdata_list.append(sparr)

        if time_idx % 100 == 0:
            print('\r%0.3f/%0.3f' % (time_each, max_t), end='', flush=True)

    Activity = pd.DataFrame(dict(tid=tid_list, t=t_list, m=m_list))
    Indata = pd.DataFrame(np.vstack(indata_list), columns=['tid', 't', 'x', 'y', 'phase'])
    SpikeData = pd.DataFrame(np.vstack(spdata_list), columns=['neuronidx', 'neuronx', 'neurony', 'tidxsp',
                                                              'tsp', 'xsp', 'ysp', 'phasesp'])
    NeuronPos = pd.DataFrame(neuron_theta, columns=['neuronx', 'neurony'])

    simdata = dict(Activity=Activity, Indata=Indata, SpikeData=SpikeData, NeuronPos=NeuronPos)
    if save_pth:
        with open(save_pth, 'wb') as fh:
            pickle.dump(simdata, fh)
    return simdata

def w_cos_scaled(pairdist1, pairdist2):
    out = cos_scaled_2d(pairdist1, pairdist2)
    return out

def gt_cos_scaled(pos_t, neuron_pos):
    pairdist1, pairdist2 = pos_t[0] - neuron_pos[:, 0], pos_t[1] - neuron_pos[:, 1]
    out = cos_scaled_2d(pairdist1, pairdist2)
    return out




def part_pair_correlations(neuron_indices, SpikeData, NeuronPos):
    phase_finder = ThetaEstimator(0.005, 0.3, [5, 12])
    alllags = []
    allxdiff = []
    progress = 0
    for neui in neuron_indices:
        for neuj in neuron_indices:
            if neui == neuj:
                continue
            print('\r%d/%d' % (progress, neuron_indices.shape[0] ** 2), end='', flush=True)
            tsp1 = SpikeData[SpikeData['neuronidx'] == neui]['tsp'].to_numpy()
            tsp2 = SpikeData[SpikeData['neuronidx'] == neuj]['tsp'].to_numpy()
            _, phaselag, _ = phase_finder.find_theta_isi_hilbert([tsp1], [tsp2])
            x1 = NeuronPos[neui, 0]
            x2 = NeuronPos[neuj, 0]
            xdiff = x1 - x2
            # xdiff = cdiff(x1, x2)
            alllags.append(phaselag)
            allxdiff.append(xdiff)
            progress += 1
    print()
    lags = np.array(alllags)
    xdiffs = np.array(allxdiff)
    nonanmask = ~np.isnan(lags)
    lags = lags[nonanmask]
    xdiffs = xdiffs[nonanmask]
    return xdiffs, lags


def plot_part_pair_correlations(Activity, Indata, SpikeData, NeuronPos, save_plot_pth, tag):
    # Define time range
    tidx1, tidx2 = 1200, 3000
    tidx_arr = np.arange(tidx1, tidx2)

    # Organize data
    NeuronPos = NeuronPos[['neuronx', 'neurony']].to_numpy()  # (num_neurons, 2)
    m = np.stack(Activity['m'].to_numpy())  # (time, num_neurons)
    trajx, trajy = Indata['x'].to_numpy(), Indata['y'].to_numpy()
    trajxy = np.stack([trajx, trajy]).T
    trajt = Indata['t'].to_numpy()
    Iphase = Indata['phase'].to_numpy()

    # Get and plot neighbors
    neighbors = np.zeros(tidx_arr.shape)
    for idx in range(tidx_arr.shape[0]):
        minidx = np.argmin(np.linalg.norm(NeuronPos - trajxy[tidx_arr[idx],].reshape(1, 2), axis=1))
        if hasattr(minidx, '__iter__'):
            minidx = minidx[0]
        neighbors[idx] = minidx
    uniidx = np.unique(neighbors, return_index=True)[1]
    neighbors = np.array([neighbors[i] for i in sorted(uniidx)]).astype(int)

    # Generate more spikes
    dt = 1e-3
    SpikeData_more_dict = {col: [] for col in SpikeData.columns}
    for i in range(10):
        print('\r Generating spikes %d' % i, end='', flush=True)
        for j in range(tidx1, tidx2):
            instantr = m[j,] * dt
            rand_p = np.random.uniform(0, 1, size=instantr.shape[0])

            neuronidx = np.where(rand_p < instantr)[0]
            tsp = np.ones(neuronidx.shape[0]) * trajt[j]

            neuronx = NeuronPos[neuronidx, 0]
            neurony = NeuronPos[neuronidx, 1]
            tidxsp = np.ones(neuronidx.shape[0]).astype(int) * j
            xsp = np.ones(neuronidx.shape[0]) * trajx[j]
            ysp = np.ones(neuronidx.shape[0]) * trajy[j]
            phasesp = np.ones(neuronidx.shape[0]) * Iphase[j]

            SpikeData_more_dict['neuronidx'].append(neuronidx)
            SpikeData_more_dict['tsp'].append(tsp)
            SpikeData_more_dict['neuronx'].append(neuronx)
            SpikeData_more_dict['neurony'].append(neurony)
            SpikeData_more_dict['tidxsp'].append(tidxsp)
            SpikeData_more_dict['xsp'].append(xsp)
            SpikeData_more_dict['ysp'].append(ysp)
            SpikeData_more_dict['phasesp'].append(phasesp)
    print()
    tmp_spdict = dict()
    for key, val in SpikeData_more_dict.items():
        if key == 'id':
            continue
        tmp_spdict[key] = np.concatenate(val)
    SpikeData_more = pd.DataFrame(tmp_spdict)

    # Filter spikes
    mask = (SpikeData['tidxsp'] < tidx2) & (SpikeData['tidxsp'] >= tidx1) & SpikeData['neuronidx'].isin(neighbors)
    mask_more = (SpikeData_more['tidxsp'] < tidx2) & (SpikeData_more['tidxsp'] >= tidx1) & \
                SpikeData_more['neuronidx'].isin(neighbors)
    spdf = SpikeData[mask].reset_index(drop=True)
    spdf_mode = SpikeData_more[mask_more].reset_index(drop=True)

    # Pair correlations
    xdiffs, lags = part_pair_correlations(neighbors, spdf, NeuronPos)
    xdiffs_more, lags_more = part_pair_correlations(neighbors, spdf_mode, NeuronPos)

    # Density
    xx, yy, zz = linear_circular_gauss_density(xdiffs, lags, cir_kappa=4 * np.pi, lin_std=0.25,
                                               xbins=100, ybins=100, xbound=(xdiffs.min(), xdiffs.max()),
                                               ybound=(-np.pi, np.pi))
    xx_more, yy_more, zz_more = linear_circular_gauss_density(xdiffs_more, lags_more, cir_kappa=4 * np.pi, lin_std=0.25,
                                                              xbins=100, ybins=100,
                                                              xbound=(xdiffs_more.min(), xdiffs_more.max()),
                                                              ybound=(-np.pi, np.pi))

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].scatter(xdiffs, lags, alpha=0.5)
    ax[0, 1].pcolormesh(xx, yy, zz)
    ax[1, 0].scatter(xdiffs_more, lags_more, alpha=0.5)
    ax[1, 1].pcolormesh(xx_more, yy_more, zz_more)
    fig.suptitle('%s\nTop: Poisson 1 times Bottom: Poisson 100 times' % tag)
    fig.savefig(save_plot_pth)
    return None

def sanity_check(data):

    spdf = data['SpikeData']
    NeuronPos = data['NeuronPos'].to_numpy()

    selected_neurons = np.array([
        [0, 0], [0, np.pi], [0, 2*np.pi],
        [np.pi, 0], [np.pi, np.pi], [np.pi, 2 * np.pi],
        [2*np.pi, 0], [2*np.pi, np.pi], [2*np.pi, 2 * np.pi],
    ])
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))

    for neurons, ax_each in zip(selected_neurons, ax.flatten()):
        neuronidx = np.argmin(np.linalg.norm(NeuronPos - neurons.reshape(1, 2), axis=1))

        neurodf = spdf[spdf['neuronidx'] == neuronidx]

        xsp, ysp = neurodf['xsp'].to_numpy(), neurodf['ysp'].to_numpy()

        ax_each.scatter(xsp, ysp, alpha=0.3)
        ax_each.set_xlim(0, 2*np.pi)
        ax_each.set_ylim(0, 2*np.pi)
    fig.savefig('tmp.png')



if __name__ == '__main__':
    raw_df_pth = 'data/emankindata_processed.pickle'

    print('Loading experiment trajectory')
    t, pos = load_exp_trajectory(raw_df_pth)

    print('Generate setting of Square environment')
    neuron_theta, I_E, I_phase, ensemble_dict = gen_setting(w_cos_scaled, gt_cos_scaled)

    print('Simulating Squarematch environment')
    simdata = simulate(t, pos, neuron_theta, I_E, I_phase, ensemble_dict,
                        max_duration=None, save_pth='results/sim/raw/squarematch.pickle')

    sanity_check(load_pickle('results/sim/raw/squarematch.pickle'))





    # Activity, Indata, SpikeData, NeuronPos = simulate(t, pos, neuron_theta, I_E, I_phase, ensemble_dict,
    #                                                   max_duration=3.2, save_pth=None)
    # # Toroidal simulation
    # print('Generate setting of Toroidal environment')
    # neuron_theta, I_E, I_phase, ensemble_dict = gen_setting(w_cos, gt_cos)
    #
    # print('Simulating Toroidal environment')
    # _ = simulate(t, pos, neuron_theta, I_E, I_phase, ensemble_dict,
    #              max_duration=None,
    #              save_pth='results/sim/raw/toroidal.pickle')



    # plot_dir = 'result_plots/romani_square_debug'
    #
    # print('Pair correlation...')
    # plot_part_pair_correlations(Activity, Indata, SpikeData, NeuronPos,
    #                             save_plot_pth=os.path.join(plot_dir, 'cos_scaled_final.png'),
    #                             tag='cos_scaled_final')
