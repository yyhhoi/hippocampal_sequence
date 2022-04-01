# Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from os.path import join
from scipy.interpolate import interp1d
from common.utils import load_pickle
from common.comput_utils import linear_transform, get_transform_params, cos_scaled_2d
from common.Ensembles import RomaniEnsemble, InputEvaluatorCustom
from common.Environments import Simulation
from common.comput_utils import pair_dist_self


def determine_thresholds(simdata, singledf):

    # Determine vthresh, sthresh and subsample_fraction
    pass

def w_cos_scaled(pairdist1, pairdist2):
    out = cos_scaled_2d(pairdist1, pairdist2)
    return out


def gt_cos_scaled(pos_t, neuron_pos):
    pairdist1, pairdist2 = pos_t[0] - neuron_pos[:, 0], pos_t[1] - neuron_pos[:, 1]
    out = cos_scaled_2d(pairdist1, pairdist2)
    return out


def load_exp_trajectory(df):
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

    t_final = np.concatenate(all_t_list)
    pos_final = np.concatenate(all_pos_list, axis=0)
    current_speed = np.mean(np.sqrt(np.sum(np.square(np.diff(pos_final, axis=0)), axis=1)) / np.diff(t_final))

    print('Experiment trajectory')
    print('Speed = ', current_speed)
    print('Time from %0.4f to %0.4f' % (t_final.min(), t_final.max()))
    print('time shape = ', t_final.shape[0])
    return t_final, pos_final, (trial_i, ca)


def gen_setting(t, pos, w_func, gt_func):
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


def simulate(t, pos, neuron_theta, I_E, I_phase, ensemble_dict, max_duration=None):
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
            tidxsp = np.ones(firing_idx.shape[0]).astype(int) * time_idx
            sparr = np.hstack([firing_idx.reshape(-1, 1), tidxsp.reshape(-1, 1)])
            spdata_list.append(sparr)

        if time_idx % 100 == 0:
            print('\r%0.3f/%0.3f' % (time_each, max_t), end='', flush=True)

    Activity = pd.DataFrame(dict(tid=tid_list, t=t_list, m=m_list))
    Indata = pd.DataFrame(np.vstack(indata_list), columns=['tid', 't', 'x', 'y', 'phase'])
    SpikeData = pd.DataFrame(np.vstack(spdata_list), columns=['neuronidx', 'tidxsp'])
    NeuronPos = pd.DataFrame(neuron_theta, columns=['neuronx', 'neurony'])

    simdata = dict(Activity=Activity, Indata=Indata, SpikeData=SpikeData, NeuronPos=NeuronPos)

    return simdata


def sanity_check(data, save_pth):
    spdf = data['SpikeData']
    NeuronPos = data['NeuronPos'].to_numpy()
    indatadf = data['Indata']
    indatadf.index = indatadf.tid

    selected_neurons = np.array([
        [0, 0], [0, np.pi], [0, 2 * np.pi],
        [np.pi, 0], [np.pi, np.pi], [np.pi, 2 * np.pi],
        [2 * np.pi, 0], [2 * np.pi, np.pi], [2 * np.pi, 2 * np.pi],
    ])
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))

    for neurons, ax_each in zip(selected_neurons, ax.flatten()):
        neuronidx = np.argmin(np.linalg.norm(NeuronPos - neurons.reshape(1, 2), axis=1))

        neurodf = spdf[spdf['neuronidx'] == neuronidx]
        tidxsp = neurodf['tidxsp'].to_numpy()
        xsp = indatadf.loc[tidxsp, 'x'].to_numpy()
        ysp = indatadf.loc[tidxsp, 'y'].to_numpy()

        ax_each.plot(indatadf['x'], indatadf['y'], linewidth=0.2, c='gray')
        ax_each.scatter(xsp, ysp, alpha=0.3, marker='.', zorder=3.2)
        ax_each.set_xlim(0, 2 * np.pi)
        ax_each.set_ylim(0, 2 * np.pi)
    fig.savefig(save_pth, dpi=200)


def main():
    input_df_pth = 'data/emankindata_processed_withwave.pickle'
    output_dict_dir = 'results/sim/raw'
    output_dict_name = 'squarematch.pickle'

    print('Load experiment data')
    input_df = pd.read_pickle(input_df_pth)

    print('Extract experiment trajectory')
    t, pos, _ = load_exp_trajectory(input_df)

    print('Generate setting of Square environment')
    neuron_theta, I_E, I_phase, ensemble_dict = gen_setting(t, pos, w_cos_scaled, gt_cos_scaled)

    print('Simulate Squarematch environment')
    simdata = simulate(t, pos, neuron_theta, I_E, I_phase, ensemble_dict, max_duration=None)

    print('Save simulated results')
    with open(join(output_dict_dir, output_dict_name), 'wb') as fh:
        pickle.dump(simdata, fh)

    print('Sanity check')
    sanity_check(data=load_pickle(join(output_dict_dir, output_dict_name)),
                 save_pth=join(output_dict_dir, output_dict_name + '.png'))


if __name__ == '__main__':
    main()
