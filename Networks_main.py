import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import torch
from scipy.interpolate import interp1d
from scipy.stats import ranksums, chi2_contingency
from torch.distributions.von_mises import VonMises
from scipy.integrate import odeint
from pycircstat.descriptive import cdiff

from common.comput_utils import linear_gauss_density, segment_passes, compute_straightness
from common.script_wrappers import DirectionalityStatsByThresh
from common.shared_vars import total_figw, ticksize, fontsize, dpi
from common.utils import load_pickle, stat_record
from common.visualization import customlegend


def cal_hd(x, y):
    hd = np.angle(np.diff(x) + 1j * np.diff(y))  # angle is converted to (-pi, pi)
    hd = np.append(hd, hd[-1])
    return hd


def extract_exp_trajectory():
    """ Extract one session of behavioural and LFP data for simulation
    """
    print('Extract one session of behavioural and LFP data for simulation')
    raw_df_pth = 'data/emankindata_processed_withwave.pickle'
    df = pd.read_pickle(raw_df_pth)

    num_trials = df.shape[0]
    for trial_i in range(num_trials):
        for ca_i in range(3):
            # Retrieve indata
            ca = 'CA%d' % (ca_i + 1)
            indata = df.loc[trial_i, ca + 'indata']
            wave = df.loc[trial_i, 'wave']
            if indata.shape[0] < 1:
                continue

            wave_t = wave['tax']
            rat_x = indata['x'].to_numpy()
            rat_y = indata['y'].to_numpy()
            rat_t = indata['t'].to_numpy()
            dt = 1e-3

            # Interpolate indata
            min_rat_t = rat_t.min()
            new_wave_t = wave_t - min_rat_t
            t_offseted = rat_t - min_rat_t

            new_rat_t = np.arange(t_offseted.min(), t_offseted.max(), step=dt)
            xinterer = interp1d(t_offseted, rat_x)
            yinterer = interp1d(t_offseted, rat_y)
            new_rat_x = xinterer(new_rat_t)
            new_rat_y = yinterer(new_rat_t)

            behdf = pd.DataFrame({'x': new_rat_x,
                                  'y': new_rat_y,
                                  't': new_rat_t,
                                  'angle': cal_hd(new_rat_x, new_rat_y)})
            wave['tax'] = new_wave_t
            wavedf = pd.DataFrame(wave)
            behdf.to_pickle('data/test_traj/behdf.pickle')
            wavedf.to_pickle('data/test_traj/wavedf.pickle')
            return


def izhikevich(v, u, gex, gexamp, w, tau_ex, Iext=0, a=0.02, b=0.2, c=-65, d=2,
               vthresh=-40):
    # Fire and reset
    fidx = torch.where(v > vthresh)[0]
    v[fidx] = c
    u[fidx] = u[fidx] + d

    # Excitatory / inhibitory conductance
    dgexdt = -gex / tau_ex + torch.sum(w[:, fidx], dim=1)

    # Soma
    Isyn = gexamp * gex
    dvdt = 0.04 * v ** 2 + 5 * v + 140 - u + Iext + Isyn
    dudt = a * (b * v - u)
    return (dvdt, dudt, dgexdt), fidx


def get_nidx(x, y, a, xxtun, yytun, aatun):
    nidx = torch.argmin((xxtun - x) ** 2 + (yytun - y) ** 2 + (aatun - a) ** 2)
    return nidx.item()
def get_nidx_np(x, y, a, xxtun, yytun, aatun):
    nidx = np.argmin((xxtun - x) ** 2 + (yytun - y) ** 2 + (aatun - a) ** 2)
    return nidx.item()

def eval_Isensory(amp, x, y, angle, xxtun, yytun, aatun):
    field_r = 10  # diamter 20 cm
    angle_bound = np.pi /2
    xdist = ((x - xxtun) ** 2) / (field_r ** 2)
    ydist = ((y - yytun) ** 2) / (field_r ** 2)
    adist = (cdiff(angle, aatun) ** 2 )/ (angle_bound ** 2)
    I_sen = amp * np.exp(- xdist - ydist - adist)
    return I_sen


def create_w(n, J):
    w = torch.ones((n, n)) * J
    return w


def pilot(plot_dir):

    # Create trajectory, running from (-40, 0) to (40, 0)
    aver_v = 15*1e-3  # cm per ms
    # total_tidx = int(80/aver_v)  # 5333 = 2666 + 2667
    # behx = np.concatenate([np.linspace(-20, 20, 2666), np.linspace(20, -20, 2667)])
    total_tidx = 1600*3
    behx = np.concatenate([np.linspace(-20, 20, 1600),
                           np.linspace(-20, 20, 1600),
                           np.linspace(-20, 20, 1600)])
    behy = np.zeros(total_tidx)


    beha = cal_hd(behx, behy)
    beht = np.linspace(0, total_tidx*1e-3, total_tidx)
    behdf = pd.DataFrame(dict(x=behx, y=behy, angle=beha, t=beht))



    dt = 1
    xmin, xmax, nx = -40, 40, 20
    ymin, ymax, ny = -40, 40, 20
    amin, amax, na = -np.pi, np.pi, 8

    n_ex = nx * ny * na
    n_total = n_ex

    xtun = torch.linspace(xmin, xmax, nx)
    ytun = torch.linspace(ymin, ymax, ny)
    atun = torch.linspace(amin, amax, na)

    xxtun, yytun, aatun = torch.meshgrid(xtun, ytun, atun)
    xxtun, yytun, aatun = xxtun.flatten(), yytun.flatten(), aatun.flatten()

    v, u = torch.ones(n_total) * -65, torch.zeros(n_total)
    gex= torch.zeros(n_total)
    gexamp = 1e-2
    tau_ex, tau_in = 5, 2.5
    w = create_w(n_total, 0)

    # LTP / LTD dynamics
    xltp, xltd = torch.zeros(n_total), torch.zeros(n_total)
    xltps = torch.zeros(n_total)
    tau_ltp, tau_ltd = 10, 10
    tau_ltp_slow = 30
    dltp_amp, dltd_amp = 1, 1
    dltps_amp = 1
    wmax = 2
    dw_amp = 1e-1


    spdf_dict = {"tidxsp": [], "fidx": []}
    states = dict(v=torch.zeros((total_tidx, n_total)),
                  Isen=torch.zeros((total_tidx, n_total)),
                  xltp=torch.zeros((total_tidx, n_total)),
                  xltd=torch.zeros((total_tidx, n_total)))
    I_base = -6
    Isen_amp = 8
    theta_amp = 8
    theta_f = 9
    theta_T = 1/theta_f
    behdf['phase'] = np.mod(behdf.t + 0.5*theta_T, theta_T)/theta_T * 2 * np.pi

    selected_xya1, selected_xya2 = np.array([-2, 0, 0]), np.array([2, 0, 0])
    prenidx = get_nidx(*selected_xya1, xxtun, yytun, aatun)
    postnidx = get_nidx(*selected_xya2, xxtun, yytun, aatun)
    selected_w = torch.zeros((total_tidx, 2))  # 0 = pre-post, 1 = post-pre
    for tidx in range(total_tidx):
        print('Sim %d/%d' % (tidx, behdf.shape[0]))

        behx, behy, beht, behangle = behdf.loc[tidx, ['x', 'y', 't', 'angle']]
        I_sen = eval_Isensory(Isen_amp, behx, behy, behangle, xxtun, yytun, aatun)

        I_theta = float(theta_amp * np.cos(2 * np.pi * theta_f * beht))
        Iext = I_base + I_theta + I_sen
        (dvdt, dudt, dgexdt), fidx = izhikevich(v, u, gex, gexamp, w, tau_ex, Iext=Iext)
        numfidx = fidx.shape[0]

        # Soma update
        v += dvdt * dt
        u += dudt * dt
        gex += dgexdt * dt

        # Synaptic dynamics update
        dxltpdt = -xltp/tau_ltp
        dxltpdt[fidx] = dxltpdt[fidx] + dltp_amp
        xltp += dxltpdt * dt

        dxltddt = -xltd/tau_ltd
        dxltddt[fidx] = dxltddt[fidx] + dltd_amp
        xltd += dxltddt * dt

        post_dw = (dw_amp * xltp).reshape(1, -1) * (wmax - w[fidx, :]) * dt * xltps[fidx].reshape(-1, 1)
        w[fidx, :] = w[fidx, :] + post_dw
        w[:, fidx] = w[:, fidx] + (dw_amp * xltd).reshape(-1, 1) * (0 - w[:, fidx]) * dt

        dxltpsdt = -xltps/tau_ltp_slow
        xltps[fidx] = dxltpsdt[fidx] + dltps_amp
        xltps += dxltpsdt * dt


        # Append data
        states['v'][tidx, :] = v.clone()
        states['Isen'][tidx, :] = I_sen.clone()
        states['xltp'][tidx, :] = xltp.clone()
        states['xltd'][tidx, :] = xltd.clone()
        num_firing = fidx.shape[0]
        spdf_dict['tidxsp'].extend([tidx] * num_firing)
        spdf_dict['fidx'].extend(fidx.tolist())
        selected_w[tidx, :] = torch.tensor([w[postnidx, prenidx], w[prenidx, postnidx]])

    spdf = pd.DataFrame(spdf_dict)


    fig_all, ax_all = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

    d1 = np.linalg.norm(behdf[['x', 'y']].to_numpy() - selected_xya1[0:2].reshape(1, 2), axis=1)
    d2 = np.linalg.norm(behdf[['x', 'y']].to_numpy() - selected_xya2[0:2].reshape(1, 2), axis=1)


    tidxsp_pre = spdf[spdf.fidx == prenidx].tidxsp.to_numpy()
    tidxsp_post = spdf[spdf.fidx == postnidx].tidxsp.to_numpy()
    phase_lines = np.where(np.diff(behdf['phase']) < -5)[0]
    c_pre, c_post, csp_pre, csp_post = 'r', 'b', 'coral', 'skyblue'
    # d1, d2, angle
    ax_all[0].plot(d1, c=c_pre, alpha=0.5)
    ax_all[0].eventplot(tidxsp_pre, lineoffsets=7.5, linelengths=2.5, colors=csp_pre)
    ax_all[0].plot(d2, c=c_post, alpha=0.5)
    ax_all[0].eventplot(tidxsp_post, lineoffsets=5, linelengths=2.5, colors=csp_post)
    ax_angle = ax_all[0].twinx()
    ax_angle.plot(np.rad2deg(behdf['angle']), c='m', alpha=0.5)

    # Soma voltage
    ax_all[1].plot(states['v'][:, prenidx], c=c_pre, alpha=0.5)
    ax_all[1].eventplot(tidxsp_pre, lineoffsets=-85, linelengths=5, colors=csp_pre)
    ax_all[1].plot(states['v'][:, postnidx], c=c_post, alpha=0.5)
    ax_all[1].eventplot(tidxsp_post, lineoffsets=-90, linelengths=5, colors=csp_post)
    ax_all[2].set_ylabel('Soma voltage')

    # xltp
    ax_all[2].plot(states['xltp'][:, prenidx].numpy(), c=c_pre, alpha=0.5)
    ax_all[2].eventplot(tidxsp_pre, lineoffsets=4, linelengths=1, colors=csp_pre)
    ax_all[2].plot(states['xltp'][:, postnidx].numpy(), c=c_post, alpha=0.5)
    ax_all[2].eventplot(tidxsp_post, lineoffsets=3, linelengths=1, colors=csp_post)
    ax_all[2].set_ylabel('LTP trace')

    # xltd
    ax_all[3].plot(states['xltd'][:, prenidx].numpy(), c=c_pre, alpha=0.5)
    ax_all[3].eventplot(tidxsp_pre, lineoffsets=4, linelengths=1, colors=csp_pre)
    ax_all[3].plot(states['xltd'][:, postnidx].numpy(), c=c_post, alpha=0.5)
    ax_all[3].eventplot(tidxsp_post, lineoffsets=3, linelengths=1, colors=csp_post)
    ax_all[3].set_ylabel('LTD trace')

    # W
    ax_all[4].plot(selected_w[:, 0].numpy(), c=c_pre, alpha=0.5) # Pre post
    ax_all[4].eventplot(tidxsp_pre, lineoffsets=0.5, linelengths=0.5, colors=csp_pre)
    ax_all[4].plot(selected_w[:, 1].numpy(), c=c_post, alpha=0.5)  # post-pre
    ax_all[4].eventplot(tidxsp_post, lineoffsets=0, linelengths=0.5, colors=csp_post)
    ax_all[4].set_ylabel('Wij & Wji')

    for idx in range(phase_lines.shape[0]-1):
        if idx % 2 == 0:
            ax_all[0].axvspan(xmin=phase_lines[idx], xmax=phase_lines[idx+1], alpha=0.2)
            ax_all[1].axvspan(xmin=phase_lines[idx], xmax=phase_lines[idx+1], alpha=0.2)
            ax_all[2].axvspan(xmin=phase_lines[idx], xmax=phase_lines[idx+1], alpha=0.2)
            ax_all[3].axvspan(xmin=phase_lines[idx], xmax=phase_lines[idx+1], alpha=0.2)
            ax_all[4].axvspan(xmin=phase_lines[idx], xmax=phase_lines[idx+1], alpha=0.2)


    fig_all.savefig(join(plot_dir, 'learning.png'))


def simulate():

    # Load trajectory
    behdf = pd.read_pickle('data/test_traj/behdf.pickle')

    total_tidx = behdf.shape[0]

    dt = 1
    xmin, xmax, nx = -40, 40, 20
    ymin, ymax, ny = -40, 40, 20
    amin, amax, na = -np.pi, np.pi, 8

    n_ex = nx * ny * na
    n_total = n_ex

    xtun = torch.linspace(xmin, xmax, nx)
    ytun = torch.linspace(ymin, ymax, ny)
    atun = torch.linspace(amin, amax, na)

    xxtun, yytun, aatun = torch.meshgrid(xtun, ytun, atun)
    xxtun, yytun, aatun = xxtun.flatten(), yytun.flatten(), aatun.flatten()

    v, u = torch.ones(n_total) * -65, torch.zeros(n_total)


    # Synaptic parameters
    gex= torch.zeros(n_total)
    gexamp = 1e-2
    tau_ex = 5
    w = create_w(n_total, 0)

    # LTP / LTD dynamics
    xltp, xltd = torch.zeros(n_total), torch.zeros(n_total)
    xltps = torch.zeros(n_total)
    tau_ltp, tau_ltd = 10, 10
    tau_ltp_slow = 30
    dltp_amp, dltd_amp = 1, 1
    dltps_amp = 1
    wmax = 2
    dw_amp = 1e-1


    # Theta input
    theta_amp = 8
    theta_f = 9
    theta_T = 1/theta_f
    behdf['phase'] = np.mod(behdf.t + 0.5*theta_T, theta_T)/theta_T * 2 * np.pi

    # Base input
    I_base = -6

    # Sensory input
    Isen_amp = 8

    spdf_dict = {"tidxsp": [], "neuronidx": []}


    for tidx in range(total_tidx):
        print('Sim %d/%d' % (tidx, behdf.shape[0]))

        behx, behy, beht, behangle = behdf.loc[tidx, ['x', 'y', 't', 'angle']]
        I_sen = eval_Isensory(Isen_amp, behx, behy, behangle, xxtun, yytun, aatun)

        I_theta = float(theta_amp * np.cos(2 * np.pi * theta_f * beht))
        Iext = I_base + I_theta + I_sen
        (dvdt, dudt, dgexdt), fidx = izhikevich(v, u, gex, gexamp, w, tau_ex, Iext=Iext)

        # Soma update
        v += dvdt * dt
        u += dudt * dt
        gex += dgexdt * dt

        # Synaptic dynamics update
        dxltpdt = -xltp/tau_ltp
        dxltpdt[fidx] = dxltpdt[fidx] + dltp_amp
        xltp += dxltpdt * dt

        dxltddt = -xltd/tau_ltd
        dxltddt[fidx] = dxltddt[fidx] + dltd_amp
        xltd += dxltddt * dt

        post_dw = (dw_amp * xltp).reshape(1, -1) * (wmax - w[fidx, :]) * dt * xltps[fidx].reshape(-1, 1)
        w[fidx, :] = w[fidx, :] + post_dw
        w[:, fidx] = w[:, fidx] + (dw_amp * xltd).reshape(-1, 1) * (0 - w[:, fidx]) * dt

        dxltpsdt = -xltps/tau_ltp_slow
        xltps[fidx] = dxltpsdt[fidx] + dltps_amp
        xltps += dxltpsdt * dt

        num_firing = fidx.shape[0]
        spdf_dict['tidxsp'].extend([tidx] * num_firing)
        spdf_dict['neuronidx'].extend(fidx.tolist())

    spdf = pd.DataFrame(spdf_dict)
    NeuronPos = pd.DataFrame(dict(neuronx=xxtun, neurony=yytun, neurona=aatun))
    simdata = dict(Indata=behdf, SpikeData=spdf, NeuronPos=NeuronPos)

    save_pth = join('results/network_proj/sim_result_dwdt.pickle')
    with open(save_pth, 'wb') as fh:
        pickle.dump(simdata, fh)
    return







def gen_IF_curve(plot_dir):

    dt = 1
    n = 1
    allI = np.arange(0, 20, step=0.1)

    allf = []
    for I_each in allI:
        print('I = %0.2fA' % I_each)
        alltidx = []
        v, u = torch.ones(n) * -65, torch.zeros(n)
        gex = torch.zeros(n)
        w = torch.zeros((n ,n))
        for tidx in range(1000):

            (dvdt, dudt, dgexdt), fidx = izhikevich(v, u, gex, 1, w, 1, Iext=I_each)
            v += dvdt * dt
            u += dudt * dt
            if fidx.shape[0]>0:
                alltidx.append(tidx)

        if len(alltidx) == 0:
            allf.append(0)
        else:
            T = np.mean(np.diff(alltidx) * 1e-3)
            f = 1/T

            allf.append(f)

    allf = np.array(allf)
    max0idx = np.max(np.where(allf == 0)[0])
    fig, ax = plt.subplots()
    ax.plot(allI, allf)
    ax.vlines(allI[max0idx], ymin=0, ymax=np.max(allf), colors='r')
    ax.set_title('Burfication: I=%0.2f (+- 0.1)'%(allI[max0idx]))
    fig.savefig(join(plot_dir, 'IF_curve.png'))


def check_tunning(plot_dir):
    field_r = 10  # diamter 20 cm
    angle_bound = np.pi /2

    xtun = np.linspace(-80, 80, 20)
    atun = np.linspace(-np.pi, np.pi, 8)

    xout = np.exp( - (xtun**2)/(field_r**2) )
    aout = np.exp( - (atun**2)/(angle_bound**2) )

    fig, ax = plt.subplots(2, 1, figsize=(4, 8))

    ax[0].plot(xtun, xout)
    ax[0].set_yticks(np.arange(0, 1.1, 0.1))
    ax[0].set_xticks([-20, -10, 0, 10, 20])
    ax[1].plot(atun, aout)
    ax[1].set_yticks(np.arange(0, 1.1, 0.1))
    ax[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    fig.savefig(join(plot_dir, 'tunning.png'))

def calc_preprocessing_info():
    # Get
    # (1) Straightness distribution of passes,
    # (2) Average spike counts per field per session
    # (3) Pass counts per passes

    simdata = load_pickle('results/network_proj/sim_result.pickle')
    Indata = simdata['Indata']
    SpikeData = simdata['SpikeData']
    NeuronPos = simdata['NeuronPos']

    all_spikecounts = []
    all_straightness = []
    radius = 10
    minpasstime = 0.6
    unique_neurons = SpikeData['neuronidx'].unique()
    num_neurons = unique_neurons.shape[0]
    for idx, nidx in enumerate(unique_neurons):

        print('%d/%d'%(idx, num_neurons))

        spks_count = SpikeData[SpikeData.neuronidx == nidx].shape[0]
        all_spikecounts.append(spks_count)

        # Get tok
        neuronx, neurony = NeuronPos.loc[nidx, ['neuronx', 'neurony']]
        dist = np.sqrt((neuronx - Indata.x) ** 2 + (neurony - Indata.y) ** 2)
        tok = dist < radius
        all_passidx = segment_passes(tok.to_numpy())

        for pid1, pid2 in all_passidx:
            pass_angles = Indata.loc[pid1:pid2, 'angle'].to_numpy()
            pass_t = Indata.loc[pid1:pid2, 't'].to_numpy()

            # Pass duration threshold
            if pass_t.shape[0] < 5:
                continue
            if (pass_t[-1] - pass_t[0]) < minpasstime:
                continue

            straightness = compute_straightness(pass_angles)
            all_straightness.append(straightness)

    print('Average spk count per field = %0.2f'% (np.mean(all_spikecounts)))
    print('0.1 quantile of straightness = %0.2f'% (np.quantile(all_straightness, 0.1)))





def sanity_check():

    simdata = load_pickle('results/network_proj/sim_result.pickle')
    indf = simdata['Indata']
    spdf = simdata['SpikeData']
    ntun = simdata['NeuronPos']
    np.random.seed(1)
    selected_nidxs = np.random.choice(spdf['neuronidx'].unique(), 4)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()

    fig_den, ax_den = plt.subplots(2, 2, figsize=(10, 10))
    ax_den = ax_den.ravel()

    for idx, nidx in enumerate(selected_nidxs):
        spdf_each = spdf[spdf.neuronidx == nidx]

        neu_x, neu_y = ntun.loc[nidx, ['neuronx', 'neurony']]
        tidxsp = spdf_each.tidxsp
        xsp = indf.x[tidxsp].to_numpy()
        ysp = indf.y[tidxsp].to_numpy()

        # Scatter
        ax[idx].plot(indf.x, indf.y)
        ax[idx].scatter(xsp, ysp, c='r', marker='.', alpha=0.5)
        ax[idx].scatter(neu_x, neu_y, c='k', marker='x')

        # Density
        xx, yy, zz = linear_gauss_density(xsp, ysp, 5, 5, 500, 500, (-40, 40), (-40, 40))
        ax_den[idx].pcolormesh(xx, yy, zz)

        ax_den[idx].scatter(neu_x, neu_y, c='k', marker='x')

    fig.savefig('result_plots/network_proj/sanity_check.png')
    fig_den.savefig('result_plots/network_proj/sanity_check_den.png')





def main():
    plot_dir = 'result_plots/network_proj'

    # simulate()
    # pilot(plot_dir)

if __name__ == '__main__':
    main()
