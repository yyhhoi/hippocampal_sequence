import numpy as np
import pandas as pd
from .comput_utils import heading


def freeforage(T, dt, boxl, speed):
    """
    Generate random artificial trajectory of an animal in a free foraging task.

    Parameters
    ----------
    T : scalar
        Total duration of the random trajectory.
    dt : scalar
        Time step or increment of simulation.
    boxl : scalar
        Maximum length of the square enclosure. Unit = 0.1m
    speed : scalar
        Speed of the animal. Unit = 10cm/s
    Returns
    -------
    tax : ndarray
        1d numpy array containing times. Shape = (time, )
    pos : ndarray
        2d numpy array containing x- and y- coordinates, in range (0, boxl). Shape = (time, 2)
    hd : ndarray
        1d numpy array containing headings in range (-pi, pi). Shape = (time, )
    """


    num_steps = int(T / dt)
    rEdge = np.ones(2) * boxl
    phi = 0
    PhiAdvance = np.pi / 36
    r = np.around(rEdge / 2)

    pos = np.zeros((num_steps, 2))
    tax = np.zeros(num_steps)

    pos[0, :] = r

    t_i = 0
    t = 0
    while t < (T - dt):
        t_i = t_i + 1
        t = dt * t_i

        phi = phi + PhiAdvance * np.random.randn()
        r1 = r + speed * dt * np.array([np.cos(phi), np.sin(phi)])
        while np.any(r1 > (rEdge - PhiAdvance * 2)) or np.any(r1 < (np.zeros(2) + PhiAdvance * 2)):
            rc = (r - boxl / 2)
            rc = rc / np.linalg.norm(rc)
            phi = np.angle(np.exp(1j * phi) - (rc[0] + 1j * rc[1]) / 5)
            r1 = r + speed * dt * np.array([np.cos(phi), np.sin(phi)])

        r = r1
        pos[t_i, :] = r
        tax[t_i] = t
    hd = heading(pos)
    return tax, pos, hd


def generate_spikes_bytuning(tax, pos, hd, center, rmax, sig, hdc, a, sighd):
    """

    Parameters
    ----------
    tax : ndarray
        Shape (time, ).
    pos : ndarray
        Shape (time, 2).
    hd : ndarray
        Heading direction, in range [-pi, pi]. Shape (time, ).
    center : ndarray
        xy coordinates of place field center. Shape (2, ).
    rmax : scalar
        Max firing rate of place field (spatial tuning)
    sig : scalar
        Standard deviation of place field strength.
    hdc : scalar
        Preferred direction of directional tuning.
    a : scalar
        Amplitude of directional tuning. Resulting in a multiplying factor in range [1 - a +..., 1].
    sighd : scalar
        Standard deviation of directional tuning.

    Returns
    -------
    tsp : ndarray
        Spike times. Shape (time, ).
    possp : ndarray
        Positions at spike times. Shape (time, 2)
    hdsp : ndarray
        Heading at spike times, in range (-pi, pi). Shape (time, 2)
    (spatial_func, direction_func) : tuple(func, func)
        Functions for sptial (arg: position as ndarray with shape (time, 2)) and directional (arg: heading as ndarray with shape (time, )) tuning.

    """

    # spatial and directional tuning
    spatial_func = lambda x: rmax * np.exp(-np.sum(np.square(x - center.reshape(1, 2)), axis=1) / (2 * (sig ** 2)))
    direction_func = lambda x: 1 - a + a * np.exp((np.cos((x - hdc)) - 1) / np.square(sighd))
    tuning = spatial_func(pos) * direction_func(hd)

    # Simulate spike events
    dt = tax[1] - tax[0]
    ilam = -np.log(np.random.rand())
    tsp = []
    possp = []
    hdsp = []

    for nt in range(tax.shape[0]):
        ilam = ilam - dt * tuning[nt]
        if ilam < 0:
            ilam = -np.log(np.random.rand())
            tsp.append(tax[nt])
            possp.append(pos[nt, :])
            hdsp.append(hd[nt])
    return np.array(tsp), np.array(possp), np.array(hdsp), (spatial_func, direction_func)

def generate_spikes_from_poisson(Indata, Activity, NeuronPos, num=100, dt=1e-3, max_tid=None):
    """
    Iterate the firing rate through the experiment and generate spikes.

    Parameters
    ----------
    Indata : Dataframe
        With columns : m, t, phase, x, y
    NeuronPos : ndarray
        xy coordinates of each neuron. Array with shape (num_neurons, 2)
    num : int
        Number of iterations of poisson spike generation
    dt : float
    max_tid : int
        Maximum time course of the experiment. Default = None (maximum of the data)

    Returns
    -------
    SpikeData_inflated : Dataframe
        With columns : neuronidx, neuronx, neurony, tidxsp, tsp, xsp, ysp, phasesp
    """

    num_neurons = NeuronPos.shape[0]
    if max_tid is None:
        max_tid = Indata.shape[0]
    all_firing_idx = []
    all_tid = []
    for tid in range(max_tid):
        m = Activity.loc[tid, 'm']
        for _ in range(num):
            firing_idx = np.where(np.random.rand(num_neurons) < (m * dt))[0]
            tid_np = np.ones(firing_idx.shape[0]) * tid
            all_firing_idx.append(firing_idx)
            all_tid.append(tid_np.astype(int))
    neuronidx = np.concatenate(all_firing_idx)
    tidxsp = np.concatenate(all_tid)
    t, phase, x, y = Indata['t'].to_numpy(), Indata['phase'].to_numpy(), Indata['x'].to_numpy(), Indata['y'].to_numpy()
    SpikeData_inflated = pd.DataFrame(dict(
        neuronidx=neuronidx,
        xsp=x[tidxsp],
        ysp=y[tidxsp],
        tidxsp=tidxsp,
        tsp=t[tidxsp],
        neuronx=NeuronPos[neuronidx, 0],
        neurony=NeuronPos[neuronidx, 1],
        phasesp=phase[tidxsp]
    ))
    return SpikeData_inflated
