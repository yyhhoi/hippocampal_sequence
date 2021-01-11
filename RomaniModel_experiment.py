#%% Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.interpolate import interp1d
from common.utils import load_pickle
from common.comput_utils import heading, TunningAnalyzer, check_border, segment_passes, aperiodcos
from common.Ensembles import RomaniEnsemble, InputEvaluatorToroidal, InputEvaluatorSquare, InputEvaluatorSquareMatch
from common.Environments import Simulation
from common.comput_utils import pair_dist_self, pair_diff, normalize_distr, linear_circular_gauss_density
from common.sql import SQLRecorder
from common.correlogram import ThetaEstimator

def linear_transform(x, trans, scaling_const):
    return (x - trans) * scaling_const


def get_transform_params(x, y):
    trans_x = np.min(x)
    trans_y = np.min(y)
    scaling_const = 2*np.pi / np.max([x-trans_x, y-trans_y])
    return trans_x, trans_y, scaling_const


#%% Load Experiment dataframe
df = load_pickle('data/emankindata_processed.pickle')

#%% Extract Experiment trajectories
dt = 1e-3
target_speed = 2*np.pi/5  # rad/second
num_trials = df.shape[0]

all_t_list, all_pos_list = [], []

for trial_i in range(num_trials):
    for ca_i in range(3):
        
        # Retrieve indata
        ca = 'CA%d' % (ca_i + 1)
        indata = df.loc[trial_i, ca+'indata']
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
        current_speed = np.mean(np.sqrt(np.diff(rat_x)**2 + np.diff(rat_y)**2)/np.diff(rat_t))
        rat_t = rat_t * current_speed / target_speed
        new_speed = np.mean(np.sqrt(np.diff(rat_x)**2 + np.diff(rat_y)**2)/np.diff(rat_t))        
        
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
        
for i in range(len(all_t_list)-1):
    all_t_list[i+1] += np.max(all_t_list[i])+dt
    
t = np.concatenate(all_t_list)  # Used name
pos = np.concatenate(all_pos_list, axis=0)  # Used name
current_speed = np.mean(np.sqrt(np.sum(np.square(np.diff(pos, axis=0)), axis=1))/np.diff(t))

print('Speed = ', current_speed)
print('Time from %0.4f to %0.4f'%(t.min(), t.max()))
print('time shape = ', t.shape[0])


#%% Setting - Toroidal

## Parameters
neuron_theta1 = np.linspace(0, 2*np.pi, num=32)
neuron_theta2 = np.linspace(0, 2*np.pi, num=32)
thethe1, thethe2 = np.meshgrid(neuron_theta1, neuron_theta2)
neuron_theta = np.stack([thethe1.reshape(-1), thethe2.reshape(-1)]).T  # (num_neurons, 2)


J1 = 35
J0 = 25
num_neurons = neuron_theta.shape[0]  
I = -7
I_L = 15
I_theta = 5
f_theta = 10

# Neurons' positions & Weights
pair_dist1 = pair_dist_self(neuron_theta[:, 0])
pair_dist2 = pair_dist_self(neuron_theta[:, 1])
w = J1 * ( np.cos(pair_dist1) + np.cos(pair_dist2) ) - J0
 
I_E = InputEvaluatorToroidal(pos, neuron_theta, t, f_theta, I_theta, I, I_L)
I_period = 1/f_theta
I_phase = (np.mod(t - (I_period/2), I_period)/I_period) * 2 * np.pi - np.pi


# Ensemble
ensemble_dict = {
    'm_rest':0,
    'x_rest':1,
    'tau':10e-3,
    'tau_R':0.8,
    'U':0.8,
    'alpha':1,
    'w':w
}

#%% Setting - SquareMatch

## Parameters
neuron_theta1 = np.linspace(0, 2*np.pi, num=32)
neuron_theta2 = np.linspace(0, 2*np.pi, num=32)
thethe1, thethe2 = np.meshgrid(neuron_theta1, neuron_theta2)
neuron_theta = np.stack([thethe1.reshape(-1), thethe2.reshape(-1)]).T  # (num_neurons, 2)

J1 = 35
J0 = 25
num_neurons = neuron_theta.shape[0]  
I = -7
I_L = 15
I_theta = 5
f_theta = 10

# Neurons' positions & Weights
pair_dist1 = pair_dist_self(neuron_theta[:, 0])
pair_dist2 = pair_dist_self(neuron_theta[:, 1])
w = J1 * ( aperiodcos(pair_dist1) + aperiodcos(pair_dist2) ) - J0
 
I_E = InputEvaluatorSquareMatch(pos, neuron_theta, t, f_theta, I_theta, I, I_L, lambda_E=0)
I_period = 1/f_theta
I_phase = (np.mod(t - (I_period/2), I_period)/I_period) * 2 * np.pi - np.pi


# Ensemble
ensemble_dict = {
    'm_rest':0,
    'x_rest':1,
    'tau':10e-3,
    'tau_R':0.8,
    'U':0.8,
    'alpha':1,
    'w':w
}


#%% Setting - SquareMatch Extended

# Determining the number of neurons and margin
for n in np.arange(32, 50):
    xinpi = n/16
    target = (2*np.pi+2)/np.pi
    diff = np.around(xinpi-target, 4)
    print(n, xinpi, diff)


## Parameters
# margin = 2.6875/2
# num1d = 43

margin = np.pi
num1d = 64
neuron_theta1 = np.linspace(0-margin, 2*np.pi+margin, num=num1d)
neuron_theta2 = np.linspace(0-margin, 2*np.pi+margin, num=num1d)
thethe1, thethe2 = np.meshgrid(neuron_theta1, neuron_theta2)
neuron_theta = np.stack([thethe1.reshape(-1), thethe2.reshape(-1)]).T  # (num_neurons, 2)

J1 = 35
J0 = 25
num_neurons = neuron_theta.shape[0]  
I = -7
I_L = 15
I_theta = 5
f_theta = 10

# Neurons' positions & Weights
pair_dist1 = pair_dist_self(neuron_theta[:, 0])
pair_dist2 = pair_dist_self(neuron_theta[:, 1])
w = J1 * ( aperiodcos(pair_dist1) + aperiodcos(pair_dist2) ) - J0
 
I_E = InputEvaluatorSquareMatch(pos, neuron_theta, t, f_theta, I_theta, I, I_L, lambda_E=0)
I_period = 1/f_theta
I_phase = (np.mod(t - (I_period/2), I_period)/I_period) * 2 * np.pi - np.pi


# Ensemble
ensemble_dict = {
    'm_rest':0,
    'x_rest':1,
    'tau':10e-3,
    'tau_R':0.8,
    'U':0.8,
    'alpha':1,
    'w':w
}

#%% Simulate

save_results_pth = 'results/sim/raw/squarematch_extended_pi.sql'
print('Result will be saved at %s'%(save_results_pth))

# Recording

sqler = SQLRecorder(num_neurons, save_results_pth, overwrite=True)
sqler.create_table()
## Simulation
simenv = Simulation(t, **dict(I_E=I_E))
ensem = RomaniEnsemble(simenv, num_neurons, **ensemble_dict)
all_m_list = list()
all_firing_list = list()
all_timeidx_list = list()
max_t = t.max()
m_neuronidx = np.arange(num_neurons).reshape(-1, 1)

# Primary keys
mid = np.arange(num_neurons).reshape(-1, 1)
indataid = 0
spdataid = 0
while ensem.simenv.check_end():
    _, m_each, firing_idx = ensem.state_update()
    
    time_each = ensem.simenv.current_time
    time_idx = ensem.simenv.current_idx
    tarr = np.array([time_idx, time_each])
    
    # Insert activity
    # data = (id, tid, t, neuronidx, m)
    # shape = (num_neurons, 4)
    m_tarr = tarr.reshape(1, 2) * np.ones((num_neurons, 1))
    marr = np.hstack([mid, m_tarr, m_neuronidx, m_each.reshape(-1, 1)])
    sqler.insert_activity(marr)
    
    # Insert Indata
    inarr = np.concatenate([np.array([indataid]), tarr, pos[time_idx, :], I_phase[[time_idx]]])
    sqler.insert_indata(inarr)
    
    # Insert SpikeData: id, neuronidx, neuronx, neurony, tidxsp, tsp, xsp, ysp, phasesp
    num_spikes = firing_idx.shape[0]
    if firing_idx.shape[0]>0:
        spdataid_row = np.arange(spdataid, spdataid+num_spikes)
        neuronxy = neuron_theta[firing_idx, :]
        tidxsp = np.ones(firing_idx.shape[0]).astype(int) * time_idx
        tsp = t[tidxsp]
        xysp = pos[tidxsp, :]
        phasesp = I_phase[tidxsp]
        sparr = np.hstack([spdataid_row.reshape(-1, 1), firing_idx.reshape(-1, 1), neuronxy, tidxsp.reshape(-1, 1), tsp.reshape(-1, 1), xysp, phasesp.reshape(-1, 1)])
        sqler.insert_spikedata(sparr)
        spdataid += num_spikes
    if time_idx % 100 == 0:
        print('\r%0.3f/%0.3f'%(time_each, max_t), end='', flush=True)
    if time_idx % 1e4 == 0:
        sqler.sqlcon.commit()
        
    mid = mid+num_neurons
    indataid += 1
    

sqler.sqlcon.commit()
sqler.close()

print('\n Simulation finished')

#%% Phase precession in 2D - Load data
# expdata = load_pickle('results/sim/raw/squarematch.pickle')
expdata = load_pickle('results/sim/raw/toroidal.pickle')
SpikeData = expdata['SpikeData']
Indata = expdata['Indata']
NeuronPos = expdata['NeuronPos']

#%% Phase precession in 2D - pair correlation
m = np.array(list(Indata['m']))
trajx, trajy = Indata['x'].to_numpy(), Indata['y'].to_numpy()
trajxy = np.stack([trajx, trajy]).T
trajt = Indata['t'].to_numpy()
trajhd = heading(np.stack([trajx, trajy]).T)
Iphase = Indata['phase'].to_numpy()

tidx1, tidx2 = 1200, 3000
tidx_arr = np.arange(tidx1, tidx2)

# Get and plot neighbors
neighbors = np.zeros(tidx_arr.shape)
for idx in range(tidx_arr.shape[0]):    
    minidx = np.argmin(np.linalg.norm(NeuronPos - trajxy[tidx_arr[idx], ].reshape(1, 2), axis=1))
    if hasattr(minidx, '__iter__'):
        minidx = minidx[0]
    neighbors[idx] = minidx
uniidx = np.unique(neighbors, return_index=True)[1]
neighbors = np.array([neighbors[i] for i in sorted(uniidx)]).astype(int)


# Generate more spikes
dt = 1e-3

SpikeData_appenddict = {col:[] for col in SpikeData.columns}
for i in range(100):
    print('\r Generating spikes %d'%i, end='', flush=True)
    for j in range(tidx1, tidx2):
        instantr = m[j, ] * dt
        rand_p = np.random.uniform(0, 1, size=instantr.shape[0])
        
        neuronidx = np.where(rand_p < instantr)[0]
        tsp = np.ones(neuronidx.shape[0]) * trajt[j]
        neuronx = NeuronPos[neuronidx, 0]
        neurony = NeuronPos[neuronidx, 1]
        tidxsp = np.ones(neuronidx.shape[0]).astype(int) * j
        xsp = np.ones(neuronidx.shape[0]) * trajx[j]
        ysp = np.ones(neuronidx.shape[0]) *trajy[j]
        phasesp = np.ones(neuronidx.shape[0]) *Iphase[j]
        
        SpikeData_appenddict['neuronidx'].append(neuronidx)
        SpikeData_appenddict['tsp'].append(tsp)
        SpikeData_appenddict['neuronx'].append(neuronx)
        SpikeData_appenddict['neurony'].append(neurony)
        SpikeData_appenddict['tidxsp'].append(tidxsp)
        SpikeData_appenddict['xsp'].append(xsp)
        SpikeData_appenddict['ysp'].append(ysp)
        SpikeData_appenddict['phasesp'].append(phasesp)
    
SpikeData_appenddict2 = dict()
for key, val in SpikeData_appenddict.items():
    SpikeData_appenddict2[key]= np.concatenate(val)
    
    

SpikeData_append = pd.DataFrame(SpikeData_appenddict2)


# Filter spikes

SpikeData_combined = pd.concat([SpikeData, SpikeData_append], axis=0, ignore_index=True)

mask = (SpikeData_combined['tidxsp'] < tidx2) & (SpikeData_combined['tidxsp'] >= tidx1) & SpikeData_combined['neuronidx'].isin(neighbors)
spdf = SpikeData_combined[mask].reset_index(drop=True)
phase_finder = ThetaEstimator(0.005, 0.3, [5, 12]) 
progress = 0

alllags = []
allxdiff = []
for neui in neighbors:
    for neuj in neighbors:
        if neui == neuj:
            continue
        
        print('\r%d/%d'%(progress, neighbors.shape[0]**2), end='', flush=True)
        
        tsp1 = spdf[spdf['neuronidx'] == neui]['tsp'].to_numpy()
        tsp2 = spdf[spdf['neuronidx'] == neuj]['tsp'].to_numpy()
        
        _, phaselag, _ = phase_finder.find_theta_isi_hilbert([tsp1], [tsp2])
        
        x1 = NeuronPos[neui, 0]
        x2 = NeuronPos[neuj, 0]
        xdiff = x1-x2
        
        alllags.append(phaselag)
        allxdiff.append(xdiff)
        
        progress += 1
        
lags = np.array(alllags)
xdiffs = np.array(allxdiff)

nonanmask = ~np.isnan(lags)
lags = lags[nonanmask]
xdiffs = xdiffs[nonanmask]

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

xx, yy, zz = linear_circular_gauss_density(xdiffs, lags, cir_kappa=4*np.pi, lin_std=0.25, 
                                           xbins=100, ybins=100, 
                                           xbound=(-4, 4), ybound=(-np.pi, np.pi))

ax[0].scatter(xdiffs, lags, alpha=0.5)
ax[1].pcolormesh(xx, yy, zz)


#%% Phase precession in 2D - plot precession
m = np.array(list(Indata['m']))
trajx, trajy = Indata['x'].to_numpy(), Indata['y'].to_numpy()
trajxy = np.stack([trajx, trajy]).T
trajt = Indata['t'].to_numpy()
trajhd = heading(np.stack([trajx, trajy]).T)
Iphase = Indata['phase'].to_numpy()

tidx1, tidx2 = 1200, 3000
tidx_arr = np.arange(tidx1, tidx2)

# Get and plot neighbors
neighbors = np.zeros(tidx_arr.shape)
for idx in range(tidx_arr.shape[0]):    
    minidx = np.argmin(np.linalg.norm(NeuronPos - trajxy[tidx_arr[idx], ].reshape(1, 2), axis=1))
    if hasattr(minidx, '__iter__'):
        minidx = minidx[0]
    neighbors[idx] = minidx
uniidx = np.unique(neighbors, return_index=True)[1]
neighbors = np.array([neighbors[i] for i in sorted(uniidx)]).astype(int)


m_slice = m[tidx1:tidx2, neighbors]
t_slice = trajt[tidx1:tidx2]
phase_slice = Iphase[tidx1:tidx2]
t1, t2 = trajt[tidx1], trajt[tidx2]
minact_idx = np.where( (phase_slice[1:] - phase_slice[:-1])< -np.pi)[0]

xPos_diff = pair_diff(trajx[tidx1:tidx2], NeuronPos[neighbors, 0 ])
yPos_diff = pair_diff(trajy[tidx1:tidx2], NeuronPos[neighbors, 1 ])

xyPos_diff = np.sqrt(xPos_diff**2 + yPos_diff**2)




# Plot trajectory
fig, ax = plt.subplots()
ax.plot(trajx[tidx1:tidx2], trajy[tidx1:tidx2])
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(0, 2*np.pi)
ax.scatter(NeuronPos[neighbors][:, 0], NeuronPos[neighbors][:, 1])



# Plot neighbors' spikes
fig, ax = plt.subplots(figsize=(16, 8))
splist = []
phasesplist = []
reldistlist = []
for i in range(neighbors.shape[0]):
    spdf_inside = SpikeData[(SpikeData['neuronidx']==neighbors[i]) & (SpikeData['tidxsp'] < tidx2) & (SpikeData['tidxsp'] > tidx1)]
    tsp = spdf_inside['tsp'].to_numpy()
    reldist = spdf_inside['xsp'].to_numpy() - NeuronPos[neighbors[i], 0]
    dist = np.linalg.norm(spdf_inside[['xsp', 'ysp']].to_numpy() - NeuronPos[neighbors[i], :].reshape(1, 2), axis=1)
    
    tidxsp = spdf_inside['tidxsp'].to_numpy()
    hdsp_complex = np.exp(1j *trajhd[tidxsp])
    hdsp = np.stack([np.real(hdsp_complex), np.imag(hdsp_complex)]).T
    center = NeuronPos[neighbors[i], :].reshape(1, 2)
    pos = np.stack([trajx[tidxsp], trajy[tidxsp]]).T
    
    drz_sign = np.sign(np.sum(hdsp*(pos-center), axis=1))
    drz_amount = np.linalg.norm(pos-center, axis=1)
    
    
    phasesp = spdf_inside['phasesp'].to_numpy()
    splist.append(tsp)
    phasesplist.append(phasesp)
    reldistlist.append(np.sign(reldist) * dist)
#     reldistlist.append(drz_sign * drz_amount)

_ = ax.eventplot(splist)
# ax[0].set_xlim(1.2, 3.0)
ax.set_ylim(0, 35)
ax.set_xlabel('time', fontsize=16)
ax.set_ylabel('Neuron Index', fontsize=16)
for i in minact_idx:
    ax.vlines(t_slice[i], ymin=0, ymax=neighbors.shape[0],color='k')


# Phase v.s. Relative distance
fig, ax = plt.subplots(figsize=(8, 8))
all_tsp = np.concatenate(splist)
all_reldist = np.concatenate(reldistlist)
all_phasesp = np.concatenate(phasesplist)
ax.scatter(all_reldist, all_phasesp, alpha=0.5)
ax.set_xlim(-4, 4)
ax.set_xlabel('Relative Distance to neuron', fontsize=16)
ax.set_ylabel('Phase', fontsize=16)



# generating more spikes

dt = 1e-3

ftidx = []
fnidx = []
phase_all = []
reldist_all = []
for i in range(100):
    for j in range(m_slice.shape[0]):
        instantr = m_slice[j, ] * dt
        rand_p = np.random.uniform(0, 1, size=instantr.shape[0])
        
        firing_idx = np.where(rand_p < instantr)[0]
        time_idx = np.ones(firing_idx.shape[0]) * t_slice[j]
        reldist = xPos_diff[j, firing_idx]
        phase_j = np.ones(firing_idx.shape[0]) * phase_slice[j]
        
        fnidx.append(firing_idx)
        phase_all.append(phase_j)
        reldist_all.append(reldist)
        ftidx.append(time_idx)
ftidx = np.concatenate(ftidx)
fnidx = np.concatenate(fnidx)
reldist_all = np.concatenate(reldist_all)
phase_all = np.concatenate(phase_all)

uninidx = np.sort(np.unique(fnidx))
sp_list = [[] for _ in range(uninidx.shape[0])]
for nidx in uninidx:
    tmpidx = np.where(fnidx == nidx)[0]
    sptimes = ftidx[tmpidx]
    sp_list[nidx] = sptimes


fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(m_slice.T, aspect='auto', cmap='jet', extent=[t1, t2, 0, neighbors.shape[0]], origin='lower')
fig.colorbar(im, ax=ax)
# ax.set_xlim(1.5, 2.5)
# ax.set_ylim(0, 35)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Neuron's Index")
_ = ax.eventplot(splist, color='k')

for i in minact_idx:
    ax.vlines(t_slice[i], ymin=0, ymax=neighbors.shape[0],color='white')

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(reldist_all, phase_all, marker='.', alpha=0.1)
ax.set_xlim(-4, 4)







#%% Test input


def modulate(refpos, pos, func, J1, J0):
    W = J1 * (func(refpos[0]-pos[:,0]) + func(refpos[1]-pos[:,1]) ) - J0
    gt = func(refpos[0]-pos[:,0]) + func(refpos[1]-pos[:,1]) 
    return W, gt

def dog(refpos, pos, J1, J0, K1, K0, sigma1, sigma2):
    dist = np.square(refpos[0]-pos[:,0]) + np.square(refpos[1]-pos[:,1])
    gau1 = np.exp(-dist/(2*np.square(sigma1)))/(2 * np.pi * sigma1 **2)
    gau2 = np.exp(-dist/(2*np.square(sigma2)))/(2 * np.pi * sigma2 **2)
    dog = gau1 - gau2
    W = J1 * dog - J0
    gt = K1 * dog - K0
    return W, gt


x = np.linspace(0, 2*np.pi, 100)
y = np.linspace(0, 2*np.pi, 100)
pos = np.stack([x, y]).T
# refpos = np.array([np.pi, np.pi])
refpos = np.array([0, 0])

# Toroidal
J1, J0 = 35, 25
diff = refpos.reshape(1, 2) - pos
dist = np.linalg.norm(diff, axis=1)
# dist[diff[:, 0] <0] = -dist[diff[:, 0] <0]

# Periodic
W, gt = modulate(refpos, pos, np.cos, J1, J0)
# Aperiodic
aW, agt = modulate(refpos, pos, aperiodcos, J1, J0)

# DoG
J1 = 5000
J0 = 100
K1 = 1
K0 = 0
sigma1, sigma2 = 2, 4
dogW, doggt = dog(refpos, pos, J1, J0, K1, K0, sigma1, sigma2)


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(dist, W, label='cos=%0.4f'%(np.sum(W)))
ax[0].plot(dist, aW, label='acos=%0.4f'%(np.sum(aW)))
ax[0].plot(dist, dogW, label='dog=%0.4f'%(np.sum(dogW)))

ax[0].set_title('W')
ax[0].legend()
ax[1].plot(dist, gt, label='cos=%0.4f'%(np.sum(gt)))
ax[1].plot(dist, agt, label='acos=%0.4f'%(np.sum(agt)))
ax[1].plot(dist, doggt, label='dog=%0.4f'%(np.sum(doggt)))
ax[1].set_title('GT')
ax[1].legend()
plt.show()
