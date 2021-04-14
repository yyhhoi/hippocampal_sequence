import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from common.generation import freeforage, generate_spikes_bytuning
from common.comput_utils import midedges, normalize_distr, DirectionerMLM, DirectionerBining
boxl = 20
# tax, pos, hd = freeforage(30 * 60, 0.005, boxl, 2)
# pos = pos / boxl

center = np.array([0.9, 0.9])
rmax = 20  # Max. firing rate of spatial tunning
sig = 0.1  # sd of spatial tuning
hdc = 1.75 * np.pi  # preferred angle of directional tuning
a = 0.6  # Amplitude of directional tuning. Resulting tunning has range [1 - a +..., 1]
sighd = 0.1  # sd of directional tuning

# tsp, possp, hdsp, tuningfuncs = generate_spikes_bytuning(tax, pos, hd, center, rmax, sig, hdc, a, sighd)
# (stuning, dtuning) = tuningfuncs

sp_binwidth = 0.05  # assumming the space is normalized to 0 - 1
a_binwidth = 2 * np.pi / 36
dt = 1  # for occupancy
minerr = 0.01

mdata = loadmat('matlab/cacucci/data.mat')
pos = mdata['data'][0][0][0].squeeze()
tax = mdata['data'][0][0][1].squeeze()
hd = mdata['data'][0][0][2].squeeze()*2*np.pi
tsp = mdata['data'][0][0][3].squeeze()
possp = mdata['data'][0][0][4].squeeze().T
hdsp = mdata['data'][0][0][5].squeeze()*2*np.pi

mlmer = DirectionerMLM(pos, hd, dt, sp_binwidth, a_binwidth, minerr)


fieldangle, fieldR, directionality = mlmer.get_directionality(possp, hdsp)


abins = np.arange(-np.pi, np.pi + a_binwidth, step=a_binwidth)
passbins, _ = np.histogram(hd, bins=abins)
spikebins, _ = np.histogram(hdsp, bins=abins)
norm_prob = normalize_distr(spikebins, passbins)


biner = DirectionerBining(abins, hd)
fieldangle, fieldR, (spike_bins, occ_bins, norm_prob_nonan) = biner.get_directionality(hdsp)


# Plot spatial and directional tunning
fig = plt.figure(figsize=(10, 15))
ax1 = fig.add_subplot(321)
ax1.plot(pos[:, 0], pos[:, 1], c='gray', alpha=0.3)
ax1.scatter(possp[:, 0], possp[:, 1], c='r', alpha=0.3)
# dax = np.linspace(-np.pi, np.pi, 100)
# ax2 = fig.add_subplot(322, projection='polar')
# ax2.plot(dax, dtuning(dax)/np.sum(dtuning(dax)))

# ax3 = fig.add_subplot(323)
# ax3.pcolormesh(midedges(xbins), midedges(ybins), positionality)

ax4 = fig.add_subplot(324, projection='polar')
ax4.plot(midedges(abins), directionality/np.sum(directionality))

ax5 = fig.add_subplot(325, projection='polar')
ax5.plot(midedges(abins), norm_prob)

ax5 = fig.add_subplot(326, projection='polar')
ax5.plot(midedges(abins), norm_prob_nonan)


plt.show()



