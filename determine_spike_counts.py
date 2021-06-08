# This script determines the spike counts per field in the experimental data, such that a proper number of spikes can
# be subsampled from simulation.
import numpy as np
from scipy.interpolate import interp1d
from common.utils import load_pickle
from common.comput_utils import IndataProcessor

rawdf_pth = 'data/emankindata_processed_withwave.pickle'
df = load_pickle(rawdf_pth)
numspikes_dict = dict(CA1=[], CA2=[], CA3=[])
num_trials = df.shape[0]
for ntrial in range(num_trials):
    for ca in ['CA%d' % (i + 1) for i in range(3)]:
        field_df = df.loc[ntrial, ca + 'fields']
        indata = df.loc[ntrial, ca + 'indata']
        num_fields = field_df.shape[0]
        if indata.shape[0] < 1:
            continue
        tunner = IndataProcessor(indata, vthresh=0, sthresh=-5, minpasstime=0)
        interpolater_angle = interp1d(tunner.t, tunner.angle)
        interpolater_x = interp1d(tunner.t, tunner.x)
        interpolater_y = interp1d(tunner.t, tunner.y)
        for nf in range(num_fields):
            print('\r%d/%d trial, %s, %d/%d field' % (ntrial, num_trials, ca, nf, num_fields), flush=True, end='')
            mask, pf, xyval = field_df.loc[nf, ['mask', 'pf', 'xyval']]
            tsp = field_df.loc[nf, 'xytsp']['tsp']
            xaxis, yaxis = pf['X'][:, 0], pf['Y'][0, :]
            # Construct passes (segment & chunk)
            tok, idin = tunner.get_idin(mask, xaxis, yaxis)
            passdf = tunner.construct_singlefield_passdf(tok, tsp, interpolater_x, interpolater_y, interpolater_angle)
            tspall = np.concatenate(passdf['tsp'].to_list())
            numspikes_dict[ca].append(tspall.shape[0])
print()
for ca in ['CA1', 'CA2', 'CA3']:
    numfields = len(numspikes_dict[ca])
    meannsp = np.median(numspikes_dict[ca])
    print('%s(n=%d), median num spikes = %0.2f' % (ca, numfields, meannsp))
