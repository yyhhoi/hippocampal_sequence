#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Wrappers enclosing scripts which are invoked more than once

"""

import numpy as np
import pandas as pd
from pycircstat.tests import rayleigh
from scipy.interpolate import interp1d
from pycircstat.descriptive import mean as circmean, cdiff, resultant_vector_length
from common.linear_circular_r import rcc
from common.comput_utils import DirectionerBining, DirectionerMLM, compute_straightness, circular_density_1d, \
    repeat_arr, shiftcyc_full2half, midedges, normalize_distr


class DirectionalityStatsByThresh:
    def __init__(self, nspikes_key, winp_key, shiftp_key, fieldR_key):
        self.nspikes_key = nspikes_key
        self.winp_key = winp_key
        self.shiftp_key = shiftp_key
        self.fieldR_key = fieldR_key

    def gen_directionality_stats_by_thresh(self, df, spike_threshs=None):
        if spike_threshs is None:
            max_nspikes = df[self.nspikes_key].max()
            spike_threshs = np.linspace(0, max_nspikes, 20)
            spike_threshs = spike_threshs[:-1]

        all_n = df.shape[0]
        stats_dict = dict(
            spike_threshs=spike_threshs,
            sigfrac_win=np.zeros(spike_threshs.shape[0]),
            sigfrac_shift=np.zeros(spike_threshs.shape[0]),
            medianR=np.zeros(spike_threshs.shape[0]),
            datafrac=np.zeros(spike_threshs.shape[0]),
            allR=[],
            win_signum=np.zeros(spike_threshs.shape[0]),
            win_nonsignum=np.zeros(spike_threshs.shape[0]),
            shift_signum=np.zeros(spike_threshs.shape[0]),
            shift_nonsignum=np.zeros(spike_threshs.shape[0]),
            n=np.zeros(spike_threshs.shape[0])

        )
        for idx, thresh in enumerate(spike_threshs):
            thresh_df = df[df[self.nspikes_key] > thresh]
            stats_dict['sigfrac_win'][idx] = np.mean(thresh_df[self.winp_key] < 0.05)
            stats_dict['sigfrac_shift'][idx] = np.mean(thresh_df[self.shiftp_key] < 0.05)
            stats_dict['medianR'][idx] = np.median(thresh_df[self.fieldR_key])
            stats_dict['datafrac'][idx] = thresh_df.shape[0] / all_n
            stats_dict['allR'].append(thresh_df[self.fieldR_key].to_numpy())
            stats_dict['win_signum'][idx] = np.sum(thresh_df[self.winp_key] < 0.05)
            stats_dict['win_nonsignum'][idx] = np.sum(thresh_df[self.winp_key] >= 0.05)
            stats_dict['shift_signum'][idx] = np.sum(thresh_df[self.shiftp_key] < 0.05)
            stats_dict['shift_nonsignum'][idx] = np.sum(thresh_df[self.shiftp_key] >= 0.05)
            stats_dict['n'][idx] = thresh_df.shape[0]

        return stats_dict


class PrecessionProcesser:
    def __init__(self, wave, vthresh, sthresh):
        self.wave = wave
        self.vthresh = vthresh
        self.sthresh = sthresh
        self.trange = None
        self.phase_inter = interp1d(wave['tax'], wave['phase'])
        self.theta_inter = interp1d(wave['tax'], wave['theta'])
        self.wave_maxt, self.wave_mint = wave['tax'].max(), wave['tax'].min()
        self.precess_keys = self._gen_precess_infokeys()

    def get_single_precession(self, passes_df, neuro_keys_dict):

        for npass in range(passes_df.shape[0]):
            result = self._get_precession(passes_df, npass, neuro_keys_dict)
            if result is None:
                continue
            else:
                yield result

    def get_pair_precession(self, passes_df, neuro_keys_dict1, neuro_keys_dict2):
        for npass in range(passes_df.shape[0]):
            result1 = self._get_precession(passes_df, npass, neuro_keys_dict1)
            result2 = self._get_precession(passes_df, npass, neuro_keys_dict2)
            if (result1 is None) or (result2 is None):
                continue
            else:
                yield (result1, result2)

    def set_trange(self, trange):
        """Set the min and max of time

        Parameters
        ----------
        trange : tuple
            Tuple which is (max_t, min_t).
        """
        self.trange = trange

    def _get_precession(self, passes_df, npass, neuro_keys_dict):
        """Receive passdf and return a dictionary containing precession information

        Parameters
        ----------
        passes_df : dataframe
            Must contain behavioural columns - 'x', 'y', 't', 'angle', and spike columns 'tsp', 'spikex', 'spikey', 'spikev', 'spikeangle'
        npass : int
            The n-th pass of the passdf
        neuro_keys_dict : dict
            Dictionary defining the keynames of the spike columns.

        Returns
        -------
        dict
            Dictionay containing all the precession information.
        """
        assert self.trange is not None
        all_maxt, all_mint = self.trange
        tsp_k = neuro_keys_dict['tsp']
        spikev_k = neuro_keys_dict['spikev']
        spikex_k = neuro_keys_dict['spikex']
        spikey_k = neuro_keys_dict['spikey']
        spikeangle_k = neuro_keys_dict['spikeangle']
        data_dict = dict()

        # Behavioural
        x, y, t, angle = passes_df.loc[npass, ['x', 'y', 't', 'angle']]
        # Neural
        tsp, vsp, xsp, ysp, anglesp = passes_df.loc[npass, [tsp_k, spikev_k, spikex_k, spikey_k, spikeangle_k]]

        # Filtering
        if (tsp.shape[0] < 1) or (tsp.shape[0] != vsp.shape[0]):
            return None

        vidx = np.where((vsp >= self.vthresh) &
                        (tsp > self.wave_mint) & (tsp <= self.wave_maxt) &
                        (tsp > all_mint) & (tsp <= all_maxt)
                        )[0]
        tsp = tsp[vidx]
        anglesp = anglesp[vidx]
        if vidx.shape[0] < 3:
            return None
        idmin = np.where(t <= tsp.min())[0][-1]
        idmax = np.where(t >= tsp.max())[0][0] + 1
        if (idmax - idmin) < 3:
            return None
        straightrank = compute_straightness(angle[idmin:idmax])

        # Compute phase and normalized passlength
        xin, yin, tin = x[idmin:idmax], y[idmin:idmax], t[idmin:idmax]
        anglein = angle[idmin:idmax]
        pass_angle = shiftcyc_full2half(circmean(anglein))
        spike_angle = shiftcyc_full2half(circmean(anglesp))

        # Pass length
        d = np.sqrt(np.square(np.diff(xin)) + np.square(np.diff(yin)))
        cumd = np.append(0, np.cumsum(d))
        cumd_norm = cumd / cumd.max()
        dinter = interp1d(tin, cumd_norm)

        # Filter tsp with no theta power
        thetasp = self.theta_inter(tsp)
        tsp_withtheta = tsp[np.abs(thetasp) > 1e-5]
        if tsp_withtheta.shape[0] < 1:
            return None

        # RCC
        phasesp = self.phase_inter(tsp_withtheta)
        dsp = dinter(tsp_withtheta)
        regress = rcc(dsp, phasesp, abound=[-2, 2])
        rcc_m, rcc_c, rcc_rho, rcc_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']

        # Wave
        try:
            wid1 = np.where(self.wave['tax'] < tin.min())[0][-1]
            wid2 = np.where(self.wave['tax'] > tin.max())[0][0]
            wave_t = self.wave['tax'][wid1:wid2]
            wave_phase = self.wave['phase'][wid1:wid2]
            wave_theta = self.wave['theta'][wid1:wid2]
        except IndexError as e:
            print(e)
            return None

        # Number of theta cycles, and cycles that have spikes
        cycidx = np.where(np.diff(wave_phase) < -(np.pi))[0]
        if cycidx.shape[0] == 0:
            wave_totalcycles = 1
            wave_truecycles = 0
            wave_maxperiod = tin.max() - tin.min()
        else:
            cyc_twindows = np.concatenate([wave_t[[0]], wave_t[cycidx], wave_t[[-1]]])
            presence = 0
            num_windows = cyc_twindows.shape[0] - 1
            for i in range(num_windows):
                t1, t2 = cyc_twindows[i], cyc_twindows[i + 1]
                if np.sum((tsp_withtheta <= t2) & (tsp_withtheta > t1)) > 0:
                    presence += 1
            wave_totalcycles = num_windows
            wave_truecycles = presence
            wave_maxperiod = np.max(np.diff(cyc_twindows))

        # Pass info
        data_dict['dsp'] = dsp
        data_dict['pass_nspikes'] = dsp.shape[0]
        data_dict['phasesp'] = phasesp
        data_dict['tsp'] = tsp
        data_dict['tsp_withtheta'] = tsp_withtheta
        data_dict['pass_angle'] = pass_angle
        data_dict['spike_angle'] = spike_angle
        data_dict['straightrank'] = straightrank
        # Pass regression
        data_dict['rcc_m'] = rcc_m
        data_dict['rcc_c'] = rcc_c
        data_dict['rcc_rho'] = rcc_rho
        data_dict['rcc_p'] = rcc_p
        # Pass LFP
        data_dict['wave_t'] = wave_t
        data_dict['wave_phase'] = wave_phase
        data_dict['wave_theta'] = wave_theta
        data_dict['wave_totalcycles'] = wave_totalcycles
        data_dict['wave_truecycles'] = wave_truecycles
        data_dict['wave_maxperiod'] = wave_maxperiod
        data_dict['cycfrac'] = wave_truecycles / wave_totalcycles
        return data_dict

    def _gen_precess_infokeys(self):
        keys_list = ['dsp', 'phasesp', 'tsp', 'tsp_withtheta', 'pass_angle', 'spike_angle', 'straightrank',
                     'rcc_m', 'rcc_c', 'rcc_rho', 'rcc_p', 'pass_nspikes',
                     'wave_t', 'wave_phase', 'wave_theta', 'cycfrac',
                     'wave_totalcycles', 'wave_truecycles', 'wave_maxperiod']
        return keys_list

    def gen_precess_dict(self, tag=''):
        precess_dict = {'%s%s'%(x, tag): [] for x in self.precess_keys}
        return precess_dict

    def append_pass_dict(self, precess_all_dict, precess_one_dict, tag=''):
        for key in self.precess_keys:
            precess_all_dict[key + tag].append(precess_one_dict[key])
        return precess_all_dict


class PrecessionFilter:
    def __init__(self):
        self.cycfrac_thresh = 0.2  # set 0.5
        self.min_num_truecycle = 2  # set 3
        self.sthresh = 3  # 9% of passes
        self.occ_thresh = 3
        self.maxperiod_thresh = 0.27  # in s, about 3 theta cycle, set 0.27

    def filter_single(self, precess_df, minocc):

        if minocc > self.occ_thresh:
            total_mask = (precess_df['cycfrac'] > self.cycfrac_thresh) & \
                         (precess_df['wave_truecycles'] > self.min_num_truecycle) & \
                         (precess_df['straightrank'] > self.sthresh) & \
                         (precess_df['wave_maxperiod'] < self.maxperiod_thresh) & \
                         (precess_df['rcc_m'] < 0) & \
                         (precess_df['rcc_m'] > -1.8)
        else:
            total_mask = [False] * precess_df.shape[0]

        precess_df['precess_exist'] = total_mask
        return precess_df

    def filter_pair(self, precess_df, minocc1, minocc2):

        if (minocc1 > self.occ_thresh) or (minocc2 > self.occ_thresh):

            cycfrac_mask = (precess_df['cycfrac1'] > self.cycfrac_thresh) & \
                           (precess_df['cycfrac2'] > self.cycfrac_thresh)
            true_cycle_mask = (precess_df['wave_truecycles1'] > self.min_num_truecycle) & (
                    precess_df['wave_truecycles2'] > self.min_num_truecycle)

            straightrank_mask = (precess_df['straightrank1'] > self.sthresh) & \
                                (precess_df['straightrank2'] > self.sthresh)
            max_period_mask = (precess_df['wave_maxperiod1'] < self.maxperiod_thresh) & \
                              (precess_df['wave_maxperiod2'] < self.maxperiod_thresh)

            slope_mask = (precess_df['rcc_m1'] < 0) & (precess_df['rcc_m1'] > -1.8) & \
                         (precess_df['rcc_m2'] < 0) & (precess_df['rcc_m2'] > -1.8)
            total_mask = cycfrac_mask & true_cycle_mask & straightrank_mask & max_period_mask & slope_mask


        else:
            total_mask = [False] * precess_df.shape[0]

        precess_df['precess_exist'] = total_mask
        return precess_df


class PrecessionStat:
    def __init__(self, precess_df, Nshuffles=200, passangle_key='spike_angle', tag=''):
        self.precess_df = precess_df
        self.startseed = 0
        self.tag = tag
        self.Nshuffles = Nshuffles
        self.passangle_key = passangle_key

    def compute_precession_stats(self, hist=False):
        filtered_precessdf = self.precess_df[self.precess_df['precess_exist']].reset_index(drop=True)

        if filtered_precessdf.shape[0] < 1:
            return filtered_precessdf, None, None
        else:

            pass_angles = filtered_precessdf[self.passangle_key + self.tag].to_numpy()
            pass_nspikes = filtered_precessdf['pass_nspikes' + self.tag].to_numpy()
            if hist:
                bestangle, R, norm_density = self.compute_R_hist(pass_angles, pass_nspikes)
            else:
                bestangle, R, norm_density = self.compute_R(pass_angles, pass_nspikes)

            info = {'norm_prob': norm_density, 'R': R}

            return filtered_precessdf, info, bestangle


    @staticmethod
    def compute_R(pass_angles, pass_nspikes):

        # Pass Counts
        pass_ax, pass_density = circular_density_1d(pass_angles, 16 * np.pi, 100, (-np.pi, np.pi))

        # Spike Counts
        anglediff_spikes = repeat_arr(pass_angles, pass_nspikes)
        spike_ax, spike_density = circular_density_1d(anglediff_spikes, 16 * np.pi, 100, (-np.pi, np.pi))

        # Normalized counts
        norm_density = normalize_distr(pass_density, spike_density)

        # Best direction
        bestangle = circmean(pass_ax, w=norm_density, d=pass_ax[1] - pass_ax[0])
        R = resultant_vector_length(pass_ax, w=norm_density, d=pass_ax[1] - pass_ax[0])

        return bestangle, R, norm_density


    @staticmethod
    def compute_R_hist(pass_angles, pass_nspikes):
        edge = np.linspace(-np.pi, np.pi, 36)
        edm = midedges(edge)
        # Pass Counts
        passbins, _ = np.histogram(pass_angles, bins=edge)

        # Spike Counts
        anglediff_spikes = repeat_arr(pass_angles, pass_nspikes)
        spikebins, _ = np.histogram(anglediff_spikes, bins=edge)

        # Normalized counts
        norm_density = normalize_distr(passbins, spikebins)

        # Best direction
        bestangle = circmean(edm, w=norm_density, d=edge[1] - edge[0])
        R = resultant_vector_length(edm, w=norm_density, d=edge[1] - edge[0])

        return bestangle, R, norm_density

def compute_precession_stats(precess_df, passangle_key='spike_angle', refangle=None, tag=''):
    if precess_df.shape[0] < 1:
        return None, None

    spike_angles = precess_df[passangle_key + tag].to_numpy()
    if refangle is not None:
        anglediff = cdiff(spike_angles, refangle)
    else:
        anglediff = spike_angles
    pass_nspikes = precess_df['pass_nspikes' + tag].to_numpy()

    # Pass Counts
    pass_ax, pass_density = circular_density_1d(anglediff, 16 * np.pi, 100, (-np.pi, np.pi))

    # Spike Counts
    anglediff_spikes = repeat_arr(anglediff, pass_nspikes)
    spike_ax, spike_density = circular_density_1d(anglediff_spikes, 16 * np.pi, 100, (-np.pi, np.pi))

    # Normalized counts
    norm_density = pass_density / spike_density
    norm_density[np.isnan(norm_density)] = 0
    norm_density[np.isinf(norm_density)] = 0
    norm_density = norm_density / np.sum(norm_density)

    # Best direction
    bestangle = circmean(pass_ax, w=norm_density, d=pass_ax[1] - pass_ax[0])
    bestangle = np.mod(bestangle + np.pi, 2 * np.pi) - np.pi  # transform range (0, 2pi) to (-pi, pi)
    R = resultant_vector_length(pass_ax, w=norm_density, d=pass_ax[1] - pass_ax[0])
    rayleigh_p, _ = rayleigh(pass_ax, w=norm_density * anglediff.shape[0])

    info = {'norm_prob': norm_density, 'R': R, 'rayleigh_p': rayleigh_p}
    return info, bestangle


def combined_average_curve(slopes, offsets, xrange=(0, 1), xbins=10):
    """

    Parameters
    ----------
    slopes : ndarray
        with shape (n, ). n is number of samples
    offsets : ndarray
    xrange : tuple
    xbins : int

    Returns
    -------

    """
    n = slopes.shape[0]
    assert n == offsets.shape[0]
    xdum = np.linspace(xrange[0], xrange[1], xbins)
    all_xdum = [xdum] * n
    all_ydum = []
    for i in range(n):
        ydum = xdum * 2 * np.pi * slopes[i] + offsets[i]
        all_ydum.append(ydum)
    all_xdum = np.concatenate(all_xdum)
    all_ydum = np.concatenate(all_ydum)
    regress = rcc(all_xdum, all_ydum)
    return regress



def permutation_test_average_slopeoffset(slopes_high, offsets_high, slopes_low, offsets_low, NShuffles=200):


    nhigh = slopes_high.shape[0]
    nlow = slopes_low.shape[0]
    ntotal = nhigh + nlow
    assert nhigh == offsets_high.shape[0]
    assert nlow == offsets_low.shape[0]

    regress_high = combined_average_curve(slopes_high, offsets_high)
    regress_low = combined_average_curve(slopes_low, offsets_low)

    slope_diff = cdiff(regress_high['aopt']*2*np.pi, regress_low['aopt']*2*np.pi)
    offset_diff = cdiff(regress_high['phi0'], regress_low['phi0'])

    pooled_slopes = np.append(slopes_high, slopes_low)
    pooled_offsets = np.append(offsets_high, offsets_low)


    all_shuf_slope_diff = np.zeros(NShuffles)
    all_shuf_offset_diff = np.zeros(NShuffles)
    for i in range(NShuffles):
        if i %5 == 0:
            print('Shuffling %d/%d'%(i, NShuffles))
        np.random.seed(i)
        ran_vec = np.random.permutation(ntotal)

        shuf_slopes = pooled_slopes[ran_vec]
        shuf_offsets = pooled_offsets[ran_vec]

        shuf_regress_high = combined_average_curve(shuf_slopes[0:nhigh], shuf_offsets[0:nhigh])
        shuf_regress_low = combined_average_curve(shuf_slopes[nhigh:], shuf_offsets[nhigh:])

        all_shuf_slope_diff[i] = cdiff(shuf_regress_high['aopt'] * 2 * np.pi, shuf_regress_low['aopt'] * 2 * np.pi)
        all_shuf_offset_diff[i] = cdiff(shuf_regress_high['phi0'], shuf_regress_low['phi0'])

    pval_slope = 1- np.mean(np.abs(slope_diff) > np.abs(all_shuf_slope_diff))
    pval_offset = 1 - np.mean(np.abs(offset_diff) > np.abs(all_shuf_offset_diff))

    return regress_high, regress_low, pval_slope, pval_offset


def get_single_precessdf(passdf, precesser, precess_filter, min_occbin, neuro_keys_dict):
    # Precession per pass
    neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                           spikeangle='spikeangle')
    pass_dict_keys = precesser._gen_precess_infokeys()
    pass_dict = {x: [] for x in pass_dict_keys}
    for precess_dict in precesser.get_single_precession(passdf, neuro_keys_dict):
        pass_dict = precesser.append_pass_dict(pass_dict, precess_dict)
    precess_df = pd.DataFrame(pass_dict)
    precessdf = precess_filter.filter_single(precess_df, minocc=min_occbin)
    precess_stater = PrecessionStat(precessdf)
    filtered_precessdf, precess_info , precess_angle = precess_stater.compute_precession_stats()
    nonprecess_df = precessdf[~precessdf['precess_exist']].reset_index(drop=True)





def construct_passdf_sim(analyzer, all_passidx, tidxsp, minpasstime=0.6):

    # x, y, t, v, angle, tsp, spikev, spikex, spikey, spikeangle
    pass_dict = dict(x=[], y=[], t=[], v=[], angle=[],
                     spikex=[], spikey=[], tsp=[], spikev=[], spikeangle=[])
    for pid1, pid2 in all_passidx:
        pass_t = analyzer.t[pid1:pid2]

        # Pass duration threshold
        if pass_t.shape[0] < 5:
            continue
        if (pass_t[-1] - pass_t[0]) < minpasstime:
            continue


        pass_dict['x'].append(analyzer.x[pid1:pid2])
        pass_dict['y'].append(analyzer.y[pid1:pid2])
        pass_dict['t'].append(pass_t)
        pass_dict['v'].append(analyzer.velocity[pid1:pid2])
        pass_dict['angle'].append(analyzer.movedir[pid1:pid2])

        tidxsp_within = tidxsp[(tidxsp >= pid1) & (tidxsp < pid2)]
        pass_dict['spikex'].append(analyzer.x[tidxsp_within])
        pass_dict['spikey'].append(analyzer.y[tidxsp_within])
        pass_dict['spikev'].append(analyzer.velocity[tidxsp_within])
        pass_dict['tsp'].append(analyzer.t[tidxsp_within])
        pass_dict['spikeangle'].append(analyzer.movedir[tidxsp_within])

    pass_df = pd.DataFrame(pass_dict)
    return pass_df



def construct_pairedpass_df_sim(analyzer, all_passidx_pair, tok1, tok2, tidxsp1, tidxsp2, minpasstime=0.6):

    # x, y, t, v, angle,
    # tsp1, spike1v, spike1x, spike1y, spike1angle
    # tsp2, spike2v, spike2x, spike2y, spike2angle
    pass_dict = dict(x=[], y=[], t=[], v=[], angle=[], infield1=[], infield2=[],
                     spike1x=[], spike1y=[], tsp1=[], spike1v=[], spike1angle=[],
                     spike2x=[], spike2y=[], tsp2=[], spike2v=[], spike2angle=[])

    for pid1, pid2 in all_passidx_pair:
        pass_t = analyzer.t[pid1:pid2]

        # Pass duration threshold
        if pass_t.shape[0] < 5:
            continue
        if (pass_t[-1] - pass_t[0]) < minpasstime:
            continue

        # Border crossing
        cross1 = np.sum(np.abs(np.diff(tok1[pid1:pid2])))
        cross2 = np.sum(np.abs(np.diff(tok2[pid1:pid2])))
        if (cross1 > 1) or (cross2 > 1):
            continue


        pass_dict['x'].append(analyzer.x[pid1:pid2])
        pass_dict['y'].append(analyzer.y[pid1:pid2])
        pass_dict['t'].append(pass_t)
        pass_dict['v'].append(analyzer.velocity[pid1:pid2])
        pass_dict['angle'].append(analyzer.movedir[pid1:pid2])
        pass_dict['infield1'].append(tok1[pid1:pid2])
        pass_dict['infield2'].append(tok2[pid1:pid2])

        tidxsp1_within = tidxsp1[(tidxsp1 >= pid1) & (tidxsp1 < pid2)]
        pass_dict['spike1x'].append(analyzer.x[tidxsp1_within])
        pass_dict['spike1y'].append(analyzer.y[tidxsp1_within])
        pass_dict['spike1v'].append(analyzer.velocity[tidxsp1_within])
        pass_dict['tsp1'].append(analyzer.t[tidxsp1_within])
        pass_dict['spike1angle'].append(analyzer.movedir[tidxsp1_within])

        tidxsp2_within = tidxsp2[(tidxsp2 >= pid1) & (tidxsp2 < pid2)]
        pass_dict['spike2x'].append(analyzer.x[tidxsp2_within])
        pass_dict['spike2y'].append(analyzer.y[tidxsp2_within])
        pass_dict['spike2v'].append(analyzer.velocity[tidxsp2_within])
        pass_dict['tsp2'].append(analyzer.t[tidxsp2_within])
        pass_dict['spike2angle'].append(analyzer.movedir[tidxsp2_within])

    pass_df = pd.DataFrame(pass_dict)
    return pass_df


