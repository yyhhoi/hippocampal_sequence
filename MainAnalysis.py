import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d, make_interp_spline
from pycircstat.descriptive import resultant_vector_length, cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import ranksums, chi2_contingency

from common.utils import load_pickle, stat_record
from common.comput_utils import check_border, normalize_distr, IndataProcessor, midedges, segment_passes, window_shuffle_gen, \
    compute_straightness, heading, pair_diff, check_border_sim, append_info_from_passes, \
    window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, DirectionerBining, DirectionerMLM, \
    window_shuffle_wrapper, timeshift_shuffle_exp_wrapper, circular_density_1d

from common.visualization import color_wheel, directionality_polar_plot, customlegend

from common.script_wrappers import DirectionalityStatsByThresh
from common.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw




def main():

    # Figure 1
    data_pth = 'results/exp/single_field/singlefield_df.pickle'
    plot_dir = 'manuscript_plots'



    return None



if __name__ == '__main__':
    main()

