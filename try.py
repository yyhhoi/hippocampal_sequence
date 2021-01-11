import numpy as np
import matplotlib.pyplot as plt

from common.utils import load_pickle



df_main = load_pickle('results/exp/single_field/singlefield_df.pickle')
df_append = load_pickle('results/exp/single_field/singlefield_df_forAppend.pickle')


df_main['nonprecess']