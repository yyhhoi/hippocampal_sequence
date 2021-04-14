import os
import shutil
import numpy as np
import skimage.io as ski
from skimage.color import rgb2gray, rgba2rgb
from os.path import join

def select_cut(src_pths, dest_pths, margin):
    min_rowl, max_rowl = [], []
    min_coll, max_coll = [], []


    # Determine frame size
    for img_pth in src_pths:
        img = ski.imread(img_pth)
        img_g = rgb2gray(rgba2rgb(img))
        rowTrue, colTrue = np.where(img_g < 0.99)
        min_row, max_row = rowTrue.min(), rowTrue.max()
        min_col, max_col = colTrue.min(), colTrue.max()
        min_rowl.append(min_row)
        max_rowl.append(max_row)
        min_coll.append(min_col)
        max_coll.append(max_col)

    min_row, max_row = np.min(min_rowl) - margin, np.max(max_rowl) + margin
    min_col, max_col = np.min(min_coll) - margin, np.max(max_coll) + margin

    min_row = max(0, min_row)
    min_col = max(0, min_col)

    # Cut frame
    for img_pth, dest_pth in zip(src_pths, dest_pths):
        img = ski.imread(img_pth)
        img_cut = img[min_row:max_row, min_col:max_col]
        ski.imsave(dest_pth, img_cut)

def fetch_figure1(resultplot_dir, dest_dir):
    sigdir = join(resultplot_dir, 'single_field')
    egdir = join(sigdir, 'example_fields')

    examples_fns = ['CA1-field151.eps',
                    'CA1-field222.eps',
                    'CA1-field389.eps',
                    'CA1-field400.eps',
                    'CA1-field409.eps',
                    'CA2-field28.eps',
                    'CA2-field47.eps',
                    'CA3-field75.eps']

    for fn in examples_fns:
        shutil.copy(join(egdir, fn), join(dest_dir, fn))

    # Directionality
    fns = ['exp_single_directionality.eps', 'colorwheel.eps']
    for fn in fns:
        shutil.copy(join(sigdir, fn), join(dest_dir, fn))

def fetch_figure2(resultplot_dir, dest_dir):

    # Fig 2D, E, F
    singlepass_dir = join(resultplot_dir, 'passes', 'single')

    fns = ['field_precess_pishift.eps', 'field_precess_R.eps', 'precess_examples.eps',
           'fraction_neg_slopes.eps']
    for fn in fns:
        shutil.copy(join(singlepass_dir, fn),
                    join(dest_dir, fn) )


def fetch_figure3(resultplot_dir, dest_dir):

    singlepass_dir = join(resultplot_dir, 'passes', 'single')

    fns = ['Marginals.eps', 'AverCurves_SpikePhase.eps', 'adiff_colorbar.eps']
    for fn in fns:
        shutil.copy(join(singlepass_dir, fn),
                    join(dest_dir, fn) )


def fetch_figure4(resultplot_dir, dest_dir):
    pairfield_dir = join(resultplot_dir, 'pair_fields')

    fns = ['examples_pair.eps', 'example_pairedspikes.eps', 'exp_pair_directionality.eps', 'exp_pair_fraction.eps']
    for fn in fns:
        shutil.copy(join(pairfield_dir, fn),
                    join(dest_dir, fn) )

# def fetch_figure5(resultplot_dir, dest_dir):
#     pairfield_dir = join(resultplot_dir, 'pair_fields')
#
#     fns = ['examples_kld.eps', 'exp_kld_thresh.eps', 'pair_single_angles.eps', 'pair_single_angles_colorbar.eps',
#            'pair_single_angles_test.eps']
#     for fn in fns:
#         shutil.copy(join(pairfield_dir, fn),
#                     join(dest_dir, fn) )

def fetch_figure5(resultplot_dir, dest_dir):

    pairfield_dir = join(resultplot_dir, 'pair_fields')
    fns = ['examples_correlogram.eps', 'exp_paircorr.eps']
    for fn in fns:
        shutil.copy(join(pairfield_dir, fn),
                    join(dest_dir, fn) )


def fetch_figure6(resultplot_dir, dest_dir, margin=3):

    pairfield_dir = join(resultplot_dir, 'pair_fields')
    fns = ['examples_exintrinsicity.eps', 'exp_exintrinsic.eps', 'exp_exintrinsic_concentration.eps']
    for fn in fns:
        shutil.copy(join(pairfield_dir, fn),
                    join(dest_dir, fn) )

    shutil.copy(join(pairfield_dir, 'pairangle_similarity', 'exp_simdissim_exintrinsic2d.eps'),
                join(dest_dir, 'exp_simdissim_exintrinsic2d.eps') )

def fetch_figure7(resultplot_dir, dest_dir):
    src_dir = join(resultplot_dir, 'pair_fields')

    fns = ['exp_4case.eps']

    for fn in fns:
        shutil.copy(join(src_dir, fn),
                    join(dest_dir, fn) )

def fetch_figure8(resultplot_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    src_dir = join(resultplot_dir, 'sim')
    fns = ['SIM_single.eps']
    for fn in fns:
        shutil.copy(join(src_dir, fn),
                    join(dest_dir, fn) )

def fetch_figure9(resultplot_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    src_dir = join(resultplot_dir, 'sim')
    fns = ['SIM_pair.eps']
    for fn in fns:
        shutil.copy(join(src_dir, fn),
                    join(dest_dir, fn) )

resultplot_dir = '/home/yiu/projects/hippocampus/result_plots'
fig_dir = '/home/yiu/projects/hippocampus/writting/figures'

# fetch_figure1(resultplot_dir, join(fig_dir, 'fig1'))
# fetch_figure2(resultplot_dir, join(fig_dir, 'fig2'))
# fetch_figure3(resultplot_dir, join(fig_dir, 'fig3'))
# fetch_figure4(resultplot_dir, join(fig_dir, 'fig4'))
# fetch_figure5(resultplot_dir, join(fig_dir, 'fig5'))
fetch_figure6(resultplot_dir, join(fig_dir, 'fig6'))
fetch_figure7(resultplot_dir, join(fig_dir, 'fig7'))
fetch_figure8(resultplot_dir, join(fig_dir, 'fig8'))
fetch_figure9(resultplot_dir, join(fig_dir, 'fig9'))
