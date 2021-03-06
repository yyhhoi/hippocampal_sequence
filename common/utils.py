import pickle
import numpy as np
import os

import skimage.io as ski
from skimage.color import rgb2gray, rgba2rgb

AB_list = [
        [1, 0, 0, 1],
        [1, 0, 1, 1],  # Run from A to A+B
        [1, 1, 0, 1],  # Run from A+B to A
]
BA_list = [
        [0, 1, 1, 0],
        [0, 1, 1, 1],  # Run from B to A+B
        [1, 1, 1, 0],  # Run from A+B to B
]

def load_pickle(fp):
    with open(fp, "rb") as f:
        f_dict = pickle.load(f)
    return f_dict



def mat2direction(loc):

    if loc in AB_list:
        return 'A->B'
    elif loc in BA_list:
        return 'B->A'
    else:
        return loc
    
def check_iter(a):
    
    if hasattr(a, '__iter__'):
        return True
    else:
        return False





def sigtext(pval, hidens=False):

    if pval < 0.0001:
        return '****'
    elif pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    elif pval > 0.05:
        if hidens:
            return ''
        else:
            return 'ns'
    else:
        return None

def p2str(pval):
    if pval < 0.0001:
        return '<0.0001'
    else:
        return '=%0.4f'%(pval)


def stat_record(fn, overwrite, *args):
    rootdir = 'writting/stats'
    pth = os.path.join(rootdir, fn)

    if overwrite:
        if os.path.isfile(pth):
            os.remove(pth)

    with open(pth, 'a') as fh:
        print_list = []
        for arg in args:

            if isinstance(arg, str):
                new_arg = arg
            elif (arg is None) or (np.isnan(arg)):
                new_arg = 'None'
            else:
                new_arg = str(np.around(arg, 4))

            print_list.append(new_arg)

        txt = ', '.join(print_list)
        fh.write(txt + '\n')
    return None



