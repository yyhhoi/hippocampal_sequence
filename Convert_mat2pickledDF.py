import pickle
import h5py
import pandas as pd
import time
from common.mat2df_conversion import clean_df
from common.mat73_to_pickle import recursive_dict

if __name__ == '__main__':
    

    # Define paths of input data (.mat v7.3 file) and output data (a pandas pickle)
    input_pth = "data/emankindata_processed.mat"
    output_pth = "data/emankindata_processed_withwave.pickle"
    print("Loading H5 file:\n", input_pth, '\noutput to:\n', output_pth)


    # Converting .mat v7.3 to pandas dataframe by recursive methods using h5py as I/O
    print("Converting")
    f = h5py.File(input_pth, mode='r')
    data = recursive_dict(f)
    df = pd.DataFrame(data['day'])

    # Converting .mat v7.3 to dataframe by recursive h5py would produce a lot of errors
    # The cleaning process below removes these errors
    print("Cleaning")
    try:
        clean_df(df, output_pth, pairs=True, passes=True, indata=True)
    except:
        import pdb
        pdb.set_trace()
    
