import pandas as pd
import numpy as np
import pickle
import copy



def clean_df(df, output_path, pairs=False, passes=True, indata=False):
    null_arr1 = np.array([0, 0])
    null_arr2 = np.array([0, 1])

    if indata:
        for concerned_key in ['CA1indata', 'CA2indata', 'CA3indata']:
            change_list = []
            for i in range(df.shape[0]):
                content_dict = df.loc[i, concerned_key]
                if isinstance(content_dict, np.ndarray):  # suspect empty row
                    if np.array_equal(content_dict, null_arr1) or np.array_equal(content_dict, null_arr2):  # confirmed empty row
                        out_df = pd.DataFrame(dict())
                    else:
                        raise AssertionError('Unknown null row type')
                elif isinstance(content_dict, dict):  # non-empty
                    out_df = pd.DataFrame(copy.deepcopy(content_dict))
                change_list.append(out_df)
            df[concerned_key] = change_list

    # fields
    for concerned_key in ['CA1fields', 'CA2fields', 'CA3fields']:
        change_list = []
        for i in range(df.shape[0]):
            content_dict = df.loc[i, concerned_key]
            if isinstance(content_dict, np.ndarray):  # suspect empty row
                if np.array_equal(content_dict, null_arr1) or np.array_equal(content_dict, null_arr2):  # confirmed empty row
                    out_df = pd.DataFrame(dict())
                else:
                    raise AssertionError('Unknown null row type')
            elif isinstance(content_dict, dict):  # non-empty
                shape_len = len(content_dict['mask'].shape)
                if shape_len == 2:  # only one entry
                    re_dict = {key:[content_dict[key]] for key in content_dict.keys()}
                elif shape_len == 3:  # Multiple entry
                    re_dict = {key: list(content_dict[key]) for key in content_dict.keys()}
                else:
                    raise AssertionError('Shape len unexpected')
                out_df = pd.DataFrame(copy.deepcopy(re_dict))
            # Passes
            if passes:
                pass_df_list = []
                for out_df_i in range(out_df.shape[0]):
                    pass_dict = out_df.loc[out_df_i, 'passes']
                    if isinstance(pass_dict, np.ndarray):
                        if np.array_equal(pass_dict, null_arr1) or np.array_equal(pass_dict, null_arr2):  # confirmed empty row
                            pass_df = pd.DataFrame(dict())
                        else:
                            raise AssertionError('Unknown null row type')
                    elif isinstance(pass_dict, dict):  # non-empty
                        try:
                            pass_df = pd.DataFrame(copy.deepcopy(pass_dict))
                        except ValueError:
                            re_dict = {key: [pass_dict[key]] for key in pass_dict.keys()}
                            pass_df = pd.DataFrame(copy.deepcopy(re_dict))
                        for col in pass_df.columns:
                            empmask1 = pass_df[col].apply(lambda x: np.array_equal(x, null_arr1))
                            empmask2 = pass_df[col].apply(lambda x: np.array_equal(x, null_arr2))
                            empmask = empmask1 | empmask2
                            for emp_idx in empmask[empmask].index:
                                pass_df.at[emp_idx, col] = np.array([])
                            try:
                                pass_df[col] = pass_df[col].apply(lambda x:x.reshape(-1))
                            except AttributeError:
                                pass

                    pass_df_list.append(pass_df.copy())
                out_df['passes'] = pass_df_list
                
            change_list.append(out_df)
        df[concerned_key] = change_list
    if pairs:
        # pairs
        for concerned_key in ['CA1pairs', 'CA2pairs', 'CA3pairs']:
            change_list = []
            for i in range(df.shape[0]):
                content_dict = df.loc[i, concerned_key]
                if isinstance(content_dict, np.ndarray):  # suspect empty row
                    if np.array_equal(content_dict, null_arr1):  # confirmed empty row
                        out_df = pd.DataFrame(dict())
                    else:
                        raise AssertionError('Unknown null row type')
                elif isinstance(content_dict, dict):  # non-empty
                    shape_len = len(content_dict['fi'].shape)
                    if shape_len == 1:
                        re_dict = {key:[content_dict[key]] for key in content_dict.keys()}
                    elif shape_len == 2:
                        re_dict = {key: list(content_dict[key]) for key in content_dict.keys()}
                    else:
                        raise AssertionError('Shape len unexpected')
                    out_df = pd.DataFrame(re_dict)
                # Passes
                pass_df_list = []
                for out_df_i in range(out_df.shape[0]):
                    pass_dict = out_df.loc[out_df_i, 'pairedpasses']
                    try:
                        pass_df = pd.DataFrame(copy.deepcopy(pass_dict))
                    except ValueError:
                        re_dict = {key: [pass_dict[key]] for key in pass_dict.keys()}
                        pass_df = pd.DataFrame(copy.deepcopy(re_dict))
                    for col in pass_df.columns:
                        empmask1 = pass_df[col].apply(lambda x: np.array_equal(x, null_arr1))
                        empmask2 = pass_df[col].apply(lambda x: np.array_equal(x, null_arr2))
                        empmask = empmask1 | empmask2
                        for emp_idx in empmask[empmask].index:
                            pass_df.at[emp_idx, col] = np.array([])
                        try:
                            pass_df[col] = pass_df[col].apply(lambda x:x.reshape(-1))
                        except AttributeError:
                            pass
                    pass_df_list.append(pass_df.copy())
                out_df['pairedpasses'] = pass_df_list
                change_list.append(out_df)
            df[concerned_key] = change_list

    with open(output_path, 'wb') as fh:
        pickle.dump(df, fh,
                protocol=pickle.HIGHEST_PROTOCOL)

    return df

if __name__ == '__main__':

    datapickle = '/home/yiu/projects/hippocampus/data/emankindata_processed.pickle'
    
    with open(datapickle, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data['day'])
    clean_df(df, datapickle)