import numpy as np
import itertools
import pandas as pd
import os

import pickle
import msgpack
import msgpack_numpy as mnp

import featuretools as ft
import math

def doCombiStep(step, field, axis) -> np.ndarray:
    if(step == 'max'):
        return np.max(field, axis=axis)
    elif (step == 'sum'):
        return np.sum(field, axis=axis)
    
def ce(data):
    summer = 0
    for i in range(len(data)-1):
        summer += math.pow(data[i] - data[i+1], 2)
    return math.sqrt(summer)

def flatten(X, pos = -1):
    """ Flatten a 3D np array """
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    if pos == -1:
        pos = X.shape[1]-1
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, pos, :]
    return(flattened_X)

def scale(X, scaler):
    """ Scale 3D array. X = 3D array, scalar = scale object from sklearn. Output = scaled 3D array. """
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

def symbolize(X, scaler):
    """ Symbolize a 3D array. X = 3D array, scalar = SAX symbolizer object. Output = symbolic 3D string array. """
    X_s = scaler.transform(X)
    return X_s

def trans(val, vocab) -> float:
    """ Translate the a string [a,e] between [-1,1] """
    for i in range(len(vocab)):
        if val == vocab[i]:
            halfSize = (len(vocab)-1)/2
            return (i - halfSize) / halfSize
    return -2

def getMapValues(size):
    """ Get the map values """
    vMap = []
    for i in range(size):
        halfSize = (size-1)/2
        vMap.append((i - halfSize) / halfSize)
    return vMap

def symbolizeTrans(X, scaler, sinfo, bins = 5):
    """ Symbolize and transform data """
    vocab = sinfo._check_params(bins)
    X_s = scaler.transform(X.tolist())
    X = np.zeros(X_s.shape)
   
    for i in range(X.shape[0]):
        X = X.astype(float)
        
        for j in range(X.shape[1]):
            X[i][j] = trans(X_s[i][j], vocab)
    return X

def symbolizeTrans2(X, scaler, bins = 5):
    """ Symbolize and transform data """
    vocab = scaler._check_params(bins)
    for i in range(X.shape[0]):
        X_s = X.astype(str) 
        z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        X_s[i, :, :][:,0] = z1
        for j in range(X.shape[1]):
            X[i][j][0] = trans(X_s[i][j][0], vocab)
    return X

def split_dataframe(df, chunk_size = 10000): 
    """ Split datafram into chunks """
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def save_obj(obj, name, method='pickle'):
    """ Save an object """
    if method == 'pickle':
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    elif method == 'msgpack':
        with open(name + '.msgpack', 'wb') as f:
            packed_data = msgpack.packb(obj, default=mnp.encode)
            f.write(packed_data)

def load_obj(name, method='pickle'):
    """ Load an object """
    if method == 'pickle':
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
        
    elif method == 'msgpack':
        with open(name + '.msgpack', 'rb') as f:
            return msgpack.unpackb(f.read(), object_hook=mnp.decode)
    
def truncate(n):
    return int(n * 1000) / 1000


def modelFidelity(modelPrediction, interpretationPrediction):
    """ Calculate fidelity """
    summer = 0
    for i in range(len(modelPrediction)):
        if modelPrediction[i] == interpretationPrediction[i]:
            summer += 1
    return summer / len(modelPrediction)

def fillOutDicWithNNOutSmall(outData):
    """ Fill result dictionary with data """
    inputDict = dict()
    inputDict['Val Accuracy'] = outData[0][0]
    inputDict['Val Precision'] = outData[0][1]
    inputDict['Val Recall'] = outData[0][2]
    inputDict['Val F1'] = outData[0][3]
    inputDict['Test Accuracy'] = outData[1][0]
    inputDict['Test Precision'] = outData[1][1]
    inputDict['Test Recall'] = outData[1][2]
    inputDict['Test F1'] = outData[1][3]
    inputDict['Train Predictions'] = outData[2][0]
    inputDict['Val Predictions'] = outData[2][1]
    inputDict['Test Predictions'] = outData[3]

    return inputDict

def to_df_tsfresh(X):
    """ Transform data into DataFrame suitable for TSFresh """
    X = X.squeeze()
   
    num_time_series = X.shape[0]
    num_timestamps = X.shape[1]

    ids = []
    times = []
    values = []

    id = 1
    for time_series in range(num_time_series):
        time = 1
        for timestamp in range(num_timestamps):
            ids.append(id)
            times.append(time)
            values.append(X[time_series][timestamp])
            time += 1
        id += 1

    data = {
        'id': ids,
        'time': times,
        'value': values
    }

    df = pd.DataFrame(data)

    return df

def y_to_tsf(y):
    """ Transform label data into TSFresh format """
    return np.asarray([np.argmax(row) for row in y])


def tsf_remove_constant_features(features):
    """ Remove constant features from TSFresh features """
    feat = features
    for key in features.keys():
        if features[key].max() - features[key].min() == 0:
            del feat[key]
    return feat

def c22_get_non_constant_features(features):
    """ Remove constant features from Catch22 features """
    column_ranges = np.ptp(features, axis=0)
    constant_column_indices = np.where(column_ranges != 0)[0]

    return constant_column_indices

def to_df_tsfeatures(X):
    """ Transform data into DataFrame suitable for tsfeatures"""
    n_timeseries = X.shape[0]
    n_timesteps = X.shape[1]

    data_flattened = X.reshape(n_timeseries, n_timesteps)

    df_list = []
    for i in range(n_timeseries):
        df_temp = pd.DataFrame({
            'unique_id': i+1,  # Unique ID for each time series
            'ds': pd.date_range(start='2024-01-01', periods=n_timesteps, freq='D'),  # Time index as dates
            'y': data_flattened[i]  # Time series values
        })
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    return df

def postprocess_tsfeatures(features, remove_constant_columns=False):
    """ Postprocessing of tsfeatures features """
    features = features.drop(columns=['unique_id'])
    # features = features.dropna(axis='columns')
    if remove_constant_columns:
        features = tsf_remove_constant_features(features)
    
    return features

def get_featuretools_features(data):
    """ Create and get feature with featuretools """
    n_timeseries = data.shape[0]
    n_timesteps = data.shape[1]

    data_flattened = data.squeeze()
    time_series_df = pd.DataFrame({
        'time_series_id': [i+1 for i in range(n_timeseries)]
    })

    observations_list = []
    for i in range(n_timeseries):
        df_temp = pd.DataFrame({
            'time_series_id': i+1, 
            'time_step': np.arange(n_timesteps),  
            'value': data_flattened[i] 
        })
        observations_list.append(df_temp)

    observations_df = pd.concat(observations_list, ignore_index=True)
    
    es = ft.EntitySet(id='time_series_data')
    es = es.add_dataframe(dataframe_name='time_series', dataframe=time_series_df, index='time_series_id')
    es = es.add_dataframe(dataframe_name='observations', dataframe=observations_df, 
                        index='observation_id',  # Auto-generated index for each observation
                        make_index=True,         # Generate the 'observation_id'
                        time_index='time_step')  # Time index
    es = es.add_relationship(parent_dataframe_name='time_series', 
                            parent_column_name='time_series_id',
                            child_dataframe_name='observations', 
                            child_column_name='time_series_id')

    features_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='time_series')

    return features_matrix

def postprocess_featuretools(features, remove_constant_columns=False):
    """ Postprocessing of featuretools """
    features = features.dropna(axis='columns')
    if remove_constant_columns:
        features = tsf_remove_constant_features(features)
    
    return features

def is_feature_extraction(method):
    """ Check if the method extracts features """
    if method == 'ORI':
        return False
    elif method == 'SAX':
        return False
    elif method == 'SFA':
        return False
    elif method == 'MCB':
        return False
    elif method == 'TSFRESH':
        return True
    elif method == 'TSFRESH_SEL':
        return True
    elif method == 'TSFRESH_SYM':
        return True
    elif method == 'TSFRESH_SEL_SYM':
        return True
    elif method == 'TSFEL':
        return True
    elif method == 'TSFEL_SYM':
        return True
    elif method == 'CATCH22':
        return True
    elif method == 'CATCH22_SYM':
        return True
    elif method == 'CATCH24':
        return True
    elif method == 'CATCH24_SYM':
        return True
    elif method == 'TSFEAT':
        return True
    elif method == 'TSFEAT_SYM':
        return True
    elif method == 'TOOLS':
        return True
    elif method == 'TOOLS_SYM':
        return True
    
    return False

def do_symbolize(method):
    """ Check if method needs to be symbolized further """
    if method == 'ORI':
        return False
    elif method == 'SAX':
        return False
    elif method == 'SFA':
        return False
    elif method == 'MCB':
        return True
    elif method == 'TSFRESH':
        return False
    elif method == 'TSFRESH_SEL':
        return False
    elif method == 'TSFRESH_SYM':
        return True
    elif method == 'TSFRESH_SEL_SYM':
        return True
    elif method == 'TSFEL':
        return False
    elif method == 'TSFEL_SYM':
        return True
    elif method == 'CATCH22':
        return False
    elif method == 'CATCH22_SYM':
        return True
    elif method == 'CATCH24':
        return False
    elif method == 'CATCH24_SYM':
        return True
    elif method == 'TSFEAT':
        return False
    elif method == 'TSFEAT_SYM':
        return True
    elif method == 'TOOLS':
        return False
    elif method == 'TOOLS_SYM':
        return True
    elif method == 'SHAPE_MCB':
        return True
    
    return False

def is_symbolified(method):
    """ Check if method was symbolized"""
    if method == 'SAX':
        return True
    elif method == 'SFA':
        return True
    elif method == 'MCB':
        return True
    elif method == 'TSFRESH_SYM':
        return True
    elif method == 'TSFRESH_SEL_SYM':
        return True
    elif method == 'TSFEL_SYM':
        return True
    elif method == 'CATCH22_SYM':
        return True
    elif method == 'CATCH24_SYM':
        return True
    elif method == 'TSFEAT_SYM':
        return True
    elif method == 'TOOLS_SYM':
        return True
    
    return False

def set_method(method):
    """ Set correct flag for method """
    calcTSF = False # calc the tsfresh model
    calcTSF_SEL = False # calc tsfresh model and select relevant features
    calcTSF_SYM = False # calc tsfresh model and symbolify
    calcTSF_SEL_SYM = False # calc tsfresh model and select relevant features and symbolify
    calcFEL = False #cal the tsfel model
    calcFEL_SYM = False # calc the tsfel model and symbolify
    calcC22 = False # calc the catch22 model
    calcC22_SYM = False # calc the catch22 model and symbolify
    calcC24 = False # calc the catch24 model
    calcC24_SYM = False # calc the catch24 model and symbolify
    calcTSFEAT = False # calc the tsfeatures model
    calcTSFEAT_SYM = False # calc the tsfeatures model and symbolify
    calcTOOLS = False # calc the featuretools model
    calcTOOLS_SYM = False # calc the featuretools model and symbolify

    
    if method == 'TSFRESH':
        calcTSF = True
    elif method == 'TSFRESH_SEL':
        calcTSF_SEL = True
    elif method == 'TSFRESH_SYM':
        calcTSF_SYM = True
    elif method == 'TSFRESH_SEL_SYM':
        calcTSF_SEL_SYM = True
    elif method == 'TSFEL':
        calcFEL = True
    elif method == 'TSFEL_SYM':
        calcFEL_SYM = True
    elif method == 'CATCH22':
        calcC22 = True
    elif method == 'CATCH22_SYM':
        calcC22_SYM = True
    elif method == 'CATCH24':
        calcC24 = True
    elif method == 'CATCH24_SYM':
        calcC24_SYM = True
    elif method == 'TSFEAT':
        calcTSFEAT = True
    elif method == 'TSFEAT_SYM':
        calcTSFEAT_SYM = True
    elif method == 'TOOLS':
        calcTOOLS = True
    elif method == 'TOOLS_SYM':
        calcTOOLS_SYM = True

    return calcTSF, calcTSF_SEL, calcTSF_SYM, calcTSF_SEL_SYM, calcFEL, calcFEL_SYM, calcC22, calcC22_SYM, calcC24, calcC24_SYM, calcTSFEAT, calcTSFEAT_SYM, calcTOOLS, calcTOOLS_SYM

def do_shapelets(method):
    """ Determine if shapelets are used """
    if (method == 'SHAPE') or (method == 'SHAPE_MCB') or (method == 'SHAPE_SAX') or (method == 'SHAPE_SFA'):
        return True

def combine_factors(factors_list) -> list[dict]:
    """
    Writes all possible combinations that can be constructed
    from a set of lists into a single list.

    Args:
        factors (dict[str, list]): each entry comprises a factor name
        associated with a list of possible values

    Returns:
        list[dict]: list of all possible factor combinations
    """
    factor_names           = [k for k in factors_list[0]]
    unnamed_combinations = []

    for factors in factors_list:
        # cartesian product of all parameter sets
        unnamed_combinations    += list(itertools.product(*factors.values()))
        
        # name combinations from keys
    return [dict(zip(factor_names, factor_l)) for factor_l in unnamed_combinations]

def get_config_factors() -> list[dict]:
    """ Set the values for the possible configs """
    config_factors = [
        {
            'symbolCount':          [6],
            'numOfAttentionLayers': [2],
            'header':               [8],
            'dmodel':               [32],
            'dff':                  [32],
            'lr':                   ['custom'],
            'warmup':               [10000]
        },
        {
            'symbolCount':          [7],
            'numOfAttentionLayers': [2],
            'header':               [8],
            'dmodel':               [32],
            'dff':                  [16],
            'lr':                   ['custom'],
            'warmup':               [10000]
        }
    ]

    return config_factors

def get_ds_methods_factors():
    """ Set combinations for datasets and methods """
    ds_methods_factors = [
        {
            'dataset':      ['ucr'],
            'takename':     [True],
            'number':       ['Meat'],#['MedicalImages'],#['SyntheticControl'],#['Wine'],#['Fish'],#['DistalPhalanxOutlineCorrect'],
            'method':       ['ORI', 'MCB', 'TSFRESH', 'TSFRESH_SEL', 'TSFRESH_SYM', 'TSFRESH_SEL_SYM', 'TSFEL', 'TSFEL_SYM', 'CATCH22', 'CATCH22_SYM', 'CATCH24', 'CATCH24_SYM', 'TSFEAT', 'TSFEAT_SYM', 'TOOLS', 'TOOLS_SYM']
        }
    ]

    return ds_methods_factors

def get_numbers() -> list:
    """ Get the datasets numbers corresponding to the name """
    factors_list = get_ds_methods_factors()
    ls = []
    for dic in factors_list:
        if dic['dataset'][0] == 'ucr':
            ls.append(dic['number'][0])
    return ls

def get_ds_numbers() ->list[dict]:
    """ Get the datasets numbers """
    factors_list = get_ds_methods_factors()
    for dic in factors_list:
        dic.pop('method')
    return combine_factors(factors_list)

def generate_ds_methods() -> list[dict]:
    """ Generate all combinations of datasets and methods """
    factors_list = get_ds_methods_factors()
    return combine_factors(factors_list)

def generate_configs() -> list[dict]:
    """ Get the datasets numbers corresponding to the name """
    factors_list = get_config_factors()
    return combine_factors(factors_list)

def merge_config_dicts() -> dict:
    """ Merges the config dicts into one dictionary (for evaluation) """
    factors_list = get_config_factors()

    # Initialize an empty dictionary to store the merged result
    merged_dict = {}

    # Iterate through each dictionary in the list
    for d in factors_list:
        for key, values in d.items():
            if key not in merged_dict:
                # Initialize a set for each key
                merged_dict[key] = set()
            # Add the value to the set (ensuring uniqueness)
            for value in values:
                merged_dict[key].add(value)

    # If you prefer lists instead of sets
    for key in merged_dict:
        merged_dict[key] = list(merged_dict[key])

    return merged_dict

def check_min_samples_before_splitting(y, k):
    unique, counts = np.unique(y, return_counts=True)
    count_y = dict(zip(unique, counts))

    for key in count_y.keys():
        if count_y[key] < k:
            print(f"Class {key} has only {count_y[key]} samples, less than {k}.")
            return False
    return True