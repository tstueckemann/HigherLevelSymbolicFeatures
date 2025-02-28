import numpy as np

import os
import shutil

from modules import helper

from sklearn.preprocessing import StandardScaler

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from tsfel import get_features_by_domain
from tsfel import time_series_features_extractor

from tsfeatures import tsfeatures

import pycatch22

def save_foldwise_features(number, dataName, method, x_train, X_test, y_train, y_test, y_trainy, y_testy, kf, saveMethod='pickle'):
    """ Extract and save the features foldwise """

    print("---------------------------------- SAVING FEATURES ----------------------------------")
    fold = 0

    for train, val in kf.split(x_train, y_trainy):
        fold += 1
        print(f"---------------------------------- FOLD {fold} ----------------------------------")
        # Define the path using number, method, and fold
        path = 'features' + '/' + str(number) + '_' + (dataName) + '/' + method + '/' + "fold_" + str(fold)
        if os.path.exists(path):
            print("File already exists")
            continue
        

        x_train1 = x_train[train]
        y_train1 = y_train[train]
        y_trainy1 = y_trainy[train]

        x_val = x_train[val]
        y_val = y_train[val]
        y_valy = y_trainy[val]

        x_test = X_test.copy()

        x_train1 = x_train1.squeeze()
        x_val = x_val.squeeze()
        x_test = x_test.squeeze()
        
        trainShape = x_train1.shape
        valShape = x_val.shape
        testShape = x_test.shape

        scaler = StandardScaler()
        scaler = scaler.fit(x_train1.reshape((-1, 1)))
        x_train1 = scaler.transform(x_train1.reshape(-1, 1)).reshape(trainShape)
        x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(valShape)
        x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(testShape)


        doTSF, doTSF_SEL, doTSF_SYM, doTSF_SEL_SYM, doFEL, doFEL_SYM, doC22, doC22_SYM, doC24, doC24_SYM, doTSFEAT, doTSFEAT_SYM, doTOOLS, doTOOLS_SYM = helper.set_method(method)

        is_multiclass = True
        if len(np.unique(y_train)) <= 2:
            is_multiclass = False

        # Extract unfiltered TSFresh features for train, validation data
        if doTSF:
            x_train1_df = helper.to_df_tsfresh(x_train1)
            extracted_features_train1 = extract_features(x_train1_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            x_train1 = extracted_features_train1.to_numpy()

            x_val_df = helper.to_df_tsfresh(x_val)
            extracted_features_val = extract_features(x_val_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            x_val = extracted_features_val.to_numpy()

            x_test_df = helper.to_df_tsfresh(x_test)
            extracted_features_test = extract_features(x_test_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            x_test = extracted_features_test.to_numpy()

        # Extract relevant TSFresh features for train, validation data
        elif doTSF_SEL:
            x_train1_df = helper.to_df_tsfresh(x_train1)
            extracted_features_train1 = extract_features(x_train1_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            selected_features_train1 = select_features(extracted_features_train1, helper.y_to_tsf(y_train1), ml_task='classification', multiclass=is_multiclass)
            x_train1 = selected_features_train1.to_numpy()
            if selected_features_train1.empty:   # No relevant features selected
                return False

            x_val_df = helper.to_df_tsfresh(x_val)
            extracted_features_val = extract_features(x_val_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            selected_features_val = extracted_features_val[selected_features_train1.columns]  # Only extract features from the val and val set that are relevant in the train set
            x_val = selected_features_val.to_numpy()

            x_test_df = helper.to_df_tsfresh(x_test)
            extracted_features_test = extract_features(x_test_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            selected_features_test = extracted_features_test[selected_features_train1.columns]  # Only extract features from the val and val set that are relevant in the train set
            x_test = selected_features_test.to_numpy()

        # Extract unfiltered TSFresh features for train, validation data and bin them via MultipleCoefficientBinning
        elif doTSF_SYM:
            x_train1_df = helper.to_df_tsfresh(x_train1)
            extracted_features_train1 = extract_features(x_train1_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            x_train1 = helper.tsf_remove_constant_features(extracted_features_train1)

            x_val_df = helper.to_df_tsfresh(x_val)
            extracted_features_val = extract_features(x_val_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            x_val = extracted_features_val[extracted_features_train1.columns] # Only extract features from the val and val set that are relevant in the train set

            x_test_df = helper.to_df_tsfresh(x_test)
            extracted_features_test = extract_features(x_test_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            x_test = extracted_features_test[extracted_features_train1.columns] # Only extract features from the val and val set that are relevant in the train set


        # Extract relevant TSFresh features for train, validation data and bin them via MultipleCoefficientBinning
        elif doTSF_SEL_SYM:
            x_train1_df = helper.to_df_tsfresh(x_train1)
            extracted_features_train1 = extract_features(x_train1_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            selected_features_train1 = select_features(extracted_features_train1, helper.y_to_tsf(y_train1), ml_task='classification', multiclass=is_multiclass)
            if selected_features_train1.empty: # No relevant features selected
                return False
            x_train1 = selected_features_train1.to_numpy()

            x_val_df = helper.to_df_tsfresh(x_val)
            extracted_features_val = extract_features(x_val_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            selected_features_val = extracted_features_val[selected_features_train1.columns]  # Only extract features from the val and val set that are relevant in the train set
            x_val = selected_features_val.to_numpy()

            x_test_df = helper.to_df_tsfresh(x_test)
            extracted_features_test = extract_features(x_test_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
            selected_features_test = extracted_features_test[selected_features_train1.columns]  # Only extract features from the val and val set that are relevant in the train set
            x_test = selected_features_test.to_numpy()

        elif doFEL:
            # tsfel needs expanded dimensions for some reason
            x_train1 = np.expand_dims(x_train1, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
            
            cfg = get_features_by_domain() # Extracts the temporal, statistical and spectral feature sets.
            
            extracted_features_train1 = time_series_features_extractor(cfg, x_train1)#, fs=100)
            x_train1 = extracted_features_train1.to_numpy()
            
            extracted_features_val = time_series_features_extractor(cfg, x_val)
            x_val = extracted_features_val.to_numpy()

            extracted_features_test = time_series_features_extractor(cfg, x_test)
            x_test = extracted_features_test.to_numpy()
        

        elif doFEL_SYM:
            # tsfel needs expanded dimensions for some reason
            x_train1 = np.expand_dims(x_train1, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
            
            cfg = get_features_by_domain() # Extracts the temporal, statistical and spectral feature sets.
            
            extracted_features_train1 = time_series_features_extractor(cfg, x_train1)#, fs=100)
            extracted_features_train1 = helper.tsf_remove_constant_features(extracted_features_train1)
            x_train1 = extracted_features_train1.to_numpy()
            
            extracted_features_val = time_series_features_extractor(cfg, x_val)
            extracted_features_val = extracted_features_val[extracted_features_train1.columns] # Only extract features from the val and val set that are relevant in the train set
            x_val = extracted_features_val.to_numpy()

            extracted_features_test = time_series_features_extractor(cfg, x_test)
            extracted_features_test = extracted_features_test[extracted_features_train1.columns] # Only extract features from the val and val set that are relevant in the train set
            x_test = extracted_features_test.to_numpy()

        elif doC22:
            x_train1 = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_train1])

            x_val = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_val]) 

            x_test = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_test]) 

        elif doC22_SYM:
            extracted_features_train1 = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_train1])
            ids = helper.c22_get_non_constant_features(extracted_features_train1) # indices of non-constant columns
            extracted_features_train1 = extracted_features_train1[:, ids]
            x_train1 = extracted_features_train1

            extracted_features_val = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_val]) 
            extracted_features_val = extracted_features_val[:, ids]
            x_val = extracted_features_val

            extracted_features_test = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_test]) 
            extracted_features_test = extracted_features_test[:, ids]
            x_test = extracted_features_test

        elif doC24:
            x_train1 = np.asarray([pycatch22.catch22_all(ts, catch24=True)['values'] for ts in x_train1])

            x_val = np.asarray([pycatch22.catch22_all(ts, catch24=True)['values'] for ts in x_val]) 

            x_test = np.asarray([pycatch22.catch22_all(ts, catch24=True)['values'] for ts in x_test]) 

        elif doC24_SYM:
            extracted_features_train1 = np.asarray([pycatch22.catch22_all(ts, catch24=True)['values'] for ts in x_train1])
            ids = helper.c22_get_non_constant_features(extracted_features_train1) # indices of non-constant columns
            extracted_features_train1 = extracted_features_train1[:, ids]
            x_train1 = extracted_features_train1

            extracted_features_val = np.asarray([pycatch22.catch22_all(ts, catch24=True)['values'] for ts in x_val]) 
            extracted_features_val = extracted_features_val[:, ids]
            x_val = extracted_features_val

            extracted_features_test = np.asarray([pycatch22.catch22_all(ts, catch24=True)['values'] for ts in x_test]) 
            extracted_features_test = extracted_features_test[:, ids]
            x_test = extracted_features_test

        elif doTSFEAT:
            df_tsfeat_train1 = helper.to_df_tsfeatures(x_train1)
            extracted_features_train1 = tsfeatures(df_tsfeat_train1)
            extracted_features_train1 = extracted_features_train1.replace(np.nan, 0.0)
            extracted_features_train1 = helper.postprocess_tsfeatures(extracted_features_train1, remove_constant_columns=False)
            x_train1 = extracted_features_train1.to_numpy()

            df_tsfeat_val = helper.to_df_tsfeatures(x_val)
            extracted_features_val = tsfeatures(df_tsfeat_val)
            x_val = extracted_features_val[extracted_features_train1.columns].replace(np.nan, 0.0).to_numpy()

            df_tsfeat_test = helper.to_df_tsfeatures(x_test)
            extracted_features_test = tsfeatures(df_tsfeat_test)
            x_test = extracted_features_test[extracted_features_train1.columns].replace(np.nan, 0.0).to_numpy()

        elif doTSFEAT_SYM:
            df_tsfeat_train1 = helper.to_df_tsfeatures(x_train1)
            extracted_features_train1 = tsfeatures(df_tsfeat_train1)
            extracted_features_train1 = extracted_features_train1.replace(np.nan, 0.0)
            extracted_features_train1 = helper.postprocess_tsfeatures(extracted_features_train1, remove_constant_columns=True)
            x_train1 = extracted_features_train1.to_numpy()

            df_tsfeat_val = helper.to_df_tsfeatures(x_val)
            extracted_features_val = tsfeatures(df_tsfeat_val)
            extracted_features_val = extracted_features_val[extracted_features_train1.columns].replace(np.nan, 0.0)
            x_val = extracted_features_val.to_numpy()

            df_tsfeat_test = helper.to_df_tsfeatures(x_test)
            extracted_features_test = tsfeatures(df_tsfeat_test)
            extracted_features_test = extracted_features_test[extracted_features_train1.columns].replace(np.nan, 0.0)
            x_test = extracted_features_test.to_numpy()

        elif doTOOLS:
            extracted_features_train1 = helper.get_featuretools_features(x_train1)
            extracted_features_train1 = helper.postprocess_featuretools(extracted_features_train1, remove_constant_columns=False)
            x_train1 = extracted_features_train1.to_numpy()

            extracted_features_val = helper.get_featuretools_features(x_val)
            extracted_features_val = extracted_features_val[extracted_features_train1.columns]
            x_val = extracted_features_val.to_numpy()

            extracted_features_test = helper.get_featuretools_features(x_test)
            extracted_features_test = extracted_features_test[extracted_features_train1.columns]
            x_test = extracted_features_test.to_numpy()
            
        elif doTOOLS_SYM:
            extracted_features_train1 = helper.get_featuretools_features(x_train1)
            extracted_features_train1 = helper.postprocess_featuretools(extracted_features_train1, remove_constant_columns=True)
            x_train1 = extracted_features_train1.to_numpy()

            extracted_features_val = helper.get_featuretools_features(x_val)
            extracted_features_val = extracted_features_val[extracted_features_train1.columns]
            x_val = extracted_features_val.to_numpy()

            extracted_features_test = helper.get_featuretools_features(x_test)
            extracted_features_test = extracted_features_test[extracted_features_train1.columns]
            x_test = extracted_features_test.to_numpy()

        x_train1 = np.expand_dims(x_train1, axis=2)
        x_val = np.expand_dims(x_val, axis=2)
        x_test = np.expand_dims(x_test, axis=2) 

        # Create the directories if they don't exist
        os.makedirs(path, exist_ok=True)

        # Save the train-test-val split
        helper.save_obj(x_train1,   path + '/X_train',  saveMethod)
        helper.save_obj(x_val,      path + '/X_val',    saveMethod)
        helper.save_obj(x_test,     path + '/X_test',   saveMethod)
        helper.save_obj(y_train1,   path + '/y_train',  saveMethod)
        helper.save_obj(y_val,      path + '/y_val',    saveMethod)
        helper.save_obj(y_test,     path + '/y_test',   saveMethod)
        helper.save_obj(y_trainy1,  path + '/y_trainy', saveMethod)
        helper.save_obj(y_valy,     path + '/y_valy',   saveMethod)
        helper.save_obj(y_testy,    path + '/y_testy',  saveMethod)

        print(f"Saved features for fold {fold}")
    return True

def load_foldwise_features(number, dataName, method, fold, saveMethod='pickle'):
    """ Load the features foldwise """
    path = 'features' + '/' + str(number) + '_' + (dataName) + '/' + method + '/' + "fold_" + str(fold)

    x_train1 =  helper.load_obj(path + '/X_train',  saveMethod)
    x_val =     helper.load_obj(path + '/X_val',    saveMethod)
    x_test =    helper.load_obj(path + '/X_test',   saveMethod)
    y_train1 =  helper.load_obj(path + '/y_train',  saveMethod)
    y_val =     helper.load_obj(path + '/y_val',    saveMethod)
    y_test =    helper.load_obj(path + '/y_test',   saveMethod)
    y_trainy1 = helper.load_obj(path + '/y_trainy', saveMethod)
    y_valy =    helper.load_obj(path + '/y_valy',   saveMethod)
    y_testy =   helper.load_obj(path + '/y_testy',  saveMethod)

    return x_train1, x_val, x_test, y_train1, y_val, y_test, y_trainy1, y_valy, y_testy

def delete_features(number, dataName, method):
    """ Deletes the complete folder of a method """
    path = 'features' + '/' + str(number) + '_' + (dataName) + '/' + method

    shutil.rmtree(path, ignore_errors=True)
