import pandas as pd
import numpy as np
import yaml

from modules import modelCreator
from modules import helper

from pyts.datasets import ucr_dataset_list

import matplotlib.pyplot as plt

def get_all_methods(paths):
    """
    Gets the names of all methods
    
    Args:
        paths (list[str]): Path(s) to the yaml file(s)

    Returns:
        method_list (list): List of all methods
    """
    methods_list = []

    for path in paths:
        # Load the YAML file
        with open(path, "r") as file:
            config = yaml.safe_load(file)

        # Get YAML grid configurations
        grid_params = config.get("grid", {})

        methods_list += grid_params.get("data.method", {}).get("options", [])

    return methods_list


def get_configs(paths):
    """
    Gets the configurations specified in yaml files
    
    Args:
        paths (list[str]): Path(s) to the yaml file(s)

    Returns:
        configs (list): List of all configurations
    """

    configs = []

    for path in paths:
        # Load the YAML file
        with open(path, "r") as file:
            config = yaml.safe_load(file)

        # Get YAML grid configurations
        grid_params = config.get("grid", {})
        small_model_params = config.get("smallModel", {}).get("grid", {})
        big_model_params = config.get("bigModel", {}).get("grid", {})


        # Get dataset numbers
        numbers_min = grid_params.get("data.number", {}).get("min", [])
        numbers_max = grid_params.get("data.number", {}).get("max", [])
        numbers_list = [x for x in range(numbers_min, numbers_max)]
       
        # Get dataset names
        dataset_list = ucr_dataset_list()

        symbolCounts_list = grid_params.get("model.symbolCount", {}).get("options", [])
        numOfAttentionLayers_list = grid_params.get("model.numOfAttentionLayers", {}).get("options", [])
        methods_list = grid_params.get("data.method", {}).get("options", [])
        headers_list = grid_params.get("model.header", {}).get("options", [])
        ncoefs_list = grid_params.get("model.ncoef", {}).get("options", [])
        strategies_list = grid_params.get("model.strategy", {}).get("options", [])
        architecture_list = grid_params.get("model.architecture", {}).get("options", [])
        dropout_list = grid_params.get("model.dropOutRate", {}).get("options", [])
        small_dmodels_list = small_model_params.get("model.dmodel", {}).get("options", [])
        small_dffs_list = small_model_params.get("model.dff", {}).get("options", [])
        big_dmodels_list = big_model_params.get("model.dmodel", {}).get("options", [])
        big_dffs_list = big_model_params.get("model.dff", {}).get("options", [])

        if len(ncoefs_list) == 0:
            ncoefs_list.append(None)
            
        ncoefs_list_filtered = []
        for i in ncoefs_list:
            if i is not None:
                ncoefs_list_filtered.append(i[0])
            else:
                ncoefs_list_filtered.append(None)

        if not helper.is_symbolified(methods_list[0]):
            strategies_list = [None]

        factors_small = {
            'number':               numbers_list,
            'symbolCount':          symbolCounts_list,
            'numOfAttentionLayers': numOfAttentionLayers_list,
            'method':               methods_list,
            'header':               headers_list,
            'dmodel':               small_dmodels_list,
            'dff':                  small_dffs_list,
            'ncoef':                ncoefs_list_filtered,
            'strategy':             strategies_list,
            'architecture':         architecture_list,  
            'dropout':              dropout_list       
        }

        factors_big = {
            'number':               numbers_list,
            'symbolCount':          symbolCounts_list,
            'numOfAttentionLayers': numOfAttentionLayers_list,
            'method':               methods_list,
            'header':               headers_list,
            'dmodel':               big_dmodels_list,
            'dff':                  big_dffs_list,
            'ncoef':                ncoefs_list_filtered,
            'strategy':             strategies_list,
            'architecture':         architecture_list,  
            'dropout':              dropout_list      
        }


        configs += helper.combine_factors([factors_small])
        configs += helper.combine_factors([factors_big])
    
        # Add name of the dataset to each config
        for i, conf in enumerate(configs):
            conf['name'] = dataset_list[conf['number']]
    
    return configs

def get_config_name(config):
    """
    Returns the name of a configuration that was used for saving the results
    
    Args:
        config (dict): Dictionary of the configuration

    Returns:
        The name under which the results were saved in the experiment
    """
    if config['architecture'] == 'transformer':
        number                  = config['number']
        name                    = config['name']
        method                  = config['method']
        symbolCount             = config['symbolCount']
        ncoef                   = config['ncoef']
        strategy                = config['strategy']
       
        numOfAttentionLayers    = config['numOfAttentionLayers']
        header                  = config['header']
        dmodel                  = config['dmodel']
        dff                     = config['dff']
        dropout                 = config['dropout']

        return modelCreator.getWeightName(number, name, 0,
                                          architecture=architecture, abstractionType=method, symbols=symbolCount, 
                                          learning=False, results=True, 
                                          layers=numOfAttentionLayers, header=header, dmodel=dmodel, dff=dff, 
                                          num_coef=ncoef, strategy=strategy, dropout=dropout)
    
    elif config['architecture'] == 'resnet':
        number                  = config['number']
        name                    = config['name']
        method                  = config['method']
        symbolCount             = config['symbolCount']
        ncoef                   = config['ncoef']
        architecture            = config['architecture']
        strategy                = config['strategy']
        num_resblocks           = config['num_resblocks']
        num_channels            = config['num_channels']
        use_1x1conv             = config['use_1x1conv']

        return modelCreator.getWeightName(number, name, 0,
                                          architecture=architecture, abstractionType=method, symbols=symbolCount,  
                                          learning = False, results = True, 
                                          num_resblocks=num_resblocks, num_channels=num_channels, use_1x1conv=use_1x1conv,
                                          num_coef=ncoef, strategy=strategy)

def get_mean_per_config(config, metric="Test Accuracy"):
    """
    Gets the mean value of a metric of a configuration across the folds

    Args:
        config (DataFrame): The configuration
        metric (str): The metric of which the mean is calculated 

    Returns:
        The mean value of the metric of the specified configuration
    """
    conf_name = get_config_name(config)
    results = helper.load_obj(conf_name, 'pickle')

    folds = 0
    mean = 0

    if config['method'].split('_')[0] != 'SHAPE':
        for result in results[config['method']]['results']:
            folds += 1
            mean += result[metric]
    else:
        for result in results[config['method']][list(results[config['method']].keys())[0]]:
            folds += 1
            mean += result[metric]

    return np.round(mean / folds, 4)

def find_methods_per_dataset(configs_df):
    """
    Finds the methods that finished for each dataset

    Args:
        configs_df (DataFrame): DataFrame with all configurations

    Returns:
        dic (Dictionary): Dictionary with datasets as keys and finished methods as value
    """
    dic = {}
    for index, config in configs_df.iterrows():
        try: 
            conf_name = get_config_name(config)
            results = helper.load_obj(conf_name, 'pickle')
            # load successful
            ds_name = config['name']
            method = config['method']
            if ds_name not in dic.keys():
                dic[ds_name] = []
            if method not in dic[ds_name]:
                dic[ds_name].append(method)
            
        except FileNotFoundError as e:
            # load not successful
            continue

    return dic

def find_datasets_each_methods_finished(configs_df, paths, ignore=[]):
    """
    Finds the dataset where each method finished at least once

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        methods (list): List with all methods

    Returns:
        finished_datasets (list): List with the finished datasets
    """
    datasets = find_methods_per_dataset(configs_df)
    methods = get_all_methods(paths)

    finished_datasets = []

    for ds in datasets.keys():
        finished_datasets.append(ds)

        for m in methods:
            if m in ignore:
                continue
            if m not in datasets[ds]:
                finished_datasets.remove(ds)
                break

    return finished_datasets

def add_results_to_df(configs_df, metric="Test Accuracy"):
    """
    Adds the results (of metric) to the configurations in the DataFrame

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to add

    Returns: 
        configs_df (DataFrame): The DataFrame with results 
    """
    configs_df[metric] = None
    for index, config in configs_df.iterrows():
        try: 
            mean = get_mean_per_config(config, metric)
            mean = np.round(mean, 4)
            configs_df.at[index, metric] = mean
        except FileNotFoundError:
            continue
    return configs_df

def get_best_per_dataset_method(configs_df, metric="Test Accuracy"):
    """
    Finds the best configuration per method on each dataset

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to add

    Returns: 
        best_results (DataFrame): The DataFrame with the best results per dataset and method
    """
    configs_results_df = add_results_to_df(configs_df, metric)
    configs_results_df = configs_results_df[configs_results_df[metric].notnull()] # filter out Nones

    best_results = configs_results_df.loc[configs_results_df.groupby(['name', 'method'])[metric].idxmax()]
    best_results[metric] = best_results[metric].astype(float).round(4)
    
    return best_results

def plot_dataframe(df, name, pdf=False):
    """
    Creates the plot of a DataFrame

    Args
        df (DataFrame): The DataFrame
        name (str): Name of the stored plot (Will be stored in plots/<name>.png)
    """
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.axis('tight')
    ax.axis('off')

    # Create a table from the DataFrame
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the figure
    if pdf:
        suffix = ".pdf"
    else:
        suffix = ".png"
    plt.savefig("plots/" + name + suffix, dpi=300, bbox_inches='tight')

def get_average_per_method_foldwise(configs_df, metric="Test Accuracy"):
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    best_results_foldwise = get_foldwise_results(best_results_ds_met)
    average_results = best_results_foldwise.groupby('method')[metric].mean().reset_index()
    stds = best_results_foldwise.groupby('method')[metric].std().reset_index()
    average_results[metric] = average_results[metric].astype(float).round(4)
    average_results['std'] = stds[metric].astype(float).round(4)

    return average_results

def filter_results_without_baseline(configs_df, baseline="ORI"):
    """
    Filters out the results for datasets, where the baseline did not finish

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baseline (str): The method that is considered the baseline

    Returns:
        filtered_df (DataFrame): The filtered dataset
    """
    datasets_with_baseline = configs_df[configs_df['method'] == baseline]['name'].unique()
    filtered_df = configs_df[configs_df['name'].isin(datasets_with_baseline)]

    return filtered_df

def add_inc_dec_to_baseline_best_foldwise(configs_df, baseline="ORI", metric="Test Accuracy", k=5):
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    best_results_folds = get_foldwise_results(best_results_ds_met)
    
    inc_dec = []
    for index, config in best_results_folds.iterrows():
        conf = config.copy()

        # if config['method'] == baseline:
        #     continue

        acc_conf = config[metric]
        acc_base = best_results_folds[(best_results_folds['name'] == config['name']) & (best_results_folds['method'] == baseline) & (best_results_folds['fold'] == config['fold'])][metric].iloc[0]

        diff = acc_conf - acc_base
        if diff >= 0:
            conf['inc'] = diff
        else:
            conf['dec'] = diff

        inc_dec.append(conf)
    inc_dec_df = pd.DataFrame(inc_dec)
    inc_dec_df['inc'] = inc_dec_df['inc'].astype(float).round(4)
    inc_dec_df['dec'] = inc_dec_df['dec'].astype(float).round(4)
    
    return inc_dec_df

def get_foldwise_results(configs_df, metric="Test Accuracy", k=5):
    results_ls = []

    for index, config in configs_df.iterrows():

        conf_name = get_config_name(config)
        results = helper.load_obj(conf_name)

        for fold in range(k):
            conf = config.copy()
            conf['fold'] = fold
            conf[metric] = results[config['method']]['results'][fold][metric]
            results_ls.append(conf)
            
    results_df = pd.DataFrame(results_ls)
    return results_df

def get_avg_inc_dec_to_baseline_best_foldwise(configs_df, baseline="ORI", metric="Test Accuracy"):
    inc_dec_df = add_inc_dec_to_baseline_best_foldwise(configs_df, baseline, metric)
    
    avg_inc_dec = inc_dec_df.groupby('method')[['inc', 'dec']].mean().reset_index()
    avg_inc_dec['inc'] = avg_inc_dec['inc'].astype(float).round(4)
    avg_inc_dec['dec'] = avg_inc_dec['dec'].astype(float).round(4)
    std_per_method = avg_inc_dec.set_index('method')[['inc', 'dec']].stack().groupby(level=0).std().to_frame('std')
    merged_df = pd.merge(avg_inc_dec, std_per_method, on='method', how='inner').replace(np.nan, 0.0)
    
    return merged_df

def add_predictions(configs_df, metric="Test Predictions"):
    """
    Adds the predictions as column to the DataFrame

     Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to add

    Returns
        configs_df (DataFrame): DataFrame with the predictions of each best method on every dataset
    """
    configs_df[metric] = None
    for index, config in configs_df.iterrows():
        try: 
            conf_name = get_config_name(config)
            results = helper.load_obj(conf_name, 'pickle')
            
            predictions = [result[metric] for result in results[config['method']]['results']]
            configs_df.at[index, metric] = predictions
            
        except FileNotFoundError:
            # Load not successful
            continue

    return configs_df

def get_best_per_dataset(configs_df, metric="Test Accuracy"):
    """
    Finds the best configuration per dataset

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to add

    Returns: 
        best_results (DataFrame): The DataFrame with the best results per dataset
    """
    configs_results_df = add_results_to_df(configs_df, metric)
    configs_results_df = configs_results_df[configs_results_df[metric].notnull()] # filter out Nones

    best_results = configs_results_df.loc[configs_results_df.groupby(['name'])[metric].idxmax()]
    best_results[metric] = best_results[metric].astype(float).round(4)
    
    return best_results


def get_flags_better_baseline(configs_df, baseline='ORI', metric='Test Accuracy'):
    """
    Counts how many times a method is better than the baseline
    Gets the best configuration per method and per dataset and adds a flag if it is better than the baseline

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baseline (str): The method that is considered the baseline

    Returns:
        better_df (DataFrame): DataFrame with the parameters and the flags
    """
    results = []
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric=metric)

    group = best_results_ds_met.groupby('name')

    for dataset, group in group:
        for index, config in group.iterrows():
            if config['method'] == baseline:
                continue

            result_dic = {}
            result_dic['name'] = dataset
            result_dic['method'] = config['method']
            result_dic['symbolCount'] = config['symbolCount']
            result_dic['strategy'] = config['strategy']
            result_dic['architecture'] = config['architecture']

            if group[group['method'] == baseline][metric].empty:
                continue
            
            baseline_res = group[(group['method'] == baseline)][metric].iloc[0]
            
            if config[metric] > baseline_res:
                result_dic[f'better_{baseline}'] = 1
            else:
                result_dic[f'better_{baseline}'] = 0
            
            results.append(result_dic)
    
    better_df = pd.DataFrame(results)
    
    return better_df

def get_flags_better_baseline_foldwise(configs_df, baseline='ORI', metric='Test Accuracy'):
    """
    Counts how many times a method is better than the baseline
    Gets the best configuration per method and per dataset and adds a flag if it is better than the baseline

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baseline (str): The method that is considered the baseline

    Returns:
        better_df (DataFrame): DataFrame with the parameters and the flags
    """
    results = []
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric=metric)
    best_results_folds = get_foldwise_results(best_results_ds_met)

    group = best_results_folds.groupby('name')

    for dataset, group in group:
        for index, config in group.iterrows():
            
            if config['method'] == baseline:
                continue

            result_dic = {}
            result_dic['name'] = dataset
            result_dic['method'] = config['method']
            result_dic['symbolCount'] = config['symbolCount']
            result_dic['strategy'] = config['strategy']
            result_dic['architecture'] = config['architecture']
            result_dic['fold'] = config['fold']

            if group[group['method'] == baseline][metric].empty:
                continue
            
            baseline_res = group[(group['method'] == baseline) & (group['fold'] == config['fold'])][metric].iloc[0]
            
            if config[metric] > baseline_res:
                result_dic[f'better_{baseline}'] = 1
            else:
                result_dic[f'better_{baseline}'] = 0
            
            results.append(result_dic)
    
    better_df = pd.DataFrame(results)
   
    return better_df

def get_flags_method_best(configs_df, metric='Test Accuracy'):
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    # Find the max accuracy per dataset
    best_results_ds_met['max_accuracy'] = best_results_ds_met.groupby('name')[metric].transform('max')

    # First new column: 1 if this method achieved the highest accuracy for the dataset
    best_results_ds_met['count'] = (best_results_ds_met[metric] == best_results_ds_met['max_accuracy']).astype(int)

    # Count how many methods achieved the max accuracy
    best_results_ds_met['count_max'] = best_results_ds_met.groupby('name')[metric].transform(lambda x: (x == x.max()).sum())

    # Second new column: 1 if this method is the *only* one with the highest accuracy
    best_results_ds_met['only_best_count'] = ((best_results_ds_met['count'] == 1) & (best_results_ds_met['count_max'] == 1)).astype(int)

    # Drop helper columns
    best_results_ds_met.drop(columns=['max_accuracy', 'count_max'], inplace=True)

    return best_results_ds_met
    
def get_flags_method_best_foldwise(configs_df, metric='Test Accuracy'):
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    best_results_folds = get_foldwise_results(best_results_ds_met)

    # Find the max accuracy per dataset
    best_results_folds['max_accuracy'] = best_results_folds.groupby(['name', 'fold'])[metric].transform('max')

    # First new column: 1 if this method achieved the highest accuracy for the dataset
    best_results_folds['count'] = (best_results_folds[metric] == best_results_folds['max_accuracy']).astype(int)

    # Count how many methods achieved the max accuracy
    best_results_folds['count_max'] = best_results_folds.groupby(['name', 'fold'])[metric].transform(lambda x: (x == x.max()).sum())

    # Second new column: 1 if this method is the *only* one with the highest accuracy
    best_results_folds['only_best_count'] = ((best_results_folds['count'] == 1) & (best_results_folds['count_max'] == 1)).astype(int)

    # Drop helper columns
    best_results_folds.drop(columns=['max_accuracy', 'count_max'], inplace=True)
    
    return best_results_folds

def filter_tsfresh_sel(configs_df):
    """
    Filters out the TSFRESH_SEL configurations

    Args:
        configs_df (DataFrame): DataFrame with all configurations

    Returns:
        filtered_df (DataFrame): The filtered DataFrame
    """
    filtered_df = configs_df[(configs_df['method'] != 'TSFRESH_SEL') & (configs_df['method'] != 'TSFRESH_SEL_SYM')]
    return filtered_df
    dataset_df = pd.read_csv("DataSummary.csv")
    types = ['All']
    types += list(np.unique(dataset_df['Type']))

    results_df = pd.DataFrame()

    for type in types:
        if type == 'All':
            best_df = get_averages_type(configs_df, type, metric=metric)
        else:
            type_df = dataset_df[dataset_df['Type'] == type]
            names_for_type = type_df[type_df['Type'] == type]['Name'].tolist()
            configs_df_type = configs_df[configs_df['name'].isin(names_for_type)]
            best_df = get_averages_type(configs_df_type, type, metric=metric)

        if results_df.empty:
            results_df = best_df
        else:
            results_df = pd.merge(results_df, best_df, on='method', how='inner')
    
    return results_df

def get_average_best_foldwise(configs_df, metric='Test Accuracy'):
    """
    Gets the average of the best results

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        avg_best (float): The average of the best results
    """
    best_results_ds_met = get_best_per_dataset(configs_df, metric)
    best_results_folds = get_foldwise_results(best_results_ds_met)

    avg_best = best_results_folds[metric].mean().astype(float).round(4)
    std_best = best_results_folds[metric].std().astype(float).round(4)

    if std_best is np.nan:
        std_best = 0.0

    return avg_best, std_best
    
def get_average_best_raw_foldwise(configs_df, metric='Test Accuracy', raw_methods=['ORI', 'TSFRESH', 'TSFRESH_SEL', 'TSFEL', 'CATCH22', 'CATCH24', 'TSFEAT', 'TOOLS']):
    """
    Gets the average of the best results of the non-symbolified methods

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        avg_best (float): The average of the best results
    """
    raw_configs_df = configs_df[configs_df['method'].isin(raw_methods)]
    best_results_ds_met = get_best_per_dataset(raw_configs_df, metric)
    best_results_folds = get_foldwise_results(best_results_ds_met)

    avg_best = best_results_folds[metric].mean().astype(float).round(4)
    std_best = best_results_folds[metric].std().astype(float).round(4)

    if std_best is np.nan:
        std_best = 0.0

    return avg_best, std_best

def get_average_best_sym_foldwise(configs_df, metric='Test Accuracy', sym_methods=['MCB', 'SAX', 'SFA', 'TSFRESH_SYM', 'TSFRESH_SEL_SYM', 'TSFEL_SYM', 'CATCH22_SYM', 'CATCH24_SYM', 'TSFEAT_SYM', 'TOOLS_SYM'], strategy=None):
    """
    Gets the average of the best results of the symbolified methods

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        avg_best (float): The average of the best results
    """
    sym_configs_df = configs_df[configs_df['method'].isin(sym_methods)]
    if strategy == 'uniform':
        sym_configs_df = sym_configs_df[sym_configs_df['strategy'] == 'uniform']
    elif strategy == 'quantile':
        sym_configs_df = sym_configs_df[sym_configs_df['strategy'] == 'quantile']

    best_results_ds_met = get_best_per_dataset(sym_configs_df, metric)
    best_results_folds = get_foldwise_results(best_results_ds_met)

    avg_best = best_results_folds[metric].mean().astype(float).round(4)
    std_best = best_results_folds[metric].std().astype(float).round(4)

    if std_best is np.nan:
        std_best = 0.0

    return avg_best, std_best

def get_averages_type_foldwise(configs_df_type, type, metric='Test Accuracy'):
    """
    Gets the average of the best results of the non-symbolified and symbolified methods

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        avg_best (float): The average of the best results
    """
    avg_best, std_best = get_average_best_foldwise(configs_df_type, metric)
    avg_best_ori, std_best_ori = get_average_best_raw_foldwise(configs_df_type, metric, raw_methods=['ORI'])
    avg_best_raw, std_best_raw = get_average_best_raw_foldwise(configs_df_type, metric)
    avg_best_raw_wo_ori, std_best_raw_wo_ori = get_average_best_raw_foldwise(configs_df_type, metric, raw_methods=['TSFRESH', 'TSFRESH_SEL', 'TSFEL', 'CATCH22', 'CATCH24', 'TSFEAT', 'TOOLS'])
    avg_best_sym, std_best_sym = get_average_best_sym_foldwise(configs_df_type, metric)
    avg_best_solo_sym, std_best_solo_sym = get_average_best_sym_foldwise(configs_df_type, metric, sym_methods=['MCB', 'SAX', 'SFA'])
    avg_best_feature_sym, std_best_feature_sym = get_average_best_sym_foldwise(configs_df_type, metric, sym_methods=['TSFRESH_SYM', 'TSFRESH_SEL_SYM', 'TSFEL_SYM', 'CATCH22_SYM', 'CATCH24_SYM', 'TSFEAT_SYM', 'TOOLS_SYM'])

    best_df = pd.DataFrame([{'method': 'All', f'{type}': f'{avg_best} +- {std_best}'}, 
                            {'method': 'Ori', f'{type}': f'{avg_best_ori} +- {std_best_ori}'},
                            {'method': 'Non-Symbolic', f'{type}': f'{avg_best_raw} +- {std_best_raw}'}, 
                            {'method': 'Feature Extractors', f'{type}': f'{avg_best_raw_wo_ori} +- {std_best_raw_wo_ori}'},
                            {'method': 'Symbolic', f'{type}': f'{avg_best_sym} +- {std_best_sym}'},
                            {'method': 'Solo Symbolic', f'{type}': f'{avg_best_solo_sym} +- {std_best_solo_sym}'},
                            {'method': 'Feature Symbolic', f'{type}': f'{avg_best_feature_sym} +- {std_best_feature_sym}'}])
    
    return best_df

def get_averages_per_type_foldwise(configs_df, metric='Test Accuracy', as_csv=False, csv_save_name="plots/averages_per_type.csv"):
    dataset_df = pd.read_csv("DataSummary.csv")
    types = ['All']
    types += list(np.unique(dataset_df['Type']))

    results_df = pd.DataFrame()

    for type in types:
        if type == 'All':
            best_df = get_averages_type_foldwise(configs_df, type, metric=metric)
        else:
            type_df = dataset_df[dataset_df['Type'] == type]
            names_for_type = type_df[type_df['Type'] == type]['Name'].tolist()
            configs_df_type = configs_df[configs_df['name'].isin(names_for_type)]
            if configs_df_type.empty:
                continue
            best_df = get_averages_type_foldwise(configs_df_type, type, metric=metric)

        if results_df.empty:
            results_df = best_df
        else:
            results_df = pd.merge(results_df, best_df, on='method', how='inner')

    results_df = results_df.set_index('method')

    if as_csv:
        results_df.to_csv(csv_save_name)
    
    return results_df
    
def get_datasets_per_type(configs_df, configs_df_sel):
    dataset_df = pd.read_csv("DataSummary.csv")
    types = ['All']
    types += list(np.unique(dataset_df['Type']))

    dic = {}
    dic_wsel = {}
    dic_wosel = {}

    finished_wsel = list(np.unique(configs_df['name']))
    finished_wosel = list(np.unique(configs_df_sel['name']))

    for type in types:
        dic[type] = 0
        dic_wsel[type] = 0
        dic_wosel[type] = 0

    
    for type in types:
        if type == 'All':
            dic[type] = np.unique(dataset_df['Name']).shape[0]
            dic_wsel[type] = np.unique(configs_df['name']).shape[0]
            dic_wosel[type] = np.unique(configs_df_sel['name']).shape[0]
        else:
            type_df = dataset_df[dataset_df['Type'] == type]
            names_for_type = type_df[type_df['Type'] == type]['Name'].tolist()

            dic[type] = len(names_for_type)

            for name in names_for_type:
                if name in finished_wsel:
                    dic_wsel[type] += 1
                if name in finished_wosel:
                    dic_wosel[type] += 1
    
    res_dic = {}
    res_dic['Total'] = dic
    res_dic['With Sel'] = dic_wsel
    res_dic['Without Sel'] = dic_wosel

    res_df = pd.DataFrame().from_dict(res_dic)

    res_df.to_csv("plots/datasets_per_type.csv")

def get_configs_resnet(paths):
    """
    Gets the configurations specified in yaml files for the resnet experiments
    
    Args:
        paths (list[str]): Path(s) to the yaml file(s)

    Returns:
        configs (list): List of all configurations
    """

    configs = []

    for path in paths:
        # Load the YAML file
        with open(path, "r") as file:
            config = yaml.safe_load(file)

        # Get YAML grid configurations
        grid_params = config.get("grid", {})

        # Get dataset numbers
        numbers_min = grid_params.get("data.number", {}).get("min", [])
        numbers_max = grid_params.get("data.number", {}).get("max", [])
        numbers_list = [x for x in range(numbers_min, numbers_max)]

        # Get dataset names
        dataset_list = ucr_dataset_list()

        symbolCounts_list = grid_params.get("model.symbolCount", {}).get("options", [])
        methods_list = grid_params.get("data.method", {}).get("options", [])
        num_resblocks_list = grid_params.get("model.num_resblocks", {}).get("options", [])
        num_channels_list = grid_params.get("model.num_channels", {}).get("options", [])
        use_1x1conv_list = grid_params.get("model.use_1x1conv", {}).get("options", [])
        lr_list = grid_params.get("model.use_1x1conv", {}).get("options", [])
        ncoefs_list = grid_params.get("model.ncoef", {}).get("options", [])
        strategies_list = grid_params.get("model.strategy", {}).get("options", [])
        architectures_list = grid_params.get("model.architecture", {}).get("options", [])
        

        if len(ncoefs_list) == 0:
            ncoefs_list.append(None)
            
        ncoefs_list_filtered = []
        for i in ncoefs_list:
            if i is not None:
                ncoefs_list_filtered.append(i[0])
            else:
                ncoefs_list_filtered.append(None)

        factors = {
            'number':               numbers_list,
            'method':               methods_list,
            'architecture':         architectures_list,
            'symbolCount':          symbolCounts_list,
            'strategy':             strategies_list,
            'num_resblocks':        num_resblocks_list,
            'num_channels':         num_channels_list,
            'use_1x1conv':          use_1x1conv_list,
            'lr':                   lr_list,
            'ncoef':                ncoefs_list_filtered
        }

        configs += helper.combine_factors([factors])

        # Add name of the dataset to each config
        for i, conf in enumerate(configs):
            conf['name'] = dataset_list[conf['number']]
    
    return configs
