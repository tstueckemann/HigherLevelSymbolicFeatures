import pandas as pd
import numpy as np
import pickle
import yaml
import os

from modules import modelCreator
from modules import helper

from pyts.datasets import ucr_dataset_list

import matplotlib.pyplot as plt

from pathlib import Path


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
    if ('architecture' not in config.keys()) or (config['architecture'] == None) or (config['architecture'] == 'transformer'):
        number                  = config['number']
        name                    = config['name']
        method                  = config['method']
        symbolCount             = config['symbolCount']
        ncoef                   = config['ncoef']
        strategy                = config['strategy']

        if method.split("_")[0] == 'SHAPE':
            # is a shapelet
            minShapeletLength   = config['minShapeletLength']
            shapeletTimeLimit   = config['shapeletTimeLimit']    

            return modelCreator.getWeightName(number, name, 0, symbolCount, abstractionType=method, learning=False, results=True, posthoc=ncoef, doShapelets=True, minShapelets=minShapeletLength, timeLimit=shapeletTimeLimit, strategy=strategy)

        else:
            numOfAttentionLayers    = config['numOfAttentionLayers']
            header                  = config['header']
            dmodel                  = config['dmodel']
            dff                     = config['dff']
            dropout                 = config['dropout']

            return modelCreator.getWeightName(number, name, 0, symbolCount, numOfAttentionLayers, method, header, dmodel, dff, learning=False, results=True, doDetails=True, posthoc=ncoef, strategy=strategy, dropout=dropout)
    
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
        lr                      = config['lr']

        return modelCreator.getWeightName(number, name, 0, symbolCount, abstractionType=method, doDetails=True, learning = False, results = True, 
                                          posthoc=ncoef, strategy=strategy, architecture=architecture, num_resblocks=num_resblocks, num_channels=num_channels, 
                                          use_1x1conv=use_1x1conv)




def get_results_per_config(configs):
    """
    Returns a list of dictionaries which contain each configuration with its results
    
    Args:
        configs (list): List of configurations

    Returns:
        results (list): List of dictionaries with configuration and their results
    """

    results = []

    for config in configs:
        conf_name = get_config_name(config)
        result = {}
        result['config'] = config

        data = helper.load_obj(conf_name, 'pickle')
        result['results'] = data[config['method']]['results']
        
        results.append(result)

    return results


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

def get_missing_configs(configs_df):
    """
    Finds all the missing configurations of the experiments

    Args:
        configs_df (DataFrame): DataFrame with all configurations

    Returns:
        None
    """
    notFound = 0

    for index, config in configs_df.iterrows():
        conf_name = get_config_name(config)
        try: 
            results = helper.load_obj(conf_name, 'pickle')
        except FileNotFoundError as e:
            print(f"Config {conf_name} idx {index} was not found!")
            notFound += 1
    print(f"{configs_df.shape[0]} configs registered.")
    print(f"{notFound} configs were not found!")
    directory_path = "results/"
    file_count = len([name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

    print(f"Number of files in '{directory_path}': {file_count}")

def get_missing_datasets(configs_df):
    """
    Finds all the datasets that do not appear in the results, i.e. they did not finish once

    Args:
        configs_df (DataFrame): DataFrame with all configurations

    Returns:
        missing_datasets (list): List with all missing datasets
    """
    found_datasets = []
    missing_datasets = []
    for index, config in configs_df.iterrows():
        try: 
            conf_name = get_config_name(config)
            name = config['name']
            results = helper.load_obj(conf_name, 'pickle')
            # load successful
            if name not in found_datasets:
                found_datasets.append(name)
        except FileNotFoundError as e:
            # load not successful
            if name not in missing_datasets and name not in found_datasets:
                missing_datasets.append(name)

    for ds in found_datasets:
        if ds in missing_datasets:
            missing_datasets.remove(ds)

    return missing_datasets

def get_found_datasets(configs_df):
    """
    Finds all the datasets that appear in the results, i.e. they did at least finish once

    Args:
        configs_df (DataFrame): DataFrame with all configurations

    Returns:
        missing_datasets (list): List with all found datasets
    """
    found_datasets = []
    for index, config in configs_df.iterrows():
        try: 
            conf_name = get_config_name(config)
            name = config['name']
            results = helper.load_obj(conf_name, 'pickle')
            # load successful
            if name not in found_datasets:
                found_datasets.append(name)
        except FileNotFoundError as e:
            # load not successful
            continue

    return found_datasets

def count_results_per_parameter(configs_df, param):
    """
    Counts how many results exist per different value of a parameter

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        param (str): The parameter

    Returns:
        dic (Dictionary): Dictionary with datasets as keys and count as value 
    """
    dic = {}
    for index, config in configs_df.iterrows():
        try: 
            conf_name = get_config_name(config)
            results = helper.load_obj(conf_name, 'pickle')
            # load successful
            if config[param] not in dic.keys():
                dic[config[param]] = 0
            dic[config[param]] += 1 
        except FileNotFoundError as e:
            # load not successful
            continue

    return dic

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

def get_dataset_method_overview(configs_df, paths):
    """
    Creates an overview of which method finished at least once on each dataset

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        paths (list): List of the configuration yaml files
    """
    data = find_methods_per_dataset(configs_df)
    all_methods = get_all_methods(paths)

    dataset_list = ucr_dataset_list()

    dataset_names_with_numbers = []

    for dataset in dataset_list:
        if dataset in data:
            number = dataset_list.index(dataset)
            name = str(number) + '_' + dataset
            dataset_names_with_numbers.append(name)

    overview_df = pd.DataFrame(index=dataset_names_with_numbers, columns=all_methods, data="")

    for dataset, methods in data.items():
        for method in methods:
            number = dataset_list.index(dataset)
            name = str(number) + '_' + dataset
            overview_df.loc[name, method] = "✔"

    return overview_df

def plot_overview(overview_df, name='overview'):
    """
    Saves the overview as plot in plots/

    Args:
        overview_df (DataFrame): DataFrame with the overview
    """
    # Save the DataFrame as an image
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.axis('tight')
    ax.axis('off')

    # Create a table from the DataFrame
    table = ax.table(cellText=overview_df.values, colLabels=overview_df.columns, rowLabels=overview_df.index, loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(overview_df.columns))))

    # Save the figure
    plt.savefig(f"plots/{name}.png", dpi=300, bbox_inches='tight')

def get_datasets_with_n_methods(configs_df, n):
    """
    Finds all dataset where a minimum of n different methods finished at least once

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        n (int): Number of different methods
    """
    data = find_methods_per_dataset(configs_df)
    datasets = []
    for ds in data.keys():
        if len(data[ds]) >= n:
            datasets.append(ds)

    return datasets

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

def get_average_per_method(configs_df, metric="Test Accuracy"):
    """
    Gets the average result per method by taking the best configuration per dataset

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to add

    Returns: 
        avg_results (DataFrame): The DataFrame with the average results per method
    """
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    average_results = best_results_ds_met.groupby('method')[metric].mean().reset_index()
    stds = best_results_ds_met.groupby('method')[metric].std().reset_index()
    average_results[metric] = average_results[metric].astype(float).round(4)
    average_results['std'] = stds[metric].astype(float).round(4)

    return average_results

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

def add_inc_dec_to_baseline_best(configs_df, baseline="ORI", metric="Test Accuracy"):
    """
    Adds the Increase or Decrease regarding the metric to the baseline to each best configuration of a method in every dataset

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baseline (str): The method that is considered the baseline
        metric (str): The metric to calculate in-/decrease on
    
    Returns:
        filtered_df (DataFrame): The filtered Dataset with 
    """
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    filtered_df = filter_results_without_baseline(best_results_ds_met, baseline)

    baseline_lookup = filtered_df[filtered_df['method'] == baseline].set_index('name')[metric]
    diff = filtered_df[metric] - filtered_df['name'].map(baseline_lookup)
    
    filtered_df['inc'] = None
    filtered_df['dec'] = None

    filtered_df.loc[diff > 0, 'inc'] = diff[diff > 0]
    filtered_df.loc[diff < 0, 'dec'] = diff[diff < 0]

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


        

def get_avg_inc_dec_to_baseline_best(configs_df, baseline="ORI", metric="Test Accuracy"):
    """
    Gets the average increase or decrease of a method to the baseline

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baseline (str): The method that is considered the baseline
        metric (str): The metric to calculate in-/decrease on
    
    Returns:
        avg_inc_dec (DataFrame): The DataFrame with the average increase and decrease
    """
    inc_dec_df = add_inc_dec_to_baseline_best(configs_df, baseline, metric)
    avg_inc_dec = inc_dec_df.groupby('method')[['inc', 'dec']].mean().reset_index()
    avg_inc_dec['inc'] = avg_inc_dec['inc'].astype(float).round(4)
    avg_inc_dec['dec'] = avg_inc_dec['dec'].astype(float).round(4)
    std_inc_dec = inc_dec_df.groupby('method')[metric].std().reset_index()
    std_inc_dec['std'] = std_inc_dec[metric].astype(float).round(4)
    # std_inc_dec['dec_std'] = std_inc_dec['dec'].astype(float).round(4)
    std_inc_dec = std_inc_dec.drop(columns=['Test Accuracy'])
    merged_df = pd.merge(avg_inc_dec, std_inc_dec, on='method', how='inner')

    return merged_df

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
            # Load successful
            if config['method'].split('_')[0] == 'SHAPE':
                # is Shapelet
                slen = float(config['minShapeletLength'])
                if slen >= 1: slen = int(slen) 
                predictions = [result[metric] for result in results[config['method']][f'slen: {slen}']]
            else:
                predictions = [result[metric] for result in results[config['method']]['results']]
            configs_df.at[index, metric] = predictions
            
        except FileNotFoundError:
            # Load not successful
            continue

    return configs_df

def calc_fidelity(a, b):
    """
    Calculates the fidelity score of two arrays

    Args:
        a (list): The first array
        b (list): The second array

    Returns
        score (float): The fidelity score of a and b
    """
    if len(a) != len(b):
        print("Predictions do not have the same length!")
        return None
    score = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            score += 1

    score /= len(a)

    return score

def get_fidelities_best(configs_df, metric="Test Predictions"):
    """
    Adds the average fidelity of the best configurations of each method on every dataset as column

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns
        results_df (DataFrame): DataFrame with the fidelities
    """
    results = []
    conf_pred = add_predictions(configs_df, metric)
    grouped = conf_pred.groupby('name')
    
    for dataset, group in grouped:
        methods = group['method'].unique()
        for method_a in methods:
            predictions_a = group[group['method'] == method_a][metric].iloc[0]

            if not predictions_a:
                continue

            fidelities_to_other_methods = []
            
            for method_b in methods:
                if method_a == method_b:
                    continue 
                
                predictions_b = group[group['method'] == method_b][metric].iloc[0]

                if not predictions_b:
                    continue
                
                fold_fidelities = [
                    calc_fidelity(predictions_a[fold], predictions_b[fold])
                    for fold in range(len(predictions_a))
                ]
                
                avg_fidelity_fold = sum(fold_fidelities) / len(fold_fidelities)
                fidelities_to_other_methods.append(avg_fidelity_fold)
            
            avg_fidelity_to_others = (
                sum(fidelities_to_other_methods) / len(fidelities_to_other_methods)
                if fidelities_to_other_methods
                else 0
            )
            
            results.append({
                'name': dataset,
                'method': method_a,
                'average_fidelity': avg_fidelity_to_others
            })
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def add_fidelities_baselines_best(configs_df, baselines=["ORI", "SAX", "SFA"], metric="Test Predictions"):
    """
    Gets the (foldwise-) average fidelity to each baseline method considering the best cofiguration for each method per dataset.
    For each dataset, the best configuration per method is found. 
    Then the fidelities between each best method and the (best) baselines is added to a new column (for each baseline).

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baselines (List): List with the methods that are considered the baselines
        metric (str): The metric to evaluate on

    Returns
        results_df (DataFrame): DataFrame with the fidelities of each best configuration to the baselines
    """
    results = []
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    conf_pred = add_predictions(best_results_ds_met, metric)

    grouped = conf_pred.groupby('name')

    for dataset, group in grouped:
        methods = group['method'].unique()

        for method in methods:
            predictions = group[group['method'] == method][metric].iloc[0]
            
            if not predictions:
                continue

            result_dic = {}
            result_dic['name'] = dataset
            result_dic['method'] = method

            for method_base in baselines:
                if method == method_base:
                    continue

                if group[group['method'] == method_base][metric].empty:
                    continue

                predictions_base = group[group['method'] == method_base][metric].iloc[0]

                if not predictions_base:
                    continue

                fold_fidelities = [
                    calc_fidelity(predictions[fold], predictions_base[fold])
                    for fold in range(len(predictions))
                ]

                # Get average fidelity over each fold
                avg_fold_fidelity = np.nanmean(fold_fidelities)
                result_dic[f'fidelity_{method_base}'] = avg_fold_fidelity

            results.append(result_dic)
    results_df = pd.DataFrame(results)
    return results_df


def get_average_fidelities_per_method_best(configs_df, baselines=["ORI", "SAX", "SFA"], metric="Test Predictions", params=['name', 'dmodel', 'dff']):
    """
    Gets the average of the fidelity per method for the best configuration to the baseline

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baselines (List): List with the methods that are considered the baselines
        metric (str): The metric to evaluate on
        params (list): The parameters that are considered to find the same configuration
        
    Returns
        results_df (DataFrame): DataFrame with the fidelities of each same configuration to the baselines
    """
    fidelities_df = add_fidelities_baselines_best(configs_df, baselines, metric)

    results_avg = []
    grouped_avg = fidelities_df.groupby(['method'])

    for config, group in grouped_avg:
        
        result_dic_avg = {}
        result_dic_avg['method'] = group['method'].iloc[0]

        for method_base in baselines:
            if group[f'fidelity_{method_base}'].empty:
                continue

            fidel_avg = np.nanmean(group[f'fidelity_{method_base}'])
            result_dic_avg[f'avg_fidelity_{method_base}'] = fidel_avg

        results_avg.append(result_dic_avg)

    results_df = pd.DataFrame(results_avg)
    return results_df



def add_fidelities_baselines_same(configs_df, baselines=["ORI", "SAX", "SFA"], metric="Test Predictions", params=['name', 'dmodel', 'dff']):
    """
    Gets the (foldwise-) average fidelity to each baseline method considering the same configuration for each method per dataset.
    For each configuration, the fidelity (averaged over the folds) to each baseline method with the same configuration is added to a new column.

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baselines (List): List with the methods that are considered the baselines
        metric (str): The metric to evaluate on
        params (list): The parameters that are considered to find the same configuration
        
    Returns
        results_df (DataFrame): DataFrame with the fidelities of each same configuration to the baselines
    """
    results = []
    conf_pred = add_predictions(configs_df, metric)

    grouped = conf_pred.groupby(params)

    for config, group in grouped:

        symbols = group['symbolCount'].unique()
        ncoefs = group['ncoef'].unique()
        
        for index, config in group.iterrows():
            predictions = config[metric]

            if not predictions:
                continue

            method = config['method']

            result_dic = {}
            result_dic['name']          = config['name']
            result_dic['method']        = config['method']
            result_dic['dmodel']        = config['dmodel']
            result_dic['dff']           = config['dff']
            result_dic['symbolCount']   = config['symbolCount']          
            result_dic['ncoef']         = config['ncoef'] 

            for method_base in baselines:
                for symbol in symbols:
                    for ncoef in ncoefs:
                        # Skip unneccessary configurations
                        if (helper.is_symbolified(method_base)) and (symbol == 0):
                            continue
                        elif (not helper.is_symbolified(method_base)) and (symbol != 0):
                            continue

                        if (method_base == 'SFA') and (ncoef is None):
                            continue
                        
                        # Skip the same configuration
                        if (method_base == method) and (symbol == config['symbolCount']) and (ncoef == config['ncoef']):
                            continue
                        
                        if group[(group['method'] == method_base) & (group['symbolCount'] == symbol)][metric].empty:
                            continue
                        
                        if method_base == 'SFA':
                            predictions_base = group[(group['method'] == method_base) & (group['symbolCount'] == symbol) & (group['ncoef'] == ncoef)][metric].iloc[0]
                        else:
                            if (method_base == method) and (config['symbolCount'] == symbol):
                                continue
                            predictions_base = group[(group['method'] == method_base) & (group['symbolCount'] == symbol)][metric].iloc[0]

                        if not predictions_base:
                            continue

                        fold_fidelities = [
                            calc_fidelity(predictions[fold], predictions_base[fold])
                            for fold in range(len(predictions))
                        ]
                        # Get average fidelity over each fold
                        avg_fold_fidelity = np.nanmean(fold_fidelities)     

                        col_name = f'fidelity_{method_base}'

                        if helper.is_symbolified(method_base):
                            col_name += f'_{symbol}'
                        if method_base == 'SFA':
                            col_name += f'_{ncoef}'

                        result_dic[col_name] = avg_fold_fidelity

            results.append(result_dic)

    results_df = pd.DataFrame(results)
    return results_df


def get_average_fidelities_per_method_same(configs_df, baselines=["ORI", "SAX", "SFA"], metric="Test Predictions", params=['name', 'dmodel', 'dff']):
    """
    Gets the average of the fidelity per method for the same configuration

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baselines (List): List with the methods that are considered the baselines
        metric (str): The metric to evaluate on
        params (list): The parameters that are considered to find the same configuration
        
    Returns
        results_df (DataFrame): DataFrame with the fidelities of each same configuration to the baselines
    """
    fidelities_df = add_fidelities_baselines_same(configs_df, baselines, metric, params)

    # Ensure 'method' column contains hashable types
    fidelities_df['ncoef'] = fidelities_df['ncoef'].replace(np.nan, None)
    results_avg = []
    grouped_avg = fidelities_df.groupby(['method', 'dmodel', 'dff'])
    
    symbols = fidelities_df['symbolCount'].unique()
    ncoefs = fidelities_df['ncoef'].unique()

    for config, group in grouped_avg:
        
        result_dic_avg = {}
        result_dic_avg['method'] = group['method'].iloc[0] 
        result_dic_avg['dmodel'] = group['dmodel'].iloc[0]
        result_dic_avg['dff'] = group['dff'].iloc[0]

        for method_base in baselines:
            for symbol in symbols:
                for ncoef in ncoefs:
                    if (helper.is_symbolified(method_base)) and (symbol == 0):
                        continue
                    elif (not helper.is_symbolified(method_base)) and (symbol != 0):
                        continue

                    if (method_base == 'SFA') and (ncoef is None):
                        continue
            
                    col_name = f'fidelity_{method_base}'

                    if helper.is_symbolified(method_base):
                        col_name += f'_{symbol}'

                    if method_base == 'SFA':
                        col_name += f'_{ncoef}'

                    if group[col_name].empty:
                        continue
                    fidel_avg = np.nanmean(group[col_name])

                    if np.isnan(fidel_avg):
                        continue
                    result_dic_avg[f'avg_{col_name}'] = fidel_avg
        
        results_avg.append(result_dic_avg)

    results_df = pd.DataFrame(results_avg)
    return results_df

def add_fidelities_symbolified_same(configs_df, metric="Test Predictions", params=['name', 'dmodel', 'dff'], sym_methods=['MCB', 'SAX', 'SFA', 'TSFRESH_SYM', 'TSFRESH_SEL_SYM', 'TSFEL_SYM', 'CATCH22_SYM', 'CATCH24_SYM', 'TSFEAT_SYM', 'TOOLS_SYM']):
    """
    Gets the (foldwise-) average fidelity of each symbolified method to the same configuration of the non-symbolified method per dataset.
    For each configuration, the fidelity (averaged over the folds) of each symbolified method to the same configuration of the non-symbolified method is added to a new column.

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on
        params (list): The parameters that are considered to find the same configuration
        sym_methods (list): List with the names of the symbolified methods
        
    Returns
        results_df (DataFrame): DataFrame with the fidelities of each symbolified configuration to its non-symbolified counterpart
    """
    results = []
    conf_pred = add_predictions(configs_df, metric)

    grouped = conf_pred.groupby(params)

    for conf, group in grouped:

        for index, config in group.iterrows():
            method = config['method']

            # skip non-symbolified methods
            if not method in sym_methods:
                continue

            predictions = config[metric]

            if not predictions:
                continue

            result_dic = {}
            result_dic['name']          = config['name']
            result_dic['method']        = config['method']
            result_dic['dmodel']        = config['dmodel']
            result_dic['dff']           = config['dff']
            result_dic['symbolCount']   = config['symbolCount']
            result_dic['ncoef']         = config['ncoef']

            raw_method = method.replace('_SYM', '')
            if (method == 'MCB') or (method == 'SAX') or (method == 'SFA'):
                raw_method = 'ORI'

            if group[(group['method'] == raw_method) & (group['dmodel'] == config['dmodel']) & (group['dff'] == config['dff'])][metric].empty:
                continue
            
            predictions_raw = group[(group['method'] == raw_method) & (group['dmodel'] == config['dmodel']) & (group['dff'] == config['dff'])][metric].iloc[0]

            if not predictions_raw:
                continue

            fold_fidelities = [
                calc_fidelity(predictions[fold], predictions_raw[fold])
                for fold in range(len(predictions))
            ]
            # Get average fidelity over each fold
            avg_fold_fidelity = np.nanmean(fold_fidelities)

            result_dic['fidelity_sym'] = avg_fold_fidelity
            results.append(result_dic)

    results_df = pd.DataFrame(results)
    return results_df

def get_average_fidelities_symbolified_same(configs_df, metric="Test Predictions", params=['name', 'dmodel', 'dff'], sym_methods=['MCB', 'SFA', 'SAX', 'TSFRESH_SYM', 'TSFRESH_SEL_SYM', 'TSFEL_SYM', 'CATCH22_SYM', 'CATCH24_SYM', 'TSFEAT_SYM', 'TOOLS_SYM']):
    """
    Gets the average of the fidelity per symbolified method for the same configuration

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on
        params (list): The parameters that are considered to find the same configuration
        sym_methods (list): List with the names of the symbolified methods
        
    Returns
        results_df (DataFrame): DataFrame with the fidelities of each symbolified configuration to its non-symbolified counterpart
    """
    fidelities_df = add_fidelities_symbolified_same(configs_df, metric, params, sym_methods)
    fidelities_df = fidelities_df.replace(np.nan, 0) # SUbstitute NaNs with 0 to allow easier grouping

    results_avg = []
    grouped_avg = fidelities_df.groupby(['method', 'dmodel', 'dff', 'symbolCount', 'ncoef'])
    
    for config, group in grouped_avg:
        result_dic_avg = {}
        result_dic_avg['method'] = group['method'].iloc[0] 
        result_dic_avg['dmodel'] = group['dmodel'].iloc[0]
        result_dic_avg['dff'] = group['dff'].iloc[0]
        result_dic_avg['symbolCount'] = group['symbolCount'].iloc[0]
        result_dic_avg['ncoef'] = group['ncoef'].iloc[0]

        if group['fidelity_sym'].empty:
            continue
        fidel_avg = np.nanmean(group['fidelity_sym']).astype(float).round(4)
        fidel_std = np.nanstd(group['fidelity_sym']).astype(float).round(4)

        if np.isnan(fidel_avg):
            continue
        result_dic_avg['avg_fidelity_sym'] = fidel_avg
        result_dic_avg['std'] = fidel_std
        
        results_avg.append(result_dic_avg)

    results_df = pd.DataFrame(results_avg)
    results_df['ncoef'] = results_df['ncoef'].replace(0, None)
    return results_df

def add_fidelities_shapelets_same(configs_df, metric="Test Predictions", params=['name', 'symbolCount'], shapelet_methods=['SHAPE', 'SHAPE_MCB', 'SHAPE_SAX', 'SHAPE_SFA']):
    """
    Gets the (foldwise-) average fidelity of each shapelet method to the (best) same configuration of the baseline method per dataset.
    For each configuration, the fidelity (averaged over the folds) of each shapelet method to the same configuration of the non-shapelet method that performs best on the dataset is added to a new column.

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on
        params (list): The parameters that are considered to find the same configuration
        sym_methods (list): List with the names of the shapelet methods
        
    Returns
        results_df (DataFrame): DataFrame with the fidelities of each shapelet configuration to its non-shapelet counterpart
    """
    results = []
    conf_pred = add_predictions(configs_df, metric)

    grouped = conf_pred.groupby(params)

    for conf, group in grouped:
        for index, config in group.iterrows():
            method = config['method']

            # skip non-symbolified methods
            if not method in shapelet_methods:
                continue

            predictions = config[metric]
        
            if not predictions:
                continue
            
            # Iterate through the model sizes to get all Fidelities (Shapelets do not have dmodel, dff parameters)
            for model_size in [[32, 16], [128, 64]]:
                result_dic = {}
                result_dic['name']              = config['name']
                result_dic['method']            = config['method']
                result_dic['symbolCount']       = config['symbolCount']
                result_dic['ncoef']             = config['ncoef']
                result_dic['minShapeletLength'] = config['minShapeletLength']
                result_dic['shapeletTimeLimit'] = config['shapeletTimeLimit']
                result_dic['dmodel']            = model_size[0]
                result_dic['dff']               = model_size[1]

                if method == 'SHAPE':
                    raw_method = 'ORI'
                else:
                    raw_method = method.split('_')[1]
                
                if raw_method == 'ORI':
                    sub_group = group[(group['method'] == raw_method) & (group['dmodel'] == model_size[0]) & (group['dff'] == model_size[1])]
                elif raw_method in ['MCB', 'SAX']:
                    sub_group = group[(group['method'] == raw_method) & (group['symbolCount'] == config['symbolCount']) & (group['dmodel'] == model_size[0]) & (group['dff'] == model_size[1])]
                elif raw_method == 'SFA':
                    sub_group = group[(group['method'] == raw_method) & (group['symbolCount'] == config['symbolCount']) & (group['ncoef'] == config['ncoef']) & (group['dmodel'] == model_size[0]) & (group['dff'] == model_size[1])]

                if sub_group[metric].empty:
                    continue

                predictions_shapelet = sub_group[metric].iloc[0]
                
                if not predictions_shapelet:
                    continue

                fold_fidelities = [
                    calc_fidelity(predictions[fold], predictions_shapelet[fold])
                    for fold in range(len(predictions))
                ]
                # Get average fidelity over each fold
                avg_fold_fidelity = np.nanmean(fold_fidelities)
                
                result_dic['fidelity_shapelet'] = avg_fold_fidelity
                results.append(result_dic)
            
    results_df = pd.DataFrame(results)
    return results_df  

def get_average_fidelities_shapelets_same(configs_df, metric="Test Predictions", params=['name', 'symbolCount'], shapelet_methods=['SHAPE', 'SHAPE_MCB', 'SHAPE_SAX', 'SHAPE_SFA']):
    """
    Gets the average fidelity of each shapelet method to it's non-shapelet counterpart of the same configuration

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on
        params (list): The parameters that are considered to find the same configuration
        sym_methods (list): List with the names of the shapelet methods
        
    Returns
        results_df (DataFrame): DataFrame with the average fidelities of each shapelet configuration to its non-shapelet counterpart
    """  
    results = []
    conf_fidel = add_fidelities_shapelets_same(configs_df, metric, params, shapelet_methods)
    conf_fidel = conf_fidel.replace(np.nan, 0)

    grouped = conf_fidel.groupby(['method', 'symbolCount', 'ncoef', 'minShapeletLength', 'shapeletTimeLimit', 'dmodel', 'dff'])
    
    for conf, group in grouped:
        result_dic = {}
        result_dic['method']            = group['method'].iloc[0]
        result_dic['symbolCount']       = group['symbolCount'].iloc[0]
        result_dic['ncoef']             = group['ncoef'].iloc[0]
        result_dic['minShapeletLength'] = group['minShapeletLength'].iloc[0]
        result_dic['shapeletTimeLimit'] = group['shapeletTimeLimit'].iloc[0]
        result_dic['dmodel']            = group['dmodel'].iloc[0]
        result_dic['dff']               = group['dff'].iloc[0]
        result_dic['avg_fidelity']      = group['fidelity_shapelet'].mean().astype(float).round(4)
        result_dic['std']               = group['fidelity_shapelet'].std().astype(float).round(4)
        
        results.append(result_dic)
    
    results_df = pd.DataFrame(results)
    results_df['ncoef'] = results_df['ncoef'].replace(0, None)
    return results_df


def get_count_better_baseline(configs_df, baseline='ORI', metric='Test Accuracy'):
    """
    Counts how many times a method is better than the baseline
    Gets the best configuration per method and per dataset and counts how many times a method is better than the baseline

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        baseline (str): The method that is considered the baseline

    Returns:
        better_df (DataFrame): DataFrame with the counts of how many times a method is better than the baseline
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

            if group[group['method'] == baseline][metric].empty:
                continue
            
            if config[metric] > group[group['method'] == baseline][metric].iloc[0]:
                result_dic[f'better_{baseline}'] = 1
            else:
                result_dic[f'better_{baseline}'] = 0
            
            results.append(result_dic)
    
    better_df = pd.DataFrame(results)
    better_df = better_df.groupby('method')[f'better_{baseline}'].sum().reset_index()

    return better_df

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

def get_count_method_best(configs_df, metric='Test Accuracy'):
    """
    Counts on how many datasets each method was the best and how many times it was the only best method,
    ensuring all methods are included even if they were never the best.

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        best_df (DataFrame): DataFrame with the counts of how many times a method was the best and the only best
    """
    best_results_ds_met = get_best_per_dataset_method(configs_df, metric)
    best_per_dataset = best_results_ds_met.groupby('name')[metric].max().reset_index()
    
    best_methods = pd.merge(best_results_ds_met, best_per_dataset, on=['name', metric])
    
    count_best = best_methods['method'].value_counts().reset_index()
    count_best.columns = ['method', 'count']
    
    best_methods['is_only_best'] = best_methods.groupby('name')['method'].transform('count') == 1
    count_only_best = best_methods[best_methods['is_only_best']]['method'].value_counts().reset_index()
    count_only_best.columns = ['method', 'only_best_count']
    
    # Ensure all methods are included, even if they were never the best
    all_methods = configs_df['method'].unique()
    all_methods_df = pd.DataFrame({'method': all_methods})
    count_best = pd.merge(all_methods_df, count_best, on='method', how='left').fillna(0)
    count_best = pd.merge(count_best, count_only_best, on='method', how='left').fillna(0)
    
    count_best['count'] = count_best['count'].astype(int)
    count_best['only_best_count'] = count_best['only_best_count'].astype(int)
    
    return count_best

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

def get_average_best(configs_df, metric='Test Accuracy'):
    """
    Gets the average of the best results

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        avg_best (float): The average of the best results
    """
    best_results_ds_met = get_best_per_dataset(configs_df, metric)
    avg_best = best_results_ds_met[metric].mean().astype(float).round(4)
    std_best = best_results_ds_met[metric].std().astype(float).round(4)

    if std_best is np.nan:
        std_best = 0.0

    return avg_best, std_best
    
def get_average_best_raw(configs_df, metric='Test Accuracy', raw_methods=['ORI', 'TSFRESH', 'TSFRESH_SEL', 'TSFEL', 'CATCH22', 'CATCH24', 'TSFEAT', 'TOOLS']):
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
    avg_best = best_results_ds_met[metric].mean().astype(float).round(4)
    std_best = best_results_ds_met[metric].std().astype(float).round(4)

    if std_best is np.nan:
        std_best = 0.0

    return avg_best, std_best

def get_average_best_sym(configs_df, metric='Test Accuracy', sym_methods=['MCB', 'SAX', 'SFA', 'TSFRESH_SYM', 'TSFRESH_SEL_SYM', 'TSFEL_SYM', 'CATCH22_SYM', 'CATCH24_SYM', 'TSFEAT_SYM', 'TOOLS_SYM'], strategy=None):
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
    avg_best = best_results_ds_met[metric].mean().astype(float).round(4)
    std_best = best_results_ds_met[metric].std().astype(float).round(4)

    if std_best is np.nan:
        std_best = 0.0

    return avg_best, std_best

def get_averages(configs_df, metric='Test Accuracy'):
    """
    Gets the average of the best results of the non-symbolified and symbolified methods

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        avg_best (float): The average of the best results
    """
    avg_best, std_best = get_average_best(configs_df, metric)
    avg_best_ori, std_best_ori = get_average_best_raw(configs_df, metric, raw_methods=['ORI'])
    avg_best_raw, std_best_raw = get_average_best_raw(configs_df, metric)
    avg_best_raw_wo_ori, std_best_raw_wo_ori = get_average_best_raw(configs_df, metric, raw_methods=['TSFRESH', 'TSFRESH_SEL', 'TSFEL', 'CATCH22', 'CATCH24', 'TSFEAT', 'TOOLS'])
    avg_best_sym, std_best_sym = get_average_best_sym(configs_df, metric)
    avg_best_sym_uniform, std_best_sym_uniform = get_average_best_sym(configs_df, metric, strategy='uniform')
    avg_best_sym_quantile, std_best_sym_quantile = get_average_best_sym(configs_df, metric, strategy='quantile')

    best_df = pd.DataFrame([{'method': 'ALL', 'avg_best': avg_best, 'std_best': std_best}, 
                            {'method': 'ORI', 'avg_best': avg_best_ori, 'std_best': std_best_ori},
                            {'method': 'RAW', 'avg_best': avg_best_raw, 'std_best': std_best_raw}, 
                            {'method': 'RAW without ORI', 'avg_best': avg_best_raw_wo_ori, 'std_best': std_best_raw_wo_ori},
                            {'method': 'SYM', 'avg_best': avg_best_sym, 'std_best': std_best_sym},
                            {'method': 'SYM uniform', 'avg_best': avg_best_sym_uniform, 'std_best': std_best_sym_uniform},
                            {'method': 'SYM quantile', 'avg_best': avg_best_sym_quantile, 'std_best': std_best_sym_quantile}])
    
    return best_df

def get_averages_type(configs_df_type, type, metric='Test Accuracy'):
    """
    Gets the average of the best results of the non-symbolified and symbolified methods

    Args:
        configs_df (DataFrame): DataFrame with all configurations
        metric (str): The metric to evaluate on

    Returns:
        avg_best (float): The average of the best results
    """
    if configs_df_type.empty:
        best_df = pd.DataFrame([{'method': 'ALL', f'{type}': '-'}, 
                                {'method': 'ORI', f'{type}': '-'},
                                {'method': 'RAW', f'{type}': '-'}, 
                                {'method': 'RAW without ORI', f'{type}': '-'},
                                {'method': 'SYM', f'{type}': '-'},
                                {'method': 'SYM uniform', f'{type}': '-'},
                                {'method': 'SYM quantile', f'{type}': '-'}])
    
    else:
        avg_best, std_best = get_average_best(configs_df_type, metric)
        avg_best_ori, std_best_ori = get_average_best_raw(configs_df_type, metric, raw_methods=['ORI'])
        avg_best_raw, std_best_raw = get_average_best_raw(configs_df_type, metric)
        avg_best_raw_wo_ori, std_best_raw_wo_ori = get_average_best_raw(configs_df_type, metric, raw_methods=['TSFRESH', 'TSFRESH_SEL', 'TSFEL', 'CATCH22', 'CATCH24', 'TSFEAT', 'TOOLS'])
        avg_best_sym, std_best_sym = get_average_best_sym(configs_df_type, metric)
        avg_best_sym_uniform, std_best_sym_uniform = get_average_best_sym(configs_df_type, metric, strategy='uniform')
        avg_best_sym_quantile, std_best_sym_quantile = get_average_best_sym(configs_df_type, metric, strategy='quantile')

        best_df = pd.DataFrame([{'method': 'ALL', f'{type}': f'{avg_best} +- {std_best}'}, 
                                {'method': 'ORI', f'{type}': f'{avg_best_ori} +- {std_best_ori}'},
                                {'method': 'RAW', f'{type}': f'{avg_best_raw} +- {std_best_raw}'}, 
                                {'method': 'RAW without ORI', f'{type}': f'{avg_best_raw_wo_ori} +- {std_best_raw_wo_ori}'},
                                {'method': 'SYM', f'{type}': f'{avg_best_sym} +- {std_best_sym}'},
                                {'method': 'SYM uniform', f'{type}': f'{avg_best_sym_uniform} +- {std_best_sym_uniform}'},
                                {'method': 'SYM quantile', f'{type}': f'{avg_best_sym_quantile} +- {std_best_sym_quantile}'}])
    
    return best_df

def get_averages_per_type(configs_df, metric='Test Accuracy'):
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


def get_configs_shapelets(paths):
    """
    Gets the configurations specified in yaml files for the shapelet experiments
    
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
        minShapeletLength_list = grid_params.get("model.minShapeletLength", {}).get("options", [])
        shapeletTimeLimit_list = grid_params.get("model.shapeletTimeLimit", {}).get("options", [])
        ncoefs_list = grid_params.get("model.ncoef", {}).get("options", [])
        

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
            'symbolCount':          symbolCounts_list,
            'method':               methods_list,
            'ncoef':                ncoefs_list_filtered,
            'minShapeletLength':    minShapeletLength_list,
            'shapeletTimeLimit':    shapeletTimeLimit_list
        }

        configs += helper.combine_factors([factors])

        # Add name of the dataset to each config
        for i, conf in enumerate(configs):
            conf['name'] = dataset_list[conf['number']]
    
    return configs

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
