import pandas as pd
import numpy as np
import os

from modules import helper
import evaluate

from pyts.datasets import ucr_dataset_list

import matplotlib.pyplot as plt

from pathlib import Path

from modules.cd_diagram import draw_cd_diagram
    
def plot_cdd(configs_df_filtered, save_name="cdd"):

    results = evaluate.get_best_per_dataset_method(configs_df_filtered)
    results = evaluate.get_foldwise_results(results)
    accs = evaluate.get_average_per_method_foldwise(configs_df_filtered)

    results = results.replace('ORI', 'ORIGINAL')
    accs = accs.replace('ORI', 'ORIGINAL')

    print(accs)

    results_dic = {}
    results_dic['classifier_name'] = list(results['method'])
    results_dic['dataset_name'] = list(results['number'].astype(str) + results['fold'].astype(str))
    results_dic['accuracy'] = list(results['Test Accuracy'])
    results_df = pd.DataFrame.from_dict(results_dic)
    draw_cd_diagram(df_perf=results_df, title='', labels=True, save_name=save_name, accs=accs)

# Plotting for results averaging over all folds
def plot_averages_foldwise(configs_df_filtered, save_name="averages", figsize=(14, 15)):
    results_df = evaluate.get_averages_per_type_foldwise(configs_df_filtered, as_csv=True, csv_save_name=f"plots/{save_name}.csv")
   
    evaluate.plot_dataframe(results_df, save_name, pdf=True)

def plot_avg_inc_dec_overall_foldwise(configs_df_filtered, save_name="avg_inc_dec_overall", figsize=(14, 15)):
    
    avg_inc_dec = evaluate.get_avg_inc_dec_to_baseline_best_foldwise(configs_df_filtered)

    allNames = list(np.unique(configs_df_filtered['method']))
    allNames.remove("ORI") # ORI is baseline for the increase/decrease
    
    kNames = []
    for name in allNames: 
        # Delete all methods that are symbolified by our preprocessing
        if not helper.do_symbolize(name) or name == 'MCB':
            kNames.append(name)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                                    figsize=figsize, layout="constrained")
    colorN = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
    plt.rcParams.update({'font.size': 18})
    
    ax.set_title('Average Increase/Decrease')
    width = 0
    n_bars = 2  
    standardWidth = 0.8
    bar_width = 0.8 / n_bars
    barInd = 0
    rects = []

    colorID = 0
    for mi, strategy in enumerate(['raw', 'symbolized']):  

        resultVs = np.zeros(len(kNames))
        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
        barInd+= 1
        width += standardWidth
       
        for pos in [True, False]: 

            lables = kNames.copy() 
            resultVstd = []

            for vi, v in enumerate(resultVs): 
                method = lables[vi]

                if method not in ['MCB', 'SAX', 'SFA'] and strategy == 'symbolized':
                    name = method + '_SYM'
                elif method in ['MCB', 'SAX', 'SFA'] and strategy == 'raw':
                    name = 'ORI'
                else: 
                    name = method

                if strategy == 'raw':
                    if pos:
                        resultVs[vi] = avg_inc_dec[avg_inc_dec['method'] == name]['inc'].iloc[0]
                    elif not pos:
                        resultVs[vi] = avg_inc_dec[avg_inc_dec['method'] == name]['dec'].iloc[0]
                    resultVstd.append(avg_inc_dec[avg_inc_dec['method'] == name]['std'].iloc[0])
                    
                elif strategy == 'symbolized':
                    if pos:
                        resultVs[vi] = avg_inc_dec[avg_inc_dec['method'] == name]['inc'].iloc[0]
                    elif not pos:
                        resultVs[vi] = avg_inc_dec[avg_inc_dec['method'] == name]['dec'].iloc[0]
                    resultVstd.append(avg_inc_dec[avg_inc_dec['method'] == name]['std'].iloc[0])
    
            counts = np.round(resultVs,4)
            e =  np.round(resultVstd,4).squeeze()

            ind = np.arange(len(counts))
            rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3, color=list(colorN)[colorID]) 
            colorID += 1
            rects.append(rect)
            ax.set_xticks(ind)
            ax.set_xticklabels(lables)
            ax.tick_params(labelrotation=90, labelsize=17)
            ax.set_ylabel('Percent', fontsize=20)

    specificFolder = 'plots/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
     
    legend = ['Inc. Non-Symbolic', 'Dec. Non-Symbolic', 'Inc. Symbolic', 'Dec. Symbolic']
    fig.legend(handles=rects, labels=legend,
                    loc="upper right", bbox_to_anchor=(0.99, 1.095))
    fig.tight_layout()
    fig.savefig(specificFolder + save_name + '.pdf', dpi = 300, bbox_inches = 'tight')
     
def plot_avg_acc_strategy_foldwise(configs_df_filtered, save_name="avg_acc_strategy", figsize=(14, 15)):

    configs_df_filtered_raw = configs_df_filtered[(configs_df_filtered['symbolCount'] == 0)]
    configs_df_filtered_uniform = configs_df_filtered[(configs_df_filtered['strategy'] == 'uniform') | (configs_df_filtered['method'] == 'ORI')]
    configs_df_filtered_quantile = configs_df_filtered[(configs_df_filtered['strategy'] == 'quantile') | (configs_df_filtered['method'] == 'ORI')]

    avg_acc_raw = evaluate.get_average_per_method_foldwise(configs_df_filtered_raw)
    avg_acc_uniform = evaluate.get_average_per_method_foldwise(configs_df_filtered_uniform)
    avg_acc_quantile = evaluate.get_average_per_method_foldwise(configs_df_filtered_quantile)

    allNames = list(np.unique(configs_df_filtered['method']))
    allNames.remove("ORI") # ORI is baseline for the increase/decrease
    
    kNames = []
    for name in allNames: 
        # Delete all methods that are symbolified by our preprocessing
        if not helper.do_symbolize(name) or name == 'MCB':
            kNames.append(name)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                                    figsize=figsize, layout="constrained")
    colorN = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
    plt.rcParams.update({'font.size': 18})
    
    ax.set_title('Average Accuracy')
    width = 0
    n_bars = 3  
    standardWidth = 0.8
    bar_width = 0.8 / n_bars
    barInd = 0
    rects = []

    colorID = 0
    for mi, strategy in enumerate(['raw', 'uniform', 'quantile']):  

        resultVs = np.zeros(len(kNames))
        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
        barInd+= 1
        width += standardWidth

        lables = kNames.copy() 
        resultVstd = []

        for vi, v in enumerate(resultVs): 
            method = lables[vi]

            if method not in ['MCB', 'SAX', 'SFA'] and strategy in ['uniform', 'quantile']:
                name = method + '_SYM'
            elif method in ['MCB', 'SAX', 'SFA'] and strategy == 'raw':
                    name = 'ORI'
            else: 
                name = method

            if strategy == 'raw':
                resultVs[vi] = avg_acc_raw[avg_acc_raw['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_raw[avg_acc_raw['method'] == name]['std'].iloc[0])
                
            elif strategy == 'uniform':
                resultVs[vi] = avg_acc_uniform[avg_acc_uniform['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_uniform[avg_acc_uniform['method'] == name]['std'].iloc[0])

            elif strategy == 'quantile':
                resultVs[vi] = avg_acc_quantile[avg_acc_quantile['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_quantile[avg_acc_quantile['method'] == name]['std'].iloc[0])

        counts = np.round(resultVs,4)
        e =  np.round(resultVstd,4).squeeze()

        ind = np.arange(len(counts))
        rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3, color=list(colorN)[colorID]) 
        colorID += 1
        rects.append(rect)
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.tick_params(labelrotation=90, labelsize=17)
        ax.set_ylabel('Percent', fontsize=20)

    specificFolder = 'plots/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
     
    legend = ['Non-Symbolic', 'Uniform', 'Quantile']
    fig.legend(handles=rects, labels=legend,
                    loc="upper right", bbox_to_anchor=(0.99, 1.095))
    fig.tight_layout()
    fig.savefig(specificFolder + save_name + '.pdf', dpi = 300, bbox_inches = 'tight')
    
def plot_avg_acc_symbolCount_foldwise(configs_df_filtered, save_name="avg_acc_symbolCount", figsize=(14, 15)):

    configs_df_filtered_raw = configs_df_filtered[(configs_df_filtered['symbolCount'] == 0)]
    configs_df_filtered_5 = configs_df_filtered[(configs_df_filtered['symbolCount'] == 5) | (configs_df_filtered['method'] == 'ORI')]
    configs_df_filtered_6 = configs_df_filtered[(configs_df_filtered['symbolCount'] == 6) | (configs_df_filtered['method'] == 'ORI')]

    avg_acc_raw = evaluate.get_average_per_method_foldwise(configs_df_filtered_raw)
    avg_acc_5 = evaluate.get_average_per_method_foldwise(configs_df_filtered_5)
    avg_acc_6 = evaluate.get_average_per_method_foldwise(configs_df_filtered_6)

    allNames = list(np.unique(configs_df_filtered['method']))
    allNames.remove("ORI") # ORI is baseline for the increase/decrease
    
    kNames = []
    for name in allNames: 
        # Delete all methods that are symbolified by our preprocessing
        if not helper.do_symbolize(name) or name == 'MCB':
            kNames.append(name)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                                    figsize=figsize, layout="constrained")
    colorN = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
    plt.rcParams.update({'font.size': 18})

    ax.set_title('Average Accuracy')
    width = 0
    n_bars = 3  
    standardWidth = 0.8
    bar_width = 0.8 / n_bars
    barInd = 0
    rects = []

    colorID = 0
    for mi, symbols in enumerate(['raw', 'symbols_5', 'symbols_6']):  

        resultVs = np.zeros(len(kNames))
        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
        barInd+= 1
        width += standardWidth

        lables = kNames.copy() 
        resultVstd = []

        for vi, v in enumerate(resultVs): 
            method = lables[vi]

            if method not in ['MCB', 'SAX', 'SFA'] and symbols in ['symbols_5', 'symbols_6']:
                name = method + '_SYM'
            elif method in ['MCB', 'SAX', 'SFA'] and symbols == 'raw':
                    name = 'ORI'
            else: 
                name = method

            if symbols == 'raw':
                resultVs[vi] = avg_acc_raw[avg_acc_raw['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_raw[avg_acc_raw['method'] == name]['std'].iloc[0])
                
            elif symbols == 'symbols_5':
                resultVs[vi] = avg_acc_5[avg_acc_5['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_5[avg_acc_5['method'] == name]['std'].iloc[0])

            elif symbols == 'symbols_6':
                resultVs[vi] = avg_acc_6[avg_acc_6['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_6[avg_acc_6['method'] == name]['std'].iloc[0])

        counts = np.round(resultVs,4)
        e =  np.round(resultVstd,4).squeeze()

        ind = np.arange(len(counts))
        rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3, color=list(colorN)[colorID]) 
        colorID += 1
        rects.append(rect)
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.tick_params(labelrotation=90)
        ax.set_ylabel('Percent')

    specificFolder = 'plots/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
     
    legend = ['Raw', 'Avg. Acc. 5 Symbols', 'Avg. Acc. 6 Symbols']
    fig.legend(handles=rects, labels=legend,
                    loc="upper right", bbox_to_anchor=(1.198, 0.08))
    fig.tight_layout()
    fig.savefig(specificFolder + save_name + '.pdf', dpi = 300, bbox_inches = 'tight')

def plot_avg_acc_architecture_foldwise(configs_df_filtered, save_name="avg_acc_architecture", figsize=(14, 15)):

    configs_df_filtered_transformer = configs_df_filtered[(configs_df_filtered['architecture'] == 'transformer')]
    configs_df_filtered_resnet = configs_df_filtered[(configs_df_filtered['architecture'] == 'resnet')]

    avg_acc_transformer = evaluate.get_average_per_method_foldwise(configs_df_filtered_transformer)
    avg_acc_resnet = evaluate.get_average_per_method_foldwise(configs_df_filtered_resnet)

    avg_acc_transformer = avg_acc_transformer.replace('ORI', 'ORIGINAL')
    avg_acc_resnet = avg_acc_resnet.replace('ORI', 'ORIGINAL')

    configs_df_filtered = configs_df_filtered.replace('ORI', 'ORIGINAL')

    allNames = list(np.unique(configs_df_filtered['method']))
    kNames = allNames
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                                    figsize=figsize, layout="constrained")
    colorN = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
    plt.rcParams.update({'font.size': 18})
    
    ax.set_title('Average Accuracy')
    width = 0
    n_bars = 2  
    standardWidth = 0.8
    bar_width = 0.8 / n_bars
    barInd = 0
    rects = []

    colorID = 0
    for mi, architecture in enumerate(['transformer', 'resnet']):  

        resultVs = np.zeros(len(kNames))
        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
        barInd+= 1
        width += standardWidth

        lables = kNames.copy() 
        resultVstd = []

        for vi, v in enumerate(resultVs): 
            method = lables[vi]
            name = method

            if architecture == 'transformer':
                resultVs[vi] = avg_acc_transformer[avg_acc_transformer['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_transformer[avg_acc_transformer['method'] == name]['std'].iloc[0])
                
            elif architecture == 'resnet':
                resultVs[vi] = avg_acc_resnet[avg_acc_resnet['method'] == name]['Test Accuracy'].iloc[0]
                resultVstd.append(avg_acc_resnet[avg_acc_resnet['method'] == name]['std'].iloc[0])

        counts = np.round(resultVs,4)
        e =  np.round(resultVstd,4).squeeze()

        ind = np.arange(len(counts))
        if 'TSFRESH_SEL' in list(np.unique(configs_df_filtered['method'])):
            ind[-1], ind[-2], ind[-3] = ind[-3], ind[-1], ind[-2]  # For better plot order
        rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3, color=list(colorN)[colorID]) 
        colorID += 1
        rects.append(rect)
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.tick_params(labelrotation=90, labelsize=17)
        ax.set_ylabel('Percent', fontsize=20)

    specificFolder = 'plots/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
     
    legend = ['Transformer', 'CNN']
    fig.legend(handles=rects, labels=legend,
            #    loc="upper right")
                    loc="upper right", bbox_to_anchor=(0.99, 1.05))
    fig.tight_layout()
    fig.savefig(specificFolder + save_name + '.pdf', dpi = 300, bbox_inches = 'tight')
    
    
def plot_better_best_table_foldwise(configs_df_filtered, baseline='ORI', save_name="count_better_best_table", folds=5, as_csv=True):
    better = evaluate.get_flags_better_baseline_foldwise(configs_df_filtered)
    best = evaluate.get_flags_method_best_foldwise(configs_df_filtered)

    num_datasets = np.unique(configs_df_filtered['name']).shape[0]

    better_dic = {}
    best_dic = {}
    abs_best_dic = {}

    grouped_better = better.groupby('method')
    for method, group in grouped_better:
        better_dic[method] = group[f'better_{baseline}'].sum() / (num_datasets * folds)
    
    grouped_best = best.groupby('method')
    for method, group in grouped_best:
        best_dic[method] = group[f'count'].sum() / (num_datasets * folds)
        abs_best_dic[method] = group[f'only_best_count'].sum() / (num_datasets * folds)

    df = pd.DataFrame([better_dic, best_dic, abs_best_dic], 
                  index=['Better Original', 'Best', 'Absolute Best']).astype(float).round(4).replace(np.nan, '-')
    
    df = df.transpose()

    if as_csv:
        df.to_csv(f"plots/{save_name}.csv")
    else:
        evaluate.plot_dataframe(df, name=save_name, pdf=True)

def get_best_tsfresh(configs_df_filtered):
    configs_df_best = evaluate.get_flags_method_best(configs_df_filtered)

    count = 0
    count_same = 0
    for index, config in configs_df_best.iterrows():
        if ((config['method'] == 'TSFRESH_SYM') or (config['method'] == 'TSFRESH_SEL_SYM')) and (config['count'] == 1):
            count += 1

        # check if both are equally good
        if (config['method'] == 'TSFRESH_SYM') and (config['count'] == 1):
            group = configs_df_best[(configs_df_best['name'] == config['name']) & (configs_df_best['method'] == 'TSFRESH_SEL_SYM')]

            for idx, conf in group.iterrows():
                if conf['count'] == 1:
                    count_same += 1

        elif (config['method'] == 'TSFRESH_SEL_SYM') and (config['count'] == 1):
            group = configs_df_best[(configs_df_best['name'] == config['name']) & (configs_df_best['method'] == 'TSFRESH_SYM')]

            for idx, conf in group.iterrows():
                if conf['count'] == 1:
                    count_same += 1


    configs_df_best_foldwise = evaluate.get_flags_method_best_foldwise(configs_df_filtered)

    count_foldwise = 0
    count_same_foldwise = 0
    for index, config in configs_df_best_foldwise.iterrows():
        if ((config['method'] == 'TSFRESH_SYM') or (config['method'] == 'TSFRESH_SEL_SYM')) and (config['count'] == 1):
            count_foldwise += 1

        # check if both are equally good
        if (config['method'] == 'TSFRESH_SYM') and (config['count'] == 1):
            group = configs_df_best_foldwise[(configs_df_best_foldwise['name'] == config['name']) & (configs_df_best_foldwise['method'] == 'TSFRESH_SEL_SYM') & (configs_df_best_foldwise['fold'] == config['fold'])]

            for idx, conf in group.iterrows():
                if conf['count'] == 1:
                    count_same_foldwise += 1

        elif (config['method'] == 'TSFRESH_SEL_SYM') and (config['count'] == 1):
            group = configs_df_best_foldwise[(configs_df_best_foldwise['name'] == config['name']) & (configs_df_best_foldwise['method'] == 'TSFRESH_SYM') & (configs_df_best_foldwise['fold'] == config['fold'])]

            for idx, conf in group.iterrows():
                if conf['count'] == 1:
                    count_same_foldwise += 1

    print(f"Tsfresh_sym/Tsfresh_sel_sym best: {count}; foldwise: {count_foldwise}")
    print(f'Same performance: {count_same} datasets, {count_same_foldwise} folds')

def plot_num_datasets_per_type(configs_df_filtered, configs_df_filtered_sel):
    evaluate.get_datasets_per_type(configs_df_filtered, configs_df_filtered_sel)

#####################
# In Paper
#####################
Path("plots").mkdir(exist_ok=True)

figsize = (12, 8)

yaml_files_transformer = ['configRawFeatures.yaml', 'configSymbolFeatures.yaml', 'configSfaFeatures.yaml']
yaml_files_resnet = ['configResNetRaw.yaml', 'configResNetSym.yaml', 'configResNetSfa.yaml']
yaml_files = yaml_files_transformer + yaml_files_resnet
configs_transformer = evaluate.get_configs(yaml_files_transformer)
configs_resnet = evaluate.get_configs_resnet(yaml_files_resnet)
configs = configs_transformer + configs_resnet
configs_df = pd.DataFrame.from_dict(configs)
configs_df = configs_df.replace(np.nan, None)

datasets = evaluate.find_datasets_each_methods_finished(configs_df, yaml_files)

# Plotting with TSFRESH_SEL
configs_df_filtered = configs_df[configs_df['name'].isin(datasets)]
print("Number of datasets with TSFRESH_SEL: ", np.unique(configs_df_filtered['name']).shape[0])

datasets_sel = evaluate.find_datasets_each_methods_finished(configs_df, yaml_files, ignore=['TSFRESH_SEL', 'TSFRESH_SEL_SYM'])
configs_df_filtered_sel = configs_df[configs_df['name'].isin(datasets_sel)]
configs_df_filtered_sel = evaluate.filter_tsfresh_sel(configs_df_filtered_sel)
print("Number of datasets without TSFRESH_SEL: ", np.unique(configs_df_filtered_sel['name']).shape[0])

plot_cdd(configs_df_filtered, save_name="plots/cdd")
plot_cdd(configs_df_filtered_sel, save_name="plots/wo_cdd")

plot_avg_acc_architecture_foldwise(configs_df_filtered, save_name="avg_acc_architecture", figsize=figsize)
plot_avg_acc_architecture_foldwise(configs_df_filtered_sel, save_name="wo_avg_acc_architecture", figsize=figsize)

plot_better_best_table_foldwise(configs_df_filtered, save_name="count_better_best_table")
plot_better_best_table_foldwise(configs_df_filtered_sel, save_name="wo_count_better_best_table")

plot_avg_acc_strategy_foldwise(configs_df_filtered, save_name="avg_acc_strategy", figsize=figsize)
plot_avg_acc_strategy_foldwise(configs_df_filtered_sel, save_name="wo_avg_acc_strategy", figsize=figsize)

plot_avg_inc_dec_overall_foldwise(configs_df_filtered, save_name="avg_inc_dec_overall", figsize=figsize)
plot_avg_inc_dec_overall_foldwise(configs_df_filtered_sel, save_name="wo_avg_inc_dec_overall", figsize=figsize)

plot_averages_foldwise(configs_df_filtered, save_name="averages_per_type")
plot_averages_foldwise(configs_df_filtered_sel, save_name="wo_averages_per_type")

plot_num_datasets_per_type(configs_df_filtered, configs_df_filtered_sel)

get_best_tsfresh(configs_df_filtered)