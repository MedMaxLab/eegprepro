import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def LearningRateSelector(Table, metric, model, pipeline, task, 
                         sampling = 125, show_plot = True, verbose = True):

    # check input parameters and extract better names for plot
    metric_list = [
        'accuracy_unbalanced', 'accuracy_weighted', 'precision_micro', 
        'precision_macro', 'precision_weighted', 'recall_micro', 'recall_macro',
        'recall_weighted', 'f1score_micro', 'f1score_macro', 'f1score_weighted',
        'rocauc_micro', 'rocauc_macro', 'rocauc_weighted', 'cohen_kappa'
    ]
    model_dict = {
        'eegnet': 'egn', 'shallownet': 'shn', 'deepconvnet': 'dcn',
        'resnet': 'res', 'eegsym': 'egs', 'atcnet': 'atc', 'hybridnet': 'hyb',
        'fbcnet': 'fbc'
    }
    pipeline_dict = {'raw': 'raw', 'filt': 'flt', 'ica': 'ica', 'icasr': 'isr'}
    task_dict = {
        'eyes': 'eye',
        'alzheimer': 'alz',
        'parkinson': 'pds',
        'motorimagery': 'mmi',
        'sleep': 'slp',
        'psychosis': 'fep'
    }

    if metric in metric_list:
        metric_full = metric.replace('_', ' ').replace('1','1-') 
    elif metric in [i.replace('_', ' ').replace('1','1-') for i in metric_list]:
        metric_full = metric
        metric = metric.replace(' ', '_').replace('1-','1')
    else:
        raise ValueError('metric not supported')
    
    if model in list(model_dict.keys()):
        model_full = model
        model = model_dict[model]
    elif model in list(model_dict.values()):
        inv_dict = {v: k for k, v in model_dict.items()}
        model_full = inv_dict[model]
    else:
        raise ValueError('model not supported')

    if pipeline in list(pipeline_dict.keys()):
        pipeline_full = pipeline
        pipeline = pipeline_dict[model]
    elif pipeline in list(pipeline_dict.values()):
        inv_dict = {v: k for k, v in pipeline_dict.items()}
        pipeline_full = inv_dict[pipeline]
    else:
        raise ValueError('pipeline not supported')

    if task in list(task_dict.keys()):
        task_full = task
        task = task_dict[task]
    elif task in list(task_dict.values()):
        inv_dict = {v: k for k, v in task_dict.items()}
        task_full = inv_dict[task]
    else:
        raise ValueError('model not supported')
    
    TableMini = Table.loc[(Table['model'] == model) & 
        (Table['sampling_rate'] == sampling) & 
        (Table['pipeline'] == pipeline) & 
        (Table['task'] == task ) &
        (Table['inner_fold'] == 1 )]
    lr_list = TableMini['learning_rate'].unique().tolist()
    lr_list.sort(reverse=True)
    lr_tick = ["{:.1e}".format(i) for i in lr_list]
    performances = [None] * len(lr_list)
    for idx, lr in enumerate(lr_list):
        performances[idx] = TableMini.loc[(TableMini['learning_rate'] == lr), metric]
        performances[idx] = performances[idx].to_list()
    
    perfmat = np.array(performances).T
    order = (-perfmat).argsort(axis=1)
    ranks = order.argsort(axis=1)
    ranksum = np.sum(ranks, axis=0)
    perfmedian = np.median(perfmat,axis=0)
    
    lr_dict = {
        'median': lr_list[np.argmax(perfmedian)],
        'rank': lr_list[np.argmin(ranksum)]
    }
    if verbose:
        print(' ')
        print(metric_full + ' median values')
        print(np.around(perfmedian, 4).tolist())
        print(' ')
        print(metric_full + ' max values')
        print(np.around( np.max(perfmat, axis = 0), 4).tolist())
        print(' ')
        print(metric_full + ' min values')
        print(np.around( np.min(perfmat, axis = 0), 4).tolist())
        print(' ')
        print(metric_full + ' rank values (lower values are better)')
        print(ranksum.tolist())
        print(' ')
        print(metric_full + ' full rank matrix')
        print(ranks)
    
    if show_plot:
        plt.figure( figsize = (15,8))
        plt.boxplot(performances, labels = lr_tick )
        plt.title(
            metric_full + ' -- model ' + model_full + \
            ', task ' + task_full + ', pipeline ' + pipeline_full + ' --',
            fontsize = 22
        )
        plt.ylabel(metric_full, fontsize=20)
        plt.yticks(fontsize=15)
        plt.xlabel('Learning Rate', fontsize=20)
        plt.xticks(fontsize=15)
        plt.show()

    return lr_dict