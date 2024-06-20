import argparse
import sys
sys.path.append('..')
from AllFnc.utilities import column_switch
import glob
import re
import pandas as pd
import pickle
import os

def gather_results(save = False, filename = None):
    
    metrics_list = [ 
        'accuracy_unbalanced', 'accuracy_weighted',
        'precision_micro',     'precision_macro',   'precision_weighted',
        'recall_micro',        'recall_macro',      'recall_weighted',
        'f1score_micro',       'f1score_macro',     'f1score_weighted',
        'rocauc_micro',        'rocauc_macro',      'rocauc_weighted',
        'cohen_kappa'
    ]
    piece_list = [
        'task', 'pipeline', 'sampling_rate', 'model', 
        'outer_fold', 'inner_fold', 'learning_rate',
        'channels', 'window'
    ]
    
    set_full = set(glob.glob('**/Results/*_061_*.pickle'))
    set_torm = set(glob.glob('Supplementary/Results/*.pickle'))
    file_list = list(set_full - set_torm)
    results_list = [None]*len(file_list)
    for i, path in enumerate(file_list):

        # Get File name
        file_name = path.split(os.sep)[-1]
        file_name = file_name[:-7]

        # Get all name pieces
        pieces = file_name.split('_')

        # convert to numerical some values
        for k in [2,4,5,6,7,8]:
            pieces[k] = int(pieces[k])
            if k == 6:
                pieces[k] = pieces[k]/1e6

        # open results
        with open(path, "rb") as f:
            mdl_res = pickle.load(f)

        # append results
        for metric in metrics_list:
            pieces.append(mdl_res[metric])

        # final list
        results_list[i] = pieces

    # convert to DataFrame and swap two columns for convenience
    results_table = pd.DataFrame(results_list, columns= piece_list + metrics_list)
    results_table = column_switch( results_table, 'model', 'sampling_rate')
    results_table.sort_values(
        ['model', 'task','pipeline','inner_fold','outer_fold'],
        ascending=[True, True, True, True, True],
        inplace=True
    )

    # store if required
    if save:
        if filename is not None:
            if filename[:-3] == 'csv':
                results_table.to_csv(filename, index=False)
            else:
                results_table.to_csv(filename + '.csv', index=False)
        results_table.to_csv('ResultsTable.csv', index=False)
    return results_table


if __name__ == '__main__':

    help_d = """
    CreateResultsTable gathers the results stored in all pickle files into a
    unique Pandas DataFrame. The DataFrame is also stored in a csv file with
    default name 'ResultsTable.csv'. A custom name can be given in input as well.
    
    Example:
    
    $ python3 CreateResultsTable.csv -n CustomName.csv
    
    """
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-n",
        "--name",
        dest      = "filename",
        metavar   = "csv filename",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = None,
        help      = """
        The results filename. it should include the .csv extension at the end. 
        However, the function can handle its absence. 
        """,
    )

    # basically we overwrite the dataPath if something was given
    args = vars(parser.parse_args())
    FilenameInput = args['filename']
    gather_results(save = True, filename = FilenameInput)
    