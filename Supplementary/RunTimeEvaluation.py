# ===========================
#  Section 1: package import
# ===========================
# This section includes all the packages to import. 
# To run this notebook, you must install in your environment. 
# They are: numpy, pandas, matplotlib, scipy, scikit-learn, pytorch, selfeeg
import time
import argparse
import glob
import os
import random
import pickle
import copy
import warnings
warnings.filterwarnings(
    "ignore", message = "Using padding='same'", category = UserWarning
)

# IMPORT STANDARD PACKAGES
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.io import loadmat
from scipy.stats import zscore

# IMPORT TORCH
import torch
from torchaudio import transforms
from torch.utils.data import DataLoader

# IMPORT SELFEEG 
import selfeeg
import selfeeg.models as zoo
import selfeeg.dataloading as dl

# IMPORT REPOSITORY FUNCTIONS
import sys
sys.path.append('..')
from AllFnc import split
from AllFnc.models import HybridNet
from AllFnc.training import (
    lossBinary, lossMulti, get_performances, GetLearningRate
)
from AllFnc.utilities import (
    restricted_float, positive_float, positive_int_nozero, positive_int, str2bool
)

def loadEEGmat(path, return_label = True):
    data = loadmat(path, simplify_cells = True)
    X = data['DATA_STRUCT']['data']
    X = X[:,::2]
    X = zscore(X,1) 
    Y = data['DATA_STRUCT']['label_group']
    if Y == 'A':
        Y = 2
    elif Y == 'C':
        Y = 0
    else:
        Y = 1
    if return_label:
        return X, Y
    else:
        return X

def loadEEGpickle(path: str, 
            return_label: bool=True, 
            downsample: bool=False,
            use_only_original: bool= False,
            eegsym_train: bool = False,
            apply_zscore: bool = True,
            onehot_label: bool = False
           ):
    # NOTE: files were converted in pickle with the 
    # MatlabToPickle Jupyter Notebook. 
    with open(path, 'rb') as eegfile:
        EEG = pickle.load(eegfile)

    # extract and adapt data to training setting
    x = EEG['data']

    # get the dataset ID to coordinate some operations
    data_id = int(path.split(os.sep)[-1].split('_')[0])
    
    # if 125 Hz take one sample every 2
    if downsample:
        if data_id == 25:
            pass
        else:
            x = x[:,::2]
    
    # if use original, interpolated channels are removed.
    # Check the dataset_info.json in each Summary folder file 
    # to know which channel was interpolated during the preprocessing
    if use_only_original:
        if data_id == 2:
            chan2dele = [34,44]
        elif data_id == 10:
            chan2dele = [ 1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 15, 
                         16, 17, 18, 19, 21, 23, 24, 26, 27, 29, 30, 
                         32, 33, 34, 36, 38, 40, 41, 42, 43, 44, 46, 
                         48, 50, 51, 52, 53, 54, 56, 58, 59]
        elif data_id == 20:
            chan2dele = [34, 44]
        elif data_id == 19:
            chan2dele = [28, 30]
        elif data_id == 25:
            chan2dele = []
        elif data_id == 7:
            chan2del = [19, 27, 29, 39, 54]
        else:
            chan2dele = [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 
                         23, 27, 29, 30, 32, 34, 36, 38, 40, 42, 44, 
                         46, 48, 50, 52, 54, 56, 58]
        x = np.delete(x, chan2dele, 0)
    
    # select the 8 channels used by EEGsym. The selection order is chosen
    # to correctly fit into the selfEEG implementation. Remember that EEGsym
    # is a deeper model compared to other ones, and that it was pretrained
    # with other 4 datasets. A single supervised stratey might produce
    # lower results due to the small amount of data used.
    if eegsym_train:
        if data_id == 25:
            # Channel order is:
            # F3 =  4,  C3 = 13,  P3 = 23,  Cz = 12,
            # Pz = 22,  P4 =  5,  C4 = 14,  F4 = 24.
            x = x[[4, 13, 23, 12, 22, 5, 14, 24],:]
        else:
            raise ValueError('EEGSym is designed for motorimagery tasks')

    # apply z-score on the channels.
    if apply_zscore:
        x = zscore(x,1)

    # GetEEGPartitionNumber doesn't want a label, so we need to add a function
    # to omit the label
    if return_label:
        y = EEG['label']
        return x, y
    else:
        return x

class EEGDataset2(torch.utils.data.Dataset):
    """
    Copy of selfeeg dataset with file load on each getitem call
    """

    def __init__(
        self,
        EEGlen: pd.DataFrame,
        EEGsplit: pd.DataFrame,
        EEGpartition_spec: list,
        mode: str = "train",
        supervised: bool = False,
        load_function: "function" = None,
        transform_function: "function" = None,
        label_function: "function" = None,
        optional_load_fun_args: list or dict = None,
        optional_transform_fun_args: list or dict = None,
        optional_label_fun_args: list or dict = None,
        label_on_load: bool = False,
        label_key: list = None,
        default_dtype=torch.float32,
    ):
        # Instantiate parent class
        super().__init__()

        # Check Partition specs
        self.freq = EEGpartition_spec[0]
        self.window = EEGpartition_spec[1]
        self.overlap = EEGpartition_spec[2]
        if (self.overlap < 0) or (self.overlap >= 1):
            raise ValueError("overlap must be a number in the interval [0,1)")
        if self.freq <= 0:
            raise ValueError("the EEG sampling rate cannot be negative")
        if self.window <= 0:
            raise ValueError("the time window cannot be negative")
        if (self.freq * self.window) != int(self.freq * self.window):
            raise ValueError("freq*window must give an integer number ")

        # Store all Input arguments
        self.default_dtype = default_dtype
        self.EEGsplit = EEGsplit
        self.EEGlen = EEGlen
        self.mode = mode
        self.supervised = supervised

        self.load_function = load_function
        self.optional_load_fun_args = optional_load_fun_args
        self.transform_function = transform_function
        self.optional_transform_fun_args = optional_transform_fun_args
        self.label_function = label_function
        self.optional_label_fun_args = optional_label_fun_args

        self.label_on_load = label_on_load
        self.given_label_keys = None
        self.curr_key = None
        if label_key is not None:
            self.given_label_keys = label_key if isinstance(label_key, list) else [label_key]
            self.curr_key = self.given_label_keys[0] if len(self.given_label_keys) == 1 else None

        # Check if the dataset is for train test or validation
        # and extract relative file names
        if mode.lower() == "train":
            FileNames = EEGsplit.loc[EEGsplit["split_set"] == 0, "file_name"].values
        elif mode.lower() == "validation":
            FileNames = EEGsplit.loc[EEGsplit["split_set"] == 1, "file_name"].values
        else:
            FileNames = EEGsplit.loc[EEGsplit["split_set"] == 2, "file_name"].values

        # initialize attributes for __len__ and __getItem__
        self.EEGlenTrain = EEGlen.loc[EEGlen["file_name"].isin(FileNames)].reset_index()
        self.EEGlenTrain = self.EEGlenTrain.drop(columns="index")
        self.DatasetSize = self.EEGlenTrain["N_samples"].sum()

        # initialize other attributes for __getItem__
        self.Nsample = int(EEGpartition_spec[0] * EEGpartition_spec[1])
        self.EEGcumlen = np.cumsum(self.EEGlenTrain["N_samples"].values)

        # Set Current EEG loaded attributes (speed up getItem method)
        # Keep in mind that multiple workers use copy of the dataset
        # saving a copy of the current loaded EEG file can use lots of memory
        # if EEGs are pretty large
        self.currEEG = None
        self.dimEEG = 0
        self.dimEEGprod = None
        self.file_path = None
        self.minIdx = -1
        self.maxIdx = -1
        self.label_info = None
        self.label_info_keys = None

        # Set attributes for lazy load. In this case the entire dataset
        # will be pre-loaded and stored in the Dataset class
        self.is_preloaded = False
        self.x_preload = None
        self.y_preload = None

    def __len__(self):
        """
        :meta private:

        """
        return self.DatasetSize

    def __getitem__(self, index):
        nameIdx = np.searchsorted(self.EEGcumlen, index, side="right")
        self.file_path = self.EEGlenTrain.iloc[nameIdx].full_path

        # load file according to given setting (custom load or not)
        if self.load_function is not None:
            if isinstance(self.optional_load_fun_args, list):
                EEG = self.load_function(self.file_path, *self.optional_load_fun_args)
            elif isinstance(self.optional_load_fun_args, dict):
                EEG = self.load_function(self.file_path, **self.optional_load_fun_args)
            else:
                EEG = self.load_function(self.file_path)
            if self.label_on_load:
                self.currEEG = EEG[0]
                if self.supervised:
                    self.label_info = EEG[1]
                    if self.given_label_keys is not None:
                        self.label_info_keys = self.label_info.keys()
                        if (self.given_label_keys is not None) and (
                            len(self.given_label_keys) > 1
                        ):
                            self.curr_key = list(
                                set(self.label_info_keys).intersection(self.given_label_keys)
                            )[0]
                        self.label = self.label_info[self.curr_key]
                    else:
                        self.label = EEG[1]
            else:
                self.currEEG = EEG
        else:
            # load things considering files coming from the auto-BIDS library
            EEG = loadmat(self.file_path, simplify_cells=True)
            self.currEEG = EEG["DATA_STRUCT"]["data"]
            if (self.supervised) and (self.label_on_load):
                self.label_info = EEG["DATA_STRUCT"]["subj_info"]
                self.label_info_keys = self.label_info.keys()
                if (self.given_label_keys is not None) and (len(self.given_label_keys) > 1):
                    self.curr_key = list(
                        set(self.label_info_keys).intersection(self.given_label_keys)
                    )[0]
                self.label = self.label_info[self.curr_key]

            # transform data if transformation function is given
            if self.transform_function is not None:
                if isinstance(self.optional_transform_fun_args, list):
                    self.currEEG = self.transform_function(
                        self.currEEG, *self.optional_transform_fun_args
                    )
                elif isinstance(self.optional_transform_fun_args, dict):
                    self.currEEG = self.transform_function(
                        self.currEEG, **self.optional_transform_fun_args
                    )
                else:
                    self.currEEG = self.transform_function(self.currEEG)

            # convert loaded eeg to torch tensor of specific dtype
            if isinstance(self.currEEG, np.ndarray):
                self.currEEG = torch.from_numpy(self.currEEG)
            if self.currEEG.dtype != self.default_dtype:
                self.currEEG = self.currEEG.to(dtype=self.default_dtype)

            # store dimensionality of EEG files (some datasets stored 3D tensors, unfortunately)
            # This might be helpful for partition selection of multiple EEG in a single file
            self.dimEEG = len(self.currEEG.shape)
            if self.dimEEG > 2:
                self.dimEEGprod = (self.EEGlenTrain.iloc[nameIdx].N_samples) / np.cumprod(
                    self.currEEG.shape[:-2]
                )
                self.dimEEGprod = self.dimEEGprod.astype(int)

            # change minimum and maximum index according to new loaded file
            self.minIdx = 0 if nameIdx == 0 else self.EEGcumlen[nameIdx - 1]
            self.maxIdx = self.EEGcumlen[nameIdx] - 1

        # Calculate start and end of the partition
        # Manage the multidimensional EEG
        # NOTE: using the if add lines but avoid making useless operation in case of 2D tensors
        partition = index - self.minIdx
        first_dims_idx = [0] * (self.dimEEG - 2)
        if self.dimEEG > 2:
            cumidx = 0
            for i in range(self.dimEEG - 2):
                first_dims_idx[i] = (partition - cumidx) // self.dimEEGprod[i]
                cumidx += first_dims_idx[i] * self.dimEEGprod[i]
            start = (self.Nsample - round(self.Nsample * self.overlap)) * (partition - cumidx)
            end = start + self.Nsample
            if end > self.currEEG.shape[-1]:  # in case of partial ending samples
                sample = self.currEEG[
                    (
                        *first_dims_idx,
                        slice(None),
                        slice(self.currEEG.shape[-1] - Nsample, self.currEEG.shape[-1]),
                    )
                ]
            else:
                sample = self.currEEG[(*first_dims_idx, slice(None), slice(start, end))]
        else:
            start = (self.Nsample - round(self.Nsample * self.overlap)) * (partition)
            end = start + self.Nsample
            if end > self.currEEG.shape[-1]:  # in case of partial ending samples
                sample = self.currEEG[..., -self.Nsample :]
            else:
                sample = self.currEEG[..., start:end]

        # extract label if training is supervised (fine-tuning purposes)
        if self.supervised:
            if self.label_on_load:
                label = self.label
            else:
                if isinstance(self.optional_label_fun_args, list):
                    label = self.label_function(
                        self.file_path,
                        [*first_dims_idx, start, end],
                        *self.optional_label_fun_args
                    )
                elif isinstance(self.optional_label_fun_args, dict):
                    label = self.label_function(
                        self.file_path,
                        [*first_dims_idx, start, end],
                        **self.optional_label_fun_args,
                    )
                else:
                    label = self.label_function(self.file_path, [*first_dims_idx, start, end])
            return sample, label
        else:
            return sample

if __name__ == '__main__':
    start_time = time.time()
    # ===========================
    #  Section 2: set parameters
    # ===========================
    # In this section all tunable parameters are instantiated. The entire training 
    # pipeline is configured here, from the task definition to the model evaluation.
    # Other code cells compute their operations using the given configuration. 
    
    help_d = """
    RunTimeEvaluation run a single training of the Alzheimer task with
    different batch construction modalities. This is to evaluate how 
    an optimized batch construction can save hours of total runtime
    
    Example:
    
    $ python3 RunTimeEvaluation -m 1
    
    """
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-m",
        "--mode",
        dest      = "mode",
        metavar   = "mode",
        type      = int,
        nargs     = '?',
        required  = False,
        default   = 1,
        choices   =[1, 2, 3, 4, 5, 6],
        help      = """
        The loading modality to evaluate. It can be one of the following:
        1) load an eeg and extract the specific sample every time using mat files;
        2) load an eeg and extract the specific sample every time using pickle files;
        3) use selfeeg dataset with mat files
        4) use selfeeg dataset with pickle files
        5) use selfeeg dataset and preload everything on cpu
        6) use selfeeg dataset and preload everything on gpu
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest      = "verbose",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
        help      = """
        Set the verbosity level of the whole script. If True, information about
        the choosen split, and the training progression will be displayed
        """
    )
    args = vars(parser.parse_args())

    evalMode       = args['mode']
    if evalMode in [1,3]:
        eegpath   = '/data/zanola/_mat_preprocessed/'
        loadEEG   = loadEEGmat
    else:
        eegpath   = '/data/delpup/eegpickle/filt'
        loadEEG   = loadEEGpickle
    pipelineToEval = 'filt'
    taskToEval     = 'alzheimer'
    modelToEval    = 'shallownet'
    outerFold      = 1
    innerFold      = 1
    downsample     = True
    z_score        = True
    rem_interp     = False
    batchsize      = 64
    overlap        = 0.0
    workers        = 0
    verbose        = args['verbose']
    lr             = 0.0
    
    # ==================================
    #  Section 3: create partition list
    # ==================================
    
    part_a = split.create_nested_kfold_subject_split([i for i in range(1,37)], 10, 5)
    part_c = split.create_nested_kfold_subject_split([i for i in range(37,66)], 10, 5)
    part_f = split.create_nested_kfold_subject_split([i for i in range(66,89)], 10, 5)
    partition_list_1 = split.merge_partition_lists(part_a, part_c, 10, 5)
    partition_list = split.merge_partition_lists(partition_list_1, part_f, 10, 5)
    
    # ======================================
    # Section 4: set the training parameters
    # =====================================
    
    # Define the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the number of Channels to use. 
    # Basically 61 due to BIDSAlign channel system alignment.
    Chan = 61
    
    # Define the sampling rate. 125
    freq = 125
    
    # Define the partition window length, in second. 4s
    window = 4.0
    
    # Define the number of classes to predict.
    nb_classes = 3
    
    # For selfEEG's models instantiation
    Samples = int(freq*window)
    
    # Set the Dataset ID for glob.glob operation in SelfEEG's GetEEGPartitionNumber().
    # It is a single number for every task except for PD that merges two datasets
    datasetID = '10'
    
    # Set the class label in case of plot of functions
    class_labels = ['CTL', 'FTD', 'AD']
    
    # =====================================================
    #  Section 5: Define pytorch's Datasets and dataloaders
    # =====================================================
    
    # Now that everything is ready, let's define the pytorch's Datasets and dataloaders. 
    # The dataset is defined by using the selfEEG EEGDataset custom class, 
    # which includes an option to preload the entire dataset.
    
    # GetEEGPartitionNumber doesn't need the labels
    if evalMode in [1,3]:
        loadEEG_args = {'return_label': False}
        glob_input = ['ds004504*_FILT/10_*']
    else:
        glob_input = [datasetID + '_*.pickle']
        loadEEG_args = {
            'return_label': False, 
            'downsample': downsample, 
            'use_only_original': rem_interp,
            'eegsym_train': True if modelToEval.casefold() == 'eegsym' else False,
            'apply_zscore': z_score
        }
    
    # calculate dataset length.
    # Basically it automatically retrieves all the partitions 
    # that can be extracted from each EEG signal
    EEGlen = dl.get_eeg_partition_number(
        eegpath,
        freq,
        window,
        overlap, 
        file_format = glob_input,
        load_function = loadEEG,
        optional_load_fun_args = loadEEG_args,
        includePartial = False if overlap == 0 else True,
        verbose = verbose
    )
    
    # Now we also need to load the labels
    loadEEG_args['return_label'] = True
    
    # Set functions to retrieve dataset, subject, and session from each filename.
    # They will be used by GetEEGSplitTable to perform a subject based split
    dataset_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[0])
    subject_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[1]) 
    session_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[2]) 
    
    # fold to eval is the correct index to get the desired train/val/test partition
    foldToEval = outerFold*5 + innerFold
    
    train_id   = partition_list[foldToEval][0]
    val_id     = partition_list[foldToEval][1]
    test_id    = partition_list[foldToEval][2]
    EEGsplit= dl.get_eeg_split_table(
        partition_table      = EEGlen,
        exclude_data_id      = None,
        val_data_id          = val_id,
        test_data_id         = test_id, 
        split_tolerance      = 0.001,
        dataset_id_extractor = subject_id_ex,
        subject_id_extractor = session_id_ex,
        perseverance         = 10000
    )
    
    if verbose:
        print(' ')
        print('Subjects used for test')
        print(test_id)
    
    
    # Define Datasets and preload all data
    if evalMode in [1,3]:
        print('using files in mat format')
    else:
        print('using files in pickle format')
    
    if evalMode in [1,2]:
        print('Batch will be created loading the entire EEG every time')
        trainset = EEGDataset2(
            EEGlen, EEGsplit, [freq, window, overlap], 'train', 
            supervised             = True, 
            label_on_load          = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )
        valset = EEGDataset2(
            EEGlen, EEGsplit, [freq, window, overlap], 'validation',
            supervised             = True, 
            label_on_load          = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )
        testset = EEGDataset2(
            EEGlen, EEGsplit, [freq, window, overlap], 'test',
            supervised             = True,
            label_on_load          = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )

    else:
        print('Batch will be created using selfeeg dataset')
        trainset = dl.EEGDataset(
            EEGlen, EEGsplit, [freq, window, overlap], 'train', 
            supervised             = True, 
            label_on_load          = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )
        
        valset = dl.EEGDataset(
            EEGlen, EEGsplit, [freq, window, overlap], 'validation',
            supervised             = True, 
            label_on_load          = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )
        
        testset = dl.EEGDataset(
            EEGlen, EEGsplit, [freq, window, overlap], 'test',
            supervised             = True,
            label_on_load          = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )

    if evalMode in [5,6]:
        print('Batch will be created using selfeeg dataset and preload on CPU')
        trainset.preload_dataset()
        valset.preload_dataset()
        testset.preload_dataset()
        trainset.y_preload = trainset.y_preload.to(dtype = torch.long)
        valset.y_preload   = valset.y_preload.to(dtype = torch.long)
        testset.y_preload  = testset.y_preload.to(dtype = torch.long)

        if evalMode == 6:
            print('Batch will be created using selfeeg dataset and preload on GPU')
            trainset.x_preload = trainset.x_preload.to(device=device)
            trainset.y_preload = trainset.y_preload.to(device=device)
            valset.x_preload = valset.x_preload.to(device=device)
            valset.y_preload = valset.y_preload.to(device=device)
            testset.x_preload = testset.x_preload.to(device=device)
            testset.y_preload = testset.y_preload.to(device=device)
        
    
    # Finally, Define Dataloaders
    # (no need to use more workers in validation and test dataloaders)
    if evalMode in [1,2]:
        trainloader = DataLoader(dataset = trainset, batch_size = batchsize,
                                 shuffle = True, num_workers = workers)
        valloader = DataLoader(dataset = valset, batch_size = batchsize,
                               shuffle = False, num_workers = workers)
        testloader = DataLoader(dataset = testset, batch_size = batchsize,
                                shuffle = False, num_workers = workers)
    elif evalMode in [3,4]:
        trainsampler = dl.EEGSampler(trainset, workers)
        valsampler = dl.EEGSampler(valset, workers, 0)
        testsampler = dl.EEGSampler(testset, workers, 0)
        trainloader = DataLoader(dataset = trainset, batch_size = batchsize,
                                 num_workers = workers)
        valloader = DataLoader(dataset = valset, batch_size = batchsize,
                               num_workers = workers)
        testloader = DataLoader(dataset = testset, batch_size = batchsize,
                                num_workers = workers)
    else:
        trainloader = DataLoader(dataset = trainset, batch_size = batchsize,
                                 shuffle = True, num_workers = 0)
        valloader = DataLoader(dataset = valset, batch_size = batchsize,
                               shuffle = False, num_workers = 0)
        testloader = DataLoader(dataset = testset, batch_size = batchsize,
                                shuffle = False, num_workers = 0)
    
    if verbose and evalMode in [2,4,5,6]:
        # plot split statistics
        labels = np.zeros(len(EEGlen))
        for i in range(len(EEGlen)):
            path = EEGlen.iloc[i,0]
            with open(path, 'rb') as eegfile:
                EEG = pickle.load(eegfile)
            labels[i] = EEG['label']
        dl.check_split(EEGlen, EEGsplit, labels)
    
    # ==================================================
    #  Section 6: define the loss, model, and optimizer
    # ==================================================
    
    # cross entropy if alzheimer vs frontotemporal vs control
    lossFnc = lossMulti
    
    
    # SET SEEDS FOR REPRODUCIBILITY
    # why this seed? It's MedMax in ASCII!
    seed = 83136297
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    
    
    Mdl = zoo.ShallowNet(nb_classes, Chan, Samples)
    Mdl.to(device = device)
    Mdl.train()
    if verbose:
        print(' ')
        ParamTab = selfeeg.utils.count_parameters(Mdl, False, True, True)
        print(' ')
    
    
    if lr == 0:
        lr = GetLearningRate(modelToEval, taskToEval)
        if verbose:
            print(' ')
            print('used learning rate', lr)
    gamma = 0.995
    optimizer = torch.optim.Adam(Mdl.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    
    # Define selfEEG's EarlyStopper with large patience to act as a model checkpoint
    earlystop = selfeeg.ssl.EarlyStopping(
        patience = 125, 
        min_delta = 1e-05, 
        record_best_weights = True
    )
    
    # =============================
    #  Section 7: train the model
    # =============================
    start_time_training = time.time()
    print('training started after', start_time_training - start_time)
    loss_summary=selfeeg.ssl.fine_tune(
            model                 = Mdl,
            train_dataloader      = trainloader,
            epochs                = 1,
            optimizer             = optimizer,
            loss_func             = lossFnc, 
            lr_scheduler          = scheduler,
            EarlyStopper          = earlystop,
            validation_dataloader = valloader,
            verbose               = verbose,
            device                = device,
            return_loss_info      = True
        )
    print('training concluded after', time.time() - start_time_training)
    
    # ===============================
    #  Section 8: evaluate the model
    # ===============================
    Mdl.eval()
    earlystop.restore_best_weights(Mdl)
    Mdl.to(device=device)
    scores = get_performances(loader2eval    = testloader, 
                              Model          = Mdl, 
                              device         = device,
                              nb_classes     = nb_classes,
                              return_scores  = True,
                              verbose        = verbose,
                              plot_confusion = False,
                              class_labels   = class_labels
                             )
    
    # ==================================
    #  Section 9: Save model and metrics
    # ==================================
    
    # Set the output path
    if taskToEval.casefold() == 'eyes':
        start_piece_mdl = 'EoecClassification/Models/'
        start_piece_res = 'EoecClassification/Results/'
        task_piece = 'eye'
    elif taskToEval.casefold() == 'alzheimer':
        start_piece_mdl = 'AlzClassification/Models/'
        start_piece_res = 'AlzClassification/Results/'
        task_piece = 'alz'
    elif taskToEval.casefold() == 'motorimagery':
        start_piece_mdl = 'MIClassification/Models/'
        start_piece_res = 'MIClassification/Results/'
        task_piece = 'mmi'
    elif taskToEval.casefold() == 'parkinson':
        start_piece_mdl = 'PDClassification/Models/'
        start_piece_res = 'PDClassification/Results/'
        task_piece = 'pds'
    elif taskToEval.casefold() == 'sleep':
        start_piece_mdl = 'SleepClassification/Models/'
        start_piece_res = 'SleepClassification/Results/'
        task_piece = 'slp'
    elif taskToEval.casefold() == 'psychosis':
        start_piece_mdl = 'FEPClassification/Models/'
        start_piece_res = 'FEPClassification/Results/'
        task_piece = 'fep'
    
    if modelToEval.casefold() == 'eegnet':
        mdl_piece = 'egn'
    elif modelToEval.casefold() == 'shallownet':
        mdl_piece = 'shn'
    elif modelToEval.casefold() == 'deepconvnet':
        mdl_piece = 'dcn'
    elif modelToEval.casefold() == 'eegsym':
        mdl_piece = 'egs'
    elif modelToEval.casefold() == 'atcnet':
        mdl_piece = 'atc'
    elif modelToEval.casefold() == 'hybridnet':
        mdl_piece = 'hyb'
    elif modelToEval.casefold() == 'resnet':
        mdl_piece = 'res'
    else:
        mdl_piece = 'fbc'
    
    if pipelineToEval.casefold() == 'raw':
        pipe_piece = 'raw'
    elif pipelineToEval.casefold() == 'filt':
        pipe_piece = 'flt'
    elif pipelineToEval.casefold() == 'ica':
        pipe_piece = 'ica'
    elif pipelineToEval.casefold() == 'icasr':
        pipe_piece = 'isr'
    
    if downsample:
        freq_piece = '125'
    else:
        if taskToEval.casefold() == 'motorimagery':
            freq_piece = '160'
        else:
            freq_piece = '250'
    
    out_piece = str(outerFold+1).zfill(3)
    in_piece = str(innerFold+1).zfill(3)
    lr_piece = str(int(lr*1e6)).zfill(6)
    chan_piece = str(Chan).zfill(3)
    win_piece = str(round(window)).zfill(3)
    
    file_name = '_'.join(
        [task_piece, pipe_piece, freq_piece, mdl_piece, 
         out_piece, in_piece, lr_piece, chan_piece, win_piece]
    )
    model_path = start_piece_mdl + file_name + '.pt'
    results_path = start_piece_res + file_name + '.pickle'
    
    if verbose:
        print('saving model and results in the following paths')
        print(model_path)
        print(results_path)
    
    print('run complete in:')
    print(time.time() - start_time)
