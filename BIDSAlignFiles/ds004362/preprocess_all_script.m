
% Script: Preprocess EEG Datasets with BIDS Structure
% Description: Preprocesses multiple EEG datasets with a BIDS structure,
% applying various preprocessing steps and saving the preprocessed data.
%
% Note: This script requires EEGLAB to be installed and configured.
% You need to set appropriate paths and parameters before running the script.
% 
% Author: [Andrea Zanola, Federico Del Pup]
% Date: [25/01/2024]

%% Clear Workspace and Initialize
%clear
close all
clc

%% add path to bidsalign functions
[~] = evalc('bidsalign nogui');

%% Set General Variables
modality              = 'modename';         % choose ['local'|'server'|'modename']
use_parpool           = false;              % use parpool if available
dataset_info_filename = '/home/pnc/Desktop/DATASET_INFO.tsv'; % Set the name of the dataset info file 
verbose               = true;               % Set the verbosity level
current_path          = pwd;                % just to return here
solve_nogui           = false;              % leave it to false, if problems set to true
setting_name          = 'test_dev_';        % test_dev_ is a set ignored by GIT

%% Select Modality (preprocess all, single dataset or single file)
single_file  = false;
dataset_name = ['EEGMMI'];
raw_filename = ['']; 


%% Set selection parameters

% Create a struct to store selection information
% selection info tells which subjects to select from 
% each dataset based on given sets
subj = 1;
sess = 1;
selection_info = struct('sub_i',           [], ...
                        'sub_f',           [], ...
                        'ses_i',           [], ...
                        'ses_f',           [], ...
                        'obj_i',           [], ...
                        'obj_f',           [], ...
                        'select_subjects', true, ...
                        'label_name',      [] , ...
                        'label_value',     [] , ...
                        'subjects_totake', {{ }} , ...
                        'session_totake',  {{ }} , ...
                        'task_totake',     {{ '_run-4_', '_run-8_' '_run-12_' }} ...
                        );

%% Set save parameters

% Create a struct to store the save information
% save_info tells which data must be saved (or not)
% after preprocessing
save_info = struct('save_data',    true, ...
                   'save_data_as', 'matrix', ...
                   'save_set',     true, ...
                   'save_struct',  true, ...
                   'save_marker',  true , ...
                   'set_label',    '_ICASR'); % '_RAW' '_FILT' '_ICA' '_ICASR'

%% Set preprocessing parameters

% Set the parameters for preprocessing
% params_info tells which operations and configuration
% use when preprocessing an eeg file
params_info = struct('low_freq',           1 , ...                % filtering                              
                     'high_freq',          45, ...                % filtering
                     'sampling_rate',      250, ...               % resampling
                     'standard_ref',       'COMMON', ...          % standard ref
                     'interpol_method',    'spherical' , ...      % interpolation
                     'flatlineC',          5 , ...                % 1° ASR
                     'channelC',           0.8 , ...              % 1° ASR
                     'lineC',              4, ...                 % 1° ASR
                     'burstC',             20 , ...               % 2° ASR
                     'windowC',            0.25 , ...             % 2° ASR
                     'burstR',             'off' , ...            % 2° ASR
                     'th_reject',          250, ...               % amplitude uV
                     'ica_type',           'fastica' , ...        % ICA
                     'non_linearity',      'tanh' , ...           % ICA
                     'n_ica',              30 , ...               % ICA
                     'ic_rej_type',        'iclabel' , ...        % 'mara','iclabel'
                     'iclabel_thresholds', [0 0.1; ...            % brain
                                            0.9 1; ...            % Muscle
                                            0.9 1; ...            % eye
                                            0.9 1; ...            % heart
                                            0.9 1; ...            % line noise
                                            0.9 1; ...            % channel noise
                                            0.9 1] , ...          % other
                     'mara_threshold',     0.9 , ...              % MARA thresh
                     'dt_i',               0 , ...                % segment removal [s]
                     'dt_f',               0 , ...                % segment removal [s]
                     'prep_steps', ...                            % steps2do
                          struct('rmchannels' , true , ...
                                 'rmsegments' , true , ...
                                 'rmbaseline' , true , ...
                                 'resampling' , false , ...
                                 'filtering'  , true , ...
                                 'ICA'        , false , ...
                                 'ICrejection', true, ...
                                 'ASR'        , true, ...
                                 'rereference', true) ...
                                 );

%% Set Paths

% currently this file allows to switch between path 
% from two modality. Since storage space and
% permission are different between local and server
% (remote) platforms, the two configurations are 
% set to local and server. Feel free to change names 
% We encourage you to work using the process_all
% function, as it allow to use an indefinite and complete
% (meaning that other structs are involved) setting
% configurations

filePath = mfilename('fullpath');
filePath = filePath(1:length(filePath)-14);
path_info = set_path_info('silence_warn',true);

if strcmp(modality,'server')

    % set various paths
    path_info.datasets_path   = ''; %#ok
    path_info.output_path     = '';
    path_info.output_mat_path = '';
    path_info.output_csv_path = '';
    path_info.output_set_path = '';
    path_info.eeglab_path     = '';
    
    path_info.git_path  = filePath(1:length(filePath)-6);
    path_info.lib_path  = filePath; 
    addpath(path_info.lib_path);

elseif strcmp(modality,'local')
    
    % set various paths
    path_info.datasets_path   = ''; %#ok
    path_info.output_path     = '';
    path_info.output_mat_path = '';
    path_info.output_csv_path = '';
    path_info.output_set_path = '';
    path_info.eeglab_path     = '';
    
    path_info.git_path = filePath(1:length(filePath)-6);
    path_info.lib_path = filePath; 
elseif strcmp(modality, 'modename')

    % load a specific setting
    path_info = load_settings(setting_name, 'path');
else
    error(['ERROR: UNRECOGNAIZED MODALITY: ' modality]); %#ok
end

%% Launch preprocessing
EEG_data = preprocess_all(  dataset_info_filename, ...
                            'path_info',           path_info, ...
                            'preprocess_info',     params_info, ...
                            'selection_info',      selection_info, ...
                            'save_info',           save_info, ...
                            'setting_name',        setting_name, ...
                            'single_file',         single_file, ...
                            'single_dataset_name', dataset_name, ...
                            'single_file_name',    raw_filename, ...
                            'use_parpool',         use_parpool, ...
                            'solve_nogui',         solve_nogui, ...
                            'verbose',             verbose ...
                         );

cd(current_path)
