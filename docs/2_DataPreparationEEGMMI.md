
This file provides an additional step-by-step instruction for preprocessing the EEGMMI dataset.

First of all, we need to create the correct MATLAB environment.

1.  Download the **BIDSAlign** library from the official GitHub repository. Here is the [GitHub link](https://github.com/MedMaxLab/BIDSAlign).
2.  Download **EEGLab** and all the required plugins. BIDSAlign provides a function called “[install_eeglab_from_scratch.m](https://github.com/MedMaxLab/BIDSAlign/blob/main/install_eeglab_from_scratch.m)" in case you want this step to be done automatically. Please remember to specify the correct installation path, which is usually inside the MATLAB folder.
3.  Download the **EEGMMI Dataset** from OpenNeuro at the following  [link](https://openneuro.org/datasets/ds004362/versions/1.0.0). There are several ways to do this. If you are familiar with Python, you can use the  [openneuro-py package](https://github.com/hoechenberger/openneuro-py).
4.  Git clone [this repository](https://github.com/MedMaxLab/eegprepro/tree/main).

These steps should set up the environment for preprocessing the EEGMMI dataset.
Now we need to add the path to BIDSAlign and EEGLab folders. Usually this can be done by:

1.   running `addpath bidsalign` and `addpath eeglab` in the Matlab console.
2.  running `bidsalign nogui` to automatically set up all other paths correctly.

To preprocess the EEGMMI dataset, we need to perform some additional steps.

1.  Copy paste the `preprocess_*` files from the [eegprepro/Preprocess](https://github.com/MedMaxLab/eegprepro/tree/main/Preprocessing) folder to the `bidsalign/__lib` folder.
2.  Set the `preprocess_all` script as described in the  [eegprepro/docs/2_DataPreparation.md](https://github.com/MedMaxLab/eegprepro/blob/main/docs/2_DataPreparation.md) file. Remember to set `dt_i` and `dt_f` to 0, `save_set` to true, and `task_totake` to `{{ 'run-4' 'run-8' ‘run-12’}}`. Remember to also disable the resampling step, as we do not want to do any upsampling. And of course add the path to the raw dataset.
3.  Each preprocessing pipeline has a set of steps that should be enabled or disabled by changing the booleans of the `prep_steps` field of the `params_info` struct variable. Depending on the pipeline, we suggest setting the `set_label` field of the `save_info` structure to `"_raw”, “_filt”, “_ica”, “_icasr”`. This is important for running the `MatlabToPickle` notebook.

Before running the `preprocess_all_script` file, we to align the dataset to the BIDS architecture.

1.  Run the command `create_dataset_architecture(‘/path/to/dataset’,’sess-01','eeg')`. Just change the path string and leave the rest as it is. NOTE that this will do inplace operations, so you may want to create a backup copy in case something goes wrong.

**NOTE**: Remember that BIDSAlign is designed specifically for preprocessing EEG datasets stored in BIDS format. It expects a specific directory structure to navigate through EEG files and retrieve necessary information. The required structure is `dataset_root —> subject X —> session X —> eeg —> eeg files`. So, if the function was called correctly, the EEGMMI dataset should now have a structure like `/Users/XXXXX/EEG/dataset/ds004362/sub-001/sess-01/eeg`.

Now you should be able to run the `preprocess_all_script` file and, hopefully, preprocess the entire dataset.

**NOTE**: in BIDSAlign, when the same information, such as the reference, can be retrieved from multiple sources, there is a specific priority order followed by the library. Warnings may be generated when something is unusual but can be handled by the library itself. You don't need to worry about it. We include these warnings to inform users about internal handlings, as these matters should not be resolved silently.

The last step consists in running the “[EEGMMI_splitter_LeftRight.m](https://github.com/MedMaxLab/eegprepro/blob/main/Preprocessing/EEGMMI_splitter_LeftRight.m)” file. You only have to change the paths and pipeline names at lines 72, 80, 85, 90.

If everything went smoothly, you will now have a Matlab file for each MI trial, excluding those of subjects 88, 92, 100, which we skipped for reasons described in the paper.
