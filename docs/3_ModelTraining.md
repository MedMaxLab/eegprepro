# Model Training

Training all the 4800 models should be pretty simple but will require lot of time.
To simplify thing, all the results are uploaded in this repository
together with a summary table created with the ``CreateResultsTable.py`` file.
However, if you want to rerun all the training here is a description of the
scripts you need to run.

The entire process relies on two files:
the ``RunKfold.py`` and the ``RunKfoldAll.py``
(or its ``RunKfoldComboB0X.py`` variants for multiple GPUs) 
files.

Basically, ``RunKfold.py`` runs a single training with a specific
set of arguments. Here is the script help:

```
RunKfold run a single training with a specific split extracted from a nested k-fold subject split
(10 outer folds, 5 inner folds).
Many parameters can be set, which will be then used
to create a custom file name.
The only one required is the root dataset path.
Others have a default in case you want to check a
single demo run.
Example: $ python RunKfold -d /path/to/data

options:
  -h, --help            show this help message and exit
  -d datasets path, --datapath datasets path
        The dataset path. This is expected to be static
        across all trainings. dataPath must point to a
        directory which contains four subdirecotries,
        one with all the pickle files containing EEGs
        preprocessed with a specific pipeline.
        Subdirectoties are expected to have the following
        names, which are the same as the preprocessing
        pipelinea to evaluate:
            1) raw; 2) filt; 3) ica; 4) icasr

  -p [preprocessing pipeline], --pipeline [preprocessing pipeline]
        The pipeline to consider.
        It can be one of the following:
            1) raw; 2) filt; 3) ica; 4) icasr

  -t [task], --task [task]
        The task to evaluate.
        It can be one of the following:
            1) eyes; 2) alzheimer; 3)parkinson;
            4) motorimagery 5) sleep; 6) psychosis

  -m [model], --model [model]
        The model to evaluate.
        It can be one of the following:
            1) eegnet; 2) shallownet; 3) deepconvnet;
            4) resnet; 5) eegsym; 6) atcnet;
            7) hybridnet;

  -f [outer fold], --outer [outer fold]
        The outer fold to evaluate.
        It can be a number between 1 and 10.

  -i [inner fold], --inner [inner fold]
        The inner fold to evaluate.
        It can be a number between 1 and 5.

  -s [downsample], --downsample [downsample]
        A boolean that set if downsampling at 125 Hz
        should be applied or not.
        The presented analysis uses 125 Hz.

  -z [zscore], --zscore [zscore]
        A boolean that set if the z-score should
        be applied or not. The presented analysis
        applied the z-score, as different
        preprocessing pipelines produce EEGs that
        evolve on different range of values.

  -r [remove interpolated], --rminterp [remove interpolated]
        A boolean that set if the interpolated channels
        should be removed or not. BIDSAlign aligns all
        EEGs to a common 61 channel template based on
        the 10_10 International System.

  -b [batch size], --batch [batch size]
        Define the Batch size. It is suggested to use
        64 or 128. The experimental analysis
        was performed on batch 64.

  -o [windows overlap], --overlap [windows overlap]
        The overlap between time windows.
        Higher values means more samples but higher
        correlation between them.
        0.25 is a good trade-off.
        Must be a value in [0,1)

  -l [learning rate], --learningrate [learning rate]
        The learning rate.
        If left to its default (zero) a proper
        learning rate will be chosen depending on the
        model and task to evaluate.
        Optimal learning rates were identified by running
        multiple trainings with different set of values.
        Must be a positive value.

  -w [dataloader workers], --workers [dataloader workers]
        The number of workers to set for the dataloader.
        Datasets are preloaded for faster computation,
        so 0 is preferred due to known issues on values
        greater than 1 for some os, and to not increase
        too much the memory usage.

  -v [VERBOSE], --verbose [VERBOSE]
        Set the verbosity level of the whole script.
        If True, information about the choosen split,
        and the training progression will be displayed
```

``RunKfoldAll.py`` creates the grid of values to parse
to ``RunKfold.py`` and run all the training sequentially.
If you have multiple GPUs, you can duplicate this code
(or use the ``RunKfoldComboB0X.py`` files)
and divide the grid of possible arguments (for example,
split the outer fold numbers) to parallelize all the
training.

Both the results on the test set and the model weights will
be stored in the relative folders. Each task has its own
folder with name ``acronym+Classification`` (e.g.,
EoecClassification).