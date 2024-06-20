# How important is EEG preprocessing in deep learning applications?

This is the official repository for the research paper 

    The more, the better? Evaluating the role of
    EEG preprocessing for deep learning applications.

Submitted to IEEE Transactions on Neural Systems and Rehabilitation Engineering.

In this work, we have investigated the effect of EEG
data preprocessing on the performances of deep learning models.
In particular, we have evaluated if raw data can be
effectively given to DL models without losing predictive
power or not.
Furthermore, we have compared pipelines with different
levels of complexity, from a minimal filtering to a richer
one with established artifact handling automated algorithms.

## How was the comparison designed

The paper describes in the detail the experimental methodology. 
Markdown files in the docs folder provide additional information
on the provided code.
Here, we report a brief description of the key points.

### models and tasks

We used six different tasks, covering a wide range of possible
clinical and non-clinical use cases,
and four different deep learning architectures.

Tasks:
* **Eye**: physiological classification of eyes open and eyes
  closed recordings.
* **MMI**: motor movement imagery, famous BCI application
  largely studied in the domain.
* **Parkinson**, **FEP**, **Alzheimer**: two and three classes
  pathology classification focused on relevant medical
  use-cases, such as Parkinson’s, Psychosis and Alzheimer’s
  diseases.
* **Sleep**: normal sleep vs sleep deprivation recognition.

Models:
* **EEGNet**
* **DeepConvNet**
* **ShallowNet**
* **FBCNet**

### model evaluation

Data were splitted using a proposed variant of the
Leave-N-Sujects-Out Cross Validation, called 
Nested-Leave-N-Subjects-Out Cross Validation, schematized in
the figure below.
Each model was evaluated with different metrics and results
were used to run a statistical analysis to assess differences
between the investigated pipelines

<img src="Images/NestedKfold4.pdf"
        alt="Picture"
        width="600"
        style="display: block; margin: 0 auto" />

### statistical analysis

We searched for differences in the pipelines both at the local
(single model single task) and the model level (single model
all tasks).
Specific stattistical tests were used for each level, 
using the balanced accuracy as the evaluation metric and the
median value as centrality measure.
Results were presented with dedicated figures. 
An example is provided below

<img src="Images/shn.pdf"
        alt="Picture"
        width="500"
        style="display: block; margin: 0 auto" />
<img src="Images/shn_CD.pdf"
        alt="Picture"
        width="600"
        style="display: block; margin: 0 auto" />


## Provided code

All the code used to produce the results presented in the paper
are in this repository. Additional instruction to how replicate
our experimental pipeline is provided in the ``docs`` folder.

## Authors and Citation

If you find codes and results useful for your research,
please concider citing our work. It would help us to continue our research.
We are working on a research paper to submit to
IEEE Transactions on Neural Systems and Rehabilitation Engineering.  


Contributors:

- Eng. Federico Del Pup
- M.Sc. Andrea Zanola
- Prof. Manfredo Atzori

## License

The code is released under the MIT License