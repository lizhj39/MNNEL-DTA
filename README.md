# MNNEL-DTA

## file list

- base_models: Functions and utility functions of the base learners.

- data_input: The dataset and the preprocessed pconsc4 file can be downloaded from https://drive.google.com/file/d/191QqrTDrcroRuEKYDeb1Cei2s4WLECCF/view?usp=sharing. The code for pconsc4 can be found at https://github.com/595693085/DGraphDTA.

- base_models_train.py：This file instantiates three base learner classes, namely MGraphDTA, TFusionDTA, and NHGNN_DTA. By calling the fit function of the instance, one-click training of the model can be achieved.

- meta_model_train.py：This file instantiates the class of the meta-learner. By calling the fit function of the instance, one-click training of the model can be achieved; by calling the val function of the instance, the test MSE, CI value, and attention weight of the model for the data set can be returned.

- meta_model_pure.py: Class of Output Meta Learner, see the paper for details.
- meta_model.py: Class of Visual Meta Learner, see the paper for details.

- drug_screening.py：Input "training data set" and "data set to be predicted" (the latter defaults to the "AD" data set, that is, FDA-approved drug-target pairs), and output the predicted DTA value.

- heatmap.py: Heat map drawing of attention weights.

- joint_plot.py: Drawing of Figure 7 in the paper.


## run code

To train the MNNEL model, train all base learners first and then train meta learner:

```cmd
python base_models_train.py
python meta_model_train.py
```

You can also invoke the class of the meta model to test on the test set or make predictions, etc.

Running drug_screening can achieve drug screening.

```cmd
python drug_screening.py
```

**All codes need minor adjustments to run properly.**

## requirement


numpy
pandas
rdkit
torch
torch_geometric
