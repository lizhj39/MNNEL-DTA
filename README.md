# MNNEL-DTA

## file list

- base_models: Functions and utility functions of the base learners.

- data_input: The dataset and the preprocessed pconsc4 file can be downloaded from https://drive.google.com/file/d/191QqrTDrcroRuEKYDeb1Cei2s4WLECCF/view?usp=sharing. The code for pconsc4 can be found at https://github.com/595693085/DGraphDTA.

- base_models_train.py：Code for training the base learner.

- meta_model_train.py：Code for training the meta learner.

- meta_model.py: Code of meta model.

- drug_screening.py：Code for application of screening drugs.

  

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
