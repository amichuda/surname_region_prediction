# Surname-Region Prediction

This repository creates a package that predicts region of origin based on surname in Uganda.

The intended idea behind using this package will be take a columns of surnames and apply these functions to the names to get predicted regions of origin.

This package uses `sklearn` and `xgboost`.

## Changelog

03/29/2021:

- Added calibrated classifier

05/31/2020:

- Added classifier for agro-ecological zones that can be accessed with the `agro_eco` flag.

06/02/2020:

- Added ability to use pandas dataframe as input in to table and classifier predictors
- added environment file to recreate conda environment.

06/04/2020:

- made fuzzy matching in the table predictor optional (with the `fuzzy` flag)
  - `False` by default
- predictions output the input names

## Installation

To install the package, you can either install with pip:

```
pip install .
```

from the folder,

or install repository:

```
pip install git+https://github.com/amichuda/surname_region_prediction
```

## Quickstart

To recreate the conda environment, do the following:

```bash

cd /path/to/surname_region_prediction

conda env -f environment.yml
```

## Directory Structure

This outlines the directory structure of the project:

```
surname_region_prediction
├── environment.yml
├── LICENSE
├── main.py
├── predictor
│   ├── classifier_prediction.py
│   ├── exceptions.py
│   ├── __init__.py
│   ├── saved_models
│   │   ├── label_encoder.joblib
│   │   ├── label_encoder_multilabel_False_nokampala_True_agro_zone_smote_False_opt.joblib
│   │   ├── label_encoder_multilabel_False_nokampala_True_gaul_smote_False_gaul_opt.joblib
│   │   ├── label_encoder_multilabel_False_nokampala_True.joblib
│   │   ├── label_encoder_multilabel_True.joblib
│   │   ├── tfidf.joblib
│   │   ├── tfidf_multilabel_False_nokampala_True_agro_zone_smote_False_opt.joblib
│   │   ├── tfidf_multilabel_False_nokampala_True_gaul_smote_False_gaul_opt.joblib
│   │   ├── tfidf_multilabel_False_nokampala_True.joblib
│   │   ├── tfidf_multilabel_True.joblib
│   │   ├── xgb_None_calibrated_gaul_opt.joblib
│   │   ├── xgb_None.joblib
│   │   ├── xgb_None_multilabel_False_add_kampala_True_agro_zone_smote_False_opt.joblib
│   │   ├── xgb_None_multilabel_False_add_kampala_True_gaul_smote_False_gaul_opt.joblib
│   │   ├── xgb_None_multilabel_False_add_kampala_True.joblib
│   │   └── xgb_None_multilabel_True.joblib
│   ├── table
│   │   ├── agro_zone_predictor.csv
│   │   ├── gaul_predictor.csv
│   │   └── table_predictor.csv
│   └── table_predictor.py
├── README.md
├── requirements.txt
└── setup.py

```

## Example

To run the predictors, see `main.py` for a runnable example. A minimal example, using a pandas dataframe is shown below:

```python

from predictor.classifier_prediction import ClassifierPredictor
from predictor.table_predictor import TablePredictor

import pandas as pd

surnames = pd.DataFrame({'names':['Ahimbisibwe', 'Auma', 'Amin', 
                         'Makubuya', 'Museveni', 'Oculi', 'Kadaga']})

# %% Table Predictor
t = TablePredictor(column_name='names')

table_predict = t.predict(surnames, n_jobs=10)

# %% Classifier for Regions
c = ClassifierPredictor(column_name='names')

predict_xgb = c.predict(surnames, 
              get_label_names=True, 
              predict_prob = True,
              df_out =True)

# %% Agro-ecological Zones Classifier
cag = ClassifierPredictor(column_name = 'names', agro_eco=True)

predict_xgb_agro_eco = cag.predict(surnames, 
              get_label_names=True, 
              predict_prob = True,
              df_out =True)
              
```