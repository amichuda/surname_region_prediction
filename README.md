# uber_surname_region_prediction
This repository creates a package that predicts region of origin based on surname in Uganda.

The intended idea behind using this package will be take a columns of surnames and apply these functions to the names to get predicted regions of origin.

## Changelog

05/31/2020:
- Added classifier for agro-ecological zones that can be accessed with the `agro_eco` flag.
06/02/2020:
- Added ability to use pandas dataframe as input in to table and classifier predictors
- added environment file to recreate conda environment.

## Table of Contents

- [uber_surname_region_prediction](#uber_surname_region_prediction)
  - [Changelog](#changelog)
  - [Table of Contents](#table-of-contents)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
- [predictor package](#predictor-package)
  - [Submodules](#submodules)
  - [predictor.classifier_prediction module](#predictorclassifier_prediction-module)
    - [class predictor.classifier_prediction.ClassifierPredictor(tfidf_path=None, model_path=None, label_encoder_path=None, \*\*kwargs)](#class-predictorclassifier_predictionclassifierpredictortfidf_pathnone-model_pathnone-label_encoder_pathnone-kwargs)
      - [\_\_init\_\_(tfidf_path=None, model_path=None, label_encoder_path=None, \*\*kwargs)](#inittfidf_pathnone-model_pathnone-label_encoder_pathnone-kwargs)
      - [load_label_encoder()](#load_label_encoder)
      - [load_model()](#load_model)
      - [load_tfidf()](#load_tfidf)
      - [predict(text=None, get_label_names=False, predict_prob=False, df_out=False)](#predicttextnone-get_label_namesfalse-predict_probfalse-df_outfalse)
      - [process_text(text)](#process_texttext)
      - [transform_text(text=None)](#transform_texttextnone)
  - [predictor.exceptions module](#predictorexceptions-module)
    - [exception predictor.exceptions.NoTextException()](#exception-predictorexceptionsnotextexception)
  - [predictor.table_predictor module](#predictortable_predictor-module)
    - [class predictor.table_predictor.TablePredictor(table_path=None)](#class-predictortable_predictortablepredictortable_pathnone)
      - [\_\_init\_\_(table_path=None)](#inittable_pathnone)
      - [predict(text, n_jobs=1)](#predicttext-n_jobs1)
  - [Module contents](#module-contents)

## Directory Structure

This outlines the directory structure of the project

```
uber_surname_region_prediction
├─ main.py
├─ predictor
│  ├─ __init__.py
│  ├─ classifier_prediction.py
│  ├─ exceptions.py
│  ├─ saved_models
│  │  ├─ label_encoder.joblib
│  │  ├─ label_encoder_multilabel_False_nokampala_True.joblib
│  │  ├─ label_encoder_multilabel_True.joblib
│  │  ├─ tfidf.joblib
│  │  ├─ tfidf_multilabel_False_nokampala_True.joblib
│  │  ├─ tfidf_multilabel_True.joblib
│  │  ├─ xgb_None.joblib
│  │  ├─ xgb_None_multilabel_False_add_kampala_True.joblib
│  │  └─ xgb_None_multilabel_True.joblib
│  ├─ table
│  │  └─ table_predictor.csv
│  └─ table_predictor.py
└─ README.md
```

## Installation

To install the package, you can either install with pip:

```
pip install .
```

from the folder,

or install repository:

```
pip install git+https://github.com/amichuda/uber_surname_region_prediction
```

## Quickstart

To recreate the conda environment, do the following:

```bash

cd /path/to/uber_surname_region_prediction

conda env -f environment.yml
```

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

# predictor package

## Submodules

## predictor.classifier_prediction module


### class predictor.classifier_prediction.ClassifierPredictor(tfidf_path=None, model_path=None, label_encoder_path=None, \*\*kwargs)
Bases: `object`


#### \_\_init\_\_(tfidf_path=None, model_path=None, label_encoder_path=None, \*\*kwargs)
This class uses a pickled trained classifier to make predictions about surnames.
It loads the pickled processors (which include a tfidf transformer and label encoder),
as well as the classifier and then trainsforms input text and give prediction

Keyword Arguments:

    tfidf_path {str} – path to trained tfidf transformer (joblib object) (default: {None})
    model_path {str} – path to joblib pickle of trained model (default: {None})
    label_encoder_path {str} – path to label encoder pickled object (default: {None})


#### load_label_encoder()
Loads label encoder. See [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
for more information


#### load_model()
Loads pickled trained classifier. Current one is an XGBoost classifier.
See: [https://xgboost.readthedocs.io/en/latest/](https://xgboost.readthedocs.io/en/latest/) for more details.


#### load_tfidf()
Loads Tfidf transformer. See 
[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

for information on all functionality.


#### predict(text=None, get_label_names=False, predict_prob=False, df_out=False)
Predicts origin based on classifier

Keyword Arguments:

    text {list} – List of strings to be predicted (default: {None})
    get_label_names {bool} – Whether to output the label names after prediction (default: {False})
    predict_prob {bool} – whether to give probabilities of coming from each region (default: {False})
    df_out {bool} – whether to output a pandas dataframe (default: {False})

    ```python
    >>> from predictor.classifier_prediction import ClassifierPredictor
    >>> # Instantiate predictor
    >>> c = ClassifierPredictor()
    >>> # Predict
    >>> prediction = c.predict(['Auma'])
    >>> print(prediction)
    ```

Returns:

    pandas.DataFrame or list – An object that contains predictions from the model


#### process_text(text)
Pre-processes input text to make it ready for prediction.

Arguments:

    text {str or list-like object} – The input string or list of strings that are to be processed

Returns:

    list – Pre-processed list of strings


#### transform_text(text=None)
## predictor.exceptions module


### exception predictor.exceptions.NoTextException()
Bases: `Exception`

Raises error if text input is not given

## predictor.table_predictor module


### class predictor.table_predictor.TablePredictor(table_path=None)
Bases: `object`


#### \_\_init\_\_(table_path=None)
A class for predicting origin based on direct frequency of name occurrences

Keyword Arguments:

    table_path {str} – path to table with occurrence probabilities (default: {None})


#### predict(text, n_jobs=1)
Predicts region probability by first trying to find an exact match and then 
fuzzy matching for any names not matched.

Arguments:

    text {list} – List of strings to predict

Keyword Arguments:

    n_jobs {int} – The number of cores to use for fuzzy-matching. This
    only applies to fuzzy-matching. If all names are found by exact match, then
    multi-processing is not used at all. (default: {1})

Returns:

    pandas.DataFrame – DataFrame of predictions

## Module contents
