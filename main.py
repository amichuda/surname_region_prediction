# %%
#! load_ext autoreload

# %%
#! autoreload 2
from predictor.classifier_prediction import ClassifierPredictor
from predictor.table_predictor import TablePredictor

import pandas as pd

"""
This script is made as an example of how to run the different predictors. 
Input has to be a list.

There is a TablePredictor and a ClassifierPredictor which should both be used for comparison.
"""

# %%
# Get some representative names
surnames = ['Ahimbisibwe', 'Auma', 'Amin', 'Makubuya', 'Museveni', 'Oculi', 'Kadaga']

# %%
t = TablePredictor()
table_predict = t.predict(surnames, n_jobs=10)
print(table_predict.to_markdown())

# %%
# Regular XGB Predictor
c = ClassifierPredictor()

predict_xgb = c.predict(surnames, 
              get_label_names=True, 
              predict_prob = True,
              df_out =True)

print(predict_xgb.to_markdown())

# %%
# Agro-Eological Zone Predictor
cag = ClassifierPredictor(agro_eco=True)

predict_xgb_agro_eco = cag.predict(surnames, 
              get_label_names=True, 
              predict_prob = True,
              df_out =True)

print(predict_xgb_agro_eco.to_markdown())

# %%
# Compare Regions and Agro-ecological zones

## First run without probability prediction
region_df = c.predict(surnames, 
              get_label_names=True, 
              predict_prob=True,
              df_out =True)

# Not predicting probabilities for agro_eco to make a smaller table
agro_eco_df = cag.predict(surnames, 
              get_label_names=True, 
              df_out =True)

comp_df = region_df.merge(agro_eco_df, 
                  left_index=True,
                  right_index = True)

print(comp_df.to_markdown())


