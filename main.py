from predictor.classifier_prediction import ClassifierPredictor
from predictor.table_predictor import TablePredictor

import pandas as pd

"""
This script is made as an example of how to run the different predictors. 
Input has to be a list.

There is a TablePredictor and a ClassifierPredictor which should both be used for comparison.
"""

t = TablePredictor()

surnames = ['Ahimbisibwe', 'Auma', 'Amin', 'Makubuya', 'Museveni', 'Oculi', 'Kadaga']

table_predict = t.predict(surnames, n_jobs=10)
print(table_predict)


# Regular XGB Predictor
c = ClassifierPredictor()

predict_xgb = c.predict(surnames, 
              get_label_names=True, 
              predict_prob = True,
              df_out =True)

print(predict_xgb)

