# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:33:57 2019

@author: Rohit Nagraj
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

diabetes = pd.read_csv('diabetes_cleaned_balanced.csv')
diabetes.drop('HbA1c', axis=1, inplace=True)

X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 8256x52
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 8256x1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37)
# X_train: 6192x52
# X_test: 2064x52

xgb = XGBClassifier()
xgb.fit(X_train, y_train.ravel())

score = xgb.score(X_test, y_test)

print(score)
# Avg accuracy: 92.3%