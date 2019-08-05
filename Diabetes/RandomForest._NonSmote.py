# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:11:29 2019

@author: Rohit Nagraj
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


diabetes = pd.read_csv('diabetes_cleaned.csv')
diabetes.drop('HbA1c', axis=1, inplace=True)

X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 8866x51
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 8866x1
  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37)
# X_train: 6649x51
# X_test: 2217x51

rf = RandomForestClassifier(n_estimators=100, random_state=0)

rf.fit(X_train, y_train.ravel())

y_pred = rf.predict(X_test)
score = rf.score(X_test, y_test)

print(f1_score(y_test, y_pred))

# Avg accuracy: 86%
# F-Score: 35%