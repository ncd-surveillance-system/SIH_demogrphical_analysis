#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:12:49 2019

@author: rohit
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import f1_score


diabetes = pd.read_csv('diabetes_cleaned_balanced_NonMedical.csv')


X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 8256x9
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 8256x1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37)
# X_train: 6192x9
# X_test: 2064x9

rf = RandomForestClassifier(n_estimators=1000, random_state=0)

rf.fit(X_train, y_train.ravel())

pickle.dump(rf, open('PredictDiabetes.pickle', 'wb'))


y_pred = rf.predict(X_test)

print(f1_score(y_test, y_pred))
# Avg accuracy: 82%
