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

diabetes = pd.read_csv('diabetes_cleaned_NonMedical.csv')


X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 4893x9
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 4893x1
# 765 positives, 4128 negatives

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37)
# X_train: 3669x9
# X_test: 1224x9
# Training positives: 575, negatives: 3094
# Testing positives:190, negatives: 1034

rf = RandomForestClassifier(n_estimators=1000, random_state=0)

rf.fit(X_train, y_train.ravel())

pickle.dump(rf, open('PredictDiabetes.pickle', 'wb'))

score = rf.score(X_test, y_test)

print(score)
# Avg accuracy: 84%
