#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:12:49 2019

@author: rohit
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

diabetes = pd.read_csv('diabetes_balanced.csv')
diabetes.drop('HbA1c', axis=1, inplace=True)

X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 8256x52
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 8256x1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)
# X_train: 6192x52
# X_test: 2064x52

logisticRegr = LogisticRegression(solver = 'liblinear', max_iter=10000)

logisticRegr.fit(X_train, y_train.ravel())

score = logisticRegr.score(X_test, y_test)

print(score)

# Avg accuracy: 76%