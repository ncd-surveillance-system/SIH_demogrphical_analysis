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
from sklearn.feature_selection import RFE

diabetes = pd.read_csv('diabetes_cleaned_balanced.csv')
diabetes.drop('HbA1c', axis=1, inplace=True)

X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 8256x52
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 8256x1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37)
# X_train: 6192x52
# X_test: 2064x52

estimator = RandomForestClassifier(n_estimators=1000, random_state=0)

selector = RFE(estimator, 5)

selector = selector.fit(X_train, y_train.ravel())
print(selector.ranking_)
# Selected non-medical attributes: ['Weight', 'BMI', 'Taking medication for hypertenstion', 'Exercise
# more than 30 minutes-category', 'Waist circumference (cm)', 'Improve lifestyle habits',
# 'Walking or physical activity-category','Quick walking-category', 'Age', 'HbA1c_category'] (In order)
