#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:08:30 2019

@author: rohit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

diabetes = pd.read_csv('diabetes_balanced_area.csv')

#diabetes = diabetes.loc[:, diabetes.columns != 'Area']

X = np.array(diabetes.loc[:, diabetes.columns != 'Outcome'])  # 710x8
y = np.array(diabetes.loc[:, diabetes.columns == 'Outcome'])  # 710x1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)

rf = RandomForestClassifier(n_estimators=1000, random_state= 77)
rf.fit(X_train, y_train.ravel())
score = rf.score(X_test, y_test)
    
print(score)
# Avg accuracy: 87.0%