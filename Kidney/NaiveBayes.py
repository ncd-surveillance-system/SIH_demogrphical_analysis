#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:12:49 2019

@author: rohit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

diabetes = pd.read_csv('kidney.csv')

X = np.array(diabetes.loc[:, diabetes.columns != 'Kidney'])  # 8256x54
y = np.array(diabetes.loc[:, diabetes.columns == 'Kidney'])  # 8256x1


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37)

nb = GaussianNB()
nb.fit(X_train, y_train.ravel())

score = nb.score(X_test, y_test)

print(score)
# Avg accuracy: 73.6%
