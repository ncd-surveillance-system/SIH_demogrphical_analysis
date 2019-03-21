# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:26:13 2019

@author: Rohit Nagraj
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

diabetes = pd.read_csv('diabetes_cleaned_balanced.csv')
diabetes.drop('HbA1c', axis=1, inplace=True)

X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 8866x51
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 8866x1
  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37)
# X_train: 6649x51
# X_test: 2217x51

mlp = MLPClassifier(hidden_layer_sizes=(10))
mlp.fit(X_train, y_train.ravel())

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
y_pred = mlp.predict(X_test)
score = mlp.score(X_test, y_test)
print("Score: ", score)
print("Precision: ",precision_score(y_test, y_pred))
print("recall: ",recall_score(y_test, y_pred))
print("F1: ",f1_score(y_test, y_pred))
# F-score: 35%
