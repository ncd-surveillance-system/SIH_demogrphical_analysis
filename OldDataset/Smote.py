# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

diabetes = pd.read_csv('diabetes.csv')
diabetes = diabetes[(diabetes[['Glucose', 'BloodPressure',
                               'SkinThickness', 'BMI', 'Age']] != 0).all(axis=1)]
diabetes = diabetes.reset_index(drop=True)
# 177 positives, 335 negatives

X = np.array(diabetes.loc[:, diabetes.columns != 'Outcome'])  # 532x8
y = np.array(diabetes.loc[:, diabetes.columns == 'Outcome'])  # 532x1

# X: 532x8  y_train: 532x1
# Out of 532 training examples, 177 are positive, 355 are negative

sm = SMOTE(random_state=2)
X_new, y_new = sm.fit_sample(X, y.ravel())
# After smote, we get 710 samples in training sets, of which 355 are positive.

diabetes = pd.DataFrame(data=X_new, columns=diabetes.columns[0:8])
diabetes['Outcome'] = y_new

diabetes.to_csv(r'diabetes_balanced.csv', index=False)
