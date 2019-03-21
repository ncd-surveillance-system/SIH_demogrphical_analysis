#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:27:03 2019

@author: rohit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('diabetes.csv')

print(diabetes.columns)

diabetes = diabetes.drop(['ID of participants', 'Year of birth', 'Month of birth',
                          'Day of birth', 'Year of data', 'Time of data', 'Fasting blood glucose',
                          'cardiograph', 'eyeground',
                          'Taking health care from doctor_category',
                          'PHN_category', 'Care-category', 'Weight changes from 20 yr-category',
                          'Alcohol amount-category'], axis=1)

diabetes = diabetes.astype(np.object)
diabetes.dropna(inplace=True, how='any')
diabetes = diabetes[(diabetes != ' ').all(axis=1)]

diabetes['HbA1c_category'].replace(1, 0, inplace=True)
diabetes['HbA1c_category'].replace([2, 3, 4], 1, inplace=True)

print(diabetes.shape)

diabetes.to_csv('diabetes_cleaned.csv', index=False)

# 5267x53
