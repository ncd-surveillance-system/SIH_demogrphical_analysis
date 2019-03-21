# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

diabetes = pd.read_csv('diabetes_cleaned.csv')  # 4893 rows, 54 columns
# 765 positives, 4128 negatives

X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 5267x52
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 5267x1


sm = SMOTE(random_state=2)
X_new, y_new = sm.fit_sample(X, y.ravel())
# 8866 rows, 54 columns

c = diabetes.columns.tolist()

c.remove('HbA1c_category')
diabetes = pd.DataFrame(data=X_new, columns=c)
diabetes['HbA1c_category'] = y_new

diabetes.to_csv(r'diabetes_cleaned_balanced.csv', index=False)

# New dataset: 8866x54
