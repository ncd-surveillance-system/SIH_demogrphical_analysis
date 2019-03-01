# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

diabetes = pd.read_csv('Kidney.csv')  # 4893 rows, 54 columns
# 765 positives, 4128 negatives

X = np.array(diabetes.loc[:, diabetes.columns != 'Kidney'])  # 4893x8
y = np.array(diabetes.loc[:, diabetes.columns == 'Kidney'])  # 4893x1


sm = SMOTE(random_state=2)
X_new, y_new = sm.fit_sample(X, y.ravel())
# 8256 rows, 54 columns

c = diabetes.columns.tolist()

c.remove('Kidney')
diabetes = pd.DataFrame(data=X_new, columns=c)
diabetes['Kidney'] = y_new

diabetes.to_csv(r'Kidney_balanced.csv', index=False)

# New dataset: 8256x54
