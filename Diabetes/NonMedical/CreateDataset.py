# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:25:32 2019

@author: Rohit Nagraj
"""

import pandas as pd
df = pd.read_csv('../diabetes_cleaned.csv')

df = df[['Weight',
         'BMI',
         'Taking medication for hypertenstion',
         'Exercise more than 30 minutes-category',
         'Waist circumference (cm)',
         'Improve lifestyle habits',
         'Walking or physical activity-category',
         'Quick walking-category',
         'Age',
         'HbA1c_category']]
df.to_csv('diabetes_cleaned_NonMedical.csv', index=False)

df = pd.read_csv('../diabetes_cleaned_balanced.csv')

df = df[['Weight',
         'BMI',
         'Taking medication for hypertenstion',
         'Exercise more than 30 minutes-category',
         'Waist circumference (cm)',
         'Improve lifestyle habits',
         'Walking or physical activity-category',
         'Quick walking-category',
         'Age',
         'HbA1c_category']]
df.to_csv('diabetes_cleaned_balanced_NonMedical.csv', index=False)
