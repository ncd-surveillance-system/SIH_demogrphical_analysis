# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:31:11 2019

@author: Rohit Nagraj
"""

import numpy as np
import pandas as pd

diabetes = pd.read_csv('../Diabetes/diabetes_cleaned_balanced.csv')
dia = pd.read_csv(
    '../Diabetes/NonMedical/diabetes_cleaned_balanced_NonMedical.csv')
dia['Kidney'] = diabetes['Triglyceride']
dia['Kidney'] = dia['Kidney'] > 150
dia['Kidney'] = dia['Kidney'].astype(int)
dia.to_csv('Kidney.csv', index=False)
