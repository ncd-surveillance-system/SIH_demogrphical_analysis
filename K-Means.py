# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

diabetes = pd.read_csv('diabetes_balanced.csv')

X = np.array(diabetes.drop(['Outcome'], axis=1).astype(float))
y = np.array(diabetes['Outcome'])

kmeans = KMeans(n_clusters=4, max_iter=10000, algorithm='auto')

kmeans.fit(X)

pred = []
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)
    pred.append(float(prediction[0]))

diabetes['Area'] = pred

diabetes.to_csv('diabetes_balanced_area.csv', index=False)