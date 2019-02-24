# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pylab as pl

df = pd.read_csv('diabetes.csv')
df = df[df.Outcome > 0]
df = df[df.Glucose > 0]
df = df[df.BloodPressure > 0]
df = df[df.SkinThickness > 0]
df = df[df.BMI > 0]
df = df[df.DiabetesPedigreeFunction > 0]
df = df[df.Age > 0]
df = df.reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != 'Outcome'], df['Outcome'], stratify=df['Outcome'], random_state=66)

diabetes = df

X = np.array(diabetes.drop(['Outcome'], axis=1).astype(float))
y = np.array(diabetes['Outcome'])
kmeans = KMeans(n_clusters=4, max_iter=600, algorithm='auto')

kmeans.fit(X)
correct = 0
pred = []
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)
    pred.append("a"+str(prediction[0]))

diabetes['area'] = pred

df.to_csv(r'Output.csv')
