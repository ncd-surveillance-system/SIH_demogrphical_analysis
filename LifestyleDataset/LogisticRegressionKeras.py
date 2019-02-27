import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.regularizers import L1L2

"""
Prepare you data, such as:
"""
diabetes = pd.read_csv('diabetes_balanced.csv')
diabetes.drop('HbA1c', axis=1, inplace=True)

#diabetes = diabetes.loc[:, diabetes.columns != 'Area']

X = np.array(diabetes.loc[:, diabetes.columns != 'HbA1c_category'])  # 710x8
y = np.array(diabetes.loc[:, diabetes.columns == 'HbA1c_category'])  # 710x1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)

"""
Set up the logistic regression model
"""
model = Sequential()
model.add(Dense(2,  # output dim is 2, one score per each class
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                input_dim=len(feature_vector)))  # input dimension = number of features your data has
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))