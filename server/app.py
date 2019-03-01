# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:45:49 2019

@author: Rohit Nagraj
"""

from flask import Flask, request
import pickle
import time

app = Flask(__name__)


@app.route("/diabetes", methods=["GET"])
def predictDiabetes():
    model = pickle.load(
        open('../Diabetes/NonMedical/PredictDiabetes.pickle', 'rb'))
    x = request.data.split()
    y = model.predict([list(map(float, x))])
    return str(y[0])


@app.route("/kidney", methods=["GET"])
def predictKidney():
    model = pickle.load(
        open('../Kidney/PredictKidney.pickle', 'rb'))
    x = request.data.split()
    y = model.predict([list(map(float, x))])
    return str(y[0])


if __name__ == '__main__':
    app.run(debug=True)
