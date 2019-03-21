# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:45:49 2019

@author: Rohit Nagraj
"""

from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import timedelta
from datetime import datetime
import io
import boto3
from io import StringIO

from config import access_key_id, secret_access_key

app = Flask(__name__)


@app.route("/diabetes", methods=["GET"])
def predictDiabetes():
    model = pickle.load(
        open('../Diabetes/NonMedical/PredictDiabetes.pickle', 'rb'))
    x = request.get_json()
    # d = {x:y for (x,y) in zip(df.columns.tolist()[:df.columns.size-1], np.array(df)[8][:df.columns.size-1])}
    y = model.predict([list(map(float, x.values()))])
    y = jsonify({'Answer': int(y[0])})
    return y


@app.route("/kidney", methods=["GET"])
def predictKidney():
    model = pickle.load(
        open('../Kidney/PredictKidney.pickle', 'rb'))
    x = request.get_json()
    # d = {x:y for (x,y) in zip(df.columns.tolist()[:df.columns.size-1], np.array(df)[1][:df.columns.size-1])}
    y = model.predict([list(map(float, x.values()))])
    y = jsonify({'Answer': int(y[0])})
    return y


@app.route("/database", methods=["POST"])
def pushData():
    global s3, airdata
    df2 = request.json
    model = pickle.load(
        open('../Diabetes/NonMedical/PredictDiabetes.pickle', 'rb'))
    y = model.predict([list(map(float, df2.values()))])
    df2['HbA1c_category'] = int(y[0])
    airdata = airdata.append(df2, ignore_index=True)

    csv_buffer = StringIO()

    airdata.to_csv(csv_buffer, index=False)
    response = s3.put_object(
        Body=csv_buffer.getvalue(),
        ContentType='application/vnd.ms-excel',
        Bucket="analysts-bucket",
        Key="non-medical.csv"
    )
    return "Success"


s3 = ''
airdata = ''


def connectDatabase():
    global s3, airdata
    s3 = boto3.client('s3',
                      aws_access_key_id=access_key_id,
                      aws_secret_access_key=secret_access_key
                      )
    obj = s3.get_object(Bucket='analysts-bucket', Key='non-medical.csv')
    airdata = pd.read_csv(io.BytesIO(obj['Body'].read()))


if __name__ == '__main__':
    connectDatabase()
    app.run(debug=True)


# Positive
{
    "Weight": "63.7",
    "BMI": "23.1",
    "Taking medication for hypertenstion": "0.0",
    "Exercise more than 30 minutes-category": "1.0",
    "Waist circumference (cm)": "88.0",
    "Improve lifestyle habits": "2.0",
    "Walking or physical activity-category": "2.0",
    "Quick walking-category": "2.0",
    "Age": "74.0"
}


# Positive
{
    "Weight": "45.4",
    "BMI": "20.6",
    "Taking medication for hypertenstion": "1.0",
    "Exercise more than 30 minutes-category": "2.0",
    "Waist circumference (cm)": "67.0",
    "Improve lifestyle habits": "2.0",
    "Walking or physical activity-category": "2.0",
    "Quick walking-category": "1.0",
    "Age": "71.0"
}

# Negative
{
    "Weight": "48.0",
    "BMI": "22.1",
    "Taking medication for hypertenstion": "0.0",
    "Exercise more than 30 minutes-category": "1.0",
    "Waist circumference (cm)": "77.0",
    "Improve lifestyle habits": "1.0",
    "Walking or physical activity-category": "1.0",
    "Quick walking-category": "2.0",
    "Age": "68.0"
}

# Negative
{
    "Weight": "47.5",
    "BMI": "21.8",
    "Taking medication for hypertenstion": "0.0",
    "Exercise more than 30 minutes-category": "1.0",
    "Waist circumference (cm)": "75.0",
    "Improve lifestyle habits": "1.0",
    "Walking or physical activity-category": "1.0",
    "Quick walking-category": "2.0",
    "Age": "69.0"
}
