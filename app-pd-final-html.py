#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from flask import Flask, request, json, jsonify

import json
import os
import logging

import pickle

import PredictionVF as pr
app = Flask(__name__)


# Configuración de registro
logging.basicConfig(filename='app.log', level=logging.DEBUG)


@app.route("/")
def hello():
    return jsonify(message="Modelo de incumplimiento de pago")


@app.route('/pagador', methods=['GET'])
def pagador():
    # Obtener los parámetros de la solicitud
    loan = float(request.args.get('LOAN'))
    value = float(request.args.get('VALUE'))
    yoj = float(request.args.get('YOJ'))
    derog = float(request.args.get('DEROG'))
    delinq = float(request.args.get('DELINQ'))
    clage = float(request.args.get('CLAGE'))
    ninq = float(request.args.get('NINQ'))
    clno = float(request.args.get('CLNO'))
    debtinc = float(request.args.get('DEBTINC'))
    reason_debtcon = float(request.args.get('REASON:DebtCon'))
    job_mgr = float(request.args.get('JOB:Mgr'))
    job_office = float(request.args.get('JOB:Office'))
    job_other = float(request.args.get('JOB:Other'))
    job_profexe = float(request.args.get('JOB:ProfExe'))
    job_sales = float(request.args.get('JOB:Sales'))
    job_self = float(request.args.get('JOB:Self'))

    # Crear una lista con los valores
    sample = [loan, value, yoj, derog, delinq, clage, ninq, clno, debtinc, reason_debtcon, job_mgr, job_office, job_other, job_profexe, job_sales, job_self]
    
    # Realizar la predicción
    prediction = pr.predict(sample)
    # Obtener el valor de predicción
    predicted_value = prediction['predicted_class']

    # Obtener la probabilidad de incumplimiento
    probability_default = prediction['predicted_probabilities'][0]

    # Determinar si es un incumplimiento o no
    is_default = probability_default > 0.5
    result = "Default" if is_default else "No Default"
    
    # Formatear la respuesta en HTML
    response_html = f"""
    <h2>Predicted Value:</h2>
    <p>{predicted_value}</p>
    <h2>Input:</h2>
    <p>{sample}</p>
    <h2>Probability of Default:</h2>
    <p>{probability_default}</p>
    <h1>Result:</h1>
    <p style="font-size: 24px;">{result}</p>
    """

    return response_html
    
    #return f'Predicted Value: {predicted_value}\nInput:{sample}\nProbability of Default: {probability_default}\nResult: {result}'


    # Realizar la predicción
    #prediction = pr.predict(sample)

    # Convertir el resultado a una lista
    #prediction_list = list(prediction.values())
    #return str(prediction)
    #return jsonify(prediction_list)

if __name__ == '__main__':    
    app.run(host='0.0.0.0', debug=True) 

