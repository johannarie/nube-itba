#!/usr/bin/env python3
import numpy as np

import os
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json


def predict(sample: list) -> dict:
    model=pickle.load(open('modelos/modelo.pkl', 'rb'))
    print(f'Received sample: {sample}')
    column_order = ['LOAN', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'REASON:DebtCon', 'JOB:Mgr', 'JOB:Office','JOB:Other', 'JOB:ProfExe', 'JOB:Sales', 'JOB:Self']
    # Organizar 'sample' en el mismo orden que las columnas de entrenamiento
    sample_ordered = [sample[column_order.index(col)] for col in column_order]

    # Define las características a las que deseas aplicar el StandardScaler
    features_to_scale = ['LOAN', 'VALUE', 'CLAGE']

    # Define los índices de las variables categóricas en 'sample'
    categorical_indices = [column_order.index(col) for col in column_order if col not in features_to_scale]
    # Separate the categorical variables from the continuous variables
    categorical_variables = [sample[i] for i in categorical_indices]
    continuous_variables = [sample[i] for i in range(len(sample)) if i not in categorical_indices]

    # Standarizamos las variables continuas
    scaler=pickle.load(open('modelos/scaler_model.pkl', 'rb'))
    scaled_continuous_variables = scaler.transform([continuous_variables])

    categorical_variables_2d = np.array(categorical_variables).reshape(1, -1)

    # Contenamos las variables ya transformadas
    input_data = np.concatenate([categorical_variables_2d, scaled_continuous_variables], axis=1)
    # Define el nuevo orden de las columnas
    new_order = [ 13, 14, 0, 1, 2, 15, 3, 4, 5, 6, 7,8, 9, 10, 11,12]

    # Reorganiza las columnas en el nuevo orden
    input_data_fin = input_data[:, new_order]
    # Realizar la predicción de regresión logística
    predicted_probabilities = model.predict_proba(input_data_fin)[:, 1]

    # Se definio un umbral de 0.5 para predecir la clase
    predicted_class = 1 if predicted_probabilities >= 0.5 else 0

    predicted_result = {
        'sample': sample,
        'predicted_class': predicted_class,
        'predicted_probabilities': predicted_probabilities
    }

    return predicted_result

