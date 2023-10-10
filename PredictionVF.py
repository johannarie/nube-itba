#!/usr/bin/env python3
import numpy as np

import os
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json

from config import *
import aws_utils as au

""" Fetch trained models, encoders and tokenizers. Make predictions. """
def fetch_pickle(bucket_name: str, folder: str, file_name: str):
    print(f'Loading {file_name} from local')    
    with open(os.path.join(folder, file_name), 'rb') as f:
        fetched_object = pickle.load(f)

    return fetched_object    

#def get_model_and_encoders():
 #   print('Fetching binaries')
  #  normalizer = fetch_pickle(BUCKET_NAME, FOLDER, 'normalizer.pkl')
  #  encoder = fetch_pickle(BUCKET_NAME, FOLDER, 'encoder.pkl')
  #  model = fetch_pickle(BUCKET_NAME, FOLDER, 'model.pkl')

   # return normalizer, encoder, model

###Prueba FINAL:FUNCIONAAAA
#from sklearn.preprocessing import StandardScaler
#import numpy as np

def predict(sample: list) -> dict:
    """
    'sample': List of feature values, including categorical and continuous variables
    """
    model =fetch_pickle(BUCKET_NAME, FOLDER, 'modelo.pkl')

    print(f'Received sample: {sample}')
    column_order = ['LOAN', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'REASON:DebtCon', 'JOB:Mgr', 'JOB:Office','JOB:Other', 'JOB:ProfExe', 'JOB:Sales', 'JOB:Self']
    # Organizamos 'sample' en el mismo orden que las columnas de entrenamiento
    sample_ordered = [sample[column_order.index(col)] for col in column_order]

    # Definimos las variables a las que aplicar el StandardScaler
    features_to_scale = ['LOAN', 'VALUE', 'CLAGE']

    # Definimos los índices de las variables categóricas en 'sample'
    categorical_indices = [column_order.index(col) for col in column_order if col not in features_to_scale]
    # Separacion entre categoricas y numericas
    categorical_variables = [sample[i] for i in categorical_indices]
    continuous_variables = [sample[i] for i in range(len(sample)) if i not in categorical_indices]

    # Aplicamos standar scaler a las numericas
    scaler =fetch_pickle(BUCKET_NAME, FOLDER, 'scaler_model.pkl')
    scaled_continuous_variables = scaler.transform([continuous_variables])

    categorical_variables_2d = np.array(categorical_variables).reshape(1, -1)

    # Unficamos categoricas y numericas
    input_data = np.concatenate([categorical_variables_2d, scaled_continuous_variables], axis=1)
    # Definimos el nuevo orden de las columnas
    new_order = [ 13, 14, 0, 1, 2, 15, 3, 4, 5, 6, 7,8, 9, 10, 11,12]

    # Reorganizamos las columnas en el nuevo orden
    input_data_fin = input_data[:, new_order]
    # Realizamos la predicción de regresión logística
    predicted_probabilities = model.predict_proba(input_data_fin)[:, 1]

    # Convertimos en clase
    predicted_class = 1 if predicted_probabilities >= 0.5 else 0

    predicted_result = {
        'sample': sample,
        'predicted_class': predicted_class,
        'predicted_probabilities': predicted_probabilities
    }

    return predicted_result

def get_deep_model():
    print('Loading deep model from disk')

  
    file_name = 'modelo.json'
    au.download_json_from_s3(BUCKET_NAME, FOLDER, file_name)    
    with open(os.path.join(FOLDER, file_name), 'r') as json_file:
        model = model_from_json(json_file.read())
    print('Model definition loaded from disk')

    
    file_name = 'modelo.h5'
    au.download_h5py_from_s3(BUCKET_NAME, FOLDER, file_name)    
    model.load_weights(os.path.join(FOLDER, file_name))
    print("Model weights loaded from disk")

    return model

