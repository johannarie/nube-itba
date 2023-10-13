# nube-itba

Implementacion de modelo de incumplimiento de pago

Fuente de Dataset:https://www.kaggle.com/code/raimar/riesgo-criditicio/input

## Descripción de variables

- **BAD:** 1 = candidato con préstamo incumplido o con mora; 0 = candidato que paga su deuda y no tiene registro negativo
- **LOAN:** Monto de solicitud de préstamo
- **MORTDUE:** Monto adeudado de la hipoteca existente
- **VALUE:** Valor actual del bien o propiedad
- **REASON:** DebtCon = consolidación de la deuda; HomeImp = mejoras para el hogar
- **JOB:** Categorías ocupacionales o profesionales
- **YOJ:** Años en su trabajo actual
- **DEROG:** Número de informes derogados o cancelados importantes
- **DELINQ:** Número de líneas de crédito morosas
- **CLAGE:** Antigüedad de la línea de crédito más antigua en meses
- **NINQ:** Número de consultas crediticias recientes
- **CLNO:** Número de líneas de crédito

Para llevar a cabo este proceso, se utilizaron las siguientes herramientas y tecnologías:
- **Entorno de Desarrollo**: Jupyter Notebook- Anaconda
- **Infraestructura en la Nube**: AWS EC2
- **Framework para Desarrollo Web**: Flask
- **Acceso a la Máquina Remota**: PuTTY

![POC drawio (2)](https://github.com/johannarie/nube-itba/assets/75706210/8d6afe08-571f-49b4-a51c-bd4c96a32aaa)

Entrenamiento de modelo

Inicialmente, se realizó una verificación en busca de variables con más del 70% de valores nulos, y ninguna se encontró.
Posteriormente, se procedió a dividir el conjunto de datos en dos subconjuntos: uno de entrenamiento (80%) y otro de prueba (20%). Además, se categorizaron las variables en categóricas y numéricas.
En el caso de las variables categóricas, se calculó el coeficiente de correlación, lo que reveló una alta correlación entre el monto adeudado (MORTDUE) y el valor actual de la propiedad (VALUE). Por lo tanto, se decidió eliminar la variable MORTDUE.
Las variables categóricas "REASON" y "JOB" se transformaron en variables dummy. Luego, se realizaron imputaciones de valores nulos con ceros en columnas específicas, mientras que para las variables "LOAN" y "VALUE," se utilizó la imputación de la media.
Además, se optó por estandarizar algunas de las variables, y se exportó la serialización del proceso de estandarización. Todos estos pasos se aplicaron tanto al conjunto de entrenamiento como al de prueba.
Una vez preparado el conjunto de datos, se procedió a entrenar un modelo de regresión logística y calcular las predicciones en el conjunto de prueba, junto con sus respectivas métricas. Este modelo fue serializado y guardado en formato Pickle para su posterior uso.

Despliegue de modelo de aprendizaje automático con Flask en el servidor local (localhost).

El script "PredictionVF.py" se utiliza para realizar predicciones de incumplimiento de pago utilizando un modelo de regresión logística previamente entrenado.Para utilizar este diccionario, es necesario que estén los archivos 'modelo.pkl' y 'scaler_model.pkl'.
Al emplear la función "predict" y proporcionar una lista de muestras como argumento, se generará como resultado un diccionario que abarca la muestra original, la clase pronosticada (0 o 1), y las probabilidades.
 ![predict test](https://github.com/johannarie/nube-itba/assets/75706210/a4e5a070-ea1e-481e-83e5-9c77b6b14968)
 
El script "app-pd-final-json.py" es una aplicación web basada en Flask que proporciona un servicio para realizar predicciones de incumplimiento de pago utilizando un modelo previamente entrenado. Los usuarios pueden acceder a la aplicación a través de una URL y proporcionar los parámetros necesarios para obtener una predicción sobre si un solicitante de préstamo incumplirá o no con su pago, junto con la probabilidad asociada. La aplicación utiliza un modelo previamente entrenado para realizar estas predicciones y devuelve los resultados en formato JSON.
Para probar la aplicación antes de implementarla en EC2, se verificó su funcionamiento en el entorno local utilizando la terminal de Anaconda PowerShell y se realizó una prueba a través de una solicitud HTTP para asegurarse de que la aplicación funciona. El proceso de prueba se llevó a cabo de la siguiente manera:
1. Se cambió el directorio de trabajo actual a la ubicación donde se encontraban los scripts de la aplicación.
2. Se activó el entorno con Python 3.10 usando el siguiente comando:
   conda activate python-3.10
3. Luego, se ejecutó el script de la aplicación con el siguiente comando:
   python app-pd-final-json.py
4. Con la aplicación en funcionamiento, se procedió a probar su correcta operación utilizando una solicitud HTTP a través de la URL en el navegador. La solicitud de prueba se realizó utilizando la siguiente URL:
http://localhost:5000/pagador?LOAN=5000&VALUE=10000&YOJ=2&DEROG=0&DELINQ=0&CLAGE=30&NINQ=1&CLNO=10&DEBTINC=35&REASON:DebtCon=1&JOB:Mgr=0&JOB:Office=1&JOB:Other=0&JOB:ProfExe=0&JOB:Sales=0&JOB:Self=0
5. El resultado de la solicitud de prueba fue una respuesta de la aplicación en formato JSON, similar a la que se muestra en la siguiente imagen:
   
![predict localhost](https://github.com/johannarie/nube-itba/assets/75706210/7d86974a-4dd2-4aaa-89a4-7df6eedb22f0)


