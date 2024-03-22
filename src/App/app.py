from flask import Flask
from flask import request
from flask import render_template
import pandas as pd
import numpy as np
import pickle
import os

os.chdir(os.path.dirname(__file__))
print(os.getcwd())

with open('../model/extra_tree_model.pkl', 'rb') as f:
        model = pickle.load(f)

df = pd.read_csv("../data/processed/test_prepared.csv")
df = df.drop(['Unnamed: 0', 'price'], axis=1)
df = df.drop(df.index)
# Crea una fila de datos con valores cero
fila_cero = pd.Series([0] * len(df.columns), index=df.columns)

# Agrega la fila de datos al DataFrame
df.loc[0] = fila_cero

app = Flask(__name__)

def obtener_modelos_por_marca(dataframe):
    modelos_por_marca = {}

    marcas = dataframe['make'].unique()  # Obtener todas las marcas únicas en el DataFrame
    
    for marca in marcas:
        if type(marca) != float:
            modelos = []
            modelos_por_marca[marca] = []
            for modelo in dataframe[dataframe['make'] == marca]['model'].unique(): # Obtener todos los modelos únicos para la marca actual
                if type(modelo) != float:
                    modelos.append(modelo)

            modelos_por_marca[marca].append(sorted(modelos))
    
    modelos_por_marca_ordenado = dict(sorted(modelos_por_marca.items()))

    return modelos_por_marca_ordenado


@app.route('/', methods=['GET'])
def home():
    df = pd.read_csv("../data/processed/coches.csv")

    return render_template('index.html', modelos_por_marca=obtener_modelos_por_marca(df))

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.form

    diccionario = df.to_dict() 

    if 'vendedor_profesional' in request.form:
        vendedor_profesional = 1
    else:
        vendedor_profesional = 0

    if 'marca' in request.form:
        marca = str(datos['marca']).replace(' ', '_')
    else:
        return 'Te falta poner la marca'
    if 'modelo' in request.form:
        modelo = str(datos['modelo']).replace(' ', '_')
    else:
        return 'Te falta poner el modelo'
    if 'combustible' in request.form:
        combustible = int(datos['combustible'])
    else:
        return 'Te falta poner el combustible'
    if 'anio' in request.form and int(datos['anio']) < 2024 and int(datos['anio']) > 1900:
        anio = int(datos['anio'])
    else:
        return 'Te falta poner el año entre 1900 y 2024'
    if 'kms' in request.form and int(datos['kms']) > 0:
        kms = int(datos['kms'])
    else:
        return 'Te falta poner los kilómetros bien'
    if 'potencia' in request.form:
        potencia = int(datos['potencia'] and int(datos['potencia']) > 0)
    else:
        return 'Te falta poner la potencia bien'
    if 'puertas' in request.form and int(datos['puertas']) > 0:
        puertas = int(datos['puertas'])
    else:
        return 'Te falta poner las puertas'
    if 'cambio' in request.form:
        cambio = int(datos['cambio'])
    else:
        return 'Te falta poner el tipo de cambio'
    
    potencia = np.log(potencia+1)
    kms = np.log(kms+1)

    if f"make_{marca}" in diccionario:
        diccionario[f"make_{marca}"] = 1
        
    if f"model_{modelo}" in diccionario:
        diccionario[f"model_{modelo}"] = 1

    diccionario['fuel'] = combustible
    diccionario['year'] = anio
    diccionario['kms'] = kms
    diccionario['power'] = potencia
    diccionario['doors'] = puertas
    diccionario['shift'] = cambio
    diccionario['is_professional'] = vendedor_profesional

    # print(datos)
    resultado = np.exp(model.predict(pd.DataFrame(diccionario)))

    return 'Precio estimado: {} €'.format(round(resultado[0], 2))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)