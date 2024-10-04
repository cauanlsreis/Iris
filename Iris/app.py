from flask import Flask, render_template, request, url_for
import torch
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

#carregar o modelo
modelo = torch.jit.load('./models/model_scripted.pt')
modelo.eval()

@app.route('/predict', methods=['POST'])
def predict():
    sepal_w = float(request.form['sepal_l'])
    sepal_l = float(request.form['sepal_w'])
    petal_w = float(request.form['petal_l'])
    petal_l = float(request.form['petal_w'])
    data = np.array([sepal_l, sepal_w, petal_l, petal_w])
    data_tensor = torch.FloatTensor(data)
    y_pred = modelo(data_tensor)
    result = torch.round(y_pred, decimals=2).tolist()
    return render_template('resultado.html', resultados=result)
    
