'''Dataset Information:

You should respect the following traininig/test split: 60,000 training examples, and 10,000 test examples. Avoid overfitting.
The data is available at: https://www.dropbox.com/s/qawunrav8ri0sp4/fashion-mnist-dataset.zip: 'train' folder (fashion-mnist_train.csv.zip) + 'test' folder (fashion-mnist_test.csv.zip)
Each row is a separate image. Column 1 is the class label. Remaining columns are pixel numbers (784 total). Each value is the darkness of the pixel (1 to 255). Dataset was converted to CSV with this script: https://pjreddie.com/projects/mnist-in-csv.'''

import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

dados_treino = pd.read_csv("/content/drive/My Drive/Colab/fashion-mnist_train.csv", sep=',')
dados_teste = pd.read_csv("/content/drive/My Drive/Colab/fashion-mnist_train.csv", sep=',')
dados_treino = dados_treino.to_numpy()
dados_teste = dados_teste.to_numpy()

#Normaliza os dados.
def normalizacao (data):
  data = data.astype(float)
  dados = data[:,1:]
  num_colunas = dados.shape[1]
  for i in range(num_colunas):
    coluna_norm = dados[:,i]
    media = coluna_norm.mean()
    max_col = coluna_norm.max()
    dados[:,i] = (dados[:,i]-media)/max_col
  data[:,1:]=dados
  return data

#Função para One-hot encoding.
def hot_encod(y, num_classes):
  aux = np.zeros((num_classes,1))
  aux[int(y)] = 1.0
  return aux.T.astype(int)

#Dados para o MLR
dados_treino = normalizacao(dados_treino)
dados_teste = normalizacao(dados_teste)

#Dados para a rede neural
#Dados de treino
y = [hot_encod(y,10) for y in dados_treino[:,0]]
x = dados_treino[:,1:]

#Dados de teste
y_teste = [hot_encod(yt,10) for yt in dados_teste[:,0]]
x_teste = dados_teste[:,1:]