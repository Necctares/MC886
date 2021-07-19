#Carregando os dados após a primeira execução.
from sklearn.model_selection import train_test_split
import gc

data = np.load('/content/drive/MyDrive/Colab/plant-pathology/img_treino.npz')

for img in data:
  imagem_treino = data[img]

print('Treino:', imagem_treino.shape, '\n')

x_treino, x_teste, y_treino, y_teste = train_test_split(imagem_treino, y_obj.to_numpy(), test_size=0.2)

print(x_treino.shape, x_teste.shape, y_treino.shape, y_teste.shape, '\n')
del imagem_treino
gc.collect()

data = np.load('/content/drive/MyDrive/Colab/plant-pathology/img_teste.npz')

for img in data:
  imagem_teste = data[img]

print('Teste:', imagem_teste.shape, '\n')
del data
gc.collect()