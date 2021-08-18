import tensorflow as tf
import numpy as np
import pandas as pd

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), "\n")


img_dir = "/content/drive/MyDrive/Colab/plant-pathology/images/"
teste_dir = "/content/drive/MyDrive/Colab/plant-pathology/test.csv"
treino_dir = "/content/drive/MyDrive/Colab/plant-pathology/train.csv"

dados_teste = pd.read_csv(teste_dir)
dados_treino = pd.read_csv(treino_dir)
img_id_teste = dados_teste['image_id']

y_obj = dados_treino[['healthy', 'multiple_diseases', 'rust', 'scab']]

#Amostra para treino.
NUM_AMOSTRA = dados_treino.shape[0]
print("Treino: ", NUM_AMOSTRA)

#Executar na primeira execução para compactar as imagens.
from PIL import Image
from tqdm.notebook import tqdm
import gc

imagem_treino = np.empty((NUM_AMOSTRA,255,255,3))
imagem_teste = np.empty((dados_teste.shape[0],255,255,3))

for i in tqdm(range(NUM_AMOSTRA)):
  imagem_treino[i] = np.uint8(Image.open(img_dir + f'Train_{i}.jpg').resize((255, 255)))

print(imagem_treino.shape, '\n')
np.savez_compressed('/content/drive/MyDrive/Colab/plant-pathology/img_treino.npz', imagem_treino)
del imagem_treino
gc.collect()

for i in tqdm(range(dados_teste.shape[0])):
  imagem_teste[i] = np.uint8(Image.open(img_dir + f'Test_{i}.jpg').resize((255, 255)))

print(imagem_teste.shape, '\n')
np.savez_compressed('/content/drive/MyDrive/Colab/plant-pathology/img_teste.npz', imagem_teste)
del imagem_teste
gc.collect()