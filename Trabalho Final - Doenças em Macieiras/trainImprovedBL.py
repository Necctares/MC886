#Treinamento

early_stopping = tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('modelo.hdf5', save_best_only=True)
reducao_reg = tf.keras.callbacks.ReduceLROnPlateau(patience=20)

modelo.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
data_aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,vertical_flip=True)
resultado = modelo.fit(data_aug.flow(x_treino,y_treino,batch_size=100),epochs=400,steps_per_epoch=x_treino.shape[0]//100, verbose=0,callbacks=[early_stopping, checkpoint, reducao_reg],validation_data=(x_teste,y_teste))
modelo = tf.keras.models.load_model('modelo.hdf5')
print('Feito.\n')

#Visualizando os resultados do treinamento.
from matplotlib import pyplot as plt

his = resultado.history

plt.figure(1,figsize=(20,10))
plt.subplot(122)
plt.ylabel('precisão')
plt.xlabel('epocas')
plt.plot(his[f'acc'], label='Treino')
plt.plot(his[f'val_acc'], label='Validação')
plt.legend()
plt.show()

#Erro por epoca.
plt.figure(1,figsize=(20,10))
plt.subplot(122)
plt.xlabel('epocas')
plt.ylabel('loss')
plt.plot(his['loss'], label='treino')
plt.plot(his['val_loss'], label='validação')
plt.legend()
plt.show()

#Matriz de confusão + Teste ROC AUC.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import matplotlib.colors

predicao = modelo.predict(x_teste)

y_pred = np.array([])
for i in predicao:
  y_pred = np.append(y_pred,np.argmax(i))

y_verdade = np.array([])
for i in y_teste:
  y_verdade = np.append(y_verdade,np.argmax(i))

acertos = 0
for i in range(y_verdade.shape[0]):
  if y_verdade[i] == y_pred[i]:
    acertos += 1
print('Acertos:', ((acertos/y_verdade.shape[0])*100), '%.\n')

acertos = 0
for i in range(4):
    score = roc_auc_score(y_teste[:, i], predicao[:, i])
    acertos += score
    print(f'{score:.2f}')

acertos /= 4
print(f'Acertos total (Media):{acertos:.2f}')

c_m = confusion_matrix(y_verdade, y_pred)

labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

figura = plt.figure(figsize=(10,9))
axis = figura.add_subplot(111)
caxis = axis.matshow(c_m)
plt.title('Matriz de Confusão: CNN')
figura.colorbar(caxis)
plt.xticks([0,1,2,3],labels)
plt.yticks([0,1,2,3],labels)
plt.xlabel('Predição')
plt.ylabel('Esperado')
plt.show()