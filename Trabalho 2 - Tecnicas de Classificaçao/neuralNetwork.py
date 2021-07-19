num_hl = x.shape[1] #Numero de neuronios na camada escondida.

np.random.seed(3772)
peso_hl = np.random.rand(x.shape[1],num_hl)*((2.0/x.shape[1])**(0.5))
peso_saida = np.random.rand(num_hl,y[0].shape[1])*((2.0/x.shape[1])**(0.5))
bias_hl = np.random.rand(1)*((2.0/x.shape[1])**(0.5))
bias_saida = np.random.rand(1)*((2.0/x.shape[1])**(0.5))

#Funcao de acertos.
def acertosneur(predicao, y):
  acertos = 0
  for i,j in zip(predicao,y):
    if np.argmax(i) == np.argmax(j):
      acertos += 1
  return ((acertos/len(y))*100)

#Função de ativação.
def ativacao(camada):  
  return (1 / (1+np.exp(-camada)))

#Função derivada da ativação (Sigmoide).
def der_ativ(camada):
  return (ativacao(camada)*(1.0-ativacao(camada)))

#Função forward propagation.
def forwardprop(x,peso_hl,peso_saida, bias_hl, bias_saida):
  hidden_layer = np.dot(x,peso_hl) + bias_hl
  hidden_layer = ativacao(hidden_layer)
  saida = np.dot(hidden_layer,peso_saida) + bias_saida
  saida = ativacao(saida)
  return hidden_layer, saida

#Função back propagation.
def backprop(x,y,peso_hl,peso_saida,bias_hl,bias_saida):
  x = x.reshape((1,-1))
  hidden_layer, saida = forwardprop(x,peso_hl,peso_saida,bias_hl,bias_saida)  

  custo = (y-saida)**2
  der_bias_saida = 2*(y-saida)*der_ativ(saida)
  der_ps = hidden_layer.T.dot(der_bias_saida)
  der_bias_hl = der_bias_saida.dot(peso_saida.T)*der_ativ(hidden_layer)
  der_phl = x.T.dot(der_bias_hl)

  return der_bias_saida.mean(), der_bias_hl.mean(), der_ps, der_phl, custo.sum()

def sgdneur(x,y,peso_hl,peso_saida,bias_hl,bias_saida,epocas,learning_rate):
  
  custo = np.array([])

  for i in range(epocas):
    treino = zip(x,y)
    for t in treino:
      dbs, dbhl, dps, dphl, custo_temp = backprop(t[0],t[1],peso_hl,peso_saida,bias_hl,bias_saida)
      custo = np.append(custo,custo_temp)

      bias_saida -= learning_rate*dbs
      bias_hl -= learning_rate*dbhl
      peso_saida -= learning_rate*dps
      peso_hl -= learning_rate*dphl

  return bias_saida, bias_hl, peso_saida, peso_hl, custo

#Treino
bs,bhl,ps,phl,custo = sgdneur(x,y,peso_hl,peso_saida,bias_hl,bias_saida,10,1e-2)
%store custo
#Teste
_, saida = forwardprop(x_teste,phl,ps, bhl, bs)
%store saida
print('Acertos com os dados de teste:', acertosneur(saida,y_teste), '%')

custo_plt_1 = np.array([])
j = 0
for i in range(0,600000,6000):
  custo_plt_1 = np.append(custo_plt_1, custo[i])
  j += 1

plt.xticks([0,20,40,60,80,100],['0','2','4','6','8','10'])
plt.xlabel('Numero de Epocas')
plt.ylabel('Custo')
plt.plot(custo_plt_1)