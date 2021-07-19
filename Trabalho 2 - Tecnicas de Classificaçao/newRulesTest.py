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

#Nesterov Acelerated Gradient
def gdnesterov(x,y,peso_hl,peso_saida,bias_hl,bias_saida,epocas,learning_rate, gama=0.9):
  
  custo = np.array([])

  for i in range(epocas):

    #Randomiza as entradas x e y
    rand_seed = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rand_seed)
    np.random.shuffle(y)
    treino = zip(x,y)

    velocidade = np.array([np.zeros(bias_saida.shape),np.zeros(bias_hl.shape),np.zeros(peso_saida.shape),np.zeros(peso_hl.shape)])

    for t in treino:
      n_bs = bias_saida - gama*velocidade[0]
      n_bhl = bias_hl - gama*velocidade[1]
      n_ps = peso_saida - gama*velocidade[2]
      n_phl = peso_hl - gama*velocidade[3]
      
      dbs, dbhl, dps, dphl, custo_temp = backprop(t[0],t[1],n_phl,n_ps,n_bhl,n_bs)
      custo = np.append(custo,custo_temp)

      velocidade[0] = gama*velocidade[0] + learning_rate*dbs
      bias_saida -= velocidade[0]

      velocidade[1] = gama*velocidade[1] + learning_rate*dbhl
      bias_hl -= velocidade[1]

      velocidade[2] = gama*velocidade[2] + learning_rate*dps
      peso_saida -= velocidade[2]

      velocidade[3] = gama*velocidade[3] + learning_rate*dphl
      peso_hl -= velocidade[3]

  return bias_saida, bias_hl, peso_saida, peso_hl, custo

#Treino
bs,bhl,ps,phl,custo_nesterov = gdnesterov(x,y,peso_hl,peso_saida,bias_hl,bias_saida,5,1e-2)
%store custo_nesterov
#Teste
_, saida_nesterov = forwardprop(x_teste,phl,ps, bhl, bs)
%store saida_nesterov
print('Acertos com os dados de teste:', acertosneur(saida_nesterov,y_teste), '%')

custo_plt_1 = np.array([])
j = 0
for i in range(0,300000,6000):
  custo_plt_1 = np.append(custo_plt_1, custo[i])
  j += 1

custo_plt_2 = np.array([])
j = 0
for i in range(0,300000,6000):
  custo_plt_2 = np.append(custo_plt_2, custo[i])
  j += 1

fig, grap = plt.subplots()
plt.xticks([0,20,40,60,80,100],['0','2','4','6','8','10'])
grap.plot(custo_plt_1, label='Custo SGD')
grap.plot(custo_plt_2, 'ro', markersize='3', label='Custo Nesterov')
leg = grap.legend()