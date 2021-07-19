#Multinomial Logistic Regression

num_classes = 10
num_features = 784
#Inicia aleatoriamente o peso e o bias para a função de predição.
peso = np.random.rand(num_classes, num_features)
bias = np.random.rand(num_classes, 1)

#Função para calcular porcentagem de acertos.
def acertos(predicao, data):
  acertos = 0
  for i in range(len(predicao)):
    if predicao[i] == data[i,0]:
      acertos += 1
  return ((acertos/len(predicao))*100)

#Função de predição.
def predlog(data,peso,bias):
  features = data[:,1:]
  logistica = np.array(np.empty([features.shape[0],peso.shape[0]]))
  for i in range(features.shape[0]):
    logistica[i] = (peso.dot(features[i].reshape(-1,1)) + bias).reshape(-1)
  return logistica

#Função para calcular a probabilidade.
def prob(logistica):
  prob = np.array(np.empty([logistica.shape[0],logistica.shape[1]]))
  for i in range(logistica.shape[0]):
    exponencial = np.exp(logistica[i])
    soma = np.sum(exponencial)
    prob[i] = exponencial/soma
  return prob

#Função para calcular Regressão Logistica.
def reglog(data,peso,bias):
  logistica = predlog(data,peso,bias)
  probabilidade = prob(logistica)
  resultado = np.empty(0)
  for i in probabilidade:
    resultado = np.append(resultado,np.argmax(i))
  return resultado.astype(int), probabilidade

#Função para calcular o custo.
def custoreglog(data, probabilidade):
  y = data[:,0].astype(int)
  tam = y.shape[0]
  log = -np.log(probabilidade[range(tam),y]+1e-9)
  custoreglog = np.sum(log) / tam
  return custoreglog

#Função para treino com SGD.
def sgdlog(data,peso,bias,epocas,learning_rate):
  y = data[:,0].astype(int)
  x = data[:,1:]
  custo = np.array([])
  for i in range(epocas):
    _ , prob = reglog(data,peso,bias)
    custo_temp = custoreglog(data,prob)
    custo = np.append(custo,custo_temp)
    prob[range(y.shape[0]),y] -= 1

    p_grad = prob.T.dot(x)
    b_grad = np.sum(prob,axis=0).reshape(-1,1)

    peso -= (learning_rate*p_grad)
    bias -= (learning_rate*b_grad)

  return custo, peso, bias 

#Treinamento
custo, peso_treinado, bias_treinado = sgdlog(dados_treino,peso,bias,100,1e-4)

#Resultado final
pred_final, prob_final = reglog(dados_teste,peso_treinado,bias_treinado)

acertos = acertos(pred_final, dados_teste)

print('Acerto usando os dados de teste: ', acertos, '%')
plt.xlabel('Epocas')
plt.ylabel('Custo')
plt.plot(custo)