def normal_eq(data):
  #Normal equation implementation
  x = data[['carat','cut','color','clarity','x','y','z','depth','table']].values
  y = data['price'].values
  x=np.c_[np.ones(len(y)),x]
  #Calculo do (x_transposto*x)^(-1)
  x_trans = np.transpose(x)
  x_trans_esc = x_trans.dot(x)
  #Calculo do (x_transposto*x)^(-1) * (x_transposto*y)
  c1 = np.linalg.inv(x_trans_esc)
  c2 = x_trans.dot(y)
  result = c1.dot(c2)
  return result

#Teste
result = normal_eq(data)
y_pred = predict(x,result)
plt.scatter(y,y_pred)