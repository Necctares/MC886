def sklearn_SGD(data): 
  from sklearn.linear_model import SGDRegressor
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
 
  x = data[['carat','cut','color','clarity','x','y','z','depth','table']]
  y = data['price']
  model = make_pipeline(StandardScaler(), SGDRegressor(alpha=0.01,max_iter=5000, tol=1e-4))
  model.fit(x,y)
  predict = model.predict(x)
  return predict

t = sklearn_SGD(data)
plt.scatter(y,t)