#Mini Batch
def create_mini_batch(data,batch_size):
    m_b = []
    x = data[['carat','cut','color','clarity','x','y','z','depth','table']].values
    y = data['price'].values
    batch = np.c_[x,y]
    np.random.shuffle(batch)
    number_of_batches = batch.shape[0]//batch_size
    i=0
    for i in range(number_of_batches+1):
        mini_batch = batch[i * batch_size:(i + 1)*batch_size, :] 
        x_m = mini_batch[:, :-1]
        y_m = mini_batch[:, -1].reshape((-1, 1))
        m_b.append((x_m, y_m))
    if data.shape[0] % batch_size != 0: 
        mini_batch = batch[i * batch_size:data.shape[0]] 
        x_mini = mini_batch[:, :-1] 
        y_mini = mini_batch[:, -1].reshape((-1, 1)) 
        m_b.append((x_mini, y_mini)) 
    return m_b

def predict(x,theta):
    x=np.c_[np.ones(len(x)),x]
    return np.dot(x, theta)

def gradient_mb(x, y, theta): 
    h = predict(x, theta)
    gradient = [0]*len(theta)
    for i in range(len(y)):
        gradient[0] += h[i] - y[i]
    for k in range(len(theta)-1):
        for j in range(len(y)):
            gradient[k+1] += (h[j] - y[j])*x[j][k]
    return gradient

def cost(x, y, theta): 
    h = predict(x, theta)
    J = 0
    for i in range(len(y)):
        J += (h[i]-y[i])**2
    J /= 2*len(y)
    return J

def mini_batch_gd(data,alpha,epochs,batch_size):
    error = []
    t = np.zeros((data.shape[1],1))
    for i in range(epochs):
        mini_batches = create_mini_batch(data,batch_size)
        for batch in mini_batches:
            x,y = batch
            gradient = gradient_mb(x,y,t)
            if len(y) > 0:
              t[0] = t[0] - (alpha*gradient[0])/len(y)
              for j in range(1,len(t)):
                  t[j] = t[j] - (alpha*gradient[j])/len(y)
              error.append(cost(x,y,t))
    return t, error

#SGD

def predict_sgd(x,theta):
    x=np.r_[1,x]
    return np.dot(x,theta)

def gradient_sgd(x, y, theta): 
    h = predict_sgd(x, theta)
    gradient = [0]*len(theta)
    gradient[0] = (h - y)
    for k in range(len(theta)-1):
        gradient[k+1] += (h - y)*x[k]
    return gradient

def cost_sgd(x, y, theta): 
    h = predict_sgd(x, theta)
    J = (h-y)**2
    J /= 2
    return J

def stochastic_gd(data,alpha,epochs):
    data_shuff = data.values
    error = []
    t = np.zeros((data.shape[1],1))
    for i in range(epochs):
        np.random.shuffle(data_shuff)
        for example in data_shuff:
            x = example[:-1]
            y = example[-1]
            gradient = gradient_sgd(x,y,t)
            t[0] = t[0] - (alpha*gradient[0])
            for j in range(1,len(t)):
              t[j] = t[j] - (alpha*gradient[j])
            error.append(cost_sgd(x,y,t))
    return t, error

x = data_test[['carat','cut','color','clarity','x','y','z','depth','table']].values
y = data_test['price'].values
t, error = stochastic_gd(data,0.0001,20)
#t, error = mini_batch_gd(data,0.0001,20,500):
y_pred = predict(x,t)
plt.scatter(y,y_pred)