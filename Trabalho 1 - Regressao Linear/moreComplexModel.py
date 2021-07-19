def predict(x,theta):
    x=np.c_[np.ones(len(x)),x]
    return np.dot(x, theta)

def gradient_b(x, y, theta): 
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

def matrix_union(A, B):
    for a, b in zip(A, B):
        yield [*a, *b]

def linear_eq_grau2(data,alpha,epochs):
    error = []
    x = data[['carat','cut','color','clarity','x','y','z','depth','table']].values
    y = data['price'].values
    t = np.zeros((data.shape[1]*2-1,1))
    c = x*x
    x = list(matrix_union(x,c))
    for e in range(epochs):
        error.append(cost(x,y,t))
        gradient = gradient_b(x,y,t)
        t[0] = t[0] - (alpha*gradient[0])/len(y)
        for j in range(1,len(t)):
            t[j] = t[j] - (alpha*gradient[j])/len(y)
    return t, error, x

t,e,x=linear_eq_grau2(data,0.0001,10)
y = data['price'].values
y_pred = predict(x,t)
#print(e[1]," vs. ", e[-1])
plt.scatter(y,y_pred)