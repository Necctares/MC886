%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

t,e,x1=linear_eq_grau2(data,0.0001,10)

y = e
x = np.zeros(len(y))
for i in range(len(y)):
  x[i] = i+1

plt.xlabel('iteration')
plt.ylabel('error')

plt.plot(x, y, 'o', color='blue');