import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv("diamonds-train.csv", na_values = ['no info', '.'])
data = data.replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [1, 2, 3, 4, 5])
data = data.replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [1, 2, 3, 4, 5, 6, 7, 8])
data = data.replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [1, 2, 3, 4, 5, 6, 7])
data_norm = data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

data_test = pd.read_csv("diamonds-test.csv", na_values = ['no info', '.'])
data_test = data_test.replace(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], [1, 2, 3, 4, 5])
data_test = data_test.replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [1, 2, 3, 4, 5, 6, 7, 8])
data_test = data_test.replace(['J', 'I', 'H', 'G', 'F', 'E', 'D'], [1, 2, 3, 4, 5, 6, 7])