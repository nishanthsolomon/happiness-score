import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt

train_data = pd.read_csv('./dataset/train.csv', delimiter = ',')

y = train_data['happiness_score']
x = train_data[['gdp','life_expectancy','freedom','generosity','corruption']]

mlp = MLPRegressor(hidden_layer_sizes=(1000), activation='relu', solver='lbfgs', max_iter=100, batch_size=10)
mlp.fit(x,y)

y_pred = mlp.predict(x)

rms = sqrt(mean_squared_error(y, y_pred))

print('Root Mean Square Error using multi layer perceptron = ' + str(rms))