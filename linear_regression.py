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

linear_regression = LinearRegression()

linear_regression.fit(x,y)

y_pred = linear_regression.predict(x)

rms = sqrt(mean_squared_error(y, y_pred))
print('Root Mean Square Error using linear regression = ' + str(rms))