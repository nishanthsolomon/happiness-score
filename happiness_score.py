import configparser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

from linear_regression import LinearRegressionHappinessScore
from mlp import MLPHappinessScore


class HappinessScore():
    def __init__(self, config):
        data_config = config['data']
        mlp_config = config['mlp']

        train_data = pd.read_csv(data_config['train_data_path'], delimiter = data_config['delimiter'])
        test_data = pd.read_csv(data_config['test_data_path'], delimiter = data_config['delimiter'])

        output_label = data_config['output_label']
        feature_labels = data_config['feature_labels'].split(',')

        self.y_train = train_data[output_label]
        self.x_train = train_data[feature_labels]

        self.y_test = test_data[output_label]
        self.x_test = test_data[feature_labels]

        self.linear_regression = LinearRegressionHappinessScore()
        self.mlp = MLPHappinessScore(mlp_config)
    
    def run_linear_regression(self):
        self.linear_regression.train(self.x_train, self.y_train)
        y_predicted = self.linear_regression.predict(self.x_test)
        rmse = self.rmse(self.y_test, y_predicted)
        return rmse

    def run_mlp(self):
        self.mlp.train(self.x_train, self.y_train)
        y_predicted = self.mlp.predict(self.x_test)
        rmse = self.rmse(self.y_test, y_predicted)
        return rmse

    def rmse(self, y, y_predicted):
        self.plot_results(y_predicted)
        rms = sqrt(mean_squared_error(y, y_predicted))
        return rms

    def plot_results(self, y_predicted):
        X = np.array(self.x_test)
        Y = np.array(self.y_test)
        Y_predicted = np.array(y_predicted)

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], Y)
        ax.scatter(X[:, 0], X[:, 1], Y, color='r', label='Actual Happiness Score')
        ax.scatter(X[:, 0], X[:, 1], Y_predicted, color='g', label='Predicted Happiness Score')
        ax.set_xlabel('GDP')
        ax.set_ylabel('Life Expectancy')
        ax.set_zlabel('Happiness score')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./happiness_score.conf')

    happiness_score = HappinessScore(config)

    rmse_linear_regression = happiness_score.run_linear_regression()
    rmse_mlp = happiness_score.run_mlp()

    print('RMS error of linear regression model = ' + str(rmse_linear_regression))
    print('RMS error of mlp model = ' + str(rmse_mlp))
