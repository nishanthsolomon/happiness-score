import configparser

import pandas as pd

from sklearn.metrics import mean_squared_error
from math import sqrt

from linear_regression import LinearRegressionHappinessScore
from mlp import MLPHappinessScore


class HappinessScore():
    def __init__(self, config):
        data_config = config['data']
        mlp_config = config['mlp']

        train_data = pd.read_csv(data_config['train_data_path'], delimiter = data_config['delimiter'])
        test_data = pd.read_csv(data_config['train_data_path'], delimiter = data_config['delimiter'])

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
        rms = sqrt(mean_squared_error(y, y_predicted))
        return rms


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./happiness_score.conf')

    happiness_score = HappinessScore(config)

    rmse_linear_regression = happiness_score.run_linear_regression()
    rmse_mlp = happiness_score.run_mlp()

    print('RMS error of linear regression model' + str(rmse_linear_regression))
    print('RMS error of mlp model' + str(rmse_mlp))