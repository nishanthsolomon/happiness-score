from sklearn.linear_model import LinearRegression


class LinearRegressionHappinessScore():
    def __init__(self):
        self.linear_regression = LinearRegression()
    
    def train(self, x, y):
        self.linear_regression.fit(x,y)

    def predict(self, x):
        y_predicted = self.linear_regression.predict(x)
        return y_predicted