from sklearn.neural_network import MLPRegressor


class MLPHappinessScore():
    def __init__(self, config):
        hidden_layer_size = int(config['hidden_layer_size'])
        activation = config['activation']
        solver = config['solver']
        max_iter = int(config['max_iter'])
        batch_size = int(config['batch_size'])
        self.mlp = MLPRegressor(hidden_layer_sizes=(hidden_layer_size), activation=activation, solver=solver, max_iter=max_iter, batch_size=batch_size)
    
    def train(self, x, y):
        self.mlp.fit(x,y)

    def predict(self, x):
        y_predicted = self.mlp.predict(x)
        return y_predicted