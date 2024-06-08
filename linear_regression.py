import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y
        
        for _ in range(self.iterations):
            self.update_weights()
    
    def update_weights(self):
        y_pred = self.predict(self.X)
        d_theta = -(2 * (self.X.T).dot(self.y - y_pred)) / self.m
        d_bias = -2 * np.sum(self.y - y_pred) / self.m
        
        self.theta -= self.learning_rate * d_theta
        self.bias -= self.learning_rate * d_bias
    
    def predict(self, X):
        return X.dot(self.theta) + self.bias

# Sample usage
if __name__ == "__main__":
    # Generating sample data
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Training the model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    
    # Predicting and plotting
    y_pred = model.predict(X)
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.show()
