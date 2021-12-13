import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, lr=0.001, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        
    def sign(self, X, w, b):
        return np.dot(X, w) + b
    
    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.core.frame.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.core.series.Series):
            y_train = y_train.to_numpy()
        
        self.w = np.ones(X_train.shape[1])
        self.b = 0
        
        for iteration in range(self.iterations):
            for i in range(X_train.shape[0]):
                y_pred = self.sign(X_train[i], self.w, self.b)
                
                if (-1) * y_train[i] * y_pred >= 0:
                    self.w += self.lr * y_train[i] * X_train[i]
                    self.b += self.lr * y_train[i]
                
    def predict(self, X_test):
        if (isinstance(X_test, pd.core.frame.DataFrame)):
            X_test = X_test.to_numpy()
        
        pred = self.sign(X_test, self.w, self.b)
        return np.where(pred >= 0, 1, -1)