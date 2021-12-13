import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, shape, lr=0.01, iterations=1000):
        self.w = np.ones(shape - 1, dtype=np.float32)
        self.b = 0
        self.lr = lr
        self.iterations = iterations

    def sign(self, x, w, b):
        return np.dot(x, w) + b

    # Stochastic Gradient Descent
    def fit(self, X_train, y_train):
        fitted = False
        iteration = 0
        while (not fitted) and (iteration < self.iterations):
            error_count = 0
            self.iterations += 1
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                y_pred = self.sign(X, self.w, self.b)

                if y * y_pred < 0:
                    self.w = self.w + self.lr * y * X
                    self.b = self.b + self.lr * y
                    error_count += 1

            if error_count == 0:
                fitted = True
