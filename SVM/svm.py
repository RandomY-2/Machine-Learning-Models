import numpy as np

# Hard Margin Linear SVM
class SVM:
    def __init__(self, lr=0.01, lamb=0.1, iterations=1000):
        self.lr = lr
        self.lamb = lamb
        self.iterations = iterations
    
    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        m, n = X.shape
        
        self.w = np.zeros(n)
        self.b = 0
        
        for _ in range(self.iterations):
            for index, Xi in enumerate(X):
                if y[index] * (np.dot(Xi, self.w) - self.b) >= 1:
                    self.w -= self.lr * 2 * self.lamb * self.w
                else:
                    self.w -= self.lr * (2 * self.lamb * self.w - np.dot(Xi, y[index]))
                    self.b -= self.lr * y[index]
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)