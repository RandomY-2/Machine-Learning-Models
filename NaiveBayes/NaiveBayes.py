import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self, epilson=1e-6):
        self.epilson = epilson
    
    def fit(self, X_train, y_train):
        self.num_examples, self.num_features = X_train.shape
        self.num_classes = len(np.unique(y_train))
        
        self.classes_mean = {}
        self.classes_variance= {}
        self.classes_prior = {}
        
        for c in range(self.num_classes):
            X_c = X_train[y_train==c]
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X_train.shape[0]
    
    def predict(self, X_test):
        probs = np.zeros((self.num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(X_test, self.classes_mean[str(c)], self.classes_variance[str(c)])
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)
    
    def density_function(self, X, mu, sigma):
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.epilson))
        probs = 0.5 * np.sum(np.power(X - mu, 2) / (sigma + self.epilson), 1)
        return const - probs
                
        
        