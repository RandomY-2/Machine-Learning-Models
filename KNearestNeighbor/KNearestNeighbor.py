import numpy as np
import pandas as pd

class KNearestNeighbor():
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p
        self.X_train = np.array([])
        self.y_train = np.array([])
    
    def distance(self, X, Xj):        
        return np.sum(np.absolute((X - Xj) ** self.p), axis=1) ** (1 / self.p)
    
    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.core.frame.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.core.series.Series):
            y_train = y_train.to_numpy()
        
        self.X_train = X_train
        self.y_train = y_train
    
    # only for one point
    def predict(self, X_test):
        if (isinstance(X_test, pd.core.frame.DataFrame)):
            X_test = X_test.to_numpy()
        
        y_pred = []
        distance_ranking = np.argpartition(self.distance(self.X_train, X_test), self.k - 1)
        
        for i in range(len(distance_ranking)):
            if distance_ranking[i] < self.k:
                y_pred.append(self.y_train[i])

        return np.argmax(np.bincount(y_pred))