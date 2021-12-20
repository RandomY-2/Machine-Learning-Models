import numpy as np

class Decision_Stump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None
        
    def predict(self, X):
        n = X.shape[0]
        X_col = X[:, self.feature_index]
        predictions = np.ones(n)
        
        if self.polarity == 1:
            predictions[X_col < self.threshold] = -1
        else:
            predictions[X_col > self.threshold] = -1
            
        return predictions

class AdaBoost:
    def __init__(self, n_classifiers=5):
        self.n_classifiers = n_classifiers
        self.classifiers = []
    
    def fit(self, X, y):
        n, m = X.shape
        w = np.full(n, 1 / n)
        
        self.classifiers = []
        for _ in range(self.n_classifiers):
            classifier = Decision_Stump()
            min_error = float('inf')
            
            for feature_index in range(m):
                X_col = X[:, feature_index]
                thresholds = np.unique(X_col)
                
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n)
                    predictions[X_col < threshold] = -1
                    
                    misclassified = w[y != predictions]
                    error = sum(misclassified)
                    
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    if error < min_error:
                        min_error = error
                        classifier.polarity = p
                        classifier.threshold = threshold
                        classifier.feature_index = feature_index
                        
            epsilon = 1e-10
            classifier.alpha = 0.5 * np.log((1.0 - min_error + epsilon) / (min_error + epsilon))
            
            predictions = classifier.predict(X)
            w *= np.exp(-classifier.alpha * y * predictions)
            w /= np.sum(w)
            
            self.classifiers.append(classifier)
    
    def predict(self, X):
        classifier_preds = [classifier.alpha * classifier.predict(X) for classifier in self.classifiers]
        y_pred = np.sum(classifier_preds, axis=0)
        return np.sign(y_pred)