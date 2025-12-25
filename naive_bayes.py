import numpy as np 
import math 
from sklearn.metrics import accuracy_score

# class basis division
class Naivebayes:
    def __init__(self, X, y):
        self.X , self.y = X, y
        self.classes = np.unique(y)
        self.paramters = []

        for i , c in enumerate(self.classes):
            self.paramters.append([])

            X_where_c = X[np.where(y ==c)]
            for col in X_where_c.T:
                params = {
                    "mean": col.mean(),
                    "var": col.var()
                }
            self.paramters[i].append(params)

    # step 2 : Compute likelihood
    def compute_likelihood(self, mean , var , x):
        eps = 1e-4
        coeff = 1/ math.sqrt(2 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    # step 3: calculte the prior for each class 
    def _calculate_prior(self, c):
        freq = np.mean(self.y == c)
        return freq

    def classify(self, sample):

        posteriors = []
        for i , c in enumerate(self.classes):
            posterior = self._calculate_prior(c)

            for feature_value , params in zip(sample, self.paramters[i]):
                likelihood = self.compute_likelihood(params['mean'], params['var'], feature_value)
                posterior *= likelihood

        posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self.classify(sample) for sample in X]
        return y_pred

X = np.array([
  [1, 20],
    [2, 22],
    [3, 21],
    [8, 70],
    [9, 72],
    [10, 71]  
])

y = np.array([0, 0, 0, 1, 1, 1])

from sklearn.model_selection import train_test_split

X_train , X_test, y_train , y_test = train_test_split(X, y, test_size= 0.2, random_state= 42) 

nb = Naivebayes(X, y)
y_pre= nb.predict(X_test) 

print("Prediction: {}".format(y_pre))
print("Actuals: {}".format(y_test.tolist()))
print(f"Accuracy: {accuracy_score(y_test, y_pre)}")