# Test Bernoulli Naive Bayes on the MNIST dataset

import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

shuffle_index = np.random.permutation(70000)
X, y = X[shuffle_index], y[shuffle_index]

# Custom transformer for feature binarization 

from sklearn.base import BaseEstimator, TransformerMixin

class BinaryRecode (BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 127):
        self.threshold = threshold
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array(X > self.threshold)

# Pipeline

from sklearn.pipeline import Pipeline
from naive_bayes import MyBernoulliNaiveBayes

my_pipeline = Pipeline([
    ('bin_rec', BinaryRecode(90)),
    ('ber_nb', MyBernoulliNaiveBayes())
])

from sklearn.naive_bayes import BernoulliNB

pipeline = Pipeline([
    ('bin_rec', BinaryRecode(90)),
    ('ber_nb', BernoulliNB())
])

# Similar results 
from sklearn.model_selection import cross_val_score
cross_val_score(my_pipeline, X, y, cv=3, scoring = "accuracy")
cross_val_score(pipeline, X, y, cv=3, scoring = "accuracy")

# Grid search for the best threshold value (takes a long time - a good threshold is 90)
from sklearn.model_selection import GridSearchCV
grid = [
    {'bin_rec__threshold': list(range(1, 255, 10))}
]
grid_search = GridSearchCV(my_pipeline, grid, cv=3, n_jobs = -1)
grid_search.fit(X, y)
grid_search.best_params_
grid_search.best_estimator_
