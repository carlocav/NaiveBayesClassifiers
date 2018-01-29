# Test Multinomial Naive Bayes on the titanic dataset

import numpy as np
import pandas as pd

titanic = pd.read_csv('titanic.csv')
titanic = titanic.loc[:, ['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
titanic = titanic.dropna(axis=0)

# I need to encode the categorical features because sklearn's MultinomialNB only work with numerical type variables
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
titanic['sex'] = enc.fit_transform(titanic['sex'])
titanic['embarked'] = enc.fit_transform(titanic['embarked'])

X = titanic.drop('survived', axis = 1)
y = titanic.loc[:, 'survived']

# Custom transformer to categorize continuous features
from sklearn.base import BaseEstimator, TransformerMixin

class ToCategorical(BaseEstimator, TransformerMixin):
    # features need to be a list
    def __init__(self, features, percentiles = 5):
        self.features = features
        self.percentiles = percentiles
    def fit(self, X, y=None):
        X_new = X.dropna(axis=0, subset=self.features)
        percentiles = np.linspace(0, 100, self.percentiles + 1)[1:]
        self.percentiles_ = np.percentile(X_new[self.features], percentiles, axis=0)
        return self
    def transform(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = X.dropna(axis=0, subset=self.features)
        # for each variable
        for i in range(self.percentiles_.shape[1]):
            results = np.zeros(X[self.features[i]].shape)
            # for each percentile
            for j in self.percentiles_[:, i]:
                results = results + (X[self.features[i]] > float(j))
            X[self.features[i]] = results
        return X

# Pipeline
from sklearn.pipeline import Pipeline
from naive_bayes import MyMultinomialNaiveBayes

my_pipeline = Pipeline([
    ('to_cat', ToCategorical(['age', 'fare'], 3)),
    ('mult_nb', MyMultinomialNaiveBayes())
])

from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('to_cat', ToCategorical(['age', 'fare'], 5)),
    ('ber_nb', MultinomialNB())
])

# Comparison
from sklearn.model_selection import cross_val_score
cross_val_score(my_pipeline, X, y, cv=3, scoring = "accuracy")
cross_val_score(pipeline, X, y, cv=3, scoring = "accuracy")
# I got similar results


# Grid search for the best percentiles value
from sklearn.model_selection import GridSearchCV
grid = [
    {'to_cat__percentiles': list(range(2,10))}
]
grid_search = GridSearchCV(my_pipeline, grid, cv=3, n_jobs = -1)
grid_search.fit(X, y)
grid_search.best_params_





