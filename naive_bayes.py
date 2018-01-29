import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class MyGaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        self.labels_ = np.unique(y)
        self.means_ = pd.DataFrame(X).groupby(y).agg(np.mean).as_matrix()
        self.stds_ = (pd.DataFrame(X).groupby(y).agg(np.std) + 10 ** -5).as_matrix()
        self.priors_ = (pd.DataFrame(X).groupby(y).agg(lambda x: len(x)) / len(X)).iloc[:,1]
        return self

    def predict(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        nlab = len(self.labels_)
        # score is an array nr.labels * nr.obs
        score = np.ones((nlab, X.shape[0]))
        for i in range(nlab):  # cycle for labels
            stds = np.repeat(self.stds_[i, :].reshape(1, -1), X.shape[0], axis=0)
            means = np.repeat(self.means_[i, :].reshape(1, -1), X.shape[0], axis=0)
            # loglik is an array nr.obs x nr.features
            loglik = -0.5 * np.log(2 * np.pi) - np.log(stds) - (X - means) ** 2 / (2 * stds ** 2)
            # loglik.sum(axis=1)  is a vector of dimension nr.obs
            score[i, :] = loglik.sum(axis=1) + np.log(self.priors_[i])
        # pred is a vector 1 * nr.obs
        pred = np.argmax(score, axis=0)
        return self.labels_.take(pred)

class MyBernoulliNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        self.labels_ = np.unique(y)
        self.probs_ = (pd.DataFrame(X).groupby(y).agg(np.mean) + 10 ** -5).as_matrix()
        self.priors_ = (pd.DataFrame(X).groupby(y).agg(lambda x: len(x)) / len(X)).iloc[:,1]
        return self

    def predict(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        nlab = len(self.labels_)
        # score is an array nr.labels * nr.obs
        score = np.ones((nlab, X.shape[0]))
        for i in range(nlab):  # cycle for labels
            probs = np.repeat(self.probs_[i, :].reshape(1, -1), X.shape[0], axis=0)
            # loglik is an array nr.obs x nr.features
            loglik = X * np.log(probs) + (1 - X) * np.log(1 - probs)
            # loglik.sum(axis=1)  is a vector of dimension nr.obs
            score[i, :] = loglik.sum(axis=1) + np.log(self.priors_[i])
        # pred is a vector 1 * nr.obs
        pred = np.argmax(score, axis=0)
        return self.labels_.take(pred)

class MyMultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        self.labels_ = np.unique(y)
        self.priors_ = (pd.DataFrame(X).groupby(y).agg(lambda x: len(x)) / len(X)).iloc[:,0]
        # params is a list of nr.features contingency tables
        params = []
        for i in range(X.shape[1]):
            params.append(pd.crosstab(y, X.iloc[:, i], normalize = 'index'))
        self.params_ = params
        return self

    def predict(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # score is an array nr.labels * nr.obs
        score = np.ones((len(self.labels_), X.shape[0]))
        for i in range(len(self.labels_)):       # cycle for the labels y
            loglik = X.copy()
            for j in range(X.shape[1]):     # cycle for the features x
                loglik.iloc[:, j] = loglik.iloc[:, j].replace(self.params_[j].iloc[i, :].index, self.params_[j].iloc[i, :].values)
            # loglik.sum(axis=1)  is a vector of dimension nr.obs
            score[i, :] = loglik.sum(axis=1) + np.log(self.priors_[i])
        # pred is a vector of dimension nr.obs
        pred = np.argmax(score, axis=0)
        return self.labels_.take(pred)