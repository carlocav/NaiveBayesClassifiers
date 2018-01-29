# Test Gaussian Naive Bayes with the Iris dataset

import numpy as np
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
new_iris = np.hstack((iris["data"],iris["target"].reshape(150,1)))
attributes = iris["feature_names"].copy()
attributes.append("flower")
new_iris = pd.DataFrame(new_iris, columns = attributes)
new_iris.iloc[:, 4] = new_iris.iloc[:, 4].replace([0, 1, 2],list(iris["target_names"]))

X = new_iris.iloc[:, :4]
y = new_iris.iloc[:, 4]

shuffle_index = np.random.permutation(150)
X = X.iloc[shuffle_index,:]
y = y.iloc[shuffle_index]

from naive_bayes import MyGaussianNaiveBayes
my_clf = MyGaussianNaiveBayes()

from sklearn.naive_bayes import GaussianNB
sk_clf = GaussianNB()

# Comparison
from sklearn.model_selection import cross_val_score
cross_val_score(my_clf, X, y, cv=3, scoring = "accuracy")
cross_val_score(sk_clf, X, y, cv=3, scoring = "accuracy")
# I got similar results