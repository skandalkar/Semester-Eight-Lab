# Practical 3: Logistic Regression using scikit-learn's built-in dataset, Iris Flowers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]
y = iris["target"] == 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(solver="lbfgs")
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolor="black")
ax[0].set_xlabel("Petal length")
ax[0].set_ylabel("Petal width")
ax[0].set_title("Actual values")

ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired, edgecolor="black")
ax[1].set_xlabel("Petal length")
ax[1].set_ylabel("Petal width")
ax[1].set_title("Predicted outputs (Accuracy: {:.2f})".format(accuracy))

plt.show()