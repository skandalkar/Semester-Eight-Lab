# Practical 2: Linear Regression using scikit-learn's built-in dataset, Boston Housing Prices

import matplotlib.pyplot as plt
import numpy as np
# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# boston = load_boston()
boston = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

highlight = abs(y_pred - y_test) < 5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

plt.scatter(y_test[highlight], y_pred[highlight], color='red', marker='o', label="Accurate")
plt.xlabel("True Values")
plt.ylabel("Predictions")
# ax1.legend()

plt.scatter(y_test, y_pred, color='blue', marker='x', label="Inaccurate")
ax1.set_xlabel("True Values")
ax1.set_ylabel("Predictions")
# plt.legend()

plt.show()
plt.savefig("output.png")

print("Mean Squared Error:", mse)
print("R-Squared:", r2)