# Practical 2: Linear Regression using scikit-learn's built-in dataset, Boston Housing Prices

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Loading Boston house price data from CSV seperately downloaded
df = pd.read_csv("boston_house_prices.csv", skiprows=1)


# Preparing the data 
X = df.iloc[:, :-1].values # Features
y = df.iloc[:, -1].values # Target variable (House Prices)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

threshold = 5
highlight = np.abs(y_pred - y_test) < threshold

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accurate and Inaccurate predictions in Same graph
accurate = highlight
inaccurate = ~highlight

ax1.scatter(
    y_test[accurate], y_pred[accurate], color="green", marker="o", label="Accurate"
)
ax1.scatter(
    y_test[inaccurate], y_pred[inaccurate], color="red", marker="x", label="Inaccurate"
)
ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
    label="Perfect Prediction",
)

ax1.set_xlabel("True Values")
ax1.set_ylabel("Predictions")
ax1.legend()
ax1.set_title("Predictions: Accurate vs Inaccurate")
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction Errors (Residuals) 
'''The Residual means the difference between the actual value and the predicted value.'''

errors = y_pred - y_test
ax2.scatter(range(len(errors)), errors, color="purple", marker="o", alpha=0.6)
ax2.axhline(y=0, color="k", linestyle="-", lw=2, label="Zero Error")
ax2.axhline(
    y=threshold,
    color="r",
    linestyle="--",
    lw=1.5,
    alpha=0.7,
    label=f"±{threshold} Error Threshold",
)
ax2.axhline(y=-threshold, color="r", linestyle="-", lw=1.5, alpha=0.7)

ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Prediction Error (Predicted - Actual)")
ax2.legend()
ax2.set_title("Prediction Errors (Residuals)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Mean Squared Error:", mse)
print("R-Squared:", r2)
print("Accuracy :", r2 * 100, "%")