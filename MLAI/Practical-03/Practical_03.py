# Practical 3: Logistic Regression using breast cancer dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

# Step 1: Load the dataset
df = pd.read_csv('breast_cancer_dataset.csv')

# Step 2: Select independent and dependent variables: Independent variables (features) 
# # These clinical attributes help predict cancer outcome

X = df[['age', 'meno', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon']].values

# Dependent variable (target)
# status indicates presence/absence of recurrence or survival outcome
y = df['status'].values

# Step 3: Split dataset into training and testing sets 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

# Step 4: Feature scaling: Logistic Regression works better when features are scaled
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Logistic Regression model
logreg = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight='balanced')
logreg.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = logreg.predict(X_test)

# Step 7: Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("Accuracy: {:.2f}%".format(accuracy))

# Display confusion matrix (important in medical diagnosis)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 8: Visualization (2D projection for illustration)
# Note:
# Although the model uses 8 features, we visualize only
# two important features: tumor size and tumor grade.
# This plot is for understanding prediction behavior only.

# Index of 'size' = 2, index of 'grade' = 3
fig, ax = plt.subplots(1, 2, figsize=(14, 4))

# Actual values
scatter1 = ax[0].scatter(X_test[:, 2], X_test[:, 3],c=y_test, cmap=plt.cm.Paired,s=100)
ax[0].set_xlabel("Tumor Size")
ax[0].set_ylabel("Tumor Grade")
ax[0].set_title("Actual Breast Cancer Status")
plt.colorbar(scatter1, ax=ax[0], label="Class")

# Predicted values
scatter2 = ax[1].scatter(X_test[:, 2], X_test[:, 3], c=y_pred, cmap=plt.cm.Paired, s=100)
ax[1].set_xlabel("Tumor Size")
ax[1].set_ylabel("Tumor Grade")
ax[1].set_title("Predicted Breast Cancer Status\nAccuracy: {:.2f}%".format(accuracy))
plt.colorbar(scatter2, ax=ax[1], label="Class")

plt.tight_layout()
plt.show()

# Step 9: Error Analysis
plt.figure(figsize=(10, 6))
for status, color, label in [(0, 'blue', 'No Cancer'),(1, 'red', 'Cancer')]:
    idx = (y_test == status)
    plt.scatter(np.where(idx)[0], y_pred[idx] - y_test[idx], c=color, label=label, edgecolor="black"
    )

plt.axhline(0, color="black", label="Zero Error")
plt.legend()
plt.title("Prediction Errors with Cancer Status")
plt.xlabel("Sample Index")
plt.ylabel("Prediction Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


''' Training a Multilayer Perceptron (MLP) Classifier for comparison, but it gives Lower Accuracy that Logistic Regression,
    before these parameters, stratify=y, class_weight="balanced", already accuracy was 57.97%, and after use it increased to 
    62.32%, and after use MLP Classifier, accuracy is 57.97%, which is lower than Logistic Regression, this shows that Logistic Regression is better for this dataset, and MLP Classifier is not suitable for this dataset, because it is a small dataset and MLP Classifier is a complex model that requires a large dataset to perform well.
'''

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("MLP Accuracy: {:.2f}%".format(accuracy))