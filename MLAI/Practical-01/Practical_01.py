import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


# Sample dataset 100 entries
np.random.seed(0)
samples = 100
X = np.random.randint(0, 2, (samples, 3))
y = (np.sum(X, axis=1) % 2).reshape(-1, 1)

weights_hidden = np.random.rand(3, 3)
weights_output = np.random.rand(3, 1)
bias_hidden = np.zeros((1, 3))
bias_output = np.zeros((1, 1))

losses = []
print()

for epoch in range(1000):

    layer_hidden = np.dot(X, weights_hidden) + bias_hidden
    activation_hidden = sigmoid(layer_hidden)
    layer_output = np.dot(activation_hidden, weights_output) + bias_output
    activation_output = sigmoid(layer_output)

    loss = mean_squared_error(activation_output, y)
    losses.append(loss)

    # Backpropagation
    error_output = activation_output - y
    derivative_output = sigmoid_derivative(activation_output)
    delta_output = error_output * derivative_output

    # Core of backpropagation
    error_hidden = delta_output.dot(weights_output.T)
    derivative_hidden = sigmoid_derivative(activation_hidden)
    delta_hidden = error_hidden * derivative_hidden

    # Backpropagation + Gradient Descent
    weights_output -= activation_hidden.T.dot(delta_output) * 0.1
    bias_output -= np.sum(delta_output, axis=0, keepdims=True) * 0.1

    weights_hidden -= X.T.dot(delta_hidden) * 0.1
    bias_hidden -= np.sum(delta_hidden, axis=0, keepdims=True) * 0.1

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

x_test = np.array([1, 0, 0])
layer_hidden = np.dot(x_test, weights_hidden) + bias_hidden
activation_hidden = sigmoid(layer_hidden)
layer_output = np.dot(activation_hidden, weights_output) + bias_output
activation_output = sigmoid(layer_output)

# Actual output
matches = np.all(X == x_test, axis=1)
actual_output = int(y[matches][0][0]) if np.any(matches) else None

print()
print(f"Predicted Output: {activation_output}")
print(f"Actual Output: {actual_output}\n")

# Visualization

# Plot 1: Loss over epochs
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses, linewidth=2.5, color="blue")
plt.title("Training Loss Over Epochs")
plt.grid(True, alpha=0.3)

# Plot 2: Predictions vs Actual
predictions = sigmoid(
    np.dot(sigmoid(np.dot(X, weights_hidden) + bias_hidden), weights_output)
    + bias_output
)
plt.subplot(1, 2, 2)
plt.xlabel("Sample Index")
plt.ylabel("Output Value")
plt.title("Predictions vs Actual Values")
plt.scatter(
    range(samples),
    predictions,
    label="Predicted Output",
    alpha=0.6,
    s=50,
    color="red",
    marker="x",
)
plt.scatter(
    range(samples), y, label="Actual Output", alpha=0.6, s=50, color="green", marker="o"
)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() 
plt.savefig("output.png")