import numpy as np
import matplotlib.pyplot as plt


# SIGMOID FUNCTION
# Purpose: Squashes input values between 0 and 1
# Formula: σ(x) = 1 / (1 + e^-x)
# Use case: Good for binary classification, output represents probability
# Disadvantage: Can cause vanishing gradient problem in deep networks
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# SIGMOID DERIVATIVE
# Purpose: Used during backpropagation to calculate gradients
# Formula: σ'(x) = σ(x) * (1 - σ(x))
# Why: Chain rule requires derivative of activation function for gradient calculation
# Input x is already sigmoid activated output, so we just multiply
def sigmoid_derivative(x):
    return x * (1 - x)


# MEAN SQUARED ERROR (MSE) LOSS
# Purpose: Calculates how far predictions are from actual values
# Formula: MSE = (1/n) * Σ(y_pred - y_true)²
# Use case: Regression tasks and classification with continuous outputs
def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


# RELU (RECTIFIED LINEAR UNIT)
# Purpose: Returns max(0, x) - either 0 or the input value
# Formula: f(x) = max(0, x)
# Advantages: Faster learning, avoids vanishing gradient, simpler computation
# Use case: Most common in hidden layers of modern neural networks
def relu(x):
    return np.maximum(0, x)

# RELU DERIVATIVE
# Purpose: Used during backpropagation to calculate gradients for ReLU layer
# Formula: f'(x) = 1 if x > 0, else 0 (either 1 or 0, no in-between)
# Impact: Sparse gradient - only positive values contribute to learning
def relu_derivative(x):
    # (x > 0) creates boolean array, .astype(float) converts to 1.0 or 0.0
    return (x > 0).astype(float)


# Create sample dataset for XOR problem (100 samples)
np.random.seed(0)  # Set seed for reproducible random numbers
samples = 100
# Input features: 3 random binary values (0 or 1) for each sample
X = np.random.randint(0, 2, (samples, 3))
# Output: XOR operation - 1 if sum is odd, 0 if even
y = (np.sum(X, axis=1) % 2).reshape(-1, 1)


# Initialize weights with random values (small random numbers for better learning)
weights_hidden = np.random.rand(3, 3)      # 3 inputs → 3 hidden neurons
weights_output = np.random.rand(3, 1)      # 3 hidden neurons → 1 output
# Initialize biases with zeros (can be updated during training)
bias_hidden = np.zeros((1, 3))             # Bias for 3 hidden neurons
bias_output = np.zeros((1, 1))             # Bias for output neuron

# List to store loss values for each epoch (for plotting)
losses = []
print()


# Train the neural network for 1000 iterations
for epoch in range(1000):

    # ---- FORWARD PROPAGATION ----
    # Calculate hidden layer: (Input × Weights_hidden) + Bias_hidden
    layer_hidden = np.dot(X, weights_hidden) + bias_hidden
    # Apply activation function to hidden layer (using ReLU)
    activation_hidden = relu(layer_hidden)
    # Calculate output layer: (Hidden × Weights_output) + Bias_output
    layer_output = np.dot(activation_hidden, weights_output) + bias_output
    # Apply sigmoid activation to output (for probability between 0 and 1)
    activation_output = sigmoid(layer_output)

    # Calculate loss: how far predictions are from actual values
    loss = mean_squared_error(activation_output, y)
    losses.append(loss)

    # ---- BACKPROPAGATION (calculate gradients) ----
    # Output layer error: difference between predicted and actual
    error_output = activation_output - y
    # Derivative of output activation (sigmoid for output layer)
    derivative_output = sigmoid_derivative(activation_output)
    # Delta (gradient) for output layer: error × derivative
    delta_output = error_output * derivative_output

    # Propagate error backwards to hidden layer
    error_hidden = delta_output.dot(weights_output.T)
    # Derivative of hidden activation (ReLU in hidden layer)
    derivative_hidden = relu_derivative(layer_hidden)
    # Delta (gradient) for hidden layer: error × derivative
    delta_hidden = error_hidden * derivative_hidden

    # ---- WEIGHT UPDATE (Gradient Descent) ----
    # Update output weights: subtract (learning_rate × gradient)
    weights_output -= activation_hidden.T.dot(delta_output) * 0.1
    # Update output bias
    bias_output -= np.sum(delta_output, axis=0, keepdims=True) * 0.1

    # Update hidden weights
    weights_hidden -= X.T.dot(delta_hidden) * 0.1
    # Update hidden bias
    bias_hidden -= np.sum(delta_hidden, axis=0, keepdims=True) * 0.1

    # Print loss every 100 epochs to monitor training progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")


# Test with a specific input vector
x_test = np.array([1, 0, 0])
# Forward pass through the trained network
layer_hidden = np.dot(x_test, weights_hidden) + bias_hidden
activation_hidden = relu(layer_hidden)
layer_output = np.dot(activation_hidden, weights_output) + bias_output
activation_output = sigmoid(layer_output)

# Find the actual output value from training data
matches = np.all(X == x_test, axis=1)  # Find rows that match test input
actual_output = int(y[matches][0][0]) if np.any(matches) else None

# Display results
print()
print(f"Predicted Output: {activation_output}")
print(f"Actual Output: {actual_output}\n")


# Create figure with 2 subplots
plt.figure(figsize=(14, 5))

# Plot 1: Training Loss Over Epochs
# This shows how the model's error decreases during training
plt.subplot(1, 2, 1)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.plot(losses, linewidth=2.5, color="blue")
plt.title("Training Loss Over Epochs")
plt.grid(True, alpha=0.5)

# Plot 2: Predictions vs Actual Values
# This compares network predictions with actual values for all samples
# Get all predictions by doing forward pass on entire training data
predictions = sigmoid(
    np.dot(relu(np.dot(X, weights_hidden) + bias_hidden), weights_output)
    + bias_output
)
plt.subplot(1, 2, 2)
plt.xlabel("Sample Index")
plt.ylabel("Output Value")
plt.title("Predictions vs Actual Values")

# Add jitter (small random offset) to x-positions to prevent overlapping
jitter = np.random.normal(0, 0.1, samples)  # Small random noise
x_jittered = np.arange(samples) + jitter

# Plot predicted values as red X's with jitter
plt.scatter(
    x_jittered,
    predictions,
    label="Predicted Output",
    alpha=0.6,
    s=50,
    color="red",
    marker="x",
)
# Plot actual values as green circles with slight offset to prevent overlap
x_jittered_actual = np.arange(samples) - jitter
plt.scatter(
    x_jittered_actual, y, label="Actual Output", alpha=0.6, s=50, color="green", marker="o"
)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()