# -*- coding: utf-8 -*-
"""Multilayer_perceptron.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Rj6LnRB0EuD-FBRLLhQy0Ep1LeNrRkms

- Saurabh Jaiswal
- 24AI60R46

## Question 1 (2 marks)
"""

print("Importing Dependency")
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import os
import torch.optim as optim
from tqdm import tqdm
print("Importing done")

print("Load and preprocess data")
def load_and_preprocess_data(val_size=0.15, test_size=0.15, random_state=42):
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                     test_size=val_size_adjusted,
                                                     random_state=random_state)

    # Standardize and scale targets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler
    print("Load and preprocess done")

print("Load and preprocess data")
(X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_preprocess_data()

"""
## Question 2 (6 marks)


"""

print("Defining Neuralnet")
class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        #initialization with regularization
        self.weights = {
            'W1': np.random.randn(input_size, hidden1_size) * np.sqrt(2.0/input_size),
            'W2': np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0/hidden1_size),
            'W3': np.random.randn(hidden2_size, output_size) * np.sqrt(2.0/hidden2_size)
        }
        self.biases = {
            'b1': np.zeros((1, hidden1_size)),
            'b2': np.zeros((1, hidden2_size)),
            'b3': np.zeros((1, output_size))
        }
        self.lambda_reg = 0.001  # L2 regularization

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.Z1 = X @ self.weights['W1'] + self.biases['b1']
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.weights['W2'] + self.biases['b2']
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2 @ self.weights['W3'] + self.biases['b3']
        return self.Z3

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        reg_term = self.lambda_reg/m

        # Output layer
        dZ3 = (output - y)
        dW3 = (self.A2.T @ dZ3)/m + reg_term*self.weights['W3']
        db3 = np.sum(dZ3, axis=0, keepdims=True)/m

        # Hidden layer 2
        dA2 = dZ3 @ self.weights['W3'].T
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = (self.A1.T @ dZ2)/m + reg_term*self.weights['W2']
        db2 = np.sum(dZ2, axis=0, keepdims=True)/m

        # Hidden layer 1
        dA1 = dZ2 @ self.weights['W2'].T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1)/m + reg_term*self.weights['W1']
        db1 = np.sum(dZ1, axis=0, keepdims=True)/m

        # Update parameters
        for param, grad in zip([self.weights, self.biases],
                             [{'W1':dW1, 'W2':dW2, 'W3':dW3},
                              {'b1':db1, 'b2':db2, 'b3':db3}]):
            for key in param:
                param[key] -= learning_rate * grad[key]
print("Defination Done")

"""## Question 3 (3 marks)


"""

print("Traing function")
def train_numpy_model(model, X_train, y_train, X_val, y_val,
                     epochs=500, batch_size=128, learning_rate=0.001):
    train_losses = []
    val_losses = []
    m = X_train.shape[0]

    # Learning rate decay
    decay_rate = 0.95
    decay_step = 100

    for epoch in range(epochs):
        if epoch % decay_step == 0 and epoch > 0:
            learning_rate *= decay_rate

        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch training
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            output = model.forward(X_batch)
            model.backward(X_batch, y_batch, output, learning_rate)

        # Calculate losses
        train_pred = model.forward(X_train)
        train_loss = np.mean((train_pred - y_train)**2)
        val_pred = model.forward(X_val)
        val_loss = np.mean((val_pred - y_val)**2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses
print("Training Done")

"""## Question 4 (3 marks)


"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Define PyTorch model architecture")
class PyTorchModel(nn.Module):
    def __init__(self, input_size, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x)
print("Done")

print("Convert data to PyTorch tensors")
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)

print("Create datasets")
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

hidden_layer_sizes = [(i,j) for i in (32,64,128) for j in (32,64,128)]
best_val_loss = float('inf')
best_hidden_sizes = None
best_model = None
print(device)
print("Starting Cross-Validation for Hidden Units:")

for h1, h2 in tqdm(hidden_layer_sizes, desc="Cross-validating architectures"):
    print(f"\nTraining model with hidden layers: ({h1}, {h2})")

    # Create and train model
    model = PyTorchModel(X_train.shape[1], h1, h2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # Training loop
    best_epoch_loss = float('inf')
    for epoch in range(100):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).squeeze()
                val_loss += criterion(preds, y_batch.squeeze()).item()

        val_loss /= len(val_loader)

        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Val Loss: {val_loss:.4f}")

    # Update best model
    if best_epoch_loss < best_val_loss:
        best_val_loss = best_epoch_loss
        best_hidden_sizes = (h1, h2)
        best_model = model

    # Save weights
    torch.save(model.state_dict(), f'model_weights_{h1}_{h2}.pth')

print(f"\nBest hidden layer sizes: {best_hidden_sizes}")

"""
## Question 5 (2 marks)

"""

print("Load the best model weights")
best_model.load_state_dict(torch.load(
    f'model_weights_{best_hidden_sizes[0]}_{best_hidden_sizes[1]}.pth',
    map_location=device  # Ensure device compatibility
))

print("Train the NumPy model with best hidden sizes ")
best_h1, best_h2 = best_hidden_sizes
numpy_model = NeuralNetwork(input_size=X_train.shape[1],
                            hidden1_size=best_h1,
                            hidden2_size=best_h2,
                            output_size=1)
numpy_train_losses, numpy_val_losses = train_numpy_model(
    numpy_model, X_train, y_train, X_val, y_val, epochs=500, learning_rate=0.001
)

print("Evaluate NumPy model on test set")
test_preds_numpy = numpy_model.forward(X_test)

from sklearn.metrics import mean_squared_error
test_mse_numpy = mean_squared_error(y_test, test_preds_numpy)
print(f"Test MSE: {test_mse_numpy:.4f}")

# Plotting functions
def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10,6))
    plt.scatter(y_true*5, y_pred*5, alpha=0.3)
    plt.plot([0,25], [0,25], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.show()

print("Plotting NumPy model results")
plot_predictions(y_test, test_preds_numpy, "NumPy Model Predictions")
plot_losses(numpy_train_losses, numpy_val_losses, "NumPy Model Training Progress")
print(f"\nTest MSE: {test_mse_numpy:.4f}")

"""
## Question 6 (3 marks)


"""

print("Create data loaders for PyTorch")
test_dataset = TensorDataset(torch.FloatTensor(X_test).to(device),
                            torch.FloatTensor(y_test).to(device))
test_loader = DataLoader(test_dataset, batch_size=128)

# Train final PyTorch model
final_torch_model = PyTorchModel(X_train.shape[1], *best_hidden_sizes).to(device)
optimizer = torch.optim.Adam(final_torch_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

torch_train_loss = []
torch_val_loss = []

print("Training PyTorch model")
for epoch in tqdm(range(500),desc="Epoch"):
    final_torch_model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = final_torch_model(X_batch).squeeze()
        loss = criterion(preds, y_batch.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    torch_train_loss.append(epoch_loss/len(train_loader))

    # Validation
    final_torch_model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = final_torch_model(X_batch).squeeze()
            val_loss += criterion(preds, y_batch.squeeze()).item()

    torch_val_loss.append(val_loss/len(val_loader))

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {torch_train_loss[-1]:.4f} | Val Loss: {torch_val_loss[-1]:.4f}")
print("Training done")

"""## Question 7 (1 marks)

"""

print("Evaluate PyTorch model on test set")
final_torch_model.eval()
test_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        preds = final_torch_model(X_batch).cpu().numpy()
        test_preds.extend(preds)

test_preds = np.array(test_preds)
torch_mse = mean_squared_error(y_test, test_preds)
print(f"\nPyTorch Test MSE: {torch_mse:.4f}")

# Plot comparisons
plot_losses(torch_train_loss, torch_val_loss, "PyTorch Model Training Progress")
plot_predictions(y_test, test_preds, "PyTorch Model Predictions")

# Final comparison
print("\nFinal Comparison:")
print(f"NumPy Implementation Test MSE: {test_mse_numpy:.4f}")
print(f"PyTorch Implementation Test MSE: {torch_mse:.4f}")

if test_mse_numpy < torch_mse:
    print("\nInference: The NumPy implementation achieved a slightly lower test MSE, suggesting it might have generalized slightly better to unseen data in this specific instance.")
    print("Possible reasons for this could be differences in weight initialization, the optimization algorithm, or subtle variations in how gradients are calculated.")
elif torch_mse < test_mse_numpy:
    print("\nInference: The PyTorch implementation achieved a slightly lower test MSE, indicating potentially better generalization.")
    print("Possible reasons could include the efficiency of PyTorch's automatic differentiation and optimization routines, or inherent benefits from the framework's design for neural networks.")
else:
    print("\nInference: Both implementations performed very similarly, with nearly identical test MSE values.")
    print("This suggests that the core model architecture and hyperparameter settings played a more significant role in the results than the choice of framework itself.")



