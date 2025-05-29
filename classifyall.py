import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extractData():
    X = pd.read_csv("traindata.txt", header=None, sep=",")
    y = pd.read_csv("trainlabels.txt", header=None, names=["label"])
    X = X.iloc[:, :-1]  # Drop trailing column if needed

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y.values.flatten(), dtype=torch.long)

# === Activation functions ===
def tanh(x):
    return torch.tanh(x)

def tanh_derivative(x):
    return 1 - torch.tanh(x)**2

def softmax(z):
    z = z - torch.max(z, dim=1, keepdim=True)[0]  # for numerical stability
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=1, keepdim=True)

def cross_entropy(y_hat, y):
    N = y_hat.shape[0]
    return -torch.log(y_hat[range(N), y]).mean()

def cross_entropy_grad(y_hat, y):
    N = y_hat.shape[0]
    grad = y_hat.clone()
    grad[range(N), y] -= 1
    return grad / N

def init_weights(in_dim, hidden_dim, out_dim):
    torch.manual_seed(42)
    W1 = torch.randn(in_dim, hidden_dim) * (1 / np.sqrt(in_dim))
    b1 = torch.zeros((1, hidden_dim))
    W2 = torch.randn(hidden_dim, out_dim) * (1 / np.sqrt(hidden_dim))
    b2 = torch.zeros((1, out_dim))
    return W1, b1, W2, b2

def train(X, y, hidden_dim=64, epochs=400, lr=0.05):
    N, D = X.shape
    K = len(torch.unique(y))
    W1, b1, W2, b2 = init_weights(D, hidden_dim, K)

    for epoch in range(epochs):
        # Forward pass
        Z1 = X @ W1 + b1
        A1 = tanh(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # Loss
        loss = cross_entropy(A2, y)
        if (epoch + 1) % 50 == 0:
            preds = torch.argmax(A2, dim=1)
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc.item()*100:.2f}%")

        # Backprop
        dZ2 = cross_entropy_grad(A2, y)
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(0, keepdim=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * tanh_derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(0, keepdim=True)
        
        lambda_reg = 0.001

        # Update
        W1 -= lr * (dW1 + lambda_reg * W1)
        b1 -= lr * db1
        W2 -= lr * (dW2 + lambda_reg * W2)
        b2 -= lr * db2

    return W1, b1, W2, b2

# === Main ===
X, y = extractData()
X_train, X_temp, y_train, y_temp = X[:200], X[200:], y[:200], y[200:]  # Your described split
X_val, X_test = X_temp[:25], X_temp[25:]
y_val, y_test = y_temp[:25], y_temp[25:]

W1, b1, W2, b2 = train(X_train, y_train)

# Validation eval
def evaluate(X, y, W1, b1, W2, b2):
    A1 = tanh(X @ W1 + b1)
    A2 = softmax(A1 @ W2 + b2)
    preds = torch.argmax(A2, dim=1)
    acc = (preds == y).float().mean()
    return acc.item() * 100

val_acc = evaluate(X_val, y_val, W1, b1, W2, b2)
test_acc = evaluate(X_test, y_test, W1, b1, W2, b2)
print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
