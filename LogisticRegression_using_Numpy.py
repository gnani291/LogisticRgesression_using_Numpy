import numpy as np
import matplotlib.pyplot as plt

# Generate simple dataset (2D points, 2 classes)
np.random.seed(0)
X_class0 = np.random.randn(50, 2) + np.array([-2, -2])
X_class1 = np.random.randn(50, 2) + np.array([2, 2])
X = np.vstack([X_class0, X_class1])
y = np.array([0]*50 + [1]*50).reshape(-1, 1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training with gradient descent
def train(X, y, lr=0.1, epochs=1000):
    n, d = X.shape
    W = np.zeros((d,1))
    b = 0
    for _ in range(epochs):
        z = X @ W + b
        y_pred = sigmoid(z)
        error = y_pred - y
        grad_W = (X.T @ error) / n
        grad_b = np.mean(error)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b

W, b = train(X, y)

# Predictions
def predict(X, W, b):
    return (sigmoid(X @ W + b) >= 0.5).astype(int)

y_pred = predict(X, W, b)
acc = (y_pred == y).mean()
print("Accuracy:", acc)

# Plot decision boundary
x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = sigmoid(grid @ W + b).reshape(xx1.shape)

plt.contourf(xx1, xx2, probs, cmap="coolwarm", alpha=0.7)
plt.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="coolwarm", edgecolors="k")
plt.title("Logistic Regression Decision Boundary")
plt.show()
