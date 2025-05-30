import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("diabetes.csv")

feature_columns = dataset.columns[:-1]  
X = (dataset[feature_columns] - dataset[feature_columns].mean()) / dataset[feature_columns].std()
y = dataset["Outcome"].values
m = X.shape[0]

def func(X, W, b):
    z = X.dot(W) + b
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

def calculate_cost(x, y_array, w, b):
    m = x.shape[0]
    p = func(x, w, b) 
    p = np.clip(p, 1e-15, 1 - 1e-15) 
    cost = -(np.sum(y_array * np.log(p) + (1 - y_array) * np.log(1 - p))) / m 
    return cost

def calculate_derivative_w(X, y, w, b): 
    m = X.shape[0]
    predictions = func(X, w, b)
    errors = predictions - y
    derivative = X.T.dot(errors) / m  
    return derivative

def calculate_derivative_b(X, y, w, b):
    m = X.shape[0]
    predictions = func(X, w, b)
    errors = predictions - y
    derivative = np.sum(errors) / m
    return derivative

a = 0.09
W = np.zeros(X.shape[1])
b = 0

iterations = 2000
costs = []
accuracies = []

for i in range(iterations):
    predictions = func(X, W, b)
    errors = predictions - y
    
    dJ_dw = X.T.dot(errors) / m
    dJ_db = np.sum(errors) / m
    
    W -= a * dJ_dw
    b -= a * dJ_db

    # --- track cost & accuracy ---
    cost_i = calculate_cost(X, y, W, b)
    pred_probs = func(X, W, b)
    preds = (pred_probs >= 0.5).astype(int)
    acc_i = np.mean(preds == y)

    costs.append(cost_i)
    accuracies.append(acc_i)

    if i % 100 == 0:
        print(f"Iter {i}: cost={cost_i:.4f}, acc={acc_i:.4f}")

# --- plotting ---
plt.figure()
plt.plot(range(iterations), costs, label="Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

plt.figure()
plt.plot(range(iterations), accuracies, label="Accuracy", color="orange")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Iterations")
plt.show()






