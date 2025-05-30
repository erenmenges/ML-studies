import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

dataset = pd.read_csv("HousingData.csv")
dataset.fillna(dataset.mean(), inplace=True)

X = dataset.drop("MEDV", axis=1).to_numpy()
y = dataset["MEDV"].values

def func(x_row, w_array, b):
    f_sum = b
    for i,x_val in enumerate(x_row):
        f_sum += x_val * w_array[i]
    return f_sum

def calculate_mse(X, y, W, b):
    cost = 0
    m = X.shape[0]
    for i in range(m):
        cost += (func(X[i], W, b) - y[i]) ** 2
    return cost / (2*m)
    
def calculate_partial_derivatives_w(X, y, W, b):
    g_derivatives = []
    for i,w in enumerate(W):
        g_sum = 0
        for j,x in enumerate(X):
            g_sum += (func(x,W,b) - y[j]) * x[i]
        g_sum /= X.shape[0]
        g_derivatives.append(g_sum)

    return g_derivatives

def calculate_partial_derivative_b(X, y, W, b):
    g_sum = 0
    for i,x in enumerate(X):
        g_sum += (func(x,W,b) - y[i])
    g_sum /= X.shape[0]
    return g_sum

# normalize x values
mu    = X.mean(axis=0)       
sigma = X.std(axis=0)        
X_norm = (X - mu) / sigma    

# training parameters & history
num_iters      = 1000
learning_rate = 0.01
cost_history  = []
W_history     = []
b_history     = []

a = 0.01
W = np.zeros(X_norm.shape[1])
b = 0

for it in range(num_iters):
    dW_list = calculate_partial_derivatives_w(X_norm, y, W, b)
    db      = calculate_partial_derivative_b   (X_norm, y, W, b)

    dW = np.array(dW_list)

    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    # record metrics
    cost_history.append(c := calculate_mse(X_norm, y, W, b))
    W_history.append(W.copy())
    b_history.append(b)

    # log every 100 iters
    if it % 100 == 0:
        logging.info(f"iter={it:4d}  cost={c:.4f}  b={b:.4f}  W={W}")


# Cost vs Iterations
plt.figure()
plt.plot(range(num_iters), cost_history, label="Cost")
plt.xlabel("Iteration")
plt.ylabel("MSE Cost")
plt.title("Cost vs. Iteration")
plt.legend()


# Predicted vs Actual
y_pred = X_norm.dot(W) + b
plt.figure()
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted")


