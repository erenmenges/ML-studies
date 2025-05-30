import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dataset = pd.read_csv("height_gender.csv")
logger.info(f"Original dataset shape: {dataset.shape}")

# Drop females above 120 pounds
dataset = dataset[~((dataset["Gender"] == "Female") & (dataset["Weight"] > 120))]
dataset = dataset[~((dataset["Gender"] == "Male") & (dataset["Weight"] < 120))]
logger.info(f"Dataset shape after filtering: {dataset.shape}")

dataset = dataset.drop(["Height","Index"], axis=1)

x = (dataset["Weight"] - dataset["Weight"].mean()) / dataset["Weight"].std()
y = dataset["Gender"].map({"Male": 1, "Female": 0}).values

logger.info(f"Weight statistics - Mean: {dataset['Weight'].mean():.2f}, Std: {dataset['Weight'].std():.2f}")
logger.info(f"Gender distribution - Males: {sum(y)}, Females: {len(y) - sum(y)}")

def func(x, w, b):
    z = w * x + b
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

def calculate_cost(x_array, y_array, w, b):
    cost = 0
    m = x_array.shape[0]
    for i, x_i in enumerate(x_array):
        p = func(x_i, w, b)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        cost += y_array[i]*np.log(p) + (1-y_array[i])*np.log(1-p)
    return -cost / m

def calculate_derivative_w(x_array, y_array, w, b):
    derivative = 0
    for i,x in enumerate(x_array):
        derivative += (func(x, w, b) - y_array[i]) * x
    return derivative/x_array.shape[0]

def calculate_derivative_b(x_array, y_array, w, b):
    derivative = 0
    for i,x in enumerate(x_array):
        derivative += (func(x, w, b) - y_array[i]) 
    return derivative/x_array.shape[0]

a = 0.01
w = 0
b = 0

logger.info(f"Starting training with learning rate: {a}")
logger.info(f"Initial parameters - w: {w}, b: {b}")

# Track cost over iterations
cost_history = []
for i in range(20000):
    # 1) compute & record current cost
    cost = calculate_cost(x, y, w, b)
    cost_history.append(cost)
    
    # Log progress every 2000 iterations
    if i % 2000 == 0:
        logger.info(f"Iteration {i}: Cost = {cost:.6f}, w = {w:.6f}, b = {b:.6f}")

    # 2) standard gradient steps
    dw = calculate_derivative_w(x, y, w, b)
    db = calculate_derivative_b(x, y, w, b)
    w  -= a * dw
    b  -= a * db

logger.info(f"Training completed. Final parameters - w: {w:.6f}, b: {b:.6f}")
logger.info(f"Final cost: {cost_history[-1]:.6f}")

# Plot cost vs. iteration
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

# --- new: plot final logistic fit against data ---
plt.figure()
plt.scatter(dataset["Weight"], y, alpha=0.5, label='Data')                                    # raw data points
h_vals = np.linspace(dataset["Weight"].min(), dataset["Weight"].max(), 100)                      # range of heights
x_vals = (h_vals - dataset["Weight"].mean()) / dataset["Weight"].std()                           # normalize them
p_vals = func(x_vals, w, b)                                                                      # predicted probabilities
plt.plot(h_vals, p_vals, color='red', label='Logistic Regression Fit')                          # fitted curve
plt.xlabel('Weight')
plt.ylabel('Probability of Male')
plt.title('Final Logistic Fit')
plt.legend()
plt.show()


