import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Salary_dataset.csv")
dataset = dataset.loc[:,["YearsExperience","Salary"]]

x_array = dataset.loc[:, "YearsExperience"]
y_array = dataset.loc[:, "Salary"]

def func(x,w,b):
    return (w * x) + b

def calculate_mse(x_values, real_y_values, w, b):
    cost = 0
    for i,x in enumerate(x_values):
        cost += (func(x,w,b) - real_y_values.iloc[i]) ** 2
    cost /= (2 * x_values.shape[0])
    return cost

def calculate_partial_derivative_w(x_values, real_y_values, w, b):
    g_sum = 0
    for i,x in enumerate(x_values):
        g_sum += (func(x,w,b) - real_y_values.iloc[i]) * x
    g_sum /= x_values.shape[0]
    return g_sum

def calculate_partial_derivative_b(x_values, real_y_values, w, b):
    g_sum = 0
    for i,x in enumerate(x_values):
        g_sum += (func(x,w,b) - real_y_values.iloc[i])
    g_sum /= x_values.shape[0]
    return g_sum

x_mean = x_array.mean()
x_std  = x_array.std()
x_norm = (x_array - x_mean) / x_std

w = 0.0  
b = 0.0  

learning_rate = 0.01
num_iterations = 10000

# For plotting the training process
cost_history = []
iteration_list = []

for i in range(num_iterations):
    dj_dw = calculate_partial_derivative_w(x_norm, y_array, w, b)
    dj_db = calculate_partial_derivative_b(x_norm, y_array, w, b)
    
    w = w - (learning_rate * dj_dw)
    b = b - (learning_rate * dj_db)

    if (i % 100 == 0) or (i == num_iterations - 1): 
        cost = calculate_mse(x_norm, y_array, w, b)
        print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")
        cost_history.append(cost)
        iteration_list.append(i)


# (Optional) Convert back to original scale for interpretation:
#    y_pred = w*( (x - x_mean)/x_std ) + b
#  ⇒ y_pred = (w/x_std)*x + (b - w*x_mean/x_std)
w_orig = w / x_std
b_orig = b - (w * x_mean / x_std)
print(f"\nOriginal‐scale model:  y = {w_orig:.2f}·x + {b_orig:.2f}")

# Plot Cost vs Iterations
plt.figure()
plt.plot(iteration_list, cost_history, label='Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.legend()
plt.show()

# Plot final regression line on original data
plt.figure()
plt.scatter(x_array, y_array, label='Data')
x_vals = np.linspace(x_array.min(), x_array.max(), 100)
y_vals = w_orig * x_vals + b_orig
plt.plot(x_vals, y_vals, 'r-', label='Model')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Salary vs Experience with fitted line')
plt.legend()
plt.show()
