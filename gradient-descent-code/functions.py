import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

# Data
data = [
        [3.5, 18],
        [3.69, 15],
        [3.44, 18],
        [3.43, 16],
        [4.34, 15],
        [4.42, 14],
        [2.37, 24]
        ]
N = len(data)
iteration = 0

# Functions to perform gradient descent

# Plot data points and regression line
def plot(w, b):
    
    # Separate into x and y lists
    x_data = [point[0] for point in data]
    y_data = [point[1] for point in data]


    # Generate line x values (cover the data range)
    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_line = w * x_line + b

    # Plot scatter and line
    plt.scatter(x_data, y_data, color='blue', label='Data points')
    plt.plot(x_line, y_line, color='red', label=f"y = {w}x + {b}")

    # Formatting
    plt.xlabel("Pounds (1000s)")
    plt.ylabel("Miles per gallon")
    plt.title("Data with Regression Line")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot MSE Losses vs Iterations
def plot_MSE(MSE_losses):

    plt.plot(MSE_losses, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MSE Loss vs Iterations")
    plt.grid(True)
    plt.show()

# Perform full batch gradient descent
def full_batch(w, b, learning_rate):
    
    # Formulas
    # Weight derivative = 2/N * (SUM(predicted y value - actual y value) * x value)
    # Bias derivative = 2/N * (SUM(predicted y value - actual y value))
    # MSE Loss = 1/N * SUM((actual - predicted) ** 2)

    weight_derivative, bias_derivative, MSE_loss = 0, 0, 0
    for actual_x, actual_y in data:
        predicted_y = w * actual_x + b
        weight_derivative += ((predicted_y - actual_y) * actual_x)
        bias_derivative += (predicted_y - actual_y)
        MSE_loss += ((actual_y - predicted_y) ** 2)

    weight_derivative *= (2/N)
    bias_derivative *= (2/N)
    MSE_loss *= (1/N)

    w -= (weight_derivative * learning_rate)
    b -= (bias_derivative * learning_rate)
    global iteration
    iteration += 1

    print(f"w = {w}, b = {b}, MSE Loss = {MSE_loss}, Iteration = {iteration}")
    return w, b, MSE_loss

# Perform stochiatic gradient descent
def stochiatic(w, b, learning_rate):
    
    # Formulas
    # Weight derivative = 2 * ((predicted y value - actual y value) * x value)
    # Bias derivative = 2 * ((predicted y value - actual y value))
    # MSE Loss = ((actual - predicted) ** 2)

    weight_derivative, bias_derivative, MSE_loss = 0, 0, 0

    # Pick random x,y pair from data
    random_idx = random.randint(0, len(data) - 1)
    actual_x, actual_y = data[random_idx]
    predicted_y = w * actual_x + b
    weight_derivative += (2 * (predicted_y - actual_y) * actual_x)
    bias_derivative += (2 * (predicted_y - actual_y))
    MSE_loss += ((actual_y - predicted_y) ** 2)



    w -= (weight_derivative * learning_rate)
    b -= (bias_derivative * learning_rate)
    global iteration
    iteration += 1

    print(f"w = {w}, b = {b}, MSE Loss = {MSE_loss}, Iteration = {iteration}")
    return w, b, MSE_loss


# Perform mini batch stochiatic gradient descent
def mini_batch(w, b, learning_rate, batch_size):
    
    # Formulas (M = batch_size)
    # Weight derivative = 2/M * (SUM(predicted y value - actual y value) * x value)
    # Bias derivative = 2/M * (SUM(predicted y value - actual y value))
    # MSE Loss = 1/M * SUM((actual - predicted) ** 2)

    weight_derivative, bias_derivative, MSE_loss = 0, 0, 0

    # Pick random batch of x,y pair from data
    random_idx_start = random.randint(0, len(data) - 1)
    random_idx_end = min(len(data), (random_idx_start + batch_size)) # end is NON INCLUSIVE

    for actual_x, actual_y in data[random_idx_start: random_idx_end]:
        predicted_y = w * actual_x + b
        weight_derivative += ((predicted_y - actual_y) * actual_x)
        bias_derivative += (predicted_y - actual_y)
        MSE_loss += ((actual_y - predicted_y) ** 2)

    weight_derivative *= (2/batch_size)
    bias_derivative *= (2/batch_size)
    MSE_loss *= (1/batch_size)

    w -= (weight_derivative * learning_rate)
    b -= (bias_derivative * learning_rate)
    global iteration
    iteration += 1

    print(f"w = {w}, b = {b}, MSE Loss = {MSE_loss}, Iteration = {iteration}")
    return w, b, MSE_loss