import matplotlib.pyplot as plt
import numpy as np

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


# Perform full batch gradient descent
def full_batch(w, b, learning_rate):
    
    # Formulas
    # Weight derivative = 2/N * (SUM(predicted y value - actual y value) * x value)
    # Bias derivative = 2/N * (SUM(predicted y value - actual y value))
    # MSE Loss = 1/N * SUM((actual - predicted) ** 2)
    weight_derivative, bias_derivative = 0, 0
    for actual_x, actual_y in data:
        predicted_y = w * actual_x + b
        weight_derivative += ((predicted_y - actual_y) * actual_x)
        bias_derivative += (predicted_y - actual_y)

    weight_derivative *= (2/N)
    bias_derivative *= (2/N)

    w -= (weight_derivative * learning_rate)
    b -= (bias_derivative * learning_rate)

    print(f"w = {w}, b = {b}")
    return w, b
