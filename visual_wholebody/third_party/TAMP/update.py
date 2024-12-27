import numpy as np
import matplotlib.pyplot as plt

# Generate x values
np.random.seed(0)  # For reproducibility
x = np.sort(np.random.uniform(0, 10, 500))

# Generate y values using a non-monotonic function of x
y = np.sin(x) * (1 + 0.5 * x) + np.random.normal(scale=0.5, size=x.size)

# Define a new y value
y_new = 5.0

# Define a tolerance epsilon
epsilon = 0.5  # Adjust as needed

# Find indices where y_i is within epsilon of y_new
indices = np.where(np.abs(y - y_new) <= epsilon)[0]

# Retrieve corresponding x_i values
x_close = x[indices]

# Check if x_close is not empty
if x_close.size == 0:
    print("No data points found within the specified epsilon.")
    # Optionally, increase epsilon or handle this case differently
else:
    # Define the uniform range
    x_min = x_close.min()
    x_max = x_close.max()
    x_range = (x_min, x_max)

    print(f"Uniform x range: {x_range}")
    print(f"Number of points in range: {x_close.size}")

    # Plot x vs. y
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Data Points', alpha=0.7)
    plt.scatter(x_close, y[indices], color='red', label='Points near y_new')
    plt.axhline(y_new, color='green', linestyle='--', label='y_new')
    plt.axhline(y_new + epsilon, color='orange', linestyle='--', label='y_new ± epsilon')
    plt.axhline(y_new - epsilon, color='orange', linestyle='--')
    plt.title('Scatter Plot of x vs. y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Histogram of x values near y_new
    plt.figure(figsize=(10, 5))
    plt.hist(x_close, bins='auto', edgecolor='black')
    plt.title('Histogram of x Values Near y_new')
    plt.xlabel('x values')
    plt.ylabel('Frequency')
    plt.show()
