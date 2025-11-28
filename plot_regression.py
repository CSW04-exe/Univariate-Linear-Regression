# ----------------------------------------------------------------
# Author : Carter Ward
# Class  : CS 430-1
# Date   : 09/28/2025
#
# Purpose: This program plots the training data and regression line.
#          1) Loads x,y pairs from "data.txt"
#          2) Uses gradient descent to compute θ0, θ1
#          3) Draws a scatter plot of data points
#          4) Overlays the fitted regression line
#          5) Saves the plot as "linreg_plot.png"
# ----------------------------------------------------------------

import matplotlib.pyplot as plt

def load_data(file_name="data.txt"):
    x_values, y_values = []
    y_values = []
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            x, y = line.split(",")
            x_values.append(float(x))
            y_values.append(float(y))
    return x_values, y_values

def compute_mean(values):
    return sum(values) / float(len(values))

def least_squares_parameters(x_values, y_values):
    x_mean = compute_mean(x_values)
    y_mean = compute_mean(y_values)
    numerator, denominator = 0.0, 0.0
    for i in range(len(x_values)):
        numerator += (x_values[i] - x_mean) * (y_values[i] - y_mean)
        denominator += (x_values[i] - x_mean) ** 2
    theta1 = numerator / denominator
    theta0 = y_mean - theta1 * x_mean
    return theta0, theta1

def compute_cost(theta0, theta1, x_values, y_values):
    total_error = 0.0
    m = len(x_values)
    for i in range(m):
        prediction = theta0 + theta1 * x_values[i]
        error = prediction - y_values[i]
        total_error += error ** 2
    return (1.0 / (2.0 * m)) * total_error

def gradient_descent(x_values, y_values, alpha=0.01, iterations=5000):
    theta0, theta1 = 0.0, 0.0
    m = len(x_values)
    for _ in range(iterations):
        sum_error_theta0, sum_error_theta1 = 0.0, 0.0
        for i in range(m):
            prediction = theta0 + theta1 * x_values[i]
            error = prediction - y_values[i]
            sum_error_theta0 += error
            sum_error_theta1 += error * x_values[i]
        theta0 -= alpha * (1.0 / m) * sum_error_theta0
        theta1 -= alpha * (1.0 / m) * sum_error_theta1
    return theta0, theta1

def main():
    x_values, y_values = load_data()  # no hard-coded path

    # --- Least Squares ---
    theta0_ls, theta1_ls = least_squares_parameters(x_values, y_values)
    cost_ls = compute_cost(theta0_ls, theta1_ls, x_values, y_values)

    # --- Gradient Descent ---
    theta0_gd, theta1_gd = gradient_descent(x_values, y_values)
    cost_gd = compute_cost(theta0_gd, theta1_gd, x_values, y_values)

    # --- Results ---
    print("=== Least Squares ===")
    print("θ0:", theta0_ls, "θ1:", theta1_ls, "Cost:", cost_ls)
    print("\n=== Gradient Descent ===")
    print("θ0:", theta0_gd, "θ1:", theta1_gd, "Cost:", cost_gd)
    print("\nDifferences:")
    print("Δθ0:", abs(theta0_ls - theta0_gd))
    print("Δθ1:", abs(theta1_ls - theta1_gd))
    print("ΔCost:", abs(cost_ls - cost_gd))

    # --- Plot ---
    x_min, x_max = min(x_values), max(x_values)
    x_line = [x_min, x_max]
    y_line = [theta0_gd + theta1_gd * x_min, theta0_gd + theta1_gd * x_max]
    plt.figure()
    plt.scatter(x_values, y_values, color="#d62728", marker="x", label="Training data")
    plt.plot(x_line, y_line, color="#1f77b4", linewidth=2.0, label="Linear regression")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.legend()
    plt.title("Linear Regression with One Feature")
    plt.tight_layout()
    plt.savefig("linreg_plot.png", dpi=150)
    print("Saved plot: linreg_plot.png")

if __name__ == "__main__":
    main()
