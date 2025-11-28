# ----------------------------------------------------------------
# Author : Carter Ward
# Class  : CS 430-1
# Date   : 09/28/2025
#
# Purpose: This program performs linear regression with one feature.
#          1) Loads x,y pairs from "data.txt"
#          2) Computes θ0, θ1 using closed-form least squares
#          3) Computes θ0, θ1 using batch gradient descent
#          4) Prints both sets of parameters and their costs
#          5) Compares least squares vs. gradient descent results
#          6) Predicts profit for populations of 35,000 and 70,000
# ----------------------------------------------------------------


def load_data(file_name="data.txt"):
    """Reads x,y pairs from the text file."""
    x_values, y_values = [], []
    with open(file_name, "r") as file:
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
    """Closed-form solution for theta0, theta1."""
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
    """Batch gradient descent."""
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
    # --- Load Data ---
    x_values, y_values = load_data("data.txt")

    # --- Least Squares ---
    theta0_ls, theta1_ls = least_squares_parameters(x_values, y_values)
    cost_ls = compute_cost(theta0_ls, theta1_ls, x_values, y_values)

    # --- Gradient Descent ---
    theta0_gd, theta1_gd = gradient_descent(x_values, y_values, alpha=0.01, iterations=5000)
    cost_gd = compute_cost(theta0_gd, theta1_gd, x_values, y_values)

    # --- Results ---
    print("=== Least Squares ===")
    print("theta0:", theta0_ls, "theta1:", theta1_ls, "cost:", cost_ls)
    print("\n=== Gradient Descent ===")
    print("theta0:", theta0_gd, "theta1:", theta1_gd, "cost:", cost_gd)

    print("\n=== Differences ===")
    print("Δtheta0:", abs(theta0_ls - theta0_gd))
    print("Δtheta1:", abs(theta1_ls - theta1_gd))
    print("Δcost:", abs(cost_ls - cost_gd))

    # --- Predictions ---
    pop1 = 3.5  # 35,000 people
    pop2 = 7.0  # 70,000 people
    pred1 = theta0_gd + theta1_gd * pop1
    pred2 = theta0_gd + theta1_gd * pop2

    print("\n=== Predictions ===")
    print(f"Profit for 35,000 people: {pred1:.4f} (≈ ${pred1*10000:,.0f})")
    print(f"Profit for 70,000 people: {pred2:.4f} (≈ ${pred2*10000:,.0f})")

if __name__ == "__main__":
    main()
