# Univariate Linear Regression

**Type:** Individual project
**Contributor:** Carter Ward
**Course:** CS 430-1 (Machine Learning) — Problem 2
**Completed:** 09/28/2025

## Purpose
I implemented single-feature linear regression from scratch in Python to understand, at the level of the actual arithmetic, how a model is fit to data and a cost function minimized — rather than calling `model.fit()`. I coded both the closed-form least squares solution and batch gradient descent myself and cross-checked that they agree.

## Problem and Approach
The task is the classic "population vs. profit" exercise: given `data.txt` (97 comma-separated `x, y` pairs — city population in 10,000s vs. food-truck profit in $10,000s), fit `y = θ0 + θ1·x`. I solved it two ways: (1) closed-form least squares via the covariance/variance formula, and (2) batch gradient descent (5000 iterations, `alpha = 0.01`) starting from `θ0 = θ1 = 0`. I then used the trained model to predict profit for cities of 35,000 and 70,000 people.

## Structure and Methodologies
- `load_data` — reads `data.txt` into parallel `x_values`/`y_values` lists
- `compute_cost` — mean-squared-error cost, `J(θ0, θ1) = (1/2m)·Σ(θ0 + θ1·x_i − y_i)²`
- `least_squares_parameters` — closed-form `θ1 = Σ(x_i−x̄)(y_i−ȳ) / Σ(x_i−x̄)²`, `θ0 = ȳ − θ1·x̄`
- `gradient_descent` — batch updates over the full dataset each step
- `Problem02.py` uses only the Python standard library
- `plot_regression.py` duplicates the same functions and adds matplotlib to plot the fit and save `linreg_plot.png`

## Process
1. Load the 97 `(x, y)` pairs from `data.txt`.
2. Compute `θ0, θ1` via closed-form least squares and its cost.
3. Compute `θ0, θ1` via batch gradient descent and its cost.
4. Compare both parameter sets and costs to confirm agreement.
5. Predict profit for populations of 35,000 and 70,000.
6. Re-run the same pipeline in `plot_regression.py` and plot the fitted line over the data.

## Outcome
Both methods converged to essentially the same line: least squares gave `θ0 = -3.8957808783, θ1 = 1.1930336442, cost = 4.4769713760`, and gradient descent gave `θ0 = -3.8957805263, θ1 = 1.1930336088, cost = 4.4769713760` — matching to about 7 decimal places (Δθ0 ≈ 3.5e-7, Δθ1 ≈ 3.5e-8). Predicted profit was ≈$4,519 for 35,000 people and ≈$45,342 for 70,000 people. This confirmed my gradient descent implementation genuinely converges to the global minimum of a convex cost function, rather than just stopping after a fixed number of iterations. The project solidified my grasp of the core supervised-learning loop — hypothesis, cost function, and minimizing it analytically or iteratively — and gave me a concrete way to validate gradient descent against ground truth instead of eyeballing the result.

## How to Run
```
python Problem02.py          # trains and prints both models' parameters/predictions
python plot_regression.py    # trains again and saves linreg_plot.png
```
Requires Python 3 (`Problem02.py` uses only the standard library); `plot_regression.py` needs `pip install matplotlib`. Keep `data.txt` in the same folder as the scripts.

## Files
- `data.txt` — training data (97 comma-separated `x,y` pairs)
- `Problem02.py` — main program: computes θ0, θ1 both ways and predicts profit
- `plot_regression.py` — trains the model again and plots the fit
- `linreg_plot.png` — output plot from `plot_regression.py`
