# Univariate Linear Regression

A from-scratch implementation of linear regression with one feature, built for CS 430-1, predicting city profit from population using both a closed-form solution and batch gradient descent.

## 1. Purpose

This project exists to build and understand the simplest possible supervised learning model — linear regression with a single feature — without hiding the math behind a library call. Rather than importing `scikit-learn` and calling `.fit()`, the goal was to implement the cost function, the closed-form least-squares solution, and batch gradient descent by hand in plain Python, so the mechanics of "how a model learns" are fully visible and verifiable. It doubles as a first, minimal introduction to supervised learning and cost-function minimization before moving on to more complex models.

## 2. Problem and approach

The problem is an assigned classroom exercise (CS 430-1, "Problem 02"): given a dataset of city populations and the profit of a business in that city, fit a line that predicts profit from population, then use that line to forecast profit for two new population sizes (35,000 and 70,000 people). This is the classic single-variable "food truck profit vs. city population" regression problem.

The approach taken was to solve the same problem two different ways and compare them:

- **Closed-form least squares** — directly compute θ0 (intercept) and θ1 (slope) using the normal-equation-style formula built from the means, covariance, and variance of the data, giving an exact analytical answer in one pass.
- **Batch gradient descent** — start θ0 and θ1 at 0, then iteratively nudge them in the direction that reduces the mean-squared-error cost function `J(θ0, θ1)`, using a learning rate (`alpha = 0.01`) and a fixed number of iterations (5,000).

Computing both let the two methods be checked against each other: if the iterative method converges to (nearly) the same parameters as the exact formula, that's strong evidence the gradient descent implementation is correct.

## 3. Structure and methodologies

The project is intentionally small and dependency-light, split into a data file and two scripts that share the same core functions:

- **`data.txt`** — 97 comma-separated `(x, y)` training pairs, where `x` is city population in units of 10,000 and `y` is profit in units of $10,000.
- **`Problem02.py`** — the main program. It contains:
  - `load_data()` — parses `data.txt` into parallel `x_values`/`y_values` lists.
  - `compute_mean()` — a small helper used by the least-squares formula.
  - `least_squares_parameters()` — the closed-form solution for θ0/θ1.
  - `compute_cost()` — the standard linear regression cost function, `J(θ) = (1/2m) * Σ(prediction - actual)²`.
  - `gradient_descent()` — batch gradient descent over all 97 examples per iteration.
  - `main()` — runs both methods, prints their parameters/costs/differences, and predicts profit for two population values.
- **`plot_regression.py`** — a companion script that reuses the same data-loading/least-squares/cost/gradient-descent logic, then uses **Matplotlib** to scatter the raw data points and overlay the fitted regression line, saving the figure to `linreg_plot.png`.
- **`linreg_plot.png`** — the generated output plot (population on the x-axis, profit on the y-axis, with the regression line drawn through the data).

No machine-learning libraries are used anywhere — `Problem02.py` has zero external dependencies (pure Python 3), and the only outside dependency in the whole project is **Matplotlib**, used solely for plotting in `plot_regression.py`. All numerical work (means, sums of squares, gradients) is done with plain loops and lists rather than NumPy arrays, which keeps every step of the math explicit at the cost of some performance.

## 4. Process

The natural build order, reflected in how the two scripts are structured, went roughly like this:

1. **Get the data pipeline working first.** `load_data()` was written to read `data.txt` line by line, split on the comma, and cast each field to `float`, giving a clean pair of `x_values`/`y_values` lists to work from.
2. **Implement the exact solution before the iterative one.** The closed-form least-squares formula (`least_squares_parameters`) was built first, since it gives a known-correct answer to validate against — computing the means of `x` and `y`, then summing `(x - x̄)(y - ȳ)` over the sum of `(x - x̄)²` to get the slope, and backing out the intercept from the means.
3. **Add a cost function.** `compute_cost()` implements the mean-squared-error cost `J(θ0, θ1)`, used both to report how well each method fits the data and, implicitly, as the function gradient descent is minimizing.
4. **Implement batch gradient descent and tune it.** `gradient_descent()` starts both parameters at 0 and repeatedly computes the error over the whole dataset, updating θ0 and θ1 by the average gradient scaled by a learning rate. `alpha = 0.01` and `iterations = 5000` were chosen as values that let the parameters converge to essentially the same result as the closed-form solution without diverging or needing per-iteration cost logging (the README notes that printing `J(θ0,θ1)` every few iterations is the recommended way to sanity-check convergence).
5. **Compare the two methods and report deltas.** `main()` prints both parameter sets, both costs, and the absolute differences between them (`Δtheta0`, `Δtheta1`, `Δcost`) — a direct, built-in check that the iterative method is behaving correctly.
6. **Turn the model into predictions.** The final gradient-descent parameters are used to predict profit at two population values (35,000 and 70,000 people), converting the model's abstract θ0/θ1 output into a concrete, interpretable answer to the original assignment question.
7. **Add visualization as a separate concern.** Rather than complicate `Problem02.py`, a second script (`plot_regression.py`) was written that duplicates the core functions and adds a Matplotlib scatter-plus-line plot, saved to `linreg_plot.png`, so the fit could be inspected visually and separately from the numeric output.

## 5. Outcome

Running `Problem02.py` on the provided 97-point dataset produces:

| Method | θ0 | θ1 | Cost J(θ0, θ1) |
|---|---|---|---|
| Least squares (closed-form) | -3.8957808783 | 1.1930336442 | 4.4769713760 |
| Batch gradient descent | -3.8957805263 | 1.1930336088 | 4.4769713760 |

The two methods agree to about 7 decimal places on both parameters and land on the *same* cost value to 10 significant figures — concrete, measurable confirmation that the hand-written gradient descent implementation converges to the true optimum found by the exact formula. Using the converged model, the project predicts a profit of **≈$4,519** for a city of 35,000 people and **≈$45,342** for a city of 70,000 people.

Beyond the numbers, this project demonstrates the ability to translate a mathematical formulation (a cost function and its gradient) into working, from-scratch code without leaning on a machine learning library, and to validate that code empirically by cross-checking an iterative method against an analytical one rather than trusting it blindly. It reinforced how learning rate and iteration count affect convergence, why initializing parameters at zero and averaging gradients over all examples is important in batch gradient descent, and how to separate a model's numeric output from a human-readable interpretation (converting scaled units back into real population and dollar figures). It also motivated a practical takeaway for future work: instrumenting gradient descent with periodic cost logging is a simple, effective way to catch divergence or a bad learning rate before it produces silently wrong results.
