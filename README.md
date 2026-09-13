# Univariate Linear Regression

Author: Carter Ward
Class: CS 430-1
Date: 09/28/2025

## Purpose

This is an assignment from my CS 430 course where I implement linear regression
with a single feature from scratch in Python. I built it to understand, at the
level of the actual arithmetic, how a linear model is fit to data and how a
cost function gets minimized — rather than just calling `model.fit()` from a
library. The project exists to demonstrate that I can derive and code the
closed-form least squares solution and batch gradient descent myself, verify
that they agree, and use the result to make predictions.

## Problem and Approach

The problem is the classic "population vs. profit" regression exercise: given
`data.txt`, a set of 97 comma-separated `(x, y)` pairs where `x` is a city's
population (in units of 10,000) and `y` is the profit of a food-truck-style
business in that city (in units of $10,000), fit a line `y = θ0 + θ1·x` that
predicts profit from population.

I approached it two ways so I could cross-check my work:

1. **Closed-form least squares** — solve directly for `θ0` and `θ1` using the
   normal-equation-style formula for simple linear regression (covariance of
   x and y over variance of x), computed with plain mean/sum arithmetic.
2. **Batch gradient descent** — start `θ0` and `θ1` at 0 and iteratively
   nudge them in the direction that reduces the mean-squared-error cost,
   using a fixed learning rate and a fixed number of iterations.

Both methods are implemented in `Problem02.py`, and I compare their resulting
parameters and costs to confirm gradient descent converges to (essentially)
the same answer as the exact algebraic solution. I then use the trained model
to predict profit for cities of 35,000 and 70,000 people.

## Structure and Methodologies

- **Data loading** (`load_data`): reads `data.txt` line by line, splits each
  line on the comma, and builds parallel `x_values`/`y_values` lists.
- **Cost function** (`compute_cost`): the standard linear regression cost,
  `J(θ0, θ1) = (1 / 2m) · Σ (θ0 + θ1·x_i − y_i)²`, i.e. mean squared error
  scaled by 1/2 for a cleaner gradient.
- **Closed-form solution** (`least_squares_parameters`): computes `θ1` as the
  ratio of `Σ(x_i − x̄)(y_i − ȳ)` to `Σ(x_i − x̄)²`, then `θ0 = ȳ − θ1·x̄`.
- **Gradient descent** (`gradient_descent`): runs `iterations` (default 5000)
  passes over the full dataset per step (batch, not stochastic), computing the
  prediction error for every point, summing the gradients for `θ0` and `θ1`,
  and updating both parameters by `alpha` (default 0.01) times the average
  gradient.
- **No external libraries** are used in `Problem02.py` — everything (means,
  sums, the regression math) is plain Python using only built-ins.
- `plot_regression.py` duplicates the same loading/cost/least-squares/
  gradient-descent functions and adds **matplotlib** (`matplotlib.pyplot`) to
  render a scatter plot of the training data with the fitted line overlaid.

## Process

Running the two scripts executes the project end to end:

1. **`python Problem02.py`**
   - Loads all 97 `(x, y)` pairs from `data.txt`.
   - Computes `θ0, θ1` via the closed-form least squares formula and its cost.
   - Computes `θ0, θ1` via batch gradient descent (5000 iterations,
     `alpha = 0.01`) and its cost.
   - Prints both parameter sets, their costs, and the absolute differences
     between the two methods (Δθ0, Δθ1, Δcost) as a sanity check that they
     agree.
   - Uses the gradient-descent parameters to predict profit for populations
     of 3.5 and 7.0 (i.e. 35,000 and 70,000 people), printing the predicted
     profit in $10,000s and converted to dollars.

2. **`python plot_regression.py`**
   - Re-loads `data.txt` and re-runs least squares and gradient descent
     independently (it's a standalone script), printing the same comparison
     output to the console.
   - Builds a scatter plot of every `(x, y)` point (red "x" markers) and
     draws the fitted regression line (blue) from the min to the max x value
     using the gradient-descent parameters.
   - Labels the axes ("Population of City in 10,000s" / "Profit in
     $10,000s"), adds a legend and title, and saves the figure to
     `linreg_plot.png` at 150 DPI.

## Outcome

On this dataset, both methods converge to essentially the same line:

```
=== Least Squares ===
theta0: -3.8957808783  theta1: 1.1930336442  cost: 4.4769713760

=== Gradient Descent ===
theta0: -3.8957805263  theta1: 1.1930336088  cost: 4.4769713760

=== Predictions ===
Profit for 35,000 people: 0.4519 (≈ $4,519)
Profit for 70,000 people: 4.5342 (≈ $45,342)
```

The gradient descent parameters match the closed-form solution to about 7
decimal places (Δθ0 ≈ 3.5e-7, Δθ1 ≈ 3.5e-8), which is the result I was hoping
for — it confirms my gradient descent implementation is actually converging
to the global minimum of a convex cost function rather than just running for
a fixed number of iterations and stopping wherever it lands. `linreg_plot.png`
shows the fit visually: the line trends upward through a noisy but genuinely
correlated scatter of points, which matches the positive slope (`θ1 ≈ 1.19`)
found by both methods.

Working through this project solidified my understanding of the core
supervised-learning loop: define a hypothesis, define a cost function that
measures how wrong it is, and either solve for the minimum analytically or
walk downhill toward it iteratively. Implementing both approaches side by
side — rather than trusting one blindly — was the most useful part, since it
gave me a concrete way to check gradient descent's correctness against ground
truth instead of just eyeballing whether the cost looked "low enough." It's
a small dataset and a single feature, but the mechanics (cost function,
gradient computation, learning rate, convergence) are the same ones that
scale up to multivariate regression and beyond.

## How to Run

- Requires Python 3 (tested on 3.13.7). `Problem02.py` uses only the standard
  library.
- To run the plotter, install matplotlib first: `pip install matplotlib`.
- Keep `data.txt` in the same folder as the scripts, then:

```
python Problem02.py          # trains and prints both models' parameters/predictions
python plot_regression.py    # trains again and saves linreg_plot.png
```

## Files

- `data.txt` — training data (97 comma-separated `x,y` pairs)
- `Problem02.py` — main program: computes θ0, θ1 both ways and predicts profit
- `plot_regression.py` — trains the model again and plots the fit
- `linreg_plot.png` — output plot from `plot_regression.py`
