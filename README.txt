========================================================
README - Linear Regression with One Feature
========================================================

Author : Carter Ward
Class  : CS 430-1
Date   : 09/28/2025

--------------------------------------------------------
Overview
--------------------------------------------------------
This project implements linear regression with one feature
(population vs. profit) in Python.

The main program (Problem02.py):
- Reads x,y values from data.txt
- Calculates θ0 and θ1 using the closed-form least squares formula
- Calculates θ0 and θ1 again using batch gradient descent
- Prints both results, their cost values, and the differences
- Uses the final parameters to predict profit for populations
  of 35,000 and 70,000 people

The plotter (plot_regression.py) is included to display the
final plot. It runs gradient descent, plots the data points
and fitted regression line, and saves the result as
linreg_plot.png.

--------------------------------------------------------
Files
--------------------------------------------------------
data.txt             -> training data (x,y pairs, comma-separated)
Problem02.py         -> main program for computing θ0, θ1
plot_regression.py   -> optional plotter
linreg_plot.png      -> output plot after running plotter

--------------------------------------------------------
Requirements
--------------------------------------------------------
- Python 3.13.7 (tested)
- No external libraries needed for Problem02.py
- To use the plotter, install matplotlib:
    pip install matplotlib

--------------------------------------------------------
How to Run
--------------------------------------------------------
1. Put data.txt in the same folder as Problem02.py
2. Open a terminal in that folder (or run from VS Code)
3. Run the main program:
    python Problem02.py

(Optional) To generate a plot:
    python plot_regression.py

--------------------------------------------------------
Sample Output
--------------------------------------------------------
=== Least Squares ===
theta0: -3.8957808783  theta1: 1.1930336442  cost: 4.4769713760

=== Gradient Descent ===
theta0: -3.8957805263  theta1: 1.1930336088  cost: 4.4769713760

=== Predictions ===
Profit for 35,000 people: 0.4519 (≈ $4,519)
Profit for 70,000 people: 4.5342 (≈ $45,342)

--------------------------------------------------------
Notes
--------------------------------------------------------
If you see "FileNotFoundError: data.txt", make sure the file
is in the same folder as Problem02.py.

A good way to check that gradient descent is working is to
print the cost function J(θ0,θ1) every few iterations and
confirm that it keeps going down.
