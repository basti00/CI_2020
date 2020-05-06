#!/usr/bin/env python
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gradient_descent as gd
import logreg as lr
import logreg_toolbox

"""
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) and Logistic Regression

This is the main file that loads the data, computes the solution and plots the results.
"""


def main():
    # Set parameters
    degree = 5
    eta = 1.
    max_iter = 20

    # Load data and expand with polynomial features
    f = open('data_logreg.json', 'r')
    data = json.load(f)
    for k, v in data.items(): data[k] = np.array(v)  # Encode list into numpy array

    # Expand with polynomial features
    X_train = logreg_toolbox.poly_2D_design_matrix(data['x1_train'], data['x2_train'], degree)
    n = X_train.shape[1]

    # Define the functions of the parameter we want to optimize
    def f(theta): return lr.cost(theta, X_train, data['y_train'])

    def df(theta): return lr.grad(theta, X_train, data['y_train'])

    # Test to verify if the computation of the gradient is correct
    logreg_toolbox.check_gradient(f, df, n)

    # Point for initialization of gradient descent
    theta0 = np.zeros(n)

    theta_opt, E_list = gd.gradient_descent(f, df, theta0, eta, max_iter)

    logreg_toolbox.plot_logreg(data, degree, theta_opt, E_list)
    plt.show()


if __name__ == '__main__':
    main()
