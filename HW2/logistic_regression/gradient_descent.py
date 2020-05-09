#!/usr/bin/env python
import numpy as np

"""
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) and Logistic Regression

This file contains generic implementation of gradient descent solver.
The functions are:
- TODO gradient_descent: for a given function with its gradient it finds the minimum with gradient descent
"""


def gradient_descent(f, df, theta0, learning_rate, max_iter):
    """
    Finds the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decreases the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument "theta" and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param theta0: initial point
    :param learning_rate:
    :param max_iter: maximal number of iterations
    :return: theta (solution), E_list (array of errors over iterations)
    """

    E_list = np.zeros(max_iter)
    theta = theta0

    for i in range(max_iter):
        E_list[i] = f(theta)
        theta = theta - df(theta) * learning_rate
        if i%5 == 0:
          print(100*i/max_iter, "%")

    return theta, E_list