import numpy as np
import matplotlib.pyplot as plt
from svm_plot import plot_decision_function 

"""
Assignment 4: Support Vector Machine, Kernels & Multiclass classification
TODOs. Fill the cost function, the gradient function and gradient descent solver.
"""

def ex_4_a(x, y):
    
    # TODO: Split x, y (take 80% of x, and corresponding y). You can simply use indexing, since the dataset is already shuffled.
    
    
    # Define the functions of the parameter we want to optimize
    f = lambda th: cost(th, X_train, y_train, C)
    df = lambda th: grad(th, X_train, y_train, C)
    
    # TODO: Initialize w and b to zeros. What is the dimensionality of w?
    
    theta_opt, E_list = gradient_descent(f, df, (w, b), eta, max_iter)
    w, b = theta_opt
    
    # TODO: Calculate the predictions using the test set
    # TODO: Calculate the accuracy
    
    # Plot the list of errors
    if len(E_list) > 0:
        fig, ax = plt.subplots(1)
        ax.plot(E_list, linewidth=2)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Error')
        ax.set_title('Error monitoring')
        
    # TODO: Call the function for plotting (plot_decision_function).


def gradient_descent(f, df, theta0, learning_rate, max_iter):
    """
    Finds the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decreases the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param theta0: initial point
    :param learning_rate:
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (array of errors over iterations)
    """
    ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm

    E_list = np.zeros(max_iter)
    w, b = theta0

    # END TODO
    ###########

    return (w, b), E_list


def cost(theta, x, y, C):
    """
    Computes the cost of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: cost
    """
    cost = 0 # TODO 

    return cost


def grad(theta, x, y, C):
    """

    Computes the gradient of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: grad_w, grad_b
    """
    w, b = theta
    
    grad_w = 0  # TODO 
    grad_b = 0  # TODO 
    
    return grad_w, grad_b
