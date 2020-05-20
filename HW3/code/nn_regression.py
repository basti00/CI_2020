import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, \
    plot_learned_function, plot_mse_vs_alpha

"""
Assignment 3: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def calculate_mse(nn, x, y):
    """
    Calculates the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO y_predict_test = nn.predict(x_test)

    mse = 0
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    for n_h in [2, 5, 50]:
        nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=5000, hidden_layer_sizes=(n_h,), alpha=0)

        nn.fit(x_train, y_train)

        y_predict_train = nn.predict(x_train)
        y_predict_test = nn.predict(x_test)

        plot_learned_function(n_h, x_train, y_train, y_predict_train, x_test, y_test, y_predict_test)


    ## TODO
    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 d)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass


def ex_1_2(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass
