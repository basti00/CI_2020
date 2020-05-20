import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
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
    ## TODO 
    y_predict_test = nn.predict(x)
    mse = mean_squared_error(y, y_predict_test)
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

    MSE_train, MSE_test = [], []

    for i in range(0, 10):
        nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=5000, hidden_layer_sizes=(5,), alpha=0, random_state=69*i)
        nn.fit(x_train, y_train)

        MSE_train.append(calculate_mse(nn, x_train, y_train))
        MSE_test.append(calculate_mse(nn, x_test, y_test))

    print(MSE_train)
    print(MSE_test)

    mean_train = np.mean(MSE_train)
    mean_test = np.mean(MSE_test)

    std_train = np.std(MSE_train)
    std_test = np.std(MSE_test)

    max_train = np.max(MSE_train)
    max_test = np.max(MSE_test)

    min_train = np.min(MSE_train)
    min_test = np.min(MSE_test)

    print(mean_train, std_train, max_train, MSE_train.index(max_train), min_train, MSE_train.index(min_train))
    print(mean_test, std_test, max_test,MSE_test.index(max_test), min_test, MSE_test.index(min_test))

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

    MSE_train, MSE_test = np.zeros((8,10)), np.zeros((8,10))

    hiddenN = [1, 2, 4, 6, 8, 12, 20, 40]
    randoms = np.random.randint(0, 1000, size=10)
    print(randoms)

    for i, n_h in enumerate(hiddenN):

        for j, rand in enumerate(randoms):
            nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=5000, hidden_layer_sizes=(n_h,), alpha=0, random_state=rand)
            nn.fit(x_train, y_train)

            MSE_train[i][j] = calculate_mse(nn, x_train, y_train)
            MSE_test[i][j]  = calculate_mse(nn, x_test, y_test)

    plot_mse_vs_neurons(MSE_train, MSE_test, hiddenN)


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

    hiddenN = [2, 5, 50]
    max_iter = 5000
    MSE_train, MSE_test = np.ndarray((len(hiddenN),max_iter)), np.ndarray((len(hiddenN),max_iter))

    for n in range(len(hiddenN)):
        nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=1, warm_start=True,
                          hidden_layer_sizes=(hiddenN[n],), alpha=0, random_state=0)
        for i in range(max_iter):
            nn.fit(x_train, y_train)

            MSE_train[n][i] = calculate_mse(nn, x_train, y_train)
            MSE_test[n][i] = calculate_mse(nn, x_test, y_test)

    plot_mse_vs_iterations(MSE_train, MSE_test, max_iter, hiddenN)

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

    alphas = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 100]
    randoms = np.random.randint(0, 1000, size=10)

    MSE_train, MSE_test = np.ndarray((len(alphas),len(randoms))), np.ndarray((len(alphas),len(randoms)))
    n_h = 50
    for a, alpha in enumerate(alphas):
        for r, random in enumerate(randoms):
            nn = MLPRegressor(activation='logistic', solver='lbfgs', max_iter=5000,
                              hidden_layer_sizes=(n_h,), alpha=alpha, random_state=random)
            nn.fit(x_train, y_train)
            MSE_train[a][r] = calculate_mse(nn, x_train, y_train)
            MSE_test[a][r] = calculate_mse(nn, x_test, y_test)

    plot_mse_vs_alpha(MSE_train, MSE_test, alphas)

