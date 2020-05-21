from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_boxplot, plot_image
import numpy as np


"""
Assignment 3: Neural networks
Part 2: Classification with Neural Networks: Fashion MNIST

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def ex_2_1(X_train, y_train, X_test, y_test):
    """
    Solution for exercise 2.1
    :param X_train: Train set
    :param y_train: Targets for the train set
    :param X_test: Test set
    :param y_test: Targets for the test set
    :return:
    """

    randomSeed = np.random.randint(1, 100, 5)

    n_hidd = [1,10,100]

    score_train, score_test = [], []

    best_score = 0

    bestNetwork = MLPClassifier()

    classes = ["T-shirt/top", "trousers/pants", "pullover shirt", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    for n, n_h in enumerate(n_hidd):
        for s, seed in enumerate(randomSeed):
            nn = MLPClassifier(hidden_layer_sizes=(n_h,), activation='tanh', max_iter=50, random_state=seed)

            nn.fit(X_train, y_train)

            scoretrain = nn.score(X_train, y_train)
            scoretest = nn.score(X_test, y_test)

            score_train.append(scoretrain)
            score_test.append(scoretest)

            if scoretest > best_score:
                bestNetwork = nn
                best_score = scoretest

            print(100 / (len(n_hidd) * len(randomSeed)) * ((n*len(randomSeed)) + (s+1)), "%")

            
    plot_boxplot(score_train, score_test)


    prediction = bestNetwork.predict(X_test)
    confusionMatrix = confusion_matrix(y_test, prediction)

    #confusion matrix
    print("Confusion matrix:")
    print(classes)
    print(confusionMatrix)

    #Weight
    plot_hidden_layer_weights(bestNetwork.coefs_[0])

    print("Misclassified Pictures")

    falseList = prediction == y_test

    indexPosList = []

    for i, index in enumerate(falseList):
        if index == False:
            indexPosList.append(i)

    print(indexPosList)

    for i in range(5):
        plot_image(X_test[indexPosList[i]])
        print("MLPClassifer think it is", prediction[indexPosList[i]], "but it is", y_test[indexPosList[i]])
    ## TODO
    pass
