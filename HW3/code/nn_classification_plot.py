import numpy as np
import matplotlib.pyplot as plt

"""
Assignment 3: Neural networks
Part 2: Classification with Neural Networks: Fashion MNIST

This file contains functions for plotting.

"""

IMAGE_DIM = (28, 28)


def plot_image(image_matrix):
    """
    Plots a single image
    :param image_matrix: 2-D matrix of dimensions IMAGE_DIM
    :return:
    """
    ax = plt.subplot()
    # Rotate the image the right way using .T
    ax.imshow(image_matrix.reshape(*IMAGE_DIM).T, cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()


def plot_random_images(inp, n_images=3):
    """
    Picks some random images from the dataset passed in (default 3) and plots them as an image.
    :param inp: The input features from the dataset. Each row has 784 values.
    :param n_images: (optional) The number of random images to plot
    :return:
    """
    fig, ax_list = plt.subplots(1, n_images)
    image_numbers = np.random.randint(len(inp), size=n_images)
    for k_i, image_number in enumerate(image_numbers):
        ax = ax_list[k_i]
        ax.set_title("Image number {}".format(image_number))
        # Rotate the image the right way using .T
        ax.imshow(inp[image_number, :].reshape(*IMAGE_DIM).T, cmap=plt.cm.gray)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


def plot_hidden_layer_weights(hidden_layer_weights, max_plot=10):
    """
    Plots the hidden layer weights passed in.
    :param hidden_layer_weights:
    :return:
    """
    k_plot = min(hidden_layer_weights.shape[1], max_plot)
    fig, ax_list = plt.subplots(1, k_plot, figsize=(10, 5))
    for hidden_neuron_num in range(k_plot):
        ax = ax_list[hidden_neuron_num]
        vmin, vmax = hidden_layer_weights.min(), hidden_layer_weights.max()
        ax.imshow(hidden_layer_weights[:, hidden_neuron_num].reshape(*IMAGE_DIM).T, cmap=plt.cm.gray,
                  vmin=.5 * vmin, vmax=.5 * vmax)
        if hidden_neuron_num == k_plot // 2:
            ax.set_title('Feature of hidden units')
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


def plot_boxplot(train_acc, test_acc):
    """
    Plots the boxplot of training and testing accuracy.
    :param train_acc: Training accuracy
    :param test_acc: Testing accuracy
    :return:
    """
    fig, ax_list = plt.subplots(1, 2)
    ax_list[0].set_title("Boxplot of training accuracy")
    ax_list[0].boxplot(train_acc)
    ax_list[0].set_ylabel('Accuracy')

    ax_list[1].set_title("Boxplot of testing accuracy")
    ax_list[1].boxplot(test_acc)
    ax_list[1].set_ylabel('Accuracy')
    plt.show()

