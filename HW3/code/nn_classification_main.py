import numpy as np

from nn_classification import ex_2_1
from nn_classification_plot import plot_image, plot_random_images

"""
Assignment 3: Neural networks
Part 2: Classification with Neural Networks: Fashion MNIST

This file contains the code to load the data and contains the top level code for various parts of the assignment.
Fill in all the sections containing TODOs.

"""


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784) / 255.

    return images, labels


def main():
    
    X_train, y_train = load_mnist('data', kind='train')
    X_test, y_test = load_mnist('data', kind='t10k')

    ## Plot some random images
    plot_random_images(X_train)
    ## End plot some random images

    ## 2.1
    ex_2_1(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
