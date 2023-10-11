import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D


def get_data(client, data='cifar', iid="iid", balanced='balanced'):
    """
    Reads csv files and prepares datasets for model training & testing.

    :param client: Client ID
    :param data: Name of dataset
    :param iid: iid or non_iid
    :param balanced: balanced or unbalanced
    :return: Train and test datasets
    """

    # initialized for Cifar-10
    pixel = 32
    rgb = 3

    if data == 'mnist':
        pixel = 28
        rgb = 1

    if data == 'fmnist':
        pixel = 28
        rgb = 1

    if client == 'server':
        # Server does not train model
        X_train = None
        y_train = None
    else:
        X_train = pd.read_csv(f'../data/{data}/{iid}_{balanced}/{client}_X_train.csv', index_col=0)
        y_train = pd.read_csv(f'../data/{data}/{iid}_{balanced}/{client}_y_train.csv', index_col=0)
        X_train = X_train / 255.0
        X_train = X_train.values.reshape(X_train.shape[0], pixel, pixel, rgb)
        y_train = y_train['label'].values
        y_train = keras.utils.to_categorical(y_train, num_classes=10)

    X_test = pd.read_csv(f'../data/{data}/test.csv', index_col=0)
    y_test = X_test['label']
    X_test.drop('label', inplace=True, axis=1)
    X_test = X_test / 255.0
    X_test = X_test.values.reshape(X_test.shape[0], pixel, pixel, rgb)
    y_test = y_test.values
    return X_train, y_train, X_test, y_test


def get_lenet5(dataset):
    """
    Creates LeNet-5 model.

    :param data_name: Name of dataset
    :return: Keras model
    """

    if dataset == "cifar-10" or dataset == 'svhn':
        pixel = 32
        rgb = 3
    else:
        pixel = 28
        rgb = 1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # LeNet-5
    model = keras.models.Sequential()
    # Convolutional layer 1
    model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation='relu', name=f'conv2d_{0}', input_shape=(pixel, pixel, rgb)))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layer 2
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name=f'conv2d_{1}'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(keras.layers.Flatten())

    # Fully connected layer 1
    model.add(keras.layers.Dense(units=120, activation='relu', name=f'dense_{0}'))

    # Fully connected layer 2
    model.add(keras.layers.Dense(units=84, activation='relu', name=f'dense_{1}'))

    # Output layer
    model.add(keras.layers.Dense(units=10, activation='softmax', name=f'dense_{2}'))
    return model


def get_lenet3(dataset):
    """
    Creates LeNet-5 model.

    :param data_name: Name of dataset
    :return: Keras model
    """

    if dataset == "cifar-10" or dataset == 'svhn':
        pixel = 32
        rgb = 3
    else:
        pixel = 28
        rgb = 1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # LeNet-5
    model = keras.models.Sequential()
    # Convolutional layer 1
    model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(pixel, pixel, rgb)))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(keras.layers.Flatten())

    # Fully connected layer 1
    model.add(keras.layers.Dense(units=120, activation='relu'))

    # Output layer
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    return model
