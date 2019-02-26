"""
from https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
"""


import numpy as np
import matplotlib.pyplot as plt
import os

import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

from sklearn.model_selection import train_test_split

import pickle


def look_on_data(train_X, train_Y, test_X, test_Y):

    """
    First look on data
    """

    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)

    # Find the unique numbers from the train labels
    classes = np.unique(train_Y)
    n_classes = len(classes)
    print('Total number of outputs : ', n_classes)
    print('Output classes : ', classes)

    plt.figure(figsize=[5, 5])

    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(train_X[0], cmap='gray')
    plt.title("Ground Truth : {}".format(train_Y[0]))

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_X[0], cmap='gray')
    plt.title("Ground Truth : {}".format(test_Y[0]))

    fig_file = 'fig/example_data_item.pdf'
    os.makedirs(fig_file.split("/")[0], exist_ok=True)
    plt.savefig(fig_file)


def pre_process(train_X, train_Y, test_X, test_Y):

    """
    Data pre-processing
    """

    # Reshape matrix 28x28 into matrix of size 28x28x1
    train_X = train_X.reshape(-1, 28, 28, 1)
    test_X = test_X.reshape(-1, 28, 28, 1)
    print('New training data shape : ', train_X.shape)
    print('New testing data shape : ', test_X.shape)

    # Convert int8 to float32,
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    # Rescale the pixel values in range 0 - 1 inclusive.
    train_X = train_X / 255.
    test_X = test_X / 255.

    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)

    # Display the change for category label using one-hot encoding
    print('Original label:', train_Y[0])
    print('After conversion to one-hot:', train_Y_one_hot[0])

    train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2,
                                                                  random_state=13)

    return train_X, valid_X, train_label, valid_label, test_X, test_Y_one_hot


def create_network(n_classes, dropout):

    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28, 28, 1)))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    if dropout:
        fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    if dropout:
        fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    if dropout:
        fashion_model.add(Dropout(0.4))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    if dropout:
        fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(n_classes, activation='softmax'))

    return fashion_model


def train_network(fashion_model,
                  train_X, valid_X, train_label, valid_label,
                  batch_size, epochs):

    # Compile the model
    fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # Visualize the model
    fashion_model.summary()

    # Train the model
    fashion_train = fashion_model.fit(
        train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

    return fashion_model, fashion_train


def save_model_and_history(backup_train_file, backup_model_file, fashion_train, fashion_model):

    # Backup in pickle
    os.makedirs(backup_train_file.split("/")[0], exist_ok=True)
    pickle.dump(fashion_train, file=open(backup_train_file, 'wb'))
    fashion_model.save(backup_model_file)


def load_model_and_history(backup_train_file, backup_model_file):

    fashion_train = pickle.load(file=open(backup_train_file, 'rb'))
    fashion_model = load_model(backup_model_file)

    return fashion_model, fashion_train


def evaluate_model(fashion_model, test_X, test_Y_one_hot):

    # Model evaluation on the test set
    test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])


def main(dropout=True):

    cond = "dropout"if dropout else "basic"

    # Backup file
    backup_train_file = f'backup/backup_train_{cond}.p'
    backup_model_file = f'backup/backup_model_{cond}.h5'

    # Load data
    (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

    # Take a look
    look_on_data(train_X, train_Y, test_X, test_Y)

    # Pre-process data
    train_X, valid_X, train_label, valid_label, test_X, test_Y_one_hot = \
        pre_process(train_X=train_X, test_X=test_X, train_Y=train_Y, test_Y=test_Y)

    if not os.path.exists(backup_train_file) or not os.path.exists(backup_model_file):

        # # Do the training

        # Parameters
        batch_size = 64
        epochs = 20
        n_classes = 10

        fashion_model = create_network(n_classes=n_classes, dropout=dropout)

        fashion_model, fashion_train = train_network(
            fashion_model=fashion_model, train_X=train_X, valid_X=valid_X,
            train_label=train_label, valid_label=valid_label, test_X=test_X, test_Y_one_hot=test_Y_one_hot,
            batch_size=batch_size, epochs=epochs)

        save_model_and_history(backup_train_file=backup_train_file, backup_model_file=backup_model_file,
                               fashion_train=fashion_train, fashion_model=fashion_model)

    else:

        # Load backup
        fashion_model, fashion_train = load_model_and_history(backup_model_file=backup_model_file,
                                                              backup_train_file=backup_train_file)

    evaluate_model(fashion_model, test_X=test_X, test_Y_one_hot=test_Y_one_hot)

    # Plot the accuracy and loss plots between training and validation data:
    accuracy = fashion_train.history['acc']
    val_accuracy = fashion_train.history['val_acc']
    loss = fashion_train.history['loss']
    val_loss = fashion_train.history['val_loss']
    epochs = range(len(accuracy))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(epochs, accuracy, 'bo', label='Training accuracy')
    ax1.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax1.set_title('Training and validation accuracy')
    ax1.legend()

    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.legend()

    plt.tight_layout()

    fig_file = f"fig/training_and_validation_loss_{cond}.pdf"
    os.makedirs(fig_file.split("/")[0], exist_ok=True)
    plt.savefig(fig_file)


if __name__ == "__main__":

    for v in (True, False):
        main(v)
