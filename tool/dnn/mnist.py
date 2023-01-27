import sys

import numpy as np
from tensorflow import keras

np.set_printoptions(threshold=sys.maxsize)


def get_data():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    num_classes = len(np.unique(y_train))
    print("num_classes: {}".format(num_classes))

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("x_train: (min: {}, max: {})".format(x_train.min(), x_train.max()))
    print("y_train: (min: {}, max: {})".format(y_train.min(), y_train.max()))
    print("x_test: (min: {}, max: {})".format(x_test.min(), x_test.max()))
    print("y_test: (min: {}, max: {})".format(y_test.min(), y_test.max()))

    print("x_train.shape: {}".format(x_train.shape))
    print("y_train.shape: {}".format(y_train.shape))
    print("x_test.shape: {}".format(x_test.shape))
    print("y_test.shape: {}".format(y_test.shape))

    return x_train, y_train, x_test, y_test


def preprocessing(x_train, y_train, x_test, y_test):

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print("preprocessing x_train: (min: {}, max: {})".format(x_train.min(), x_train.max()))
    print("preprocessing x_test: (min: {}, max: {})".format(x_test.min(), x_test.max()))

    return x_train, y_train, x_test, y_test


def train_convnet(msg, x_train, y_train, x_test, y_test):
    print("-------- train_convnet {} --------".format(msg))

    input_shape = (28, 28, 1)
    num_classes = y_train.shape[-1]
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 20
    optimizer = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def train_densenet(msg, x_train, y_train, x_test, y_test):
    print("-------- train_densenet {} --------".format(msg))

    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))

    input_shape = (28*28)
    num_classes = y_train.shape[-1]
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Dense(36, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 10
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.0)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def main():
    x_train, y_train, x_test, y_test = get_data()
    x_train_p, y_train_p, x_test_p, y_test_p = preprocessing(x_train, y_train, x_test, y_test)

    tests = [
        train_densenet,
        # train_convnet
    ]

    for test in tests:
        test("without preprocessing", x_train, y_train, x_test, y_test)
        test("with preprocessing", x_train_p, y_train_p, x_test_p, y_test_p)


if __name__ == '__main__':
    main()