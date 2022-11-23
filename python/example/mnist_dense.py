#!/usr/bin/python3

##############################################################################
#            mnist_dense.py
#
#  Copyright  2006-20  Mirco De Marchi
##############################################################################

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

import os
import time

import pyedgelearning as el


def main():
    SEED = 134234563
    BATCH_SIZE = 64
    EPOCHS = 1
    LEARNING_RATE = 0.01

    # Load MNIST dataset
    resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_mnist, test_mnist = el.parser.load_mnist(resource_path)
    input_size = len(train_mnist.input_idx)
    output_size = len(train_mnist.label_idx)

    # Normalization
    train_mnist = train_mnist.min_max_normalization(0, 255, train_mnist.input_idx)
    test_mnist = test_mnist.min_max_normalization(0, 255, test_mnist.input_idx)

    # Neural Network definition
    layers = [
        el.Input("input_layer",   input_size),
        el.Dense("hidden_layer1", 250, el.ActivationType.RELU),
        el.Dense("hidden_layer2", 200, el.ActivationType.RELU),
        # el.Dense("hidden_layer3", 150, el.ActivationType.RELU),
        # el.Dense("hidden_layer4", 100, el.ActivationType.RELU),
        # el.Dense("hidden_layer5", 50,  el.ActivationType.RELU),
        el.Dense("output_layer",  output_size, el.ActivationType.SOFTMAX),
    ]
    model = el.FNN(layers, "model")
    model.compile(loss=el.LossType.CCE, optimizer=el.OptimizerType.GRADIENT_DESCENT)

    print("--- Training")
    start = time.time()
    model.fit(train_mnist, EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED)
    end = time.time()
    print("elapsed: {} ms".format((end - start) * 1e3))

    print("--- Testing")
    start = time.time()
    score = model.evaluate(test_mnist)
    print("Loss {}, Accuracy: {}%, Error rate: {}%".format(
        score.loss, score.accuracy_perc, score.error_rate_perc))
    end = time.time()
    print("elapsed: {} ms".format((end - start) * 1e3))

    print("--- Prediction")
    observations = test_mnist.inputs()
    prediction = model.predict(observations)
    for i in range(min(test_mnist.size(), 10)):
        entry = test_mnist.input(i)
        expected_output = test_mnist.label(i)
        predicted_output = prediction.entry(i)

        print("INPUT{}: {} EXPECTED: {} PREDICTED: {}".format(
            i, entry, expected_output, predicted_output
        ))

    print("End")


if __name__ == '__main__':
    main()

