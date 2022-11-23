#!/usr/bin/python3

##############################################################################
#            simple_classification.py
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

import time
import numpy as np

import pyedgelearning as el


def generate_inputs(random=True, entry_amount=0, input_size=0, value_from=0.0, value_to=0.0, seed=-1):
    if not random:
        ret = np.array([
            [10.0,  1.0,  10.0, 1.0],
            [1.0,  3.0,  8.0,  3.0],
            [8.0,  1.0,  8.0,  1.0],
            [1.0,  1.5,  8.0,  1.5],
            [-1.0,  2.5, -1.0,  1.5],
            [8.0, -2.5,  1.0, -3.0],
            [1.0,  2.5, -1.0,  1.5],
            [8.0,  2.5,  1.0, -3.0],
            [0.0,  0.0,  0.0,  0.0],
            [1.0,  1.0,  1.0,  1.0],
        ])
        return ret
    if seed != -1:
        np.random.seed(seed)
    return np.random.uniform(low=value_from, high=value_to, size=(entry_amount, input_size))


def generate_labels(inputs, f):
    return np.array([f(v) for v in inputs])


def main():
    SEED = 134234563
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.01

    # Load dataset
    ENTRY_AMOUNT = 1000
    INPUT_SIZE = 4
    FROM_RANDOM_VALUE = -1.0
    TO_RANDOM_VALUE = 1.0
    def classification_function(vec):
        flag = (vec[0] > 0.5 or vec[1] > 0.0) and (vec[2] > 0.0 or vec[3] > 0.5)
        return [flag, not flag]

    inputs = generate_inputs(True, ENTRY_AMOUNT, INPUT_SIZE,
                             FROM_RANDOM_VALUE, TO_RANDOM_VALUE, SEED)
    labels = generate_labels(inputs, classification_function)
    input_size = inputs.shape[1]
    output_size = labels.shape[1]

    inputs = (inputs - inputs.min(axis=0)) / (inputs.max(axis=0) - inputs.min(axis=0))
    data = np.concatenate((inputs, labels), axis=1)
    label_indexes = set(range(input_size, input_size+output_size))
    ds = el.data.Dataset(data, label_indexes)

    # Neural Network definition
    layers = [
        el.Input("input_layer",   input_size),
        el.Dense("hidden_layer1", 200, el.ActivationType.RELU),
        el.Dense("hidden_layer2", 100, el.ActivationType.RELU),
        el.Dense("output_layer",  output_size, el.ActivationType.SOFTMAX),
    ]
    model = el.FNN(layers, "model")
    model.compile(loss=el.LossType.CCE, optimizer=el.OptimizerType.GRADIENT_DESCENT)

    print("--- Training")
    start = time.time()
    model.fit(ds, EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED)
    end = time.time()
    print("elapsed: {} ms".format((end - start) * 1e3))

    print("--- Validation")
    start = time.time()
    score = model.evaluate(ds)
    print("Loss {}, Accuracy: {}%, Error rate: {}%".format(
        score.loss, score.accuracy_perc, score.error_rate_perc))
    end = time.time()
    print("elapsed: {} ms".format((end - start) * 1e3))

    print("--- Prediction")
    observations = ds.inputs()
    prediction = model.predict(observations)
    for i in range(min(ds.size(), 10)):
        entry = ds.input(i)
        expected_output = ds.label(i)
        predicted_output = prediction.entry(i)

        print("INPUT{}: {} EXPECTED: {} PREDICTED: {}".format(
            i, entry, expected_output, predicted_output
        ))

    print("End")


if __name__ == '__main__':
    main()

