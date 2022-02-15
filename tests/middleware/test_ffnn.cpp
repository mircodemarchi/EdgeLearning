/***************************************************************************
 *            tests/middleware/test_ffnn.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of EdgeLearning.
 *
 *  EdgeLearning is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  EdgeLearning is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with EdgeLearning.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "test.hpp"
#include "middleware/ffnn.hpp"

using namespace std;
using namespace EdgeLearning;


class TestFFNN {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_train());
        EDGE_LEARNING_TEST_CALL(test_predict());
    }

private:
    const std::size_t BATCH_SIZE = 2;
    const std::size_t EPOCHS     = 50;

    void test_train() {
        Dataset<NumType>::Mat data = {
            {10.0, 1.0, 10.0, 1.0, 1.0, 0.0},
            {1.0,  3.0, 8.0,  3.0, 0.0, 1.0},
            {8.0,  1.0, 8.0,  1.0, 1.0, 0.0},
            {1.0,  1.5, 8.0,  1.5, 0.0, 1.0},
        };
        Dataset<NumType> dataset{data, 1, {4, 5}};

        FFNN::LayerDescVec layers_descriptor(
            {{"hidden_layer", 8UL, Activation::ReLU  },
             {"output_layer", 2UL, Activation::Linear   }}
            );
        EDGE_LEARNING_TEST_TRY(
            auto m = FFNN(layers_descriptor, 4,
                 LossType::MSE, BATCH_SIZE,
                 "regressor_model"));
        auto m = FFNN(layers_descriptor,
                      4, LossType::MSE, BATCH_SIZE,
                      "regressor_model");
        EDGE_LEARNING_TEST_TRY(
            m.fit<NumType>(dataset,
                           EPOCHS,
                           OptimizerType::GradientDescent,
                           0.03));
    }

    void test_predict() {
        Dataset<NumType>::Mat data = {
            {10.0, 1.0, 10.0, 1.0},
            {1.0,  3.0, 8.0,  3.0},
            {8.0,  1.0, 8.0,  1.0},
            {1.0,  1.5, 8.0,  1.5},
        };
        Dataset<NumType> dataset(data, 1, {4, 5});

        FFNN::LayerDescVec layers_descriptor(
            {{"hidden_layer", 8UL, Activation::ReLU  },
             {"output_layer", 2UL, Activation::Linear   }}
        );
        EDGE_LEARNING_TEST_TRY(
            auto m = FFNN(layers_descriptor, 4,
                 LossType::MSE, BATCH_SIZE,
                 "regressor_model"));
        auto m = FFNN(layers_descriptor,
                      4, LossType::MSE, BATCH_SIZE,
                      "regressor_model");

        Dataset<NumType> predicted_labels;
        EDGE_LEARNING_TEST_TRY(
            predicted_labels = m.predict<NumType>(dataset));
        EDGE_LEARNING_TEST_EQUALS(predicted_labels.size(), dataset.size());
        EDGE_LEARNING_TEST_EQUALS(predicted_labels.feature_size(),
                                  dataset.feature_size());
        for (const auto& e: predicted_labels.data())
        {
            EDGE_LEARNING_TEST_PRINT(e);
        }

    }
};

int main() {
    TestFFNN().test();
    return EDGE_LEARNING_TEST_FAILURES;
}