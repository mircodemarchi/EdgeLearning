/***************************************************************************
 *            middleware/test_fnn.cpp
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
#include "middleware/fnn.hpp"

using namespace std;
using namespace EdgeLearning;


class TestFNN {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_train());
        EDGE_LEARNING_TEST_CALL(test_predict());
        EDGE_LEARNING_TEST_CALL(test_evaluate());
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
            {8.0,  1.0, 8.0,  1.0, 1.0, 0.0},
            {1.0,  1.5, 8.0,  1.5, 0.0, 1.0},
        };
        Dataset<NumType> dataset{data, 1, {4, 5}};

        NNDescriptor layers_descriptor(
            {Input{"input_layer",            4UL},
             Dense{"hidden_layer_relu",      8UL, ActivationType::ReLU     },
             Dense{"hidden_layer_softmax",   8UL, ActivationType::Softmax  },
             Dense{"hidden_layer_tanh",      8UL, ActivationType::TanH     },
             Dense{"hidden_layer_linear",    8UL, ActivationType::Linear   },
             Dense{"output_layer",           2UL, ActivationType::Linear   }}
            );
        auto m = CompileFNN<>(layers_descriptor, "regressor_model");
        EDGE_LEARNING_TEST_TRY(m.fit(dataset, EPOCHS, BATCH_SIZE, 0.03));

        auto m_runtime_err = CompileFNN<>({}, "regressor_model_runtime_err");
        EDGE_LEARNING_TEST_THROWS(
            m_runtime_err.fit(dataset, EPOCHS, BATCH_SIZE, 0.03),
            std::runtime_error);

        auto m_thread_parallelism_on_data_entry = CompileFNN<
            LossType::MSE,
            OptimizerType::GRADIENT_DESCENT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_ENTRY
            >(layers_descriptor, "regressor_model");
        EDGE_LEARNING_TEST_TRY(m_thread_parallelism_on_data_entry.fit(dataset, EPOCHS, BATCH_SIZE, 0.03));

        auto m_thread_parallelism_on_data_batch = CompileFNN<
            LossType::MSE,
            OptimizerType::GRADIENT_DESCENT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH
        >(layers_descriptor, "regressor_model");
        EDGE_LEARNING_TEST_TRY(m_thread_parallelism_on_data_batch.fit(dataset, EPOCHS, BATCH_SIZE, 0.03));
    }

    void test_predict() {
        const std::size_t OUTPUT_SIZE = 2;
        Dataset<NumType>::Mat data = {
            {10.0, 1.0, 10.0, 1.0},
            {1.0,  3.0, 8.0,  3.0},
            {8.0,  1.0, 8.0,  1.0},
            {1.0,  1.5, 8.0,  1.5},
        };
        Dataset<NumType> dataset(data, 1, {4, 5});

        NNDescriptor layers_descriptor(
            {Input{"input_layer",            4UL},
             Dense{"hidden_layer_relu",      8UL, ActivationType::ReLU     },
             Dense{"hidden_layer_elu",       8UL, ActivationType::ELU      },
             Dense{"hidden_layer_softmax",   8UL, ActivationType::Softmax  },
             Dense{"hidden_layer_tanh",      8UL, ActivationType::TanH     },
             Dense{"hidden_layer_sigmoid",   8UL, ActivationType::Sigmoid  },
             Dense{"hidden_layer_linear",    8UL, ActivationType::Linear   },
             Dense{"output_layer",   OUTPUT_SIZE, ActivationType::Linear   }}
        );
        auto m = CompileFNN<>(layers_descriptor, "regressor_model");

        Dataset<NumType> predicted_labels;
        EDGE_LEARNING_TEST_TRY(predicted_labels = m.predict(dataset));
        EDGE_LEARNING_TEST_EQUALS(predicted_labels.size(), dataset.size());
        EDGE_LEARNING_TEST_EQUALS(predicted_labels.feature_size(), OUTPUT_SIZE);
        for (const auto& e: predicted_labels.data())
        {
            EDGE_LEARNING_TEST_PRINT(e);
        }
    }

    void test_evaluate() {
        Dataset<NumType>::Mat data = {
            {10.0, 1.0, 10.0, 1.0, 1.0, 0.0},
            {1.0,  3.0, 8.0,  3.0, 0.0, 1.0},
            {8.0,  1.0, 8.0,  1.0, 1.0, 0.0},
            {1.0,  1.5, 8.0,  1.5, 0.0, 1.0},
            {8.0,  1.0, 8.0,  1.0, 1.0, 0.0},
            {1.0,  1.5, 8.0,  1.5, 0.0, 1.0},
        };
        Dataset<NumType> dataset{data, 1, {4, 5}};

        NNDescriptor layers_descriptor(
            {Input{"input_layer",            4UL},
             Dense{"hidden_layer_relu",      8UL, ActivationType::ReLU     },
             Dense{"hidden_layer_softmax",   8UL, ActivationType::Softmax  },
             Dense{"hidden_layer_tanh",      8UL, ActivationType::TanH     },
             Dense{"hidden_layer_linear",    8UL, ActivationType::Linear   },
             Dense{"output_layer",           2UL, ActivationType::Linear   }}
        );
        auto m = CompileFNN<>(layers_descriptor, "regressor_model");
        m.fit(dataset, EPOCHS, BATCH_SIZE, 0.03);

        NN<NumType>::EvaluationResult performance_metrics;
        EDGE_LEARNING_TEST_TRY(performance_metrics = m.evaluate(dataset));
        EDGE_LEARNING_TEST_PRINT(performance_metrics.loss);
        EDGE_LEARNING_TEST_PRINT(performance_metrics.accuracy);
        EDGE_LEARNING_TEST_EQUAL(performance_metrics.accuracy_perc,
                                 performance_metrics.accuracy * 100.0);
        EDGE_LEARNING_TEST_EQUAL(performance_metrics.error_rate,
                                 1.0 - performance_metrics.accuracy);
        EDGE_LEARNING_TEST_EQUAL(performance_metrics.error_rate_perc,
                                 performance_metrics.error_rate * 100.0);
    }
};

int main() {
    TestFNN().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
