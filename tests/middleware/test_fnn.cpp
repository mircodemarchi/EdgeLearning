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
        EDGE_LEARNING_TEST_CALL(test_dynamic());
    }

private:
    const std::size_t BATCH_SIZE = 2;
    const std::size_t EPOCHS     = 50;

    template <
        Framework F,
        LossType LT,
        InitType IT,
        ParallelizationLevel PL,
        typename T>
    struct TestMapModel {
        using type = typename MapModel<F, LT, IT, PL, T>::feedforward_model;
    };

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

        NeuralNetworkDescriptor layers_descriptor(
            {Input{"input_layer",            4UL},
             Dense{"hidden_layer_relu",      8UL, ActivationType::ReLU     },
             Dense{"hidden_layer_softmax",   8UL, ActivationType::Softmax  },
             Dense{"hidden_layer_tanh",      8UL, ActivationType::TanH     },
             Dense{"hidden_layer_linear",    8UL, ActivationType::Linear   },
             Dense{"output_layer",           2UL, ActivationType::Linear   }}
            );
        auto m = CompileFeedforwardNeuralNetwork<>(layers_descriptor, "regressor_model");
        EDGE_LEARNING_TEST_TRY(m.fit(dataset, OptimizerType::GRADIENT_DESCENT, EPOCHS, BATCH_SIZE, 0.03));
        EDGE_LEARNING_TEST_TRY(m.fit(dataset, OptimizerType::ADAM, EPOCHS, BATCH_SIZE, 0.03));

        auto m_runtime_err = CompileFeedforwardNeuralNetwork<>({}, "regressor_model_runtime_err");
        EDGE_LEARNING_TEST_THROWS(
            m_runtime_err.fit(dataset, OptimizerType::GRADIENT_DESCENT, EPOCHS, BATCH_SIZE, 0.03),
            std::runtime_error);
        EDGE_LEARNING_TEST_THROWS(
            m_runtime_err.fit(dataset, OptimizerType::ADAM, EPOCHS, BATCH_SIZE, 0.03),
            std::runtime_error);

        NeuralNetworkDescriptor bad_layers_descriptor({
            Input("input_layer", 4),
            Input("bad_input_layer", 4)
        });
        EDGE_LEARNING_TEST_FAIL(
            CompileFeedforwardNeuralNetwork<>(bad_layers_descriptor, "bad_regressor_model"));
        EDGE_LEARNING_TEST_THROWS(
            CompileFeedforwardNeuralNetwork<>(bad_layers_descriptor, "bad_regressor_model"),
            std::runtime_error);

        auto m_thread_parallelism_on_data_entry = CompileFeedforwardNeuralNetwork<
            LossType::MSE,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_ENTRY
            >(layers_descriptor, "regressor_model");
        EDGE_LEARNING_TEST_TRY(m_thread_parallelism_on_data_entry.fit(dataset, OptimizerType::GRADIENT_DESCENT, EPOCHS, BATCH_SIZE, 0.03));
        EDGE_LEARNING_TEST_TRY(m_thread_parallelism_on_data_entry.fit(dataset, OptimizerType::ADAM, EPOCHS, BATCH_SIZE, 0.03));

        auto m_thread_parallelism_on_data_batch = CompileFeedforwardNeuralNetwork<
            LossType::MSE,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH
        >(layers_descriptor, "regressor_model");
        EDGE_LEARNING_TEST_TRY(m_thread_parallelism_on_data_batch.fit(dataset, OptimizerType::GRADIENT_DESCENT, EPOCHS, BATCH_SIZE, 0.03));
        EDGE_LEARNING_TEST_TRY(m_thread_parallelism_on_data_batch.fit(dataset, OptimizerType::ADAM, EPOCHS, BATCH_SIZE, 0.03));
    }

    void test_predict() {
        const std::size_t OUTPUT_SIZE = 2;
        Dataset<NumType>::Mat data = {
            {10.0, 1.0, 10.0, 1.0},
            {1.0,  3.0, 8.0,  3.0},
            {8.0,  1.0, 8.0,  1.0},
            {1.0,  1.5, 8.0,  1.5},
        };
        Dataset<NumType> dataset(data);

        NeuralNetworkDescriptor layers_descriptor(
            {Input{"input_layer",            4UL},
             Dense{"hidden_layer_relu",      8UL, ActivationType::ReLU     },
             Dense{"hidden_layer_elu",       8UL, ActivationType::ELU      },
             Dense{"hidden_layer_softmax",   8UL, ActivationType::Softmax  },
             Dense{"hidden_layer_tanh",      8UL, ActivationType::TanH     },
             Dense{"hidden_layer_sigmoid",   8UL, ActivationType::Sigmoid  },
             Dense{"hidden_layer_linear",    8UL, ActivationType::Linear   },
             Dense{"output_layer",   OUTPUT_SIZE, ActivationType::Linear   }}
        );
        auto m = CompileFeedforwardNeuralNetwork<>(layers_descriptor, "regressor_model");

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

        NeuralNetworkDescriptor layers_descriptor(
            {Input{"input_layer",            4UL},
             Dense{"hidden_layer_relu",      8UL, ActivationType::ReLU     },
             Dense{"hidden_layer_softmax",   8UL, ActivationType::Softmax  },
             Dense{"hidden_layer_tanh",      8UL, ActivationType::TanH     },
             Dense{"hidden_layer_linear",    8UL, ActivationType::Linear   },
             Dense{"output_layer",           2UL, ActivationType::Linear   }}
        );
        auto m_gd = CompileFeedforwardNeuralNetwork<>(layers_descriptor, "regressor_model");
        m_gd.fit(dataset, OptimizerType::GRADIENT_DESCENT, EPOCHS, BATCH_SIZE, 0.03);

        CompileFeedforwardNeuralNetwork<>::EvaluationResult gd_performance_metrics;
        EDGE_LEARNING_TEST_TRY(gd_performance_metrics = m_gd.evaluate(dataset));
        EDGE_LEARNING_TEST_PRINT(gd_performance_metrics.loss);
        EDGE_LEARNING_TEST_PRINT(gd_performance_metrics.accuracy);
        EDGE_LEARNING_TEST_EQUAL(gd_performance_metrics.accuracy_perc,
                                 gd_performance_metrics.accuracy * 100.0);
        EDGE_LEARNING_TEST_EQUAL(gd_performance_metrics.error_rate,
                                 1.0 - gd_performance_metrics.accuracy);
        EDGE_LEARNING_TEST_EQUAL(gd_performance_metrics.error_rate_perc,
                                 gd_performance_metrics.error_rate * 100.0);

        auto m_adam = CompileFeedforwardNeuralNetwork<>(layers_descriptor, "regressor_model");
        m_adam.fit(dataset, OptimizerType::ADAM, EPOCHS, BATCH_SIZE, 0.03);

        CompileFeedforwardNeuralNetwork<>::EvaluationResult adam_performance_metrics;
        EDGE_LEARNING_TEST_TRY(adam_performance_metrics = m_adam.evaluate(dataset));
        EDGE_LEARNING_TEST_PRINT(adam_performance_metrics.loss);
        EDGE_LEARNING_TEST_PRINT(adam_performance_metrics.accuracy);
        EDGE_LEARNING_TEST_EQUAL(adam_performance_metrics.accuracy_perc,
                                 adam_performance_metrics.accuracy * 100.0);
        EDGE_LEARNING_TEST_EQUAL(adam_performance_metrics.error_rate,
                                 1.0 - adam_performance_metrics.accuracy);
        EDGE_LEARNING_TEST_EQUAL(adam_performance_metrics.error_rate_perc,
                                 adam_performance_metrics.error_rate * 100.0);
    }

    void test_dynamic()
    {
        Dataset<NumType>::Mat data = {
            {10.0, 1.0, 10.0, 1.0, 1.0, 0.0},
            {1.0,  3.0, 8.0,  3.0, 0.0, 1.0},
            {8.0,  1.0, 8.0,  1.0, 1.0, 0.0},
            {1.0,  1.5, 8.0,  1.5, 0.0, 1.0},
            {8.0,  1.0, 8.0,  1.0, 1.0, 0.0},
            {1.0,  1.5, 8.0,  1.5, 0.0, 1.0},
        };
        Dataset<NumType> dataset{data, 1, {4, 5}};

        NeuralNetworkDescriptor layers_descriptor(
            {Input{"input_layer",            4UL},
             Conv{"hidden_layer_conv",       {1, {1}}, ActivationType::ReLU},
             MaxPool{"hidden_layer_max_pool",   {{1}}, ActivationType::ReLU},
             AvgPool{"hidden_layer_avg_pool",   {{1}}, ActivationType::ReLU},
             Dropout{"hidden_layer_dropout",    {0.0}, ActivationType::ReLU},
             Dense{"hidden_layer_relu",      8UL, ActivationType::ReLU     },
             Dense{"hidden_layer_softmax",   8UL, ActivationType::Softmax  },
             Dense{"hidden_layer_tanh",      8UL, ActivationType::TanH     },
             Dense{"hidden_layer_linear",    8UL, ActivationType::Linear   },
             Dense{"output_layer",           2UL, ActivationType::Linear   }}
        );

        using TestDynamicModel = DynamicNeuralNetwork<
            TestMapModel,
            Framework::EDGE_LEARNING,
            ParallelizationLevel::SEQUENTIAL,
            NumType>;

        TestDynamicModel::EvaluationResult score0;
        EDGE_LEARNING_TEST_TRY(TestDynamicModel(layers_descriptor, "dynamic_model"));
        auto dynamic_m = TestDynamicModel(layers_descriptor, "dynamic_model");
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile());
        EDGE_LEARNING_TEST_EQUAL(dynamic_m.input_size(), 4);
        EDGE_LEARNING_TEST_EQUAL(dynamic_m.output_size(), 2);
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_PRINT(score0.loss);
        EDGE_LEARNING_TEST_PRINT(score0.accuracy);
        EDGE_LEARNING_TEST_EQUAL(score0.accuracy_perc,
                                 score0.accuracy * 100.0);
        EDGE_LEARNING_TEST_EQUAL(score0.error_rate,
                                 1.0 - score0.accuracy);
        EDGE_LEARNING_TEST_EQUAL(score0.error_rate_perc,
                                 score0.error_rate * 100.0);
        Dataset<NumType> prediction;
        auto train_dataset = dataset.trainset();
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_EQUAL(prediction.feature_size(), dataset.labels_idx().size());
        EDGE_LEARNING_TEST_EQUAL(prediction.size(), dataset.size());

        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::MSE, OptimizerType::GRADIENT_DESCENT, InitType::AUTO));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::CCE, OptimizerType::GRADIENT_DESCENT, InitType::AUTO));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::MSE, OptimizerType::ADAM, InitType::AUTO));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::CCE, OptimizerType::ADAM, InitType::AUTO));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::MSE, OptimizerType::GRADIENT_DESCENT, InitType::XAVIER_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::CCE, OptimizerType::GRADIENT_DESCENT, InitType::XAVIER_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::MSE, OptimizerType::ADAM, InitType::XAVIER_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::CCE, OptimizerType::ADAM, InitType::XAVIER_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::MSE, OptimizerType::GRADIENT_DESCENT, InitType::HE_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::CCE, OptimizerType::GRADIENT_DESCENT, InitType::HE_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::MSE, OptimizerType::ADAM, InitType::HE_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_TRY(dynamic_m.compile(LossType::CCE, OptimizerType::ADAM, InitType::HE_INIT));
        EDGE_LEARNING_TEST_TRY(dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score0 = dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_m.predict(train_dataset));

        auto fail_dynamic_m = TestDynamicModel(layers_descriptor, "fail_dynamic_model");
        EDGE_LEARNING_TEST_FAIL(fail_dynamic_m.compile(
            static_cast<LossType>(-1), OptimizerType::ADAM, static_cast<InitType>(-1)));
        EDGE_LEARNING_TEST_THROWS(fail_dynamic_m.compile(
            static_cast<LossType>(-1), OptimizerType::ADAM, static_cast<InitType>(-1)), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(fail_dynamic_m.fit(dataset));
        EDGE_LEARNING_TEST_THROWS(fail_dynamic_m.fit(dataset), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(fail_dynamic_m.evaluate(dataset));
        EDGE_LEARNING_TEST_THROWS(fail_dynamic_m.evaluate(dataset), std::runtime_error);

        auto safe_dynamic_m = TestDynamicModel(layers_descriptor, "safe_dynamic_model");
        EDGE_LEARNING_TEST_EQUAL(safe_dynamic_m.input_size(), 4);
        safe_dynamic_m = TestDynamicModel(layers_descriptor, "safe_dynamic_model");
        EDGE_LEARNING_TEST_EQUAL(safe_dynamic_m.output_size(), 2);
        safe_dynamic_m = TestDynamicModel(layers_descriptor, "safe_dynamic_model");
        EDGE_LEARNING_TEST_TRY(prediction = safe_dynamic_m.predict(train_dataset));
        EDGE_LEARNING_TEST_EQUAL(prediction.feature_size(), dataset.labels_idx().size());
        EDGE_LEARNING_TEST_EQUAL(prediction.size(), dataset.size());

        EDGE_LEARNING_TEST_TRY(
            DynamicFeedforwardNeuralNetwork<>(
                layers_descriptor, "regressor_model"));
        auto dynamic_fnn_m = DynamicFeedforwardNeuralNetwork<>(
            layers_descriptor, "dynamic_fnn_model");

        DynamicFeedforwardNeuralNetwork<>::EvaluationResult score;
        EDGE_LEARNING_TEST_TRY(dynamic_fnn_m.compile(Framework::EDGE_LEARNING));
        EDGE_LEARNING_TEST_TRY(dynamic_fnn_m.fit(Framework::EDGE_LEARNING, dataset));
        EDGE_LEARNING_TEST_TRY(score = dynamic_fnn_m.evaluate(Framework::EDGE_LEARNING, dataset));
        EDGE_LEARNING_TEST_PRINT(score.loss);
        EDGE_LEARNING_TEST_PRINT(score.accuracy);
        EDGE_LEARNING_TEST_EQUAL(score.accuracy_perc,
                                 score.accuracy * 100.0);
        EDGE_LEARNING_TEST_EQUAL(score.error_rate,
                                 1.0 - score.accuracy);
        EDGE_LEARNING_TEST_EQUAL(score.error_rate_perc,
                                 score.error_rate * 100.0);
        train_dataset = dataset.trainset();
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_fnn_m.predict(Framework::EDGE_LEARNING, train_dataset));
        EDGE_LEARNING_TEST_EQUAL(prediction.feature_size(), dataset.labels_idx().size());
        EDGE_LEARNING_TEST_EQUAL(prediction.size(), dataset.size());

        std::map<Framework, DynamicFeedforwardNeuralNetwork<>::EvaluationResult> score_map;
        EDGE_LEARNING_TEST_TRY(dynamic_fnn_m.compile());
        EDGE_LEARNING_TEST_TRY(dynamic_fnn_m.fit(dataset));
        EDGE_LEARNING_TEST_TRY(score_map = dynamic_fnn_m.evaluate(dataset));
        EDGE_LEARNING_TEST_ASSERT(!score_map.empty());
        std::map<Framework, Dataset<NumType>> prediction_map;
        EDGE_LEARNING_TEST_TRY(prediction_map = dynamic_fnn_m.predict(train_dataset));
        EDGE_LEARNING_TEST_ASSERT(prediction_map.size() >= 1);

        auto fail_dynamic_fnn_m = DynamicFeedforwardNeuralNetwork<>(
            layers_descriptor, "fail_dynamic_fnn_model");
        EDGE_LEARNING_TEST_FAIL(fail_dynamic_fnn_m.fit(Framework::EDGE_LEARNING, dataset));
        EDGE_LEARNING_TEST_THROWS(fail_dynamic_fnn_m.fit(Framework::EDGE_LEARNING, dataset),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(fail_dynamic_fnn_m.evaluate(Framework::EDGE_LEARNING, dataset));
        EDGE_LEARNING_TEST_THROWS(fail_dynamic_fnn_m.evaluate(Framework::EDGE_LEARNING, dataset),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(fail_dynamic_fnn_m.fit(dataset));
        EDGE_LEARNING_TEST_THROWS(fail_dynamic_fnn_m.fit(dataset), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(fail_dynamic_fnn_m.evaluate(dataset));
        EDGE_LEARNING_TEST_THROWS(fail_dynamic_fnn_m.evaluate(dataset), std::runtime_error);

        auto safe_dynamic_fnn_m = DynamicFeedforwardNeuralNetwork<>(
            layers_descriptor, "fail_dynamic_fnn_model");
        EDGE_LEARNING_TEST_TRY(prediction = dynamic_fnn_m.predict(Framework::EDGE_LEARNING, train_dataset));
        EDGE_LEARNING_TEST_EQUAL(prediction.feature_size(), dataset.labels_idx().size());
        EDGE_LEARNING_TEST_EQUAL(prediction.size(), dataset.size());
        EDGE_LEARNING_TEST_TRY(prediction_map = dynamic_fnn_m.predict(train_dataset));
        EDGE_LEARNING_TEST_ASSERT(prediction_map.size() >= 1);
    }
};

int main() {
    TestFNN().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
