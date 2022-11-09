/***************************************************************************
 *            dnn/test_model.cpp
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
#include "dnn/layer.hpp"
#include "dnn/model.hpp"
#include "dnn/dense.hpp"
#include "dnn/activation.hpp"
#include "dnn/recurrent.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/gd_optimizer.hpp"
#include "data/path.hpp"

using namespace std;
using namespace EdgeLearning;

class CustomLossLayer: public LossLayer {
public:
    CustomLossLayer(SizeType input_size = 0, SizeType batch_size = 1)
        : LossLayer(input_size, batch_size, "custom_loss_layer_test")
        , _i{0}
    {
        _params.resize(input_size);
    }

    SizeType param_count() const noexcept override { return input_size(); }
    NumType& param(SizeType index) override { return _params[index]; }

    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override
    {
        std::copy(inputs.begin(),
                  inputs.begin() + static_cast<std::int64_t>(input_size()),
                  _params.begin());
        _last_input = inputs.data();
        if (_i++ % 2 == 0) ++_correct; else ++_incorrect;
        _cumulative_loss += 2.0;
        return inputs;
    }
    const std::vector<NumType>& backward(
        const std::vector<NumType>&gradients) override { return gradients; }

    [[nodiscard]] SharedPtr clone() const override
    { return std::make_shared<CustomLossLayer>(*this); }

private:
    std::vector<NumType> _params;
    SizeType _i;
};

class TestModel {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_model());
        EDGE_LEARNING_TEST_CALL(test_getter());
        EDGE_LEARNING_TEST_CALL(test_load_save());
        EDGE_LEARNING_TEST_CALL(test_classifier_model());
        EDGE_LEARNING_TEST_CALL(test_classifier_model_predict());
        EDGE_LEARNING_TEST_CALL(test_regressor_model());
        EDGE_LEARNING_TEST_CALL(test_regressor_model_predict());
        EDGE_LEARNING_TEST_CALL(test_recursive_model());
    }

private:
    const std::size_t BATCH_SIZE = 2;
    const std::size_t EPOCHS     = 50;

    void test_model() 
    {
        EDGE_LEARNING_TEST_EXECUTE(auto m = Model{"model"});
        EDGE_LEARNING_TEST_TRY(auto m = Model{"model"});
        Model m{"model"};
        EDGE_LEARNING_TEST_EQUAL(m.name(), "model");
        EDGE_LEARNING_TEST_EQUAL(m.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(m.output_size(), 0);

        EDGE_LEARNING_TEST_EXECUTE(auto m_noname = Model{});
        EDGE_LEARNING_TEST_TRY(auto m_noname = Model{});
        Model m_noname{};
        EDGE_LEARNING_TEST_ASSERT(!m_noname.name().empty());
        EDGE_LEARNING_TEST_EQUAL(m_noname.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(m_noname.output_size(), 0);

        EDGE_LEARNING_TEST_EXECUTE(Model m_copy{m});
        EDGE_LEARNING_TEST_TRY(Model m_copy{m});
        Model m_copy{m};
        EDGE_LEARNING_TEST_EQUAL(m_copy.name(), "model");
        EDGE_LEARNING_TEST_EQUAL(m_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(m_copy.output_size(), 0);

        EDGE_LEARNING_TEST_EXECUTE(Model m_assign; m_assign = m);
        EDGE_LEARNING_TEST_TRY(Model m_assign; m_assign = m);
        Model m_assign; m_assign = m;
        EDGE_LEARNING_TEST_EQUAL(m_assign.name(), "model");
        EDGE_LEARNING_TEST_EQUAL(m_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(m_assign.output_size(), 0);

        EDGE_LEARNING_TEST_FAIL(m.predict({}));
        EDGE_LEARNING_TEST_THROWS(m.predict({}), std::runtime_error);

        SizeType input_size = 4;
        SizeType output_size = 8;
        EDGE_LEARNING_TEST_TRY(m.add_layer<DenseLayer>(
            "first", input_size, output_size));
        EDGE_LEARNING_TEST_TRY(m.add_layer<ReluLayer>(
            "first_relu", output_size));
        EDGE_LEARNING_TEST_TRY(m.add_layer<DenseLayer>(
            "second", input_size, output_size));
        EDGE_LEARNING_TEST_TRY(m.add_layer<ReluLayer>(
            "second_relu", output_size));

        auto l1 = m.add_layer<DenseLayer>("first", input_size, output_size);
        auto l1_relu = m.add_layer<ReluLayer>("first_relu", output_size);
        auto loss = m.add_loss<CustomLossLayer>(
            output_size, BATCH_SIZE);
        EDGE_LEARNING_TEST_TRY(m.create_edge(l1, l1_relu));
        EDGE_LEARNING_TEST_TRY(m.create_edge(l1_relu, loss));
        EDGE_LEARNING_TEST_TRY(m.init());
        EDGE_LEARNING_TEST_TRY(m.init(Model::InitializationFunction::KAIMING));
        EDGE_LEARNING_TEST_TRY(m.init(Model::InitializationFunction::XAVIER));
    }

    void test_getter()
    {
        SizeType input_size = 4;
        SizeType output_size = 8;
        Model m{"model"};
        EDGE_LEARNING_TEST_EQUAL(m.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(m.output_size(), 0);
        auto l1 = m.add_layer<DenseLayer>("first", input_size, output_size);
        auto l1_relu = m.add_layer<ReluLayer>("first_relu", output_size);
        auto loss = m.add_loss<CustomLossLayer>(
            output_size, BATCH_SIZE);
        m.create_edge(l1, l1_relu);
        m.create_loss_edge(l1_relu, loss);
        EDGE_LEARNING_TEST_EQUAL(m.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(m.input_size(1), 0);
        EDGE_LEARNING_TEST_EQUAL(m.output_size(), output_size);
        EDGE_LEARNING_TEST_EQUAL(m.output_size(1), 0);
        EDGE_LEARNING_TEST_EQUAL(m.layers().size(), 3);
        EDGE_LEARNING_TEST_EQUAL(m.layers()[0], l1);
        EDGE_LEARNING_TEST_EQUAL(m.input_layers().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(m.input_layers()[0], l1);
        EDGE_LEARNING_TEST_EQUAL(m.output_layers().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(m.output_layers()[0], l1_relu);
        EDGE_LEARNING_TEST_EQUAL(m.loss_layers().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(m.loss_layers()[0], loss);

        std::vector<NumType> input{1,2,3,4};
        std::vector<NumType> target{1,2,3,4,5,6,7,8};
        EDGE_LEARNING_TEST_TRY(m.step(input, target));
        EDGE_LEARNING_TEST_TRY(m.step(input, target));
        EDGE_LEARNING_TEST_EQUAL(m.accuracy(), 0.5);
        EDGE_LEARNING_TEST_EQUAL(m.avg_loss(), 2.0);
    }

    void test_load_save()
    {
        Model m{"test_model_load_save"};
        auto first_layer = m.add_layer<DenseLayer>("first", 4, 8);
        auto first_relu = m.add_layer<ReluLayer>("first_relu", 8);
        auto output_layer = m.add_layer<DenseLayer>("second", 8, 2);
        auto output_linear = m.add_layer<LinearLayer>("second_relu", 2);
        auto loss_layer = m.add_loss<CustomLossLayer>(
                2, BATCH_SIZE);
        m.create_edge(first_layer, first_relu);
        m.create_edge(first_relu, output_layer);
        m.create_edge(output_layer, output_linear);
        m.create_loss_edge(output_linear, loss_layer);

        EDGE_LEARNING_TEST_TRY(m.init());
        std::ofstream ofile{
            std::filesystem::path{"classifier_weight.json"},
            std::ios::trunc};
        EDGE_LEARNING_TEST_TRY(m.dump(ofile));
        ofile.close();

        EDGE_LEARNING_TEST_TRY(m.init());
        std::ifstream ifile{std::filesystem::path{"classifier_weight.json"}};
        EDGE_LEARNING_TEST_TRY(m.load(ifile));
        ifile.close();
    }

    void test_classifier_model() {
        // Input definition.
        std::vector<std::vector<NumType>> inputs = {
            {10.0, 1.0, 10.0, 1.0},
            {1.0,  3.0, 8.0,  3.0},
            {8.0,  1.0, 8.0,  1.0},
            {1.0,  1.5, 8.0,  1.5},
        };

        std::vector<std::vector<NumType>> targets = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {0.0, 1.0},
        };
        
        // Model definition.
        GradientDescentOptimizer o{NumType{0.5}};
        Model m = TestModel::_create_binary_classifier_model();
        m.init();
        m.print();

        for (std::size_t e = 0; e < EPOCHS; ++e)
        {
            std::cout << "EPOCH " << e << std::endl;
            for (std::size_t i = 0; i < inputs.size();)
            {
                for (std::size_t b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
                {
                    m.step(inputs[i], targets[i]);
                }
                m.train(o);
                std::cout << "Step " << i 
                    << " - loss: " << m.avg_loss()
                    << ", accuracy: " << m.accuracy()
                    << std::endl;
                m.reset_score();
            }
        }

        std::cout << "Final result - " << std::endl;
        m.print();

        std::ofstream params_file{
            std::filesystem::path{"classifier_weight.json"}, 
            std::ios::trunc};
        EDGE_LEARNING_TEST_TRY(m.dump(params_file));
        params_file.close();

        Model m_copy(m);
        EDGE_LEARNING_TEST_EQUAL(m_copy.name(), m.name());
        EDGE_LEARNING_TEST_EQUAL(m_copy.layers().size(), m.layers().size());
        for (std::size_t i = 0; i < m.layers().size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(m_copy.layers()[i]->name(),
                                     m.layers()[i]->name());
        }
    }

    void test_classifier_model_predict() {
        GradientDescentOptimizer o{NumType{0.3}};
        Model m = TestModel::_create_binary_classifier_model();

        std::ifstream params_file{
            std::filesystem::path{"classifier_weight.json"}};
        EDGE_LEARNING_TEST_TRY(m.load(params_file));
        params_file.close();
    }

    void test_regressor_model() {
        // Input definition.

        std::vector<std::vector<NumType>> inputs = {
            {10.0, 1.0, 10.0, 1.0},
            {1.0,  3.0, 8.0,  3.0},
            {8.0,  1.0, 8.0,  1.0},
            {1.0,  1.5, 8.0,  1.5},
        };

        std::vector<std::vector<NumType>> targets = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {0.0, 1.0},
        };
        
        // Model definition.
        GradientDescentOptimizer o{NumType{0.01}};
        Model m = TestModel::_create_regressor_model();
        m.init();
        m.print();

        for (std::size_t e = 0; e < EPOCHS; ++e)
        {
            std::cout << "EPOCH " << e << std::endl;
            for (std::size_t i = 0; i < inputs.size();)
            {
                for (std::size_t b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
                {
                    m.step(inputs[i], targets[i]);
                }
                m.train(o);
                std::cout << "Step " << i 
                    << " - loss: " << m.avg_loss()
                    << ", accuracy: " << m.accuracy()
                    << std::endl;
                m.reset_score();
            }
        }

        std::cout << "Final result - " << std::endl;
        m.print();

        std::ofstream params_file{
            std::filesystem::path{"regressor_weight.json"}, 
            std::ios::trunc};
        EDGE_LEARNING_TEST_TRY(m.dump(params_file));
        params_file.close();

        Model m_copy(m);
        EDGE_LEARNING_TEST_EQUAL(m_copy.name(), m.name());
        EDGE_LEARNING_TEST_EQUAL(m_copy.layers().size(), m.layers().size());
        for (std::size_t i = 0; i < m.layers().size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(m_copy.layers()[i]->name(),
                                     m.layers()[i]->name());
        }
    }

    void test_regressor_model_predict() {
        GradientDescentOptimizer o{NumType{0.3}};
        Model m = TestModel::_create_regressor_model();

        std::ifstream params_file{
            std::filesystem::path{"regressor_weight.json"}};
        EDGE_LEARNING_TEST_TRY(m.load(params_file));
        params_file.close();
    }

    void test_recursive_model() {
        // Input definition.
        std::size_t time_steps = 2;

        std::size_t input_size = 3;
        std::vector<std::vector<NumType>> inputs = {
            {10.0, 1.0, 10.0, 1.0, 10.0, 1.0},
            {1.0,  3.0, 8.0,  3.0, 1.0,  3.0,},
            {8.0,  1.0, 8.0,  1.0, 8.0,  1.0,},
            {1.0,  1.5, 8.0,  1.5, 8.0,  1.5},
        };

        std::size_t output_size = 2;
        std::vector<std::vector<NumType>> targets = {
            {1.0, 2.0, 1.0, 2.0},
            {1.0, 2.0, 1.0, 2.0},
            {1.0, 0.0, 1.0, 0.0},
            {1.0, 0.0, 1.0, 0.0},
        };

        // Model definition.
        Model m{"recurrent"};
        auto first_layer = m.add_layer<DenseLayer>(
            "hidden", input_size * time_steps, input_size * time_steps);
        auto first_layer_relu = m.add_layer<ReluLayer>(
            "hidden_relu", input_size * time_steps);
        auto output_layer = m.add_layer<RecurrentLayer>(
            "output", input_size, output_size, 2);
        output_layer->hidden_state({0.01, 0.01});
        output_layer->time_steps(time_steps);
        output_layer->hidden_state({0.0, 0.0});
        auto loss_layer = m.add_loss<MeanSquaredLossLayer>("loss",
            time_steps * output_size, BATCH_SIZE, 0.5);
        GradientDescentOptimizer o{NumType{0.01}};
        m.create_edge(first_layer, first_layer_relu);
        m.create_edge(first_layer_relu, output_layer);
        m.create_loss_edge(output_layer, loss_layer);
        m.init();
        m.print();

        for (std::size_t e = 0; e < EPOCHS; ++e)
        {
            std::cout << "EPOCH " << e << std::endl;
            for (std::size_t i = 0; i < inputs.size();)
            {
                for (std::size_t b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
                {
                    m.step(inputs[i], targets[i]);
                }
                m.train(o);
                std::cout << "Step " << i
                    << " - loss: " << m.avg_loss()
                    << ", accuracy: " << m.accuracy()
                    << std::endl;
                m.reset_score();
            }
        }

        std::cout << "Final result - " << std::endl;
        m.print();

        output_layer->reset_hidden_state();

        Model m_copy(m);
        EDGE_LEARNING_TEST_EQUAL(m_copy.name(), m.name());
        EDGE_LEARNING_TEST_EQUAL(m_copy.layers().size(), m.layers().size());
        for (std::size_t i = 0; i < m.layers().size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(m_copy.layers()[i]->name(),
                                     m.layers()[i]->name());
        }
    }

    Model _create_binary_classifier_model()
    {
        Model m{"binary_classifier"};
        auto first_layer = m.add_layer<DenseLayer>(
            "hidden", 4, 8);
        auto first_layer_relu = m.add_layer<ReluLayer>(
            "hidden_relu", 8);
        auto output_layer = m.add_layer<DenseLayer>(
            "output", 8, 2);
        auto output_layer_softmax = m.add_layer<SoftmaxLayer>(
            "output_softmax", 2);
        auto loss_layer = m.add_loss<CategoricalCrossEntropyLossLayer>(
            "loss", 2, BATCH_SIZE);
        m.create_edge(first_layer, first_layer_relu);
        m.create_edge(first_layer_relu, output_layer);
        m.create_edge(output_layer, output_layer_softmax);
        m.create_loss_edge(output_layer_softmax, loss_layer);
        return m;
    }

    Model _create_regressor_model()
    {
        Model m{"regressor"};
        auto first_layer = m.add_layer<DenseLayer>(
            "hidden", 4, 8);
        auto first_layer_relu = m.add_layer<ReluLayer>(
            "hidden_relu", 8);
        auto output_layer = m.add_layer<DenseLayer>(
            "output", 8, 2);
        auto output_layer_linear = m.add_layer<LinearLayer>(
            "output_linear", 2);
        auto loss_layer = m.add_loss<MeanSquaredLossLayer>(
            "loss", 2, BATCH_SIZE, 0.5);
        m.create_edge(first_layer, first_layer_relu);
        m.create_edge(first_layer_relu, output_layer);
        m.create_edge(output_layer, output_layer_linear);
        m.create_loss_edge(output_layer_linear, loss_layer);
        return m;
    }
};

int main() {
    TestModel().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
