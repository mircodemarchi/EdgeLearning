/***************************************************************************
 *            tests/dnn/test_model.cpp
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
#include "dnn/recurrent.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/gd_optimizer.hpp"

#include <filesystem>

using namespace std;
using namespace EdgeLearning;

class TestModel {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_model());
        EDGE_LEARNING_TEST_CALL(test_classifier_model());
        EDGE_LEARNING_TEST_CALL(test_classifier_model_predict());
        EDGE_LEARNING_TEST_CALL(test_regressor_model());
        EDGE_LEARNING_TEST_CALL(test_regressor_model_predict());
        EDGE_LEARNING_TEST_CALL(test_recurisive_model());
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

        EDGE_LEARNING_TEST_EXECUTE(auto m_noname = Model{});
        EDGE_LEARNING_TEST_TRY(auto m_noname = Model{});
        Model m_noname{};
        EDGE_LEARNING_TEST_ASSERT(!m_noname.name().empty());

        EDGE_LEARNING_TEST_EXECUTE(Model m_copy{m});
        EDGE_LEARNING_TEST_TRY(Model m_copy{m});
        Model m_copy{m};
        EDGE_LEARNING_TEST_EQUAL(m_copy.name(), "model");

        EDGE_LEARNING_TEST_EXECUTE(Model m_assign; m_assign = m);
        EDGE_LEARNING_TEST_TRY(Model m_assign; m_assign = m);
        Model m_assign; m_assign = m;
        EDGE_LEARNING_TEST_EQUAL(m_assign.name(), "model");
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
        GDOptimizer o{NumType{0.5}};
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
                    m.step(inputs[i].data(), targets[i].data());
                }

                std::cout << "Step " << i 
                    << " - loss: " << m.avg_loss()
                    << ", accuracy: " << m.accuracy()
                    << std::endl;
                m.train(o);
            }
        }

        std::cout << "Final result - " << std::endl;
        m.print();

        std::ofstream params_file{
            std::filesystem::path{"classifier.weight"}, 
            std::ios::binary};
        m.save(params_file);
    }

    void test_classifier_model_predict() {
        GDOptimizer o{NumType{0.3}};
        Model m = TestModel::_create_binary_classifier_model();

        std::ifstream params_file{
            std::filesystem::path{"classifier.weight"}, 
            std::ios::binary};
        m.load(params_file);
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
        GDOptimizer o{NumType{0.01}};
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
                    m.step(inputs[i].data(), targets[i].data());
                }

                std::cout << "Step " << i 
                    << " - loss: " << m.avg_loss()
                    << ", accuracy: " << m.accuracy()
                    << std::endl;
                m.train(o);
            }
        }

        std::cout << "Final result - " << std::endl;
        m.print();

        std::ofstream params_file{
            std::filesystem::path{"regressor.weight"}, 
            std::ios::binary};
        m.save(params_file);
    }

    void test_regressor_model_predict() {
        GDOptimizer o{NumType{0.3}};
        Model m = TestModel::_create_regressor_model();

        std::ifstream params_file{
            std::filesystem::path{"regressor.weight"}, 
            std::ios::binary};
        m.load(params_file);
    }

    void test_recurisive_model() {
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
        auto input_layer = m.add_layer<RecurrentLayer>("hidden",
                                                        static_cast<SizeType>(output_size),
                                                        static_cast<SizeType>(input_size), 2);
        input_layer->set_initial_hidden_state({0.01, 0.01});
        input_layer->set_time_steps(time_steps);
        input_layer->set_initial_hidden_state({0.0, 0.0});
        auto loss_layer = m.add_loss<MSELossLayer>("loss",
            static_cast<SizeType>(time_steps * output_size), BATCH_SIZE, 0.5);
        GDOptimizer o{NumType{0.01}};
        m.create_edge(input_layer, loss_layer);
        m.init();
        m.print();

        for (std::size_t e = 0; e < EPOCHS; ++e)
        {
            std::cout << "EPOCH " << e << std::endl;
            for (std::size_t i = 0; i < inputs.size();)
            {
                for (std::size_t b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
                {
                    m.step(inputs[i].data(), targets[i].data());
                }

                std::cout << "Step " << i 
                    << " - loss: " << m.avg_loss()
                    << ", accuracy: " << m.accuracy()
                    << std::endl;
                m.train(o);
            }
        }

        std::cout << "Final result - " << std::endl;
        m.print();

        input_layer->reset_hidden_state();
    }

    Model _create_binary_classifier_model()
    {
        Model m{"binary_classifier"};
        auto first_layer = m.add_layer<DenseLayer>(
                "hidden", Activation::ReLU, 8, 4);
        auto output_layer = m.add_layer<DenseLayer>(
                "output", Activation::Softmax, 2, 8);
        auto loss_layer = m.add_loss<CCELossLayer>(
            "loss", 2, BATCH_SIZE);
        m.create_edge(first_layer, output_layer);
        m.create_edge(output_layer, loss_layer);
        return m;
    }

    Model _create_regressor_model()
    {
        Model m{"regressor"};
        auto first_layer = m.add_layer<DenseLayer>(
                "hidden", Activation::ReLU, 8, 4);
        auto output_layer = m.add_layer<DenseLayer>(
                "output", Activation::Linear, 2, 8);
        auto loss_layer = m.add_loss<MSELossLayer>(
            "loss", 2, BATCH_SIZE, 0.5);
        m.create_edge(first_layer, output_layer);
        m.create_edge(output_layer, loss_layer);
        return m;
    }
};

int main() {
    TestModel().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
