/***************************************************************************
 *            tests/test_dlmath.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
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
using namespace Ariadne;

class TestModel {
public:
    void test() {
        ARIADNE_TEST_CALL(test_classifier_model());
        ARIADNE_TEST_CALL(test_classifier_model_predict());
        ARIADNE_TEST_CALL(test_regressor_model());
        ARIADNE_TEST_CALL(test_regressor_model_predict());
        ARIADNE_TEST_CALL(test_recurisive_model());
    }

private:
    const size_t BATCH_SIZE = 2;
    const size_t EPOCHS     = 50;

    void test_classifier_model() {
        // Input definition.
        NumType* input  = nullptr;
        NumType* target = nullptr;

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
        DenseLayer* input_layer;
        CCELossLayer* loss_layer;
        GDOptimizer o{NumType{0.5}};
        Model m = TestModel::_create_binary_classifier_model(&input_layer, 
            &loss_layer);
        m.init();
        m.print();

        for (size_t e = 0; e < EPOCHS; ++e)
        {
            std::printf("EPOCH %zu\n", e);
            for (size_t i = 0; i < inputs.size();)
            {
                loss_layer->reset_score();

                for (size_t b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
                {
                    input = inputs[i].data();
                    target = targets[i].data();
                    loss_layer->set_target(target);
                    input_layer->forward(input);
                    loss_layer->reverse();
                }

                std::printf("Step %zu - ", i);
                loss_layer->print();

                m.train(o);
            }
        }

        std::printf("Final result - ");
        loss_layer->print();
        m.print();

        std::ofstream params_file{
            std::filesystem::path{"classifier.weight"}, 
            std::ios::binary};
        m.save(params_file);
    }

    void test_classifier_model_predict() {
        DenseLayer* input_layer;
        CCELossLayer* loss_layer;
        GDOptimizer o{NumType{0.3}};
        Model m = TestModel::_create_binary_classifier_model(&input_layer, 
            &loss_layer);

        std::ifstream params_file{
            std::filesystem::path{"classifier.weight"}, 
            std::ios::binary};
        m.load(params_file);
    }

    void test_regressor_model() {
        // Input definition.
        NumType* input  = nullptr;
        NumType* target = nullptr;

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
        DenseLayer* input_layer;
        MSELossLayer* loss_layer;
        GDOptimizer o{NumType{0.01}};
        Model m = TestModel::_create_regressor_model(&input_layer, 
            &loss_layer);
        m.init();
        m.print();

        for (size_t e = 0; e < EPOCHS; ++e)
        {
            std::printf("EPOCH %zu\n", e);
            for (size_t i = 0; i < inputs.size();)
            {
                loss_layer->reset_score();

                for (size_t b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
                {
                    input = inputs[i].data();
                    target = targets[i].data();
                    loss_layer->set_target(target);
                    input_layer->forward(input);
                    loss_layer->reverse();
                }

                std::printf("Step %zu - ", i);
                loss_layer->print();

                m.train(o);
            }
        }

        std::printf("Final result - ");
        loss_layer->print();
        m.print();

        std::ofstream params_file{
            std::filesystem::path{"regressor.weight"}, 
            std::ios::binary};
        m.save(params_file);
    }

    void test_regressor_model_predict() {
        DenseLayer* input_layer;
        MSELossLayer* loss_layer;
        GDOptimizer o{NumType{0.3}};
        Model m = TestModel::_create_regressor_model(&input_layer, 
            &loss_layer);

        std::ifstream params_file{
            std::filesystem::path{"regressor.weight"}, 
            std::ios::binary};
        m.load(params_file);
    }

    void test_recurisive_model() {
        // Input definition.
        NumType* input  = nullptr;
        NumType* target = nullptr;
        size_t time_steps = 2;

        size_t input_size = 3;
        std::vector<std::vector<NumType>> inputs = {
            {10.0, 1.0, 10.0, 1.0, 10.0, 1.0},
            {1.0,  3.0, 8.0,  3.0, 1.0,  3.0,},
            {8.0,  1.0, 8.0,  1.0, 8.0,  1.0,},
            {1.0,  1.5, 8.0,  1.5, 8.0,  1.5},
        };

        size_t output_size = 2;
        std::vector<std::vector<NumType>> targets = {
            {1.0, 2.0, 1.0, 2.0},
            {1.0, 2.0, 1.0, 2.0},
            {1.0, 0.0, 1.0, 0.0},
            {1.0, 0.0, 1.0, 0.0},
        };
        
        // Model definition.
        Model m{"recurrent"};
        RecurrentLayer& input_layer = m.add_node<RecurrentLayer>("hidden", 
            output_size, input_size, 2);
        input_layer.set_initial_hidden_state({0.01, 0.01});
        input_layer.set_time_steps(time_steps);
        input_layer.set_initial_hidden_state({0.0, 0.0});
        MSELossLayer& loss_layer = m.add_node<MSELossLayer>("loss", 
            time_steps * output_size, BATCH_SIZE, 0.5);
        GDOptimizer o{NumType{0.01}};
        m.create_edge(loss_layer, input_layer);
        m.init();
        m.print();

        for (size_t e = 0; e < EPOCHS; ++e)
        {
            std::printf("EPOCH %zu\n", e);
            for (size_t i = 0; i < inputs.size();)
            {
                loss_layer.reset_score();

                for (size_t b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
                {
                    input = inputs[i].data();
                    target = targets[i].data();
                    loss_layer.set_target(target);
                    input_layer.forward(input);
                    loss_layer.reverse();
                }

                std::printf("Step %zu - ", i);
                loss_layer.print();

                m.train(o);
            }
        }

        std::printf("Final result - ");
        loss_layer.print();
        m.print();

        input_layer.reset_hidden_state();
    }

    Model _create_binary_classifier_model(DenseLayer** first_layer, 
        CCELossLayer** loss_layer)
    {
        Model m{"binary_classifier"};
        *first_layer = &m.add_node<DenseLayer>("hidden", 
            Activation::ReLU, 8, 4);
        DenseLayer& output_layer = m.add_node<DenseLayer>("output", 
            Activation::Softmax, 2, 8);

        *loss_layer = &m.add_node<CCELossLayer>("loss", 2, BATCH_SIZE);
        m.create_edge(output_layer, **first_layer);
        m.create_edge(**loss_layer, output_layer);
        return m;
    }

    Model _create_regressor_model(DenseLayer** first_layer, 
        MSELossLayer** loss_layer)
    {
        Model m{"regressor"};
        *first_layer = &m.add_node<DenseLayer>("hidden", 
            Activation::ReLU, 8, 4);
        DenseLayer& output_layer = m.add_node<DenseLayer>("output", 
            Activation::Linear, 2, 8);

        *loss_layer = &m.add_node<MSELossLayer>("loss", 2, BATCH_SIZE, 0.5);
        m.create_edge(output_layer, **first_layer);
        m.create_edge(**loss_layer, output_layer);
        return m;
    }
};

int main() {
    TestModel().test();
    return ARIADNE_TEST_FAILURES;
}
