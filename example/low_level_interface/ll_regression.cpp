/***************************************************************************
 *            example/low_level_interface/regression.cpp
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

#include "edge_learning.hpp"

using namespace EdgeLearning;

int main()
{
    const SizeType BATCH_SIZE    = 2;
    const SizeType EPOCHS        = 50;
    const SizeType INPUT_SIZE    = 4;
    const SizeType OUTPUT_SIZE   = 2;
    const NumType  LEARNING_RATE = 0.03;

    std::vector<std::vector<NumType>> inputs = {
        {10.0, 1.0, 10.0, 1.0},
        {1.0,  3.0, 8.0,  3.0},
        {8.0,  1.0, 8.0,  1.0},
        {1.0,  1.5, 8.0,  1.5},
    };

    std::vector<std::vector<NumType>> targets = {
        {1.0, 0.0},
        {1.0, 0.4},
        {1.0, 0.0},
        {1.0, 0.4},
    };
    
    // Model definition.
    GDOptimizer o{NumType{LEARNING_RATE}};
    Model m{"regressor"};
    auto first_layer = m.add_layer<DenseLayer>(
        "hidden", Activation::ReLU, 8, INPUT_SIZE);
    auto output_layer = m.add_layer<DenseLayer>(
        "output", Activation::Linear, OUTPUT_SIZE, 8);
    auto loss_layer = m.add_loss<MSELossLayer>(
        "loss", OUTPUT_SIZE, BATCH_SIZE, 0.5);
    m.create_edge(first_layer, output_layer);
    m.create_back_arc(output_layer, loss_layer);

    for (SizeType e = 0; e < EPOCHS; ++e)
    {
        std::cout << "EPOCH " << e << std::endl;
        for (SizeType i = 0; i < inputs.size();)
        {
            for (SizeType b = 0; b < BATCH_SIZE && i < inputs.size(); ++b, ++i)
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
    std::cout << "Training End" << std::endl;

    std::vector<NumType> new_data = {9.0,  1.0, 9.0,  1.0};
    auto result = m.predict(new_data.data());
    std::cout << "Predict: {";
    for (SizeType i = 0; i < m.input_size() - 1; ++i)
    {
        std::cout << new_data[i] << ", ";
    }
    std::cout << new_data[m.input_size() - 1] << "} -> {"<< std::endl;
    for (SizeType i = 0; i < m.output_size() - 1; ++i)
    {
        std::cout << result[i] << ", ";
    }
    std::cout << result[m.output_size() - 1] << "}"<< std::endl;
}