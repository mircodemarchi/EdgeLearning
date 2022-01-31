/***************************************************************************
 *            tests/test_dlmath.cpp
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
    const SizeType BATCH_SIZE = 2;
    const SizeType EPOCHS     = 50;

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
    Model m{"regressor"};
    auto first_layer = m.add_node<DenseLayer>(
        "hidden", Activation::ReLU, 8, 4);
    auto output_layer = m.add_node<DenseLayer>(
        "output", Activation::Linear, 2, 8);
    auto loss_layer = m.add_loss<MSELossLayer>(
        "loss", 2, BATCH_SIZE, 0.5);
    m.create_edge(first_layer, output_layer);
    m.create_edge(output_layer, loss_layer);

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
    std::cout << "End" << std::endl; 
}