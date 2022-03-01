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
    const SizeType BATCH_SIZE    = 1;
    const SizeType EPOCHS        = 100;
    const SizeType INPUT_SIZE    = 4;
    const SizeType OUTPUT_SIZE   = 2;
    const NumType  LEARNING_RATE = 0.03;

    Dataset<NumType>::Mat data = {
        {10.0, 1.0, 10.0, 1.0, 1.0, 0.0},
        {1.0,  3.0, 8.0,  3.0, 1.0, 0.4},
        {8.0,  1.0, 8.0,  1.0, 1.0, 0.0},
        {1.0,  1.5, 8.0,  1.5, 1.0, 0.4},
    };
    Dataset<NumType> dataset(data, 1, {4, 5});

    LayerDescriptorVector layers_descriptor(
        {
            {"input_layer",   INPUT_SIZE,  Activation::Linear },
            {"hidden_layer1", 8UL,         Activation::ReLU   },
            {"hidden_layer2", 32UL,        Activation::ReLU   },
            {"hidden_layer3", 16UL,        Activation::ReLU   },
            {"output_layer",  OUTPUT_SIZE, Activation::Linear }
        }
    );
    SizeType input_size = 4;
    FNN<Framework::EDGE_LEARNING,
        LossType::MSE,
        OptimizerType::GRADIENT_DESCENT,
        InitType::AUTO> m(layers_descriptor, "regressor_model");
    m.fit(dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE);
    std::cout << "Training End" << std::endl;

    Dataset<NumType> new_data({9.0,  1.0, 9.0,  1.0}, INPUT_SIZE);
    auto result = m.predict(new_data);
    std::cout << "Predict: {";
    for (SizeType i = 0; i < INPUT_SIZE - 1; ++i)
    {
        std::cout << new_data.data()[i] << ", ";
    }
    std::cout << new_data.data()[INPUT_SIZE - 1] << "} -> {";
    for (SizeType i = 0; i < OUTPUT_SIZE - 1; ++i)
    {
        std::cout << result.data()[i] << ", ";
    }
    std::cout << result.data()[OUTPUT_SIZE - 1] << "}"<< std::endl;
}