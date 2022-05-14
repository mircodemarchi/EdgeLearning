/***************************************************************************
 *            dnn/dropout.cpp
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

#include "dropout.hpp"

#include "dlmath.hpp"

#include <algorithm>
#include <cstdio>

namespace EdgeLearning {

DropoutLayer::DropoutLayer(Model& model, std::string name,
                           Activation activation, SizeType size,
                           NumType drop_probability, RneType random_generator)
    : FeedforwardLayer(model, size, size, activation,
                       std::move(name), "dropout_layer_")
    , _drop_probability{drop_probability}
    , _scale{(_drop_probability == 1.0) ? 1.0 : 1.0 / (1.0 - drop_probability)}
    , _random_generator{random_generator}
    , _zero_mask_idxs{}
{

}

const std::vector<NumType>& DropoutLayer::training_forward(
    const std::vector<NumType>& inputs)
{
    Layer::_check_training_input(inputs);
    // Last input not used for backpropagation.
    _last_input = inputs.data();

    auto dist = DLMath::uniform_pdf<NumType>(0.5, 1.0);

    // Input size is equal to the output size.
    _zero_mask_idxs.clear();
    for (SizeType i = 0; i < _output_size; ++i)
    {
        auto random_value = dist(_random_generator);
        if (random_value > _drop_probability)
        {
            _activations[i] = inputs[i] * _scale;
        }
        else
        {
            _activations[i] = NumType(0.0);
            _zero_mask_idxs.push_back(i);
        }
    }

    FeedforwardLayer::forward(_activations);
    return Layer::forward(_activations);
}

const std::vector<NumType>& DropoutLayer::backward(
    const std::vector<NumType>& gradients)
{
    FeedforwardLayer::backward(gradients);

    // Input size is equal to the output size.
    DLMath::arr_mul(_input_gradients.data(), _activation_gradients.data(),
                    _scale, _input_size);
    for (const auto& i: _zero_mask_idxs)
    {
        _input_gradients[i] = 0;
    }

    return Layer::backward(_input_gradients);
}

void DropoutLayer::print() const
{
    std::cout << _name << std::endl;
    std::cout << "No learnable parameters" << std::endl;
    std::cout << std::endl;
}

void DropoutLayer::input_size(DLMath::Shape3d input_size)
{
    FeedforwardLayer::input_size(input_size);
    _output_size = input_size.size();
    _activations.resize(_output_size);
    _activation_gradients.resize(_output_size);
}

} // namespace EdgeLearning
