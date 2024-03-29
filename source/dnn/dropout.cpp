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

const std::string DropoutLayer::TYPE = "Dropout";

DropoutLayer::DropoutLayer(std::string name, SizeType size,
                           NumType drop_probability, RneType random_generator)
    : FeedforwardLayer(size, size, std::move(name), "dropout_layer_")
    , _drop_probability{drop_probability}
    , _scale{(_drop_probability == 1.0) ? 1.0 : 1.0 / (1.0 - drop_probability)}
    , _random_generator{random_generator}
    , _zero_mask_idxs{}
{

}

const std::vector<NumType>& DropoutLayer::training_forward(
    const std::vector<NumType>& inputs)
{
    Layer::training_forward(inputs);
    auto dist = DLMath::uniform_pdf<NumType>(0.5, 1.0);

    // Input size is equal to the output size.
    _zero_mask_idxs.clear();
    for (SizeType i = 0; i < output_size(); ++i)
    {
        auto random_value = dist(_random_generator);
        if (random_value > _drop_probability)
        {
            _output_activations[i] = inputs[i] * _scale;
        }
        else
        {
            _output_activations[i] = NumType(0.0);
            _zero_mask_idxs.push_back(i);
        }
    }

    return FeedforwardLayer::forward(_output_activations);
}

const std::vector<NumType>& DropoutLayer::backward(
    const std::vector<NumType>& gradients)
{
    // Input size is equal to the output size.
    DLMath::arr_mul(_input_gradients.data(), gradients.data(),
                    _scale, input_size());
    for (const auto& i: _zero_mask_idxs)
    {
        _input_gradients[i] = 0;
    }

    return FeedforwardLayer::backward(_input_gradients);
}

void DropoutLayer::print() const
{
    std::cout << _shared_fields->name() << std::endl;
    std::cout << "No learnable parameters" << std::endl;
    std::cout << std::endl;
}

Json DropoutLayer::dump() const
{
    Json out = FeedforwardLayer::dump();

    Json others;
    others["drop_probability"] = _drop_probability;
    out[dump_fields.at(DumpFields::OTHERS)] = others;
    return out;
}

void DropoutLayer::load(const Json& in)
{
    FeedforwardLayer::load(in);

    _drop_probability = in.at(dump_fields.at(DumpFields::OTHERS))
        .at("drop_probability").as<NumType>();
    _scale = (_drop_probability == 1.0) ? 1.0 : 1.0 / (1.0 - _drop_probability);
}

void DropoutLayer::_set_input_shape(LayerShape input_shape)
{
    FeedforwardLayer::_set_input_shape(input_shape);
    _shared_fields->output_shape() = input_shape;
    _output_activations.resize(output_size());
}

} // namespace EdgeLearning
