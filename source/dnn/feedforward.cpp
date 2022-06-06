/***************************************************************************
 *            dnn/feedforward.cpp
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

#include "feedforward.hpp"


namespace EdgeLearning {

FeedforwardLayer::FeedforwardLayer(
    Model& model, DLMath::Shape3d input_shape, DLMath::Shape3d output_shape,
    std::string name, std::string prefix_name)
    : Layer(model, input_shape, output_shape, std::move(name),
            prefix_name.empty() ? "feedforward_layer_" : prefix_name)
    , _output_activations{}
    , _input_gradients{}
{
    _output_activations.resize(output_shape.size());
    _input_gradients.resize(input_shape.size());
}

void FeedforwardLayer::input_shape(DLMath::Shape3d input_shape)
{
    Layer::input_shape(input_shape);
    _input_gradients.resize(input_shape.size());
}

void FeedforwardLayer::dump(Json& out) const
{
    Layer::dump(out);
}

void FeedforwardLayer::load(Json& in)
{
    Layer::load(in);
    _output_activations.resize(output_size());
    _input_gradients.resize(input_size());
}

} // namespace EdgeLearning
