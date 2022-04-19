/***************************************************************************
 *            dnn/max_pooling.cpp
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

#include "max_pooling.hpp"

#include <utility>

namespace EdgeLearning {

MaxPoolingLayer::MaxPoolingLayer(
    Model& model, std::string name, Activation activation,
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    DLMath::Shape2d stride)
    : PoolingLayer(model, activation, input_shape, kernel_shape, stride,
                   std::move(name), "max_pooling_layer_")
{}

void MaxPoolingLayer::forward(const NumType *inputs)
{
    // Remember the last input data for backpropagation.
    _last_input = inputs;

    /*
     * Perform convolution with n_filters of kernel size contained in
     * _weights vector on the input 3D matrix.
     */
    DLMath::max_pool<NumType>(_activations.data(), inputs, _input_shape,
                              _kernel_shape, _stride);

    FeedforwardLayer::forward(_activations.data());

    FeedforwardLayer::next();
}

void MaxPoolingLayer::reverse(const NumType *gradients)
{
    FeedforwardLayer::reverse(gradients);

    FeedforwardLayer::previous();
}

} // namespace EdgeLearning
