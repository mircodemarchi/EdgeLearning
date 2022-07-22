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

const std::string MaxPoolingLayer::TYPE = "MaxPool";

MaxPoolingLayer::MaxPoolingLayer(
    std::string name, DLMath::Shape3d input_shape,
    DLMath::Shape2d kernel_shape, DLMath::Shape2d stride)
    : PoolingLayer(input_shape, kernel_shape, stride,
                   std::move(name), "max_pooling_layer_")
{}

const std::vector<NumType>& MaxPoolingLayer::forward(
    const std::vector<NumType>& inputs)
{
    // Remember the last input data for backpropagation.
    _last_input = inputs.data();

    DLMath::max_pool<NumType>(_output_activations.data(), inputs.data(),
                              _input_shape, _kernel_shape, _stride);

    return PoolingLayer::forward(_output_activations);
}

const std::vector<NumType>& MaxPoolingLayer::backward(
    const std::vector<NumType>& gradients)
{
    std::fill(_input_gradients.begin(), _input_gradients.end(), 0);
    auto gradients_op = [&](
        NumType* dst, DLMath::Shape2d dst_shape, DLMath::Coord2d dst_coord,
        const NumType* src, DLMath::Shape3d src_shape,
        const NumType* k, DLMath::Shape2d k_shape, SizeType n_filters,
        int64_t row, int64_t col)
    {
        // TODO: evaluate to optimize using the activations vector and the calculated max
        (void) dst;
        (void) k;
        (void) n_filters;
        auto src_step = src_shape.width * src_shape.channels;
        auto dst_step = dst_shape.width * src_shape.channels;
        for (SizeType c = 0; c < src_shape.channels; ++c)
        {
            auto output_gradient = gradients[
                dst_coord.row * dst_step
                + dst_coord.col * src_shape.channels
                + c];

            NumType max = src[row * static_cast<int64_t>(src_step)
                              + col + static_cast<int64_t>(c)];
            int64_t max_row = 0, max_col = 0;
            for (SizeType k_i = 1; k_i < k_shape.height * k_shape.width; ++k_i)
            {
                auto row_k = k_i / k_shape.width;
                auto col_k = k_i % k_shape.width;
                auto row_src = (row + static_cast<int64_t>(row_k))
                    * static_cast<int64_t>(src_step);
                auto col_src = col
                    + static_cast<int64_t>(col_k * src_shape.channels)
                    + static_cast<int64_t>(c);
                auto curr_val = src[row_src + col_src];
                if (curr_val > max)
                {
                    max = curr_val;
                    max_row = row_src;
                    max_col = col_src;
                }
            }
            _input_gradients[static_cast<std::size_t>(max_row + max_col)]
                += output_gradient;
        }
    };
    DLMath::kernel_slide<NumType>(
        gradients_op, nullptr, _last_input, _input_shape,
        nullptr, _kernel_shape,
        1, _stride, {0, 0});

    return PoolingLayer::backward(_input_gradients);
}

} // namespace EdgeLearning
