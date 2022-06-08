/***************************************************************************
 *            dnn/pooling.cpp
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

#include "pooling.hpp"

#include <utility>

namespace EdgeLearning {

static inline SizeType pooling_output_side(
    SizeType input_side, SizeType kernel_side, SizeType stride
    )
{
    return input_side == 0 ? 0 : ((input_side - kernel_side) / stride) + 1ULL;
}

static inline DLMath::Shape3d pooling_output_shape(
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    DLMath::Shape2d stride
)
{
    return {
        pooling_output_side(input_shape.height, kernel_shape.height,
                            stride.height),
        pooling_output_side(input_shape.width, kernel_shape.width,
                            stride.width),
        input_shape.channels};
}

PoolingLayer::PoolingLayer(
    Model& model, DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    DLMath::Shape2d stride, std::string name, std::string prefix_name)
    : FeedforwardLayer(
        model, input_shape,
        pooling_output_shape(input_shape, kernel_shape, stride), std::move(name),
        prefix_name.empty() ? "pooling_layer_" : prefix_name)
    , _kernel_shape(kernel_shape)
    , _stride(stride)
{}

void PoolingLayer::print() const
{
    std::cout << _name << std::endl;
    std::cout << "No learnable parameters" << std::endl;
    std::cout << std::endl;
}

void PoolingLayer::input_shape(DLMath::Shape3d input_shape)
{
    FeedforwardLayer::input_shape(input_shape);

    // Update input and output shape accordingly (see this constructor).
    _input_shape = input_shape;
    _output_shape = pooling_output_shape(input_shape, _kernel_shape, _stride);

    // Update output size accordingly (see Layer and FeedforwardLayer constr.).
    _output_activations.resize(output_size());
}

void PoolingLayer::dump(Json& out) const
{
    FeedforwardLayer::dump(out);

    Json others;
    std::vector<std::size_t> kernel_size = {
        _kernel_shape.height, _kernel_shape.width
    };
    others["kernel_size"] = Json(kernel_size);
    std::vector<std::size_t> stride = { _stride.height, _stride.width };
    others["stride"] = Json(stride);

    out[dump_fields.at(DumpFields::OTHERS)] = others;
}

void PoolingLayer::load(Json& in)
{
    FeedforwardLayer::load(in);

    auto kernel_size = in[dump_fields.at(DumpFields::OTHERS)]["kernel_size"]
        .as_vec<std::size_t>();
    _kernel_shape = DLMath::Shape2d(kernel_size.at(0), kernel_size.at(1));
    auto stride = in[dump_fields.at(DumpFields::OTHERS)]["stride"]
        .as_vec<std::size_t>();
    _stride = DLMath::Shape2d(stride.at(0), stride.at(1));
}

} // namespace EdgeLearning
