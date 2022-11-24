/***************************************************************************
 *            dnn/concatenate.cpp
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

#include "concatenate.hpp"

#include "dlmath.hpp"

#include <algorithm>
#include <cstdio>

namespace EdgeLearning {

const std::string ConcatenateLayer::TYPE = "Concatenate";

static inline void concatenate_check_shape(
        std::vector<DLMath::Shape3d> shapes, SizeType axis)
{
    if (shapes.empty())
    {
        throw std::runtime_error("concatenate layer error: empty shapes");
    }
    if (axis >= DLMath::Shape3d::SIZE)
    {
        throw std::runtime_error("concatenate layer error: axis overload");
    }

    for (SizeType shape_idx = 1; shape_idx < shapes.size(); ++shape_idx)
    {
        for (SizeType i = 0; i < DLMath::Shape3d::SIZE; ++i)
        {
            if (i != axis && shapes[shape_idx - 1][i] != shapes[shape_idx][i])
            {
                throw std::runtime_error("concatenate layer error: "
                                         "shapes invalid.");
            }
        }
    }
}

static inline DLMath::Shape3d concatenate_input_shape(
        std::vector<DLMath::Shape3d> shapes, SizeType axis)
{
    concatenate_check_shape(shapes, axis);
    DLMath::Shape3d ret(shapes[0]);
    for (SizeType shape_idx = 1; shape_idx < shapes.size(); ++shape_idx)
    {
        if (ret[axis] < shapes[shape_idx][axis])
        {
            ret[axis] = shapes[shape_idx][axis];
        }
    }
    return ret;
}

static inline DLMath::Shape3d concatenate_output_shape(
        std::vector<DLMath::Shape3d> shapes, SizeType axis)
{
    concatenate_check_shape(shapes, axis);
    DLMath::Shape3d ret(shapes[0]);
    for (SizeType shape_idx = 1; shape_idx < shapes.size(); ++shape_idx)
    {
        ret[axis] += shapes[shape_idx][axis];
    }
    return ret;
}

ConcatenateLayer::ConcatenateLayer(std::string name,
                                   std::vector<DLMath::Shape3d> shapes,
                                   SizeType axis)
    : FeedforwardLayer(concatenate_input_shape(shapes, axis),
                       concatenate_output_shape(shapes, axis),
                       std::move(name), "concatenate_layer_")
    , _axis{axis}
    , _current_input_layer{0}
    , _current_output_shape{0, 0, 0}
{

}

const std::vector<NumType>& ConcatenateLayer::forward(
    const std::vector<NumType>& inputs)
{
    if (_current_input_layer == 0)
    {
        _current_output_shape = input_shapes()[0];
    }

    DLMath::append<NumType>(
        _output_activations.data(), _shared_fields->output_shape().shape(),
        inputs.data(), input_shapes().at(_current_input_layer).at(_axis),
        _axis, _current_output_shape[_axis]);

    if (_current_input_layer > 0)
    {
        _current_output_shape[_axis] +=
            input_shapes()[_current_input_layer].at(_axis);
    }

    if (_current_input_layer < input_layers())
    {
        _current_input_layer++;
        return _output_activations;
    }
    _current_input_layer = 0;
    return FeedforwardLayer::forward(_output_activations);
}

const std::vector<NumType>& ConcatenateLayer::backward(
    const std::vector<NumType>& gradients)
{
    (void) gradients;

    return _input_gradients;
}

void ConcatenateLayer::print() const 
{
    std::cout << _shared_fields->name() << std::endl;
    std::cout << "No learnable parameters" << std::endl;
    std::cout << std::endl;
}

Json ConcatenateLayer::dump() const
{
    return FeedforwardLayer::dump();
}

void ConcatenateLayer::load(const Json& in)
{
    FeedforwardLayer::load(in);
}

void ConcatenateLayer::_set_input_shape(LayerShape input_shape)
{
    FeedforwardLayer::input_shape(input_shape);
}

} // namespace EdgeLearning
