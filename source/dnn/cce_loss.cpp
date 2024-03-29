/***************************************************************************
 *            dnn/cce_loss.cpp
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

#include "cce_loss.hpp"

#include "dlmath.hpp"

#include <tuple>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace EdgeLearning {

const std::string CategoricalCrossEntropyLossLayer::TYPE = "CCELoss";

CategoricalCrossEntropyLossLayer::CategoricalCrossEntropyLossLayer(
    std::string name,
    SizeType input_size,
    SizeType batch_size)
    : LossLayer(input_size,
                batch_size,
                std::move(name),
                "cce_loss_layer_")
    , _active{}
{ 

}

const std::vector<NumType>& CategoricalCrossEntropyLossLayer::forward(
    const std::vector<NumType>& inputs)
{
    SizeType in_size = inputs.size();

    if (_target.empty())
    {
        throw std::runtime_error("_target is empty, set_target not called");
    }
    _loss = DLMath::cross_entropy(_target.data(), inputs.data(), in_size);
    _cumulative_loss += _loss;
    
    auto max = DLMath::max_and_argmax(inputs.data(), in_size);
    // NumType max_value = std::get<0>(max);
    SizeType max_index = std::get<1>(max);

    _active = _argactive();
    if (max_index == _active)
    {
        ++_correct;
    }
    else 
    {
        ++_incorrect;
    }

    // No more forward.
    return inputs;
}

const std::vector<NumType>& CategoricalCrossEntropyLossLayer::backward(
    const std::vector<NumType>& gradients)
{
    // Parameter ignored because it is a loss layer.
    (void) gradients;

    DLMath::cross_entropy_1(_gradients.data(), _target.data(), _last_input,
        _inv_batch_size, _gradients.size());

    return LossLayer::backward(_gradients);
}

SizeType CategoricalCrossEntropyLossLayer::_argactive() const
{
    for (SizeType i = 0; i < _shared_fields->input_size(); ++i)
    {
        if (_target[i] != NumType{0.0})
        {
            return i;
        }
    }

    throw std::runtime_error("_target is an array of 0.0 values");
}

} // namespace EdgeLearning