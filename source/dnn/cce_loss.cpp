/***************************************************************************
 *            cce_loss.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "cce_loss.hpp"

#include "dlmath.hpp"

#include <tuple>
#include <cstdio>

namespace Ariadne {

CCELossLayer::CCELossLayer(Model& model, std::string name, 
    uint16_t input_size, size_t batch_size)
    : Layer(model, name)
    , _input_size{input_size}
    , _inv_batch_size{NumType{1.0} / batch_size}
{ 
    /* 
     * When we deliver a gradient back, we deliver just the loss gradient with
     * respect to any input and the index that was "hot" in the second argument.
     */
    _gradients.resize(_input_size);
}

void CCELossLayer::forward(NumType* inputs)
{
    _loss = DLMath::cross_entropy(_target, inputs, _input_size);
    _cumulative_loss += _loss;
    
    auto max = DLMath::max_and_argmax(inputs, _input_size);
    // NumType max_value = std::get<0>(max);
    size_t max_index = std::get<1>(max);

    _active = _argactive();
    if (max_index == _active)
    {
        ++_correct;
    }
    else 
    {
        ++_incorrect;
    }

    // Store the data pointer to compute gradients later.
    _last_input = inputs;
}

void CCELossLayer::reverse(NumType* gradients)
{
    // Parameter ignored because it is a loss layer.
    (void) gradients;

    DLMath::cross_entropy_1(_gradients.data(), _target, _last_input, 
        _inv_batch_size, _input_size);

    for (auto* l: _antecedents)
    {
        l->reverse(_gradients.data());
    }
}

void CCELossLayer::print() const
{
    std::printf("avg loss: %f\t%f%% correct\n", avg_loss(), accuracy() * 100.0);
}

void CCELossLayer::set_target(NumType const* target)
{
    _target = target;
}

NumType CCELossLayer::accuracy() const
{
    return static_cast<NumType>(_correct) 
         / static_cast<NumType>(_correct + _incorrect);
}

NumType CCELossLayer::avg_loss() const
{
    return static_cast<NumType>(_cumulative_loss) 
         / static_cast<NumType>(_correct + _incorrect);
}

void CCELossLayer::reset_score()
{
    _cumulative_loss = 0.0;
    _correct         = 0.0;
    _incorrect       = 0.0;
}

size_t CCELossLayer::_argactive() const
{
    if (_target == nullptr)
    {
        std::runtime_error("_target is null, call set_target before");
    }

    for (size_t i = 0; i < _input_size; ++i)
    {
        if (_target[i] != NumType{0.0})
        {
            return i;
        }
    }
    
    std::runtime_error("_target is an array of 0.0 values");
    return 0;
}

} // namespace Ariadne