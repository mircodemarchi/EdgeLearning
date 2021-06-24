/***************************************************************************
 *            mse_loss.cpp
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

#include "mse_loss.hpp"

#include "dlmath.hpp"

namespace Ariadne {

MSELossLayer::MSELossLayer(Model& model, std::string name, 
    uint16_t input_size, size_t batch_size, num_t loss_tolerance)
    : Layer(model, name)
    , _input_size{input_size}
    , _inv_batch_size{num_t{1.0} / batch_size}
    , _loss_tolerance{loss_tolerance}
{ 
    /* 
     * When we deliver a gradient back, we deliver just the loss gradient with
     * respect to any input and the index that was "hot" in the second argument.
     */
    _gradients.resize(_input_size);
}

void MSELossLayer::forward(num_t* inputs)
{
    _loss = DLMath::mean_squared_error(_target, inputs, _input_size);
    _cumulative_loss += _loss;

    if (-_loss_tolerance <= _loss && _loss <= _loss_tolerance)
    {
        _correct++;
    }
    else 
    {
        _incorrect++;
    }

    // Store the data pointer to compute gradients later.
    _last_input = inputs;
}

void MSELossLayer::reverse(num_t* inputs)
{
    // Parameter ignored because it is a loss layer.
    (void) inputs;

    DLMath::mean_squared_error_1(_gradients.data(), _target, _last_input, 
        _inv_batch_size, _input_size);

    for (auto *l: _antecedents)
    {
        l->reverse(_gradients.data());
    }
}

void MSELossLayer::print() const
{
    std::printf("Avg Loss: %f\t%f%% correct\n", avg_loss(), accuracy() * 100.0);
}

void MSELossLayer::set_target(num_t const* target)
{
    _target = target;
}

num_t MSELossLayer::accuracy() const
{
    return static_cast<num_t>(_correct) 
         / static_cast<num_t>(_correct + _incorrect);
}

num_t MSELossLayer::avg_loss() const
{
    return static_cast<num_t>(_cumulative_loss) 
         / static_cast<num_t>(_correct + _incorrect);
}

void MSELossLayer::reset_score()
{
    _cumulative_loss = 0.0;
    _correct         = 0.0;
    _incorrect       = 0.0;
}


} // namespace Ariadne