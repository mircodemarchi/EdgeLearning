/***************************************************************************
 *            dnn/mse_loss.cpp
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

#include "mse_loss.hpp"

#include "dlmath.hpp"

namespace EdgeLearning {

MSELossLayer::MSELossLayer(Model& model, std::string name, 
    SizeType input_size, SizeType batch_size, NumType loss_tolerance)
    : LossLayer(model, input_size, batch_size,
                std::move(name), "mse_loss_layer_")
    , _loss_tolerance{loss_tolerance}
{ 

}

void MSELossLayer::forward(const NumType *inputs)
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

void MSELossLayer::reverse(const NumType *gradients)
{
    // Parameter ignored because it is a loss layer.
    (void) gradients;

    DLMath::mean_squared_error_1(_gradients.data(), _target, _last_input, 
        _inv_batch_size, _input_size);

    for (auto l: _antecedents)
    {
        l->reverse(_gradients.data());
    }
}

} // namespace EdgeLearning