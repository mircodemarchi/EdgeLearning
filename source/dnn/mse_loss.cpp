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

const std::string MeanSquaredLossLayer::TYPE = "MSELoss";

MeanSquaredLossLayer::MeanSquaredLossLayer(std::string name,
                                           SizeType input_size, SizeType batch_size, NumType loss_tolerance)
    : LossLayer(input_size, batch_size,
                std::move(name), "mse_loss_layer_")
    , _loss_tolerance{loss_tolerance}
{ 

}

const std::vector<NumType>& MeanSquaredLossLayer::forward(
    const std::vector<NumType>& inputs)
{
    SizeType in_size = inputs.size();

    if (_target == nullptr)
    {
        throw std::runtime_error("_target is null, set_target not called");
    }
    _loss = DLMath::mean_squared_error(_target, inputs.data(), in_size);
    _cumulative_loss += _loss;

    if (-_loss_tolerance <= _loss && _loss <= _loss_tolerance)
    {
        _correct++;
    }
    else 
    {
        _incorrect++;
    }

    // No more forward.
    return inputs;
}

const std::vector<NumType>& MeanSquaredLossLayer::backward(
    const std::vector<NumType>& gradients)
{
    // Parameter ignored because it is a loss layer.
    (void) gradients;

    DLMath::mean_squared_error_1(_gradients.data(), _target, _last_input,
        _inv_batch_size, _gradients.size());

    return LossLayer::backward(_gradients);
}

Json MeanSquaredLossLayer::dump() const
{
    Json out = LossLayer::dump();

    Json others;
    others["loss_tolerance"] = _loss_tolerance;
    out[dump_fields.at(DumpFields::OTHERS)] = others;
    return out;
}

void MeanSquaredLossLayer::load(const Json& in)
{
    LossLayer::load(in);

    _loss_tolerance = in.at(dump_fields.at(DumpFields::OTHERS))
        .at("loss_tolerance").as<NumType>();
}

} // namespace EdgeLearning