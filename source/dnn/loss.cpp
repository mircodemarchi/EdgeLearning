/***************************************************************************
 *            dnn/loss.cpp
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

#include "loss.hpp"

#include "dlmath.hpp"


namespace EdgeLearning {

const std::string LossLayer::TYPE = "Loss";

LossLayer::LossLayer(
    SizeType input_size, SizeType batch_size,
    std::string name, std::string prefix_name)
    : Layer(std::move(name), input_size, 0,
            prefix_name.empty() ? "loss_layer_" : prefix_name)
    , _loss{}
    , _target{}
    , _gradients{}
    , _inv_batch_size{
        NumType{1.0} / static_cast<NumType>(std::max(batch_size, SizeType{1}))}
    , _cumulative_loss{0.0}
    , _correct{0}
    , _incorrect{0}
{
    _gradients.resize(input_size);
}

void LossLayer::set_target(const std::vector<NumType>& target)
{
    _target = target.data();
}

NumType LossLayer::accuracy() const
{
    return static_cast<NumType>(_correct) 
         / static_cast<NumType>(_correct + _incorrect);
}

NumType LossLayer::avg_loss() const
{
    return static_cast<NumType>(_cumulative_loss) 
         / static_cast<NumType>(_correct + _incorrect);
}

void LossLayer::reset_score()
{
    _cumulative_loss = 0.0;
    _correct         = 0;
    _incorrect       = 0;
}

void LossLayer::print() const
{
    std::cout << "Avg Loss: " << avg_loss()
        << "\t" << (accuracy() * 100.0) << "%% correct" << std::endl;
}

void LossLayer::dump(Json& out) const
{
    Layer::dump(out);
}

void LossLayer::load(Json& in)
{
    Layer::load(in);
    _gradients.resize(input_size());
}

void LossLayer::_set_input_shape(LayerShape input_shape) {
    Layer::_set_input_shape(input_shape);
    _gradients.resize(input_shape.size());
}

} // namespace EdgeLearning
