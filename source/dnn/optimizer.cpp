/***************************************************************************
 *            dnn/optimizer.cpp
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

#include "optimizer.hpp"

#include <stdexcept>

namespace EdgeLearning {

void Optimizer::train(Layer& layer_from, Layer& layer_to)
{
    _train(layer_from, layer_to);
}

void Optimizer::train(Layer& layer)
{
    train(layer, layer);
}

void Optimizer::train_check(Layer& layer_from, Layer& layer_to)
{
    if (layer_from.param_count() != layer_to.param_count())
    {
        throw std::runtime_error("Layers have different amount of params");
    }
    train(layer_from, layer_to);
}

} // namespace EdgeLearning
