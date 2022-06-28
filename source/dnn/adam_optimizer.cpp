/***************************************************************************
 *            dnn/adam_optimizer.cpp
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

#include "adam_optimizer.hpp"

#include <cmath>

namespace EdgeLearning {

AdamOptimizer::AdamOptimizer(
    NumType eta, NumType beta_1, NumType beta_2, NumType epsilon)
    : Optimizer()
    , _eta{eta}
    , _beta_1{beta_1}
    , _beta_2{beta_2}
    , _epsilon{epsilon}
    , _m{0.0}
    , _v{0.0}
    , _t{1}
{ }

void AdamOptimizer::train(Layer& layer)
{
    SizeType param_count = layer.param_count();
    for (SizeType i = 0; i < param_count; ++i)
    {
        NumType& param    = layer.param(i);
        NumType& gradient = layer.gradient(i);

        // beta 1 - Momentum optimizer.
        _m = _beta_1 * _m + (1 - _beta_1) * gradient;

        // beta 2 - RMSProp optimizer.
        _v = _beta_2 * _v + (1 - _beta_2) * (gradient * gradient);

        // bias correction.
        auto m_corrected = _m / (1 - std::pow(_beta_1, _t));
        auto v_corrected = _v / (1 - std::pow(_beta_2, _t));

        param -= _eta * (m_corrected / (std::sqrt(v_corrected) + _epsilon));

        ++_t;

        // Reset the gradient accumulated again in the next training epoch.
        gradient = NumType{0.0};
    }
}

void AdamOptimizer::reset()
{
    _t = 1;
    _m = 0;
    _v = 0;
}

} // namespace EdgeLearning
