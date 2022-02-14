/***************************************************************************
 *            time_estimator.cpp
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

#include "middleware/feed_forward.hpp"

#include <utility>

namespace EdgeLearning {

FeedForward::FeedForward(
    std::map<std::string, std::tuple<SizeType, Activation>> layers,
    LossType loss, OptimizerType optimizer,
    std::string name)
    : _layers{std::move(layers)}
    , _loss{loss}
    , _optimizer{optimizer}
    , _name{name}
#if ENABLE_MLPACK
#else
    , _m{name}
#endif
{ }

} // namespace EdgeLearning
