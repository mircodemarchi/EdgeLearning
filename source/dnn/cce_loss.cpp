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

namespace Ariadne {

CCELossLayer::CCELossLayer(Model& model, std::string name, 
    uint16_t input_size, size_t batch_size)
    : Layer(model, name)
    , _input_size{input_size}
    , _inv_batch_size{num_t{1.0} / batch_size}
{ 
    /* 
     * When we deliver a gradient back, we deliver just the loss gradient with
     * respect to any input and the index that was "hot" in the second argument.
     */
    _gradients.resize(_input_size);
}

void CCELossLayer::forward(num_t* inputs)
{

}

void CCELossLayer::reverse(num_t* inputs)
{

}

void CCELossLayer::print() const
{

}

void CCELossLayer::set_target(num_t const* target)
{

}

num_t CCELossLayer::accuracy() const
{

}

num_t CCELossLayer::avg_loss() const
{

}

void CCELossLayer::reset_score()
{

}

size_t CCELossLayer::argactive() const
{
    if (_target == nullptr)
    {
        std::runtime_error("_target is null, call set_target before");
    }

    for (size_t i = 0; i < _input_size; ++i)
    {
        if (_target[i] != num_t{0.0})
        {
            return i;
        }
    }
    
    std::runtime_error("_target is an array of 0.0 values");
}

} // namespace Ariadne