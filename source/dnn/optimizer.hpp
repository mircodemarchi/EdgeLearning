/***************************************************************************
 *            optimizer.hpp
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

/*! \file optimizer.hpp
 *  \brief Optimizer interface.
 */

#ifndef ARIADNE_DNN_OPTIMIZER_HPP
#define ARIADNE_DNN_OPTIMIZER_HPP

#include "layer.hpp"


namespace Ariadne {

/**
 * \brief Base class of optimizer used to train a model.
 */
class Optimizer
{
public:
    virtual void train(Layer& layer) = 0;
};

} // namespace Ariadne
 
#endif // ARIADNE_DNN_OPTIMIZER_HPP