/***************************************************************************
 *            gd_optimizer.hpp
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

/*! \file gd_optimizer.hpp
 *  \brief Gradient Descent Optimizer class.
 */

#ifndef ARIADNE_DNN_GD_OPTIMIZER_HPP
#define ARIADNE_DNN_GD_OPTIMIZER_HPP

#include "optimizer.hpp"


namespace Ariadne {

/**
 * \brief Class that defines the general gradient descent algorithm
 * It can be used as part of the *Stochastic* gradient descent algorithm (SGD) 
 * by invoking it after smaller batches of training data are evaluated.
 */
class GDOptimizer : public Optimizer
{
public:
    /**
     * \brief Construct a new GDOptimizer object.
     * Given a loss gradient dL/dp for some parameter p, during gradient 
     * descent, p will be adjusted such that p' = p - eta * dL/dp.
     * \param eta commonly accepted character used to denote the learning rate.
     */
    GDOptimizer(NumType eta);

    /**
     * \brief Invoked at the end of each batch's evaluation.
     * The interface technically permits the use of different optimizers for
     * different segments of the computational graph.
     * \param layer
     */
    void train(Layer& layer) override;

private:
    NumType _eta; ///< Learning rate.
};

} // namespace Ariadne
 
#endif // ARIADNE_DNN_GD_OPTIMIZER_HPP