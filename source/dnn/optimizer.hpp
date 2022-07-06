/***************************************************************************
 *            dnn/optimizer.hpp
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

/*! \file  dnn/optimizer.hpp
 *  \brief Optimizer interface.
 */

#ifndef EDGE_LEARNING_DNN_OPTIMIZER_HPP
#define EDGE_LEARNING_DNN_OPTIMIZER_HPP

#include "layer.hpp"


namespace EdgeLearning {

/**
 * \brief Base class of optimizer used to train a model.
 */
class Optimizer
{
public:
    Optimizer() {}

    /**
     * \brief Wrapper of private train method:
     * run the optimization process taking the gradients from a layer
     * and apply the optimization to another layer.
     * \param layer_from Layer from taking the gradients.
     * \param layer_to   Layer to apply the optimization.
     */
    virtual void train(Layer& layer_from, Layer& layer_to);

    /**
     * \brief Run the optimization process to layer.
     * \param layer The layer to optimize.
     */
    virtual void train(Layer& layer);

    /**
     * \brief Run the optimization process as the \a train method but it
     * throws a runtime_error if layers have a different amount of training
     * parameters.
     * \param layer_from Layer from taking the gradients.
     * \param layer_to   Layer to apply the optimization.
     */
    virtual void train_check(Layer& layer_from, Layer& layer_to);

    /**
     * \brief Reset optimizer internal state.
     */
    virtual void reset() {};

protected:
    /**
     * \brief Run the optimization process taking the gradients from a layer
     * and apply the optimization to another layer.
     * \param layer_from Layer from taking the gradients.
     * \param layer_to   Layer to apply the optimization.
     */
    virtual void _train(Layer& layer_from, Layer& layer_to) = 0;

};

} // namespace EdgeLearning
 
#endif // EDGE_LEARNING_DNN_OPTIMIZER_HPP