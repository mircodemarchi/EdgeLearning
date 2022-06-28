/***************************************************************************
 *            dnn/adam_optimizer.hpp
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

/*! \file  dnn/gd_optimizer.hpp
 *  \brief Adam Optimizer class.
 */

#ifndef EDGE_LEARNING_DNN_ADAM_OPTIMIZER_HPP
#define EDGE_LEARNING_DNN_ADAM_OPTIMIZER_HPP

#include "optimizer.hpp"


namespace EdgeLearning {

/**
 * \brief Class that defines the Adam optimization algorithm.
 *
 * Algorithm:
 * input: dw, eta, b_1, b_2, eps, t
 * output: w
 *
 * v = b_1 * v + (1 - b_1) * dw
 * s = b_2 * s + (1 - b_2) * dw^2
 *
 * v_corrected = v / (1 - b_1^t)
 * s_corrected = s / (1 - b_2^t)
 *
 * w = w - eta * v_corrected / (sqrt(s_corrected) + eps)
 */
class AdamOptimizer : public Optimizer
{
public:
    /**
     * \brief Construct a new AdamOptimizer object.
     * \param eta commonly accepted character used to denote the learning rate.
     * \param beta_1 Exponential decay of the rate for the first moment.
     * \param beta_2 Exponential decay of the rate for the second moment.
     * \param epsilon A value near to zero to prevent division by zero.
     */
    AdamOptimizer(NumType eta,
                  NumType beta_1 = 0.9,
                  NumType beta_2 = 0.999,
                  NumType epsilon = 1e-8);

    /**
     * \brief Optimization invoked at the end of each batch's evaluation.
     * \param layer Layer& Layer to train.
     */
    void train(Layer& layer) override;

    /**
     * \brief Reset timestamp, first and second moment.
     */
    void reset() override;

private:
    NumType _eta;     ///< \brief Learning rate.
    NumType _beta_1;  ///< \brief Exponential decay for the first moment.
    NumType _beta_2;  ///< \brief Exponential decay for the second moment.
    NumType _epsilon; ///< \brief A value near to zero.

    NumType _m;  ///< \brief The first moment (momentum optimizer algorithm).
    NumType _v;  ///< \brief The second moment (RMSProp optimizer algorithm).
    SizeType _t; ///< \brief Incremental timestamp (temperature).
};

} // namespace EdgeLearning
 
#endif // EDGE_LEARNING_DNN_ADAM_OPTIMIZER_HPP