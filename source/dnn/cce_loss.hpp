/***************************************************************************
 *            cce_loss.hpp
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

/*! \file cce_loss.hpp
 *  \brief Categorical Cross-Entropy Loss layer.
 */

#ifndef EDGE_LEARNING_DNN_CCE_LOSS_HPP
#define EDGE_LEARNING_DNN_CCE_LOSS_HPP

#include "loss.hpp"
#include "model.hpp"

#include <string>

namespace EdgeLearning {

class CCELossLayer : public LossLayer {
public:
    /**
     * @brief Construct a new CCELossLayer object.
     * @param model 
     * @param name 
     * @param input_size 
     * @param batch_size 
     */
    CCELossLayer(Model& model, std::string name = std::string(), 
        uint16_t input_size = 0, size_t batch_size = 1);

    void forward(NumType* inputs) override;

    /**
     * \brief As a loss node, the argument to this method is ignored (the 
     * gradient of the loss with respect to itself is unity).
     * \param gradients
     */
    void reverse(NumType* gradients = nullptr) override;

private:
    /**
     * \brief Find the argument of _target array that is active.
     * \return size_t
     */
    size_t _argactive() const;

    // Last active classification in the target one-hot encoding. 
    size_t _active; 
};

} // namespace EdgeLearning
 
#endif // EDGE_LEARNING_DNN_CCE_LOSS_HPP