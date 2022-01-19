/***************************************************************************
 *            mse_loss.hpp
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

/*! \file mse_loss.hpp
 *  \brief Mean Squared Error Loss layer.
 */

#ifndef EDGE_LEARNING_DNN_MSE_LOSS_HPP
#define EDGE_LEARNING_DNN_MSE_LOSS_HPP

#include "loss.hpp"

#include "model.hpp"

#include <string>

namespace EdgeLearning {

class MSELossLayer : public LossLayer {
public:
    MSELossLayer(Model& model, std::string name, uint16_t input_size, 
        size_t batch_size, NumType loss_tolerance=0.1);

    /**
     * \brief No initiallization is needed for this layer.
     * \param rne
     */
    void init(RneType& rne) override { (void) rne; };

    void forward(NumType* inputs) override;

    /**
     * \brief As a loss node, the argument to this method is ignored (the 
     * gradient of the loss with respect to itself is unity).
     * \param gradients
     */
    void reverse(NumType* gradients = nullptr) override;
    
private:
    /// @brief Tollerange to ensure that the prediction produced is correct. 
    NumType _loss_tolerance;
};

} // namespace EdgeLearning
 
#endif // EDGE_LEARNING_DNN_MSE_LOSS_HPP