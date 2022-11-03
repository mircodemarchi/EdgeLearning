/***************************************************************************
 *            dnn/mse_loss.hpp
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

/*! \file  dnn/mse_loss.hpp
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
    static const std::string TYPE;

    MSELossLayer(std::string name = std::string(),
                 SizeType input_size = 0, SizeType batch_size = 1,
                 NumType loss_tolerance=0.1);

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;

    /**
     * \brief As a loss node, the argument to this method is ignored (the 
     * gradient of the loss with respect to itself is unity).
     * \param gradients
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    [[nodiscard]] SharedPtr clone() const override
    {
        return std::make_shared<MSELossLayer>(*this);
    }

    /**
     * \brief Save the layer infos to disk.
     * \return Json Layer dump.
     */
    Json dump() const override;

    /**
     * \brief Load the layer infos from disk.
     * \param in const Json& Json to read.
     */
    void load(const Json& in) override;
    
private:
    /// \brief Tollerange to ensure that the prediction produced is correct.
    NumType _loss_tolerance;
};

} // namespace EdgeLearning
 
#endif // EDGE_LEARNING_DNN_MSE_LOSS_HPP