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

#include "layer.hpp"
#include "model.hpp"

#include <string>

namespace EdgeLearning {

class CCELossLayer : public Layer {
public:
    CCELossLayer(Model& model, std::string name, uint16_t input_size, 
        size_t batch_size);

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

    void print() const override;

    /**
     * \brief Set the target object.
     * During training, this must be set to the expected target distribution for 
     * a given sample.
     * \param target
     */
    void set_target(NumType const* target);

    NumType accuracy() const;
    NumType avg_loss() const;
    void reset_score();

private:
    /**
     * \brief Find the argument of _target array that is active.
     * \return size_t
     */
    size_t _argactive() const;

    uint16_t _input_size;
    NumType _loss;
    const NumType* _target;
    NumType* _last_input;

    std::vector<NumType> _gradients;

    NumType _inv_batch_size; ///< Used to scale with batch size.

    // Last active classification in the target one-hot encoding. 
    size_t _active; 
    NumType _cumulative_loss{0.0};
    
    // Running counts of correct and incorrect predictions.
    size_t _correct{0};
    size_t _incorrect{0};
};

} // namespace EdgeLearning
 
#endif // EDGE_LEARNING_DNN_CCE_LOSS_HPP