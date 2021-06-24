/***************************************************************************
 *            mse_loss.hpp
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

/*! \file mse_loss.hpp
 *  \brief Mean Squared Error Loss layer.
 */

#ifndef ARIADNE_DNN_MSE_LOSS_HPP
#define ARIADNE_DNN_MSE_LOSS_HPP

#include "layer.hpp"
#include "model.hpp"

#include <string>

namespace Ariadne {

class MSELossLayer : public Layer {
public:
    MSELossLayer(Model& model, std::string name, uint16_t input_size, 
        size_t batch_size, num_t loss_tolerance=0.1);

    /**
     * \brief No initiallization is needed for this layer.
     * \param rne
     */
    void init(rne_t& rne) override { (void) rne; };

    void forward(num_t* inputs) override;

    /**
     * \brief As a loss node, the argument to this method is ignored (the 
     * gradient of the loss with respect to itself is unity).
     * \param inputs
     */
    void reverse(num_t* inputs) override;

    void print() const override;

    /**
     * \brief Set the target object.
     * During training, this must be set to the expected target distribution for 
     * a given sample.
     * \param target
     */
    void set_target(num_t const* target);

    num_t accuracy() const;
    num_t avg_loss() const;
    void reset_score();

private:
    uint16_t _input_size;
    num_t _loss;
    num_t _loss_tolerance;
    const num_t* _target;
    num_t* _last_input;

    std::vector<num_t> _gradients;

    num_t _inv_batch_size; ///< Used to scale with batch size.

    // Last active classification in the target one-hot encoding. 
    size_t _active; 
    num_t _cumulative_loss{0.0};
    
    // Running counts of correct and incorrect predictions.
    size_t _correct{0};
    size_t _incorrect{0};
};

} // namespace Ariadne
 
#endif // ARIADNE_DNN_MSE_LOSS_HPP