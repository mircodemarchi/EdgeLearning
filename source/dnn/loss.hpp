/***************************************************************************
 *            loss.hpp
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

/*! \file loss.hpp
 *  \brief High level loss layer of a deep neural network.
 */

#ifndef EDGE_LEARNING_DNN_LOSS_HPP
#define EDGE_LEARNING_DNN_LOSS_HPP

#include "layer.hpp"
#include "model.hpp"


namespace EdgeLearning {

class Model;

/**
 * \brief Base class of computational LossLayers in a model.
 */
class LossLayer : public Layer 
{
public:
    /**
     * @brief Construct a new LossLayer object.
     * @param model 
     * @param name 
     */
    LossLayer(Model& model, std::string name = std::string(), 
        uint16_t input_size = 0, size_t batch_size = 1);

    /**
     * @brief Destroy the LossLayer object.
     */
    virtual ~LossLayer() {};

    /**
     * \brief No initialization is needed for loss layers.
     * \param rne Not used.
     */
    void init(RneType& rne) override { (void) rne; };

    /**
     * @brief Loss layers do not have params.
     * @return size_t 0
     */
    size_t param_count() const noexcept override { return 0; }

    /**
     * @brief Loss layers do not have params.
     * @param index Not used.
     * @return NumType* nullptr
     */
    NumType* param(size_t index) override { (void) index; return nullptr; }

    /**
     * @brief Loss layers do not have gradients.
     * @param index Not used.
     * @return NumType* nullptr
     */
    NumType* gradient(size_t index) override { (void) index; return nullptr; }

    /**
     * \brief Setter of the target object.
     * During training, this must be set to the expected target distribution for 
     * a given sample.
     * \param target
     */
    void set_target(const NumType* target);

    /**
     * @brief Calculate and return the accuracy until the last forward 
     * iteration.
     * @return NumType 
     */
    NumType accuracy() const;

    /**
     * @brief Calculate and return the average loss until the last forward 
     * iteration.
     * @return NumType 
     */
    NumType avg_loss() const;

    /**
     * @brief Reset loss statistics. 
     */
    void reset_score();

    /**
     * @brief Print loss info.
     */
    virtual void print() const override;

protected:
    uint16_t _input_size;
    NumType _loss;
    const NumType* _target;
    NumType* _last_input;

    /**
     * The loss delivered back gradient with respect to any input.
     */
    std::vector<NumType> _gradients;

    NumType _inv_batch_size; ///< Used to scale with batch size.

    /// Last active classification in the target one-hot encoding. 
    NumType _cumulative_loss;
    
    /// Running counts of correct and incorrect predictions.
    size_t _correct;
    size_t _incorrect;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_LOSS_HPP
