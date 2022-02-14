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


namespace EdgeLearning {

/**
 * \brief Base class of computational LossLayers in a model.
 */
class LossLayer : public Layer 
{
public:
    /**
     * \brief Construct a new LossLayer object.
     * \param model
     * \param name
     * \param input_size
     * \param batch_size
     * \param prefix_name
     */
    LossLayer(Model& model,
        SizeType input_size = 0, SizeType batch_size = 1,
        std::string name = std::string(),
        std::string prefix_name = std::string());

    /**
     * \brief Destroy the LossLayer object.
     */
    virtual ~LossLayer() {};

    /**
     * \brief No initialization is needed for loss layers.
     * \param rne Not used.
     */
    void init(RneType& rne) override { (void) rne; };

    /**
     * \brief Method inheritance.
     * \param inputs
     */
    virtual void forward(NumType* inputs) override = 0;

    /**
     * \brief Method inheritance with default parameter overriding. 
     * \param gradients
     */
    virtual void reverse(NumType* gradients = nullptr) override = 0;

    /**
     * \brief Loss layers do not have params.
     * \return SizeType 0
     */
    [[nodiscard]] SizeType param_count() const noexcept override { return 0; }

    /**
     * \brief Loss layers do not have params.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType* param(SizeType index) override { (void) index; return nullptr; }

    /**
     * \brief Loss layers do not have gradients.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType* gradient(SizeType index) override { (void) index; return nullptr; }

    std::vector<NumType> last_input() override;
    std::vector<NumType> last_output() override;

    /**
     * \brief Setter of the target object.
     * During training, this must be set to the expected target distribution for 
     * a given sample.
     * \param target
     */
    void set_target(const NumType* target);

    /**
     * \brief Calculate and return the accuracy until the last forward
     * iteration.
     * \return NumType
     */
    NumType accuracy() const;

    /**
     * \brief Calculate and return the average loss until the last forward
     * iteration.
     * \return NumType
     */
    NumType avg_loss() const;

    /**
     * \brief Reset loss statistics.
     */
    void reset_score();

    /**
     * \brief Print loss info.
     */
    virtual void print() const override;

protected:
    SizeType _input_size;
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
    SizeType _correct;
    SizeType _incorrect;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_LOSS_HPP
