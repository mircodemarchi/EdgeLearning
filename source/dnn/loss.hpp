/***************************************************************************
 *            dnn/loss.hpp
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

/*! \file  dnn/loss.hpp
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
    static const std::string TYPE;

    /**
     * \brief Construct a new LossLayer object.
     * \param input_size
     * \param batch_size
     * \param name
     * \param prefix_name
     */
    LossLayer(
        SizeType input_size = 0, SizeType batch_size = 1,
        std::string name = std::string(),
        std::string prefix_name = std::string());

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    /**
     * \brief Destroy the LossLayer object.
     */
    virtual ~LossLayer() {};

    /**
     * \brief No initialization is needed for loss layers.
     * \param init  Not used.
     * \param pdf   Not used.
     * \param rne   Not used.
     */
    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType(std::random_device{}()))
        override
    {
        (void) init;
        (void) pdf;
        (void) rne;
    };

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
    NumType& param(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Loss layers do not have params");
    }

    /**
     * \brief Loss layers do not have gradients.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& gradient(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Loss layers do not have gradients");
    }

    /**
     * \brief Setter of the target object.
     * During training, this must be set to the expected target distribution for 
     * a given sample.
     * \param target
     */
    void set_target(const std::vector<NumType>& target);

    /**
     * \brief Calculate and return the accuracy until the last forward
     * iteration.
     * \return NumType
     */
    [[nodiscard]] NumType accuracy() const;

    /**
     * \brief Calculate and return the average loss until the last forward
     * iteration.
     * \return NumType
     */
    [[nodiscard]] NumType avg_loss() const;

    /**
     * \brief Reset loss statistics.
     */
    void reset_score();

    /**
     * \brief Print loss info.
     */
    virtual void print() const override;

    const std::vector<NumType>& last_input_gradient() override
    {
        return _gradients;
    }

    /**
     * \brief Loss layers do not have output.
     * \return No return, always throw a std::runtime_error.
     */
    const std::vector<NumType>& last_output() override
    {
        throw std::runtime_error("Loss layers do not have output");
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

protected:

    void _set_input_shape(LayerShape input_shape) override;

    NumType _loss;
    Params _target;

    /**
     * The loss delivered back gradient with respect to any input.
     */
    Params _gradients;

    NumType _inv_batch_size; ///< Used to scale with batch size.

    /// Last active classification in the target one-hot encoding. 
    NumType _cumulative_loss;
    
    /// Running counts of correct and incorrect predictions.
    SizeType _correct;
    SizeType _incorrect;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_LOSS_HPP
