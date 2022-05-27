/***************************************************************************
 *            dnn/dropout.hpp
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

/*! \file  dnn/dropout.hpp
 *  \brief Dropout layer.
 */

#ifndef EDGE_LEARNING_DNN_DROPOUT_HPP
#define EDGE_LEARNING_DNN_DROPOUT_HPP

#include "feedforward.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

class DropoutLayer : public FeedforwardLayer
{
public:
    static const std::string TYPE;

    DropoutLayer(Model& model,
                 std::string name = std::string(), SizeType size = 0,
                 NumType drop_probability = 0.1,
                 RneType random_generator = RneType(std::random_device{}()));

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    /**
     * \brief No initialization is needed for Pooling layers.
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

    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override
    {
        _last_input = inputs.data();
        return FeedforwardLayer::forward(inputs);
    }

    /**
     * \brief Dropout layer's training forward has a different implementation
     * of the forward method because the Dropout becomes useful only during
     * training. Instead the forward method only pass the inputs to the next
     * layers, as the default method implement.
     * \param inputs const std::vector<NumType>& The inputs to dropout.
     * \return const std::vector<NumType>& The inputs with some neuron
     * dropped out.
     */
    const std::vector<NumType>& training_forward(
        const std::vector<NumType>& inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients const std::vector<NumType>&
     * \return const std::vector<NumType>&
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    /**
     * \brief Dropout layer doesn't have any learnable parameters.
     * \return SizeType 0.
     */
    [[nodiscard]] SizeType param_count() const noexcept override
    {
        return 0;
    }

    /**
     * \brief Dropout layers do not have params.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& param(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Dropout layers do not have params");
    }

    /**
     * \brief Dropout layers do not have gradients.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& gradient(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Dropout layers do not have gradients");
    }

    void print() const override;

    [[nodiscard]] const DLMath::Shape3d& input_shape() const override
    {
        return FeedforwardLayer::input_shape();
    }
    void input_shape(DLMath::Shape3d input_shape) override;

private:

    NumType _drop_probability;  ///< @brief Probability to drop a value.
    NumType _scale;             ///< @brief scale = 1 / (1 - drop_p).
    RneType _random_generator;  ///< @brief Random generator.

    /// @brief Vector of indexes for gradients to set to zero.
    std::vector<SizeType> _zero_mask_idxs;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_DROPOUT_HPP
