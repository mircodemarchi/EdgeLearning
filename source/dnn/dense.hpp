/***************************************************************************
 *            dnn/dense.hpp
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

/*! \file  dnn/dense.hpp
 *  \brief Dense layer.
 */

#ifndef EDGE_LEARNING_DNN_DENSE_HPP
#define EDGE_LEARNING_DNN_DENSE_HPP

#include "feedforward.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

class DenseLayer : public FeedforwardLayer
{
public:
    DenseLayer(Model& model,
               std::string name = std::string(),
               SizeType input_size = 0, SizeType output_size = 0);

    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType(std::random_device{}()))
        override;

    /**
     * \brief The input data should have size _input_size.
     * \param inputs
     */
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    /**
     * \brief Weight matrix entries + bias entries.
     * \return SizeType
     */
    [[nodiscard]] SizeType param_count() const noexcept override
    {
        return (_input_size + 1UL) * _output_size;
    }

    NumType& param(SizeType index) override;
    NumType& gradient(SizeType index) override;

    void print() const override;

    /**
     * \brief Getter of input_size class field.
     * \return The size of the layer input.
     */
    [[nodiscard]] virtual SizeType input_size() const override
    {
        return FeedforwardLayer::input_size();
    }

    /**
     * \brief Setter of input_size class field.
     * \param input_size DLMath::Shape3d Shape param used to take the size and
     * assign it to input_size.
     * The operation also performs a resize of the weights and its gradients.
     */
    void input_size(DLMath::Shape3d input_size) override;
private:

    // == Layer parameters ==
    /// \brief Weights of the layer. Size: _output_size * _input_size.
    std::vector<NumType> _weights;
    /// \brief Biases of the layer. Size: _output_size. 
    std::vector<NumType> _biases;

    // == Loss Gradients ==
    /// \brief Weight gradients of the layer. Size: _output_size * _input_size.
    std::vector<NumType> _weight_gradients;
    /// \brief Biase gradients of the layer. Size: _output_size. 
    std::vector<NumType> _bias_gradients;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_DENSE_HPP
