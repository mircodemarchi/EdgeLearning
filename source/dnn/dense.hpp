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
    static const std::string TYPE;

    DenseLayer(std::string name = std::string(),
               SizeType input_size = 0, SizeType output_size = 0);

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType(std::random_device{}()))
        override;

    /**
     * \brief The input data should have size _input_size.
     * \param inputs const std::vector<NumType>& Layer input in forward.
     */
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients const std::vector<NumType>& Layer gradients in backward.
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    /**
     * \brief Weight matrix entries + bias entries.
     * \return SizeType input_size*output_size + bias_size(1)*output_size.
     */
    [[nodiscard]] SizeType param_count() const noexcept override
    {
        return (_input_size + 1UL) * _output_size;
    }

    NumType& param(SizeType index) override;
    NumType& gradient(SizeType index) override;

    [[nodiscard]] SharedPtr clone() const override
    {
        return std::make_shared<DenseLayer>(*this);
    }

    void print() const override;

    /**
     * \brief Save the layer infos and weights to disk.
     * \param out Json& out Json to write.
     */
    void dump(Json& out) const override;

    /**
     * \brief Load the layer infos and weights from disk.
     * \param in const Json& Json to read.
     */
    void load(Json& in) override;

protected:
    /**
     * \brief Setter of input_shape class field.
     * \param input_shape DLMath::Shape3d Shape param used to take the size and
     * assign it to input_shape.
     * The operation also performs a resize of the weights and its gradients.
     */
    void _set_input_shape(LayerShape input_shape) override;

private:

    // == Layer parameters ==
    /// \brief Weights of the layer. Size: _output_size * _input_size.
    SharedParams _weights;
    /// \brief Biases of the layer. Size: _output_size. 
    SharedParams _biases;

    // == Loss Gradients ==
    /// \brief Weight gradients of the layer. Size: _output_size * _input_size.
    Params _weight_gradients;
    /// \brief Biase gradients of the layer. Size: _output_size. 
    Params _bias_gradients;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_DENSE_HPP
