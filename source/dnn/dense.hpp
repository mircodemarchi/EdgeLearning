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

#include "layer.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

enum class Activation
{
    ReLU,
    Softmax,
    Linear
};

class DenseLayer : public Layer 
{
public: 
    DenseLayer(Model& model,
               std::string name = std::string(),
               Activation activation = Activation::ReLU,
               SizeType output_size = 0, SizeType input_size = 0);

    void init(RneType& rne) override;

    /**
     * \brief The input data should have size _input_size.
     * \param inputs
     */
    void forward(const NumType *inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients
     */
    void reverse(const NumType *gradients) override;

    const NumType* last_input() override;
    const NumType* last_output() override;

    /**
     * \brief Weight matrix entries + bias entries.
     * \return SizeType
     */
    SizeType param_count() const noexcept override
    {
        return (_input_size + 1UL) * _output_size;
    }

    NumType* param(SizeType index) override;
    NumType* gradient(SizeType index) override;

    void print() const override;

    [[nodiscard]] SizeType input_size() const override
    {
        return Layer::input_size();
    }
    void input_size(SizeType input_size) override;

private:
    Activation _activation;

    // == Layer parameters ==
    /// \brief Weights of the layer. Size: _output_size * _input_size.
    std::vector<NumType> _weights;
    /// \brief Biases of the layer. Size: _output_size. 
    std::vector<NumType> _biases;
    /// \brief Activations of the layer. Size: _output_size. 
    std::vector<NumType> _activations;

    // == Loss Gradients ==
    /// \brief Weight gradients of the layer. Size: _output_size * _input_size.
    std::vector<NumType> _weight_gradients;
    /// \brief Biase gradients of the layer. Size: _output_size. 
    std::vector<NumType> _bias_gradients;
    /// \brief Activation gradients of the layer. Size: _output_size. 
    std::vector<NumType> _activation_gradients;
    /**
     * \brief Input gradients of the layer. Size: _input_size. 
     * This buffer is used to store temporary gradients used in a **singe** 
     * backpropagation pass. Note that this does not accumulate like the weight 
     * and bias gradients do.
     */
    std::vector<NumType> _input_gradients;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_DENSE_HPP
