/***************************************************************************
 *            dense.hpp
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

/*! \file dense.hpp
 *  \brief Dense layer.
 */

#ifndef ARIADNE_DNN_DENSE_HPP
#define ARIADNE_DNN_DENSE_HPP

#include "layer.hpp"

#include <string>
#include <vector>


namespace Ariadne {

enum class Activation
{
    ReLU,
    Softmax,
    Linear
};

class DenseLayer : public Layer 
{
public: 
    DenseLayer(Model& model, std::string name, Activation activation, 
        uint16_t output_size, uint16_t input_size);

    void init(RneType& rne) override;

    /**
     * \brief The input data should have size _input_size.
     * \param inputs
     */
    void forward(NumType* inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients
     */
    void reverse(NumType* gradients) override;

    /**
     * \brief Weight matrix entries + bias entries.
     * \return size_t
     */
    size_t param_count() const noexcept override
    {
        return (_input_size + 1UL) * _output_size;
    }

    NumType* param(size_t index) override;
    NumType* gradient(size_t index) override;

    void print() const override;

private:
    Activation _activation;
    uint16_t _output_size;
    uint16_t _input_size;

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
    /**
     * \brief The last input passed to the layer. It is needed to compute loss 
     * gradients with respect to the weights during backpropagation.
     */
    NumType* _last_input;
};

} // namespace Ariadne

#endif // ARIADNE_DNN_DENSE_HPP
