/***************************************************************************
 *            dense.cpp
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

#include "dense.hpp"

#include "dlmath.i.hpp"


namespace Ariadne {

DenseLayer::DenseLayer(Model& model, std::string name, Activation activation, 
    uint16_t output_size, uint16_t input_size)
    : Layer(model, std::move(name))
    , _activation{activation}
    , _output_size{output_size}
    , _input_size{input_size}
{
    
}

void DenseLayer::init(rne_t& rne)
{
    num_t sigma;
    switch (_activation)
    {
        case Activation::ReLU:
        {   
            /*
             * Kaiming He, et. al. weight initialization for ReLU networks 
             * https://arxiv.org/pdf/1502.01852.pdf
             * Suggests using a normal distribution with variance := 2 / n_in
             */
            sigma = std::sqrt(2.0 / static_cast<num_t>(_input_size));
            break;
        }
        case Activation::Softmax:
        default:
        {
            /* 
             * LeCun initialization for "Self-Normalizing Neural Networks"
             * https://arxiv.org/pdf/1706.02515.pdf
             * Suggests using a normal distribution with variance := 1 / n_in
             */
            sigma = std::sqrt(1.0 / static_cast<num_t>(_input_size));
            break;
        }

        /*
         * The C++ standard does not guarantee that the results obtained from a 
         * distribution function will be identical given the same inputs across 
         * different compilers and platforms, therefore I use my own 
         * distributions to provide deterministic results.
         */
        auto dist = dlmath::normal_pdf<num_t>(0.0, sigma);

        for (num_t& w: _weights)
        {
            w = dist(rne);
        }

        /*
         * Setting biases to zero is a common practice, as is initializing the 
         * bias to a small value (e.g. on the order of 0.01). The thinking is
         * that a non-zero bias will ensure that the neuron always "fires" at 
         * the beginning to produce a signal.
         */
        for (num_t& b: _biases)
        {
            b = 0.01; ///< You can try also with 0.0 or other strategies.
        }
    }
}

void DenseLayer::forward(num_t* inputs) 
{
    // Remember the last input data for backpropagation.
    _last_input = inputs;

    /* 
     * Compute the product of the input data with the weight add the bias.
     * z = W * x + b
     */
    dlmath::matarr_mul<num_t>(_activations.data(), _weights.data(), inputs, 
        _output_size, _input_size);
    dlmath::arr_sum<num_t>(_activations.data(), _activations.data(), 
        _biases.data(), _output_size);

    switch (_activation)
    {
        case Activation::ReLU:
        {
            dlmath::relu<num_t>(_activations.data(), _activations.data(), 
                size_t(_output_size));
        }
        case Activation::Softmax:
        default:
        {
            dlmath::softmax<num_t>(_activations.data(), _activations.data(), 
                size_t(_output_size));
        }
    }

    // Forward to the next layers.
    for (auto *layer: this->_subsequents)
    {
        layer->forward(_activations.data());
    }
}

void DenseLayer::reverse(num_t* gradients)
{
    // Calculate dg(z)/dz and put in _activation_gradients.
    switch (_activation)
    {
        case Activation::ReLU:
        {
            /*
             * The input for ReLU derivation is the _activations vector, that 
             * is filled with the ReLU of vector z, and not directly the vector
             * z. Considering that the input of ReLU derivation is used only 
             * to check if it is > 0, and that if z > 0 then ReLU(z) > 0 and 
             * viceversa, using ReLU of vector z or using directly the vector z
             * there is no differences.  
             */
            dlmath::relu_1<num_t>(
                _activation_gradients.data(), 
                _activations.data(), 
                _output_size);
        }
        case Activation::Softmax:
        default:
        {
            /*
             * The softmax derivation explits the calculus of softmax performed 
             * previously and saved in _activations vector.
             */
            dlmath::softmax_1_opt<num_t>(
                _activation_gradients.data(),
                _activations.data(), 
                _output_size);
        }
    }

    // Calculate dJ/dz = dJ/dg(z) * dg(z)/dz.
    dlmath::arr_mul(_activation_gradients.data(), _activation_gradients.data(),
        gradients, _output_size);

    /*
     * Bias gradient.
     * Calculate dJ/db = dJ/dg(z) * dg(z)/db
     *                 = dJ/dg(z) * dg(z)/dz * dz/db            <- z = Wx+b
     *                 = dJ/dg(z) * dg(Wx+b)/dz * d(Wx+b)/db 
     *                 = dJ/dg(z) * dg(Wx+b)/dz * 1
     *                 = dJ/dg(z) * dg(z)/dz
     *                 = dJ/dz
     */
    dlmath::arr_sum(_bias_gradients.data(), _bias_gradients.data(), 
        _activation_gradients.data(), _output_size);

    /*
     * Weight gradient.
     * Calculate dJ/dw_i_j = dJ/dg(z) * dg(z)/dw_i_j
     *                     = dJ/dg(z) * dg(z)/dz * dz/dw_i_j    <- z = Wx+b
     *                     = dJ/dg(z) * dg(Wx+b)/dz * d(Wx+b)/dw_i_j 
     *                     = dJ/dg(z) * dg(Wx+b)/dz * x_j
     *                     = dJ/dg(z) * dg(z)/dz * x_j
     *                     = dJ/dz * x_j
     */
    for (size_t i = 0; i < _output_size; ++i)
    {
        for (size_t j = 0; j < _input_size; ++j)
        {
            _weight_gradients[(i * _input_size) + j] += 
                _activation_gradients[i] * _last_input[j];
        }
    }

    /* 
     * Input gradient.
     * Calculate dJ/dx = dJ/dg(z) * dg(z)/x
     *                 = dJ/dg(z) * dg(z)/dz * dz/x             <- z = Wx+b
     *                 = dJ/dg(z) * dg(Wx+b)/dz * d(Wx+b)/x 
     *                 = dJ/dg(z) * dg(Wx+b)/dz * W
     *                 = dJ/dg(z) * dg(z)/dz * W
     *                 = dJ/dz * W
     */
    std::fill(_input_gradients.begin(), _input_gradients.end(), 0);
    for (size_t i = 0; i < _output_size; ++i)
    {
        for (size_t j = 0; j < _input_size; ++j)
        {
            _input_gradients[j] += 
                _activation_gradients[i] * _weights[(i * _input_size) + j];
        }
    }
}

num_t* DenseLayer::param(size_t index)
{

}

num_t* DenseLayer::gradient(size_t index)
{

}

void DenseLayer::print() const 
{

}

} // namespace Ariadne
