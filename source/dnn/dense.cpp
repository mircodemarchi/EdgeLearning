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
    // Remember the last input data for backpropagation later.
    _last_input = inputs;

    /* 
     * Compute the product of the input data with the weight add the bias.
     * z = W * x + b
     */
    for (size_t i = 0; i < _output_size; ++i)
    {
        num_t z{0.0};

        size_t offset = i * _input_size;
        for (size_t j = 0; j < _input_size; ++j)
        {
            z += _weights[offset + j] * inputs[j];
        }
        z += _biases[i];

        // Save the result in activations vector.
        _activations[i] = z;
    }

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
