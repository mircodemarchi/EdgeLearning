/***************************************************************************
 *            dnn/dense.cpp
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

#include "dense.hpp"

#include "dlmath.hpp"

#include <algorithm>
#include <cstdio>

namespace EdgeLearning {

DenseLayer::DenseLayer(Model& model, std::string name, Activation activation, 
    SizeType input_size, SizeType output_size)
    : Layer(model, input_size, output_size, std::move(name))
    , _activation{activation}
{
    // std::cout << _name << ": " << input_size
    //    << " -> " << output_size << std::endl;

    // The weight parameters of a FF-layer are an NxM matrix.
    _weights.resize(output_size * input_size);

    // Each node in this layer is assigned a bias.
    _biases.resize(output_size);

    // The outputs of each neuron within the layer is an "activation".
    _activations.resize(output_size);

    _activation_gradients.resize(output_size);
    _weight_gradients.resize(output_size * input_size);
    _bias_gradients.resize(output_size);
    _input_gradients.resize(input_size);
}

void DenseLayer::init(RneType& rne)
{
    NumType sigma;
    switch (_activation)
    {
        case Activation::ReLU:
        {   
            /*
             * Kaiming He, et. al. weight initialization for ReLU networks 
             * https://arxiv.org/pdf/1502.01852.pdf
             * Nrmal distribution with variance := sqrt( 2 / n_in )
             */
            sigma = std::sqrt(2.0 / static_cast<NumType>(_input_size));
            break;
        }
        case Activation::Softmax:
        case Activation::Linear:
        default:
        {
            /* 
             * Xavier initialization
             * https://arxiv.org/pdf/1706.02515.pdf
             * Normal distribution with variance := sqrt( 1 / n_in )
             */
            sigma = std::sqrt(1.0 / static_cast<NumType>(_input_size));
            break;
        }
    }

    /*
     * The C++ standard does not guarantee that the results obtained from a 
     * distribution function will be identical given the same inputs across 
     * different compilers and platforms, therefore I use my own 
     * distributions to provide deterministic results.
     */
    auto dist = DLMath::normal_pdf<NumType>(0.0, sigma);

    for (NumType& w: _weights)
    {
        w = dist(rne);
    }

    /*
     * Setting biases to zero is a common practice, as is initializing the 
     * bias to a small value (e.g. on the order of 0.01). The thinking is
     * that a non-zero bias will ensure that the neuron always "fires" at 
     * the beginning to produce a signal.
     */
    for (NumType& b: _biases)
    {
        b = 0.01; ///< You can try also with 0.0 or other strategies.
    }
}

void DenseLayer::forward(const NumType *inputs)
{
    // Remember the last input data for backpropagation.
    _last_input = inputs;

    /* 
     * Compute the product of the input data with the weight add the bias.
     * z = W * x + b
     */
    DLMath::matarr_mul<NumType>(_activations.data(), _weights.data(), inputs, 
        _output_size, _input_size);
    DLMath::arr_sum<NumType>(_activations.data(), _activations.data(), 
        _biases.data(), _output_size);

    switch (_activation)
    {
        case Activation::ReLU:
        {
            DLMath::relu<NumType>(_activations.data(), _activations.data(), 
                SizeType(_output_size));
            break;
        }
        case Activation::Softmax:
        {
            DLMath::softmax<NumType>(_activations.data(), _activations.data(), 
                SizeType(_output_size));
            break;
        }
        case Activation::Linear:
        default:
        {
            // Linear activation disables non-linear function.
            break;
        }
    }

    // Forward to the next layers.
    for (const auto& layer: this->_subsequents)
    {
        layer->forward(_activations.data());
    }
}

void DenseLayer::reverse(const NumType *gradients)
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
            DLMath::relu_1<NumType>(
                _activation_gradients.data(), 
                _activations.data(), 
                _output_size);
            break;
        }
        case Activation::Softmax:
        {
            /*
             * The softmax derivation explits the calculus of softmax performed 
             * previously and saved in _activations vector.
             */
            DLMath::softmax_1_opt<NumType>(
                _activation_gradients.data(),
                _activations.data(), 
                _output_size);
            break;
        }
        case Activation::Linear:
        default:
        {
            std::fill(_activation_gradients.begin(), 
                _activation_gradients.end(), NumType{1.0});
            break;
        }
    }

    // Calculate dJ/dz = dJ/dg(z) * dg(z)/dz.
    DLMath::arr_mul(_activation_gradients.data(), _activation_gradients.data(),
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
    DLMath::arr_sum(_bias_gradients.data(), _bias_gradients.data(), 
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
    for (SizeType i = 0; i < _output_size; ++i)
    {
        for (SizeType j = 0; j < _input_size; ++j)
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
    for (SizeType i = 0; i < _output_size; ++i)
    {
        for (SizeType j = 0; j < _input_size; ++j)
        {
            _input_gradients[j] += 
                _activation_gradients[i] * _weights[(i * _input_size) + j];
        }
    }

    for (auto l: _antecedents)
    {
        l->reverse(_input_gradients.data());
    }
}

const NumType* DenseLayer::last_input()
{
    return _last_input;
}

const NumType* DenseLayer::last_output()
{
    return _activations.data();
}

NumType* DenseLayer::param(SizeType index)
{
    if (index < _weights.size())
    {
        return &_weights[index];
    }
    return &_biases[index - _weights.size()];
}

NumType* DenseLayer::gradient(SizeType index)
{
    if (index < _weight_gradients.size())
    {
        return &_weight_gradients[index];
    }
    return &_bias_gradients[index - _weight_gradients.size()];
}

void DenseLayer::print() const 
{
    std::cout << _name << std::endl;
    std::cout << "Weights (" << _output_size << " x " << _input_size << ")"
        << std::endl;
    for (SizeType i = 0; i < _output_size; ++i)
    {
        SizeType offset = i * _input_size;
        for (SizeType j = 0; j < _input_size; ++j)
        {
            std::cout << "\t[" << (offset + j) << "]" << _weights[offset + j]
                << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Biases (" << _output_size << " x 1)" << std::endl;
    for (SizeType i = 0; i < _output_size; ++i)
    {
        std::cout << "\t" << _biases[i] << std::endl;
    }
    std::cout << std::endl;
}

void DenseLayer::input_size(SizeType input_size)
{
    Layer::input_size(input_size);
    _weights.resize(_output_size * input_size);
    _weight_gradients.resize(_output_size * input_size);
    _input_gradients.resize(input_size);
}

} // namespace EdgeLearning
