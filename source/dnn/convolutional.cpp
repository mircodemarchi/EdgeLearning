/***************************************************************************
 *            dnn/convolutional.cpp
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

#include "convolutional.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace EdgeLearning {

static inline SizeType convolutional_output_side(
    SizeType input_side, SizeType kernel_side,
    SizeType padding, SizeType stride
    )
{
    return ((input_side - kernel_side + (2 * padding)) / stride) + 1ULL;
}

static inline DLMath::Shape3d convolutional_output_shape(
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    DLMath::Shape2d stride, DLMath::Shape2d padding, SizeType n_filters
)
{
    return {
        convolutional_output_side(input_shape.width, kernel_shape.width,
                                  padding.width, stride.width),
        convolutional_output_side(input_shape.height, kernel_shape.height,
                                  padding.height, stride.height),
        n_filters};
}

static inline SizeType convolutional_output_size(
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    DLMath::Shape2d stride, DLMath::Shape2d padding, SizeType n_filters
)
{
    return convolutional_output_shape(input_shape, kernel_shape,
                                      stride, padding, n_filters).size();
}

ConvolutionalLayer::ConvolutionalLayer(
    Model& model, std::string name, Activation activation,
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    SizeType n_filters, DLMath::Shape2d stride, DLMath::Shape2d padding)
    : FeedforwardLayer(
        model, input_shape.size(),
        convolutional_output_size(input_shape, kernel_shape,
                                  stride, padding, n_filters),
        activation, std::move(name), "convolutional_layer_")
    , _input_shape(input_shape)
    , _output_shape(
        convolutional_output_shape(input_shape, kernel_shape,
                                   stride, padding, n_filters))
    , _kernel_shape(kernel_shape)
    , _n_filters(n_filters)
    , _stride(stride)
    , _padding(padding)
{
    // The weight parameters are composed by n_filters of kernel size.
    _weights.resize(_kernel_shape.size() * input_shape.channels * n_filters);

    // The bias is incremented to the result of each filter.
    _biases.resize(n_filters);

    _weight_gradients.resize(_kernel_shape.size() * input_shape.channels
                             * n_filters);
    _bias_gradients.resize(n_filters);
}

void ConvolutionalLayer::init(RneType& rne)
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

void ConvolutionalLayer::forward(const NumType *inputs)
{
    // Remember the last input data for backpropagation.
    _last_input = inputs;

    /*
     * Perform convolution with n_filters of kernel size contained in
     * _weights vector on the input 3D matrix.
     */
    DLMath::conv4d<NumType>(_activations.data(), inputs, _input_shape,
                            _weights.data(), _kernel_shape, _n_filters,
                            _stride, _padding);

    FeedforwardLayer::forward(_activations.data());

    FeedforwardLayer::next();
}

void ConvolutionalLayer::reverse(const NumType *gradients)
{
    FeedforwardLayer::reverse(gradients);

    // TODO:

    FeedforwardLayer::previous();
}

NumType* ConvolutionalLayer::param(SizeType index)
{
    if (index < _weights.size())
    {
        return &_weights[index];
    }
    return &_biases[index - _weights.size()];
}

NumType* ConvolutionalLayer::gradient(SizeType index)
{
    if (index < _weight_gradients.size())
    {
        return &_weight_gradients[index];
    }
    return &_bias_gradients[index - _weight_gradients.size()];
}

void ConvolutionalLayer::print() const
{
    std::cout << _name << std::endl;
    std::cout << "Weights ("
        << _kernel_shape.height << " x "
        << _kernel_shape.width  << " x "
        << _input_shape.channels  << " x "
        << _n_filters << ")" << std::endl;

    for (SizeType r = 0; r < _kernel_shape.height; ++r)
    {
        SizeType r_offset = r * _kernel_shape.width * _input_shape.channels
                            * _n_filters;
        for (SizeType c = 0; c < _kernel_shape.width; ++c)
        {
            SizeType c_offset = c * _input_shape.channels * _n_filters;
            for (SizeType ch = 0; ch < _input_shape.channels; ++ch)
            {
                SizeType ch_offset = ch * _n_filters;
                std::cout << "\t[" << r << "," << c << "," << ch << ",0:"
                          << _n_filters << "]" << std::endl;
                for (SizeType f = 0; f < _n_filters - 1; ++f)
                {
                    std::cout << _weights[r_offset + c_offset + ch_offset + f]
                        << ", ";
                }
                std::cout << _weights[r_offset + c_offset + ch_offset
                                      + _n_filters - 1] << std::endl;
            }
        }
    }
    std::cout << "Biases (1 x 1 x " << _n_filters << ")" << std::endl;
    for (SizeType i = 0; i < _n_filters; ++i)
    {
        std::cout << "\t" << _biases[i] << std::endl;
    }
    std::cout << std::endl;
}

void ConvolutionalLayer::input_size(DLMath::Shape3d input_shape)
{
    FeedforwardLayer::input_size(input_shape);
    _weights.resize(_kernel_shape.size() * input_shape.channels * _n_filters);
    _weight_gradients.resize(_kernel_shape.size() * input_shape.channels
                             * _n_filters);

    // Update input and output shape accordingly (see this constructor).
    _input_shape = input_shape;
    _output_shape = convolutional_output_shape(input_shape, _kernel_shape,
                                               _stride, _padding, _n_filters);

    // Update output size accordingly (see Layer and FeedforwardLayer constr.).
    _output_size = _output_shape.size();
    _activations.resize(_output_size);
    _activation_gradients.resize(_output_size);
}

} // namespace EdgeLearning
