/***************************************************************************
 *            dnn/feedforward.cpp
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

#include "feedforward.hpp"


namespace EdgeLearning {

FeedforwardLayer::FeedforwardLayer(
    Model& model, SizeType input_size, SizeType output_size,
    Activation activation, std::string name, std::string prefix_name)
    : Layer(model, input_size, output_size, activation, std::move(name),
            prefix_name.empty() ? "feedforward_layer_" : prefix_name)
    , _activations{}
    , _activation_gradients{}
    , _input_gradients{}
{
    // The outputs of each neuron within the layer is an "activation".
    _activations.resize(output_size);
    _activation_gradients.resize(output_size);

    _input_gradients.resize(input_size);
}


void FeedforwardLayer::forward(const NumType *inputs)
{
    switch (_activation)
    {
        case Activation::ReLU:
        {
            DLMath::relu<NumType>(_activations.data(), inputs, _output_size);
            break;
        }
        case Activation::Softmax:
        {
            DLMath::softmax<NumType>(_activations.data(), inputs, _output_size);
            break;
        }
        case Activation::TanH:
        {
            DLMath::tanh<NumType>(_activations.data(), inputs, _output_size);
            break;
        }
        case Activation::Linear:
        default:
        {
            // Linear activation disables non-linear function.
            break;
        }
    }
}

void FeedforwardLayer::reverse(const NumType *gradients)
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
        case Activation::TanH:
        {
            DLMath::tanh_1<NumType>(
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
}

void FeedforwardLayer::next(const NumType *activations)
{
    (void) activations;
    Layer::next(_activations.data());
}

void FeedforwardLayer::previous(const NumType *gradients)
{
    (void) gradients;
    Layer::next(_input_gradients.data());
}

void FeedforwardLayer::input_size(DLMath::Shape3d input_size) {
    Layer::input_size(input_size);
    _input_gradients.resize(input_size.size());
}


} // namespace EdgeLearning
