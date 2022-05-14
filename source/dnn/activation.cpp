/***************************************************************************
 *            dnn/activation.cpp
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

#include "activation.hpp"

#include "dlmath.hpp"

#include <algorithm>
#include <cstdio>

namespace EdgeLearning {

ActivationLayer::ActivationLayer(Model& model, SizeType size,
                                 std::string name, std::string prefix_name)
    : FeedforwardLayer(model, size, size, std::move(name),
                       prefix_name.empty() ? "activation_layer_" : prefix_name)
{ }

void ActivationLayer::print() const
{
    std::cout << _name << std::endl;
    std::cout << "No learnable parameters" << std::endl;
    std::cout << std::endl;
}

void ActivationLayer::input_size(DLMath::Shape3d input_size)
{
    FeedforwardLayer::input_size(input_size);

    // Update output size accordingly (see Layer and FeedforwardLayer constr.).
    _output_size = input_size.size();
    _output_activations.resize(_output_size);
}
// =============================================================================

// ================================= ReLU ======================================
ReluLayer::ReluLayer(Model& model, std::string name, SizeType size)
    : ActivationLayer(model, size, std::move(name), "relu_layer_")
{ }

const std::vector<NumType>& ReluLayer::forward(
    const std::vector<NumType>& inputs)
{
    _last_input = inputs.data();
    DLMath::relu<NumType>(_output_activations.data(), inputs.data(),
                          _output_size);
    return ActivationLayer::forward(_output_activations);
}

const std::vector<NumType>& ReluLayer::backward(
    const std::vector<NumType>& gradients)
{
    /*
     * Calculate dg(z)/dz and put in _activation_gradients.
     * The input for ReLU derivation is the _activations vector, that
     * is filled with the ReLU of vector z, and not directly the vector
     * z. Considering that the input of ReLU derivation is used only
     * to check if it is > 0, and that if z > 0 then ReLU(z) > 0 and
     * viceversa, using ReLU of vector z or using directly the vector z
     * there is no differences.
     */
    DLMath::relu_1<NumType>(_input_gradients.data(), _output_activations.data(),
                            _output_size);
    // Calculate dJ/dz = dJ/dg(z) * dg(z)/dz.
    DLMath::arr_mul(_input_gradients.data(), _input_gradients.data(),
                    gradients.data(), _output_size);
    return ActivationLayer::backward(_input_gradients);
}
// =============================================================================

// ================================ Softmax ====================================
SoftmaxLayer::SoftmaxLayer(Model& model, std::string name, SizeType size)
    : ActivationLayer(model, size, std::move(name), "softmax_layer_")
{ }

const std::vector<NumType>& SoftmaxLayer::forward(
    const std::vector<NumType>& inputs)
{
    _last_input = inputs.data();
    DLMath::softmax<NumType>(_output_activations.data(), inputs.data(),
                             _output_size);
    return ActivationLayer::forward(_output_activations);
}

const std::vector<NumType>& SoftmaxLayer::backward(
    const std::vector<NumType>& gradients)
{
    /*
     * Calculate dJ/dz.
     * The softmax derivation exploits the calculus of softmax performed
     * previously and saved in _activations vector.
     */
    DLMath::softmax_1_opt_no_check<NumType>(
        _input_gradients.data(),
        _output_activations.data(), gradients.data(),
        _output_size);
    return ActivationLayer::backward(_input_gradients);
}
// =============================================================================

// ================================= TanH ======================================
TanhLayer::TanhLayer(Model& model, std::string name, SizeType size)
    : ActivationLayer(model, size, std::move(name), "tanh_layer_")
{ }

const std::vector<NumType>& TanhLayer::forward(
    const std::vector<NumType>& inputs)
{
    _last_input = inputs.data();
    DLMath::tanh<NumType>(_output_activations.data(), inputs.data(),
                          _output_size);
    return ActivationLayer::forward(_output_activations);
}

const std::vector<NumType>& TanhLayer::backward(
    const std::vector<NumType>& gradients)
{
    // Calculate dg(z)/dz and put in _activation_gradients.
    DLMath::tanh_1<NumType>(_input_gradients.data(), _output_activations.data(),
                            _output_size);
    // Calculate dJ/dz = dJ/dg(z) * dg(z)/dz.
    DLMath::arr_mul(_input_gradients.data(), _input_gradients.data(),
                    gradients.data(), _output_size);
    return ActivationLayer::backward(_input_gradients);
}
// =============================================================================

// ================================ Linear =====================================
LinearLayer::LinearLayer(Model& model, std::string name, SizeType size)
    : ActivationLayer(model, size, std::move(name), "linear_layer_")
{ }

const std::vector<NumType>& LinearLayer::forward(
    const std::vector<NumType>& inputs)
{
    _last_input = inputs.data();
    _output_activations = inputs;
    return ActivationLayer::forward(_output_activations);
}

const std::vector<NumType>& LinearLayer::backward(
    const std::vector<NumType>& gradients)
{
    // Linear activation: dg(z)/dz = 1.
    std::copy(gradients.begin(),
              gradients.begin() + static_cast<std::int64_t>(_output_size),
              _input_gradients.begin());
    return ActivationLayer::backward(_input_gradients);
}
// =============================================================================

} // namespace EdgeLearning
