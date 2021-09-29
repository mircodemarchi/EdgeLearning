/***************************************************************************
 *            recurrent.cpp
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

#include "recurrent.hpp"

#include "dlmath.hpp"

#include <algorithm>
#include <cstdio>

namespace Ariadne {

RecurrentLayer::RecurrentLayer(Model& model, std::string name, 
    uint16_t output_size, uint16_t input_size, uint16_t hidden_size, 
    uint16_t time_steps, 
    OutputActivation output_activation, HiddenActivation hidden_activation)
    : Layer(model, std::move(name))
    , _output_activation{output_activation}
    , _hidden_activation{hidden_activation}
    , _output_size{output_size}
    , _input_size{input_size}
    , _hidden_size{hidden_size}
    , _time_steps{time_steps}
{
    std::printf("%s: %d -{%d}-> %d\n", 
        _name.c_str(), _input_size, _hidden_size, _output_size);

    // The weight input to hidden parameters are an HxI matrix.
    _weights_i_to_h.resize(_hidden_size * _input_size);
    // The weight input to hidden parameters are an HxH matrix.
    _weights_h_to_h.resize(_hidden_size * _hidden_size);
    // The weight input to hidden parameters are an OxH matrix.
    _weights_h_to_o.resize(_output_size * _hidden_size);

    // The bias to hidden parameters are a Hx1 vector. 
    _biases_to_h.resize(_hidden_size);
    // The bias to output parameters are a Ox1 vector. 
    _biases_to_o.resize(_output_size);

    // The outputs of each neuron within the layer is an "activation".
    _activations.resize(_output_size * _time_steps);

    // The hidden state is of hidden_size for each time step of the sequences.
    _hidden_state = std::vector<NumType>(
        size_t(_hidden_size * std::max(_time_steps, uint16_t(1U))), 0.0);

    _activation_gradients.resize(_output_size);
    _weights_i_to_h_gradients.resize(_hidden_size * _input_size);
    _weights_h_to_h_gradients.resize(_hidden_size * _hidden_size);
    _weights_h_to_o_gradients.resize(_output_size * _hidden_size);
    _biases_to_h_gradients.resize(_hidden_size);
    _biases_to_o_gradients.resize(_output_size);
    _input_gradients.resize(_input_size);
}

void RecurrentLayer::init(RneType& rne)
{
    NumType sigma_i, sigma_h;
    switch (_output_activation)
    {
        // case OutputActivation::ReLU:
        // {   
        //     /*
        //      * Kaiming He, et. al. weight initialization for ReLU networks 
        //      * https://arxiv.org/pdf/1502.01852.pdf
        //      * Nrmal distribution with variance := sqrt( 2 / n_in )
        //      */
        //     sigma_i = std::sqrt(2.0 / static_cast<NumType>(_input_size));
        //     sigma_h = std::sqrt(2.0 / static_cast<NumType>(_hidden_size));
        //     break;
        // }
        // case OutputActivation::Softmax:
        case OutputActivation::Linear:
        default:
        {
            /* 
             * Xavier initialization
             * https://arxiv.org/pdf/1706.02515.pdf
             * Normal distribution with variance := sqrt( 1 / n_in )
             */
            sigma_i = std::sqrt(1.0 / static_cast<NumType>(_input_size));
            sigma_h = std::sqrt(1.0 / static_cast<NumType>(_hidden_size));
            break;
        }
    }

    /*
     * The C++ standard does not guarantee that the results obtained from a 
     * distribution function will be identical given the same inputs across 
     * different compilers and platforms, therefore I use my own 
     * distributions to provide deterministic results.
     */
    auto dist_i = DLMath::normal_pdf<NumType>(0.0, sigma_i);
    auto dist_h = DLMath::normal_pdf<NumType>(0.0, sigma_h);

    for (NumType& w: _weights_i_to_h)
    {
        w = dist_i(rne);
    }
    for (NumType& w: _weights_h_to_h)
    {
        w = dist_h(rne);
    }
    for (NumType& w: _weights_h_to_o)
    {
        w = dist_h(rne);
    }

    /*
     * Setting biases to zero is a common practice, as is initializing the 
     * bias to a small value (e.g. on the order of 0.01). The thinking is
     * that a non-zero bias will ensure that the neuron always "fires" at 
     * the beginning to produce a signal.
     */
    for (NumType& b: _biases_to_h)
    {
        b = 0.01; ///< You can try also with 0.0 or other strategies.
    }
    for (NumType& b: _biases_to_o)
    {
        b = 0.01; ///< You can try also with 0.0 or other strategies.
    }

    // Init first hidden state.
    for (NumType& hs: _hidden_state)
    {
        hs = 0.0;
    }
}

void RecurrentLayer::forward(NumType* inputs) 
{
    NumType* curr_sequence;     //< Ptr to the current sequence to forward.
    size_t curr_hs_idx;         //< Current hidden state index.
    size_t next_hs_idx;         //< Next hidden state index.

    NumType *tmp_mul = new NumType[_hidden_size];

    // Loop the time sequences.
    for (size_t i = 0; i < _time_steps; ++i)
    {
        curr_sequence = inputs + i * _input_size;
        curr_hs_idx = i;
        next_hs_idx = (i == (_time_steps - 1)) ? 0UL : i + 1;

        /*
         * Compute the product of the input with its input_to_hidden weights.
         * h(t+1) = W_ih * x 
         */
        DLMath::matarr_mul<NumType>(
            _hidden_state.data() + next_hs_idx * _hidden_size, 
            _weights_i_to_h.data(),
            inputs, _hidden_size, _input_size);

        /*
         * Compute the product of the hidden state with its 
         * hidden_to_hidden weights and the sum with the to_hidden bias.
         * h(t+1) += W_hh * h(t) + b_h
         */
        DLMath::matarr_mul<NumType>(tmp_mul, _weights_h_to_h.data(),
            _hidden_state.data() + curr_hs_idx * _hidden_size, 
            _hidden_size, _hidden_size);
        DLMath::arr_sum<NumType>(
            _hidden_state.data() + next_hs_idx * _hidden_size,
            _hidden_state.data() + next_hs_idx * _hidden_size,
            tmp_mul, _hidden_size);
        DLMath::arr_sum<NumType>(
            _hidden_state.data() + next_hs_idx * _hidden_size,
            _hidden_state.data() + next_hs_idx * _hidden_size,
            _biases_to_h.data(), _hidden_size);

        switch (_hidden_activation)
        {
            // TODO: to test.
            // case HiddenActivation::ReLU:
            // {
            //     /*
            //      * Compute the relu of the new hidden state.
            //      * h(t+1) = relu(h(t+1))
            //      */ 
            //     DLMath::relu(
            //         _hidden_state.data() + next_hs_idx * _hidden_size,
            //         _hidden_state.data() + next_hs_idx * _hidden_size, 
            //         _hidden_size);
            //     break;
            // }
            // case HiddenActivation::Linear:
            // {
            //     // Linear activation disables non-linear function.
            //     break;
            // }
            case HiddenActivation::TanH:
            default:
            {
                /*
                 * Compute the tanh of the new hidden state.
                 * h(t+1) = tanh(h(t+1))
                 */ 
                DLMath::tanh(
                    _hidden_state.data() + next_hs_idx * _hidden_size,
                    _hidden_state.data() + next_hs_idx * _hidden_size, 
                    _hidden_size);
                break;
            }
        }

        /*
         * Compute the product of the hidden state with the 
         * hidden_to_output weights and the sum with the to_output bias.
         * a(t) = W_ho * h(t + 1) + b_o
         */
        DLMath::matarr_mul<NumType>(_activations.data() + i * _output_size, 
            _weights_h_to_o.data(),
            _hidden_state.data() + next_hs_idx * _hidden_size, 
            _output_size, _hidden_size);
        DLMath::arr_sum<NumType>(
            _activations.data() + i * _output_size,
            _activations.data() + i * _output_size,
            _biases_to_o.data(), _output_size);

        switch (_output_activation)
        {
            // TODO: to test.
            // case OutputActivation::Softmax:
            // {
            //     DLMath::softmax<NumType>(
            //         _activations.data() + i * _output_size, 
            //         _activations.data() + i * _output_size,
            //         size_t(_output_size));
            //     break;
            // }
            case OutputActivation::Linear:
            default:
            {
                // Linear activation disables non-linear function.
                break;
            }
        }
    }

    delete[] tmp_mul;

    // Forward to the next layers.
    for (auto *layer: this->_subsequents)
    {
        layer->forward(_activations.data());
    }
}

void RecurrentLayer::reverse(NumType* gradients)
{
    
}

NumType* RecurrentLayer::param(size_t index)
{
    size_t acc_size = 0;
    if (index < _weights_i_to_h.size())
    {
        return &_weights_i_to_h[index];
    }
    acc_size += _weights_i_to_h.size();
    if (index < acc_size + _weights_h_to_h.size())
    {
        return &_weights_h_to_h[index - acc_size];
    }
    acc_size += _weights_h_to_h.size();
    if (index < acc_size + _biases_to_h.size())
    {
        return &_biases_to_h[index - acc_size];
    }
    acc_size += _biases_to_h.size();
    if (index < acc_size + _weights_h_to_o.size())
    {
        return &_weights_h_to_o[index - acc_size];
    }
    acc_size += _weights_h_to_o.size();
    return &_biases_to_o[index - acc_size];
}

NumType* RecurrentLayer::gradient(size_t index)
{
    size_t acc_size = 0;
    if (index < _weights_i_to_h_gradients.size())
    {
        return &_weights_i_to_h_gradients[index];
    }
    acc_size += _weights_i_to_h_gradients.size();
    if (index < acc_size + _weights_h_to_h_gradients.size())
    {
        return &_weights_h_to_h_gradients[index - acc_size];
    }
    acc_size += _weights_h_to_h_gradients.size();
    if (index < acc_size + _biases_to_h_gradients.size())
    {
        return &_biases_to_h_gradients[index - acc_size];
    }
    acc_size += _biases_to_h_gradients.size();
    if (index < acc_size + _weights_h_to_o_gradients.size())
    {
        return &_weights_h_to_o_gradients[index - acc_size];
    }
    acc_size += _weights_h_to_o_gradients.size();
    return &_biases_to_o_gradients[index - acc_size];
}

void RecurrentLayer::print() const 
{
    std::printf("%s\n", _name.c_str());

    std::printf("Weights input to hidden (%d x %d)\n", 
        _hidden_size, _input_size);
    for (size_t i = 0; i < _hidden_size; ++i)
    {
        size_t offset = i * _input_size;
        for (size_t j = 0; j < _input_size; ++j)
        {
            std::printf("\t[%zu]%f", offset + j, _weights_i_to_h[offset + j]);
        }
        std::printf("\n");
    }
    std::printf("Weights hidden to hidden (%d x %d)\n", 
        _hidden_size, _hidden_size);
    for (size_t i = 0; i < _hidden_size; ++i)
    {
        size_t offset = i * _hidden_size;
        for (size_t j = 0; j < _hidden_size; ++j)
        {
            std::printf("\t[%zu]%f", offset + j, _weights_h_to_h[offset + j]);
        }
        std::printf("\n");
    }
    std::printf("Weights hidden to output (%d x %d)\n", 
        _output_size, _hidden_size);
    for (size_t i = 0; i < _output_size; ++i)
    {
        size_t offset = i * _hidden_size;
        for (size_t j = 0; j < _hidden_size; ++j)
        {
            std::printf("\t[%zu]%f", offset + j, _weights_h_to_o[offset + j]);
        }
        std::printf("\n");
    }
    std::printf("Biases to hidden (%d x 1)\n", _hidden_size);
    for (size_t i = 0; i < _hidden_size; ++i)
    {
        std::printf("\t%f\n", _biases_to_h[i]);
    }
    std::printf("Biases to output (%d x 1)\n", _output_size);
    for (size_t i = 0; i < _hidden_size; ++i)
    {
        std::printf("\t%f\n", _biases_to_o[i]);
    }
    std::printf("\n");
}

} // namespace Ariadne
