/***************************************************************************
 *            dnn/recurrent.cpp
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

#include "recurrent.hpp"

#include "dlmath.hpp"

#include <algorithm>
#include <cstdio>

namespace EdgeLearning {

RecurrentLayer::RecurrentLayer(Model& model, std::string name, 
    SizeType input_size, SizeType output_size, SizeType hidden_size,
    SizeType time_steps, 
    OutputActivation output_activation, HiddenActivation hidden_activation)
    : Layer(model, input_size, output_size,
            std::move(name), "recurrent_layer_")
    , _output_activation{output_activation}
    , _hidden_activation{hidden_activation}
    , _hidden_size{hidden_size}
    , _time_steps{time_steps}
{
    // std::cout << _name << ": " << _input_size
    //    << " -{" << _hidden_size << "}-> " << _output_size << std::endl;

    auto ih_size = _input_size * _hidden_size;
    auto hh_size = _hidden_size * _hidden_size;
    auto ho_size = _hidden_size * _output_size;

    // The weight input to hidden parameters are an HxI matrix.
    _weights_i_to_h.resize(ih_size);
    // The weight input to hidden parameters are an HxH matrix.
    _weights_h_to_h.resize(hh_size);
    // The weight input to hidden parameters are an OxH matrix.
    _weights_h_to_o.resize(ho_size);

    // The bias to hidden parameters are a Hx1 vector. 
    _biases_to_h.resize(_hidden_size);
    // The bias to output parameters are a Ox1 vector. 
    _biases_to_o.resize(_output_size);

    // The outputs of each neuron within the layer is an "activation".
    _activations.resize(_output_size * _time_steps);

    // The hidden state is of hidden_size for each time step of the sequences.
    _hidden_state = std::vector<NumType>(
        _hidden_size * std::max(_time_steps, SizeType(1U)), 0.0);

    _activation_gradients.resize(_output_size * _time_steps);
    _weights_i_to_h_gradients.resize(ih_size);
    _weights_h_to_h_gradients.resize(hh_size);
    _weights_h_to_o_gradients.resize(ho_size);
    _biases_to_h_gradients.resize(_hidden_size);
    _biases_to_o_gradients.resize(_output_size);
    _input_gradients.resize(_input_size * _time_steps);
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

void RecurrentLayer::forward(const NumType *inputs)
{
    // Remember the last input data for backpropagation.
    _last_input = inputs;

    const NumType* curr_sequence; //< Ptr to the current sequence to forward.
    SizeType curr_hs_idx;         //< Current hidden state index.
    SizeType next_hs_idx;         //< Next hidden state index.

    auto *tmp_mul = new NumType[_hidden_size];

    // Loop the time sequences.
    for (SizeType t = 0; t < _time_steps; ++t)
    {
        curr_sequence = inputs + t * _input_size;
        curr_hs_idx = t;
        next_hs_idx = (t == (_time_steps - 1)) ? 0 : t + 1;

        /*
         * Compute the product of the input with its input_to_hidden weights.
         * h(t+1) = W_ih * x 
         */
        DLMath::matarr_mul<NumType>(
            _hidden_state.data() + next_hs_idx * _hidden_size, 
            _weights_i_to_h.data(),
            curr_sequence, _hidden_size, _input_size);

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

        // Calculate hidden activations.
        switch (_hidden_activation)
        {
            // TODO: to test.
            // case HiddenActivation::ReLU:
            // {
            //     /*
            //      * Compute the relu of the new hidden state.
            //      * h(t+1) = relu(h(t+1))
            //      */ 
            //     DLMath::relu<NumType>(
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
                DLMath::tanh<NumType>(
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
        DLMath::matarr_mul<NumType>(_activations.data() + t * _output_size, 
            _weights_h_to_o.data(),
            _hidden_state.data() + next_hs_idx * _hidden_size, 
            _output_size, _hidden_size);
        DLMath::arr_sum<NumType>(
            _activations.data() + t * _output_size,
            _activations.data() + t * _output_size,
            _biases_to_o.data(), _output_size);

        // Calculate output activations.
        switch (_output_activation)
        {
            // TODO: to test.
            // case OutputActivation::Softmax:
            // {
            //     DLMath::softmax<NumType>(
            //         _activations.data() + i * _output_size, 
            //         _activations.data() + i * _output_size,
            //         SizeType(_output_size));
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
    for (const auto& l: this->_subsequents)
    {
        l->forward(_activations.data());
    }
}

void RecurrentLayer::reverse(const NumType *gradients)
{
    const NumType* curr_sequence_gradients; //< Ptr to the current gradients.
    SizeType curr_hs_idx; //< Current hidden state index.
    SizeType prev_hs_idx; //< Previous hidden state index.

    std::vector<NumType> next_hidden_state(
        static_cast<std::size_t>(_hidden_size), NumType{0.0});
    auto *tmp_mul = new NumType[_hidden_size];

    // Loop the gradient sequences in reverse.
    for (SizeType t = _time_steps; t > 0; --t)
    {
        SizeType t_idx = t - 1;
        curr_hs_idx = (t >= _time_steps) ? 0 : t;
        prev_hs_idx = t_idx;
        curr_sequence_gradients = gradients + (t_idx * _output_size);

        /* 
         * Calculate gradient of output activation and put in 
         * _activation_gradients.
         */
        NumType* curr_activation_gradients = _activation_gradients.data() 
            + (t_idx * _output_size);
        switch (_output_activation)
        {
            // TODO: to test.
            // case OutputActivation::Softmax:
            // {
            //     DLMath::softmax_1_opt<NumType>(
            //         curr_activation_gradients,
            //         _activations.data() + t_idx * _output_size, 
            //         _output_size);
            //     break;
            // }
            case OutputActivation::Linear:
            default:
            {
                std::fill(curr_activation_gradients, 
                    curr_activation_gradients + _output_size, NumType{1.0});
                break;
            }
        }

        // Calculate dJ/dz = dJ/dg(z) * dg(z)/dz.
        DLMath::arr_mul(
            curr_activation_gradients, curr_activation_gradients,
            curr_sequence_gradients, _output_size);

        // Bias gradient to output.
        DLMath::arr_sum(
            _biases_to_o_gradients.data(), 
            _biases_to_o_gradients.data(), 
            curr_activation_gradients, _output_size);

        // Weight gradient hidden to output.
        for (SizeType i = 0; i < _output_size; ++i)
        {
            for (SizeType j = 0; j < _hidden_size; ++j)
            {
                _weights_h_to_o_gradients[(i * _hidden_size) + j] += 
                    curr_activation_gradients[i] 
                    * _hidden_state[(SizeType(curr_hs_idx) * _hidden_size) + j];
            }
        }

        // Hidden state gradient for next time step gradient propagation.
        for (SizeType j = 0; j < _hidden_size; ++j)
        {
            tmp_mul[j] = NumType(0.0);
            for (SizeType i = 0; i < _output_size; ++i)
            {
                tmp_mul[j] += _weights_h_to_o[(i * _hidden_size) + j] 
                    * curr_activation_gradients[i];
            }
        }
        DLMath::arr_sum(
            tmp_mul, 
            tmp_mul, 
            next_hidden_state.data(), _hidden_size);

        // Calculate gradient of hidden activation and put in next_hidden_state.
        switch (_hidden_activation)
        {
            // TODO: to test.
            // case HiddenActivation::ReLU:
            // {
            //     DLMath::relu_1<NumType>(next_hidden_state.data(), 
            //         _hidden_state.data() + curr_hs_idx * _hidden_size, 
            //         _hidden_size);
            //     break;
            // }
            // case HiddenActivation::Linear:
            // {
            //     std::fill(next_hidden_state.begin(), next_hidden_state.end(),
            //         NumType{1.0});
            //     break;
            // }
            case HiddenActivation::TanH:
            default:
            {
                DLMath::tanh_1<NumType>(
                    next_hidden_state.data(), 
                    _hidden_state.data() + curr_hs_idx * _hidden_size, 
                    _hidden_size);
                break;
            }
        }
        DLMath::arr_mul<NumType>(next_hidden_state.data(), 
            next_hidden_state.data(), tmp_mul, _hidden_size);

        // Bias gradient to hidden.
        DLMath::arr_sum<NumType>(
            _biases_to_h_gradients.data(), 
            _biases_to_h_gradients.data(), 
            next_hidden_state.data(), _hidden_size);

        // Weight gradient input to hidden.
        for (SizeType i = 0; i < _hidden_size; ++i)
        {
            for (SizeType j = 0; j < _input_size; ++j)
            {
                _weights_i_to_h_gradients[(i * _input_size) + j] += 
                    next_hidden_state[i] 
                    * _last_input[(t_idx * _input_size) + j];
            }
        }

        // Weight gradient hidden to hidden.
        for (SizeType i = 0; i < _hidden_size; ++i)
        {
            for (SizeType j = 0; j < _hidden_size; ++j)
            {
                _weights_h_to_h_gradients[(i * _hidden_size) + j] += 
                    next_hidden_state[i] 
                    * _hidden_state[(SizeType(prev_hs_idx) * _hidden_size) + j];
            }
        }

        // Input gradient.
        NumType* curr_input_gradients = _input_gradients.data() 
            + (t_idx * _input_size);
        std::fill(curr_input_gradients, curr_input_gradients + _input_size, 0);
        for (SizeType i = 0; i < _hidden_size; ++i)
        {
            for (SizeType j = 0; j < _input_size; ++j)
            {
                curr_input_gradients[j] += 
                    next_hidden_state[i] 
                        * _weights_i_to_h[(i * _input_size) + j];
            }
        }

        // Update next hidden state for next time step.
        for (SizeType j = 0; j < _hidden_size; ++j)
        {
            tmp_mul[j] = NumType(0.0);
            for (SizeType i = 0; i < _hidden_size; ++i)
            {
                tmp_mul[j] += _weights_h_to_h[(i * _hidden_size) + j] 
                    * next_hidden_state[i];
            }
        }
        next_hidden_state = std::vector<NumType>(tmp_mul, 
            tmp_mul + _hidden_size);
    }

    delete[] tmp_mul;

    for (const auto& l: _antecedents)
    {
        l->reverse(_input_gradients.data());
    }
}

const NumType* RecurrentLayer::last_input()
{
    return _last_input;
}

const NumType* RecurrentLayer::last_output()
{
    return _activations.data();
}

NumType* RecurrentLayer::param(SizeType index)
{
    SizeType acc_size = 0;
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

NumType* RecurrentLayer::gradient(SizeType index)
{
    SizeType acc_size = 0;
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
    std::cout << _name << std::endl;
    std::cout << "Weights input to hidden (" 
        << _hidden_size << " x " << _input_size << ")" << std::endl;
    for (SizeType i = 0; i < _hidden_size; ++i)
    {
        SizeType offset = i * _input_size;
        for (SizeType j = 0; j < _input_size; ++j)
        {
            std::cout << "\t[" << (offset + j) << "]" 
                << _weights_i_to_h[offset + j];
        }
        std::cout << std::endl;
    }
    std::cout << "Weights hidden to hidden (" 
        << _hidden_size << " x " << _input_size << ")" << std::endl;
    for (SizeType i = 0; i < _hidden_size; ++i)
    {
        SizeType offset = i * _hidden_size;
        for (SizeType j = 0; j < _hidden_size; ++j)
        {
            std::cout << "\t[" << (offset + j) << "]" 
                << _weights_h_to_h[offset + j];
        }
        std::cout << std::endl;
    }
    std::cout << "Weights hidden to output (" 
        << _output_size << " x " << _input_size << ")" << std::endl;
    for (SizeType i = 0; i < _output_size; ++i)
    {
        SizeType offset = i * _hidden_size;
        for (SizeType j = 0; j < _hidden_size; ++j)
        {
            std::cout << "\t[" << (offset + j) << "]" 
                << _weights_h_to_o[offset + j];
        }
        std::cout << std::endl;
    }
    std::cout << "Biases to hidden (" << _hidden_size << " x 1)" << std::endl;
    for (SizeType i = 0; i < _hidden_size; ++i)
    {
        std::cout << "\t" << _biases_to_h[i] << std::endl;
    }
    std::cout << "Biases to output (" << _output_size << " x 1)" << std::endl;
    for (SizeType i = 0; i < _hidden_size; ++i)
    {
        std::cout << "\t" << _biases_to_h[i] << std::endl;
    }
    std::cout << std::endl;
}

void RecurrentLayer::input_size(SizeType input_size) {
    Layer::input_size(input_size);
    auto ih_size = input_size * _hidden_size;
    _weights_i_to_h.resize(ih_size);
    _weights_i_to_h_gradients.resize(ih_size);
    _input_gradients.resize(input_size * _time_steps);
}

} // namespace EdgeLearning
