/***************************************************************************
 *            dnn/recurrent.hpp
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

/*! \file  dnn/recurrent.hpp
 *  \brief Recurrent layer.
 */

#ifndef EDGE_LEARNING_DNN_RECURRENT_HPP
#define EDGE_LEARNING_DNN_RECURRENT_HPP

#include "layer.hpp"

#include <string>
#include <vector>
#include <stdexcept>


namespace EdgeLearning {

// TODO: test softmax output activation.
enum class OutputActivation
{
    Linear,
    // Softmax, 
};

// TODO: test relu and linear hidden activation.
enum class HiddenActivation
{
    TanH,
    // ReLU,
    // Linear,
};

class RecurrentLayer : public Layer 
{
public: 
    RecurrentLayer(Model& model, std::string name = std::string(),
        SizeType output_size = 0, SizeType input_size = 0, SizeType hidden_size = 0,
        SizeType time_steps = 0,
        OutputActivation output_activation = OutputActivation::Linear, 
        HiddenActivation hidden_activation = HiddenActivation::TanH);

    void init(RneType& rne) override;

    /**
     * \brief The input data should have size _input_size.
     * \param inputs
     */
    void forward(const NumType *inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * \param gradients
     */
    void reverse(const NumType *gradients) override;

    const NumType* last_input() override;
    const NumType* last_output() override;

    /**
     * \brief Weight input to hidden, hidden to hidden and hidden to output.
     * Bias to hidden and to output. 
     * \return SizeType
     */
    [[nodiscard]] SizeType param_count() const noexcept override
    {
        return (_input_size + _hidden_size + 1UL) * _hidden_size
             + _hidden_size + 1UL * _output_size;
    }

    NumType* param(SizeType index) override;
    NumType* gradient(SizeType index) override;

    void print() const override;

    void set_initial_hidden_state(std::vector<NumType> initial_hidden_state) 
    {
        if (initial_hidden_state.size() > _hidden_size)
        {
            throw std::runtime_error("initial hidden state exceeds the hidden size");
        }
        std::copy(initial_hidden_state.begin(), initial_hidden_state.end(), 
            _hidden_state.begin());
    }

    void set_time_steps(SizeType time_steps)
    {
        _time_steps = time_steps;
        _hidden_state.resize(
                _hidden_size * std::max(_time_steps, SizeType(1U)));
        _activations.resize(_output_size * _time_steps);
        _activation_gradients.resize(_output_size * _time_steps);
        _input_gradients.resize(_input_size * _time_steps);
    }

    void reset_hidden_state()
    {
        for (NumType& s: _hidden_state)
        {
            s = 0.0;
        }
    }

    void input_size(SizeType input_size) override;

private:
    OutputActivation _output_activation;
    HiddenActivation _hidden_activation;
    SizeType _hidden_size;

    std::vector<NumType> _hidden_state;
    SizeType _time_steps;

    // == Layer parameters ==
    /**
     * \brief Weights input to hidden of the layer. 
     * Size: _hidden_size * _input_size.
     */
    std::vector<NumType> _weights_i_to_h;
    /**
     * \brief Weights hidden to hidden of the layer. 
     * Size: _hidden_size * _hidden_size.
     */
    std::vector<NumType> _weights_h_to_h;
    /**
     * \brief Weights hidden to output of the layer. 
     * Size: _output_size * _hidden_size.
     */
    std::vector<NumType> _weights_h_to_o;

    /// \brief Biases to hidden of the layer. Size: _hidden_size. 
    std::vector<NumType> _biases_to_h;
    /// \brief Biases to output of the layer. Size: _output_size. 
    std::vector<NumType> _biases_to_o;

    /// \brief Activations of the layer. Size: _output_size. 
    std::vector<NumType> _activations;

    /**
     * \brief Weights gradients input to hidden of the layer. 
     * Size: _hidden_size * _input_size.
     */
    std::vector<NumType> _weights_i_to_h_gradients;
    /**
     * \brief Weights gradients hidden to hidden of the layer. 
     * Size: _hidden_size * _hidden_size.
     */
    std::vector<NumType> _weights_h_to_h_gradients;
    /**
     * \brief Weights gradients hidden to output of the layer. 
     * Size: _output_size * _hidden_size.
     */
    std::vector<NumType> _weights_h_to_o_gradients;

    /// \brief Biases gradients to hidden of the layer. Size: _hidden_size. 
    std::vector<NumType> _biases_to_h_gradients;
    /// \brief Biases gradients to output of the layer. Size: _output_size. 
    std::vector<NumType> _biases_to_o_gradients;

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

#endif // EDGE_LEARNING_DNN_RECURRENT_HPP
