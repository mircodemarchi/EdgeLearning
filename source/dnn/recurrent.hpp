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

class RecurrentLayer : public Layer 
{
public:
    static const std::string TYPE;

    enum class HiddenActivation : int
    {
        ReLU,
        TanH,
        Linear
    };

    // TODO: test softmax output activation.
    // TODO: test relu and linear hidden activation.

    RecurrentLayer(std::string name = std::string(),
        SizeType input_size = 0, SizeType output_size = 0,
        SizeType hidden_size = 0, SizeType time_steps = 0,
        HiddenActivation hidden_activation = HiddenActivation::TanH);

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType(std::random_device{}()))
        override;

    /**
     * \brief The input data should have size input_size().
     * \param inputs
     */
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;

    /**
     * \brief The gradient data should have size output_size().
     * \param gradients
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    const std::vector<NumType>& last_output() override;

    /**
     * \brief Weight input to hidden, hidden to hidden and hidden to output.
     * Bias to hidden and to output. 
     * \return SizeType
     */
    [[nodiscard]] SizeType param_count() const noexcept override
    {
        return (input_size() + _hidden_size + 1UL) * _hidden_size
             + (_hidden_size + 1UL) * output_size();
    }

    NumType& param(SizeType index) override;
    NumType& gradient(SizeType index) override;

    [[nodiscard]] SharedPtr clone() const override
    {
        return std::make_shared<RecurrentLayer>(*this);
    }

    void print() const override;

    void hidden_state(std::vector<NumType> hidden_state)
    {
        if (hidden_state.size() > _hidden_size)
        {
            throw std::runtime_error("hidden state exceeds the hidden size");
        }
        std::copy(hidden_state.begin(), hidden_state.end(),
                  _hidden_state.begin());
    }

    void time_steps(SizeType time_steps)
    {
        _time_steps = time_steps;
        _hidden_state.resize(
                _hidden_size * std::max(_time_steps, SizeType(1U)));
        _output_activations.resize(output_size() * _time_steps);
        _input_gradients.resize(input_size() * _time_steps);
    }

    void reset_hidden_state()
    {
        for (NumType& s: _hidden_state)
        {
            s = 0.0;
        }
    }

    [[nodiscard]] const DLMath::Shape3d & input_shape() const override
    {
        return Layer::input_shape();
    }
    void input_shape(DLMath::Shape3d input_shape) override;

    /**
     * \brief Save the layer infos to disk.
     * \param out Json& out Json to write.
     */
    void dump(Json& out) const override;

    /**
     * \brief Load the layer infos from disk.
     * \param in const Json& Json to read.
     */
    void load(Json& in) override;

private:
    HiddenActivation _hidden_activation;
    SizeType _hidden_size;

    std::vector<NumType> _hidden_state;
    SizeType _time_steps;

    // == Layer parameters ==
    /**
     * \brief Weights input to hidden of the layer. 
     * Size: _hidden_size * _input_size.
     */
    SharedParams _weights_i_to_h;
    /**
     * \brief Weights hidden to hidden of the layer. 
     * Size: _hidden_size * _hidden_size.
     */
    SharedParams _weights_h_to_h;
    /**
     * \brief Weights hidden to output of the layer. 
     * Size: _output_size * _hidden_size.
     */
    SharedParams _weights_h_to_o;

    /// \brief Biases to hidden of the layer. Size: _hidden_size. 
    SharedParams _biases_to_h;
    /// \brief Biases to output of the layer. Size: output_size(). 
    SharedParams _biases_to_o;

    /// \brief Activations of the layer. Size: output_size().
    std::vector<NumType> _output_activations;

    /**
     * \brief Weights gradients input to hidden of the layer. 
     * Size: _hidden_size * input_size().
     */
    Params _weights_i_to_h_gradients;
    /**
     * \brief Weights gradients hidden to hidden of the layer. 
     * Size: _hidden_size * _hidden_size.
     */
    Params _weights_h_to_h_gradients;
    /**
     * \brief Weights gradients hidden to output of the layer. 
     * Size: output_size() * _hidden_size.
     */
    Params _weights_h_to_o_gradients;

    /// \brief Biases gradients to hidden of the layer. Size: _hidden_size. 
    Params _biases_to_h_gradients;
    /// \brief Biases gradients to output of the layer. Size: output_size(). 
    Params _biases_to_o_gradients;

    /**
     * \brief Input gradients of the layer. Size: input_size().
     * This buffer is used to store temporary gradients used in a **single**
     * backpropagation pass. Note that this does not accumulate like the weight
     * and bias gradients do.
     */
    std::vector<NumType> _input_gradients;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_RECURRENT_HPP
