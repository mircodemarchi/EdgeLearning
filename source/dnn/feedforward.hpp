/***************************************************************************
 *            dnn/feedforward.hpp
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

/*! \file  dnn/feedforward.hpp
 *  \brief High level Feedforward layer of a deep neural network.
 */

#ifndef EDGE_LEARNING_DNN_FEEDFORWARD_HPP
#define EDGE_LEARNING_DNN_FEEDFORWARD_HPP

#include "layer.hpp"


namespace EdgeLearning {

class FeedforwardLayer : public Layer
{
public:
    /**
     * \brief Construct a new Layer object.
     * \param model       The model in which the layer takes part.
     * \param input_size  The size of inputs of the layer.
     * \param output_size The size of outputs of the layer.
     * \param name        The name of the layer.
     * If empty, a default generated one is chosen.
     * \param prefix_name The prefix name of the default generated name.
     */
    FeedforwardLayer(Model& model,
                     SizeType input_size = 0, SizeType output_size = 0,
                     std::string name = std::string(),
                     std::string prefix_name = std::string());

    /**
     * \brief The last output of a feedforward layer will be the activation
     * vector.
     * \return The reference to the activation vector field.
     */
    const std::vector<NumType>& last_output() override
    {
        return _output_activations;
    }

    /**
     * \brief Getter of input_size class field.
     * \return The size of the layer input.
     */
    [[nodiscard]] virtual SizeType input_size() const override
    {
        return Layer::input_size();
    }

    /**
     * \brief Setter of input_size class field.
     * \param input_size DLMath::Shape3d Shape param used to take the size and
     * assign it to input_size.
     * The operation also performs a resize of the input_gradients.
     */
    virtual void input_size(DLMath::Shape3d input_size) override;

protected:
    /// \brief Activations of the layer. Size: _output_size.
    std::vector<NumType> _output_activations;

    /**
     * \brief Input gradients of the layer. Size: _input_size.
     * This buffer is used to store temporary gradients used in a **singe**
     * backpropagation pass. Note that this does not accumulate like the weight
     * and bias gradients do.
     */
    std::vector<NumType> _input_gradients;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_FEEDFORWARD_HPP
