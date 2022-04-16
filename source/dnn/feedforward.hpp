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
     * \param activation  The output activation function.
     * \param prefix_name The prefix name of the default generated name.
     */
    FeedforwardLayer(Model& model,
                     SizeType input_size = 0, SizeType output_size = 0,
                     Activation activation = Activation::Linear,
                     std::string name = std::string(),
                     std::string prefix_name = std::string());

    /**
     * \brief Virtual method used to perform forward propagations. During 
     * forward propagation nodes transform input data and feed results to all 
     * subsequent nodes.
     * \param inputs NumType ptr
     */
    virtual void forward(const NumType *inputs) override;

    /**
     * \brief Virtual method used to perform reverse propagations. During 
     * reverse propagation nodes receive loss gradients to its previous outputs
     * and compute gradients with respect to each tunable parameter.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz.
     * \param gradients NumType ptr dJ/dg(z)
     */
    virtual void reverse(const NumType *gradients) override;

    const NumType* last_output() override { return _activations.data(); }

    void next(const NumType *activations = nullptr) override;
    void previous(const NumType *gradients = nullptr) override;

    /**
     * \brief Getter of input_size class field.
     * \return The size of the layer input.
     */
    virtual void input_size(DLMath::Shape3d input_size) override;

protected:
    // == Layer parameters ==
    /// \brief Activations of the layer. Size: _output_size.
    std::vector<NumType> _activations;

    // == Loss Gradients ==
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

#endif // EDGE_LEARNING_DNN_FEEDFORWARD_HPP
