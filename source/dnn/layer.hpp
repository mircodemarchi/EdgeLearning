/***************************************************************************
 *            dnn/layer.hpp
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

/*! \file  dnn/layer.hpp
 *  \brief High level layer of a deep neural network.
 */

#ifndef EDGE_LEARNING_DNN_LAYER_HPP
#define EDGE_LEARNING_DNN_LAYER_HPP

#include "type.hpp"
#include "dlmath.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>


namespace EdgeLearning {

class Model;

/**
 * \brief Base class of computational layers in a model.
 */
class Layer 
{
public:
    static const std::string Type;

    enum class Activation
    {
        ReLU,
        Softmax,
        TanH,
        // Sigmoid, // TODO: Implement sigmoid activation function.
        Linear,
        None
    };

    using ProbabilityDensityFunction = DLMath::ProbabilityDensityFunction;
    using SharedPtr = std::shared_ptr<Layer>;

    /**
     * \brief Construct a new Layer object.
     * \param model       The model in which the layer takes part.
     * \param input_size  The size of inputs of the layer.
     * \param output_size The size of outputs of the layer.
     * \param activation  The output activation function.
     * \param name        The name of the layer.
     * If empty, a default generated one is chosen.
     * \param prefix_name The prefix name of the default generated name.
     */
    Layer(Model& model, SizeType input_size = 0, SizeType output_size = 0,
          Activation activation = Activation::None,
          std::string name = std::string(),
          std::string prefix_name = std::string());

    /**
     * \brief Copy constructor of a new Layer object.
     * \param obj Layer object to copy.
     */
    Layer(const Layer& obj);

    /**
     * \brief Destroy the Layer object.
     */
    virtual ~Layer() {};

    /**
     * \brief Assignment operator of a Layer object.
     * \param obj       Layer object to copy.
     * \return Layer&   The assigned Layer object.
     */
    Layer& operator=(const Layer& obj);

    /**
     * \brief Virtual method used to describe how a layer should be 
     * initialized.
     * \param pdf ProbabilityDensityFunction The distribution used.
     * \param rne RneType                    The random generator.
     */
    virtual void init(
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType(std::random_device{}()))
        = 0;

    /**
     * \brief Virtual method used to perform forward propagations. During 
     * forward propagation nodes transform input data and feed results to all 
     * subsequent layers.
     * By default it passes activations vector to the subsequent layers.
     * \param inputs const std::vector<NumType>& Vector of inputs.
     * \return const std::vector<NumType>& The computed activations passed to
     * the subsequent layers.
     */
    virtual const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs);

    /**
     * \brief Virtual method used to perform forward propagations during model
     * training. By default it call the forward method.
     * \param inputs const std::vector<NumType>& Vector of inputs.
     * \return const std::vector<NumType>& The computed activations passed to
     * the subsequent layers.
     */
    virtual const std::vector<NumType>& training_forward(
        const std::vector<NumType>& inputs)
    {
        if (_input_size == 0)
        {
            input_size(inputs.size());
        }
        else if (_input_size != inputs.size())
        {
            throw std::runtime_error(
                "Training forward input catch an unpredicted input size: "
                + std::to_string(_input_size)
                + " != " + std::to_string(inputs.size()));
        }
        return forward(inputs);
    }

    /**
     * \brief Virtual method used to perform reverse propagations. During 
     * reverse propagation nodes receive loss gradients to its previous outputs
     * and compute gradients with respect to each tunable parameter.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz.
     * By default it passes the input_gradients vector to the antecedent layers.
     * \param gradients const std::vector<NumType>& Vector of gradients dJ/dg(z)
     * \return const std::vector<NumType>& The computed gradients passed to the
     * antecedent layers.
     */
    virtual const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients);

    /**
     * \brief Return the last input of the layer.
     * \return const NumType* The last input of the layer of input size.
     */
    std::vector<NumType> last_input()
    {
        return _last_input
            ? std::vector<NumType>{_last_input, _last_input + _input_size}
            : std::vector<NumType>{};
    };

    /**
     * \brief Return the last output of the layer.
     * \return const NumType* The last output of the layer of output size.
     */
    virtual const std::vector<NumType>& last_output() = 0;

    /**
     * \brief Virtual method that return the number of tunable parameters. 
     * This methos should be overridden to reflect the quantity of tunable 
     * parameters.
     * \return SizeType The amount of tunable parameters. 
     */
    [[nodiscard]] virtual SizeType param_count() const noexcept = 0;

    /**
     * \brief Virtual method accessor for parameter by index.
     * \param index SizeType Parameter index.
     * \return NumType* Pointer to parameter.
     */
    virtual NumType& param(SizeType index) = 0;

    /**
     * \brief Virtual method accessor for loss-gradient with respect to a 
     * parameter specified by index.
     * \param index SizeType Parameter index.
     * \return NumType* Pointer to gradient value of parameter.
     */
    virtual NumType& gradient(SizeType index) = 0;

    /**
     * \brief Print layer info.
     */
    virtual void print() const = 0;

    /**
     * \brief Virtual method information dump for debugging purposes.
     * \return std::string const& The layer name.
     */
    [[nodiscard]] std::string const& name() const noexcept 
    { 
        return _name; 
    }

    /**
     * \brief Getter of input_size class field.
     * \return The size of the layer input.
     */
    [[nodiscard]] virtual SizeType input_size() const;
    virtual void input_size(DLMath::Shape3d input_size);

    /**
     * \brief Getter of input_size class field.
     * \return The size of the layer output.
     */
    [[nodiscard]] SizeType output_size() const;

protected:
    friend class Model;

    Model& _model;                         ///< Model reference.
    std::string _name;                     ///< Layer name (for debug).
    std::vector<SharedPtr> _antecedents;   ///< List of previous layers.
    std::vector<SharedPtr> _subsequents;   ///< List of followers layers.
    SizeType _input_size;                  ///< Layer input size.
    SizeType _output_size;                 ///< Layer output size.

    Activation _activation;

    /**
     * \brief The last input passed to the layer. It is needed to compute loss
     * gradients with respect to the weights during backpropagation.
     */
    const NumType* _last_input;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_LAYER_HPP
