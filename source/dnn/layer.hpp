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
#include "parser/json.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <map>


namespace EdgeLearning {

class Model;

/**
 * \brief Learning parameters of a layer that can't be shared and will be
 * copied.
 */
using Params = std::vector<NumType>;

/**
 * \brief Learning parameters of a layer that can be shared.
 */
class SharedParams {
public:
    struct Iterator
    {
        using pointer   = NumType*;
        using reference = NumType&;

        Iterator(Params::iterator ptr) : _iter(ptr) {}

        reference operator*() const { return _iter.operator*(); }
        pointer operator->() { return _iter.operator->(); }
        Iterator& operator++() { _iter++; return *this; }
        Iterator operator++(int)
        { Iterator tmp = *this; ++(*this); return tmp; }
        friend bool operator== (const Iterator& a, const Iterator& b)
        { return a._iter == b._iter; };
        friend bool operator!= (const Iterator& a, const Iterator& b)
        { return a._iter != b._iter; };

    private:
        Params::iterator _iter;
    };

    SharedParams()
        : _p(std::make_shared<Params>())
    { }

    void resize(std::size_t length) const { (*_p).resize(length); }
    NumType& operator[](std::size_t i) const { return (*_p)[i]; }
    [[nodiscard]] const NumType& at(std::size_t i) const { return (*_p).at(i); }
    NumType* data() { return (*_p).data(); }
    std::size_t size() { return (*_p).size(); }

    Iterator begin() { return Iterator((*_p).begin()); }
    Iterator end()   { return Iterator((*_p).end());   }

private:
    std::shared_ptr<Params> _p;
};

/**
 * \brief Base class of computational layers in a model.
 */
class Layer 
{
public:
    enum class DumpFields
    {
        TYPE,
        NAME,
        INPUT_SIZE,
        OUTPUT_SIZE,
        WEIGHTS,
        BIASES,
        ANTECEDENTS,
        SUBSEQUENTS,
        OTHERS
    };

    static const std::string TYPE;
    static const std::map<DumpFields, std::string> dump_fields;

    using InitializationFunction = DLMath::InitializationFunction;
    using ProbabilityDensityFunction = DLMath::ProbabilityDensityFunction;
    using SharedPtr = std::shared_ptr<Layer>;

    /**
     * \brief Construct a new Layer object.
     * \param model        The model in which the layer takes part.
     * \param input_shape  The shape of inputs of the layer.
     * \param output_shape The shape of outputs of the layer.
     * \param name         The name of the layer.
     * If empty, a default generated one is chosen.
     * \param prefix_name The prefix name of the default generated name.
     */
    Layer(Model& model,
          DLMath::Shape3d input_shape = 0, DLMath::Shape3d output_shape = 0,
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
     * \param init InitializationFunction    The initialization used.
     * \param pdf ProbabilityDensityFunction The distribution used.
     * \param rne RneType                    The random generator.
     */
    virtual void init(
        InitializationFunction init = InitializationFunction::KAIMING,
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
        const std::vector<NumType>& inputs);

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
     * \brief Getter of layer type.
     * \return std::string The layer type.
     */
    [[nodiscard]] virtual inline const std::string& type() const
    { return TYPE; }

    /**
     * \brief Check if this instance is of type LayerT.
     * \tparam LayerT The Layer type to check.
     * \return bool True if this is of type LayerT, otherwise false.
     */
    template<class LayerT>
    [[nodiscard]] bool is_type() const
    {
        return dynamic_cast<const LayerT*>(this) != nullptr;
    }

    /**
     * \brief Return the last input of the layer.
     * \return const NumType* The last input of the layer of input size.
     */
    std::vector<NumType> last_input()
    {
        return _last_input
            ? std::vector<NumType>{
                _last_input, _last_input + _input_shape.size()}
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
     * \brief Clone the layer with its custom parameters.
     * \return std::shared_prt<Layer> The pointer to the cloned layer.
     */
    [[nodiscard]] virtual SharedPtr clone() const = 0;

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
     * \brief Getter of input_shape class field.
     * \return The size of the layer input.
     */
    [[nodiscard]] virtual const DLMath::Shape3d& input_shape() const;

    /**
     * \brief Setter of input_shape class field.
     * \param input_shape DLMath::Shape3d Shape param used to take the size and
     * assign it to input_shape.
     */
    virtual void input_shape(DLMath::Shape3d input_shape);

    /**
     * \brief Getter of output_shape class field.
     * \return The size of the layer output.
     */
    [[nodiscard]] virtual const DLMath::Shape3d& output_shape() const;

    /**
     * \brief Getter of input_shape class field.
     * \return The size of the layer input.
     */
    [[nodiscard]] SizeType input_size() const;

    /**
     * \brief Getter of output_shape class field.
     * \return The size of the layer output.
     */
    [[nodiscard]] SizeType output_size() const;

    /**
     * \brief Save the layer infos to disk.
     * \param out Json& out Json to write.
     */
    virtual void dump(Json& out) const;

    /**
     * \brief Load the layer infos from disk.
     * \param in const Json& Json to read.
     */
    virtual void load(Json& in);

protected:
    /**
     * \brief Check the input of the forward pass during layer training.
     * \param inputs The input of the layer.
     */
    void _check_training_input(const std::vector<NumType>& inputs);

    friend class Model;

    Model& _model;                         ///< Model reference.
    std::string _name;                     ///< Layer name (for debug).
    std::vector<SharedPtr> _antecedents;   ///< List of previous layers.
    std::vector<SharedPtr> _subsequents;   ///< List of followers layers.
    DLMath::Shape3d _input_shape;          ///< Layer input shape.
    DLMath::Shape3d _output_shape;         ///< Layer output shape.

    /**
     * \brief The last input passed to the layer. It is needed to compute loss
     * gradients with respect to the weights during backpropagation.
     */
    const NumType* _last_input;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_LAYER_HPP
