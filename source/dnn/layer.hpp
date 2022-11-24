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
#include <utility>
#include <vector>
#include <algorithm>
#include <map>


namespace EdgeLearning {

class DLGraph;

class LayerShape {
public:
    LayerShape(std::vector<DLMath::Shape3d> shape_vec);
    LayerShape(DLMath::Shape3d shape);
    LayerShape(SizeType size);
    LayerShape();

    [[nodiscard]] const std::vector<DLMath::Shape3d>& shapes() const;

    [[nodiscard]] const DLMath::Shape3d& shape(SizeType idx = 0) const;
    [[nodiscard]] SizeType size(SizeType idx = 0) const;
    [[nodiscard]] SizeType width(SizeType idx = 0) const;
    [[nodiscard]] SizeType height(SizeType idx = 0) const;
    [[nodiscard]] SizeType channels(SizeType idx = 0) const;

    [[nodiscard]] SizeType amount_shapes() const;

private:
    std::vector<DLMath::Shape3d> _shape_vec;
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

    class Fields {
    public:
        Fields(const std::string& name, const LayerShape& input_shape,
               const LayerShape& output_shape)
            : _name(name)
            , _input_shape(input_shape)
            , _input_size(_input_shape.size())
            , _output_shape(output_shape)
            , _output_size(_output_shape.size())
        { }

        std::string& name() { return _name; }
        [[nodiscard]] const std::string& name() const
        { return _name; }

        LayerShape& input_shape() { return _input_shape; }
        [[nodiscard]] const LayerShape& input_shape() const
        { return _input_shape; }

        SizeType& input_size() { return _input_size; }
        [[nodiscard]] const SizeType& input_size() const
        { return _input_size; }

        LayerShape& output_shape() { return _output_shape; }
        [[nodiscard]] const LayerShape& output_shape() const
        { return _output_shape; }

        SizeType& output_size() { return _output_size; }
        [[nodiscard]] const SizeType& output_size() const
        { return _output_size; }

    private:
        std::string _name;        ///< Layer name.
        LayerShape _input_shape;  ///< Layer input shape.
        SizeType   _input_size;   ///< Layer input size.
        LayerShape _output_shape; ///< Layer output shape.
        SizeType   _output_size;  ///< Layer output size.
    };

    /**
     * \brief Construct a new Layer object.
     * \param name         The name of the layer.
     * If empty, a default generated one is chosen.
     * \param input_shape  The shape of inputs of the layer.
     * \param output_shape The shape of outputs of the layer.
     * \param prefix_name The prefix name of the default generated name.
     */
    Layer(std::string name = std::string(), LayerShape input_shape = 0,
          LayerShape output_shape = 0, std::string prefix_name = std::string());

    /**
     * \brief Destroy the Layer object.
     */
    virtual ~Layer() = default;

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
    std::vector<NumType> last_input();
    virtual const std::vector<NumType>& last_input_gradient() = 0;

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
    { return _shared_fields->name(); }

    /**
     * \brief Getter of input_shape class field.
     * \return The size of the layer input.
     */
    [[nodiscard]] const LayerShape& input_shape() const;

    /**
     * \brief Setter of input_shape class field. Wrapper of protected virtual
     * method used in inheritance layers to override this setter.
     * \param input_shape LayerShape Shape of the layer input tensor.
     */
    void input_shape(LayerShape input_shape);

    /**
     * \brief Getter of output_shape class field.
     * \return The size of the layer output.
     */
    [[nodiscard]] const LayerShape& output_shape() const;

    /**
     * \brief Getter of the list of input shapes of the layer.
     * \return const std::vector<DLMath::Shape3d>& The vector of shapes.
     */
    [[nodiscard]] const std::vector<DLMath::Shape3d>& input_shapes() const;

    /**
     * \brief Getter of the list of output shapes of the layer.
     * \return const std::vector<DLMath::Shape3d>& The vector of shapes.
     */
    [[nodiscard]] const std::vector<DLMath::Shape3d>& output_shapes() const;

    /**
     * \brief Getter of input_shape class field.
     * \param input_idx SizeType The index of the input to obtain the size.
     * \return The size of the layer input.
     */
    [[nodiscard]] SizeType input_size(SizeType input_idx = 0) const;

    /**
     * \brief Getter of output_shape class field.
     * \param output_idx SizeType The index of the output to obtain the size.
     * \return The size of the layer output.
     */
    [[nodiscard]] SizeType output_size(SizeType output_idx = 0) const;

    /**
     * \brief Getter of the input layers amount.
     * \return SizeType The input layers amount.
     */
    [[nodiscard]] SizeType input_layers();

    /**
     * \brief Getter of the output layers amount.
     * \return SizeType The output layers amount.
     */
    [[nodiscard]] SizeType output_layers();

    /**
     * \brief Save the layer infos to disk.
     * \return Json Layer dump.
     */
    virtual Json dump() const;

    /**
     * \brief Load the layer infos from disk.
     * \param in const Json& Json to read.
     */
    virtual void load(const Json& in);

protected:
    friend class Model;

    /**
     * \brief Setter of input_shape class field.
     * \param input_shape LayerShape Shape param used to take the size and
     * assign it to input_shape.
     */
    virtual void _set_input_shape(LayerShape input_shape);

    std::shared_ptr<Fields> _shared_fields; ///< Layer shared fields.

    /**
     * \brief The last input passed to the layer. It is needed to compute loss
     * gradients with respect to the weights during backpropagation.
     */
    const NumType* _last_input;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_LAYER_HPP
